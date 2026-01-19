"""Multi-Signal Analysis module for batch signal evaluation.

This module provides efficient analysis of 50-200 signals with:
- Parallel computation via joblib
- Smart caching with Polars fingerprinting
- FDR and FWER multiple testing corrections
- Signal selection algorithms for comparison
- Focus + Context visualization patterns

References
----------
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate"
Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure"
López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from tqdm import tqdm

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.caching.smart_cache import SmartCache
from ml4t.diagnostic.config.multi_signal_config import MultiSignalAnalysisConfig
from ml4t.diagnostic.evaluation.signal_selector import SignalSelector
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr, holm_bonferroni
from ml4t.diagnostic.results.multi_signal_results import ComparisonResult, MultiSignalSummary
from ml4t.diagnostic.signal import SignalResult, analyze_signal

if TYPE_CHECKING:
    import pandas as pd


class MultiSignalAnalysis:
    """Batch analysis of multiple signals with statistical corrections.

    Efficiently analyze 50-200 signals with parallel computation,
    smart caching, and multiple testing corrections.

    Parameters
    ----------
    signals : dict[str, pl.DataFrame | pd.DataFrame]
        Dictionary mapping signal names to factor DataFrames.
        Each DataFrame must have columns: date, asset, factor
    prices : pl.DataFrame | pd.DataFrame
        Price data with columns: date, asset, price
    config : MultiSignalAnalysisConfig | None
        Configuration object. If None, uses defaults.

    Examples
    --------
    >>> # Basic usage
    >>> signals = {
    ...     'momentum_12m': mom_df,
    ...     'value_btm': val_df,
    ...     'quality': qual_df,
    ... }
    >>> analyzer = MultiSignalAnalysis(signals, prices)
    >>> summary = analyzer.compute_summary()
    >>> print(f"Significant: {summary.n_fdr_significant}/{summary.n_signals}")

    >>> # Compare top uncorrelated signals
    >>> comparison = analyzer.compare(selection="uncorrelated", n=5)
    >>> comparison.save_html("top_signals.html")

    >>> # Custom configuration
    >>> config = MultiSignalAnalysisConfig(
    ...     fdr_alpha=0.01,
    ...     fwer_alpha=0.01,
    ...     n_jobs=-1,  # All cores
    ... )
    >>> analyzer = MultiSignalAnalysis(signals, prices, config=config)
    """

    def __init__(
        self,
        signals: dict[str, pl.DataFrame | pd.DataFrame],
        prices: pl.DataFrame | pd.DataFrame,
        config: MultiSignalAnalysisConfig | None = None,
    ) -> None:
        """Initialize MultiSignalAnalysis."""
        self.config = config or MultiSignalAnalysisConfig()

        # Convert signals to Polars
        self._signals: dict[str, pl.DataFrame] = {}
        for name, df in signals.items():
            converted, _ = DataFrameAdapter.to_polars(df)
            self._signals[name] = converted

        # Convert prices to Polars
        self._prices, _ = DataFrameAdapter.to_polars(prices)

        # Validate inputs
        self._validate_inputs()

        # Initialize cache if enabled
        self._cache: SmartCache | None = None
        if self.config.cache_enabled:
            self._cache = SmartCache(
                max_items=self.config.cache_max_items,
                ttl_seconds=self.config.cache_ttl,
            )

        # Cached results
        self._summary: MultiSignalSummary | None = None
        self._individual_results: dict[str, SignalResult] = {}
        self._correlation_matrix: pl.DataFrame | None = None

    def _validate_inputs(self) -> None:
        """Validate input data structure."""
        if not self._signals:
            raise ValueError("No signals provided")

        # Check each signal has required columns
        required_cols = {"date", "asset", "factor"}
        for name, df in self._signals.items():
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Signal '{name}' missing required columns: {missing}")

        # Check prices
        price_required = {"date", "asset", "price"}
        missing_price = price_required - set(self._prices.columns)
        if missing_price:
            raise ValueError(f"Price data missing required columns: {missing_price}")

    @property
    def signal_names(self) -> list[str]:
        """List of signal names."""
        return list(self._signals.keys())

    @property
    def n_signals(self) -> int:
        """Number of signals."""
        return len(self._signals)

    def get_individual(self, signal_name: str) -> SignalResult:
        """Get or create SignalResult for a specific signal.

        Parameters
        ----------
        signal_name : str
            Name of signal

        Returns
        -------
        SignalResult
            Analysis result for the signal
        """
        if signal_name not in self._signals:
            raise ValueError(f"Signal '{signal_name}' not found. Available: {self.signal_names}")

        if signal_name not in self._individual_results:
            self._individual_results[signal_name] = analyze_signal(
                self._signals[signal_name],
                self._prices,
                periods=tuple(self.config.signal_config.periods),
                quantiles=self.config.signal_config.quantiles,
                filter_zscore=self.config.signal_config.filter_zscore,
                compute_turnover_flag=self.config.signal_config.compute_turnover,
            )

        return self._individual_results[signal_name]

    def _compute_signal_metrics(self, signal_name: str) -> dict[str, Any]:
        """Compute metrics for a single signal.

        This is the parallelizable unit of work.
        """
        # Check cache
        cache_key = None
        if self._cache is not None:
            cache_key = self._cache.make_key(
                signal_name,
                self._signals[signal_name],
                self.config.signal_config,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Compute metrics using new functional API
        try:
            result = analyze_signal(
                self._signals[signal_name],
                self._prices,
                periods=tuple(self.config.signal_config.periods),
                quantiles=self.config.signal_config.quantiles,
                filter_zscore=self.config.signal_config.filter_zscore,
                compute_turnover_flag=self.config.signal_config.compute_turnover,
            )

            # Extract metrics for first period (most common use case)
            period = self.config.signal_config.periods[0]
            period_key = f"{period}D"

            metrics = {
                "signal_name": signal_name,
                "ic_mean": result.ic.get(period_key, np.nan),
                "ic_std": result.ic_std.get(period_key, np.nan),
                "ic_t_stat": result.ic_t_stat.get(period_key, np.nan),
                "ic_p_value": result.ic_p_value.get(period_key, np.nan),
                "ic_ir": result.ic_ir.get(period_key, np.nan),
                "ic_positive_pct": result.ic_positive_pct.get(period_key, np.nan),
                "n_observations": result.n_dates,
                "error": None,
            }

            # Add turnover if computed
            if result.turnover is not None:
                metrics["turnover_mean"] = result.turnover.get(period_key, np.nan)
            else:
                metrics["turnover_mean"] = np.nan

            if result.autocorrelation is not None and len(result.autocorrelation) > 0:
                metrics["autocorr_1"] = result.autocorrelation[0]
            else:
                metrics["autocorr_1"] = np.nan

        except Exception as e:
            metrics = {
                "signal_name": signal_name,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ic_t_stat": np.nan,
                "ic_p_value": np.nan,
                "ic_ir": np.nan,
                "ic_positive_pct": np.nan,
                "n_observations": 0,
                "turnover_mean": np.nan,
                "autocorr_1": np.nan,
                "error": str(e),
            }

        # Cache result
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, metrics)

        return metrics

    def compute_summary(
        self,
        progress: bool = True,
    ) -> MultiSignalSummary:
        """Compute summary metrics for all signals with FDR/FWER correction.

        Parameters
        ----------
        progress : bool, default True
            Show progress bar

        Returns
        -------
        MultiSignalSummary
            Summary with metrics and multiple testing corrections
        """
        if self._summary is not None:
            return self._summary

        # Compute metrics for all signals
        if self.config.n_jobs == 1:
            # Serial execution
            results = []
            iterator = tqdm(self.signal_names, disable=not progress, desc="Analyzing signals")
            for name in iterator:
                results.append(self._compute_signal_metrics(name))
        else:
            # Parallel execution
            try:
                from joblib import Parallel, delayed

                results = Parallel(
                    n_jobs=self.config.n_jobs,
                    backend=self.config.backend,
                )(
                    delayed(self._compute_signal_metrics)(name)
                    for name in tqdm(
                        self.signal_names, disable=not progress, desc="Analyzing signals"
                    )
                )
            except ImportError:
                warnings.warn(
                    "joblib not available, falling back to serial execution",
                    UserWarning,
                    stacklevel=2,
                )
                results = []
                iterator = tqdm(self.signal_names, disable=not progress, desc="Analyzing signals")
                for name in iterator:
                    results.append(self._compute_signal_metrics(name))

        # Build summary DataFrame
        summary_data: dict[str, list[Any]] = {
            "signal_name": [],
            "ic_mean": [],
            "ic_std": [],
            "ic_t_stat": [],
            "ic_p_value": [],
            "ic_ir": [],
            "ic_positive_pct": [],
            "n_observations": [],
            "turnover_mean": [],
            "autocorr_1": [],
        }

        for r in results:
            for key in summary_data:
                summary_data[key].append(r.get(key, np.nan))

        # Apply FDR correction
        p_values = summary_data["ic_p_value"]
        valid_p_values = [p if not np.isnan(p) else 1.0 for p in p_values]

        fdr_result = benjamini_hochberg_fdr(
            valid_p_values,
            alpha=self.config.fdr_alpha,
            return_details=True,
        )
        summary_data["fdr_significant"] = list(fdr_result["rejected"])
        summary_data["fdr_adjusted_p"] = list(fdr_result["adjusted_p_values"])

        # Apply FWER correction
        fwer_result = holm_bonferroni(valid_p_values, alpha=self.config.fwer_alpha)
        summary_data["fwer_significant"] = fwer_result["rejected"]
        summary_data["fwer_adjusted_p"] = fwer_result["adjusted_p_values"]

        # Count significant
        n_fdr_sig = sum(summary_data["fdr_significant"])
        n_fwer_sig = sum(summary_data["fwer_significant"])

        # Create result
        self._summary = MultiSignalSummary(
            summary_data=summary_data,
            n_signals=self.n_signals,
            n_fdr_significant=n_fdr_sig,
            n_fwer_significant=n_fwer_sig,
            periods=self.config.signal_config.periods,
            fdr_alpha=self.config.fdr_alpha,
            fwer_alpha=self.config.fwer_alpha,
        )

        return self._summary

    def correlation_matrix(
        self,
        method: Literal["returns", "ic"] = "returns",
    ) -> pl.DataFrame:
        """Compute pairwise signal correlation matrix.

        Parameters
        ----------
        method : str, default "returns"
            Correlation method:
            - "returns": Correlation of signal-weighted returns
            - "ic": Correlation of IC time series

        Returns
        -------
        pl.DataFrame
            Correlation matrix with signal names as columns
        """
        if self._correlation_matrix is not None:
            return self._correlation_matrix

        # For now, use simple cross-sectional correlation of factor values
        # This is a reasonable approximation for signal similarity

        # Get all dates that appear in all signals
        all_dates: set[Any] | None = None
        for df in self._signals.values():
            dates = set(df["date"].unique().to_list())
            if all_dates is None:
                all_dates = dates
            else:
                all_dates = all_dates.intersection(dates)

        if not all_dates:
            raise ValueError("No overlapping dates across signals")

        # Build correlation matrix
        n = self.n_signals
        corr_matrix = np.eye(n)

        for i, name_i in enumerate(self.signal_names):
            for j, name_j in enumerate(self.signal_names):
                if i >= j:
                    continue

                # Get factor values for common dates and assets
                df_i = self._signals[name_i].filter(pl.col("date").is_in(list(all_dates)))
                df_j = self._signals[name_j].filter(pl.col("date").is_in(list(all_dates)))

                # Join on date and asset
                merged = df_i.select(["date", "asset", "factor"]).join(
                    df_j.select(["date", "asset", pl.col("factor").alias("factor_j")]),
                    on=["date", "asset"],
                    how="inner",
                )

                if merged.height > 10:
                    corr = np.corrcoef(
                        merged["factor"].to_numpy(),
                        merged["factor_j"].to_numpy(),
                    )[0, 1]
                    if not np.isnan(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr

        # Convert to DataFrame
        self._correlation_matrix = pl.DataFrame(
            corr_matrix,
            schema=self.signal_names,
        )

        return self._correlation_matrix

    def compare(
        self,
        selection: Literal["top_n", "uncorrelated", "pareto", "cluster", "manual"] = "top_n",
        n: int = 10,
        signals: list[str] | None = None,
        **kwargs: Any,
    ) -> ComparisonResult:
        """Create detailed comparison of selected signals.

        Parameters
        ----------
        selection : str, default "top_n"
            Selection method:
            - "top_n": Best N by metric (default: ic_ir)
            - "uncorrelated": Diverse signals with low correlation
            - "pareto": Signals on efficient frontier
            - "cluster": Representative from each cluster
            - "manual": Use provided signal list
        n : int, default 10
            Number of signals to select (ignored for "manual")
        signals : list[str] | None
            Signal names for "manual" selection
        **kwargs : Any
            Additional parameters for selection methods

        Returns
        -------
        ComparisonResult
            Detailed comparison with tear sheet data
        """
        # Ensure summary is computed
        summary = self.compute_summary(progress=False)
        summary_df = summary.get_dataframe()

        # Get correlation matrix if needed
        corr_matrix = None
        if selection in ("uncorrelated", "cluster"):
            corr_matrix = self.correlation_matrix()

        # Select signals
        if selection == "manual":
            if signals is None:
                raise ValueError("signals parameter required for manual selection")
            selected = signals
        elif selection == "top_n":
            metric = kwargs.get("metric", self.config.default_selection_metric)
            selected = SignalSelector.select_top_n(summary_df, n=n, metric=metric, **kwargs)
        elif selection == "uncorrelated":
            if corr_matrix is None:
                raise ValueError("Correlation matrix required for uncorrelated selection")
            max_corr = kwargs.get("max_correlation", self.config.default_correlation_threshold)
            selected = SignalSelector.select_uncorrelated(
                summary_df, corr_matrix, n=n, max_correlation=max_corr, **kwargs
            )
        elif selection == "pareto":
            selected = SignalSelector.select_pareto_frontier(summary_df, **kwargs)
            if len(selected) > n:
                selected = selected[:n]
        elif selection == "cluster":
            if corr_matrix is None:
                raise ValueError("Correlation matrix required for cluster selection")
            n_clusters = kwargs.get("n_clusters", n)
            selected = SignalSelector.select_by_cluster(
                corr_matrix, summary_df, n_clusters=n_clusters, **kwargs
            )
        else:
            raise ValueError(f"Unknown selection method: {selection}")

        # Limit to max comparison signals
        if len(selected) > self.config.max_signals_comparison:
            selected = selected[: self.config.max_signals_comparison]

        # Compute tear sheets (signal results) for selected signals
        tear_sheets: dict[str, dict[str, Any]] = {}
        for name in selected:
            try:
                result = self.get_individual(name)
                tear_sheets[name] = result.to_dict()
            except Exception as e:
                warnings.warn(
                    f"Failed to analyze signal {name}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                tear_sheets[name] = {"error": str(e)}

        # Get correlation matrix for selected signals
        full_corr = self.correlation_matrix()
        selected_corr: dict[str, list[float]] = {}
        for name in selected:
            if name in full_corr.columns:
                idx = self.signal_names.index(name)
                selected_corr[name] = [full_corr[s][idx] for s in selected]
            else:
                selected_corr[name] = [np.nan] * len(selected)

        return ComparisonResult(
            signals=selected,
            selection_method=selection,
            selection_params={"n": n, **kwargs},
            tear_sheets=tear_sheets,
            correlation_matrix=selected_corr,
        )

    def cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics.

        Returns
        -------
        dict | None
            Cache statistics if caching enabled, else None
        """
        if self._cache is None:
            return None
        return self._cache.stats

    def clear_cache(self) -> None:
        """Clear the cache."""
        if self._cache is not None:
            self._cache.clear()
        self._summary = None
        self._individual_results.clear()
        self._correlation_matrix = None

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"MultiSignalAnalysis(n_signals={self.n_signals}, "
            f"cache={'enabled' if self._cache else 'disabled'})"
        )
