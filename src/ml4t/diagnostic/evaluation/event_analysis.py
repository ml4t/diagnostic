"""Event Study Analysis Module.

This module implements event study methodology following MacKinlay (1997)
"Event Studies in Economics and Finance" for measuring abnormal returns
around corporate events, announcements, or other market events.

Classes
-------
EventStudyAnalysis
    Main class for conducting event studies

References
----------
MacKinlay, A.C. (1997). "Event Studies in Economics and Finance",
    Journal of Economic Literature, 35(1), 13-39.
Boehmer, E., Musumeci, J., Poulsen, A.B. (1991). "Event-study methodology
    under conditions of event-induced variance", Journal of Financial Economics.
Corrado, C.J. (1989). "A nonparametric test for abnormal security-price
    performance in event studies", Journal of Financial Economics.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats

from ml4t.diagnostic.config.event_config import EventConfig
from ml4t.diagnostic.results.event_results import AbnormalReturnResult, EventStudyResult

if TYPE_CHECKING:
    import pandas as pd


class EventStudyAnalysis:
    """Event study analysis for measuring abnormal returns around events.

    Implements the standard event study methodology with support for:
    - Market model (CAPM-based expected returns)
    - Mean-adjusted model
    - Market-adjusted model

    And statistical tests:
    - Standard t-test
    - BMP test (Boehmer et al. 1991, robust to event-induced variance)
    - Corrado rank test (non-parametric)

    Parameters
    ----------
    returns : pl.DataFrame
        Asset returns in long format with columns: [date, asset, return].
        Returns should be simple returns (not log returns).
    events : pl.DataFrame
        Events to analyze with columns: [date, asset]. Optionally
        includes [event_type, event_id] for grouping.
    benchmark : pl.DataFrame
        Market/benchmark returns with columns: [date, return].
    config : EventConfig, optional
        Configuration for the analysis.

    Examples
    --------
    >>> returns_df = pl.DataFrame({
    ...     'date': [...],
    ...     'asset': [...],
    ...     'return': [...]
    ... })
    >>> events_df = pl.DataFrame({
    ...     'date': ['2023-01-15', '2023-02-20'],
    ...     'asset': ['AAPL', 'MSFT']
    ... })
    >>> benchmark_df = pl.DataFrame({
    ...     'date': [...],
    ...     'return': [...]  # Market returns
    ... })
    >>> analysis = EventStudyAnalysis(returns_df, events_df, benchmark_df)
    >>> result = analysis.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        returns: pl.DataFrame | pd.DataFrame,
        events: pl.DataFrame | pd.DataFrame,
        benchmark: pl.DataFrame | pd.DataFrame,
        config: EventConfig | None = None,
    ) -> None:
        """Initialize event study analysis."""
        self.config = config or EventConfig()

        # Convert to Polars if needed
        self._returns = self._to_polars(returns)
        self._events = self._to_polars(events)
        self._benchmark = self._to_polars(benchmark)

        # Validate inputs
        self._validate_inputs()

        # Prepare data
        self._prepare_data()

        # Cache for computed results
        self._ar_results: list[AbnormalReturnResult] | None = None
        self._aggregated_result: EventStudyResult | None = None

    def _to_polars(self, df: Any) -> pl.DataFrame:
        """Convert DataFrame to Polars if needed."""
        if isinstance(df, pl.DataFrame):
            return df
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(df)}")

    def _validate_inputs(self) -> None:
        """Validate input DataFrames have required columns."""
        # Check returns
        required_return_cols = {"date", "asset", "return"}
        if not required_return_cols.issubset(set(self._returns.columns)):
            raise ValueError(
                f"returns DataFrame missing columns: {required_return_cols - set(self._returns.columns)}"
            )

        # Check events
        required_event_cols = {"date", "asset"}
        if not required_event_cols.issubset(set(self._events.columns)):
            raise ValueError(
                f"events DataFrame missing columns: {required_event_cols - set(self._events.columns)}"
            )

        # Check benchmark
        required_bench_cols = {"date", "return"}
        if not required_bench_cols.issubset(set(self._benchmark.columns)):
            raise ValueError(
                f"benchmark DataFrame missing columns: {required_bench_cols - set(self._benchmark.columns)}"
            )

        # Check we have events
        if len(self._events) == 0:
            raise ValueError("No events provided")

    def _prepare_data(self) -> None:
        """Prepare data for analysis (sorting, date alignment)."""
        # Sort by date
        self._returns = self._returns.sort("date")
        self._benchmark = self._benchmark.sort("date")

        # Create date-indexed lookup for benchmark
        self._benchmark_dict: dict[Any, float] = dict(
            zip(
                self._benchmark["date"].to_list(),
                self._benchmark["return"].to_list(),
                strict=False,
            )
        )

        # Get unique dates for index mapping
        self._all_dates = sorted(self._returns["date"].unique().to_list())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}

        # Add event_id if not present
        if "event_id" not in self._events.columns:
            self._events = self._events.with_row_index("event_id").with_columns(
                pl.col("event_id").cast(pl.Utf8).alias("event_id")
            )

    def _get_estimation_window_data(
        self, asset: str, event_date: Any
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get returns for estimation window.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            (asset_returns, market_returns) for estimation window,
            or None if insufficient data.
        """
        est_start, est_end = self.config.window.estimation_window

        # Find event date index
        if event_date not in self._date_to_idx:
            return None
        event_idx = self._date_to_idx[event_date]

        # Calculate estimation window indices
        start_idx = event_idx + est_start
        end_idx = event_idx + est_end

        if start_idx < 0:
            return None

        # Get dates in estimation window
        est_dates = self._all_dates[start_idx : end_idx + 1]

        if len(est_dates) < self.config.min_estimation_obs:
            return None

        # Get asset returns
        asset_data = self._returns.filter(
            (pl.col("asset") == asset) & (pl.col("date").is_in(est_dates))
        ).sort("date")

        if len(asset_data) < self.config.min_estimation_obs:
            return None

        # Get benchmark returns
        asset_returns = []
        market_returns = []
        for row in asset_data.iter_rows(named=True):
            date = row["date"]
            if date in self._benchmark_dict:
                asset_returns.append(row["return"])
                market_returns.append(self._benchmark_dict[date])

        if len(asset_returns) < self.config.min_estimation_obs:
            return None

        return np.array(asset_returns), np.array(market_returns)

    def _estimate_market_model(
        self, asset_returns: np.ndarray, market_returns: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Estimate market model parameters via OLS.

        AR = R - (α + β*Rm)

        Returns
        -------
        tuple[float, float, float, float]
            (alpha, beta, r_squared, residual_std)
        """
        # OLS regression: R_asset = alpha + beta * R_market + epsilon
        X = np.column_stack([np.ones(len(market_returns)), market_returns])
        y = asset_returns

        # Solve normal equations
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coeffs[0], coeffs[1]

            # Calculate R-squared
            y_pred = alpha + beta * market_returns
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Residual standard deviation
            residual_std = np.std(y - y_pred, ddof=2)

            return alpha, beta, r_squared, residual_std
        except Exception:
            return 0.0, 1.0, 0.0, np.std(asset_returns)

    def _get_event_window_data(
        self, asset: str, event_date: Any
    ) -> dict[int, tuple[float, float]] | None:
        """Get returns for event window.

        Returns
        -------
        dict[int, tuple[float, float]] | None
            {relative_day: (asset_return, market_return)}
        """
        evt_start, evt_end = self.config.window.event_window

        if event_date not in self._date_to_idx:
            return None
        event_idx = self._date_to_idx[event_date]

        result = {}
        for rel_day in range(evt_start, evt_end + 1):
            day_idx = event_idx + rel_day
            if 0 <= day_idx < len(self._all_dates):
                date = self._all_dates[day_idx]

                # Get asset return
                asset_ret = self._returns.filter(
                    (pl.col("asset") == asset) & (pl.col("date") == date)
                )

                if len(asset_ret) > 0 and date in self._benchmark_dict:
                    result[rel_day] = (
                        asset_ret["return"][0],
                        self._benchmark_dict[date],
                    )

        return result if result else None

    def _compute_abnormal_return_single(
        self, event_row: dict[str, Any]
    ) -> AbnormalReturnResult | None:
        """Compute abnormal returns for a single event."""
        asset = event_row["asset"]
        event_date = event_row["date"]
        event_id = str(event_row.get("event_id", f"{asset}_{event_date}"))

        # Get estimation window data
        est_data = self._get_estimation_window_data(asset, event_date)
        if est_data is None:
            return None

        asset_est_returns, market_est_returns = est_data

        # Estimate model parameters
        alpha, beta, r2, residual_std = 0.0, 1.0, 0.0, 0.0

        if self.config.model == "market_model":
            alpha, beta, r2, residual_std = self._estimate_market_model(
                asset_est_returns, market_est_returns
            )
        elif self.config.model == "mean_adjusted":
            alpha = float(np.mean(asset_est_returns))
            beta = 0.0
            residual_std = float(np.std(asset_est_returns, ddof=1))
        elif self.config.model == "market_adjusted":
            alpha = 0.0
            beta = 1.0
            residual_std = float(np.std(asset_est_returns - market_est_returns, ddof=1))

        # Get event window data
        event_data = self._get_event_window_data(asset, event_date)
        if event_data is None:
            return None

        # Compute abnormal returns
        ar_by_day: dict[int, float] = {}
        for rel_day, (asset_ret, market_ret) in event_data.items():
            if self.config.model == "market_model":
                expected_ret = alpha + beta * market_ret
            elif self.config.model == "mean_adjusted":
                expected_ret = alpha
            else:  # market_adjusted
                expected_ret = market_ret

            ar_by_day[rel_day] = asset_ret - expected_ret

        # Compute CAR
        car = sum(ar_by_day.values())

        return AbnormalReturnResult(
            event_id=event_id,
            asset=asset,
            event_date=str(event_date),
            ar_by_day=ar_by_day,
            car=car,
            estimation_alpha=alpha if self.config.model == "market_model" else None,
            estimation_beta=beta if self.config.model == "market_model" else None,
            estimation_r2=r2 if self.config.model == "market_model" else None,
            estimation_residual_std=residual_std,
        )

    def compute_abnormal_returns(self) -> list[AbnormalReturnResult]:
        """Compute abnormal returns for all events.

        Returns
        -------
        list[AbnormalReturnResult]
            Abnormal return results for each valid event.
        """
        if self._ar_results is not None:
            return self._ar_results

        results = []
        n_skipped = 0

        for row in self._events.iter_rows(named=True):
            result = self._compute_abnormal_return_single(row)
            if result is not None:
                results.append(result)
            else:
                n_skipped += 1

        if n_skipped > 0:
            warnings.warn(
                f"Skipped {n_skipped} events due to insufficient data",
                stacklevel=2,
            )

        self._ar_results = results
        return results

    def aggregate(self, group_by: str | None = None) -> EventStudyResult:
        """Aggregate individual results to AAR and CAAR.

        Parameters
        ----------
        group_by : str | None
            Column to group by (e.g., 'event_type'). If None,
            aggregates all events together.

        Returns
        -------
        EventStudyResult
            Aggregated event study results.
        """
        ar_results = self.compute_abnormal_returns()

        if len(ar_results) == 0:
            raise ValueError("No valid events to aggregate")

        # Collect all relative days
        all_days = set()
        for r in ar_results:
            all_days.update(r.ar_by_day.keys())
        sorted_days = sorted(all_days)

        # Compute AAR (average AR across events for each day)
        aar_by_day: dict[int, float] = {}
        ar_matrix: dict[int, list[float]] = {d: [] for d in sorted_days}

        for r in ar_results:
            for day in sorted_days:
                if day in r.ar_by_day:
                    ar_matrix[day].append(r.ar_by_day[day])

        for day in sorted_days:
            if ar_matrix[day]:
                aar_by_day[day] = float(np.mean(ar_matrix[day]))
            else:
                aar_by_day[day] = 0.0

        # Compute CAAR and its statistics
        caar_values = []
        caar_std = []
        cumsum = 0.0

        for day in sorted_days:
            cumsum += aar_by_day[day]
            caar_values.append(cumsum)

            # Cross-sectional standard deviation at this day
            if ar_matrix[day]:
                caar_std.append(float(np.std(ar_matrix[day], ddof=1)))
            else:
                caar_std.append(0.0)

        # Compute confidence intervals
        n_events = len(ar_results)
        z_score = stats.norm.ppf(1 - self.config.alpha / 2)

        caar_ci_lower = []
        caar_ci_upper = []
        for caar, std in zip(caar_values, caar_std, strict=False):
            se = std / np.sqrt(n_events) if n_events > 0 else 0.0
            caar_ci_lower.append(caar - z_score * se)
            caar_ci_upper.append(caar + z_score * se)

        # Run statistical test
        test_stat, p_value = self._run_statistical_test(ar_results, ar_matrix)

        result = EventStudyResult(
            aar_by_day=aar_by_day,
            caar=caar_values,
            caar_dates=sorted_days,
            caar_std=caar_std,
            caar_ci_lower=caar_ci_lower,
            caar_ci_upper=caar_ci_upper,
            test_statistic=test_stat,
            p_value=p_value,
            test_name=self.config.test,
            n_events=n_events,
            model_name=self.config.model,
            event_window=self.config.window.event_window,
            confidence_level=self.config.confidence_level,
            individual_results=ar_results,
        )

        self._aggregated_result = result
        return result

    def _run_statistical_test(
        self,
        ar_results: list[AbnormalReturnResult],
        ar_matrix: dict[int, list[float]],
    ) -> tuple[float, float]:
        """Run statistical significance test.

        Returns
        -------
        tuple[float, float]
            (test_statistic, p_value)
        """
        if self.config.test == "t_test":
            return self._t_test(ar_results, ar_matrix)
        elif self.config.test == "boehmer":
            return self._bmp_test(ar_results)
        elif self.config.test == "corrado":
            return self._corrado_test(ar_results, ar_matrix)
        else:
            return self._t_test(ar_results, ar_matrix)

    def _t_test(
        self,
        ar_results: list[AbnormalReturnResult],
        ar_matrix: dict[int, list[float]],
    ) -> tuple[float, float]:
        """Standard parametric t-test on CAAR.

        H0: CAAR = 0
        Test statistic: t = CAAR / SE(CAAR)
        """
        # Get CARs for all events
        cars = [r.car for r in ar_results]
        n = len(cars)

        if n < 2:
            return 0.0, 1.0

        mean_car = np.mean(cars)
        std_car = np.std(cars, ddof=1)
        se_car = std_car / np.sqrt(n)

        if se_car == 0:
            return 0.0, 1.0

        t_stat = mean_car / se_car
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        return float(t_stat), float(p_value)

    def _bmp_test(self, ar_results: list[AbnormalReturnResult]) -> tuple[float, float]:
        """Boehmer, Musumeci, Poulsen (1991) test.

        Robust to event-induced variance changes by standardizing
        ARs by their estimation period volatility.

        SAR_i = AR_i / σ_i
        Test statistic: Z = (1/N) * Σ SAR_i / SE(SAR)
        """
        # Compute standardized abnormal returns
        sars = []
        for r in ar_results:
            if r.estimation_residual_std and r.estimation_residual_std > 0:
                sar = r.car / r.estimation_residual_std
            else:
                sar = r.car  # Fallback to unstandardized
            sars.append(sar)

        n = len(sars)
        if n < 2:
            return 0.0, 1.0

        mean_sar = np.mean(sars)
        std_sar = np.std(sars, ddof=1)
        se_sar = std_sar / np.sqrt(n)

        if se_sar == 0:
            return 0.0, 1.0

        z_stat = mean_sar / se_sar
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return float(z_stat), float(p_value)

    def _corrado_test(
        self,
        ar_results: list[AbnormalReturnResult],
        ar_matrix: dict[int, list[float]],
    ) -> tuple[float, float]:
        """Corrado (1989) non-parametric rank test.

        Robust to non-normality in returns. Uses ranks instead of
        raw abnormal returns.
        """
        n_events = len(ar_results)
        if n_events < 2:
            return 0.0, 1.0

        # For simplicity, test at t=0 (event day)
        if 0 not in ar_matrix or len(ar_matrix[0]) < 2:
            # Fallback to t-test
            return self._t_test(ar_results, ar_matrix)

        event_day_ars = np.array(ar_matrix[0])

        # Rank the ARs
        ranks = stats.rankdata(event_day_ars)
        expected_rank = (n_events + 1) / 2

        # Compute test statistic
        rank_deviations = ranks - expected_rank
        mean_deviation = np.mean(rank_deviations)

        # Standard deviation of ranks under null
        std_rank = np.std(rank_deviations, ddof=1)
        se_rank = std_rank / np.sqrt(n_events)

        if se_rank == 0:
            return 0.0, 1.0

        z_stat = mean_deviation / se_rank
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return float(z_stat), float(p_value)

    def run(self) -> EventStudyResult:
        """Run complete event study analysis.

        This is the main entry point that computes abnormal returns,
        aggregates results, and runs statistical tests.

        Returns
        -------
        EventStudyResult
            Complete event study results.

        Examples
        --------
        >>> analysis = EventStudyAnalysis(returns, events, benchmark)
        >>> result = analysis.run()
        >>> print(result.summary())
        >>> if result.is_significant:
        ...     print("Significant abnormal returns detected!")
        """
        return self.aggregate()

    def create_tear_sheet(self) -> EventStudyResult:
        """Alias for run() - creates complete event study results."""
        return self.run()

    @property
    def n_events(self) -> int:
        """Number of events in the study."""
        return len(self._events)

    @property
    def n_valid_events(self) -> int:
        """Number of events with sufficient data for analysis."""
        ar_results = self.compute_abnormal_returns()
        return len(ar_results)

    @property
    def assets(self) -> list[str]:
        """List of unique assets in the events."""
        return self._events["asset"].unique().sort().to_list()

    @property
    def date_range(self) -> tuple[Any, Any]:
        """Date range of the returns data."""
        return self._all_dates[0], self._all_dates[-1]
