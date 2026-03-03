"""Signal analysis result dataclass.

Simple, immutable result container for signal analysis.
No Pydantic, no inheritance - just a frozen dataclass.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml4t.diagnostic.results.signal_results.ic import SignalICResult
    from ml4t.diagnostic.results.signal_results.quantile import QuantileAnalysisResult
    from ml4t.diagnostic.results.signal_results.tearsheet import SignalTearSheet


@dataclass(frozen=True)
class SignalResult:
    """Immutable result from signal analysis.

    All metrics are keyed by period (e.g., "1D", "5D", "21D").

    Attributes
    ----------
    ic : dict[str, float]
        Mean IC by period.
    ic_std : dict[str, float]
        IC standard deviation by period.
    ic_t_stat : dict[str, float]
        T-statistic for IC != 0.
    ic_p_value : dict[str, float]
        P-value for IC significance.
    ic_ir : dict[str, float]
        Information Ratio (IC mean / IC std) by period.
    ic_positive_pct : dict[str, float]
        Percentage of periods with positive IC.
    ic_series : dict[str, list[float]]
        IC time series by period.
    quantile_returns : dict[str, dict[int, float]]
        Mean returns by period and quantile.
    spread : dict[str, float]
        Top minus bottom quantile spread.
    spread_t_stat : dict[str, float]
        T-statistic for spread.
    spread_p_value : dict[str, float]
        P-value for spread significance.
    monotonicity : dict[str, float]
        Rank correlation of quantile returns (how monotonic).
    turnover : dict[str, float] | None
        Mean turnover rate by period.
    autocorrelation : list[float] | None
        Factor autocorrelation at lags 1, 2, ...
    half_life : float | None
        Estimated signal half-life in periods.
    n_assets : int
        Number of unique assets.
    n_dates : int
        Number of unique dates.
    date_range : tuple[str, str]
        (first_date, last_date).
    periods : tuple[int, ...]
        Forward return periods analyzed.
    quantiles : int
        Number of quantiles used.
    """

    # IC metrics
    ic: dict[str, float]
    ic_std: dict[str, float]
    ic_t_stat: dict[str, float]
    ic_p_value: dict[str, float]
    ic_ir: dict[str, float] = field(default_factory=dict)  # Information Ratio (ic/ic_std)
    ic_positive_pct: dict[str, float] = field(default_factory=dict)  # % of positive ICs
    ic_series: dict[str, list[float]] = field(default_factory=dict)

    # Quantile metrics
    quantile_returns: dict[str, dict[int, float]] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    spread_t_stat: dict[str, float] = field(default_factory=dict)
    spread_p_value: dict[str, float] = field(default_factory=dict)
    monotonicity: dict[str, float] = field(default_factory=dict)

    # IC dates (per-period, for bridge to SignalICResult)
    ic_dates: dict[str, list[str]] = field(default_factory=dict)

    # Quantile detail (for bridge to QuantileAnalysisResult)
    quantile_returns_std: dict[str, dict[int, float]] = field(default_factory=dict)
    count_by_quantile: dict[int, int] = field(default_factory=dict)

    # Spread std (standard error of spread, for CI computation)
    spread_std: dict[str, float] = field(default_factory=dict)

    # Turnover (optional)
    turnover: dict[str, float] | None = None
    autocorrelation: list[float] | None = None
    half_life: float | None = None

    # Metadata
    n_assets: int = 0
    n_dates: int = 0
    date_range: tuple[str, str] = ("", "")
    periods: tuple[int, ...] = ()
    quantiles: int = 5

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Signal Analysis: {self.n_assets} assets, {self.n_dates} dates",
            f"Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"Periods: {self.periods}, Quantiles: {self.quantiles}",
            "",
            "IC Summary:",
        ]

        for period in [f"{p}D" for p in self.periods]:
            ic_val = self.ic.get(period, float("nan"))
            t = self.ic_t_stat.get(period, float("nan"))
            p = self.ic_p_value.get(period, float("nan"))
            ir = self.ic_ir.get(period, float("nan"))
            pos_pct = self.ic_positive_pct.get(period, float("nan"))
            sig = "*" if p < 0.05 else ""
            lines.append(
                f"  {period}: IC={ic_val:+.4f} (t={t:.2f}, p={p:.3f}){sig}, IR={ir:.2f}, +%={pos_pct:.0f}%"
            )

        lines.append("\nSpread (Top - Bottom):")
        for period in [f"{p}D" for p in self.periods]:
            spread = self.spread.get(period, float("nan"))
            t = self.spread_t_stat.get(period, float("nan"))
            p = self.spread_p_value.get(period, float("nan"))
            sig = "*" if p < 0.05 else ""
            lines.append(f"  {period}: {spread:+.4f} (t={t:.2f}, p={p:.3f}){sig}")

        lines.append("\nMonotonicity:")
        for period in [f"{p}D" for p in self.periods]:
            mono = self.monotonicity.get(period, float("nan"))
            lines.append(f"  {period}: {mono:+.3f}")

        if self.turnover:
            lines.append("\nTurnover:")
            for period in [f"{p}D" for p in self.periods]:
                t = self.turnover.get(period, float("nan"))
                lines.append(f"  {period}: {t:.1%}")

        if self.half_life is not None:
            lines.append(f"\nHalf-life: {self.half_life:.1f} periods")

        return "\n".join(lines)

    # =========================================================================
    # Bridge Methods (convert to Pydantic viz models)
    # =========================================================================

    def to_ic_result(self, period: int | str | None = None) -> SignalICResult:
        """Convert to SignalICResult for visualization functions.

        Parameters
        ----------
        period : int | str | None
            Specific period (e.g. 21 or "21D"). If None, includes all periods
            aligned to their common date intersection.

        Returns
        -------
        SignalICResult
            Pydantic model compatible with plot_ic_ts, plot_ic_histogram, etc.

        Raises
        ------
        ValueError
            If ic_dates is empty (result created without date capture).

        Examples
        --------
        >>> result = analyze_signal(factor_df, prices_df)
        >>> plot_ic_ts(result.to_ic_result())
        >>> plot_ic_ts(result.to_ic_result(period=21))
        """
        from ml4t.diagnostic.results.signal_results.ic import SignalICResult

        if not self.ic_dates:
            raise ValueError("ic_dates not available. Re-run analyze_signal() to capture dates.")

        period_keys: list[str]
        if period is not None:
            key = f"{period}D" if isinstance(period, int) else str(period)
            if not key.endswith("D"):
                key = f"{key}D"
            if key not in self.ic:
                raise ValueError(f"Period '{key}' not found. Available: {list(self.ic.keys())}")
            period_keys = [key]
        else:
            period_keys = [f"{p}D" for p in self.periods]

        # Find common date intersection across requested periods
        date_sets = [set(self.ic_dates[k]) for k in period_keys if k in self.ic_dates]
        if not date_sets:
            common_dates: list[str] = []
        else:
            common = date_sets[0]
            for ds in date_sets[1:]:
                common = common & ds
            common_dates = sorted(common)

        # Build ic_by_date aligned to common dates
        ic_by_date: dict[str, list[float]] = {}
        for key in period_keys:
            if key not in self.ic_dates or key not in self.ic_series:
                ic_by_date[key] = []
                continue
            date_to_ic = dict(zip(self.ic_dates[key], self.ic_series[key], strict=False))
            ic_by_date[key] = [date_to_ic[d] for d in common_dates if d in date_to_ic]

        return SignalICResult(
            ic_by_date=ic_by_date,
            dates=common_dates,
            ic_mean={k: self.ic[k] for k in period_keys},
            ic_std={k: self.ic_std[k] for k in period_keys},
            ic_t_stat={k: self.ic_t_stat[k] for k in period_keys},
            ic_p_value={k: self.ic_p_value[k] for k in period_keys},
            ic_positive_pct={k: self.ic_positive_pct.get(k, 0.0) for k in period_keys},
            ic_ir={k: self.ic_ir.get(k, 0.0) for k in period_keys},
        )

    def to_quantile_result(self) -> QuantileAnalysisResult:
        """Convert to QuantileAnalysisResult for visualization functions.

        Returns
        -------
        QuantileAnalysisResult
            Pydantic model compatible with plot_quantile_returns_bar, etc.

        Examples
        --------
        >>> result = analyze_signal(factor_df, prices_df)
        >>> plot_quantile_returns_bar(result.to_quantile_result())
        """
        from ml4t.diagnostic.results.signal_results.quantile import QuantileAnalysisResult

        period_keys = [f"{p}D" for p in self.periods]
        quantile_labels = [f"Q{i}" for i in range(1, self.quantiles + 1)]

        # Convert int quantile keys to string labels: {1: val} -> {"Q1": val}
        def _relabel(d: dict[int, float]) -> dict[str, float]:
            return {f"Q{k}": v for k, v in sorted(d.items())}

        mean_returns = {pk: _relabel(self.quantile_returns[pk]) for pk in period_keys}

        # std_returns: use captured data or fill with 0.0
        std_returns: dict[str, dict[str, float]] = {}
        for pk in period_keys:
            if pk in self.quantile_returns_std:
                std_returns[pk] = _relabel(self.quantile_returns_std[pk])
            else:
                std_returns[pk] = dict.fromkeys(quantile_labels, 0.0)

        # count_by_quantile with string labels
        count_by_q: dict[str, int]
        if self.count_by_quantile:
            count_by_q = {f"Q{k}": v for k, v in sorted(self.count_by_quantile.items())}
        else:
            count_by_q = dict.fromkeys(quantile_labels, 0)

        # Spread metrics
        spread_mean = {pk: self.spread.get(pk, 0.0) for pk in period_keys}
        spread_std_d = {pk: self.spread_std.get(pk, 0.0) for pk in period_keys}
        spread_t = {pk: self.spread_t_stat.get(pk, 0.0) for pk in period_keys}
        spread_p = {pk: self.spread_p_value.get(pk, 1.0) for pk in period_keys}

        # Confidence intervals: spread ± 1.96 * spread_std
        z = 1.96
        spread_ci_lower: dict[str, float] = {}
        spread_ci_upper: dict[str, float] = {}
        for pk in period_keys:
            s = spread_mean[pk]
            se = spread_std_d[pk]
            if math.isfinite(se):
                spread_ci_lower[pk] = s - z * se
                spread_ci_upper[pk] = s + z * se
            else:
                spread_ci_lower[pk] = float("nan")
                spread_ci_upper[pk] = float("nan")

        # Monotonicity derivation
        is_monotonic: dict[str, bool] = {}
        monotonicity_direction: dict[str, str] = {}
        rank_correlation: dict[str, float] = {}
        for pk in period_keys:
            rho = self.monotonicity.get(pk, 0.0)
            rank_correlation[pk] = rho
            is_monotonic[pk] = abs(rho) > 0.8
            if rho > 0.8:
                monotonicity_direction[pk] = "increasing"
            elif rho < -0.8:
                monotonicity_direction[pk] = "decreasing"
            else:
                monotonicity_direction[pk] = "none"

        return QuantileAnalysisResult(
            n_quantiles=self.quantiles,
            quantile_labels=quantile_labels,
            periods=period_keys,
            mean_returns=mean_returns,
            std_returns=std_returns,
            count_by_quantile=count_by_q,
            spread_mean=spread_mean,
            spread_std=spread_std_d,
            spread_t_stat=spread_t,
            spread_p_value=spread_p,
            spread_ci_lower=spread_ci_lower,
            spread_ci_upper=spread_ci_upper,
            is_monotonic=is_monotonic,
            monotonicity_direction=monotonicity_direction,
            rank_correlation=rank_correlation,
        )

    def to_tear_sheet(self, signal_name: str = "signal") -> SignalTearSheet:
        """Convert to full SignalTearSheet for dashboard display.

        Bundles to_ic_result() and to_quantile_result() into a SignalTearSheet.

        Parameters
        ----------
        signal_name : str
            Name for the signal (used in dashboard title).

        Returns
        -------
        SignalTearSheet
            Pydantic model with IC and quantile analysis components.

        Examples
        --------
        >>> result = analyze_signal(factor_df, prices_df)
        >>> tear_sheet = result.to_tear_sheet("momentum_21d")
        >>> tear_sheet.show()
        """
        from ml4t.diagnostic.results.signal_results.tearsheet import SignalTearSheet

        ic_result = self.to_ic_result() if self.ic_dates else None
        quantile_result = self.to_quantile_result()

        return SignalTearSheet(
            signal_name=signal_name,
            n_assets=self.n_assets,
            n_dates=self.n_dates,
            date_range=self.date_range,
            ic_analysis=ic_result,
            quantile_analysis=quantile_result,
        )

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        return asdict(self)

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """Export to JSON string or file.

        Parameters
        ----------
        path : str | None
            If provided, write to file. Otherwise return string.
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            JSON string.
        """
        data = self.to_dict()

        def convert(obj: Any) -> Any:
            if isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        def serialize(d: Any) -> Any:
            if isinstance(d, dict):
                return {str(k): serialize(v) for k, v in d.items()}
            if isinstance(d, list):
                return [serialize(v) for v in d]
            return convert(d)

        serialized = serialize(data)
        json_str = json.dumps(serialized, indent=indent)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, path: str) -> SignalResult:
        """Load from JSON file.

        Parameters
        ----------
        path : str
            Path to JSON file.

        Returns
        -------
        SignalResult
            Loaded result.
        """
        with open(path) as f:
            data = json.load(f)

        # Convert lists back to tuples for immutable fields
        if "date_range" in data:
            data["date_range"] = tuple(data["date_range"])
        if "periods" in data:
            data["periods"] = tuple(data["periods"])

        # Convert quantile keys back to int
        if "quantile_returns" in data:
            data["quantile_returns"] = {
                period: {int(k): v for k, v in qr.items()}
                for period, qr in data["quantile_returns"].items()
            }
        if "quantile_returns_std" in data:
            data["quantile_returns_std"] = {
                period: {int(k): v for k, v in qr.items()}
                for period, qr in data["quantile_returns_std"].items()
            }
        if "count_by_quantile" in data:
            data["count_by_quantile"] = {int(k): v for k, v in data["count_by_quantile"].items()}

        return cls(**data)


__all__ = ["SignalResult"]
