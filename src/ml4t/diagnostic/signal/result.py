"""Signal analysis result dataclass.

Simple, immutable result container for signal analysis.
No Pydantic, no inheritance - just a frozen dataclass.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any

from ml4t.diagnostic.utils.formatting import format_finite

if TYPE_CHECKING:
    import polars as pl

    from ml4t.diagnostic.results.signal_results.ic import SignalICResult
    from ml4t.diagnostic.results.signal_results.quantile import QuantileAnalysisResult
    from ml4t.diagnostic.results.signal_results.tearsheet import SignalTearSheet
    from ml4t.diagnostic.results.signal_results.turnover import TurnoverAnalysisResult


_OWNED_FRAMES = (
    ("_prepared_data", "prepared_data"),
    ("_dated_turnover", "dated_turnover"),
    ("_dated_autocorrelation", "dated_autocorrelation"),
    ("_quantile_observations", "quantile_observations"),
)


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
        Mean turnover rate at each period's lag in observed factor dates.
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
    horizon_labels: tuple[str, ...] = ()
    quantile_ids: tuple[int, ...] = ()
    preparation_metadata: dict[str, Any] = field(default_factory=dict)
    _prepared_data: bytes | None = field(default=None, repr=False, compare=False)
    _dated_turnover: bytes | None = field(default=None, repr=False, compare=False)
    _dated_autocorrelation: bytes | None = field(default=None, repr=False, compare=False)
    _quantile_observations: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def period_keys(self) -> tuple[str, ...]:
        """Exact analysis labels, with compatibility for older integer-day results."""
        return self.horizon_labels or tuple(f"{period}D" for period in self.periods)

    @property
    def prepared_data(self) -> pl.DataFrame | None:
        """Return an independent prepared snapshot, when this result owns one."""
        if self._prepared_data is None:
            return None
        import polars as pl

        return pl.read_ipc(BytesIO(self._prepared_data))

    @property
    def dated_turnover(self) -> pl.DataFrame | None:
        """Return owned entrants/current observations, not native scalar means."""
        if self._dated_turnover is None:
            return None
        import polars as pl

        return pl.read_ipc(BytesIO(self._dated_turnover))

    @property
    def dated_autocorrelation(self) -> pl.DataFrame | None:
        """Return owned full-date-rank Pearson observations and actual counts."""
        if self._dated_autocorrelation is None:
            return None
        import polars as pl

        return pl.read_ipc(BytesIO(self._dated_autocorrelation))

    @property
    def quantile_observations(self) -> pl.DataFrame | None:
        """Return actual dated bucket means/counts, not asset samples or simulations."""
        if self._quantile_observations is None:
            return None
        import polars as pl

        return pl.read_ipc(BytesIO(self._quantile_observations))

    def to_turnover_result(self) -> TurnoverAnalysisResult | None:
        """Convert actual dated observations without synthesizing data from means.

        Native scalar summaries retain their separate definitions. Older results
        without dated observations and analyses with persistence disabled return
        None; neither missing dates/buckets nor reference ranks can be recovered
        from their scalar summaries.
        """
        turnover, rank = self.dated_turnover, self.dated_autocorrelation
        if turnover is None or rank is None:
            return None
        from ml4t.diagnostic.results.signal_results.turnover import TurnoverAnalysisResult

        return TurnoverAnalysisResult.from_observations(turnover, rank)

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Signal Analysis: {self.n_assets} assets, {self.n_dates} dates",
            f"Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"Periods: {self.periods}, Quantiles: {self.quantiles}"
            if self.periods
            else f"Horizons: {self.period_keys}, Quantiles: {self.quantiles}",
            "",
            "IC Summary:",
        ]

        for period in self.period_keys:
            ic_val = self.ic.get(period, float("nan"))
            t = self.ic_t_stat.get(period, float("nan"))
            p = self.ic_p_value.get(period, float("nan"))
            ir = self.ic_ir.get(period, float("nan"))
            pos_pct = self.ic_positive_pct.get(period, float("nan"))
            sig = "*" if p < 0.05 else ""
            ic_display = format_finite(ic_val, "+.4f")
            t_display = format_finite(t, ".2f")
            p_display = format_finite(p, ".3f")
            ir_display = format_finite(ir, ".2f")
            pos_display = format_finite(pos_pct / 100, ".0%")
            lines.append(
                f"  {period}: IC={ic_display} (t={t_display}, p={p_display}){sig}, "
                f"IR={ir_display}, +%={pos_display}"
            )

        lines.append("\nSpread (Top - Bottom):")
        for period in self.period_keys:
            spread = self.spread.get(period, float("nan"))
            t = self.spread_t_stat.get(period, float("nan"))
            p = self.spread_p_value.get(period, float("nan"))
            sig = "*" if p < 0.05 else ""
            spread_display = format_finite(spread, "+.4f")
            t_display = format_finite(t, ".2f")
            p_display = format_finite(p, ".3f")
            lines.append(f"  {period}: {spread_display} (t={t_display}, p={p_display}){sig}")

        lines.append("\nMonotonicity:")
        for period in self.period_keys:
            mono = self.monotonicity.get(period, float("nan"))
            lines.append(f"  {period}: {format_finite(mono, '+.3f')}")

        if self.turnover:
            lines.append("\nTurnover:")
            for period in self.turnover:
                t = self.turnover.get(period, float("nan"))
                lines.append(f"  {period}: {format_finite(t, '.1%')}")

        if self.half_life is not None:
            lines.append(f"\nHalf-life: {format_finite(self.half_life, '.1f')} periods")

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
            Its statistics describe the retained common-date observations, not
            the original per-horizon samples. Select one period to retain that
            horizon's complete finite IC sample. Empty intersections have
            unavailable summaries; singletons retain only defined point values.

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
        from ml4t.diagnostic.signal.signal_ic import compute_ic_summary

        if not self.ic_dates:
            raise ValueError("ic_dates not available. Re-run analyze_signal() to capture dates.")

        period_keys: list[str]
        if period is not None:
            key = f"{period}D" if isinstance(period, int) else str(period)
            if key not in self.ic and key.isdecimal():
                key = f"{key}D"
            if key not in self.ic:
                raise ValueError(f"Period '{key}' not found. Available: {list(self.ic.keys())}")
            period_keys = [key]
        else:
            period_keys = list(self.period_keys)

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

        summaries = {key: compute_ic_summary(values) for key, values in ic_by_date.items()}
        return SignalICResult(
            ic_by_date=ic_by_date,
            dates=common_dates,
            ic_mean={key: summary["mean"] for key, summary in summaries.items()},
            ic_std={key: summary["std"] for key, summary in summaries.items()},
            ic_t_stat={key: summary["t_stat"] for key, summary in summaries.items()},
            ic_p_value={key: summary["p_value"] for key, summary in summaries.items()},
            ic_positive_pct={
                key: summary["pct_positive"] * 100 for key, summary in summaries.items()
            },
            ic_ir={
                key: summary["mean"] / summary["std"] if summary["std"] > 0 else float("nan")
                for key, summary in summaries.items()
            },
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

        period_keys = self.period_keys
        bucket_ids = self.quantile_ids or tuple(range(1, self.quantiles + 1))
        quantile_labels = [f"Q{i}" for i in bucket_ids]

        # Convert int quantile keys to string labels: {1: val} -> {"Q1": val}
        def _relabel(d: dict[int, float]) -> dict[str, float]:
            return {f"Q{k}": v for k, v in sorted(d.items())}

        mean_returns = {pk: _relabel(self.quantile_returns[pk]) for pk in period_keys}

        # std_returns: use captured data or mark unavailable values as NaN
        std_returns: dict[str, dict[str, float]] = {}
        for pk in period_keys:
            if pk in self.quantile_returns_std:
                std_returns[pk] = _relabel(self.quantile_returns_std[pk])
            else:
                std_returns[pk] = dict.fromkeys(quantile_labels, float("nan"))

        # count_by_quantile with string labels
        count_by_q: dict[str, int]
        count_by_q = {f"Q{k}": self.count_by_quantile.get(k, 0) for k in bucket_ids}

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
            quantile_ids=list(bucket_ids),
            periods=list(period_keys),
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
            observations_ipc_base64=base64.b64encode(self._quantile_observations).decode("ascii")
            if self._quantile_observations is not None
            else None,
        )

    def to_tear_sheet(self, signal_name: str = "signal") -> SignalTearSheet:
        """Convert available analysis components for dashboard display.

        Bundle IC, quantile summaries and actual dated persistence when retained.
        This does not establish cumulative paths or complete legacy tear sheets.

        Parameters
        ----------
        signal_name : str
            Name for the signal (used in dashboard title).

        Returns
        -------
        SignalTearSheet
            Pydantic model with available IC, quantile and persistence components.

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
            turnover_analysis=self.to_turnover_result(),
        )

    # =========================================================================
    # Export Methods
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary."""
        data = asdict(self)
        for private, public in _OWNED_FRAMES:
            payload = data.pop(private)
            if payload is not None:
                data[public] = {
                    "format": "arrow-ipc-base64",
                    "data": base64.b64encode(payload).decode("ascii"),
                }
        if not data["preparation_metadata"]:
            data.pop("preparation_metadata")
        return data

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
            with open(path, "w", encoding="utf-8") as f:
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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for private, public in _OWNED_FRAMES:
            if private in data:
                raise ValueError(f"private {public} data is not a supported JSON field")
            encoded = data.pop(public, None)
            if encoded is not None:
                if not isinstance(encoded, dict) or encoded.get("format") != "arrow-ipc-base64":
                    raise ValueError(f"unsupported {public} format")
                import polars as pl

                try:
                    payload = base64.b64decode(encoded["data"], validate=True)
                    pl.read_ipc(BytesIO(payload))
                except (KeyError, ValueError, TypeError, OSError, pl.exceptions.PolarsError) as exc:
                    raise ValueError(f"invalid {public} payload") from exc
                data[private] = payload
        if (data.get("_dated_turnover") is None) != (data.get("_dated_autocorrelation") is None):
            raise ValueError(
                "dated persistence requires both turnover and autocorrelation observations"
            )

        # Convert lists back to tuples for immutable fields
        if "date_range" in data:
            data["date_range"] = tuple(data["date_range"])
        if "periods" in data:
            data["periods"] = tuple(data["periods"])
        for name in ("horizon_labels", "quantile_ids"):
            if name in data:
                data[name] = tuple(data[name])

        def restore_numeric(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: restore_numeric(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore_numeric(item) for item in value]
            return float("nan") if value is None else value

        for name in (
            "ic",
            "ic_std",
            "ic_t_stat",
            "ic_p_value",
            "ic_ir",
            "ic_positive_pct",
            "ic_series",
            "quantile_returns",
            "quantile_returns_std",
            "spread",
            "spread_t_stat",
            "spread_p_value",
            "spread_std",
            "monotonicity",
            "turnover",
            "autocorrelation",
        ):
            if name in data and data[name] is not None:
                data[name] = restore_numeric(data[name])

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
