"""Signal analysis result dataclass.

Simple, immutable result container for signal analysis.
No Pydantic, no inheritance - just a frozen dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


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

        return cls(**data)


__all__ = ["SignalResult"]
