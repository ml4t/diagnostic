"""Price path excursion analysis for TP/SL parameter selection.

This module provides tools to analyze price movement distributions over various
horizons, helping traders set take-profit and stop-loss levels based on
empirical price behavior.

**Key Distinction from Trade MFE/MAE**:

- **Trade MFE/MAE** (backtest library): Tracks best/worst unrealized return
  during actual trades. Used for exit efficiency analysis.

- **Price Excursion Analysis** (this module): Analyzes potential price movements
  over horizons BEFORE trading. Used for parameter selection (TP/SL levels).

Example workflow:
    >>> # 1. Analyze historical price movements
    >>> result = analyze_excursions(prices, horizons=[30, 60, 120])
    >>>
    >>> # 2. See distribution of movements
    >>> print(result.percentiles)
    >>>
    >>> # 3. Choose TP/SL based on percentiles
    >>> # e.g., 75th percentile MFE at 60 bars = 2.5% → use 2% take-profit
    >>> tp_level = result.get_percentile(horizon=60, percentile=75, side="mfe")
    >>>
    >>> # 4. Use these informed parameters in triple barrier labeling
    >>> from ml4t.engineer.labeling import triple_barrier_labels
    >>> labels = triple_barrier_labels(prices, upper_barrier=tp_level, ...)

Warning:
    ⚠️ FORWARD-LOOKING ANALYSIS
    This computes future price movements for parameter selection.
    DO NOT use excursion values as ML features (data leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


@dataclass
class ExcursionStats:
    """Statistics for excursions at a single horizon."""

    horizon: int
    n_samples: int

    # MFE (Maximum Favorable Excursion) stats
    mfe_mean: float
    mfe_std: float
    mfe_median: float
    mfe_skewness: float

    # MAE (Maximum Adverse Excursion) stats
    mae_mean: float
    mae_std: float
    mae_median: float
    mae_skewness: float

    # Percentiles (stored as dicts)
    mfe_percentiles: dict[float, float] = field(default_factory=dict)
    mae_percentiles: dict[float, float] = field(default_factory=dict)


@dataclass
class ExcursionAnalysisResult:
    """Result container for price excursion analysis.

    Attributes:
        horizons: List of horizons analyzed
        n_samples: Number of valid samples used
        return_type: Type of returns computed ('pct', 'log', 'abs')
        statistics: Per-horizon statistics
        percentile_matrix: DataFrame with horizons × percentiles
        excursions: Raw excursion values (optional, can be large)
        rolling_stats: Rolling statistics over time (optional)
    """

    horizons: list[int]
    n_samples: int
    return_type: str
    statistics: dict[int, ExcursionStats]
    percentile_matrix: pl.DataFrame
    excursions: pl.DataFrame | None = None
    rolling_stats: pl.DataFrame | None = None

    def get_percentile(self, horizon: int, percentile: float, side: Literal["mfe", "mae"]) -> float:
        """Get a specific percentile value.

        Args:
            horizon: The horizon to query
            percentile: Percentile (0-100)
            side: 'mfe' for favorable or 'mae' for adverse

        Returns:
            The percentile value

        Example:
            >>> result.get_percentile(horizon=60, percentile=75, side="mfe")
            0.025  # 75th percentile MFE at 60 bars is 2.5%
        """
        if horizon not in self.statistics:
            raise ValueError(f"Horizon {horizon} not in analysis. Available: {self.horizons}")

        stats = self.statistics[horizon]
        percentiles = stats.mfe_percentiles if side == "mfe" else stats.mae_percentiles

        if percentile not in percentiles:
            raise ValueError(
                f"Percentile {percentile} not computed. Available: {list(percentiles.keys())}"
            )

        return percentiles[percentile]

    def summary(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            "Price Excursion Analysis Summary",
            "=" * 40,
            f"Samples: {self.n_samples:,}",
            f"Return type: {self.return_type}",
            f"Horizons: {self.horizons}",
            "",
            "MFE (Maximum Favorable Excursion):",
        ]

        for h in self.horizons:
            stats = self.statistics[h]
            p50 = stats.mfe_percentiles.get(50, stats.mfe_median)
            p90 = stats.mfe_percentiles.get(90, 0)
            lines.append(f"  {h:3d} bars: median={p50:+.2%}, 90th={p90:+.2%}")

        lines.append("")
        lines.append("MAE (Maximum Adverse Excursion):")

        for h in self.horizons:
            stats = self.statistics[h]
            p50 = stats.mae_percentiles.get(50, stats.mae_median)
            p10 = stats.mae_percentiles.get(10, 0)
            lines.append(f"  {h:3d} bars: median={p50:+.2%}, 10th={p10:+.2%}")

        return "\n".join(lines)


def compute_excursions(
    prices: pl.Series | pd.Series | NDArray,
    horizons: list[int],
    return_type: Literal["pct", "log", "abs"] = "pct",
) -> pl.DataFrame:
    """Compute MFE/MAE for each horizon.

    For each bar t and horizon h:
    - MFE[t,h] = max(prices[t:t+h]) / prices[t] - 1  (for pct)
    - MAE[t,h] = min(prices[t:t+h]) / prices[t] - 1  (for pct)

    Args:
        prices: Price series (close prices typically)
        horizons: List of horizons to compute (e.g., [15, 30, 60])
        return_type: How to compute returns:
            - 'pct': Percentage returns (default)
            - 'log': Log returns
            - 'abs': Absolute price changes

    Returns:
        DataFrame with columns: mfe_{h}, mae_{h} for each horizon h

    Example:
        >>> prices = pl.Series([100, 102, 98, 105, 103, 101])
        >>> result = compute_excursions(prices, horizons=[2, 3])
        >>> print(result)
        shape: (3, 4)
        ┌──────────┬──────────┬──────────┬──────────┐
        │ mfe_2    ┆ mae_2    ┆ mfe_3    ┆ mae_3    │
        │ ---      ┆ ---      ┆ ---      ┆ ---      │
        │ f64      ┆ f64      ┆ f64      ┆ f64      │
        ╞══════════╪══════════╪══════════╪══════════╡
        │ 0.02     ┆ -0.02    ┆ 0.05     ┆ -0.02    │
        │ ...      ┆ ...      ┆ ...      ┆ ...      │
        └──────────┴──────────┴──────────┴──────────┘
    """
    # Convert to numpy for computation
    if isinstance(prices, pl.Series):
        price_array = prices.to_numpy()
    elif isinstance(prices, np.ndarray):
        price_array = prices
    elif hasattr(prices, "to_numpy"):  # pandas Series
        price_array = prices.to_numpy()
    else:
        price_array = np.asarray(prices)

    price_array = price_array.astype(np.float64)
    n = len(price_array)

    # Validate
    if n < max(horizons) + 1:
        raise ValueError(
            f"Price series too short ({n}) for max horizon ({max(horizons)}). Need at least {max(horizons) + 1} prices."
        )

    # Pre-allocate result arrays
    results = {}
    max_horizon = max(horizons)

    for h in horizons:
        mfe = np.full(n - max_horizon, np.nan)
        mae = np.full(n - max_horizon, np.nan)

        for i in range(n - max_horizon):
            entry_price = price_array[i]
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            window = price_array[i : i + h + 1]  # Include entry price
            if np.any(np.isnan(window)) or np.any(window <= 0):
                continue

            max_price = np.max(window)
            min_price = np.min(window)

            if return_type == "pct":
                mfe[i] = (max_price - entry_price) / entry_price
                mae[i] = (min_price - entry_price) / entry_price
            elif return_type == "log":
                mfe[i] = np.log(max_price / entry_price)
                mae[i] = np.log(min_price / entry_price)
            elif return_type == "abs":
                mfe[i] = max_price - entry_price
                mae[i] = min_price - entry_price
            else:
                raise ValueError(f"Unknown return_type: {return_type}")

        results[f"mfe_{h}"] = mfe
        results[f"mae_{h}"] = mae

    return pl.DataFrame(results)


def analyze_excursions(
    prices: pl.Series | pd.Series | NDArray,
    horizons: list[int] | None = None,
    return_type: Literal["pct", "log", "abs"] = "pct",
    percentiles: list[float] | None = None,
    keep_raw: bool = False,
    rolling_window: int | None = None,
) -> ExcursionAnalysisResult:
    """Analyze price excursions with statistics and percentiles.

    This is the main entry point for price excursion analysis. It computes
    MFE/MAE distributions and provides statistics useful for setting
    take-profit and stop-loss levels.

    Args:
        prices: Price series (close prices typically)
        horizons: List of horizons to analyze. Default: [15, 30, 60]
        return_type: How to compute returns ('pct', 'log', 'abs')
        percentiles: Percentiles to compute. Default: [10, 25, 50, 75, 90]
        keep_raw: If True, include raw excursion values in result
        rolling_window: If provided, compute rolling statistics over this window

    Returns:
        ExcursionAnalysisResult with statistics and percentiles

    Example:
        >>> import polars as pl
        >>> prices = pl.Series(np.random.randn(1000).cumsum() + 100)
        >>> result = analyze_excursions(prices, horizons=[30, 60, 120])
        >>>
        >>> # View summary
        >>> print(result.summary())
        >>>
        >>> # Get specific percentile for parameter selection
        >>> tp_level = result.get_percentile(horizon=60, percentile=75, side="mfe")
        >>> sl_level = result.get_percentile(horizon=60, percentile=25, side="mae")
        >>> print(f"Suggested TP: {tp_level:.2%}, SL: {sl_level:.2%}")
    """
    # Defaults
    if horizons is None:
        horizons = [15, 30, 60]
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    # Sort horizons
    horizons = sorted(horizons)

    # Compute raw excursions
    excursions = compute_excursions(prices, horizons, return_type)
    n_samples = len(excursions)

    # Compute statistics per horizon
    statistics = {}
    percentile_rows = []

    for h in horizons:
        mfe_col = f"mfe_{h}"
        mae_col = f"mae_{h}"

        mfe_values = excursions[mfe_col].drop_nulls().to_numpy()
        mae_values = excursions[mae_col].drop_nulls().to_numpy()

        # Skip if no valid data
        if len(mfe_values) == 0:
            continue

        # Compute percentiles
        mfe_pcts = {p: float(np.percentile(mfe_values, p)) for p in percentiles}
        mae_pcts = {p: float(np.percentile(mae_values, p)) for p in percentiles}

        # Compute statistics
        from scipy.stats import skew

        stats = ExcursionStats(
            horizon=h,
            n_samples=len(mfe_values),
            mfe_mean=float(np.mean(mfe_values)),
            mfe_std=float(np.std(mfe_values)),
            mfe_median=float(np.median(mfe_values)),
            mfe_skewness=float(skew(mfe_values)) if len(mfe_values) > 2 else 0.0,
            mae_mean=float(np.mean(mae_values)),
            mae_std=float(np.std(mae_values)),
            mae_median=float(np.median(mae_values)),
            mae_skewness=float(skew(mae_values)) if len(mae_values) > 2 else 0.0,
            mfe_percentiles=mfe_pcts,
            mae_percentiles=mae_pcts,
        )
        statistics[h] = stats

        # Build percentile matrix row
        row = {"horizon": h, "side": "mfe"}
        row.update({f"p{int(p)}": v for p, v in mfe_pcts.items()})
        percentile_rows.append(row)

        row = {"horizon": h, "side": "mae"}
        row.update({f"p{int(p)}": v for p, v in mae_pcts.items()})
        percentile_rows.append(row)

    percentile_matrix = pl.DataFrame(percentile_rows)

    # Compute rolling stats if requested
    rolling_stats = None
    if rolling_window is not None:
        rolling_stats = _compute_rolling_excursion_stats(excursions, horizons, rolling_window)

    return ExcursionAnalysisResult(
        horizons=horizons,
        n_samples=n_samples,
        return_type=return_type,
        statistics=statistics,
        percentile_matrix=percentile_matrix,
        excursions=excursions if keep_raw else None,
        rolling_stats=rolling_stats,
    )


def _compute_rolling_excursion_stats(
    excursions: pl.DataFrame, horizons: list[int], window: int
) -> pl.DataFrame:
    """Compute rolling statistics for excursions.

    This allows seeing how excursion distributions change over time,
    useful for detecting regime changes.
    """
    results = []

    for h in horizons:
        mfe_col = f"mfe_{h}"
        mae_col = f"mae_{h}"

        # Rolling median and std
        rolling_df = excursions.select(
            [
                pl.col(mfe_col).rolling_median(window).alias(f"mfe_{h}_median"),
                pl.col(mfe_col).rolling_std(window).alias(f"mfe_{h}_std"),
                pl.col(mae_col).rolling_median(window).alias(f"mae_{h}_median"),
                pl.col(mae_col).rolling_std(window).alias(f"mae_{h}_std"),
            ]
        )
        results.append(rolling_df)

    # Combine all horizons
    if results:
        return pl.concat(results, how="horizontal")
    return pl.DataFrame()
