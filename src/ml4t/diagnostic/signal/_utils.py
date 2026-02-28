"""Internal utilities for signal analysis.

Simple, pure functions for data preparation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

from ml4t.diagnostic.evaluation.metrics.basic import (
    compute_forward_returns as compute_forward_returns_core,
)

if TYPE_CHECKING:
    import pandas as pd


class QuantileMethod(str, Enum):
    """Method for quantile assignment."""

    QUANTILE = "quantile"  # Equal frequency (rank-based)
    UNIFORM = "uniform"  # Equal width


def ensure_polars(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars if needed.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df
    # Pandas DataFrame
    return pl.from_pandas(df)


def filter_outliers(
    data: pl.DataFrame,
    z_threshold: float = 3.0,
    factor_col: str = "factor",
    date_col: str = "date",
) -> pl.DataFrame:
    """Filter outliers using cross-sectional z-score.

    Removes observations where factor z-score exceeds threshold
    within each date's cross-section.

    Parameters
    ----------
    data : pl.DataFrame
        Data with date and factor columns.
    z_threshold : float, default 3.0
        Z-score threshold. Values <= 0 disable filtering.
    factor_col : str, default "factor"
        Factor column name.
    date_col : str, default "date"
        Date column name.

    Returns
    -------
    pl.DataFrame
        Data with outliers removed.
    """
    if z_threshold <= 0:
        return data

    # Cross-sectional z-score with std=0 edge case
    data = data.with_columns(
        pl.when(pl.col(factor_col).std().over(date_col) > 0)
        .then(
            (pl.col(factor_col) - pl.col(factor_col).mean().over(date_col))
            / pl.col(factor_col).std().over(date_col)
        )
        .otherwise(pl.lit(None))
        .alias("_zscore")
    )

    # Keep rows within threshold or with null z-score (constant cross-section)
    data = data.filter(pl.col("_zscore").is_null() | (pl.col("_zscore").abs() <= z_threshold))
    return data.drop("_zscore")


def quantize_factor(
    data: pl.DataFrame,
    n_quantiles: int = 5,
    method: QuantileMethod = QuantileMethod.QUANTILE,
    factor_col: str = "factor",
    date_col: str = "date",
) -> pl.DataFrame:
    """Assign quantile labels to factor values within each date.

    Parameters
    ----------
    data : pl.DataFrame
        Data with date and factor columns.
    n_quantiles : int, default 5
        Number of quantiles.
    method : QuantileMethod, default QUANTILE
        QUANTILE = equal frequency, UNIFORM = equal width.
    factor_col : str, default "factor"
        Factor column name.
    date_col : str, default "date"
        Date column name.

    Returns
    -------
    pl.DataFrame
        Data with "quantile" column (1 = lowest, n = highest).
    """
    if method == QuantileMethod.QUANTILE:
        # Rank-based (equal count per quantile)
        data = data.with_columns(
            (
                (pl.col(factor_col).rank().over(date_col) - 1)
                / pl.col(factor_col).count().over(date_col)
                * n_quantiles
            )
            .floor()
            .cast(pl.Int32)
            .clip(0, n_quantiles - 1)
            .alias("_rank")
        )
        data = data.with_columns((pl.col("_rank") + 1).alias("quantile"))
        return data.drop("_rank")
    else:
        # Equal width
        data = data.with_columns(
            (
                (pl.col(factor_col) - pl.col(factor_col).min().over(date_col))
                / (
                    pl.col(factor_col).max().over(date_col)
                    - pl.col(factor_col).min().over(date_col)
                    + 1e-10
                )
                * n_quantiles
            )
            .floor()
            .cast(pl.Int32)
            .clip(0, n_quantiles - 1)
            .alias("_pct")
        )
        data = data.with_columns((pl.col("_pct") + 1).alias("quantile"))
        return data.drop("_pct")


def compute_forward_returns(
    data: pl.DataFrame,
    prices: pl.DataFrame,
    periods: tuple[int, ...],
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "price",
) -> pl.DataFrame:
    """Compute forward returns for each period.

    Forward returns are computed on the full price universe (per asset) using
    the shared core metrics implementation, then joined back to signal rows.
    This preserves trading-day semantics when factor dates are a subset of the
    available price dates.

    Parameters
    ----------
    data : pl.DataFrame
        Factor data with date and asset columns.
    prices : pl.DataFrame
        Price data with date, asset, and price columns.
    periods : tuple[int, ...]
        Forward return periods in trading days.
    date_col, asset_col, price_col : str
        Column names.

    Returns
    -------
    pl.DataFrame
        Data with forward return columns (e.g., "1D_fwd_return").
    """
    if data.is_empty():
        for p in periods:
            data = data.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{p}D_fwd_return"))
        return data

    # Compute forward returns on full price history for each asset/date.
    price_with_returns = compute_forward_returns_core(
        prices,
        periods=list(periods),
        price_col=price_col,
        group_col=asset_col,
        date_col=date_col,
    )

    # Keep only keys + forward-return columns, then join onto factor rows.
    join_cols = [date_col, asset_col]
    ret_cols = [f"fwd_ret_{p}" for p in periods]
    returns_lookup = price_with_returns.select(join_cols + ret_cols)

    result = data.join(returns_lookup, on=join_cols, how="left")

    # Match signal naming convention.
    rename_map = {f"fwd_ret_{p}": f"{p}D_fwd_return" for p in periods}
    result = result.rename(rename_map)

    # Preserve signal API semantics: invalid numeric returns are null, not NaN.
    return_cols = list(rename_map.values())
    for col in return_cols:
        result = result.with_columns(
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        )

    return result


__all__ = [
    "QuantileMethod",
    "ensure_polars",
    "filter_outliers",
    "quantize_factor",
    "compute_forward_returns",
]
