"""Internal utilities for signal analysis.

Simple, pure functions for data preparation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import polars as pl

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
    """Compute forward returns for each period using vectorized operations.

    For each (date, asset), computes return from date to date + period.
    Forward returns are computed using the factor data's date universe,
    so period N means "N dates forward in the factor dates", not calendar days.

    Parameters
    ----------
    data : pl.DataFrame
        Factor data with date and asset columns.
    prices : pl.DataFrame
        Price data with date, asset, and price columns.
    periods : tuple[int, ...]
        Forward return periods in trading days (factor date indices).
    date_col, asset_col, price_col : str
        Column names.

    Returns
    -------
    pl.DataFrame
        Data with forward return columns (e.g., "1D_fwd_return").
    """
    if data.is_empty():
        # Add empty columns for each period
        for p in periods:
            data = data.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{p}D_fwd_return"))
        return data

    # 1. Create date index mapping from FACTOR data (not prices)
    # This ensures forward returns align with factor date universe
    factor_dates = data.select(date_col).unique().sort(date_col)
    factor_dates = factor_dates.with_row_index("_factor_date_idx")

    # 2. Join data with current prices
    result = data.join(
        prices.select([date_col, asset_col, price_col]).rename({price_col: "_current_price"}),
        on=[date_col, asset_col],
        how="left",
    )

    # 3. Join to get factor date index for each row
    result = result.join(factor_dates, on=date_col, how="left")

    # 4. For each period, compute forward return via joins
    for p in periods:
        col_name = f"{p}D_fwd_return"

        # Create mapping: current_factor_idx -> future_factor_date
        # future_factor_idx = current_factor_idx + p
        future_date_map = factor_dates.with_columns(
            (pl.col("_factor_date_idx") - p).alias("_current_idx")
        ).filter(pl.col("_current_idx") >= 0)

        # Join to get future date (from factor date sequence)
        result = result.join(
            future_date_map.select([date_col, "_current_idx"]).rename(
                {date_col: f"_future_date_{p}"}
            ),
            left_on="_factor_date_idx",
            right_on="_current_idx",
            how="left",
        )

        # Join to get future price (from price data)
        result = result.join(
            prices.select([date_col, asset_col, price_col]).rename(
                {price_col: f"_future_price_{p}"}
            ),
            left_on=[f"_future_date_{p}", asset_col],
            right_on=[date_col, asset_col],
            how="left",
        )

        # Compute return: (future - current) / current
        # Handle NaN in current price (use is_nan check)
        result = result.with_columns(
            pl.when(
                pl.col("_current_price").is_not_null()
                & pl.col("_current_price").is_not_nan()
                & pl.col(f"_future_price_{p}").is_not_null()
                & pl.col(f"_future_price_{p}").is_not_nan()
                & (pl.col("_current_price") != 0)
            )
            .then(
                (pl.col(f"_future_price_{p}") - pl.col("_current_price")) / pl.col("_current_price")
            )
            .otherwise(None)
            .alias(col_name)
        )

    # 5. Clean up temporary columns
    temp_cols = [c for c in result.columns if c.startswith("_")]
    return result.drop(temp_cols)


__all__ = [
    "QuantileMethod",
    "ensure_polars",
    "filter_outliers",
    "quantize_factor",
    "compute_forward_returns",
]
