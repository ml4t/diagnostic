"""Basic metrics: hit rate and forward returns calculation.

This module provides fundamental building blocks for feature evaluation.
"""

from typing import TYPE_CHECKING, Union, cast

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.utils.validation import _validate_observation_keys

if TYPE_CHECKING:
    from numpy.typing import NDArray


def hit_rate(
    predictions: Union[pl.Series, pd.Series, "NDArray"],
    returns: Union[pl.Series, pd.Series, "NDArray"],
) -> float:
    """Calculate hit rate (percentage of correct directional predictions).

    Hit rate measures what percentage of predictions correctly identify the
    direction of subsequent returns (positive/negative).

    Parameters
    ----------
    predictions : Union[pl.Series, pd.Series, np.ndarray]
        Model predictions or scores
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Forward returns corresponding to predictions

    Returns
    -------
    float
        Hit rate as a percentage (0-100)

    Examples
    --------
    >>> predictions = np.array([0.1, -0.2, 0.3, -0.1])
    >>> returns = np.array([0.02, -0.01, 0.05, 0.01])  # Note: last one wrong direction
    >>> hr = hit_rate(predictions, returns)
    >>> print(f"Hit Rate: {hr:.1f}%")
    Hit Rate: 75.0%
    """
    # Convert inputs to numpy
    pred_array = DataFrameAdapter.to_numpy(predictions).flatten()
    ret_array = DataFrameAdapter.to_numpy(returns).flatten()

    # Validate inputs
    if len(pred_array) != len(ret_array):
        raise ValueError("Predictions and returns must have the same length")

    # Remove NaN pairs
    valid_mask = ~(np.isnan(pred_array) | np.isnan(ret_array))
    pred_clean = pred_array[valid_mask]
    ret_clean = ret_array[valid_mask]

    if len(pred_clean) == 0:
        return np.nan

    # Calculate directional accuracy
    pred_direction = np.sign(pred_clean)
    ret_direction = np.sign(ret_clean)

    # Count correct predictions (same sign)
    correct_predictions = pred_direction == ret_direction

    # Handle zero returns/predictions by considering them neutral (correct)
    zero_mask = (pred_clean == 0) | (ret_clean == 0)
    correct_predictions[zero_mask] = True  # Conservative approach

    hit_rate_value = np.mean(correct_predictions) * 100

    return float(hit_rate_value)


def _relative_price_return(
    exit_price: pl.Expr, reference_price: pl.Expr, *, integer_prices: bool
) -> pl.Expr:
    if integer_prices:
        difference = exit_price.cast(pl.Decimal(38, 0)) - reference_price.cast(pl.Decimal(38, 0))
        return difference.cast(pl.Float64) / reference_price.cast(pl.Float64)
    exit_value = exit_price.cast(pl.Float64)
    reference = reference_price.cast(pl.Float64)
    difference = exit_value - reference
    return (
        pl.when(difference.is_finite())
        .then(difference / reference)
        .otherwise(exit_value / reference - 1)
    )


def compute_forward_returns(
    prices: pl.DataFrame | pd.DataFrame,
    periods: int | list[int] = 1,
    price_col: str = "close",
    group_col: str | None = None,
    date_col: str | None = "date",
    output_col_template: str = "fwd_ret_{period}",
    nan_to_null: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Compute forward returns for given periods.

    Row shifts are within each asset after sorting, not independently validated
    trading days or bars. Returns use Float64; integer differences are taken
    before floating conversion so one-tick changes are not rounded to zero.

    Parameters
    ----------
    prices : Union[pl.DataFrame, pd.DataFrame]
        Price data with at least price_col and optionally group_col
    periods : Union[int, list[int]], default 1
        Forward observation counts within each asset (e.g., [1, 5, 21]).
    price_col : str, default "close"
        Column name containing prices
    group_col : str | None, default None
        Column for grouping (e.g., 'symbol' for multi-asset)
    date_col : str | None, default "date"
        Optional date/timestamp column used to sort before computing
        forward returns. Present dated keys must be unique within each group.
        Set to None to use input row order without timestamp-key uniqueness.
    output_col_template : str, default "fwd_ret_{period}"
        Template for generated forward return column names.
        Use ``{period}`` placeholder for the horizon integer.
    nan_to_null : bool, default False
        Convert NaN forward return values to null/None in Polars output.

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        DataFrame with forward return columns: fwd_ret_1, fwd_ret_5, etc.

    Examples
    --------
    >>> prices = pl.DataFrame({
    ...     "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    ...     "close": [100.0, 102.0, 101.0]
    ... })
    >>> fwd_returns = compute_forward_returns(prices, periods=[1, 2])
    >>> print(fwd_returns.columns)
    ['date', 'close', 'fwd_ret_1', 'fwd_ret_2']
    """
    output_as_pandas = isinstance(prices, pd.DataFrame)

    # Ensure periods is a list
    if isinstance(periods, int):
        periods = [periods]

    df = (
        prices.clone()
        if isinstance(prices, pl.DataFrame)
        else pl.from_pandas(cast(pd.DataFrame, prices))
    )

    keys = [group_col] if group_col is not None else []
    dated = date_col is not None and date_col in df.columns
    if dated:
        keys.insert(0, date_col)
    _validate_observation_keys(df, keys, "price", unique=dated)

    if date_col is not None and date_col in df.columns:
        if group_col is not None and group_col in df.columns:
            df = df.sort([group_col, date_col])
        else:
            df = df.sort(date_col)

    if group_col is not None:
        # Group-wise forward returns
        return_cols = []
        for period in periods:
            col_name = output_col_template.format(period=period)
            return_cols.append(col_name)
            df = df.with_columns(
                _relative_price_return(
                    pl.col(price_col).shift(-period).over(group_col),
                    pl.col(price_col),
                    integer_prices=df.schema[price_col].is_integer(),
                ).alias(col_name)
            )
    else:
        # Simple forward returns
        return_cols = []
        for period in periods:
            col_name = output_col_template.format(period=period)
            return_cols.append(col_name)
            df = df.with_columns(
                _relative_price_return(
                    pl.col(price_col).shift(-period),
                    pl.col(price_col),
                    integer_prices=df.schema[price_col].is_integer(),
                ).alias(col_name)
            )

    if nan_to_null:
        df = df.with_columns(
            [
                pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
                for c in return_cols
            ]
        )

    if output_as_pandas:
        return df.to_pandas()
    return df
