"""Basic metrics: hit rate and forward returns calculation.

This module provides fundamental building blocks for feature evaluation.
"""

from typing import TYPE_CHECKING, Union, cast

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter

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


def compute_forward_returns(
    prices: pl.DataFrame | pd.DataFrame,
    periods: int | list[int] = 1,
    price_col: str = "close",
    group_col: str | None = None,
) -> pl.DataFrame | pd.DataFrame:
    """Compute forward returns for given periods.

    This is a helper function for IC analysis, computing the forward-looking
    returns that will be correlated with predictions/features.

    Parameters
    ----------
    prices : Union[pl.DataFrame, pd.DataFrame]
        Price data with at least price_col and optionally group_col
    periods : Union[int, list[int]], default 1
        Forward periods to compute (e.g., [1, 5, 21] for 1d, 1w, 1m)
    price_col : str, default "close"
        Column name containing prices
    group_col : str | None, default None
        Column for grouping (e.g., 'symbol' for multi-asset)

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
    is_polars = isinstance(prices, pl.DataFrame)

    # Ensure periods is a list
    if isinstance(periods, int):
        periods = [periods]

    if is_polars:
        df = cast(pl.DataFrame, prices).clone()

        if group_col is not None:
            # Group-wise forward returns (e.g., per symbol)
            for period in periods:
                col_name = f"fwd_ret_{period}"
                df = df.with_columns(
                    [
                        (
                            pl.col(price_col).shift(-period).over(group_col) / pl.col(price_col) - 1
                        ).alias(col_name)
                    ]
                )
        else:
            # Simple forward returns
            for period in periods:
                col_name = f"fwd_ret_{period}"
                df = df.with_columns(
                    [(pl.col(price_col).shift(-period) / pl.col(price_col) - 1).alias(col_name)]
                )

        return df

    # pandas - use different variable name to avoid type conflict
    df_pd = cast(pd.DataFrame, prices).copy()

    if group_col is not None:
        # Group-wise forward returns
        for period in periods:
            col_name = f"fwd_ret_{period}"
            df_pd[col_name] = df_pd.groupby(group_col)[price_col].pct_change(period).shift(-period)
    else:
        # Simple forward returns
        for period in periods:
            col_name = f"fwd_ret_{period}"
            df_pd[col_name] = df_pd[price_col].pct_change(period).shift(-period)

    return df_pd
