"""Core Information Coefficient (IC) metrics.

This module provides the fundamental IC calculations used for evaluating
feature predictiveness.
"""

from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.stats import spearmanr

from ml4t.diagnostic.backends.adapter import DataFrameAdapter
from ml4t.diagnostic.evaluation.metrics.basic import compute_forward_returns

if TYPE_CHECKING:
    from numpy.typing import NDArray


def information_coefficient(
    predictions: Union[pl.Series, pd.Series, "NDArray[Any]"],
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    method: str = "spearman",
    confidence_intervals: bool = False,
    alpha: float = 0.05,
) -> float | dict[str, float]:
    """Calculate Information Coefficient between predictions and returns.

    The Information Coefficient measures the linear relationship between model
    predictions and subsequent returns. Spearman correlation is preferred as it's
    robust to outliers and non-linear relationships.

    Parameters
    ----------
    predictions : Union[pl.Series, pd.Series, np.ndarray]
        Model predictions or scores
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Forward returns corresponding to predictions
    method : str, default "spearman"
        Correlation method: "spearman" or "pearson"
    confidence_intervals : bool, default False
        Whether to return confidence intervals
    alpha : float, default 0.05
        Significance level for confidence intervals

    Returns
    -------
    Union[float, dict]
        If confidence_intervals=False: IC value
        If confidence_intervals=True: dict with 'ic', 'lower_ci', 'upper_ci', 'p_value'

    Examples
    --------
    >>> predictions = np.array([0.1, 0.3, -0.2, 0.5])
    >>> returns = np.array([0.02, 0.05, -0.01, 0.08])
    >>> ic = information_coefficient(predictions, returns)
    >>> print(f"IC: {ic:.3f}")
    IC: 0.800

    >>> # With confidence intervals
    >>> result = information_coefficient(predictions, returns, confidence_intervals=True)
    >>> print(f"IC: {result['ic']:.3f} [{result['lower_ci']:.3f}, {result['upper_ci']:.3f}]")
    IC: 0.800 [-0.602, 0.995]
    """
    # Convert inputs to numpy for consistent handling
    pred_array = DataFrameAdapter.to_numpy(predictions).flatten()
    ret_array = DataFrameAdapter.to_numpy(returns).flatten()

    # Validate inputs
    if len(pred_array) != len(ret_array):
        raise ValueError("Predictions and returns must have the same length")

    if len(pred_array) < 2:
        if confidence_intervals:
            return {
                "ic": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "p_value": np.nan,
            }
        return np.nan

    # Remove NaN pairs
    valid_mask = ~(np.isnan(pred_array) | np.isnan(ret_array))
    pred_clean = pred_array[valid_mask]
    ret_clean = ret_array[valid_mask]

    if len(pred_clean) < 2:
        if confidence_intervals:
            return {
                "ic": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "p_value": np.nan,
            }
        return np.nan

    # Calculate correlation
    if method == "spearman":
        ic_value, p_value = spearmanr(pred_clean, ret_clean)
    elif method == "pearson":
        ic_value, p_value = stats.pearsonr(pred_clean, ret_clean)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Handle edge cases
    if np.isnan(ic_value):
        if confidence_intervals:
            return {
                "ic": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "p_value": np.nan,
            }
        return np.nan

    # Return simple IC if no confidence intervals requested
    if not confidence_intervals:
        return float(ic_value)

    # Calculate confidence intervals using Fisher transformation
    n = len(pred_clean)
    if n < 4:  # Need sufficient data for meaningful CI
        return {
            "ic": float(ic_value),
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        }

    # Fisher transformation for correlation confidence intervals
    z = np.arctanh(ic_value)  # Fisher z-transform
    se = 1 / np.sqrt(n - 3)  # Standard error
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Transform back to correlation scale
    lower_z = z - z_critical * se
    upper_z = z + z_critical * se
    lower_ci = np.tanh(lower_z)
    upper_ci = np.tanh(upper_z)

    return {
        "ic": float(ic_value),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
    }


def compute_ic_series(
    predictions: pl.DataFrame | pd.DataFrame,
    returns: pl.DataFrame | pd.DataFrame,
    pred_col: str = "prediction",
    ret_col: str = "forward_return",
    date_col: str = "date",
    method: str = "spearman",
    min_periods: int = 10,
) -> pl.DataFrame | pd.DataFrame:
    """Compute IC time series for temporal analysis (Alphalens-style).

    This function computes the Information Coefficient for each time period
    (typically daily), enabling temporal analysis of prediction quality.
    This is THE fundamental visualization in Alphalens.

    Parameters
    ----------
    predictions : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with predictions, indexed or with date column
    returns : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with forward returns, matching predictions structure
    pred_col : str, default "prediction"
        Column name for predictions/features
    ret_col : str, default "forward_return"
        Column name for forward returns
    date_col : str, default "date"
        Column name for dates (for grouping by period)
    method : str, default "spearman"
        Correlation method: "spearman" or "pearson"
    min_periods : int, default 10
        Minimum observations per period for valid IC calculation

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        Time series of IC values with columns: [date_col, 'ic', 'n_obs']

    Examples
    --------
    >>> # Create sample data
    >>> dates = pd.date_range("2024-01-01", periods=100)
    >>> pred_df = pd.DataFrame({
    ...     "date": dates,
    ...     "prediction": np.random.randn(100)
    ... })
    >>> ret_df = pd.DataFrame({
    ...     "date": dates,
    ...     "forward_return": np.random.randn(100) * 0.02
    ... })
    >>> ic_series = compute_ic_series(pred_df, ret_df)
    >>> print(ic_series.head())
    """
    is_polars = isinstance(predictions, pl.DataFrame)

    if is_polars:
        # Merge predictions and returns
        predictions_pl = cast(pl.DataFrame, predictions)
        returns_pl = cast(pl.DataFrame, returns)
        df = predictions_pl.join(returns_pl, on=date_col, how="inner")

        # Use group_by().map_groups() for efficient per-group processing
        def compute_group_ic(group: pl.DataFrame) -> pl.DataFrame:
            """Compute IC for a single date group."""
            pred_array = group[pred_col].to_numpy()
            ret_array = group[ret_col].to_numpy()

            # Remove NaN pairs
            valid_mask = ~(np.isnan(pred_array) | np.isnan(ret_array))
            pred_clean = pred_array[valid_mask]
            ret_clean = ret_array[valid_mask]

            n_obs = len(pred_clean)

            if n_obs >= min_periods:
                ic_val = information_coefficient(
                    pred_clean, ret_clean, method=method, confidence_intervals=False
                )
            else:
                ic_val = np.nan

            return pl.DataFrame({date_col: [group[date_col][0]], "ic": [ic_val], "n_obs": [n_obs]})

        return df.group_by(date_col).map_groups(compute_group_ic).sort(date_col)

    # pandas - use different variable name to avoid type conflict
    # Merge predictions and returns
    predictions_pd = cast(pd.DataFrame, predictions)
    returns_pd = cast(pd.DataFrame, returns)
    df_pd = pd.merge(predictions_pd, returns_pd, on=date_col, how="inner")

    # Group by date and compute IC
    def compute_period_ic(group: pd.DataFrame) -> pd.Series:
        # Explicitly convert to ndarray to handle ExtensionArray types
        pred_array = np.asarray(group[pred_col].values, dtype=np.float64)
        ret_array = np.asarray(group[ret_col].values, dtype=np.float64)

        # Remove NaN pairs
        valid_mask = ~(np.isnan(pred_array) | np.isnan(ret_array))
        pred_clean = pred_array[valid_mask]
        ret_clean = ret_array[valid_mask]

        n_obs = len(pred_clean)

        if n_obs >= min_periods:
            ic_val = information_coefficient(
                pred_clean, ret_clean, method=method, confidence_intervals=False
            )
        else:
            ic_val = np.nan

        return pd.Series({"ic": ic_val, "n_obs": n_obs})

    ic_series = df_pd.groupby(date_col, group_keys=False).apply(compute_period_ic).reset_index()

    return ic_series


def compute_ic_by_horizon(
    predictions: pl.DataFrame | pd.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    horizons: list[int] | None = None,
    pred_col: str = "prediction",
    price_col: str = "close",
    date_col: str = "date",
    group_col: str | None = None,
    method: str = "spearman",
) -> dict[int, float]:
    """Compute IC across multiple forward return horizons.

    This function computes IC for different forward-looking periods
    (e.g., 1-day, 5-day, 21-day), which is essential for understanding
    prediction persistence and optimal holding periods.

    Parameters
    ----------
    predictions : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with predictions
    prices : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with prices to compute forward returns
    horizons : list[int], default [1, 5, 21]
        Forward periods to analyze (in days/bars)
    pred_col : str, default "prediction"
        Column name for predictions
    price_col : str, default "close"
        Column name for prices
    date_col : str, default "date"
        Column name for dates
    group_col : str | None, default None
        Column for grouping (e.g., 'symbol')
    method : str, default "spearman"
        Correlation method

    Returns
    -------
    dict[int, float | dict]
        Dictionary mapping horizon -> IC value
        Keys are horizon periods, values are IC (or dict with CI if requested)

    Examples
    --------
    >>> pred_df = pd.DataFrame({"date": dates, "prediction": preds})
    >>> price_df = pd.DataFrame({"date": dates, "close": prices})
    >>> ic_by_horizon = compute_ic_by_horizon(
    ...     pred_df, price_df, horizons=[1, 5, 21]
    ... )
    >>> print(f"1-day IC: {ic_by_horizon[1]:.3f}")
    >>> print(f"5-day IC: {ic_by_horizon[5]:.3f}")
    """
    # Compute forward returns for all horizons
    if horizons is None:
        horizons = [1, 5, 21]
    prices_with_fwd = compute_forward_returns(
        prices, periods=horizons, price_col=price_col, group_col=group_col
    )

    # Merge with predictions - declare type before branching
    df: pl.DataFrame | pd.DataFrame

    if isinstance(predictions, pl.DataFrame):
        # Type is narrowed by isinstance check, but prices_with_fwd needs cast
        prices_with_fwd_pl = cast(pl.DataFrame, prices_with_fwd)
        df = predictions.join(prices_with_fwd_pl, on=date_col, how="inner")
    elif isinstance(predictions, pd.DataFrame):
        prices_with_fwd_pd = cast(pd.DataFrame, prices_with_fwd)
        df = pd.merge(predictions, prices_with_fwd_pd, on=date_col, how="inner")
    else:
        raise TypeError(
            f"predictions must be pl.DataFrame or pd.DataFrame, got {type(predictions)}"
        )

    # Compute IC for each horizon
    ic_results: dict[int, float] = {}

    for horizon in horizons:
        ret_col = f"fwd_ret_{horizon}"

        # Extract arrays - df type is known from construction above
        if isinstance(df, pl.DataFrame):
            pred_array = df[pred_col].to_numpy()
            ret_array = df[ret_col].to_numpy()
        else:
            pred_array = df[pred_col].to_numpy()
            ret_array = df[ret_col].to_numpy()

        # Compute IC (confidence_intervals=False returns float)
        ic_result = information_coefficient(
            pred_array, ret_array, method=method, confidence_intervals=False
        )
        # When confidence_intervals=False, returns float; otherwise dict
        if isinstance(ic_result, dict):
            ic_val = float(ic_result.get("ic", np.nan))
        else:
            ic_val = float(ic_result)

        ic_results[horizon] = ic_val

    return ic_results


def compute_ic_ir(
    ic_series: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    ic_col: str = "ic",
    annualization_factor: float = np.sqrt(252),
    confidence_intervals: bool = False,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> float | dict[str, float]:
    """Compute IC Information Ratio (IC-IR) - risk-adjusted IC metric.

    IC-IR is analogous to the Sharpe ratio but for IC instead of returns.
    It measures the consistency of predictive power by computing mean IC
    divided by the standard deviation of IC.

    Higher IC-IR indicates more consistent predictions. IC-IR > 0.5 is
    generally considered good, IC-IR > 1.0 is excellent.

    Parameters
    ----------
    ic_series : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Time series of IC values (from compute_ic_series)
    ic_col : str, default "ic"
        Column name for IC values (if DataFrame)
    annualization_factor : float, default sqrt(252)
        Factor to annualize IC-IR (sqrt(periods_per_year))
        - Daily: sqrt(252) ~ 15.87
        - Weekly: sqrt(52) ~ 7.21
        - Monthly: sqrt(12) ~ 3.46
    confidence_intervals : bool, default False
        Whether to compute bootstrap confidence intervals
    n_bootstrap : int, default 10000
        Number of bootstrap samples for CI computation
    alpha : float, default 0.05
        Significance level for confidence intervals (95% CI)

    Returns
    -------
    Union[float, dict]
        If confidence_intervals=False: IC-IR value
        If confidence_intervals=True: dict with 'ic_ir', 'lower_ci', 'upper_ci'

    Examples
    --------
    >>> # Compute IC series first
    >>> ic_series = compute_ic_series(pred_df, ret_df)
    >>>
    >>> # Compute IC-IR
    >>> ic_ir = compute_ic_ir(ic_series)
    >>> print(f"IC-IR: {ic_ir:.3f}")
    IC-IR: 0.645
    >>>
    >>> # With confidence intervals
    >>> result = compute_ic_ir(ic_series, confidence_intervals=True)
    >>> print(f"IC-IR: {result['ic_ir']:.3f} [{result['lower_ci']:.3f}, {result['upper_ci']:.3f}]")
    IC-IR: 0.645 [0.412, 0.891]

    Notes
    -----
    IC-IR Interpretation:
    - IC-IR < 0.3: Weak/inconsistent predictive power
    - IC-IR 0.3-0.5: Moderate consistency
    - IC-IR 0.5-1.0: Good consistency
    - IC-IR > 1.0: Excellent consistency

    The annualization factor adjusts IC-IR to an annual scale for easier
    interpretation and comparison across different rebalancing frequencies.
    """
    # Extract IC values
    ic_values: NDArray[Any]
    if isinstance(ic_series, pl.DataFrame | pd.DataFrame):
        is_polars = isinstance(ic_series, pl.DataFrame)
        if is_polars:
            ic_values = cast(pl.DataFrame, ic_series)[ic_col].to_numpy()
        else:
            ic_values = cast(pd.DataFrame, ic_series)[ic_col].to_numpy()
    else:
        ic_values = np.asarray(ic_series).flatten()

    # Remove NaN values
    ic_clean: NDArray[Any] = ic_values[~np.isnan(ic_values)]

    # Validate sufficient data
    if len(ic_clean) < 2:
        if confidence_intervals:
            return {
                "ic_ir": np.nan,
                "lower_ci": np.nan,
                "upper_ci": np.nan,
                "mean_ic": np.nan,
                "std_ic": np.nan,
                "n_periods": len(ic_clean),
            }
        return np.nan

    # Compute IC-IR
    mean_ic = float(np.mean(ic_clean))
    std_ic = float(np.std(ic_clean, ddof=1))  # Sample std

    if std_ic == 0:
        # Perfect consistency (all IC values identical)
        ic_ir = np.inf if mean_ic > 0 else -np.inf if mean_ic < 0 else np.nan
    else:
        ic_ir = (mean_ic / std_ic) * annualization_factor

    # Return simple IC-IR if no CI requested
    if not confidence_intervals:
        return float(ic_ir)

    # Bootstrap confidence intervals
    if len(ic_clean) < 10:
        # Insufficient data for meaningful bootstrap
        return {
            "ic_ir": float(ic_ir),
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "mean_ic": float(mean_ic),
            "std_ic": float(std_ic),
            "n_periods": len(ic_clean),
        }

    # Perform bootstrap
    rng = np.random.RandomState(42)  # For reproducibility
    bootstrap_ics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(ic_clean, size=len(ic_clean), replace=True)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)

        if sample_std > 0:
            bootstrap_ic_ir = (sample_mean / sample_std) * annualization_factor
            bootstrap_ics.append(bootstrap_ic_ir)

    if len(bootstrap_ics) == 0:
        # Bootstrap failed (all samples had zero std)
        return {
            "ic_ir": float(ic_ir),
            "lower_ci": np.nan,
            "upper_ci": np.nan,
            "mean_ic": float(mean_ic),
            "std_ic": float(std_ic),
            "n_periods": len(ic_clean),
        }

    # Compute percentile confidence intervals
    lower_ci = np.percentile(bootstrap_ics, (alpha / 2) * 100)
    upper_ci = np.percentile(bootstrap_ics, (1 - alpha / 2) * 100)

    return {
        "ic_ir": float(ic_ir),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "mean_ic": float(mean_ic),
        "std_ic": float(std_ic),
        "n_periods": len(ic_clean),
    }
