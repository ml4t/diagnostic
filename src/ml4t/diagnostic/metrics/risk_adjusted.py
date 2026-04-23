"""Risk-adjusted performance metrics: Sharpe, Sortino, Maximum Drawdown.

This module provides standard risk-adjusted return metrics used in portfolio
and strategy evaluation.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _clean_returns(returns: Union[pl.Series, pd.Series, "NDArray[Any]"]) -> "NDArray[Any]":
    """Convert returns to a flat numpy array and drop NaNs."""
    ret_array = DataFrameAdapter.to_numpy(returns).flatten()
    return ret_array[~np.isnan(ret_array)]


def _periodic_risk_free_rate(annual_risk_free_rate: float, periods_per_year: int) -> float:
    """Convert an annual risk-free rate to a per-period rate."""
    return float((1 + annual_risk_free_rate) ** (1 / periods_per_year) - 1)


def sharpe_ratio(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    confidence_intervals: bool = False,
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    random_state: int | None = None,
) -> float | dict[str, float]:
    """Calculate annualized Sharpe ratio.

    The Sharpe Ratio measures risk-adjusted returns by dividing excess returns
    by return volatility. Higher values indicate better risk-adjusted performance.

    Parameters
    ----------
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Time series of periodic returns
    risk_free_rate : float, default 0.0
        Annual risk-free rate
    periods_per_year : int, default 252
        Number of return periods per year
    confidence_intervals : bool, default False
        Whether to compute bootstrap confidence intervals
    alpha : float, default 0.05
        Significance level for confidence intervals
    bootstrap_samples : int, default 1000
        Number of bootstrap samples for confidence intervals
    random_state : Optional[int], default None
        Random seed for reproducible bootstrap samples

    Returns
    -------
    Union[float, dict]
        If confidence_intervals=False: Sharpe ratio value
        If confidence_intervals=True: dict with 'sharpe', 'lower_ci', 'upper_ci'

    Examples
    --------
    >>> returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
    >>> sharpe = sharpe_ratio(returns, periods_per_year=252)
    >>> print(f"Sharpe Ratio: {sharpe:.3f}")

    >>> # With confidence intervals
    >>> result = sharpe_ratio(returns, confidence_intervals=True, random_state=42)
    >>> print(f"Sharpe: {result['sharpe']:.3f}")
    """
    if confidence_intervals:
        return sharpe_ratio_with_ci(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            alpha=alpha,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )
    return periodic_sharpe_ratio(
        returns,
        periodic_risk_free_rate=_periodic_risk_free_rate(risk_free_rate, periods_per_year),
        annualization_factor=periods_per_year,
    )


def periodic_sharpe_ratio(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    periodic_risk_free_rate: float = 0.0,
    annualization_factor: float | None = None,
) -> float:
    """Calculate Sharpe ratio from same-frequency returns and risk-free rate.

    Parameters
    ----------
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Time series of periodic returns
    periodic_risk_free_rate : float, default 0.0
        Risk-free rate (same frequency as returns)
    annualization_factor : Optional[float], default None
        Factor to annualize the ratio

    Returns
    -------
    float
        Sharpe ratio value
    """
    ret_clean = _clean_returns(returns)

    if len(ret_clean) < 2:
        return np.nan

    excess_returns = ret_clean - periodic_risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        if mean_excess > 0:
            return np.inf
        if mean_excess < 0:
            return -np.inf
        return np.nan

    sharpe = mean_excess / std_excess

    if annualization_factor is not None and not np.isinf(sharpe) and not np.isnan(sharpe):
        sharpe *= np.sqrt(annualization_factor)

    return float(sharpe) if not np.isinf(sharpe) else sharpe


def sharpe_ratio_with_ci(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    alpha: float = 0.05,
    bootstrap_samples: int = 1000,
    random_state: int | None = None,
) -> dict[str, float]:
    """Calculate annualized Sharpe ratio with bootstrap confidence intervals.

    Parameters
    ----------
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Time series of periodic returns
    risk_free_rate : float, default 0.0
        Annual risk-free rate
    periods_per_year : int, default 252
        Number of return periods per year
    alpha : float, default 0.05
        Significance level for confidence intervals
    bootstrap_samples : int, default 1000
        Number of bootstrap samples for confidence intervals
    random_state : Optional[int], default None
        Random seed for reproducible bootstrap samples

    Returns
    -------
    dict[str, float]
        Dict with 'sharpe', 'lower_ci', 'upper_ci' keys
    """
    periodic_rf = _periodic_risk_free_rate(risk_free_rate, periods_per_year)
    sharpe = periodic_sharpe_ratio(
        returns,
        periodic_risk_free_rate=periodic_rf,
        annualization_factor=periods_per_year,
    )

    if np.isnan(sharpe) or np.isinf(sharpe):
        return {"sharpe": sharpe, "lower_ci": np.nan, "upper_ci": np.nan}

    ret_clean = _clean_returns(returns)

    if len(ret_clean) < 10:
        return {"sharpe": sharpe, "lower_ci": np.nan, "upper_ci": np.nan}

    if random_state is not None:
        np.random.seed(random_state)

    bootstrap_sharpes = []
    for _ in range(bootstrap_samples):
        bootstrap_sample = np.random.choice(ret_clean, size=len(ret_clean), replace=True)
        bootstrap_excess = bootstrap_sample - periodic_rf
        bootstrap_mean = np.mean(bootstrap_excess)
        bootstrap_std = np.std(bootstrap_excess, ddof=1)

        if bootstrap_std > 0:
            bs_sharpe = bootstrap_mean / bootstrap_std
            bs_sharpe *= np.sqrt(periods_per_year)
            bootstrap_sharpes.append(bs_sharpe)

    if len(bootstrap_sharpes) == 0:
        return {"sharpe": sharpe, "lower_ci": np.nan, "upper_ci": np.nan}

    lower_ci = np.percentile(bootstrap_sharpes, (alpha / 2) * 100)
    upper_ci = np.percentile(bootstrap_sharpes, (1 - alpha / 2) * 100)

    return {"sharpe": sharpe, "lower_ci": float(lower_ci), "upper_ci": float(upper_ci)}


def maximum_drawdown(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    cumulative: bool = False,
) -> dict[str, float]:
    """Calculate Maximum Drawdown and related statistics.

    Maximum Drawdown measures the largest peak-to-trough decline in cumulative
    returns. It represents the worst-case loss an investor would experience.

    Parameters
    ----------
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Time series of returns (or cumulative returns if cumulative=True)
    cumulative : bool, default False
        Whether input is already cumulative returns

    Returns
    -------
    dict
        Dictionary with 'max_drawdown', 'max_drawdown_duration', 'peak_date', 'trough_date'

    Examples
    --------
    >>> returns = np.array([0.10, -0.05, 0.08, -0.12, 0.03])
    >>> dd = maximum_drawdown(returns)
    >>> print(f"Max Drawdown: {dd['max_drawdown']:.3f}")
    Max Drawdown: -0.102
    """
    # Import here to avoid circular dependency
    from ml4t.diagnostic.core.numba_utils import calculate_drawdown_numba

    # Convert to numpy array
    ret_array = DataFrameAdapter.to_numpy(returns).flatten()

    # Remove NaN values
    ret_clean = ret_array[~np.isnan(ret_array)]

    if len(ret_clean) == 0:
        return {
            "max_drawdown": np.nan,
            "max_drawdown_duration": np.nan,
            "peak_date": np.nan,
            "trough_date": np.nan,
        }

    # Calculate cumulative returns if needed
    if cumulative:
        cum_returns = ret_clean
    else:
        cum_returns = np.cumprod(1 + ret_clean) - 1  # Compound returns

    # Use Numba-optimized function
    max_drawdown_val, dd_duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)

    # Handle case where no drawdown was found
    if peak_idx == -1:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "peak_date": 0,
            "trough_date": 0,
        }

    return {
        "max_drawdown": float(max_drawdown_val),
        "max_drawdown_duration": int(dd_duration),
        "peak_date": int(peak_idx),
        "trough_date": int(trough_idx),
    }


def sortino_ratio(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """Calculate annualized Sortino ratio.

    The Sortino Ratio is similar to Sharpe ratio but only penalizes downside
    volatility, making it more appropriate for asymmetric return distributions.

    Parameters
    ----------
    returns : Union[pl.Series, pd.Series, np.ndarray]
        Time series of periodic returns
    risk_free_rate : float, default 0.0
        Annual risk-free rate
    periods_per_year : int, default 252
        Number of return periods per year
    target_return : float, default 0.0
        Periodic target threshold after subtracting the risk-free rate

    Returns
    -------
    float
        Sortino ratio value

    Examples
    --------
    >>> returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    >>> sortino = sortino_ratio(returns, periods_per_year=252)
    >>> print(f"Sortino Ratio: {sortino:.3f}")
    Sortino Ratio: 0.894
    """
    return periodic_sortino_ratio(
        returns,
        periodic_risk_free_rate=_periodic_risk_free_rate(risk_free_rate, periods_per_year),
        annualization_factor=periods_per_year,
        target_return=target_return,
    )


def periodic_sortino_ratio(
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    periodic_risk_free_rate: float = 0.0,
    annualization_factor: float | None = None,
    target_return: float = 0.0,
) -> float:
    """Calculate Sortino ratio from same-frequency returns and risk-free rate."""
    ret_clean = _clean_returns(returns)

    if len(ret_clean) < 2:
        return np.nan

    excess_returns = ret_clean - periodic_risk_free_rate - target_return

    # Calculate downside returns (only negative excess returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # No downside - infinite Sortino ratio if mean is positive
        mean_excess = np.mean(excess_returns)
        if mean_excess > 0:
            return np.inf
        if mean_excess < 0:
            return -np.inf
        return np.nan

    # Calculate Sortino ratio
    mean_excess = np.mean(excess_returns)
    downside_std = np.sqrt(np.mean(downside_returns**2))  # Downside deviation

    if downside_std == 0:
        return np.nan

    sortino = mean_excess / downside_std

    # Apply annualization if specified
    if annualization_factor is not None:
        sortino *= np.sqrt(annualization_factor)

    return float(sortino)
