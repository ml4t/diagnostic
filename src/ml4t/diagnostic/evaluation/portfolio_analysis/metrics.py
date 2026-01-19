"""Core metric functions for portfolio analysis.

This module provides standalone utility functions for computing
portfolio performance metrics:
- Risk-adjusted returns (Sharpe, Sortino, Calmar, Omega, Tail)
- Return metrics (annual return, volatility, max drawdown)
- Risk metrics (VaR, CVaR)
- Benchmark-relative metrics (alpha, beta, information ratio, capture ratios)
- Portfolio turnover
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import polars as pl

# Type aliases - use Union for Python 3.9 compatibility
ArrayLike = Union[np.ndarray, "pl.Series", "list[float]"]


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert various types to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, "to_numpy"):  # Polars Series
        return np.asarray(cast(Any, data).to_numpy())
    elif hasattr(data, "values"):  # pandas Series
        return np.asarray(cast(Any, data).values)
    else:
        return np.asarray(data)


def _safe_prod(arr: np.ndarray) -> float:
    """Compute product, ignoring NaN values.

    Uses np.nanprod to handle NaN gracefully instead of propagating NaN
    through the entire result.
    """
    return float(np.nanprod(arr))


def _safe_cumprod(arr: np.ndarray) -> np.ndarray:
    """Compute cumulative product with NaN handling.

    If NaN values are present, they are forward-filled from the previous
    valid cumulative product value. This prevents NaN from corrupting
    the entire equity curve.
    """
    if not np.any(np.isnan(arr)):
        return np.cumprod(arr)

    # Handle NaN by treating as 1.0 (no change) in the product
    arr_clean = np.where(np.isnan(arr), 1.0, arr)
    return np.cumprod(arr_clean)


def _annualization_factor(periods_per_year: int = 252) -> float:
    """Get annualization factor."""
    return np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Daily returns (non-cumulative)
        risk_free: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    returns = _to_numpy(returns)

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

    excess_returns = returns - daily_rf

    if len(excess_returns) < 2:
        return np.nan

    mean_excess = np.nanmean(excess_returns)
    std_excess = np.nanstd(excess_returns, ddof=1)

    if std_excess == 0:
        return np.nan

    return (mean_excess / std_excess) * _annualization_factor(periods_per_year)


def sortino_ratio(
    returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio.

    Uses downside deviation (semi-deviation) instead of full volatility.

    Args:
        returns: Daily returns (non-cumulative)
        risk_free: Annual risk-free rate
        periods_per_year: Trading periods per year
        target: Target return threshold for downside calculation (daily, relative
            to risk-free rate). When target=0, downside is measured below the
            risk-free rate.

    Returns:
        Annualized Sortino ratio
    """
    returns = _to_numpy(returns)

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

    excess_returns = returns - daily_rf

    # Downside returns: excess returns below target
    # Uses excess returns for consistency with numerator
    downside_returns = np.minimum(excess_returns - target, 0)

    if len(downside_returns) < 2:
        return np.nan

    mean_excess = np.nanmean(excess_returns)
    downside_std = np.sqrt(np.nanmean(downside_returns**2))

    if downside_std == 0:
        return np.nan

    return (mean_excess / downside_std) * _annualization_factor(periods_per_year)


def calmar_ratio(
    returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """Compute Calmar ratio (annual return / max drawdown).

    Args:
        returns: Daily returns (non-cumulative)
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    returns = _to_numpy(returns)

    ann_return = annual_return(returns, periods_per_year)
    max_dd = max_drawdown(returns)

    if max_dd == 0:
        return np.nan

    return ann_return / abs(max_dd)


def omega_ratio(
    returns: ArrayLike,
    threshold: float = 0.0,
) -> float:
    """Compute Omega ratio.

    Omega = P(gain) * E[gain|gain] / (P(loss) * E[loss|loss])

    Args:
        returns: Daily returns (non-cumulative)
        threshold: Return threshold (default 0)

    Returns:
        Omega ratio
    """
    returns = _to_numpy(returns)

    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns <= threshold]

    sum_above = np.sum(returns_above)
    sum_below = np.sum(returns_below)

    if sum_below == 0:
        return np.inf if sum_above > 0 else np.nan

    return sum_above / sum_below


def tail_ratio(returns: ArrayLike) -> float:
    """Compute tail ratio (95th percentile / abs(5th percentile)).

    Measures asymmetry of return distribution tails.

    Args:
        returns: Daily returns

    Returns:
        Tail ratio (>1 means right tail heavier)
    """
    returns = _to_numpy(returns)

    p95 = np.nanpercentile(returns, 95)
    p5 = np.nanpercentile(returns, 5)

    if p5 == 0:
        return np.nan

    # Docstring: p95 / abs(p5) - use abs on denominator only, not the whole ratio
    return float(p95 / abs(p5))


def max_drawdown(returns: ArrayLike) -> float:
    """Compute maximum drawdown.

    Args:
        returns: Daily returns (non-cumulative)

    Returns:
        Maximum drawdown (negative value)
    """
    returns = _to_numpy(returns)

    # Compute cumulative returns
    cum_returns = _safe_cumprod(1 + returns)

    # Running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Drawdown
    drawdown = (cum_returns - running_max) / running_max

    return np.nanmin(drawdown)


def annual_return(
    returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return (CAGR).

    Args:
        returns: Daily returns (non-cumulative)
        periods_per_year: Trading periods per year

    Returns:
        Annualized return
    """
    returns = _to_numpy(returns)

    total = _safe_prod(1 + returns)
    n_periods = len(returns)

    if n_periods == 0:
        return np.nan

    years = n_periods / periods_per_year

    if years == 0:
        return np.nan

    return total ** (1 / years) - 1


def annual_volatility(
    returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized volatility.

    Args:
        returns: Daily returns (non-cumulative)
        periods_per_year: Trading periods per year

    Returns:
        Annualized volatility
    """
    returns = _to_numpy(returns)
    return float(np.nanstd(returns, ddof=1) * _annualization_factor(periods_per_year))


def value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
) -> float:
    """Compute Value at Risk.

    Args:
        returns: Daily returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        VaR (negative value representing potential loss)
    """
    returns = _to_numpy(returns)
    return float(np.nanpercentile(returns, (1 - confidence) * 100))


def conditional_var(
    returns: ArrayLike,
    confidence: float = 0.95,
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Daily returns
        confidence: Confidence level

    Returns:
        CVaR (expected loss given loss exceeds VaR)
    """
    returns = _to_numpy(returns)
    var = value_at_risk(returns, confidence)
    return float(np.nanmean(returns[returns <= var]))


def stability_of_timeseries(returns: ArrayLike) -> float:
    """Compute stability (R² of cumulative returns vs time).

    Higher stability indicates more consistent returns.

    Args:
        returns: Daily returns

    Returns:
        R² value (0 to 1)
    """
    returns = _to_numpy(returns)

    cum_returns = _safe_cumprod(1 + returns)

    # Fit linear regression
    x = np.arange(len(cum_returns))

    # Handle NaN
    mask = ~np.isnan(cum_returns)
    if mask.sum() < 2:
        return np.nan

    slope, intercept, r_value, _, _ = stats.linregress(x[mask], cum_returns[mask])

    return r_value**2


def alpha_beta(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Compute CAPM alpha and beta.

    Args:
        returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns
        risk_free: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        (alpha, beta) tuple - alpha is annualized
    """
    returns = _to_numpy(returns)
    benchmark = _to_numpy(benchmark_returns)

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

    # Excess returns
    excess_returns = returns - daily_rf
    excess_benchmark = benchmark - daily_rf

    # Align lengths
    min_len = min(len(excess_returns), len(excess_benchmark))
    excess_returns = excess_returns[:min_len]
    excess_benchmark = excess_benchmark[:min_len]

    # Remove NaN
    mask = ~(np.isnan(excess_returns) | np.isnan(excess_benchmark))
    if mask.sum() < 2:
        return np.nan, np.nan

    # Linear regression
    slope, intercept, _, _, _ = stats.linregress(excess_benchmark[mask], excess_returns[mask])

    beta = slope
    # Annualize alpha
    alpha = intercept * periods_per_year

    return alpha, beta


def information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """Compute Information Ratio (alpha / tracking error).

    Args:
        returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns
        periods_per_year: Trading periods per year

    Returns:
        Information ratio
    """
    returns = _to_numpy(returns)
    benchmark = _to_numpy(benchmark_returns)

    # Align lengths
    min_len = min(len(returns), len(benchmark))
    returns = returns[:min_len]
    benchmark = benchmark[:min_len]

    # Active return
    active_return = returns - benchmark

    # Tracking error (annualized)
    tracking_error = np.nanstd(active_return, ddof=1) * _annualization_factor(periods_per_year)

    if tracking_error == 0:
        return np.nan

    # Annualized active return
    ann_active = np.nanmean(active_return) * periods_per_year

    return ann_active / tracking_error


def up_down_capture(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
) -> tuple[float, float]:
    """Compute up and down capture ratios.

    Args:
        returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        (up_capture, down_capture) tuple
    """
    returns = _to_numpy(returns)
    benchmark = _to_numpy(benchmark_returns)

    # Align lengths
    min_len = min(len(returns), len(benchmark))
    returns = returns[:min_len]
    benchmark = benchmark[:min_len]

    # Up markets
    up_mask = benchmark > 0
    if up_mask.sum() > 0:
        up_capture = _safe_prod(1 + returns[up_mask]) / _safe_prod(1 + benchmark[up_mask])
    else:
        up_capture = np.nan

    # Down markets
    down_mask = benchmark < 0
    if down_mask.sum() > 0:
        down_capture = _safe_prod(1 + returns[down_mask]) / _safe_prod(1 + benchmark[down_mask])
    else:
        down_capture = np.nan

    return up_capture, down_capture


def compute_portfolio_turnover(
    weights: ArrayLike,
    dates: ArrayLike | None = None,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute portfolio turnover from a time series of weights.

    Turnover measures how much the portfolio is traded over time. It's defined
    as the average absolute change in weights across all positions.

    **Definition**:
        Turnover_t = (1/2) * Σ_i |w_{i,t} - w_{i,t-1}|

    The 1/2 factor accounts for the fact that selling one asset requires
    buying another (double-counting).

    **Interpretation**:
    - Turnover = 0%: Buy-and-hold (no rebalancing)
    - Turnover = 100%: Full portfolio replacement each period
    - Turnover > 200%: Aggressive trading (likely high transaction costs)

    Parameters
    ----------
    weights : array-like, shape (n_periods, n_assets)
        Portfolio weights over time. Each row should sum to 1 (or close to it).
    dates : array-like, optional
        Date index for the weights. If provided, used for reporting.
    annualize : bool, default=True
        Whether to annualize the turnover (multiply by periods_per_year).
    periods_per_year : int, default=252
        Number of trading periods per year.

    Returns
    -------
    dict[str, float]
        - 'turnover_mean': Mean turnover per period (or annualized)
        - 'turnover_median': Median turnover per period
        - 'turnover_std': Standard deviation of turnover
        - 'turnover_max': Maximum single-period turnover
        - 'turnover_total': Total turnover over the entire period
        - 'n_periods': Number of periods in the sample
        - 'is_annualized': Whether turnover_mean is annualized
    """
    weights = np.asarray(weights)

    if weights.ndim != 2:
        raise ValueError(
            f"weights must be 2D array (n_periods, n_assets), got shape {weights.shape}"
        )

    n_periods, n_assets = weights.shape

    if n_periods < 2:
        raise ValueError(f"Need at least 2 periods for turnover, got {n_periods}")

    # Compute period-by-period turnover
    # Turnover_t = (1/2) * sum(|w_t - w_{t-1}|)
    weight_changes = np.abs(np.diff(weights, axis=0))  # (n_periods-1, n_assets)
    period_turnover = 0.5 * weight_changes.sum(axis=1)  # (n_periods-1,)

    # Compute statistics
    mean_turnover = float(np.mean(period_turnover))
    median_turnover = float(np.median(period_turnover))
    std_turnover = float(np.std(period_turnover))
    max_turnover = float(np.max(period_turnover))
    total_turnover = float(np.sum(period_turnover))

    # Annualize if requested
    if annualize:
        mean_turnover_output = mean_turnover * periods_per_year
    else:
        mean_turnover_output = mean_turnover

    return {
        "turnover_mean": mean_turnover_output * 100,  # As percentage
        "turnover_median": median_turnover * 100,
        "turnover_std": std_turnover * 100,
        "turnover_max": max_turnover * 100,
        "turnover_total": total_turnover * 100,
        "n_periods": n_periods,
        "is_annualized": annualize,
        "periods_per_year": periods_per_year,
    }


__all__ = [
    # Internal helpers (exported for testing)
    "_to_numpy",
    "_safe_prod",
    "_safe_cumprod",
    "_annualization_factor",
    # Risk-adjusted return metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "tail_ratio",
    # Return metrics
    "max_drawdown",
    "annual_return",
    "annual_volatility",
    # Risk metrics
    "value_at_risk",
    "conditional_var",
    # Stability
    "stability_of_timeseries",
    # Benchmark-relative
    "alpha_beta",
    "information_ratio",
    "up_down_capture",
    # Portfolio turnover
    "compute_portfolio_turnover",
]
