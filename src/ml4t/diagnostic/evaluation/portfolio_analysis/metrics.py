"""Core metric functions for portfolio analysis.

This module provides standalone utility functions for computing
portfolio performance metrics:
- Return metrics (annual return, volatility, max drawdown)
- Risk metrics (VaR, CVaR)
- Benchmark-relative metrics (alpha, beta, information ratio, capture ratios)
- Portfolio turnover
"""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from scipy import stats

from ml4t.diagnostic.metrics.risk_adjusted import (
    _drawdown_from_wealth,
    _sample_standard_deviation,
    _sample_standard_deviation_of_difference,
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# Type aliases - use Union for Python 3.9 compatibility
ArrayLike = Union[np.ndarray, "pl.Series", "pd.Series", "list[float]"]


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


def calmar_ratio(
    returns: ArrayLike,
    periods_per_year: float = 252,
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
    *,
    no_loss: str = "infinity",
) -> float:
    """Compute Omega ratio.

    Omega = P(gain) * E[gain|gain] / (P(loss) * E[loss|loss])

    Args:
        returns: Daily returns (non-cumulative)
        threshold: Return threshold (default 0)
        no_loss: Return infinity or NaN when gains exist without losses.

    Returns:
        Omega ratio
    """
    if no_loss not in {"infinity", "nan"}:
        raise ValueError("no_loss must be 'infinity' or 'nan'")
    returns = _to_numpy(returns)

    returns_above = returns[returns > threshold] - threshold
    returns_below = threshold - returns[returns <= threshold]

    sum_above = np.sum(returns_above)
    sum_below = np.sum(returns_below)

    if sum_below == 0:
        return np.inf if sum_above > 0 and no_loss == "infinity" else np.nan

    return sum_above / sum_below


def tail_ratio(returns: ArrayLike, *, absolute: bool = False) -> float:
    """Compute tail ratio (95th percentile / abs(5th percentile)).

    Measures asymmetry of return distribution tails.

    Args:
        returns: Daily returns
        absolute: Use magnitudes of both tails rather than a signed numerator.

    Returns:
        Tail ratio (>1 means right tail heavier)
    """
    returns = _to_numpy(returns)

    p95 = np.nanpercentile(returns, 95)
    p5 = np.nanpercentile(returns, 5)

    if absolute:
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.abs(p95) / np.abs(p5))
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
    if len(cum_returns) == 0:
        return np.nan
    return float(np.nanmin(_drawdown_from_wealth(cum_returns)))


def annual_return(
    returns: ArrayLike,
    periods_per_year: float = 252,
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
    return _annualize_growth(total, n_periods, periods_per_year)


def _annualize_growth(total: float, n_periods: int, periods_per_year: float) -> float:
    """Annualize an observed product or one periodic fitted-intercept growth."""

    if n_periods == 0:
        return np.nan

    years = n_periods / periods_per_year

    if years == 0:
        return np.nan

    with np.errstate(invalid="ignore", over="ignore"):
        return float(np.power(total, 1 / years) - 1)


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
    return float(_sample_standard_deviation(returns) * _annualization_factor(periods_per_year))


def value_at_risk(
    returns: ArrayLike,
    confidence: float = 0.95,
    *,
    nan_policy: str = "omit",
) -> float:
    """Compute Value at Risk.

    Args:
        returns: Daily returns
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        nan_policy: Omit missing observations or propagate an undefined sample.

    Returns:
        VaR (negative value representing potential loss)
    """
    returns = _to_numpy(returns)
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'")
    quantile = np.nanpercentile if nan_policy == "omit" else np.percentile
    return float(quantile(returns, (1 - confidence) * 100))


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def gaussian_value_at_risk(returns: ArrayLike, sigma: float = 2.0) -> float:
    """Return mean minus sigma sample deviations in the input return period.

    NaNs are omitted; this Gaussian return estimate is neither an empirical
    quantile nor a cash loss. Aggregate supplied observations explicitly before
    calling when a different civil period is required.
    """
    width = _finite_real(sigma, "sigma")
    if width < 0:
        raise ValueError("sigma must be nonnegative")
    values = np.asarray(_to_numpy(returns), dtype=float)
    if values.ndim != 1:
        raise ValueError("Gaussian return VaR requires one series; apply per column")
    known = values[~np.isnan(values)]
    if len(known) < 2:
        return np.nan
    with np.errstate(invalid="ignore"):
        return float(np.mean(known) - width * np.std(known, ddof=1))


def normal_cash_value_at_risk(
    portfolio_value: float, confidence: float, mean: float = 0.0, std: float = 1.0
) -> float:
    """Scale a modeled lower normal return quantile into a signed cash loss.

    Mean and standard deviation are periodic decimal-return parameters. Positive
    output denotes loss; a positive lower return quantile gives a negative cash
    loss rather than a clipped zero. This is not an empirical return statistic.
    """
    capital = _finite_real(portfolio_value, "portfolio_value")
    confidence = _finite_real(confidence, "confidence")
    mean, std = _finite_real(mean, "mean"), _finite_real(std, "std")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between zero and one")
    if std <= 0:
        raise ValueError("std must be positive for a nondegenerate normal model")
    return float(-capital * stats.norm.ppf(1 - confidence, loc=mean, scale=std))


def conditional_var(
    returns: ArrayLike,
    confidence: float = 0.95,
    *,
    method: str = "threshold",
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Daily returns
        confidence: Confidence level
        method: Include every threshold tie or the lowest fixed count of observations.

    Returns:
        CVaR (expected loss given loss exceeds VaR)
    """
    returns = _to_numpy(returns)
    if method not in {"threshold", "fixed_count"}:
        raise ValueError("method must be 'threshold' or 'fixed_count'")
    if method == "fixed_count":
        return _fixed_count_tail(returns, 1 - confidence)
    var = value_at_risk(returns, confidence)
    return float(np.nanmean(returns[returns <= var]))


def _fixed_count_tail(returns: np.ndarray, cutoff: float) -> float:
    """Choose the count from the original cutoff, without a complement round trip."""
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between zero and one")
    known = returns[~np.isnan(returns)]
    if len(known) == 0:
        return np.nan
    index = int((len(known) - 1) * cutoff)
    return float(np.mean(np.partition(known, index)[: index + 1]))


def stability_of_timeseries(returns: ArrayLike, *, method: str = "wealth") -> float:
    """Compute stability (R² of cumulative returns vs time).

    Higher stability indicates more consistent returns.

    Args:
        returns: Daily returns
        method: Fit compounded wealth or cumulative log returns on observed rows.

    Returns:
        R² value (0 to 1)
    """
    returns = _to_numpy(returns)
    if method not in {"wealth", "log"}:
        raise ValueError("method must be 'wealth' or 'log'")
    if method == "log":
        known = returns[~np.isnan(returns)]
        if len(known) < 2:
            return np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_returns = np.log1p(known).cumsum()
        if not np.all(np.isfinite(cum_returns)):
            return np.nan
    else:
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
    if returns.ndim != 1 or benchmark.ndim != 1 or returns.shape != benchmark.shape:
        raise ValueError("benchmark and returns must be one-dimensional with equal length")

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free) ** (1 / periods_per_year) - 1

    # Excess returns
    excess_returns = returns - daily_rf
    excess_benchmark = benchmark - daily_rf

    intercept, beta = _paired_linear_regression(excess_returns, excess_benchmark)
    return intercept * periods_per_year, beta


def _paired_linear_regression(
    returns: np.ndarray,
    benchmark: np.ndarray,
    *,
    variance_floor: float = 0.0,
    center_dependent: bool = True,
) -> tuple[float, float]:
    """Shared one-dimensional paired OLS, with an explicit optional variance floor."""
    excess_returns, excess_benchmark = returns, benchmark
    mask = ~(np.isnan(excess_returns) | np.isnan(excess_benchmark))
    if mask.sum() < 2 or np.ptp(excess_benchmark[mask]) == 0:
        return np.nan, np.nan
    if variance_floor and np.var(excess_benchmark[mask]) < variance_floor:
        return np.nan, np.nan
    if not center_dependent:
        # The equivalent uncentered dependent moment preserves legacy infinity
        # propagation; the native SciPy fit and its defaults remain unchanged.
        independent = excess_benchmark[mask]
        dependent = excess_returns[mask]
        centered = independent - np.mean(independent)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = float(np.nanmean(centered * dependent) / np.nanmean(centered**2))
            intercept = _mean_regression_residual(dependent, independent, slope)
        return intercept, slope

    # Linear regression
    slope, intercept, _, _, _ = stats.linregress(excess_benchmark[mask], excess_returns[mask])

    return intercept, slope


def _mean_regression_residual(returns: np.ndarray, benchmark: np.ndarray, beta: float) -> float:
    """Mean paired residual for a supplied slope, retaining infinite domain outcomes."""
    residual = returns - beta * benchmark
    known = residual[~np.isnan(residual)]
    return float(np.mean(known)) if len(known) else np.nan


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

    if returns.ndim != 1 or benchmark.ndim != 1 or returns.shape != benchmark.shape:
        raise ValueError("benchmark and returns must be one-dimensional with equal length")

    # Active return
    active_return = returns - benchmark

    # Tracking error (annualized)
    tracking_error = _sample_standard_deviation_of_difference(
        returns, benchmark
    ) * _annualization_factor(periods_per_year)

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

    if returns.ndim != 1 or benchmark.ndim != 1 or returns.shape != benchmark.shape:
        raise ValueError("benchmark and returns must be one-dimensional with equal length")

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
