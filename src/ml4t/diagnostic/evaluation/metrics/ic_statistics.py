"""IC statistical analysis: HAC-adjusted significance and decay analysis.

This module provides advanced statistical analysis for IC time series,
including autocorrelation-robust significance tests and decay analysis.
"""

from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_by_horizon

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_ic_hac_stats(
    ic_series: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    ic_col: str = "ic",
    maxlags: int | None = None,
    kernel: str = "bartlett",
    use_correction: bool = True,
) -> dict[str, float]:
    """Compute HAC-adjusted significance statistics for IC time series.

    Uses Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent)
    standard errors to account for autocorrelation in IC time series. This
    provides robust t-statistics and p-values when IC exhibits serial correlation.

    The Newey-West estimator accounts for:
    1. Heteroskedasticity: Non-constant variance in IC over time
    2. Autocorrelation: Serial correlation in IC values
    3. Lag selection: Automatic selection of optimal lag window

    Parameters
    ----------
    ic_series : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Time series of IC values (from compute_ic_series)
    ic_col : str, default "ic"
        Column name for IC values (if DataFrame)
    maxlags : int | None, default None
        Maximum lag for HAC adjustment. If None, uses Newey-West formula:
        maxlags = floor(4 * (T/100)^(2/9))
        where T is the sample size
    kernel : str, default "bartlett"
        Kernel function for lag weighting:
        - "bartlett": Triangular kernel (Newey-West default)
        - "uniform": Equal weights
        - "parzen": Parzen kernel
    use_correction : bool, default True
        Apply small-sample correction to standard errors

    Returns
    -------
    dict[str, float]
        Dictionary with HAC-adjusted statistics:
        - mean_ic: Mean IC across time series
        - hac_se: HAC-adjusted standard error
        - t_stat: t-statistic (mean_ic / hac_se)
        - p_value: Two-tailed p-value for H0: IC = 0
        - n_periods: Number of observations
        - effective_lags: Number of lags used in HAC adjustment
        - naive_se: Standard OLS standard error (for comparison)
        - naive_t_stat: Naive t-statistic without HAC adjustment

    Examples
    --------
    >>> # Compute IC series first
    >>> ic_series = compute_ic_series(pred_df, ret_df)
    >>>
    >>> # Compute HAC-adjusted statistics
    >>> stats = compute_ic_hac_stats(ic_series)
    >>> print(f"Mean IC: {stats['mean_ic']:.4f}")
    >>> print(f"HAC t-stat: {stats['t_stat']:.2f}")
    >>> print(f"P-value: {stats['p_value']:.4f}")
    >>> print(f"Significant: {stats['p_value'] < 0.05}")
    Mean IC: 0.0234
    HAC t-stat: 2.14
    P-value: 0.0327
    Significant: True
    >>>
    >>> # Compare with naive statistics
    >>> print(f"Naive t-stat: {stats['naive_t_stat']:.2f}")
    >>> print(f"HAC adjustment factor: {stats['naive_se'] / stats['hac_se']:.2f}x")
    Naive t-stat: 3.45
    HAC adjustment factor: 1.61x

    Notes
    -----
    HAC Adjustment Interpretation:
    - HAC SE > Naive SE: Positive autocorrelation detected
    - HAC SE < Naive SE: Negative autocorrelation (rare)
    - HAC SE ~ Naive SE: Little autocorrelation

    The Newey-West automatic lag selection formula is:
        maxlags = floor(4 * (T/100)^(2/9))

    For example:
    - T=100 -> maxlags=4
    - T=252 -> maxlags=5
    - T=500 -> maxlags=6

    References
    ----------
    .. [1] Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
           Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
           Econometrica, 55(3), 703-708.
    .. [2] Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation
           Consistent Covariance Matrix Estimation." Econometrica, 59(3), 817-858.
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
    n = len(ic_clean)
    if n < 3:
        return {
            "mean_ic": np.nan,
            "hac_se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_periods": n,
            "effective_lags": 0,
            "naive_se": np.nan,
            "naive_t_stat": np.nan,
        }

    # Compute mean IC
    mean_ic = float(np.mean(ic_clean))

    # Compute naive (OLS) standard error
    naive_var = float(np.var(ic_clean, ddof=1))  # Sample variance
    naive_se = np.sqrt(naive_var / n)  # Standard error of mean
    naive_t_stat = mean_ic / naive_se if naive_se > 0 else np.nan

    # Determine optimal lags if not specified
    if maxlags is None:
        # Newey-West automatic lag selection formula
        # maxlags = floor(4 * (T/100)^(2/9))
        maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        maxlags = max(1, maxlags)  # At least 1 lag
        maxlags = min(maxlags, n // 2)  # No more than T/2

    # Fit OLS model: IC ~ constant (testing if mean IC != 0)
    # This is equivalent to a one-sample t-test
    exog = np.ones((n, 1))  # Just constant term
    y = ic_clean.reshape(-1, 1)

    # Compute HAC covariance matrix
    try:
        # Fit OLS model
        model = OLS(y, exog)
        ols_results = model.fit()

        # Get HAC-robust covariance matrix
        hac_cov = cov_hac(
            ols_results,
            nlags=maxlags,
            weights_func=_get_kernel_weights(kernel),
            use_correction=use_correction,
        )

        # Extract HAC variance (it's a 1x1 matrix for the constant)
        hac_var = hac_cov[0, 0]
        hac_se = np.sqrt(hac_var)

    except Exception as e:
        # If HAC computation fails, fall back to naive SE
        print(f"Warning: HAC computation failed ({e}), using naive SE")
        hac_se = naive_se

    # Compute HAC-adjusted t-statistic
    t_stat = mean_ic / hac_se if hac_se > 0 else np.nan

    # Compute two-tailed p-value
    # Use t-distribution with n-1 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    return {
        "mean_ic": float(mean_ic),
        "hac_se": float(hac_se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_periods": n,
        "effective_lags": maxlags,
        "naive_se": float(naive_se),
        "naive_t_stat": float(naive_t_stat),
    }


def _get_kernel_weights(kernel: str):
    """Get kernel weight function for HAC estimation.

    Parameters
    ----------
    kernel : str
        Kernel name: "bartlett", "uniform", or "parzen"

    Returns
    -------
    callable
        Weight function that takes nlags and returns array of weights
    """
    if kernel == "bartlett":
        # Bartlett kernel: weights decline linearly (Newey-West default)
        def bartlett_weights(nlags):
            return np.array([1 - h / (nlags + 1) for h in range(nlags + 1)])

        return bartlett_weights

    if kernel == "uniform":
        # Uniform kernel: equal weights
        def uniform_weights(nlags):
            return np.ones(nlags + 1)

        return uniform_weights

    if kernel == "parzen":
        # Parzen kernel: smoother decay
        def parzen_weights(nlags):
            weights = np.zeros(nlags + 1)
            for h in range(nlags + 1):
                z = h / (nlags + 1)
                if z <= 0.5:
                    weights[h] = 1 - 6 * z**2 + 6 * z**3
                else:
                    weights[h] = 2 * (1 - z) ** 3
            return weights

        return parzen_weights

    raise ValueError(f"Unknown kernel: {kernel}. Use 'bartlett', 'uniform', or 'parzen'.")


def compute_ic_decay(
    predictions: pl.DataFrame | pd.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    horizons: list[int] | None = None,
    pred_col: str = "prediction",
    price_col: str = "close",
    date_col: str = "date",
    group_col: str | None = None,
    method: str = "spearman",
    estimate_half_life: bool = True,
) -> dict[str, Any]:
    """Analyze how IC decays over prediction horizons.

    Computes IC at multiple forward-looking horizons to understand how long
    predictions retain predictive power. Faster IC decay indicates shorter
    signal persistence.

    This is critical for:
    1. Determining optimal holding periods
    2. Understanding alpha decay dynamics
    3. Identifying when to retrain models
    4. Avoiding stale predictions

    Parameters
    ----------
    predictions : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with predictions, must have pred_col, date_col, and optionally group_col
    prices : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with prices, must have price_col, date_col, and optionally group_col
    horizons : list[int] | None, default None
        List of forward horizons in days. If None, uses [1, 2, 5, 10, 21]
    pred_col : str, default "prediction"
        Column name for predictions
    price_col : str, default "close"
        Column name for prices
    date_col : str, default "date"
        Column name for dates
    group_col : str | None, default None
        Column name for grouping (e.g., "symbol" for multi-asset)
    method : str, default "spearman"
        Correlation method: "spearman" or "pearson"
    estimate_half_life : bool, default True
        Whether to estimate IC half-life (horizon where IC drops to 50% of initial)

    Returns
    -------
    dict[str, Any]
        Dictionary with decay analysis:
        - ic_by_horizon: dict mapping horizon -> IC value
        - horizons: list of horizons analyzed
        - decay_rate: exponential decay rate (if estimable)
        - half_life: estimated half-life in days (if estimate_half_life=True)
        - optimal_horizon: horizon with highest IC
        - n_observations: number of observations per horizon

    Examples
    --------
    >>> # Analyze IC decay for multi-asset predictions
    >>> decay = compute_ic_decay(
    ...     predictions=pred_df,
    ...     prices=price_df,
    ...     horizons=[1, 2, 5, 10, 21],
    ...     group_col="symbol"
    ... )
    >>> print(f"IC at 1-day: {decay['ic_by_horizon'][1]:.3f}")
    >>> print(f"IC at 21-day: {decay['ic_by_horizon'][21]:.3f}")
    >>> print(f"Half-life: {decay['half_life']:.1f} days")
    >>> print(f"Optimal horizon: {decay['optimal_horizon']} days")
    IC at 1-day: 0.045
    IC at 21-day: 0.012
    Half-life: 8.3 days
    Optimal horizon: 1 days

    Notes
    -----
    IC Decay Patterns:
    - Fast decay: IC drops >50% within 5 days -> high-frequency signal
    - Moderate decay: IC half-life 5-20 days -> medium-term signal
    - Slow decay: IC half-life >20 days -> long-term signal
    - No decay: IC stable -> structural/fundamental signal

    Half-life is estimated by fitting exponential decay:
        IC(h) = IC(0) * exp(-lambda * h)
        half_life = ln(2) / lambda

    Optimal horizon is the horizon with maximum IC, useful for determining
    best rebalancing frequency.

    References
    ----------
    .. [1] Kakushadze, Z. (2016). "101 Formulaic Alphas." Wilmott, 2016(84), 72-81.
    """
    # Set default horizons if not provided
    if horizons is None:
        horizons = [1, 2, 5, 10, 21]

    # Ensure horizons are sorted
    horizons = sorted(horizons)

    # Compute IC for each horizon using compute_ic_by_horizon
    ic_results = compute_ic_by_horizon(
        predictions=predictions,
        prices=prices,
        horizons=horizons,
        pred_col=pred_col,
        price_col=price_col,
        date_col=date_col,
        group_col=group_col,
        method=method,
    )

    # Extract IC values and observation counts
    ic_by_horizon: dict[int, float] = {}
    n_obs_by_horizon: dict[int, int] = {}

    for horizon, ic_value in ic_results.items():
        ic_by_horizon[horizon] = ic_value
        # Note: compute_ic_by_horizon returns just IC values, not counts
        # We'll approximate n_obs from the input data
        n_obs_by_horizon[horizon] = len(predictions)

    # Find optimal horizon (highest absolute IC)
    optimal_ic: float
    optimal_horizon: int | None
    if ic_by_horizon:
        optimal_horizon = max(ic_by_horizon.keys(), key=lambda h: abs(ic_by_horizon[h]))
        optimal_ic = ic_by_horizon[optimal_horizon]
    else:
        optimal_horizon = None
        optimal_ic = np.nan

    # Estimate decay rate and half-life
    decay_rate = np.nan
    half_life = np.nan

    if estimate_half_life and len(ic_by_horizon) >= 2:
        # Extract horizons and IC values for fitting
        h_vals = np.array(list(ic_by_horizon.keys()))
        ic_vals = np.array([ic_by_horizon[h] for h in h_vals])

        # Remove NaN values
        valid_mask = ~np.isnan(ic_vals)
        h_vals = h_vals[valid_mask]
        ic_vals = ic_vals[valid_mask]

        if len(h_vals) >= 2 and np.all(ic_vals > 0):
            # Fit exponential decay: IC(h) = IC(0) * exp(-lambda * h)
            # Take log: ln(IC(h)) = ln(IC(0)) - lambda * h
            # This is linear regression: y = a + b*x where b = -lambda

            try:
                log_ic = np.log(ic_vals)

                # Linear regression
                coeffs = np.polyfit(h_vals, log_ic, deg=1)
                decay_rate = -coeffs[0]  # -lambda from the linear fit

                # Half-life: t_{1/2} = ln(2) / lambda
                if decay_rate > 0:
                    half_life = np.log(2) / decay_rate
                elif decay_rate < 0:
                    # Negative decay rate means IC is increasing (unusual)
                    half_life = np.inf
                else:
                    half_life = np.nan

            except (ValueError, np.linalg.LinAlgError):
                # Fitting failed (e.g., all IC values identical)
                decay_rate = np.nan
                half_life = np.nan

        elif len(h_vals) >= 2:
            # Can't fit exponential if IC values are not all positive
            # Try fitting to absolute values
            try:
                abs_ic_vals = np.abs(ic_vals)
                if np.all(abs_ic_vals > 0):
                    log_abs_ic = np.log(abs_ic_vals)
                    coeffs = np.polyfit(h_vals, log_abs_ic, deg=1)
                    decay_rate = -coeffs[0]

                    half_life = np.log(2) / decay_rate if decay_rate > 0 else np.nan
            except (ValueError, np.linalg.LinAlgError):
                pass

    return {
        "ic_by_horizon": ic_by_horizon,
        "horizons": horizons,
        "decay_rate": float(decay_rate) if not np.isnan(decay_rate) else None,
        "half_life": float(half_life)
        if not np.isnan(half_life) and not np.isinf(half_life)
        else None,
        "optimal_horizon": optimal_horizon,
        "optimal_ic": optimal_ic if not np.isnan(optimal_ic) else None,
        "n_observations": n_obs_by_horizon,
    }
