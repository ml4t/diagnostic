"""Model validation diagnostics: QLIKE, MALV, and residual tests.

Implements model quality metrics that are more informative than R² for
factor model evaluation.

References
----------
- Patton (2011): Volatility forecast comparison using QLIKE
- Ljung & Box (1978): Residual autocorrelation test
- Jarque & Bera (1980): Residual normality test
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .results import FactorModelResult, ModelValidationResult


def validate_factor_model(
    model_result: FactorModelResult,
    factor_returns: np.ndarray,
    *,
    max_acf_lags: int = 10,
    qlike_window: int = 21,
) -> ModelValidationResult:
    """Run model quality diagnostics on a fitted factor model.

    Parameters
    ----------
    model_result : FactorModelResult
        Fitted model result from compute_factor_model().
    factor_returns : np.ndarray
        Factor returns matrix (T, K) used in estimation.
    max_acf_lags : int
        Number of lags for Ljung-Box test.
    qlike_window : int
        Rolling window for QLIKE and MALV computation.

    Returns
    -------
    ModelValidationResult
        Comprehensive model diagnostics.
    """
    residuals = model_result.residuals
    T = len(residuals)

    # Fitted values
    betas = np.array([model_result.betas[f] for f in model_result.factor_names])
    fitted = model_result.alpha + factor_returns[:T] @ betas

    # QLIKE: quasi-likelihood loss for variance prediction (time-varying)
    qlike = _compute_qlike(residuals, fitted, window=qlike_window)

    # MALV: mean absolute log variance ratio (vectorized)
    malv = _compute_malv(residuals, fitted, window=qlike_window)

    # Ljung-Box test for residual autocorrelation
    lb_stat, lb_p = _ljung_box(residuals, max_acf_lags)

    # Jarque-Bera test for residual normality
    jb_stat, jb_p = sp_stats.jarque_bera(residuals)

    # Condition number of design matrix
    X = np.column_stack([np.ones(T), factor_returns[:T]])
    cond = float(np.linalg.cond(X))

    return ModelValidationResult(
        qlike=qlike,
        malv=malv,
        r_squared=model_result.r_squared,
        t_stat_pct=model_result.t_stat_pct_above_2,
        durbin_watson=model_result.durbin_watson,
        ljung_box_p=float(lb_p),
        jarque_bera_p=float(jb_p),
        condition_number=cond,
    )


def _compute_qlike(residuals: np.ndarray, fitted: np.ndarray, window: int = 21) -> float:
    """QLIKE loss for variance prediction quality (Patton 2011).

    Computes time-varying QLIKE by comparing rolling realized variance
    (from residuals + fitted = actual returns) against rolling predicted
    variance (from fitted values alone):

        QLIKE_t = sigma²_actual_t / sigma²_predicted_t
                  - log(sigma²_actual_t / sigma²_predicted_t) - 1

    Returns the mean QLIKE across all rolling windows.
    Lower is better; minimum of 0 when prediction is perfect.
    """
    actual = residuals + fitted
    n = len(actual)
    if n < window + 1:
        return float("nan")

    # Vectorized rolling variance using stride tricks
    actual_rolling_var = _rolling_variance(actual, window)
    fitted_rolling_var = _rolling_variance(fitted, window)

    # Mask out zero/negative variances
    valid = (actual_rolling_var > 1e-20) & (fitted_rolling_var > 1e-20)
    if not np.any(valid):
        return float("nan")

    ratio = actual_rolling_var[valid] / fitted_rolling_var[valid]
    qlike_vals = ratio - np.log(ratio) - 1.0
    return float(np.mean(qlike_vals))


def _compute_malv(residuals: np.ndarray, fitted: np.ndarray, window: int = 21) -> float:
    """Mean Absolute Log Variance ratio.

    Compares local variance of residuals vs fitted values in rolling windows.
    Lower values indicate better model fit (residual variance << fitted variance).
    """
    n = len(residuals)
    if n < window + 1:
        return float("nan")

    resid_var = _rolling_variance(residuals, window)
    fitted_var = _rolling_variance(fitted, window)

    valid = (resid_var > 1e-20) & (fitted_var > 1e-20)
    if not np.any(valid):
        return float("nan")

    ratio = resid_var[valid] / fitted_var[valid]
    return float(np.mean(np.abs(np.log(ratio))))


def _rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling variance using cumulative sums."""
    n = len(x)
    # Use Welford-like approach via cumulative sums
    cumsum = np.cumsum(x)
    cumsum2 = np.cumsum(x**2)

    # For window [i-window+1, i], sum = cumsum[i] - cumsum[i-window]
    sum_x = cumsum[window - 1 :].copy()
    sum_x[1:] -= cumsum[: n - window]
    sum_x2 = cumsum2[window - 1 :].copy()
    sum_x2[1:] -= cumsum2[: n - window]

    mean_x = sum_x / window
    var_x = sum_x2 / window - mean_x**2
    # Clamp small negatives from floating point
    return np.maximum(var_x, 0.0)


def _ljung_box(residuals: np.ndarray, max_lags: int) -> tuple[float, float]:
    """Ljung-Box test for residual autocorrelation.

    Uses the standard Q-statistic:
        Q = T(T+2) * sum_{k=1}^{m} rho_k^2 / (T-k)
    """
    T = len(residuals)
    max_lags = min(max_lags, T // 4)
    if max_lags < 1:
        return 0.0, 1.0

    # Compute ACF via numpy correlation
    mean_resid = np.mean(residuals)
    centered = residuals - mean_resid
    c0 = np.sum(centered**2)

    if c0 == 0:
        return 0.0, 1.0

    acf = np.zeros(max_lags)
    for k in range(1, max_lags + 1):
        acf[k - 1] = np.sum(centered[: T - k] * centered[k:]) / c0

    # Q statistic
    denominators = np.arange(T - 1, T - max_lags - 1, -1, dtype=np.float64)
    q = T * (T + 2) * np.sum(acf**2 / denominators)
    p_value = 1.0 - sp_stats.chi2.cdf(q, df=max_lags)

    return float(q), float(p_value)
