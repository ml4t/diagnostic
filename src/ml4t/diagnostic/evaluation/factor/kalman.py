"""Kalman filter for time-varying factor betas (Tier 2).

State-space model:
    beta_t = beta_{t-1} + eta_t    (state transition)
    r_t = X_t' beta_t + eps_t      (observation)

Custom scipy implementation using the numerically stable Joseph form
for the covariance update. The inner loop is JIT-compiled with Numba.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .data import FactorData
from .results import RollingExposureResult, StabilityDiagnostics
from .static_model import _align_and_prepare

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """Fallback: no-op decorator when Numba is not installed."""

        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


def compute_kalman_betas(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    observation_noise: float | None = None,
    state_noise: float | None = None,
    optimize_noise: bool = True,
) -> RollingExposureResult:
    """Estimate time-varying betas via Kalman filter.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    observation_noise : float | None
        Observation noise variance (sigma^2_eps). Estimated if None.
    state_noise : float | None
        State transition noise variance (sigma^2_eta). Estimated if None.
    optimize_noise : bool
        Optimize noise parameters via MLE.

    Returns
    -------
    RollingExposureResult
        Time-varying betas with confidence bands from filter covariance.
    """
    y, X, timestamps = _align_and_prepare(returns, factor_data)
    T, K = X.shape
    factor_names = factor_data.factor_names

    # Add intercept
    X_aug = np.column_stack([np.ones(T), X])
    K_aug = K + 1

    # Initialize noise params
    if observation_noise is None or state_noise is None:
        if optimize_noise:
            obs_noise, st_noise = _optimize_noise(y, X_aug, K_aug)
        else:
            obs_noise = float(np.var(y)) * 0.5
            st_noise = obs_noise * 0.01
        observation_noise = observation_noise or obs_noise
        state_noise = state_noise or st_noise

    # Run Kalman filter
    beta_filtered, P_filtered = _kalman_filter(y, X_aug, K_aug, observation_noise, state_noise)

    # Extract results (skip intercept column for betas)
    rolling_betas = {}
    for k, f in enumerate(factor_names):
        rolling_betas[f] = beta_filtered[:, k + 1]

    rolling_alpha = beta_filtered[:, 0]

    # Rolling R² using expanding window (minimum 20 periods)
    rolling_r2 = _compute_rolling_r2(y, X_aug, beta_filtered, min_window=20)

    # Stability
    stability = _compute_kalman_stability(rolling_betas, rolling_r2, factor_names)

    return RollingExposureResult(
        timestamps=timestamps,
        rolling_betas=rolling_betas,
        rolling_alpha=rolling_alpha,
        rolling_r_squared=rolling_r2,
        stability=stability,
        window=0,  # 0 indicates Kalman (no fixed window)
        factor_names=factor_names,
    )


@njit(cache=True)
def _kalman_filter_inner(
    y: np.ndarray,
    X: np.ndarray,
    K: int,
    obs_noise: float,
    state_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-JIT Kalman filter inner loop using Joseph form.

    Parameters
    ----------
    y : (T,) observations
    X : (T, K) design matrix (with intercept)
    K : number of state variables (K_aug = n_factors + 1)
    obs_noise : observation noise variance
    state_noise : state transition noise variance

    Returns
    -------
    beta_filtered : (T, K) filtered state estimates
    P_diag : (T, K) diagonal of filtered covariance (for memory efficiency)
    """
    T = len(y)

    beta = np.zeros(K)
    P = np.eye(K) * 1.0
    Q = np.eye(K) * state_noise
    I_K = np.eye(K)

    beta_filtered = np.zeros((T, K))
    P_diag = np.zeros((T, K))

    for t in range(T):
        # Predict (random walk)
        beta_pred = beta.copy()
        P_pred = P + Q

        # Update
        x_t = X[t]
        y_pred = 0.0
        for j in range(K):
            y_pred += x_t[j] * beta_pred[j]
        innovation = y[t] - y_pred

        # S = x' P_pred x + R (scalar)
        S = obs_noise
        for j in range(K):
            for m in range(K):
                S += x_t[j] * P_pred[j, m] * x_t[m]

        if S <= 1e-20:
            beta_filtered[t] = beta_pred
            for j in range(K):
                P_diag[t, j] = P_pred[j, j]
            beta = beta_pred
            P = P_pred
            continue

        # Kalman gain K_gain = P_pred @ x / S
        K_gain = np.zeros(K)
        for j in range(K):
            for m in range(K):
                K_gain[j] += P_pred[j, m] * x_t[m]
            K_gain[j] /= S

        # State update
        beta = beta_pred + K_gain * innovation

        # Joseph form: P = (I - K*x') @ P_pred @ (I - K*x')' + R * K*K'
        IKH = np.zeros((K, K))
        for j in range(K):
            for m in range(K):
                IKH[j, m] = I_K[j, m] - K_gain[j] * x_t[m]

        # P = IKH @ P_pred @ IKH' + R * outer(K_gain, K_gain)
        tmp = np.zeros((K, K))
        for j in range(K):
            for m in range(K):
                for n in range(K):
                    tmp[j, m] += IKH[j, n] * P_pred[n, m]

        P_new = np.zeros((K, K))
        for j in range(K):
            for m in range(K):
                for n in range(K):
                    P_new[j, m] += tmp[j, n] * IKH[m, n]
                P_new[j, m] += obs_noise * K_gain[j] * K_gain[m]

        # Symmetry enforcement
        for j in range(K):
            for m in range(j + 1, K):
                avg = (P_new[j, m] + P_new[m, j]) * 0.5
                P_new[j, m] = avg
                P_new[m, j] = avg

        P = P_new
        beta_filtered[t] = beta
        for j in range(K):
            P_diag[t, j] = P[j, j]

    return beta_filtered, P_diag


def _kalman_filter(
    y: np.ndarray,
    X: np.ndarray,
    K: int,
    obs_noise: float,
    state_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Kalman filter predict/update loop using Joseph form.

    Dispatches to Numba-JIT version if available, otherwise
    falls back to pure NumPy.

    Returns
    -------
    beta_filtered : (T, K) array of filtered state estimates
    P_filtered : (T, K, K) array of filtered state covariances
    """
    if _HAS_NUMBA:
        beta_filtered, P_diag = _kalman_filter_inner(
            y.astype(np.float64),
            X.astype(np.float64),
            K,
            float(obs_noise),
            float(state_noise),
        )
        # Reconstruct minimal P_filtered from diagonal for compatibility
        T = len(y)
        P_filtered = np.zeros((T, K, K))
        for t in range(T):
            np.fill_diagonal(P_filtered[t], P_diag[t])
        return beta_filtered, P_filtered

    return _kalman_filter_numpy(y, X, K, obs_noise, state_noise)


def _kalman_filter_numpy(
    y: np.ndarray,
    X: np.ndarray,
    K: int,
    obs_noise: float,
    state_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy fallback for Kalman filter."""
    T = len(y)
    beta = np.zeros(K)
    P = np.eye(K) * 1.0
    R = obs_noise
    Q = np.eye(K) * state_noise
    I_K = np.eye(K)

    beta_filtered = np.zeros((T, K))
    P_filtered = np.zeros((T, K, K))

    for t in range(T):
        beta_pred = beta
        P_pred = P + Q

        x_t = X[t]
        innovation = y[t] - x_t @ beta_pred
        S = x_t @ P_pred @ x_t + R

        if S <= 1e-20:
            beta_filtered[t] = beta_pred
            P_filtered[t] = P_pred
            beta = beta_pred
            P = P_pred
            continue

        K_gain = P_pred @ x_t / S
        beta = beta_pred + K_gain * innovation

        IKH = I_K - np.outer(K_gain, x_t)
        P = IKH @ P_pred @ IKH.T + R * np.outer(K_gain, K_gain)
        P = (P + P.T) * 0.5

        beta_filtered[t] = beta
        P_filtered[t] = P

    return beta_filtered, P_filtered


def _compute_rolling_r2(
    y: np.ndarray,
    X: np.ndarray,
    beta_filtered: np.ndarray,
    min_window: int = 20,
) -> np.ndarray:
    """Compute rolling R² from Kalman-filtered betas.

    Uses vectorized expanding window: fitted[t] = X[t] @ beta[t],
    then cumulative sums for SS_res and SS_tot.
    """
    T = len(y)
    rolling_r2 = np.full(T, np.nan)

    # Vectorized: fitted values for all t
    fitted = np.sum(X * beta_filtered, axis=1)  # (T,)
    residuals = y - fitted

    # Cumulative sums for SS computation
    cum_res2 = np.cumsum(residuals**2)
    cum_y = np.cumsum(y)
    cum_y2 = np.cumsum(y**2)

    for t in range(min_window - 1, T):
        n = t + 1
        ss_res = cum_res2[t]
        ss_tot = cum_y2[t] - cum_y[t] ** 2 / n
        rolling_r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return rolling_r2


def _optimize_noise(y: np.ndarray, X: np.ndarray, K: int) -> tuple[float, float]:
    """Optimize noise parameters via negative log-likelihood."""
    var_y = float(np.var(y))

    if _HAS_NUMBA:
        _neg_loglik_jit = _make_neg_loglik_jit()

        def neg_loglik(params: np.ndarray) -> float:
            return _neg_loglik_jit(params, y, X, K)
    else:

        def neg_loglik(params: np.ndarray) -> float:
            return _neg_loglik_numpy(params, y, X, K)

    x0 = np.array([np.log(var_y * 0.5), np.log(var_y * 0.005)])
    result = minimize(neg_loglik, x0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4})

    if not result.success:
        warnings.warn(
            f"Kalman noise optimization did not converge: {result.message}. "
            "Using best estimate found.",
            stacklevel=3,
        )

    return float(np.exp(result.x[0])), float(np.exp(result.x[1]))


def _neg_loglik_numpy(params: np.ndarray, y: np.ndarray, X: np.ndarray, K: int) -> float:
    """Pure numpy negative log-likelihood for Kalman filter."""
    obs_noise = np.exp(params[0])
    state_noise = np.exp(params[1])

    T = len(y)
    beta = np.zeros(K)
    P = np.eye(K) * 1.0
    Q = np.eye(K) * state_noise
    I_K = np.eye(K)

    ll = 0.0
    for t in range(T):
        beta_pred = beta
        P_pred = P + Q

        x_t = X[t]
        y_pred = x_t @ beta_pred
        S = x_t @ P_pred @ x_t + obs_noise

        if S <= 1e-20:
            return 1e10

        innovation = y[t] - y_pred
        ll += -0.5 * (np.log(S) + innovation**2 / S)

        K_gain = P_pred @ x_t / S
        beta = beta_pred + K_gain * innovation
        IKH = I_K - np.outer(K_gain, x_t)
        P = IKH @ P_pred @ IKH.T + obs_noise * np.outer(K_gain, K_gain)
        P = (P + P.T) * 0.5

    return -ll


def _make_neg_loglik_jit():  # type: ignore[no-untyped-def]
    """Create JIT-compiled negative log-likelihood function."""

    @njit(cache=True)
    def _neg_loglik_inner(params: np.ndarray, y: np.ndarray, X: np.ndarray, K: int) -> float:
        obs_noise = np.exp(params[0])
        state_noise = np.exp(params[1])

        T = len(y)
        beta = np.zeros(K)
        P = np.eye(K) * 1.0
        Q = np.eye(K) * state_noise
        I_K = np.eye(K)

        ll = 0.0
        for t in range(T):
            beta_pred = beta.copy()
            P_pred = P + Q

            x_t = X[t]
            y_pred = 0.0
            for j in range(K):
                y_pred += x_t[j] * beta_pred[j]

            S = obs_noise
            for j in range(K):
                for m in range(K):
                    S += x_t[j] * P_pred[j, m] * x_t[m]

            if S <= 1e-20:
                return 1e10

            innovation = y[t] - y_pred
            ll += -0.5 * (np.log(S) + innovation * innovation / S)

            # Kalman gain
            K_gain = np.zeros(K)
            for j in range(K):
                for m in range(K):
                    K_gain[j] += P_pred[j, m] * x_t[m]
                K_gain[j] /= S

            beta = beta_pred + K_gain * innovation

            # Joseph form
            IKH = np.zeros((K, K))
            for j in range(K):
                for m in range(K):
                    IKH[j, m] = I_K[j, m] - K_gain[j] * x_t[m]

            tmp = np.zeros((K, K))
            for j in range(K):
                for m in range(K):
                    for n in range(K):
                        tmp[j, m] += IKH[j, n] * P_pred[n, m]

            P_new = np.zeros((K, K))
            for j in range(K):
                for m in range(K):
                    for n in range(K):
                        P_new[j, m] += tmp[j, n] * IKH[m, n]
                    P_new[j, m] += obs_noise * K_gain[j] * K_gain[m]

            for j in range(K):
                for m in range(j + 1, K):
                    avg = (P_new[j, m] + P_new[m, j]) * 0.5
                    P_new[j, m] = avg
                    P_new[m, j] = avg

            P = P_new

        return -ll

    return _neg_loglik_inner


def _compute_kalman_stability(
    rolling_betas: dict[str, np.ndarray],
    rolling_r2: np.ndarray,
    factor_names: list[str],
) -> StabilityDiagnostics:
    """Compute stability diagnostics from Kalman betas."""
    beta_std: dict[str, float] = {}
    sign_consistency: dict[str, float] = {}
    max_abs_change: dict[str, float] = {}

    for f in factor_names:
        betas = rolling_betas[f]
        valid = betas[np.isfinite(betas)]
        if len(valid) < 2:
            beta_std[f] = float("nan")
            sign_consistency[f] = float("nan")
            max_abs_change[f] = float("nan")
            continue

        beta_std[f] = float(np.std(valid))
        full_sign = np.sign(np.mean(valid))
        sign_consistency[f] = float(np.mean(np.sign(valid) == full_sign)) if full_sign != 0 else 0.5
        diffs = np.abs(np.diff(valid))
        max_abs_change[f] = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    valid_r2 = rolling_r2[np.isfinite(rolling_r2)]
    r2_mean = float(np.mean(valid_r2)) if len(valid_r2) > 0 else float("nan")
    r2_std = float(np.std(valid_r2)) if len(valid_r2) > 0 else float("nan")

    return StabilityDiagnostics(
        beta_std=beta_std,
        sign_consistency=sign_consistency,
        max_abs_change=max_abs_change,
        vif=None,
        r_squared_mean=r2_mean,
        r_squared_std=r2_std,
    )
