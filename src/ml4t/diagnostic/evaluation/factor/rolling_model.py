"""Rolling factor model estimation for time-varying exposures.

Implements vectorized rolling OLS to estimate how factor exposures
evolve over time, with stability diagnostics.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .data import FactorData
from .results import RollingExposureResult, StabilityDiagnostics
from .static_model import _align_and_prepare


def compute_rolling_exposures(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    window: int = 63,
    expanding: bool = False,
    min_periods: int | None = None,
    compute_vif: bool = False,
) -> RollingExposureResult:
    """Estimate rolling factor exposures via windowed OLS.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    window : int
        Rolling window size in periods (default 63 ~ 1 quarter).
    expanding : bool
        Use expanding window instead of rolling.
    min_periods : int | None
        Minimum observations per window. Defaults to window // 2.
    compute_vif : bool
        Compute VIF per window (slower but useful for multicollinearity).

    Returns
    -------
    RollingExposureResult
        Rolling betas, alpha, R², and stability diagnostics.
    """
    y, X, timestamps = _align_and_prepare(returns, factor_data)
    T, K = X.shape
    factor_names = factor_data.factor_names

    if min_periods is None:
        min_periods = max(K + 2, window // 2)

    n_windows = T - window + 1 if not expanding else T - min_periods + 1
    if n_windows < 1:
        raise ValueError(
            f"Not enough data ({T} obs) for window={window}. Need at least {window} observations."
        )

    if expanding:
        out_betas, out_alpha, out_r2 = _rolling_ols_expanding(y, X, min_periods, n_windows)
        out_timestamps = np.array(
            [timestamps[min_periods + i - 1] for i in range(n_windows)]
        )
    else:
        out_betas, out_alpha, out_r2 = _rolling_ols_vectorized(y, X, window)
        out_timestamps = np.array([timestamps[i + window - 1] for i in range(n_windows)])

    # Compute stability diagnostics
    rolling_betas_dict = {}
    for k, f in enumerate(factor_names):
        rolling_betas_dict[f] = out_betas[:, k]

    stability = _compute_stability(rolling_betas_dict, out_r2, factor_names, X, compute_vif)

    return RollingExposureResult(
        timestamps=out_timestamps,
        rolling_betas=rolling_betas_dict,
        rolling_alpha=out_alpha,
        rolling_r_squared=out_r2,
        stability=stability,
        window=window,
        factor_names=factor_names,
    )


def _rolling_ols_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized rolling OLS using cumulative sum trick.

    Computes X'X and X'y via running sums, avoiding per-window dot products.
    O(T*K²) total instead of O(T*window*K²) for the naive loop.

    Parameters
    ----------
    y : (T,) array
    X : (T, K) array of factor returns (no intercept)
    window : int

    Returns
    -------
    betas : (n_windows, K) — factor betas per window
    alpha : (n_windows,) — intercept per window
    r2 : (n_windows,) — R² per window
    """
    T, K = X.shape
    n_windows = T - window + 1
    K_aug = K + 1  # with intercept

    # Augment X with intercept column
    ones = np.ones((T, 1))
    X_aug = np.column_stack([ones, X])  # (T, K+1)

    # Compute cumulative sums for X'X and X'y
    # XtX[i,j] for window [s, s+w) = sum_{t=s}^{s+w-1} X_aug[t,i]*X_aug[t,j]
    # Use cumsum of outer products
    XtX_cum = np.zeros((T + 1, K_aug, K_aug))
    Xty_cum = np.zeros((T + 1, K_aug))
    y_cum = np.zeros(T + 1)
    y2_cum = np.zeros(T + 1)

    for t in range(T):
        x_t = X_aug[t]
        XtX_cum[t + 1] = XtX_cum[t] + np.outer(x_t, x_t)
        Xty_cum[t + 1] = Xty_cum[t] + x_t * y[t]
        y_cum[t + 1] = y_cum[t] + y[t]
        y2_cum[t + 1] = y2_cum[t] + y[t] ** 2

    # Extract per-window XtX and Xty via subtraction
    out_betas = np.full((n_windows, K), np.nan)
    out_alpha = np.full(n_windows, np.nan)
    out_r2 = np.full(n_windows, np.nan)

    for i in range(n_windows):
        s = i
        e = i + window

        XtX_win = XtX_cum[e] - XtX_cum[s]
        Xty_win = Xty_cum[e] - Xty_cum[s]

        try:
            params = np.linalg.solve(XtX_win, Xty_win)
        except np.linalg.LinAlgError:
            continue

        out_alpha[i] = params[0]
        out_betas[i] = params[1:]

        # R²: 1 - SS_res / SS_tot
        # SS_tot = sum(y²) - n*mean(y)² = sum(y²) - (sum(y))²/n
        sum_y = y_cum[e] - y_cum[s]
        sum_y2 = y2_cum[e] - y2_cum[s]
        ss_tot = sum_y2 - sum_y**2 / window

        # SS_res = y'y - 2*params'*X'y + params'*X'X*params
        ss_res = sum_y2 - 2.0 * params @ Xty_win + params @ XtX_win @ params
        out_r2[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return out_betas, out_alpha, out_r2


def _rolling_ols_expanding(
    y: np.ndarray,
    X: np.ndarray,
    min_periods: int,
    n_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expanding window OLS using cumulative sums."""
    T, K = X.shape
    K_aug = K + 1

    ones = np.ones((T, 1))
    X_aug = np.column_stack([ones, X])

    # Build cumulative sums incrementally
    XtX = np.zeros((K_aug, K_aug))
    Xty = np.zeros(K_aug)
    sum_y = 0.0
    sum_y2 = 0.0

    out_betas = np.full((n_windows, K), np.nan)
    out_alpha = np.full(n_windows, np.nan)
    out_r2 = np.full(n_windows, np.nan)

    for t in range(T):
        x_t = X_aug[t]
        XtX += np.outer(x_t, x_t)
        Xty += x_t * y[t]
        sum_y += y[t]
        sum_y2 += y[t] ** 2

        n = t + 1
        if n < min_periods:
            continue

        idx = n - min_periods
        if idx >= n_windows:
            break

        try:
            params = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        out_alpha[idx] = params[0]
        out_betas[idx] = params[1:]

        ss_tot = sum_y2 - sum_y**2 / n
        ss_res = sum_y2 - 2.0 * params @ Xty + params @ XtX @ params
        out_r2[idx] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return out_betas, out_alpha, out_r2


def _compute_stability(
    rolling_betas: dict[str, np.ndarray],
    rolling_r2: np.ndarray,
    factor_names: list[str],
    X: np.ndarray,
    compute_vif: bool,
) -> StabilityDiagnostics:
    """Compute stability diagnostics from rolling estimates."""
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

        # Sign consistency: fraction same sign as full-sample mean
        full_sign = np.sign(np.mean(valid))
        if full_sign == 0:
            sign_consistency[f] = 0.5
        else:
            sign_consistency[f] = float(np.mean(np.sign(valid) == full_sign))

        # Max single-step change
        diffs = np.abs(np.diff(valid))
        max_abs_change[f] = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    # VIF from full-sample factor correlation
    vif = None
    if compute_vif and X.shape[1] > 1:
        vif = _compute_vif(X, factor_names)

    valid_r2 = rolling_r2[np.isfinite(rolling_r2)]
    r2_mean = float(np.mean(valid_r2)) if len(valid_r2) > 0 else float("nan")
    r2_std = float(np.std(valid_r2)) if len(valid_r2) > 0 else float("nan")

    return StabilityDiagnostics(
        beta_std=beta_std,
        sign_consistency=sign_consistency,
        max_abs_change=max_abs_change,
        vif=vif,
        r_squared_mean=r2_mean,
        r_squared_std=r2_std,
    )


def _compute_vif(X: np.ndarray, factor_names: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factors: VIF_k = 1/(1 - R²_k)."""
    K = X.shape[1]
    vif = {}
    for k in range(K):
        y_k = X[:, k]
        X_k = np.delete(X, k, axis=1)
        X_k = np.column_stack([np.ones(len(y_k)), X_k])

        try:
            params = np.linalg.lstsq(X_k, y_k, rcond=None)[0]
            fitted = X_k @ params
            ss_res = np.sum((y_k - fitted) ** 2)
            ss_tot = np.sum((y_k - np.mean(y_k)) ** 2)
            r2_k = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif[factor_names[k]] = 1.0 / (1.0 - r2_k) if r2_k < 1.0 else float("inf")
        except np.linalg.LinAlgError:
            vif[factor_names[k]] = float("nan")

    return vif
