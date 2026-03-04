"""Return attribution with lagged betas and confidence intervals.

Implements:
1. Standard return attribution using lagged rolling betas (no look-ahead)
2. Maximal attribution for resolving correlated-factor ambiguity (Paleologo Ch 14)

References
----------
- Paleologo (2025), Ch 14: Attribution CIs and maximal attribution
- Brinson, Hood & Beebower (1986): Performance attribution framework
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from .data import FactorData
from .results import AttributionResult, FactorModelResult, MaximalAttributionResult
from .rolling_model import compute_rolling_exposures
from .static_model import _align_and_prepare


def compute_return_attribution(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    window: int = 63,
    lag: int = 1,
    confidence_level: float = 0.95,
) -> AttributionResult:
    """Compute return attribution using lagged rolling betas.

    Attribution formula (no look-ahead):
        factor_contrib_k[t] = beta_k[t-lag] * factor_return_k[t]
        alpha_contrib[t] = alpha[t-lag]
        residual[t] = return[t] - sum(factor_contribs) - alpha

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    window : int
        Rolling window for beta estimation.
    lag : int
        Number of periods to lag betas (default 1, no look-ahead).
    confidence_level : float
        Confidence level for attribution CIs.

    Returns
    -------
    AttributionResult
        Per-period and cumulative attribution with CIs.
    """
    y, X, timestamps = _align_and_prepare(returns, factor_data)
    T, K = X.shape
    factor_names = factor_data.factor_names

    # Get rolling betas — pass already-excess returns with rf_rate stripped
    # to avoid double-subtracting the risk-free rate (attribution.py already
    # called _align_and_prepare which subtracts rf_rate).
    factor_data_no_rf = FactorData(
        returns=factor_data.returns,
        rf_rate=None,
        factor_names=factor_data.factor_names,
        source=factor_data.source,
        frequency=factor_data.frequency,
    )
    rolling_result = compute_rolling_exposures(y, factor_data_no_rf, window=window)

    # Determine the attribution period (after window + lag)
    n_rolling = len(rolling_result.timestamps)
    start_idx = window + lag - 1
    if start_idx >= T:
        raise ValueError(f"window ({window}) + lag ({lag}) exceeds data length ({T})")

    n_attr = min(n_rolling - lag, T - start_idx)
    if n_attr < 1:
        raise ValueError("Not enough data for attribution after window + lag")

    attr_timestamps = timestamps[start_idx : start_idx + n_attr]
    factor_contributions: dict[str, np.ndarray] = {}
    alpha_contribution = np.zeros(n_attr)
    total_factor_contrib = np.zeros(n_attr)

    for f in factor_names:
        lagged_betas = rolling_result.rolling_betas[f][:n_attr]
        factor_rets = X[start_idx : start_idx + n_attr, factor_names.index(f)]
        contrib = lagged_betas * factor_rets
        factor_contributions[f] = contrib
        total_factor_contrib += contrib

    lagged_alpha = rolling_result.rolling_alpha[:n_attr]
    alpha_contribution = lagged_alpha.copy()

    actual_returns = y[start_idx : start_idx + n_attr]
    residual_arr = actual_returns - total_factor_contrib - alpha_contribution

    # Cumulative returns via compounding
    cumulative_factor = {}
    for f in factor_names:
        cumulative_factor[f] = np.cumprod(1 + factor_contributions[f]) - 1

    cumulative_alpha = np.cumprod(1 + alpha_contribution) - 1
    cumulative_residual = np.cumprod(1 + residual_arr) - 1
    cumulative_total = np.cumprod(1 + actual_returns) - 1

    # Summary percentages
    total_return = cumulative_total[-1] if len(cumulative_total) > 0 else 0.0
    summary_pct: dict[str, float] = {}
    if abs(total_return) > 1e-12:
        for f in factor_names:
            summary_pct[f] = cumulative_factor[f][-1] / total_return
        summary_pct["alpha"] = cumulative_alpha[-1] / total_return
        summary_pct["residual"] = cumulative_residual[-1] / total_return
    else:
        for f in factor_names:
            summary_pct[f] = 0.0
        summary_pct["alpha"] = 0.0
        summary_pct["residual"] = 0.0

    # Attribution CIs (Paleologo Ch 14)
    attribution_se, attribution_ci, idio_se = _compute_attribution_ci(
        X[start_idx : start_idx + n_attr],
        factor_contributions,
        residual_arr,
        factor_names,
        confidence_level,
    )

    return AttributionResult(
        timestamps=attr_timestamps,
        factor_contributions=factor_contributions,
        alpha_contribution=alpha_contribution,
        residual=residual_arr,
        cumulative_factor=cumulative_factor,
        cumulative_alpha=cumulative_alpha,
        cumulative_residual=cumulative_residual,
        cumulative_total=cumulative_total,
        summary_pct=summary_pct,
        attribution_se=attribution_se,
        attribution_ci=attribution_ci,
        idiosyncratic_se=idio_se,
        factor_names=factor_names,
        window=window,
        lag=lag,
        confidence_level=confidence_level,
    )


def _compute_attribution_ci(
    X: np.ndarray,
    factor_contribs: dict[str, np.ndarray],
    residuals: np.ndarray,
    factor_names: list[str],
    confidence_level: float,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], float]:
    """Compute attribution standard errors and CIs.

    For cumulative attribution A_k = sum_t(beta_k[t] * F_k[t]), the SE
    accounts for estimation uncertainty in betas flowing through to attribution.

    Using the delta method on the static model:
        Var(beta_k) = sigma²_eps * (X'X)^{-1}_{kk}
        SE(A_k) = sqrt(sum_t(F_k[t]²)) * SE(beta_k)

    This gives the SE of the cumulative factor contribution under the
    assumption that beta estimation error is the dominant source of
    uncertainty (Paleologo Ch 14, Insight 14.1).

    Returns (attribution_se, attribution_ci, idiosyncratic_se).
    """
    T, K = X.shape
    # Residual variance with proper degrees of freedom (K factors + intercept)
    dof = T - K - 1
    sigma2 = float(np.sum(residuals**2) / max(dof, 1))

    # (X'X)^{-1} for beta covariance: Var(beta) = sigma² * (X'X)^{-1}
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    z = sp_stats.norm.ppf((1 + confidence_level) / 2)
    attribution_se: dict[str, float] = {}
    attribution_ci: dict[str, tuple[float, float]] = {}

    for k, f in enumerate(factor_names):
        # SE of beta_k
        beta_se_k = np.sqrt(sigma2 * XtX_inv[k, k])

        # Cumulative factor return (the "exposure" to beta uncertainty)
        # A_k = beta_k * sum(F_k[t]) under static beta assumption
        # SE(A_k) = |sum(F_k[t])| * SE(beta_k) via delta method
        # For rolling betas, use sqrt(sum(F_k[t]²)) * SE(beta_k) which
        # accounts for the time-varying nature of the factor returns
        factor_k = X[:, k]
        factor_norm = float(np.sqrt(np.sum(factor_k**2)))

        se = float(beta_se_k * factor_norm)
        se = max(se, 1e-12)

        cum_attrib = float(np.sum(factor_contribs[f]))
        attribution_se[f] = se
        attribution_ci[f] = (cum_attrib - z * se, cum_attrib + z * se)

    # Idiosyncratic SE: SE of the cumulative residual = sqrt(T) * sigma_eps
    # This is the standard error of the sum of T iid residuals
    idio_se = float(np.sqrt(sigma2 * T)) if T > 0 else 0.0

    return attribution_se, attribution_ci, idio_se


def compute_maximal_attribution(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    factors_of_interest: list[str],
    *,
    model_result: FactorModelResult | None = None,
) -> MaximalAttributionResult:
    """Maximal attribution for correlated factors (Paleologo Ch 14).

    Resolves ambiguity when factors are correlated (e.g., HML and CMA
    at rho~0.7) by computing the maximum PnL attributable to a subset
    S of factors of interest.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns.
    factor_data : FactorData
        Factor return data.
    factors_of_interest : list[str]
        Subset S of factors to maximize attribution for.
    model_result : FactorModelResult | None
        Pre-computed model. If None, computed internally.

    Returns
    -------
    MaximalAttributionResult
        Adjusted betas and maximal PnL.
    """
    from .static_model import compute_factor_model

    if model_result is None:
        model_result = compute_factor_model(returns, factor_data)

    factor_names = factor_data.factor_names

    # Validate factors_of_interest
    invalid = set(factors_of_interest) - set(factor_names)
    if invalid:
        raise ValueError(f"Unknown factors: {invalid}")
    other_factors = [f for f in factor_names if f not in factors_of_interest]
    if not other_factors:
        raise ValueError("Need at least one factor outside the interest set")

    # Factor covariance matrix
    X = factor_data.get_factor_array()
    Omega = np.cov(X, rowvar=False)

    # Indices
    s_idx = [factor_names.index(f) for f in factors_of_interest]
    u_idx = [factor_names.index(f) for f in other_factors]

    # Submatrices
    Omega_SS = Omega[np.ix_(s_idx, s_idx)]
    Omega_US = Omega[np.ix_(u_idx, s_idx)]

    # Rotation matrix: A = Omega_US @ inv(Omega_SS)
    try:
        Omega_SS_inv = np.linalg.inv(Omega_SS)
    except np.linalg.LinAlgError:
        Omega_SS_inv = np.linalg.pinv(Omega_SS)

    A = Omega_US @ Omega_SS_inv

    # Betas
    beta_S = np.array([model_result.betas[f] for f in factors_of_interest])
    beta_U = np.array([model_result.betas[f] for f in other_factors])

    # Adjusted betas: beta_adj = beta_S + A' @ beta_U
    beta_adj = beta_S + A.T @ beta_U

    # Maximal PnL: adjusted_beta * mean_factor_return * T
    mean_factor_returns = np.mean(X, axis=0)
    T = len(X)

    adjusted_betas = {f: float(beta_adj[i]) for i, f in enumerate(factors_of_interest)}
    maximal_pnl = {
        f: float(beta_adj[i] * mean_factor_returns[s_idx[i]] * T)
        for i, f in enumerate(factors_of_interest)
    }

    # Orthogonal residual
    total_pnl = float(
        np.sum(np.array([model_result.betas[f] for f in factor_names]) * mean_factor_returns * T)
    )
    max_pnl_total = sum(maximal_pnl.values())
    orthogonal_residual = total_pnl - max_pnl_total

    return MaximalAttributionResult(
        adjusted_betas=adjusted_betas,
        maximal_pnl=maximal_pnl,
        orthogonal_residual=orthogonal_residual,
        rotation_matrix=A,
        factors_of_interest=factors_of_interest,
        all_factor_names=factor_names,
    )
