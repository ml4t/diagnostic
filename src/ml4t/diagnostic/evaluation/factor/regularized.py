"""Regularized factor models: Ridge, LASSO, ElasticNet (Tier 2).

Uses sklearn for estimation and bootstrap for standard errors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from .data import FactorData
from .results import FactorModelResult
from .static_model import _align_and_prepare, _durbin_watson


def compute_regularized_model(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    method: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    confidence_level: float = 0.95,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> FactorModelResult:
    """Fit regularized factor model with bootstrap standard errors.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    method : str
        "ridge", "lasso", or "elastic_net".
    alpha : float
        Regularization strength.
    l1_ratio : float
        ElasticNet mixing parameter (0=ridge, 1=lasso). Only for elastic_net.
    confidence_level : float
        CI confidence level.
    n_bootstrap : int
        Number of bootstrap resamples for SEs.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    FactorModelResult
        Model results with bootstrap SEs.
    """
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    y, X, timestamps = _align_and_prepare(returns, factor_data)
    T, K = X.shape
    factor_names = factor_data.factor_names

    # Fit model
    if method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True)
    elif method == "lasso":
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    elif method == "elastic_net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=10000)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, y)

    alpha_val = float(model.intercept_)
    betas = {f: float(model.coef_[i]) for i, f in enumerate(factor_names)}

    residuals = y - model.predict(X)

    # Bootstrap SEs
    rng = np.random.RandomState(random_state)
    boot_alphas = np.zeros(n_bootstrap)
    boot_betas = np.zeros((n_bootstrap, K))

    for b in range(n_bootstrap):
        idx = rng.choice(T, size=T, replace=True)
        y_boot = y[idx]
        X_boot = X[idx]

        if method == "ridge":
            m = Ridge(alpha=alpha, fit_intercept=True)
        elif method == "lasso":
            m = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        else:
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=10000)

        m.fit(X_boot, y_boot)
        boot_alphas[b] = m.intercept_
        boot_betas[b] = m.coef_

    alpha_se = float(np.std(boot_alphas, ddof=1))
    alpha_t = alpha_val / alpha_se if alpha_se > 0 else 0.0
    alpha_p = float(2 * (1 - sp_stats.norm.cdf(abs(alpha_t))))

    beta_ses = {}
    beta_ts = {}
    beta_ps = {}
    z = sp_stats.norm.ppf((1 + confidence_level) / 2)
    beta_cis = {}

    for i, f in enumerate(factor_names):
        se = float(np.std(boot_betas[:, i], ddof=1))
        beta_ses[f] = se
        t = betas[f] / se if se > 0 else 0.0
        beta_ts[f] = t
        beta_ps[f] = float(2 * (1 - sp_stats.norm.cdf(abs(t))))
        beta_cis[f] = (betas[f] - z * se, betas[f] + z * se)

    # R²
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1 - (1 - r_squared) * (T - 1) / (T - K - 1) if T > K + 1 else r_squared

    return FactorModelResult(
        alpha=alpha_val,
        alpha_se=alpha_se,
        alpha_t=alpha_t,
        alpha_p=alpha_p,
        betas=betas,
        beta_ses=beta_ses,
        beta_ts=beta_ts,
        beta_ps=beta_ps,
        beta_cis=beta_cis,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        residuals=residuals,
        durbin_watson=_durbin_watson(residuals),
        factor_names=factor_names,
        n_obs=T,
        method=method,
        hac=False,
        confidence_level=confidence_level,
    )
