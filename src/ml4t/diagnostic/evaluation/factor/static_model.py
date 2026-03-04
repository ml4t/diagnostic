"""Static factor model estimation via OLS with HAC standard errors.

Implements time-series regression of portfolio returns on factor returns
with Newey-West (HAC) standard errors using Bartlett kernel.

References
----------
- Newey & West (1987): Bartlett kernel HAC estimator
- Andrews (1991): Automatic bandwidth selection: 4*(T/100)^(2/9)
- Paleologo (2025), Ch 5: t-stat percentage > 2 as model quality metric
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from .data import FactorData
from .results import FactorModelResult


def compute_factor_model(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    method: Literal["ols"] = "ols",
    hac: bool = True,
    max_lags: int | None = None,
    confidence_level: float = 0.95,
) -> FactorModelResult:
    """Estimate static factor model via OLS with optional HAC standard errors.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio excess returns (T,). If FactorData has rf_rate,
        the risk-free rate is subtracted automatically.
    factor_data : FactorData
        Factor return data.
    method : str
        Estimation method. Currently only "ols" supported here;
        ridge/lasso/elastic_net delegate to regularized.py.
    hac : bool
        Use Newey-West HAC standard errors (Bartlett kernel).
    max_lags : int | None
        Maximum lags for HAC. Default: Andrews rule 4*(T/100)^(2/9).
    confidence_level : float
        Confidence level for CIs (e.g. 0.95).

    Returns
    -------
    FactorModelResult
        Complete regression output with betas, SEs, t-stats, CIs.
    """
    import statsmodels.api as sm

    y, X, timestamps = _align_and_prepare(returns, factor_data)
    T, K = X.shape

    if max_lags is None:
        max_lags = int(4 * (T / 100) ** (2 / 9))
        max_lags = max(1, max_lags)

    # Add constant for alpha
    X_with_const = sm.add_constant(X)

    model = sm.OLS(y, X_with_const)
    if hac:
        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    else:
        result = model.fit()

    # Extract results
    params = result.params
    bse = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues

    alpha = float(params[0])
    alpha_se = float(bse[0])
    alpha_t = float(tvalues[0])
    alpha_p = float(pvalues[0])

    factor_names = factor_data.factor_names
    betas = {f: float(params[i + 1]) for i, f in enumerate(factor_names)}
    beta_ses = {f: float(bse[i + 1]) for i, f in enumerate(factor_names)}
    beta_ts = {f: float(tvalues[i + 1]) for i, f in enumerate(factor_names)}
    beta_ps = {f: float(pvalues[i + 1]) for i, f in enumerate(factor_names)}

    # Use t-distribution with T-K-1 degrees of freedom (not normal)
    dof = max(T - K - 1, 1)
    t_crit = sp_stats.t.ppf((1 + confidence_level) / 2, df=dof)
    beta_cis = {}
    for f in factor_names:
        b = betas[f]
        se = beta_ses[f]
        beta_cis[f] = (b - t_crit * se, b + t_crit * se)

    residuals = np.asarray(result.resid, dtype=np.float64)

    # Durbin-Watson
    dw = _durbin_watson(residuals)

    return FactorModelResult(
        alpha=alpha,
        alpha_se=alpha_se,
        alpha_t=alpha_t,
        alpha_p=alpha_p,
        betas=betas,
        beta_ses=beta_ses,
        beta_ts=beta_ts,
        beta_ps=beta_ps,
        beta_cis=beta_cis,
        r_squared=float(result.rsquared),
        adj_r_squared=float(result.rsquared_adj),
        residuals=residuals,
        durbin_watson=dw,
        factor_names=factor_names,
        n_obs=T,
        method=method,
        hac=hac,
        confidence_level=confidence_level,
    )


def _align_and_prepare(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align returns with factor data and subtract risk-free rate.

    Returns
    -------
    tuple of (y, X, timestamps)
        y: (T,) excess returns
        X: (T, K) factor returns matrix
        timestamps: (T,) aligned timestamps
    """
    if isinstance(returns, pl.Series):
        y = returns.to_numpy().astype(np.float64)
    else:
        y = np.asarray(returns, dtype=np.float64)

    X = factor_data.get_factor_array().astype(np.float64)
    timestamps = factor_data.get_timestamps()

    # Align lengths (use minimum) — warn if significant mismatch
    len_y, len_x = len(y), len(X)
    if len_y != len_x:
        warnings.warn(
            f"Returns length ({len_y}) differs from factor data length ({len_x}). "
            f"Truncating to {min(len_y, len_x)} observations.",
            stacklevel=3,
        )
    T = min(len_y, len_x)
    y = y[:T]
    X = X[:T]
    timestamps = timestamps[:T]

    # Subtract risk-free rate
    if factor_data.rf_rate is not None:
        rf = factor_data.rf_rate.to_numpy().astype(np.float64)[:T]
        y = y - rf

    # Drop rows with NaN
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    timestamps = timestamps[mask]

    if len(y) < X.shape[1] + 2:
        raise ValueError(
            f"Not enough observations ({len(y)}) for {X.shape[1]} factors. "
            f"Need at least {X.shape[1] + 2}."
        )

    return y, X, timestamps


def _durbin_watson(residuals: np.ndarray) -> float:
    """Compute Durbin-Watson statistic (2.0 = no autocorrelation)."""
    diff = np.diff(residuals)
    ss_resid = np.sum(residuals**2)
    if ss_resid == 0:
        return 2.0  # No residuals means no autocorrelation
    return float(np.sum(diff**2) / ss_resid)
