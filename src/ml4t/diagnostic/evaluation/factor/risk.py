"""Variance-based risk decomposition and marginal contribution to risk.

Decomposes portfolio variance into factor-explained and idiosyncratic
components using Euler decomposition for exact additivity.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from .data import FactorData
from .results import FactorModelResult, RiskAttributionResult
from .static_model import _align_and_prepare


def compute_risk_attribution(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    model_result: FactorModelResult | None = None,
    shrinkage: Literal["none", "ledoit_wolf", "oracle"] = "ledoit_wolf",
) -> RiskAttributionResult:
    """Decompose portfolio risk into factor and idiosyncratic components.

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    model_result : FactorModelResult | None
        Pre-computed model. If None, computed internally.
    shrinkage : str
        Covariance shrinkage method:
        - "none": sample covariance
        - "ledoit_wolf": Ledoit-Wolf shrinkage (default, stable with 5+ factors)
        - "oracle": Oracle approximating shrinkage

    Returns
    -------
    RiskAttributionResult
        Variance decomposition with factor contributions and MCTR.
    """
    from .static_model import compute_factor_model

    if model_result is None:
        model_result = compute_factor_model(returns, factor_data)

    y, X, timestamps = _align_and_prepare(returns, factor_data)
    factor_names = factor_data.factor_names

    betas = np.array([model_result.betas[f] for f in factor_names])

    # Factor covariance matrix
    Sigma_F = _compute_factor_covariance(X, shrinkage)

    # Factor variance: beta' Sigma_F beta
    factor_variance = float(betas @ Sigma_F @ betas)

    # Idiosyncratic variance: variance of residuals
    idiosyncratic_variance = float(np.var(model_result.residuals, ddof=1))

    # Total variance
    total_variance = factor_variance + idiosyncratic_variance

    # Euler decomposition: contribution_k = beta_k * (Sigma_F @ beta)_k
    Sigma_beta = Sigma_F @ betas
    factor_contributions: dict[str, float] = {}
    factor_contributions_pct: dict[str, float] = {}
    for k, f in enumerate(factor_names):
        contrib = float(betas[k] * Sigma_beta[k])
        factor_contributions[f] = contrib
        factor_contributions_pct[f] = contrib / total_variance if total_variance > 0 else 0.0

    # MCTR: (Sigma_F @ beta)_k / sigma_portfolio
    total_vol = np.sqrt(total_variance) if total_variance > 0 else 1e-10
    mctr: dict[str, float] = {}
    for k, f in enumerate(factor_names):
        mctr[f] = float(Sigma_beta[k] / total_vol)

    return RiskAttributionResult(
        total_variance=total_variance,
        factor_variance=factor_variance,
        idiosyncratic_variance=idiosyncratic_variance,
        factor_contributions=factor_contributions,
        factor_contributions_pct=factor_contributions_pct,
        mctr=mctr,
        factor_names=factor_names,
        shrinkage=shrinkage,
    )


def _compute_factor_covariance(
    X: np.ndarray,
    shrinkage: str,
) -> np.ndarray:
    """Compute factor covariance matrix with optional shrinkage."""
    if shrinkage == "none":
        return np.cov(X, rowvar=False)

    if shrinkage == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(X)
        return lw.covariance_

    if shrinkage == "oracle":
        from sklearn.covariance import OAS

        oas = OAS().fit(X)
        return oas.covariance_

    raise ValueError(f"Unknown shrinkage method: {shrinkage}")
