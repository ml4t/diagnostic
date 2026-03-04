"""Reference validation: maximal attribution against hand-computed example.

Verifies the Paleologo Ch 14 maximal attribution formula using a
2-factor system where we can compute the answer analytically.

Setup:
    Factors: F1, F2 with known correlation rho
    Betas: beta_1, beta_2 (from OLS)
    Factor of interest: S = {F1}
    Nuisance factor: U = {F2}

    Rotation matrix: A = Cov(F2,F1) / Var(F1) = rho * sigma_2 / sigma_1
    Adjusted beta: beta_adj = beta_1 + A * beta_2
    Maximal PnL: beta_adj * mean(F1) * T
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.attribution import compute_maximal_attribution
from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


class TestMaximalAttributionHandComputed:
    """Verify against analytically-derived answers."""

    def test_uncorrelated_factors(self) -> None:
        """When factors are uncorrelated, adjusted beta = original beta."""
        np.random.seed(42)
        T = 10000  # large T for tight convergence

        # Generate truly independent factors
        f1 = np.random.normal(0.001, 0.01, T)
        f2 = np.random.normal(0.0005, 0.008, T)

        # True model
        beta1, beta2 = 1.2, 0.5
        eps = np.random.normal(0, 0.002, T)
        returns = 0.0001 + beta1 * f1 + beta2 * f2 + eps

        dates = pl.date_range(date(2010, 1, 1), date(2049, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "F1": f1, "F2": f2}))

        result = compute_maximal_attribution(returns, fd, factors_of_interest=["F1"])

        # With uncorrelated factors, adjusted beta ≈ original beta
        model = compute_factor_model(returns, fd)
        assert abs(result.adjusted_betas["F1"] - model.betas["F1"]) < 0.05

    def test_correlated_factors_analytical(self) -> None:
        """Verify formula: beta_adj = beta_S + A' @ beta_U.

        Construct factors with known correlation via Cholesky, then verify
        the maximal attribution matches the analytical formula.
        """
        np.random.seed(123)
        T = 10000

        # Generate correlated factors via Cholesky
        rho = 0.6
        sigma1, sigma2 = 0.01, 0.008
        chol = np.array([[1, 0], [rho, np.sqrt(1 - rho**2)]])
        z = np.random.normal(0, 1, (T, 2))
        factors = z @ chol.T
        f1 = factors[:, 0] * sigma1
        f2 = factors[:, 1] * sigma2

        # True model
        beta1, beta2 = 0.9, 0.4
        eps = np.random.normal(0, 0.002, T)
        returns = beta1 * f1 + beta2 * f2 + eps

        dates = pl.date_range(date(2010, 1, 1), date(2049, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "F1": f1, "F2": f2}))

        result = compute_maximal_attribution(returns, fd, factors_of_interest=["F1"])

        # Analytical: A = Cov(F2,F1) / Var(F1) (scalar for 1 interest factor)
        cov_matrix = np.cov(f1, f2)
        a_analytical = cov_matrix[1, 0] / cov_matrix[0, 0]

        model = compute_factor_model(returns, fd, hac=False)
        beta_adj_analytical = model.betas["F1"] + a_analytical * model.betas["F2"]

        assert abs(result.adjusted_betas["F1"] - beta_adj_analytical) < 1e-10
        assert abs(result.rotation_matrix[0, 0] - a_analytical) < 1e-10

    def test_maximal_pnl_formula(self) -> None:
        """Verify: maximal_pnl = adjusted_beta * mean(factor) * T."""
        np.random.seed(456)
        T = 5000

        f1 = np.random.normal(0.001, 0.01, T)
        f2 = np.random.normal(0.0005, 0.008, T)
        eps = np.random.normal(0, 0.002, T)
        returns = 0.8 * f1 + 0.3 * f2 + eps

        dates = pl.date_range(date(2010, 1, 1), date(2029, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "F1": f1, "F2": f2}))

        result = compute_maximal_attribution(returns, fd, factors_of_interest=["F1"])

        expected_pnl = result.adjusted_betas["F1"] * np.mean(f1) * T
        assert abs(result.maximal_pnl["F1"] - expected_pnl) < 1e-10

    def test_orthogonal_residual(self) -> None:
        """Orthogonal residual = total_factor_pnl - maximal_pnl."""
        np.random.seed(789)
        T = 5000

        f1 = np.random.normal(0.001, 0.01, T)
        f2 = np.random.normal(0.0005, 0.008, T)
        eps = np.random.normal(0, 0.002, T)
        returns = 1.0 * f1 + 0.5 * f2 + eps

        dates = pl.date_range(date(2010, 1, 1), date(2029, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "F1": f1, "F2": f2}))

        model = compute_factor_model(returns, fd)
        result = compute_maximal_attribution(
            returns, fd, factors_of_interest=["F1"], model_result=model
        )

        # Total factor PnL = sum(beta_k * mean(F_k) * T)
        total_factor_pnl = sum(
            model.betas[f] * np.mean(fd.get_factor_array()[:, i]) * T
            for i, f in enumerate(fd.factor_names)
        )
        expected_residual = total_factor_pnl - result.maximal_pnl["F1"]
        assert abs(result.orthogonal_residual - expected_residual) < 1e-10

    def test_symmetry_swap_interest(self) -> None:
        """Swapping factor of interest should give different adjusted betas."""
        np.random.seed(42)
        T = 5000

        f1 = np.random.normal(0.001, 0.01, T)
        f2 = np.random.normal(0.0005, 0.008, T)
        eps = np.random.normal(0, 0.002, T)
        returns = 1.0 * f1 + 0.3 * f2 + eps

        dates = pl.date_range(date(2010, 1, 1), date(2029, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "F1": f1, "F2": f2}))

        r1 = compute_maximal_attribution(returns, fd, factors_of_interest=["F1"])
        r2 = compute_maximal_attribution(returns, fd, factors_of_interest=["F2"])

        # Adjusted betas should differ (asymmetric by construction)
        assert r1.adjusted_betas["F1"] != r2.adjusted_betas["F2"]
