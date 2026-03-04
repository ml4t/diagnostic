"""Reference validation: verify factor module against raw statsmodels.

These tests confirm that our implementation produces the same results
as a direct statsmodels regression, serving as a cross-check.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from scipy import stats as sp_stats

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


@pytest.fixture
def reference_data() -> tuple[np.ndarray, np.ndarray, FactorData]:
    """Generate data and return (returns, factor_matrix, FactorData)."""
    np.random.seed(123)
    T = 1000
    mkt = np.random.normal(0.0004, 0.012, T)
    smb = np.random.normal(0.0001, 0.006, T)
    hml = np.random.normal(0.0001, 0.006, T)
    eps = np.random.normal(0, 0.004, T)
    returns = 0.0003 + 0.95 * mkt + 0.4 * smb - 0.15 * hml + eps

    dates = pl.date_range(date(2016, 1, 1), date(2019, 12, 31), eager=True)[:T]
    X = np.column_stack([mkt, smb, hml])
    factor_df = pl.DataFrame({"timestamp": dates, "Mkt-RF": mkt, "SMB": smb, "HML": hml})
    fd = FactorData.from_dataframe(factor_df)
    return returns, X, fd


class TestVsStatsmodelsOLS:
    """Verify our OLS matches statsmodels exactly."""

    def test_betas_match(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        # Direct statsmodels
        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        for i, f in enumerate(fd.factor_names):
            assert abs(our.betas[f] - sm_result.params[i + 1]) < 1e-10

    def test_alpha_matches(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        assert abs(our.alpha - sm_result.params[0]) < 1e-10

    def test_standard_errors_match(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        assert abs(our.alpha_se - sm_result.bse[0]) < 1e-10
        for i, f in enumerate(fd.factor_names):
            assert abs(our.beta_ses[f] - sm_result.bse[i + 1]) < 1e-10

    def test_r_squared_matches(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        assert abs(our.r_squared - sm_result.rsquared) < 1e-10
        assert abs(our.adj_r_squared - sm_result.rsquared_adj) < 1e-10

    def test_t_stats_match(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        assert abs(our.alpha_t - sm_result.tvalues[0]) < 1e-8
        for i, f in enumerate(fd.factor_names):
            assert abs(our.beta_ts[f] - sm_result.tvalues[i + 1]) < 1e-8

    def test_p_values_match(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit()

        assert abs(our.alpha_p - sm_result.pvalues[0]) < 1e-8
        for i, f in enumerate(fd.factor_names):
            assert abs(our.beta_ps[f] - sm_result.pvalues[i + 1]) < 1e-8

    def test_hac_standard_errors_match(self, reference_data: tuple) -> None:
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=True, max_lags=5)

        X_const = sm.add_constant(X)
        sm_result = sm.OLS(returns, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        assert abs(our.alpha_se - sm_result.bse[0]) < 1e-10
        for i, f in enumerate(fd.factor_names):
            assert abs(our.beta_ses[f] - sm_result.bse[i + 1]) < 1e-10


class TestBetaRecoveryTight:
    """Verify beta recovery with tight tolerances on large sample."""

    def test_beta_recovery_1000obs(self, reference_data: tuple) -> None:
        """With T=1000, betas should be recovered within ~0.05."""
        returns, X, fd = reference_data
        our = compute_factor_model(returns, fd, hac=False)

        assert abs(our.betas["Mkt-RF"] - 0.95) < 0.05
        assert abs(our.betas["SMB"] - 0.4) < 0.05
        assert abs(our.betas["HML"] - (-0.15)) < 0.05
        assert abs(our.alpha - 0.0003) < 0.001

    def test_true_betas_inside_ci(self, reference_data: tuple) -> None:
        """True parameter values should lie within 95% CIs."""
        returns, _, fd = reference_data
        our = compute_factor_model(returns, fd, confidence_level=0.95)

        true = {"Mkt-RF": 0.95, "SMB": 0.4, "HML": -0.15}
        for f, true_val in true.items():
            lo, hi = our.beta_cis[f]
            assert lo < true_val < hi, f"True {f}={true_val} outside CI [{lo}, {hi}]"


class TestEdgeCases:
    def test_single_factor(self) -> None:
        """Model works with just one factor."""
        np.random.seed(42)
        T = 200
        mkt = np.random.normal(0, 0.01, T)
        returns = 0.8 * mkt + np.random.normal(0, 0.005, T)
        dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "Mkt": mkt}))
        result = compute_factor_model(returns, fd)
        assert abs(result.betas["Mkt"] - 0.8) < 0.1

    def test_many_factors(self) -> None:
        """Model works with 10 factors (still identified)."""
        np.random.seed(42)
        T = 500
        n_factors = 10
        X = np.random.normal(0, 0.01, (T, n_factors))
        betas_true = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        returns = X @ betas_true + np.random.normal(0, 0.005, T)

        dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
        cols = {"timestamp": dates}
        for k in range(n_factors):
            cols[f"F{k}"] = X[:, k]
        fd = FactorData.from_dataframe(pl.DataFrame(cols))
        result = compute_factor_model(returns, fd)

        # First 3 (strong) betas should be well-estimated
        assert abs(result.betas["F0"] - 1.0) < 0.1
        assert abs(result.betas["F1"] - 0.5) < 0.1
        assert abs(result.betas["F2"] - (-0.3)) < 0.1

    def test_with_nan_values(self) -> None:
        """Model handles NaN values in returns gracefully."""
        np.random.seed(42)
        T = 200
        mkt = np.random.normal(0, 0.01, T)
        returns = 1.0 * mkt + np.random.normal(0, 0.005, T)
        returns[10] = np.nan
        returns[50] = np.nan

        dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "Mkt": mkt}))
        result = compute_factor_model(returns, fd)
        assert result.n_obs == T - 2  # 2 NaN rows dropped

    def test_zero_variance_returns(self) -> None:
        """Model handles zero-variance returns (all zeros).

        When y is all zeros, SST=0 → R² is NaN (statsmodels convention).
        Betas and residuals should all be exactly zero.
        """
        np.random.seed(42)
        T = 100
        mkt = np.random.normal(0, 0.01, T)
        returns = np.zeros(T)

        dates = pl.date_range(date(2020, 1, 1), date(2020, 6, 30), eager=True)[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({"timestamp": dates, "Mkt": mkt}))
        result = compute_factor_model(returns, fd)
        assert np.isnan(result.r_squared) or result.r_squared < 0.01
        assert abs(result.betas["Mkt"]) < 1e-10
        assert np.all(np.abs(result.residuals) < 1e-10)
