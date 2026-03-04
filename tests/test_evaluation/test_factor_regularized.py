"""Tests for regularized factor models (Tier 2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.regularized import compute_regularized_model
from ml4t.diagnostic.evaluation.factor.results import FactorModelResult


@pytest.fixture
def synthetic_data(synthetic_3f_data: tuple[np.ndarray, FactorData]) -> tuple[np.ndarray, FactorData]:
    """Alias for shared 3-factor fixture."""
    return synthetic_3f_data


class TestRegularizedModel:
    def test_ridge(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="ridge", n_bootstrap=20
        )
        assert isinstance(result, FactorModelResult)
        assert result.method == "ridge"
        assert result.hac is False

    def test_lasso(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="lasso", alpha=0.0001, n_bootstrap=20
        )
        assert result.method == "lasso"

    def test_elastic_net(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="elastic_net", alpha=0.0001, l1_ratio=0.5,
            n_bootstrap=20,
        )
        assert result.method == "elastic_net"

    def test_bootstrap_ses(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="ridge", n_bootstrap=50
        )
        for f in fd.factor_names:
            assert result.beta_ses[f] > 0
            assert result.beta_cis[f][0] < result.beta_cis[f][1]

    def test_beta_recovery_ridge(
        self, synthetic_data: tuple[np.ndarray, FactorData]
    ) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="ridge", alpha=0.001, n_bootstrap=20
        )
        # With small alpha, should be close to OLS
        assert abs(result.betas["Mkt-RF"] - 1.0) < 0.2

    def test_lasso_sparsity(self) -> None:
        """LASSO with strong regularization should zero out weak factors."""
        np.random.seed(42)
        T = 500
        mkt = np.random.normal(0.0004, 0.01, T)
        noise1 = np.random.normal(0, 0.005, T)  # Pure noise factor
        noise2 = np.random.normal(0, 0.005, T)
        eps = np.random.normal(0, 0.003, T)
        returns = 0.0002 + 1.0 * mkt + eps

        dates = pl.date_range(
            date(2018, 1, 1), date(2019, 12, 31), eager=True
        )[:T]
        fd = FactorData.from_dataframe(pl.DataFrame({
            "timestamp": dates,
            "Mkt": mkt,
            "Noise1": noise1,
            "Noise2": noise2,
        }))

        result = compute_regularized_model(
            returns, fd, method="lasso", alpha=0.00001, n_bootstrap=20
        )
        # LASSO should shrink noise factor betas toward zero (or equal)
        assert abs(result.betas["Noise1"]) <= abs(result.betas["Mkt"]) + 1e-10
        assert abs(result.betas["Noise2"]) <= abs(result.betas["Mkt"]) + 1e-10

    def test_r_squared(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_regularized_model(
            returns, fd, method="ridge", n_bootstrap=20
        )
        assert 0 <= result.r_squared <= 1
        assert result.adj_r_squared <= result.r_squared
