"""Tests for risk attribution (variance decomposition)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import RiskAttributionResult
from ml4t.diagnostic.evaluation.factor.risk import compute_risk_attribution
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


@pytest.fixture
def synthetic_data(
    synthetic_2f_data: tuple[np.ndarray, FactorData],
) -> tuple[np.ndarray, FactorData]:
    """Alias for shared 2-factor fixture."""
    return synthetic_2f_data


class TestRiskAttribution:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        assert isinstance(result, RiskAttributionResult)
        assert result.total_variance > 0
        assert result.factor_variance >= 0
        assert result.idiosyncratic_variance >= 0

    def test_variance_decomposition_sums(
        self, synthetic_data: tuple[np.ndarray, FactorData]
    ) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        # Factor + idiosyncratic should equal total
        assert (
            abs(result.factor_variance + result.idiosyncratic_variance - result.total_variance)
            < 1e-8
        )

    def test_euler_decomposition_sums(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        """Euler contributions should sum to factor variance."""
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        total_contrib = sum(result.factor_contributions.values())
        assert abs(total_contrib - result.factor_variance) < 1e-8

    def test_percentages_sum(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        total_pct = sum(result.factor_contributions_pct.values())
        assert abs(total_pct - result.factor_variance_pct) < 1e-6

    def test_mctr(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        for f in fd.factor_names:
            assert f in result.mctr
            assert np.isfinite(result.mctr[f])

    def test_shrinkage_methods(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data

        result_none = compute_risk_attribution(returns, fd, shrinkage="none")
        result_lw = compute_risk_attribution(returns, fd, shrinkage="ledoit_wolf")
        result_oas = compute_risk_attribution(returns, fd, shrinkage="oracle")

        assert result_none.shrinkage == "none"
        assert result_lw.shrinkage == "ledoit_wolf"
        assert result_oas.shrinkage == "oracle"

        # All should produce valid results
        for r in [result_none, result_lw, result_oas]:
            assert r.total_variance > 0

    def test_with_precomputed_model(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        model = compute_factor_model(returns, fd)
        result = compute_risk_attribution(returns, fd, model_result=model)
        assert result.total_variance > 0

    def test_to_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "factor" in df.columns
        assert "variance_contribution" in df.columns
        assert "mctr" in df.columns

    def test_to_dict(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)
        d = result.to_dict()

        assert "total_variance" in d
        assert "factor_variance_pct" in d
        assert "mctr" in d

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)
        s = result.summary()

        assert "Risk Attribution" in s
        assert "Mkt-RF" in s
        assert "MCTR" in s

    def test_mkt_dominates_risk(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        """With beta_mkt=1.0 and beta_smb=0.3, Mkt should dominate risk."""
        returns, fd = synthetic_data
        result = compute_risk_attribution(returns, fd)

        assert result.factor_contributions_pct["Mkt-RF"] > result.factor_contributions_pct["SMB"]
