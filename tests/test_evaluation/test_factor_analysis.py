"""Tests for FactorAnalysis orchestrator (integration tests)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.analysis import FactorAnalysis
from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import (
    AttributionResult,
    FactorModelResult,
    ModelValidationResult,
    RiskAttributionResult,
    RollingExposureResult,
)


@pytest.fixture
def analysis() -> FactorAnalysis:
    np.random.seed(42)
    T = 500
    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    hml = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb - 0.1 * hml + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame(
        {
            "timestamp": dates,
            "Mkt-RF": mkt,
            "SMB": smb,
            "HML": hml,
        }
    )
    factor_data = FactorData.from_dataframe(factor_df)
    return FactorAnalysis(returns, factor_data)


class TestFactorAnalysis:
    def test_properties(self, analysis: FactorAnalysis) -> None:
        assert analysis.factor_names == ["Mkt-RF", "SMB", "HML"]
        assert analysis.n_periods == 500

    def test_static_model(self, analysis: FactorAnalysis) -> None:
        result = analysis.static_model()
        assert isinstance(result, FactorModelResult)
        assert abs(result.betas["Mkt-RF"] - 1.0) < 0.15

    def test_static_model_cached(self, analysis: FactorAnalysis) -> None:
        r1 = analysis.static_model()
        r2 = analysis.static_model()
        assert r1 is r2  # Same object from cache

    def test_rolling_model(self, analysis: FactorAnalysis) -> None:
        result = analysis.rolling_model(window=63)
        assert isinstance(result, RollingExposureResult)
        assert result.window == 63

    def test_attribution(self, analysis: FactorAnalysis) -> None:
        result = analysis.attribution(window=63)
        assert isinstance(result, AttributionResult)
        assert len(result.timestamps) > 0

    def test_risk_attribution(self, analysis: FactorAnalysis) -> None:
        result = analysis.risk_attribution()
        assert isinstance(result, RiskAttributionResult)
        assert result.total_variance > 0

    def test_validate_model(self, analysis: FactorAnalysis) -> None:
        result = analysis.validate_model()
        assert isinstance(result, ModelValidationResult)
        assert result.r_squared > 0

    def test_maximal_attribution(self, analysis: FactorAnalysis) -> None:
        result = analysis.maximal_attribution(factors_of_interest=["Mkt-RF"])
        assert "Mkt-RF" in result.adjusted_betas

    def test_clear_cache(self, analysis: FactorAnalysis) -> None:
        analysis.static_model()
        assert len(analysis._cache) > 0
        analysis.clear_cache()
        assert len(analysis._cache) == 0

    def test_full_pipeline(self, analysis: FactorAnalysis) -> None:
        """Test complete analysis pipeline end-to-end."""
        model = analysis.static_model()
        rolling = analysis.rolling_model(window=63)
        attr = analysis.attribution(window=63)
        risk = analysis.risk_attribution()
        validation = analysis.validate_model()

        # All results should be non-trivial
        assert model.r_squared > 0
        assert len(rolling.timestamps) > 0
        assert len(attr.timestamps) > 0
        assert risk.total_variance > 0
        assert validation.r_squared > 0

        # Model quality check
        assert model.durbin_watson > 0
        assert validation.condition_number > 0


class TestFactorAnalysisTier2:
    def test_factor_timing(self, analysis: FactorAnalysis) -> None:
        result = analysis.factor_timing(window=63)
        for f in analysis.factor_names:
            assert f in result.correlations
            assert np.isfinite(result.correlations[f]) or np.isnan(result.correlations[f])

    def test_kalman_model(self, analysis: FactorAnalysis) -> None:
        result = analysis.kalman_model()
        assert isinstance(result, RollingExposureResult)
        assert result.window == 0  # Kalman indicator
        for f in analysis.factor_names:
            assert f in result.rolling_betas
