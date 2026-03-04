"""Tests for rolling factor model."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import RollingExposureResult
from ml4t.diagnostic.evaluation.factor.rolling_model import compute_rolling_exposures


@pytest.fixture
def synthetic_data(
    synthetic_2f_data: tuple[np.ndarray, FactorData],
) -> tuple[np.ndarray, FactorData]:
    """Alias for shared 2-factor fixture."""
    return synthetic_2f_data


class TestRollingExposures:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)

        assert isinstance(result, RollingExposureResult)
        assert result.window == 63
        assert len(result.timestamps) > 0
        assert len(result.factor_names) == 2

    def test_output_shapes(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)

        n = len(result.timestamps)
        assert result.rolling_alpha.shape == (n,)
        assert result.rolling_r_squared.shape == (n,)
        for f in fd.factor_names:
            assert result.rolling_betas[f].shape == (n,)

    def test_beta_values_reasonable(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=126)

        # Rolling betas should center around true values (mean over windows)
        mkt_betas = result.rolling_betas["Mkt-RF"]
        valid = mkt_betas[np.isfinite(mkt_betas)]
        assert abs(np.mean(valid) - 1.0) < 0.15

    def test_stability_diagnostics(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)

        stab = result.stability
        for f in fd.factor_names:
            assert f in stab.beta_std
            assert f in stab.sign_consistency
            assert f in stab.max_abs_change
            assert stab.sign_consistency[f] >= 0
            assert stab.sign_consistency[f] <= 1

        assert stab.r_squared_mean >= 0

    def test_expanding_window(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63, expanding=True, min_periods=30)
        assert len(result.timestamps) > 0

    def test_vif_computation(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63, compute_vif=True)

        assert result.stability.vif is not None
        for f in fd.factor_names:
            assert f in result.stability.vif
            assert result.stability.vif[f] >= 1.0  # VIF >= 1 by definition

    def test_insufficient_window_raises(
        self, synthetic_data: tuple[np.ndarray, FactorData]
    ) -> None:
        returns, fd = synthetic_data
        with pytest.raises(ValueError, match="Not enough data"):
            compute_rolling_exposures(returns[:10], fd, window=100)

    def test_to_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)
        df = result.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "factor" in df.columns
        assert "beta" in df.columns

    def test_to_wide_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)
        df = result.to_wide_dataframe()
        assert "Mkt-RF" in df.columns
        assert "alpha" in df.columns
        assert "r_squared" in df.columns

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_rolling_exposures(returns, fd, window=63)
        s = result.summary()
        assert "Rolling Exposure" in s
        assert "Window: 63" in s
