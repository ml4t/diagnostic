"""Tests for Kalman filter factor model (Tier 2)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.kalman import compute_kalman_betas
from ml4t.diagnostic.evaluation.factor.results import RollingExposureResult


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, FactorData]:
    np.random.seed(42)
    T = 300
    mkt = np.random.normal(0.0004, 0.01, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 0.8 * mkt + eps

    dates = pl.date_range(date(2019, 1, 1), date(2020, 6, 30), eager=True)[:T]
    factor_df = pl.DataFrame(
        {
            "timestamp": dates,
            "Mkt-RF": mkt,
        }
    )
    return returns, FactorData.from_dataframe(factor_df)


class TestKalmanBetas:
    def test_basic(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_kalman_betas(returns, fd, optimize_noise=False)

        assert isinstance(result, RollingExposureResult)
        assert result.window == 0
        assert len(result.timestamps) == len(returns)

    def test_beta_recovery(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_kalman_betas(returns, fd, optimize_noise=False)

        # Kalman should converge to true beta (~0.8) in steady state
        mkt_betas = result.rolling_betas["Mkt-RF"]
        # Use second half where filter has converged
        steady_state = mkt_betas[len(mkt_betas) // 2 :]
        assert abs(np.mean(steady_state) - 0.8) < 0.3

    def test_with_noise_optimization(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_kalman_betas(returns, fd, optimize_noise=True)
        assert len(result.timestamps) > 0

    def test_manual_noise_params(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_kalman_betas(
            returns,
            fd,
            observation_noise=0.00001,
            state_noise=0.000001,
        )
        assert len(result.timestamps) > 0

    def test_stability_diagnostics(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, fd = synthetic_data
        result = compute_kalman_betas(returns, fd, optimize_noise=False)

        stab = result.stability
        assert "Mkt-RF" in stab.beta_std
        assert stab.r_squared_mean is not None

    def test_multi_factor(self) -> None:
        np.random.seed(42)
        T = 300
        mkt = np.random.normal(0.0004, 0.01, T)
        smb = np.random.normal(0.0001, 0.005, T)
        eps = np.random.normal(0, 0.003, T)
        returns = 0.0002 + 1.0 * mkt + 0.3 * smb + eps

        dates = pl.date_range(date(2019, 1, 1), date(2020, 6, 30), eager=True)[:T]
        fd = FactorData.from_dataframe(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "Mkt-RF": mkt,
                    "SMB": smb,
                }
            )
        )
        result = compute_kalman_betas(returns, fd, optimize_noise=False)
        assert len(result.rolling_betas) == 2
