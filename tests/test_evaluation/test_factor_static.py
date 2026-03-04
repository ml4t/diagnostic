"""Tests for static factor model (OLS + HAC)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import FactorModelResult
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model


@pytest.fixture
def synthetic_data(
    synthetic_3f_data: tuple[np.ndarray, FactorData],
) -> tuple[np.ndarray, FactorData]:
    """Alias for shared 3-factor fixture."""
    return synthetic_3f_data


class TestComputeFactorModel:
    def test_basic_ols(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data, hac=False)

        assert isinstance(result, FactorModelResult)
        assert result.method == "ols"
        assert result.n_obs == 500
        assert len(result.factor_names) == 3

    def test_beta_recovery(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        """Test that OLS recovers known betas within tolerance.

        With T=500, sigma_eps=0.003, sigma_mkt=0.01: SE(beta_mkt)~0.013.
        So 0.05 tolerance is ~4 SE — generous but physically meaningful.
        """
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data, hac=False)

        assert abs(result.betas["Mkt-RF"] - 1.0) < 0.05
        assert abs(result.betas["SMB"] - 0.3) < 0.08
        assert abs(result.betas["HML"] - (-0.1)) < 0.08

    def test_alpha_near_zero(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        assert abs(result.alpha) < 0.002

    def test_hac_standard_errors(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result_hac = compute_factor_model(returns, factor_data, hac=True)
        result_ols = compute_factor_model(returns, factor_data, hac=False)

        # HAC SEs should generally differ from OLS SEs
        assert result_hac.hac is True
        assert result_ols.hac is False

        # Both should have reasonable SEs
        for f in factor_data.factor_names:
            assert result_hac.beta_ses[f] > 0
            assert result_ols.beta_ses[f] > 0

    def test_r_squared(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        # With known factors and moderate noise, R² should be high
        assert result.r_squared > 0.7
        assert result.adj_r_squared <= result.r_squared

    def test_confidence_intervals(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data, confidence_level=0.95)

        for f in factor_data.factor_names:
            ci = result.beta_cis[f]
            # CI should contain the beta estimate
            assert ci[0] < result.betas[f] < ci[1]
            # CI width should be positive
            assert ci[1] > ci[0]

    def test_durbin_watson(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        # DW near 2 means no autocorrelation
        assert 1.0 < result.durbin_watson < 3.0

    def test_residuals_shape(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        assert result.residuals.shape == (result.n_obs,)

    def test_t_stat_pct(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        # With true betas of 1.0, 0.3, -0.1, at least Mkt should be significant
        assert result.t_stat_pct_above_2 > 0

    def test_significant_factors(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        # Mkt-RF with beta=1.0 should be significant
        assert "Mkt-RF" in result.significant_factors

    def test_polars_series_input(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        series = pl.Series("returns", returns)
        result = compute_factor_model(series, factor_data)
        assert result.n_obs > 0

    def test_insufficient_data_raises(self) -> None:
        factor_df = pl.DataFrame(
            {
                "timestamp": [date(2020, 1, i) for i in range(1, 4)],
                "A": [0.01, 0.02, -0.01],
                "B": [-0.01, 0.01, 0.02],
                "C": [0.005, -0.005, 0.01],
            }
        )
        fd = FactorData.from_dataframe(factor_df)
        returns = np.array([0.01, -0.02, 0.005])
        with pytest.raises(ValueError, match="Not enough observations"):
            compute_factor_model(returns, fd)


class TestFactorModelResult:
    def test_to_dict(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        d = result.to_dict()

        assert "alpha" in d
        assert "betas" in d
        assert "r_squared" in d
        assert "t_stat_pct_above_2" in d
        assert "significant_factors" in d

    def test_to_dataframe(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 3  # 3 factors
        assert "factor" in df.columns
        assert "beta" in df.columns
        assert "se" in df.columns
        assert "p_value" in df.columns

    def test_summary(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data)
        s = result.summary()

        assert "Factor Model Results" in s
        assert "Mkt-RF" in s
        assert "R²" in s

    def test_custom_confidence_level(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result_90 = compute_factor_model(returns, factor_data, confidence_level=0.90)
        result_99 = compute_factor_model(returns, factor_data, confidence_level=0.99)

        # 99% CI should be wider than 90% CI
        for f in factor_data.factor_names:
            ci_90 = result_90.beta_cis[f]
            ci_99 = result_99.beta_cis[f]
            assert (ci_99[1] - ci_99[0]) > (ci_90[1] - ci_90[0])

    def test_custom_max_lags(self, synthetic_data: tuple[np.ndarray, FactorData]) -> None:
        returns, factor_data = synthetic_data
        result = compute_factor_model(returns, factor_data, max_lags=10)
        assert result.hac is True
        assert result.n_obs > 0
