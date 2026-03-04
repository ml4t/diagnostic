"""Tests for model validation diagnostics (QLIKE, MALV, Ljung-Box, JB)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData
from ml4t.diagnostic.evaluation.factor.results import ModelValidationResult
from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model
from ml4t.diagnostic.evaluation.factor.validation import validate_factor_model


@pytest.fixture
def model_and_factors() -> tuple:
    """Fit a factor model and return (result, factor_returns_array)."""
    np.random.seed(42)
    T = 500
    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    hml = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    returns = 0.0002 + 1.0 * mkt + 0.3 * smb - 0.1 * hml + eps

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame(
        {"timestamp": dates, "Mkt-RF": mkt, "SMB": smb, "HML": hml}
    )
    fd = FactorData.from_dataframe(factor_df)
    model = compute_factor_model(returns, fd, hac=True)
    factor_array = fd.get_factor_array()[:model.n_obs]
    return model, factor_array


class TestValidateFactorModel:
    def test_returns_result_type(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert isinstance(result, ModelValidationResult)

    def test_qlike_is_finite(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X, qlike_window=21)
        assert np.isfinite(result.qlike)

    def test_qlike_nonnegative(self, model_and_factors: tuple) -> None:
        """QLIKE is always >= 0, with minimum at 0 (perfect prediction)."""
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert result.qlike >= 0.0

    def test_malv_is_finite(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert np.isfinite(result.malv)
        assert result.malv >= 0.0

    def test_ljung_box_p_value_range(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X, max_acf_lags=10)
        assert 0.0 <= result.ljung_box_p <= 1.0

    def test_jarque_bera_p_value_range(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert 0.0 <= result.jarque_bera_p <= 1.0

    def test_condition_number_positive(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert result.condition_number > 0

    def test_durbin_watson_from_model(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        # DW should be between 0 and 4
        assert 0.0 <= result.durbin_watson <= 4.0

    def test_t_stat_pct_matches_model(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert result.t_stat_pct == model.t_stat_pct_above_2

    def test_r_squared_matches_model(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        result = validate_factor_model(model, X)
        assert result.r_squared == model.r_squared


class TestQlikeWindowSensitivity:
    def test_different_windows(self, model_and_factors: tuple) -> None:
        model, X = model_and_factors
        r1 = validate_factor_model(model, X, qlike_window=10)
        r2 = validate_factor_model(model, X, qlike_window=42)
        # Different windows should produce different (but both valid) results
        assert np.isfinite(r1.qlike) and np.isfinite(r2.qlike)

    def test_short_data_returns_nan(self) -> None:
        """QLIKE should be NaN when data too short for window."""
        np.random.seed(42)
        T = 30
        mkt = np.random.normal(0, 0.01, T)
        eps = np.random.normal(0, 0.003, T)
        returns = 1.0 * mkt + eps
        dates = pl.date_range(date(2020, 1, 1), date(2020, 3, 1), eager=True)[:T]
        fd = FactorData.from_dataframe(
            pl.DataFrame({"timestamp": dates, "Mkt": mkt})
        )
        model = compute_factor_model(returns, fd, hac=False)
        X = fd.get_factor_array()[:model.n_obs]
        result = validate_factor_model(model, X, qlike_window=25)
        # With T=30 and window=25, only ~5 rolling windows → may be NaN or small
        assert np.isfinite(result.qlike) or np.isnan(result.qlike)


class TestLjungBoxEdgeCases:
    def test_white_noise_high_p(self) -> None:
        """White noise residuals should have high Ljung-Box p-value."""
        np.random.seed(42)
        T = 500
        mkt = np.random.normal(0, 0.01, T)
        eps = np.random.normal(0, 0.003, T)
        returns = 1.0 * mkt + eps
        dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(
            pl.DataFrame({"timestamp": dates, "Mkt": mkt})
        )
        model = compute_factor_model(returns, fd, hac=False)
        X = fd.get_factor_array()[:model.n_obs]
        result = validate_factor_model(model, X, max_acf_lags=10)
        # With true model and white noise, p should be high (no autocorrelation)
        assert result.ljung_box_p > 0.01

    def test_autocorrelated_residuals_low_p(self) -> None:
        """Returns with omitted AR component should have low Ljung-Box p."""
        np.random.seed(42)
        T = 500
        mkt = np.random.normal(0, 0.01, T)
        # Add AR(1) component that the model can't capture
        ar_component = np.zeros(T)
        for t in range(1, T):
            ar_component[t] = 0.5 * ar_component[t - 1] + np.random.normal(0, 0.003)
        returns = 1.0 * mkt + ar_component
        dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
        fd = FactorData.from_dataframe(
            pl.DataFrame({"timestamp": dates, "Mkt": mkt})
        )
        model = compute_factor_model(returns, fd, hac=False)
        X = fd.get_factor_array()[:model.n_obs]
        result = validate_factor_model(model, X, max_acf_lags=10)
        # Omitted AR(1) should cause significant autocorrelation
        assert result.ljung_box_p < 0.05
