"""Regression tests for factor module bugs found during code review (2026-03-04).

Tests that would have caught:
- Finding #1: RF double-subtraction in attribution
- Finding #2: Validation NaN misalignment
- Finding #4: Utf8 timestamp normalization
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor.data import FactorData


@pytest.fixture
def rf_factor_data() -> tuple[np.ndarray, FactorData]:
    """Synthetic data with non-zero risk-free rate."""
    np.random.seed(42)
    T = 300

    mkt = np.random.normal(0.0004, 0.01, T)
    smb = np.random.normal(0.0001, 0.005, T)
    eps = np.random.normal(0, 0.003, T)
    rf = np.full(T, 0.0002)  # constant 2bp daily RF
    returns = 0.0003 + 1.0 * mkt + 0.3 * smb + eps + rf  # total returns (not excess)

    dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
    factor_df = pl.DataFrame({
        "timestamp": dates,
        "Mkt-RF": mkt,
        "SMB": smb,
        "RF": rf,
    })
    fd = FactorData.from_dataframe(factor_df, rf_column="RF")
    return returns, fd


class TestRFInvariance:
    """Verify risk-free rate is subtracted exactly once in all code paths."""

    def test_attribution_rf_consistency(self, rf_factor_data):
        """Attribution with RF should match attribution on pre-subtracted excess returns."""
        from ml4t.diagnostic.evaluation.factor.attribution import compute_return_attribution

        returns, fd_with_rf = rf_factor_data

        # Path A: pass total returns + factor_data with rf
        attr_with_rf = compute_return_attribution(returns, fd_with_rf, window=63)

        # Path B: manually subtract RF, pass excess returns + factor_data without rf
        rf = fd_with_rf.rf_rate.to_numpy().astype(np.float64)
        excess_returns = returns - rf[: len(returns)]
        fd_no_rf = FactorData(
            returns=fd_with_rf.returns,
            rf_rate=None,
            factor_names=fd_with_rf.factor_names,
            source=fd_with_rf.source,
            frequency=fd_with_rf.frequency,
        )
        attr_no_rf = compute_return_attribution(excess_returns, fd_no_rf, window=63)

        # Both paths should produce identical attribution
        for f in fd_with_rf.factor_names:
            np.testing.assert_allclose(
                attr_with_rf.factor_contributions[f],
                attr_no_rf.factor_contributions[f],
                atol=1e-10,
                err_msg=f"Factor {f} attribution differs with/without RF",
            )
        np.testing.assert_allclose(
            attr_with_rf.alpha_contribution,
            attr_no_rf.alpha_contribution,
            atol=1e-10,
            err_msg="Alpha attribution differs with/without RF",
        )

    def test_static_model_rf_consistency(self, rf_factor_data):
        """Static model with RF should match model on pre-subtracted excess returns."""
        from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model

        returns, fd_with_rf = rf_factor_data

        model_with_rf = compute_factor_model(returns, fd_with_rf)

        rf = fd_with_rf.rf_rate.to_numpy().astype(np.float64)
        excess_returns = returns - rf[: len(returns)]
        fd_no_rf = FactorData(
            returns=fd_with_rf.returns,
            rf_rate=None,
            factor_names=fd_with_rf.factor_names,
            source=fd_with_rf.source,
            frequency=fd_with_rf.frequency,
        )
        model_no_rf = compute_factor_model(excess_returns, fd_no_rf)

        for f in fd_with_rf.factor_names:
            assert abs(model_with_rf.betas[f] - model_no_rf.betas[f]) < 1e-10, (
                f"Beta for {f} differs with/without RF"
            )
        assert abs(model_with_rf.alpha - model_no_rf.alpha) < 1e-10

    def test_rolling_model_rf_consistency(self, rf_factor_data):
        """Rolling model with RF should match model on pre-subtracted excess returns."""
        from ml4t.diagnostic.evaluation.factor.rolling_model import compute_rolling_exposures

        returns, fd_with_rf = rf_factor_data

        roll_with_rf = compute_rolling_exposures(returns, fd_with_rf, window=63)

        rf = fd_with_rf.rf_rate.to_numpy().astype(np.float64)
        excess_returns = returns - rf[: len(returns)]
        fd_no_rf = FactorData(
            returns=fd_with_rf.returns,
            rf_rate=None,
            factor_names=fd_with_rf.factor_names,
            source=fd_with_rf.source,
            frequency=fd_with_rf.frequency,
        )
        roll_no_rf = compute_rolling_exposures(excess_returns, fd_no_rf, window=63)

        for f in fd_with_rf.factor_names:
            np.testing.assert_allclose(
                roll_with_rf.rolling_betas[f],
                roll_no_rf.rolling_betas[f],
                atol=1e-10,
                err_msg=f"Rolling beta for {f} differs with/without RF",
            )


class TestNaNAlignment:
    """Verify NaN handling produces consistent results across all code paths."""

    @pytest.fixture
    def nan_factor_data(self) -> tuple[np.ndarray, FactorData]:
        """Synthetic data with NaN values in factor returns."""
        np.random.seed(123)
        T = 300

        mkt = np.random.normal(0.0004, 0.01, T)
        smb = np.random.normal(0.0001, 0.005, T)
        eps = np.random.normal(0, 0.003, T)
        returns = 0.0002 + 1.0 * mkt + 0.3 * smb + eps

        # Inject NaNs at specific positions
        mkt[10] = np.nan
        mkt[50] = np.nan
        smb[100] = np.nan
        returns[150] = np.nan

        dates = pl.date_range(date(2018, 1, 1), date(2019, 12, 31), eager=True)[:T]
        factor_df = pl.DataFrame({
            "timestamp": dates,
            "Mkt-RF": mkt,
            "SMB": smb,
        })
        fd = FactorData.from_dataframe(factor_df)
        return returns, fd

    def test_validation_with_nans_does_not_crash(self, nan_factor_data):
        """validate_model should not crash when factor data has NaNs."""
        from ml4t.diagnostic.evaluation.factor.analysis import FactorAnalysis

        returns, fd = nan_factor_data
        fa = FactorAnalysis(returns, fd)
        result = fa.validate_model()

        # Should return valid results (not NaN everywhere)
        assert np.isfinite(result.condition_number)
        assert np.isfinite(result.ljung_box_p)
        assert np.isfinite(result.jarque_bera_p)

    def test_validation_uses_aligned_rows(self, nan_factor_data):
        """validate_model should use the same rows as the fitted model."""
        from ml4t.diagnostic.evaluation.factor.analysis import FactorAnalysis

        returns, fd = nan_factor_data
        fa = FactorAnalysis(returns, fd)
        model = fa.static_model()
        validation = fa.validate_model()

        # The number of residuals should match the model's n_obs
        assert len(model.residuals) == model.n_obs
        # Condition number should be computed on the aligned design matrix
        # (not raw data with NaN rows)
        assert validation.condition_number > 0
        assert validation.condition_number < 1e15  # not pathological

    def test_static_model_drops_nan_rows(self, nan_factor_data):
        """Static model should silently drop NaN rows and report correct n_obs."""
        from ml4t.diagnostic.evaluation.factor.static_model import compute_factor_model

        returns, fd = nan_factor_data
        model = compute_factor_model(returns, fd)

        # We injected 4 NaN values across different rows
        # n_obs should be less than 300
        assert model.n_obs < 300
        assert model.n_obs >= 296  # at most 4 rows dropped
        assert len(model.residuals) == model.n_obs


class TestUtf8Timestamps:
    """Verify string timestamps are properly coerced."""

    def test_utf8_timestamps_coerced_to_date(self):
        """FactorData should accept string timestamps and coerce to Date."""
        df = pl.DataFrame({
            "timestamp": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Mkt-RF": [0.01, -0.005, 0.003],
        })
        fd = FactorData.from_dataframe(df)
        assert fd.returns["timestamp"].dtype == pl.Date

    def test_utf8_combine_with_date(self):
        """Combining string-timestamp and Date-timestamp FactorData should work."""
        df1 = pl.DataFrame({
            "timestamp": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "Mkt-RF": [0.01, -0.005, 0.003],
        })
        df2 = pl.DataFrame({
            "timestamp": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "SMB": [0.002, 0.001, -0.001],
        })
        fd1 = FactorData.from_dataframe(df1)
        fd2 = FactorData.from_dataframe(df2)

        combined = FactorData.combine(fd1, fd2)
        assert combined.n_factors == 2
        assert len(combined.returns) == 3
