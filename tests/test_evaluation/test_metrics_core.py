"""Tests for core metrics modules: ic.py, risk_adjusted.py, basic.py, monotonicity.py.

These tests cover the fundamental metric calculations used throughout the library.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

# =============================================================================
# Test Information Coefficient (ic.py)
# =============================================================================


class TestInformationCoefficient:
    """Tests for information_coefficient function."""

    def test_basic_spearman(self):
        """Test basic Spearman correlation."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns, method="spearman")

        # Perfect monotonic relationship
        assert ic == pytest.approx(1.0)

    def test_negative_correlation(self):
        """Test negative correlation."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([0.05, 0.04, 0.03, 0.02, 0.01])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(-1.0)

    def test_pearson_correlation(self):
        """Test Pearson correlation method."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns, method="pearson")

        assert ic == pytest.approx(1.0)

    def test_with_confidence_intervals(self):
        """Test IC with confidence intervals."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        rng = np.random.RandomState(42)
        predictions = rng.randn(100)
        returns = predictions * 0.5 + rng.randn(100) * 0.3

        result = information_coefficient(
            predictions, returns, confidence_intervals=True, alpha=0.05
        )

        assert isinstance(result, dict)
        assert "ic" in result
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert "p_value" in result
        assert result["lower_ci"] < result["ic"] < result["upper_ci"]

    def test_insufficient_data(self):
        """Test with insufficient data (< 2 samples)."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0])
        returns = np.array([0.01])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_insufficient_data_with_ci(self):
        """Test insufficient data returns NaN dict with CI."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0])
        returns = np.array([0.01])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert np.isnan(result["ic"])
        assert np.isnan(result["lower_ci"])

    def test_nan_handling(self):
        """Test NaN values are handled correctly."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        returns = np.array([0.01, 0.02, np.nan, 0.04, 0.05])

        # After removing NaN pairs, should have 3 valid observations
        ic = information_coefficient(predictions, returns)

        assert not np.isnan(ic)

    def test_mismatched_lengths(self):
        """Test error with mismatched input lengths."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0])
        returns = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="same length"):
            information_coefficient(predictions, returns)

    def test_unknown_method(self):
        """Test error with unknown correlation method."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0])
        returns = np.array([0.01, 0.02, 0.03])

        with pytest.raises(ValueError, match="Unknown correlation method"):
            information_coefficient(predictions, returns, method="unknown")

    def test_polars_input(self):
        """Test with Polars Series input."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0)

    def test_pandas_input(self):
        """Test with Pandas Series input."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0)

    def test_small_sample_ci(self):
        """Test CI with small sample (< 4)."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
            information_coefficient,
        )

        predictions = np.array([1.0, 2.0, 3.0])
        returns = np.array([0.01, 0.02, 0.03])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert not np.isnan(result["ic"])
        # CI should be NaN for small samples
        assert np.isnan(result["lower_ci"])


class TestComputeICIR:
    """Tests for compute_ic_ir function."""

    def test_basic_ic_ir(self):
        """Test basic IC-IR calculation."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        # Create IC series
        rng = np.random.RandomState(42)
        ic_values = rng.randn(100) * 0.05 + 0.03  # Mean 0.03, std ~0.05

        ic_series = pl.DataFrame({"ic": ic_values})

        ic_ir = compute_ic_ir(ic_series, annualization_factor=1.0)

        assert isinstance(ic_ir, float)
        assert not np.isnan(ic_ir)
        # Positive mean, positive IC-IR expected
        assert ic_ir > 0

    def test_ic_ir_with_numpy_array(self):
        """Test IC-IR with numpy array input."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        ic_values = np.array([0.05, 0.03, 0.04, 0.02, 0.06])

        ic_ir = compute_ic_ir(ic_values, annualization_factor=1.0)

        assert isinstance(ic_ir, float)

    def test_ic_ir_with_confidence_intervals(self):
        """Test IC-IR with bootstrap confidence intervals."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        rng = np.random.RandomState(42)
        ic_values = rng.randn(50) * 0.03 + 0.02

        result = compute_ic_ir(ic_values, confidence_intervals=True, n_bootstrap=100)

        assert isinstance(result, dict)
        assert "ic_ir" in result
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert "mean_ic" in result
        assert "std_ic" in result

    def test_ic_ir_insufficient_data(self):
        """Test IC-IR with insufficient data."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        ic_values = np.array([0.05])  # Only one value

        ic_ir = compute_ic_ir(ic_values)

        assert np.isnan(ic_ir)

    def test_ic_ir_zero_std(self):
        """Test IC-IR when all IC values are identical."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        ic_values = np.array([0.05, 0.05, 0.05, 0.05])  # Zero std

        ic_ir = compute_ic_ir(ic_values, annualization_factor=1.0)

        assert np.isinf(ic_ir)

    def test_ic_ir_pandas_dataframe(self):
        """Test IC-IR with pandas DataFrame."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        df = pd.DataFrame({"ic": [0.05, 0.03, 0.04, 0.02, 0.06]})

        ic_ir = compute_ic_ir(df, annualization_factor=1.0)

        assert isinstance(ic_ir, float)

    def test_annualization(self):
        """Test annualization factor application."""
        from ml4t.diagnostic.evaluation.metrics.information_coefficient import compute_ic_ir

        ic_values = np.array([0.05, 0.03, 0.04, 0.02, 0.06])

        ic_ir_1 = compute_ic_ir(ic_values, annualization_factor=1.0)
        ic_ir_sqrt252 = compute_ic_ir(ic_values, annualization_factor=np.sqrt(252))

        # Annualization factor is multiplied directly
        # sqrt(252) ~ 15.87, so ic_ir_sqrt252 should be ~15.87x ic_ir_1
        expected_ratio = np.sqrt(252)
        assert ic_ir_sqrt252 == pytest.approx(ic_ir_1 * expected_ratio, rel=0.01)


# =============================================================================
# Test Risk-Adjusted Metrics (risk_adjusted.py)
# =============================================================================


class TestSharpeRatio:
    """Tests for sharpe_ratio function."""

    def test_basic_sharpe(self):
        """Test basic Sharpe ratio calculation."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        # Positive returns
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

        sharpe = sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_sharpe_with_risk_free_rate(self):
        """Test Sharpe with risk-free rate."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        rf_rate = 0.005

        sharpe = sharpe_ratio(returns, risk_free_rate=rf_rate)

        # Sharpe should be lower with positive risk-free rate
        sharpe_no_rf = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe < sharpe_no_rf

    def test_sharpe_annualization(self):
        """Test Sharpe ratio annualization."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])

        sharpe_daily = sharpe_ratio(returns, annualization_factor=None)
        sharpe_ann = sharpe_ratio(returns, annualization_factor=252)

        # Annualized should be sqrt(252) times daily
        assert sharpe_ann == pytest.approx(sharpe_daily * np.sqrt(252), rel=0.01)

    def test_sharpe_with_confidence_intervals(self):
        """Test Sharpe with bootstrap confidence intervals."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        rng = np.random.RandomState(42)
        returns = rng.randn(100) * 0.02 + 0.001

        result = sharpe_ratio(
            returns, confidence_intervals=True, bootstrap_samples=100, random_state=42
        )

        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "lower_ci" in result
        assert "upper_ci" in result

    def test_sharpe_insufficient_data(self):
        """Test Sharpe with insufficient data."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([0.01])

        sharpe = sharpe_ratio(returns)

        assert np.isnan(sharpe)

    def test_sharpe_zero_volatility(self):
        """Test Sharpe when returns have zero volatility."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([0.01, 0.01, 0.01, 0.01])  # Zero std

        sharpe = sharpe_ratio(returns)

        # Positive mean, zero std -> infinity
        assert np.isinf(sharpe) and sharpe > 0

    def test_sharpe_negative_zero_vol(self):
        """Test Sharpe with negative returns and zero volatility."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([-0.01, -0.01, -0.01, -0.01])

        sharpe = sharpe_ratio(returns)

        assert np.isinf(sharpe) and sharpe < 0

    def test_sharpe_polars_input(self):
        """Test Sharpe with Polars Series."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = pl.Series([0.01, 0.02, -0.01, 0.03, 0.01])

        sharpe = sharpe_ratio(returns)

        assert isinstance(sharpe, float)

    def test_sharpe_nan_handling(self):
        """Test Sharpe handles NaN values."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sharpe_ratio

        returns = np.array([0.01, np.nan, 0.02, 0.03, np.nan])

        sharpe = sharpe_ratio(returns)

        assert not np.isnan(sharpe)


class TestSortinoRatio:
    """Tests for sortino_ratio function."""

    def test_basic_sortino(self):
        """Test basic Sortino ratio calculation."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sortino_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])

        sortino = sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_sortino_vs_sharpe(self):
        """Test Sortino is >= Sharpe for typical returns."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import (
            sharpe_ratio,
            sortino_ratio,
        )

        # Returns with positive mean and some downside
        returns = np.array([0.02, 0.03, -0.01, 0.04, -0.02, 0.03, 0.01])

        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)

        # Sortino should be >= Sharpe (only penalizes downside)
        assert sortino >= sharpe

    def test_sortino_no_downside(self):
        """Test Sortino with no downside returns."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sortino_ratio

        returns = np.array([0.01, 0.02, 0.01, 0.03, 0.02])  # All positive

        sortino = sortino_ratio(returns)

        assert np.isinf(sortino)

    def test_sortino_annualization(self):
        """Test Sortino ratio annualization."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sortino_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])

        sortino_daily = sortino_ratio(returns, annualization_factor=None)
        sortino_ann = sortino_ratio(returns, annualization_factor=252)

        assert sortino_ann == pytest.approx(sortino_daily * np.sqrt(252), rel=0.01)

    def test_sortino_insufficient_data(self):
        """Test Sortino with insufficient data."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sortino_ratio

        returns = np.array([0.01])

        sortino = sortino_ratio(returns)

        assert np.isnan(sortino)

    def test_sortino_with_target_return(self):
        """Test Sortino with non-zero target return."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import sortino_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])

        sortino_zero = sortino_ratio(returns, target_return=0.0)
        sortino_high = sortino_ratio(returns, target_return=0.02)

        # Higher target should result in lower Sortino
        assert sortino_high < sortino_zero


class TestMaximumDrawdown:
    """Tests for maximum_drawdown function."""

    def test_basic_drawdown(self):
        """Test basic maximum drawdown calculation."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import maximum_drawdown

        # Returns with clear drawdown
        returns = np.array([0.10, -0.05, -0.10, 0.08, 0.05])

        result = maximum_drawdown(returns)

        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert "max_drawdown_duration" in result
        assert result["max_drawdown"] < 0  # Drawdowns are negative

    def test_drawdown_empty_returns(self):
        """Test drawdown with empty returns."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import maximum_drawdown

        returns = np.array([])

        result = maximum_drawdown(returns)

        assert np.isnan(result["max_drawdown"])

    def test_drawdown_all_positive(self):
        """Test drawdown when all returns positive."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import maximum_drawdown

        returns = np.array([0.01, 0.02, 0.01, 0.03])

        result = maximum_drawdown(returns)

        # No drawdown when always going up
        assert result["max_drawdown"] == 0.0

    def test_drawdown_cumulative_input(self):
        """Test drawdown with cumulative returns input."""
        from ml4t.diagnostic.evaluation.metrics.risk_adjusted import maximum_drawdown

        # Cumulative returns
        cum_returns = np.array([0.10, 0.08, 0.05, 0.12, 0.15])

        result = maximum_drawdown(cum_returns, cumulative=True)

        assert result["max_drawdown"] < 0


# =============================================================================
# Test Basic Metrics (basic.py)
# =============================================================================


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""

    def test_basic_forward_returns(self):
        """Test basic forward returns computation."""
        from ml4t.diagnostic.evaluation.metrics.basic import compute_forward_returns

        prices = pl.DataFrame(
            {
                "date": [datetime(2024, 1, i) for i in range(1, 11)],
                "close": [100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 107.0, 108.0, 106.0, 110.0],
            }
        )

        result = compute_forward_returns(prices, periods=[1, 2])

        assert "fwd_ret_1" in result.columns
        assert "fwd_ret_2" in result.columns
        # Forward return 1-day: (101-100)/100 = 0.01 for first row
        assert result["fwd_ret_1"][0] == pytest.approx(0.01, rel=0.01)

    def test_forward_returns_pandas(self):
        """Test forward returns with pandas input."""
        from ml4t.diagnostic.evaluation.metrics.basic import compute_forward_returns

        prices = pd.DataFrame(
            {
                "date": [datetime(2024, 1, i) for i in range(1, 6)],
                "close": [100.0, 101.0, 103.0, 102.0, 105.0],
            }
        )

        result = compute_forward_returns(prices, periods=[1])

        assert "fwd_ret_1" in result.columns
        assert isinstance(result, pd.DataFrame)

    def test_forward_returns_with_grouping(self):
        """Test forward returns with group column."""
        from ml4t.diagnostic.evaluation.metrics.basic import compute_forward_returns

        prices = pl.DataFrame(
            {
                "date": [datetime(2024, 1, i) for i in range(1, 6)] * 2,
                "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
                "close": [100.0, 101.0, 103.0, 102.0, 105.0, 50.0, 51.0, 52.0, 51.0, 53.0],
            }
        )

        result = compute_forward_returns(prices, periods=[1], group_col="symbol")

        assert "fwd_ret_1" in result.columns
        # Each group should have independent forward returns


# =============================================================================
# Test Monotonicity (monotonicity.py)
# =============================================================================


class TestMonotonicity:
    """Tests for monotonicity metrics."""

    def test_basic_monotonicity(self):
        """Test basic monotonicity computation."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        # Perfect monotonic relationship: higher features -> higher outcomes
        features = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
        outcomes = features * 0.01 + np.random.randn(100) * 0.001

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        assert result["is_monotonic"] is True
        assert result["direction"] == "increasing"
        assert result["monotonicity_score"] == 1.0

    def test_decreasing_monotonicity(self):
        """Test decreasing monotonicity."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        # Inverse relationship
        features = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
        outcomes = -features * 0.01 + np.random.randn(100) * 0.001

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        assert result["is_monotonic"] is True
        assert result["direction"] == "decreasing"

    def test_non_monotonic(self):
        """Test non-monotonic relationship (U-shape)."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        # U-shaped relationship
        features = np.linspace(-2, 2, 100)
        outcomes = features**2 + np.random.randn(100) * 0.1

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        # U-shape is not monotonic
        assert result["monotonicity_score"] < 1.0

    def test_monotonicity_score(self):
        """Test monotonicity score calculation."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        # Strong but not perfect relationship
        features = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
        outcomes = features * 0.01 + np.random.randn(100) * 0.005

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        assert "monotonicity_score" in result
        assert 0 <= result["monotonicity_score"] <= 1

    def test_insufficient_data(self):
        """Test with insufficient data for quantile analysis."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        features = np.array([1, 2, 3])
        outcomes = np.array([0.01, 0.02, 0.03])

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        assert result["direction"] == "insufficient_data"
        assert np.isnan(result["correlation"])

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics.monotonicity import (
            compute_monotonicity,
        )

        df_features = pl.DataFrame({"signal": np.linspace(0, 1, 100)})
        df_outcomes = pl.DataFrame({"returns": np.linspace(0, 0.1, 100)})

        result = compute_monotonicity(
            df_features, df_outcomes, feature_col="signal", outcome_col="returns", n_quantiles=5
        )

        assert result["is_monotonic"] is True
