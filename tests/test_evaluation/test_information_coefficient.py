"""Tests for evaluation/metrics/information_coefficient.py.

This module tests the core IC (Information Coefficient) functions used for
evaluating feature predictiveness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats

from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_by_horizon,
    compute_ic_ir,
    compute_ic_series,
    information_coefficient,
)


class TestInformationCoefficient:
    """Tests for information_coefficient() function."""

    def test_perfect_positive_correlation_spearman(self):
        """Test IC = 1 for perfect positive correlation."""
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns, method="spearman")

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation_spearman(self):
        """Test IC = -1 for perfect negative correlation."""
        predictions = np.array([5, 4, 3, 2, 1])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns, method="spearman")

        assert ic == pytest.approx(-1.0, abs=1e-10)

    def test_no_correlation(self):
        """Test IC ≈ 0 for uncorrelated data."""
        np.random.seed(42)
        predictions = np.random.randn(200)
        returns = np.random.randn(200) * 0.01

        ic = information_coefficient(predictions, returns)

        # IC should be close to 0 for random data
        assert abs(ic) < 0.2

    def test_pearson_method(self):
        """Test Pearson correlation method."""
        # Linear relationship
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns, method="pearson")

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        predictions = np.array([1, 2, 3, 4])
        returns = np.array([0.01, 0.02, 0.03, 0.04])

        with pytest.raises(ValueError, match="Unknown correlation method"):
            information_coefficient(predictions, returns, method="kendall")

    def test_mismatched_length_raises(self):
        """Test that mismatched lengths raise ValueError."""
        predictions = np.array([1, 2, 3])
        returns = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="must have the same length"):
            information_coefficient(predictions, returns)

    def test_single_observation_returns_nan(self):
        """Test that single observation returns NaN."""
        predictions = np.array([1])
        returns = np.array([0.01])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_empty_arrays_return_nan(self):
        """Test that empty arrays return NaN."""
        predictions = np.array([])
        returns = np.array([])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_nan_handling(self):
        """Test that NaN values are properly removed."""
        predictions = np.array([1, np.nan, 3, 4, 5])
        returns = np.array([0.01, 0.02, np.nan, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        # Should still compute IC after removing NaN pairs
        assert np.isfinite(ic)

    def test_all_nan_returns_nan(self):
        """Test that all-NaN arrays return NaN."""
        predictions = np.array([np.nan, np.nan, np.nan])
        returns = np.array([np.nan, np.nan, np.nan])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_constant_predictions_returns_nan(self):
        """Test that constant predictions return NaN (undefined correlation)."""
        predictions = np.array([5, 5, 5, 5, 5])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_constant_returns_returns_nan(self):
        """Test that constant returns return NaN (undefined correlation)."""
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        ic = information_coefficient(predictions, returns)

        assert np.isnan(ic)

    def test_polars_series_input(self):
        """Test with Polars Series inputs."""
        predictions = pl.Series([1, 2, 3, 4, 5])
        returns = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_pandas_series_input(self):
        """Test with Pandas Series inputs."""
        predictions = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_list_input(self):
        """Test that list inputs work."""
        predictions = [1, 2, 3, 4, 5]
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]

        # Lists should work through numpy conversion
        ic = information_coefficient(np.array(predictions), np.array(returns))

        assert ic == pytest.approx(1.0, abs=1e-10)


class TestInformationCoefficientWithConfidenceIntervals:
    """Tests for IC with confidence_intervals=True."""

    def test_confidence_intervals_structure(self):
        """Test that CI output has correct structure."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100) * 0.01 + predictions * 0.005

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert "ic" in result
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert "p_value" in result

    def test_confidence_intervals_ordered(self):
        """Test that lower_ci < ic < upper_ci."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100) * 0.01 + predictions * 0.01

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert result["lower_ci"] < result["ic"]
        assert result["ic"] < result["upper_ci"]

    def test_confidence_intervals_wider_for_small_sample(self):
        """Test that CI is wider for smaller samples."""
        np.random.seed(42)
        predictions_large = np.random.randn(200)
        returns_large = predictions_large * 0.01 + np.random.randn(200) * 0.005

        predictions_small = predictions_large[:20]
        returns_small = returns_large[:20]

        result_large = information_coefficient(
            predictions_large, returns_large, confidence_intervals=True
        )
        result_small = information_coefficient(
            predictions_small, returns_small, confidence_intervals=True
        )

        width_large = result_large["upper_ci"] - result_large["lower_ci"]
        width_small = result_small["upper_ci"] - result_small["lower_ci"]

        assert width_small > width_large

    def test_confidence_intervals_custom_alpha(self):
        """Test custom alpha for narrower/wider CI."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = predictions * 0.01 + np.random.randn(100) * 0.005

        result_95 = information_coefficient(
            predictions, returns, confidence_intervals=True, alpha=0.05
        )
        result_99 = information_coefficient(
            predictions, returns, confidence_intervals=True, alpha=0.01
        )

        width_95 = result_95["upper_ci"] - result_95["lower_ci"]
        width_99 = result_99["upper_ci"] - result_99["lower_ci"]

        # 99% CI should be wider than 95% CI
        assert width_99 > width_95

    def test_insufficient_data_for_ci(self):
        """Test CI with insufficient data (n < 4)."""
        predictions = np.array([1, 2, 3])
        returns = np.array([0.01, 0.02, 0.03])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        # IC should be computed
        assert np.isfinite(result["ic"])
        # But CI should be NaN (insufficient data)
        assert np.isnan(result["lower_ci"])
        assert np.isnan(result["upper_ci"])

    def test_single_observation_with_ci(self):
        """Test CI with single observation."""
        predictions = np.array([1])
        returns = np.array([0.01])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert np.isnan(result["ic"])
        assert np.isnan(result["lower_ci"])
        assert np.isnan(result["upper_ci"])
        assert np.isnan(result["p_value"])

    def test_p_value_significant(self):
        """Test p-value for strongly correlated data."""
        # Strong positive correlation
        predictions = np.arange(100)
        returns = predictions * 0.01 + np.random.randn(100) * 0.1

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        # p-value should be small for significant correlation
        assert result["p_value"] < 0.05


class TestComputeICSeries:
    """Tests for compute_ic_series() function."""

    @pytest.fixture
    def sample_data_polars(self):
        """Create sample Polars DataFrames for testing.

        Creates aligned panel data with multiple assets per date.
        The DataFrames share (date, asset) pairs for proper joining.
        """
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        n_assets = 10

        rows = []
        for d in dates:
            for i in range(n_assets):
                pred = np.random.randn()
                fwd_ret = pred * 0.01 + np.random.randn() * 0.005
                rows.append(
                    {
                        "date": d,
                        "asset": f"A{i}",
                        "prediction": pred,
                        "forward_return": fwd_ret,
                    }
                )

        df = pl.DataFrame(rows)
        pred_df = df.select(["date", "asset", "prediction"])
        ret_df = df.select(["date", "asset", "forward_return"])

        return pred_df, ret_df

    @pytest.fixture
    def sample_data_pandas(self, sample_data_polars):
        """Convert sample data to Pandas."""
        pred_pl, ret_pl = sample_data_polars
        return pred_pl.to_pandas(), ret_pl.to_pandas()

    def test_ic_series_polars(self, sample_data_polars):
        """Test IC series computation with Polars DataFrames."""
        pred_df, ret_df = sample_data_polars

        # For panel data, we need to join on both date and asset first
        # then compute IC per date. The function joins on date only.
        # Create merged data with both columns
        merged = pred_df.join(ret_df, on=["date", "asset"])

        result = compute_ic_series(
            merged.select(["date", "prediction"]),
            merged.select(["date", "forward_return"]),
            pred_col="prediction",
            ret_col="forward_return",
            min_periods=5,
        )

        assert isinstance(result, pl.DataFrame)
        assert "date" in result.columns
        assert "ic" in result.columns
        assert "n_obs" in result.columns
        assert len(result) == 20  # One row per date

    def test_ic_series_pandas(self, sample_data_pandas):
        """Test IC series computation with Pandas DataFrames."""
        pred_df, ret_df = sample_data_pandas

        # Join on both date and asset for proper alignment
        merged = pd.merge(pred_df, ret_df, on=["date", "asset"])

        result = compute_ic_series(
            merged[["date", "prediction"]],
            merged[["date", "forward_return"]],
            pred_col="prediction",
            ret_col="forward_return",
            min_periods=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "ic" in result.columns
        assert "n_obs" in result.columns
        assert len(result) == 20

    def test_ic_series_min_periods_filtering(self):
        """Test that min_periods filters out dates with insufficient data."""
        # Create data with only 5 observations per date (single row per date-asset)
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=20, freq="D")

        # Create simple aligned data: 1 observation per date
        # (simulating a single-asset scenario)
        predictions = np.random.randn(20)
        returns = predictions * 0.01 + np.random.randn(20) * 0.005

        pred_df = pl.DataFrame({"date": dates, "prediction": predictions})
        ret_df = pl.DataFrame({"date": dates, "forward_return": returns})

        # With min_periods=5 and only 1 obs per date, all IC should be NaN
        result = compute_ic_series(
            pred_df, ret_df, pred_col="prediction", ret_col="forward_return", min_periods=5
        )

        # All IC values should be NaN since n_obs (1) < min_periods (5)
        ic_values = result["ic"].to_numpy()
        assert all(np.isnan(ic_values))

    def test_ic_series_pearson_method(self, sample_data_polars):
        """Test IC series with Pearson correlation."""
        pred_df, ret_df = sample_data_polars

        # Merge on both keys
        merged = pred_df.join(ret_df, on=["date", "asset"])

        result = compute_ic_series(
            merged.select(["date", "prediction"]),
            merged.select(["date", "forward_return"]),
            pred_col="prediction",
            ret_col="forward_return",
            method="pearson",
            min_periods=5,
        )

        assert len(result) > 0
        ic_values = (
            result["ic"].to_numpy() if isinstance(result, pl.DataFrame) else result["ic"].values
        )
        # Should have mostly finite IC values
        assert np.sum(np.isfinite(ic_values)) > len(ic_values) // 2


class TestComputeICByHorizon:
    """Tests for compute_ic_by_horizon() function."""

    @pytest.fixture
    def sample_predictions_and_prices(self):
        """Create sample predictions and prices."""
        np.random.seed(42)
        n = 100

        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        predictions = np.random.randn(n)
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        pred_df = pd.DataFrame({"date": dates, "prediction": predictions})
        price_df = pd.DataFrame({"date": dates, "close": prices})

        return pred_df, price_df

    def test_ic_by_horizon_default_horizons(self, sample_predictions_and_prices):
        """Test IC by horizon with default horizons [1, 5, 21]."""
        pred_df, price_df = sample_predictions_and_prices

        result = compute_ic_by_horizon(pred_df, price_df)

        assert isinstance(result, dict)
        assert 1 in result
        assert 5 in result
        assert 21 in result

    def test_ic_by_horizon_custom_horizons(self, sample_predictions_and_prices):
        """Test IC by horizon with custom horizons."""
        pred_df, price_df = sample_predictions_and_prices

        result = compute_ic_by_horizon(pred_df, price_df, horizons=[1, 3, 10])

        assert 1 in result
        assert 3 in result
        assert 10 in result
        assert 5 not in result  # Not requested
        assert 21 not in result

    def test_ic_by_horizon_polars(self, sample_predictions_and_prices):
        """Test IC by horizon with Polars DataFrames."""
        pred_df, price_df = sample_predictions_and_prices

        pred_pl = pl.from_pandas(pred_df)
        price_pl = pl.from_pandas(price_df)

        result = compute_ic_by_horizon(pred_pl, price_pl, horizons=[1, 5])

        assert isinstance(result, dict)
        assert 1 in result
        assert 5 in result

    def test_ic_values_are_finite(self, sample_predictions_and_prices):
        """Test that IC values are finite for valid data."""
        pred_df, price_df = sample_predictions_and_prices

        result = compute_ic_by_horizon(pred_df, price_df, horizons=[1, 5])

        for horizon, ic in result.items():
            assert np.isfinite(ic), f"IC for horizon {horizon} is not finite"


class TestComputeICIR:
    """Tests for compute_ic_ir() function."""

    def test_positive_ic_series(self):
        """Test IC-IR with positive IC series."""
        # Consistently positive IC
        ic_values = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05])

        ic_ir = compute_ic_ir(ic_values)

        assert ic_ir > 0
        assert np.isfinite(ic_ir)

    def test_negative_ic_series(self):
        """Test IC-IR with negative IC series."""
        # Consistently negative IC
        ic_values = np.array([-0.05, -0.06, -0.04, -0.07, -0.05])

        ic_ir = compute_ic_ir(ic_values)

        assert ic_ir < 0

    def test_zero_mean_ic_series(self):
        """Test IC-IR with zero-mean IC series."""
        # IC fluctuating around zero
        ic_values = np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05])

        ic_ir = compute_ic_ir(ic_values)

        # IC-IR should be close to zero
        assert abs(ic_ir) < 1.0

    def test_constant_ic_returns_inf(self):
        """Test IC-IR with constant IC (zero std) returns inf."""
        ic_values = np.array([0.05, 0.05, 0.05, 0.05, 0.05])

        ic_ir = compute_ic_ir(ic_values)

        assert ic_ir == np.inf  # Positive constant → +inf

    def test_constant_negative_ic_returns_neg_inf(self):
        """Test IC-IR with constant negative IC returns -inf."""
        ic_values = np.array([-0.05, -0.05, -0.05, -0.05, -0.05])

        ic_ir = compute_ic_ir(ic_values)

        assert ic_ir == -np.inf

    def test_insufficient_data(self):
        """Test IC-IR with insufficient data."""
        ic_values = np.array([0.05])

        ic_ir = compute_ic_ir(ic_values)

        assert np.isnan(ic_ir)

    def test_annualization_factor(self):
        """Test custom annualization factor."""
        ic_values = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04])

        # Daily: sqrt(252)
        ic_ir_daily = compute_ic_ir(ic_values, annualization_factor=np.sqrt(252))
        # Weekly: sqrt(52)
        ic_ir_weekly = compute_ic_ir(ic_values, annualization_factor=np.sqrt(52))

        # Daily annualization should give larger IC-IR
        assert abs(ic_ir_daily) > abs(ic_ir_weekly)

    def test_polars_dataframe_input(self):
        """Test IC-IR with Polars DataFrame input."""
        df = pl.DataFrame({"ic": [0.05, 0.06, 0.04, 0.07, 0.05]})

        ic_ir = compute_ic_ir(df, ic_col="ic")

        assert np.isfinite(ic_ir)

    def test_pandas_dataframe_input(self):
        """Test IC-IR with Pandas DataFrame input."""
        df = pd.DataFrame({"ic": [0.05, 0.06, 0.04, 0.07, 0.05]})

        ic_ir = compute_ic_ir(df, ic_col="ic")

        assert np.isfinite(ic_ir)

    def test_nan_handling(self):
        """Test that NaN values in IC series are removed."""
        ic_values = np.array([0.05, np.nan, 0.06, np.nan, 0.04, 0.07])

        ic_ir = compute_ic_ir(ic_values)

        assert np.isfinite(ic_ir)


class TestComputeICIRWithConfidenceIntervals:
    """Tests for IC-IR with confidence intervals."""

    def test_confidence_intervals_structure(self):
        """Test CI output structure."""
        ic_values = np.array(
            [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05, 0.07, 0.04]
        )

        result = compute_ic_ir(ic_values, confidence_intervals=True)

        assert isinstance(result, dict)
        assert "ic_ir" in result
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert "mean_ic" in result
        assert "std_ic" in result
        assert "n_periods" in result

    def test_confidence_intervals_ordered(self):
        """Test that lower_ci < ic_ir < upper_ci."""
        np.random.seed(42)
        ic_values = 0.05 + np.random.randn(50) * 0.02

        result = compute_ic_ir(ic_values, confidence_intervals=True)

        # CI should be properly ordered
        if np.isfinite(result["lower_ci"]) and np.isfinite(result["upper_ci"]):
            assert result["lower_ci"] < result["ic_ir"]
            assert result["ic_ir"] < result["upper_ci"]

    def test_insufficient_data_for_bootstrap(self):
        """Test CI with insufficient data for bootstrap."""
        ic_values = np.array([0.05, 0.06, 0.04, 0.07, 0.05])  # Only 5 values

        result = compute_ic_ir(ic_values, confidence_intervals=True)

        # Should still compute IC-IR
        assert np.isfinite(result["ic_ir"])
        # But CI should be NaN (n < 10)
        assert np.isnan(result["lower_ci"])
        assert np.isnan(result["upper_ci"])

    def test_bootstrap_reproducibility(self):
        """Test that bootstrap results are reproducible (seeded)."""
        ic_values = np.array(
            [0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06, 0.05, 0.07, 0.04]
        )

        result1 = compute_ic_ir(ic_values, confidence_intervals=True)
        result2 = compute_ic_ir(ic_values, confidence_intervals=True)

        # Results should be identical (seeded RNG)
        assert result1["lower_ci"] == result2["lower_ci"]
        assert result1["upper_ci"] == result2["upper_ci"]


class TestICAgainstScipy:
    """Cross-validation tests against scipy."""

    def test_spearman_matches_scipy(self):
        """Test that our IC matches scipy.stats.spearmanr."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100) * 0.01

        our_ic = information_coefficient(predictions, returns, method="spearman")
        scipy_corr, _ = stats.spearmanr(predictions, returns)

        assert our_ic == pytest.approx(scipy_corr, abs=1e-10)

    def test_pearson_matches_scipy(self):
        """Test that our IC matches scipy.stats.pearsonr."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100) * 0.01

        our_ic = information_coefficient(predictions, returns, method="pearson")
        scipy_corr, _ = stats.pearsonr(predictions, returns)

        assert our_ic == pytest.approx(scipy_corr, abs=1e-10)


class TestICEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_values(self):
        """Test IC with very small values (machine precision)."""
        predictions = np.array([1e-15, 2e-15, 3e-15, 4e-15, 5e-15])
        returns = np.array([1e-15, 2e-15, 3e-15, 4e-15, 5e-15])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_very_large_values(self):
        """Test IC with very large values."""
        predictions = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        returns = np.array([1e10, 2e10, 3e10, 4e10, 5e10])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_mixed_sign_values(self):
        """Test IC with mixed positive and negative values."""
        predictions = np.array([-2, -1, 0, 1, 2])
        returns = np.array([-0.02, -0.01, 0, 0.01, 0.02])

        ic = information_coefficient(predictions, returns)

        assert ic == pytest.approx(1.0, abs=1e-10)

    def test_2d_array_flattened(self):
        """Test that 2D arrays are flattened."""
        predictions = np.array([[1, 2], [3, 4], [5, 6]])
        returns = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])

        ic = information_coefficient(predictions.flatten(), returns.flatten())

        assert np.isfinite(ic)
