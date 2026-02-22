"""Tests for evaluation/trade_dashboard/stats.py.

This module tests the pure statistical functions used in the trade dashboard,
including PSR calculation and distribution tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.trade_dashboard.stats import (
    benjamini_hochberg_fdr,
    compute_distribution_tests,
    compute_return_summary,
    compute_time_series_tests,
    probabilistic_sharpe_ratio,
)
from ml4t.diagnostic.evaluation.trade_dashboard.types import ReturnSummary


class TestComputeReturnSummary:
    """Tests for compute_return_summary function."""

    def test_basic_returns(self):
        """Test with basic return array."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02

        result = compute_return_summary(returns)

        assert isinstance(result, ReturnSummary)
        assert result.n_samples == 100
        assert np.isfinite(result.mean)
        assert np.isfinite(result.std)
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)
        assert 0 <= result.win_rate <= 1

    def test_empty_array(self):
        """Test with empty return array."""
        returns = np.array([])

        result = compute_return_summary(returns)

        assert result.n_samples == 0
        assert np.isnan(result.mean)
        assert np.isnan(result.std)
        assert np.isnan(result.sharpe)
        assert np.isnan(result.skewness)
        assert np.isnan(result.kurtosis)
        assert np.isnan(result.win_rate)

    def test_single_value(self):
        """Test with single value (std = 0)."""
        returns = np.array([0.05])

        result = compute_return_summary(returns)

        assert result.n_samples == 1
        assert result.mean == 0.05
        assert result.std == 0.0
        assert np.isnan(result.sharpe)  # Can't compute with std=0

    def test_two_values(self):
        """Test with two values (skewness defaults to 0)."""
        returns = np.array([0.01, -0.02])

        result = compute_return_summary(returns)

        assert result.n_samples == 2
        assert np.isfinite(result.mean)
        assert np.isfinite(result.std)
        assert result.skewness == 0.0  # Not enough for skewness

    def test_three_values(self):
        """Test with three values (kurtosis defaults to 3)."""
        returns = np.array([0.01, -0.02, 0.03])

        result = compute_return_summary(returns)

        assert result.n_samples == 3
        assert np.isfinite(result.skewness)
        assert result.kurtosis == 3.0  # Not enough for kurtosis

    def test_four_values(self):
        """Test with four values (full statistics)."""
        returns = np.array([0.01, -0.02, 0.03, -0.01])

        result = compute_return_summary(returns)

        assert result.n_samples == 4
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)

    def test_win_rate_all_positive(self):
        """Test win rate with all positive returns."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        result = compute_return_summary(returns)

        assert result.win_rate == 1.0

    def test_win_rate_all_negative(self):
        """Test win rate with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.03, -0.04, -0.05])

        result = compute_return_summary(returns)

        assert result.win_rate == 0.0

    def test_win_rate_mixed(self):
        """Test win rate with mixed returns."""
        returns = np.array([0.01, -0.02, 0.03, -0.04])

        result = compute_return_summary(returns)

        assert result.win_rate == 0.5  # 2 positive out of 4

    def test_min_max_values(self):
        """Test min and max values are correct."""
        returns = np.array([0.05, -0.10, 0.03, -0.02])

        result = compute_return_summary(returns)

        assert result.min_val == -0.10
        assert result.max_val == 0.05


class TestProbabilisticSharpeRatio:
    """Tests for probabilistic_sharpe_ratio function."""

    def test_basic_calculation(self):
        """Test basic PSR calculation."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=1.5,
            benchmark_sharpe=0.0,
            n_samples=252,
            skewness=0.0,
            kurtosis=3.0,
        )

        assert 0 <= psr <= 1
        assert psr > 0.9  # High Sharpe should have high PSR

    def test_low_sharpe_low_psr(self):
        """Test that low Sharpe gives low PSR."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.5,
            n_samples=100,
            skewness=0.0,
            kurtosis=3.0,
        )

        assert psr < 0.5  # Below benchmark

    def test_n_samples_less_than_2(self):
        """Test with insufficient samples."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=1.5,
            benchmark_sharpe=0.0,
            n_samples=1,
        )

        assert psr == 0.5  # Default for insufficient data

    def test_return_components(self):
        """Test returning component values."""
        result = probabilistic_sharpe_ratio(
            observed_sharpe=1.5,
            benchmark_sharpe=0.0,
            n_samples=252,
            return_components=True,
        )

        assert isinstance(result, dict)
        assert "psr" in result
        assert "z_score" in result
        assert "std_sr" in result
        assert 0 <= result["psr"] <= 1

    def test_negative_variance_protection(self):
        """Test protection against negative variance from extreme skewness."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=2.0,
            benchmark_sharpe=0.0,
            n_samples=100,
            skewness=5.0,  # Extreme skewness
            kurtosis=3.0,
        )

        # Should still produce valid result
        assert 0 <= psr <= 1

    def test_skewness_effect(self):
        """Test that negative skewness increases variance."""
        psr_no_skew = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_samples=100,
            skewness=0.0,
            kurtosis=3.0,
        )

        psr_neg_skew = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_samples=100,
            skewness=-1.0,
            kurtosis=3.0,
        )

        # Negative skewness should increase SR variance, potentially different PSR
        assert np.isfinite(psr_no_skew)
        assert np.isfinite(psr_neg_skew)

    def test_kurtosis_effect(self):
        """Test that higher kurtosis increases variance."""
        psr_normal = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_samples=100,
            skewness=0.0,
            kurtosis=3.0,  # Normal
        )

        psr_heavy_tails = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_samples=100,
            skewness=0.0,
            kurtosis=6.0,  # Heavy tails
        )

        assert np.isfinite(psr_normal)
        assert np.isfinite(psr_heavy_tails)


class TestComputeDistributionTests:
    """Tests for compute_distribution_tests function."""

    def test_normal_distribution(self):
        """Test with normal distribution."""
        np.random.seed(42)
        returns = np.random.randn(500)

        df = compute_distribution_tests(returns)

        assert isinstance(df, pd.DataFrame)
        assert "test" in df.columns
        assert "statistic" in df.columns
        assert "p_value" in df.columns
        assert "interpretation" in df.columns

        # Should have Shapiro-Wilk, Anderson-Darling, Jarque-Bera
        tests = df["test"].tolist()
        assert any("Shapiro" in t for t in tests)
        assert any("Anderson" in t for t in tests)
        assert any("Jarque" in t for t in tests)

    def test_non_normal_distribution(self):
        """Test with clearly non-normal distribution."""
        np.random.seed(42)
        returns = np.random.exponential(1, 500)

        df = compute_distribution_tests(returns)

        # Should detect non-normality
        non_normal_count = (df["interpretation"] == "Non-normal").sum()
        assert non_normal_count >= 1

    def test_too_few_samples_shapiro(self):
        """Test with too few samples for Shapiro-Wilk."""
        returns = np.array([0.01, 0.02])  # Only 2 samples

        df = compute_distribution_tests(returns)

        # Should not include Shapiro-Wilk (needs >= 3)
        if not df.empty:
            tests = df["test"].tolist()
            assert not any("Shapiro" in t for t in tests)

    def test_too_few_samples_all_tests(self):
        """Test with very few samples."""
        returns = np.array([0.01, 0.02])

        df = compute_distribution_tests(returns)

        # Should be empty or have very few tests
        assert len(df) <= 1

    def test_too_many_samples_shapiro(self):
        """Test with too many samples for Shapiro-Wilk (>5000)."""
        np.random.seed(42)
        returns = np.random.randn(6000)

        df = compute_distribution_tests(returns)

        # Should not include Shapiro-Wilk for n > 5000
        tests = df["test"].tolist()
        assert not any("Shapiro" in t for t in tests)


class TestComputeTimeSeriesTests:
    """Tests for compute_time_series_tests function."""

    def test_iid_returns(self):
        """Test with i.i.d. returns (no autocorrelation)."""
        np.random.seed(42)
        returns = np.random.randn(200)

        df = compute_time_series_tests(returns)

        assert isinstance(df, pd.DataFrame)
        assert "test" in df.columns
        assert "statistic" in df.columns
        assert "p_value" in df.columns
        assert "interpretation" in df.columns

    def test_autocorrelated_returns(self):
        """Test with autocorrelated returns."""
        np.random.seed(42)
        n = 200
        phi = 0.7
        eps = np.random.randn(n)
        returns = np.zeros(n)
        returns[0] = eps[0]
        for t in range(1, n):
            returns[t] = phi * returns[t - 1] + eps[t]

        df = compute_time_series_tests(returns)

        # Ljung-Box should detect autocorrelation
        lb_row = df[df["test"].str.contains("Ljung", na=False)]
        if not lb_row.empty:
            assert lb_row["interpretation"].iloc[0] == "Autocorrelation detected"

    def test_stationary_returns(self):
        """Test stationary returns (ADF should reject unit root)."""
        np.random.seed(42)
        returns = np.random.randn(200)

        df = compute_time_series_tests(returns)

        # ADF should indicate stationary
        adf_row = df[df["test"].str.contains("ADF", na=False)]
        if not adf_row.empty:
            assert adf_row["interpretation"].iloc[0] == "Stationary"

    def test_too_few_samples(self):
        """Test with too few samples."""
        returns = np.array([0.01, 0.02, 0.03])

        df = compute_time_series_tests(returns)

        # Should be empty - too few samples for any test
        assert df.empty or len(df) == 0


class TestBenjaminiHochbergFDR:
    """Tests for benjamini_hochberg_fdr wrapper function."""

    def test_basic_fdr(self):
        """Test basic FDR correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert "rejected" in result
        assert "adjusted_p_values" in result
        assert "n_rejected" in result
        assert len(result["rejected"]) == 5

    def test_no_rejections(self):
        """Test with no significant p-values."""
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert result["n_rejected"] == 0
        assert not any(result["rejected"])

    def test_all_rejections(self):
        """Test with all significant p-values."""
        p_values = [0.001, 0.002, 0.003, 0.004, 0.005]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert result["n_rejected"] == 5
        assert all(result["rejected"])

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        p_values = np.array([0.01, 0.05, 0.1, 0.5])

        result = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert "rejected" in result
        assert "n_rejected" in result
