"""Tests for evaluation/stats/moments.py.

This module tests the return statistics computation functions used as
building blocks for DSR/PSR calculations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from ml4t.diagnostic.evaluation.stats.moments import (
    compute_autocorrelation,
    compute_kurtosis,
    compute_return_statistics,
    compute_sharpe,
    compute_skewness,
)


class TestComputeReturnStatistics:
    """Tests for compute_return_statistics function."""

    def test_basic_normal_returns(self):
        """Test with standard normal returns."""
        np.random.seed(42)
        returns = np.random.randn(1000)

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 1000
        # Sharpe should be close to mean/std which is near 0 for standard normal
        assert abs(sharpe) < 0.5
        # Skewness should be close to 0 for normal
        assert abs(skewness) < 0.3
        # Kurtosis should be close to 3 (Pearson convention)
        assert 2.5 < kurtosis < 3.5
        # Autocorrelation should be close to 0 for i.i.d.
        assert abs(autocorr) < 0.1

    def test_positive_mean_returns(self):
        """Test with consistently positive returns."""
        np.random.seed(42)
        returns = np.random.randn(500) + 0.5  # Shift mean to 0.5

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 500
        # Sharpe should be positive for positive mean returns
        assert sharpe > 0

    def test_two_observations_minimum(self):
        """Test with minimum 2 observations."""
        returns = np.array([0.01, -0.02])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 2
        assert np.isfinite(sharpe)
        # Autocorrelation should be 0 with only 2 observations
        assert autocorr == 0.0

    def test_three_observations(self):
        """Test with 3 observations (minimum for autocorrelation)."""
        returns = np.array([0.01, 0.02, -0.01])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 3
        assert np.isfinite(sharpe)
        assert np.isfinite(autocorr)

    def test_single_observation_raises(self):
        """Test that single observation raises ValueError."""
        returns = np.array([0.01])

        with pytest.raises(ValueError, match="at least 2 return observations"):
            compute_return_statistics(returns)

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        returns = np.array([])

        with pytest.raises(ValueError, match="at least 2 return observations"):
            compute_return_statistics(returns)

    def test_zero_variance_raises(self):
        """Test that zero variance raises ValueError."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        with pytest.raises(ValueError, match="zero variance"):
            compute_return_statistics(returns)

    def test_nan_values_removed(self):
        """Test that NaN values are properly removed."""
        returns = np.array([0.01, np.nan, -0.02, 0.03, np.nan, -0.01])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 4  # 2 NaN values removed

    def test_all_nan_raises(self):
        """Test that all-NaN array raises ValueError."""
        returns = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(ValueError, match="at least 2 return observations"):
            compute_return_statistics(returns)

    def test_list_input(self):
        """Test that list input works (converted to array)."""
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 5
        assert np.isfinite(sharpe)

    def test_2d_array_flattened(self):
        """Test that 2D array is flattened."""
        returns = np.array([[0.01, -0.02], [0.03, -0.01]])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 4

    def test_autocorrelation_with_nan_corrcoef(self):
        """Test autocorrelation handles NaN from corrcoef gracefully."""
        # Edge case: if first n-1 or last n-1 values are constant
        # corrcoef might return NaN
        returns = np.array([1.0, 1.0, 1.0, 2.0])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        # Should handle gracefully (autocorr might be 0.0 due to NaN handling)
        assert np.isfinite(autocorr) or autocorr == 0.0

    def test_reference_scipy_skewness(self):
        """Test skewness matches scipy.stats.skew."""
        np.random.seed(123)
        returns = np.random.randn(200)

        _, skewness, _, _, _ = compute_return_statistics(returns)
        scipy_skew = scipy_stats.skew(returns)

        # Should be close but formula may differ slightly
        assert abs(skewness - scipy_skew) < 0.1

    def test_reference_scipy_kurtosis(self):
        """Test kurtosis matches scipy.stats.kurtosis (Pearson)."""
        np.random.seed(456)
        returns = np.random.randn(200)

        _, _, kurtosis, _, _ = compute_return_statistics(returns)
        scipy_kurtosis = scipy_stats.kurtosis(returns, fisher=False)

        # Should be close
        assert abs(kurtosis - scipy_kurtosis) < 0.5


class TestComputeSharpe:
    """Tests for compute_sharpe wrapper function."""

    def test_basic(self):
        """Test basic Sharpe computation."""
        np.random.seed(42)
        returns = np.random.randn(100) + 0.5  # Strong positive mean

        sharpe = compute_sharpe(returns)

        assert sharpe > 0
        assert np.isfinite(sharpe)

    def test_negative_sharpe(self):
        """Test negative Sharpe for negative mean returns."""
        np.random.seed(42)
        returns = np.random.randn(100) - 0.3  # Negative mean

        sharpe = compute_sharpe(returns)

        assert sharpe < 0

    def test_consistency_with_main_function(self):
        """Test consistency with compute_return_statistics."""
        np.random.seed(42)
        returns = np.random.randn(100)

        sharpe1 = compute_sharpe(returns)
        sharpe2, _, _, _, _ = compute_return_statistics(returns)

        assert sharpe1 == sharpe2


class TestComputeSkewness:
    """Tests for compute_skewness wrapper function."""

    def test_basic(self):
        """Test basic skewness computation."""
        np.random.seed(42)
        returns = np.random.randn(100)

        skewness = compute_skewness(returns)

        assert np.isfinite(skewness)
        assert abs(skewness) < 1.0  # Normal distribution has skewness ~0

    def test_positive_skew(self):
        """Test detection of positive skewness."""
        # Exponential distribution has positive skew
        np.random.seed(42)
        returns = np.random.exponential(1, 500)

        skewness = compute_skewness(returns)

        assert skewness > 0

    def test_negative_skew(self):
        """Test detection of negative skewness."""
        # Negated exponential has negative skew
        np.random.seed(42)
        returns = -np.random.exponential(1, 500)

        skewness = compute_skewness(returns)

        assert skewness < 0

    def test_consistency_with_main_function(self):
        """Test consistency with compute_return_statistics."""
        np.random.seed(42)
        returns = np.random.randn(100)

        skewness1 = compute_skewness(returns)
        _, skewness2, _, _, _ = compute_return_statistics(returns)

        assert skewness1 == skewness2


class TestComputeKurtosis:
    """Tests for compute_kurtosis wrapper function."""

    def test_excess_kurtosis_default(self):
        """Test default excess kurtosis (Fisher convention)."""
        np.random.seed(42)
        returns = np.random.randn(500)

        kurtosis = compute_kurtosis(returns, excess=True)

        # Normal distribution has excess kurtosis ~0
        assert abs(kurtosis) < 0.5

    def test_pearson_kurtosis(self):
        """Test Pearson kurtosis (normal = 3)."""
        np.random.seed(42)
        returns = np.random.randn(500)

        kurtosis = compute_kurtosis(returns, excess=False)

        # Normal distribution has Pearson kurtosis ~3
        assert 2.5 < kurtosis < 3.5

    def test_excess_vs_pearson_difference(self):
        """Test that excess = Pearson - 3."""
        np.random.seed(42)
        returns = np.random.randn(100)

        excess = compute_kurtosis(returns, excess=True)
        pearson = compute_kurtosis(returns, excess=False)

        assert abs(excess - (pearson - 3)) < 1e-10

    def test_heavy_tails(self):
        """Test detection of heavy tails (excess kurtosis > 0)."""
        # Student's t with low df has heavy tails
        np.random.seed(42)
        returns = np.random.standard_t(3, 500)

        kurtosis = compute_kurtosis(returns, excess=True)

        assert kurtosis > 0  # Heavy tails

    def test_consistency_with_main_function(self):
        """Test consistency with compute_return_statistics."""
        np.random.seed(42)
        returns = np.random.randn(100)

        kurtosis_excess = compute_kurtosis(returns, excess=True)
        _, _, kurtosis_pearson, _, _ = compute_return_statistics(returns)

        # Main function returns Pearson, so excess = pearson - 3
        assert abs(kurtosis_excess - (kurtosis_pearson - 3)) < 1e-10


class TestComputeAutocorrelation:
    """Tests for compute_autocorrelation wrapper function."""

    def test_basic_iid(self):
        """Test autocorrelation for i.i.d. returns."""
        np.random.seed(42)
        returns = np.random.randn(200)

        autocorr = compute_autocorrelation(returns, lag=1)

        # Should be close to 0 for i.i.d.
        assert abs(autocorr) < 0.15

    def test_positive_autocorrelation(self):
        """Test detection of positive autocorrelation."""
        # Create autocorrelated series using AR(1)
        np.random.seed(42)
        phi = 0.7
        n = 200
        returns = np.zeros(n)
        eps = np.random.randn(n)
        returns[0] = eps[0]
        for t in range(1, n):
            returns[t] = phi * returns[t - 1] + eps[t]

        autocorr = compute_autocorrelation(returns, lag=1)

        # Should detect positive autocorrelation
        assert autocorr > 0.4

    def test_lag_not_1_raises(self):
        """Test that lag != 1 raises ValueError."""
        returns = np.random.randn(100)

        with pytest.raises(ValueError, match="Only lag=1"):
            compute_autocorrelation(returns, lag=2)

        with pytest.raises(ValueError, match="Only lag=1"):
            compute_autocorrelation(returns, lag=0)

    def test_consistency_with_main_function(self):
        """Test consistency with compute_return_statistics."""
        np.random.seed(42)
        returns = np.random.randn(100)

        autocorr1 = compute_autocorrelation(returns, lag=1)
        _, _, _, autocorr2, _ = compute_return_statistics(returns)

        assert autocorr1 == autocorr2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_returns(self):
        """Test with very small return values."""
        returns = np.array([1e-10, -1e-10, 2e-10, -2e-10, 1e-10])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert np.isfinite(sharpe)
        assert np.isfinite(skewness)
        assert np.isfinite(kurtosis)

    def test_very_large_returns(self):
        """Test with very large return values."""
        np.random.seed(42)
        returns = np.random.randn(100) * 1e6

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert np.isfinite(sharpe)
        assert np.isfinite(skewness)
        assert np.isfinite(kurtosis)

    def test_mixed_positive_negative(self):
        """Test with alternating positive and negative returns."""
        returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert n == 8
        # Alternating returns should show negative autocorrelation
        assert autocorr < 0

    def test_return_types(self):
        """Test that all return values are correct types."""
        np.random.seed(42)
        returns = np.random.randn(100)

        sharpe, skewness, kurtosis, autocorr, n = compute_return_statistics(returns)

        assert isinstance(sharpe, float)
        assert isinstance(skewness, float)
        assert isinstance(kurtosis, float)
        assert isinstance(autocorr, float)
        assert isinstance(n, int)
