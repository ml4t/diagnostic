"""Tests for Numba-optimized utility functions."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.diagnostic.core.numba_utils import (
    _rank_data_numba,
    block_bootstrap_numba,
    calculate_drawdown_numba,
    calculate_ic_vectorized,
    embargo_indices_numba,
    purge_indices_numba,
    rolling_sharpe_numba,
)


class TestDrawdownNumba:
    """Test Numba-optimized drawdown calculation."""

    def test_no_drawdown(self):
        """Test case with no drawdown (monotonic increase)."""
        cum_returns = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        assert max_dd == 0.0
        assert duration == 0

    def test_single_drawdown(self):
        """Test case with single drawdown period."""
        cum_returns = np.array([0.0, 0.1, 0.05, -0.02, 0.08])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        assert max_dd == pytest.approx(-0.12, rel=1e-6)
        assert peak_idx == 1
        assert trough_idx == 3
        assert duration == 2

    def test_multiple_drawdowns(self):
        """Test case with multiple drawdown periods."""
        cum_returns = np.array([0.0, 0.1, 0.05, 0.15, 0.08, 0.02, 0.12])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        assert max_dd == pytest.approx(-0.13, rel=1e-6)
        assert peak_idx == 3
        assert trough_idx == 5

    def test_empty_array(self):
        """Test with empty array."""
        cum_returns = np.array([])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        assert np.isnan(max_dd)
        assert duration == -1

    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_drawdown_properties(self, returns):
        """Property: Drawdown should always be <= 0."""
        cum_returns = np.cumprod(1 + np.array(returns)) - 1
        max_dd, _, _, _ = calculate_drawdown_numba(cum_returns)
        assert max_dd <= 0.0


class TestPurgeIndicesNumba:
    """Test Numba-optimized purge indices calculation."""

    def test_basic_purge(self):
        """Test basic purging calculation."""
        indices = purge_indices_numba(50, 60, 5, 100)
        expected = np.arange(45, 50)
        np.testing.assert_array_equal(indices, expected)

    def test_purge_at_start(self):
        """Test purging when test set is at the start."""
        indices = purge_indices_numba(5, 10, 10, 100)
        expected = np.arange(0, 5)
        np.testing.assert_array_equal(indices, expected)

    def test_no_purge_needed(self):
        """Test when no purging is needed."""
        indices = purge_indices_numba(0, 10, 0, 100)
        assert len(indices) == 0

    def test_large_horizon(self):
        """Test with large label horizon."""
        indices = purge_indices_numba(50, 60, 100, 100)
        expected = np.arange(0, 50)
        np.testing.assert_array_equal(indices, expected)


class TestEmbargoIndicesNumba:
    """Test Numba-optimized embargo indices calculation."""

    def test_basic_embargo(self):
        """Test basic embargo calculation."""
        indices = embargo_indices_numba(50, 10, 100)
        expected = np.arange(50, 60)
        np.testing.assert_array_equal(indices, expected)

    def test_embargo_at_end(self):
        """Test embargo when test set is at the end."""
        indices = embargo_indices_numba(95, 10, 100)
        expected = np.arange(95, 100)
        np.testing.assert_array_equal(indices, expected)

    def test_no_embargo(self):
        """Test when embargo size is 0."""
        indices = embargo_indices_numba(50, 0, 100)
        assert len(indices) == 0

    def test_embargo_beyond_end(self):
        """Test embargo that would go beyond data end."""
        indices = embargo_indices_numba(100, 10, 100)
        assert len(indices) == 0


class TestBlockBootstrapNumba:
    """Test Numba-optimized block bootstrap."""

    def test_basic_bootstrap(self):
        """Test basic block bootstrap sampling."""
        indices = np.arange(100)
        result = block_bootstrap_numba(indices, 80, 10, 42)
        assert len(result) == 80
        assert np.all(np.isin(result, indices))

    def test_sample_length_exceeds_data(self):
        """Test when sample length exceeds data length."""
        indices = np.arange(10)
        result = block_bootstrap_numba(indices, 5, 20, 42)
        assert len(result) == 5
        np.testing.assert_array_equal(result, indices[:5])

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        indices = np.arange(100)
        result1 = block_bootstrap_numba(indices, 50, 5, 42)
        result2 = block_bootstrap_numba(indices, 50, 5, 42)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        indices = np.arange(100)
        result1 = block_bootstrap_numba(indices, 50, 5, 42)
        result2 = block_bootstrap_numba(indices, 50, 5, 43)
        assert not np.array_equal(result1, result2)


class TestRollingSharpeNumba:
    """Test Numba-optimized rolling Sharpe ratio."""

    def test_constant_returns(self):
        """Test with constant returns."""
        returns = np.full(100, 0.01)
        result = rolling_sharpe_numba(returns, 20)
        # Constant returns should have very high Sharpe (std ~0)
        assert np.all(np.isnan(result[:19]))  # First 19 are NaN
        assert result[19] > 100  # Very high Sharpe for constant returns

    def test_zero_returns(self):
        """Test with zero returns."""
        returns = np.zeros(100)
        result = rolling_sharpe_numba(returns, 20)
        assert np.all(np.isnan(result[:19]))
        # Zero returns with zero std should give 0 Sharpe
        assert np.all(result[19:] == 0)

    def test_window_larger_than_data(self):
        """Test when window is larger than data."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        result = rolling_sharpe_numba(returns, 10)
        assert np.all(np.isnan(result))

    @given(
        window=st.integers(min_value=2, max_value=50),
        n_returns=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=20)
    def test_rolling_sharpe_shape(self, window, n_returns):
        """Property: Output shape should match input."""
        returns = np.random.randn(n_returns) * 0.01
        result = rolling_sharpe_numba(returns, window)
        assert len(result) == n_returns


class TestICVectorized:
    """Test Numba-optimized IC calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert ic == pytest.approx(1.0, rel=1e-6)

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert ic == pytest.approx(-1.0, rel=1e-6)

    def test_no_correlation(self):
        """Test no correlation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert abs(ic) <= 0.8  # Should not be perfectly correlated

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        predictions = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        returns = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        # Should only use [1.0, 5.0] and [1.0, 5.0]
        assert ic == pytest.approx(1.0, rel=1e-6)

    def test_spearman_vs_pearson(self):
        """Test Spearman vs Pearson correlation."""
        # Non-linear but monotonic relationship
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # Squared

        pearson_ic = calculate_ic_vectorized(predictions, returns, 0)
        spearman_ic = calculate_ic_vectorized(predictions, returns, 1)

        # Spearman should be perfect for monotonic relationship
        assert spearman_ic == pytest.approx(1.0, rel=1e-6)
        # Pearson should be less than perfect for non-linear
        assert pearson_ic < 1.0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        predictions = np.array([1.0])
        returns = np.array([1.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert np.isnan(ic)

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        predictions = np.array([1.0, 2.0, 3.0])
        returns = np.array([1.0, 2.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert np.isnan(ic)

    def test_constant_predictions(self):
        """Test with constant predictions (zero variance)."""
        predictions = np.array([1.0, 1.0, 1.0, 1.0])
        returns = np.array([1.0, 2.0, 3.0, 4.0])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert ic == 0.0  # Zero variance in predictions => IC = 0

    def test_all_nan(self):
        """Test with all NaN values."""
        predictions = np.array([np.nan, np.nan])
        returns = np.array([np.nan, np.nan])
        ic = calculate_ic_vectorized(predictions, returns, 0)
        assert np.isnan(ic)


class TestRankDataNumba:
    """Test Numba-optimized ranking helper."""

    def test_simple_ranking(self):
        """Test basic ranking."""
        data = np.array([3.0, 1.0, 2.0])
        ranks = _rank_data_numba(data)
        expected = np.array([3.0, 1.0, 2.0])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)

    def test_ties_averaged(self):
        """Test that ties are averaged."""
        data = np.array([1.0, 2.0, 2.0, 4.0])
        ranks = _rank_data_numba(data)
        # Values 2.0 at indices 1,2 should share ranks 2 and 3 => avg = 2.5
        expected = np.array([1.0, 2.5, 2.5, 4.0])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)

    def test_all_same(self):
        """Test all identical values."""
        data = np.array([5.0, 5.0, 5.0])
        ranks = _rank_data_numba(data)
        # All same => average rank = (1+2+3)/3 = 2
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)

    def test_reverse_order(self):
        """Test reverse ordered data."""
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ranks = _rank_data_numba(data)
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)

    def test_single_value(self):
        """Test single value."""
        data = np.array([42.0])
        ranks = _rank_data_numba(data)
        expected = np.array([1.0])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)

    def test_multiple_ties(self):
        """Test multiple groups of ties."""
        data = np.array([1.0, 1.0, 3.0, 3.0])
        ranks = _rank_data_numba(data)
        # First two share ranks 1,2 => 1.5
        # Last two share ranks 3,4 => 3.5
        expected = np.array([1.5, 1.5, 3.5, 3.5])
        np.testing.assert_allclose(ranks, expected, rtol=1e-10)


class TestNumbaEdgeCases:
    """Additional edge case tests for Numba functions."""

    def test_drawdown_single_value(self):
        """Test drawdown with single value."""
        cum_returns = np.array([1.0])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        assert max_dd == 0.0
        assert duration == 0

    def test_drawdown_negative_values(self):
        """Test drawdown with negative cumulative returns."""
        cum_returns = np.array([-0.1, 0.0, -0.2, 0.1, -0.3])
        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)
        # Max drawdown from peak at 0.1 to trough at -0.3
        assert max_dd == pytest.approx(-0.4, rel=1e-6)

    def test_bootstrap_more_samples_than_indices(self):
        """Test bootstrap when requesting more samples than available indices with large block."""
        indices = np.arange(5, dtype=np.int64)
        # Need 15 samples from 5 indices
        result = block_bootstrap_numba(indices, 15, 10, 42)
        assert len(result) == 15

    def test_rolling_sharpe_with_risk_free(self):
        """Test rolling Sharpe with non-zero risk-free rate."""
        np.random.seed(42)
        returns = np.random.randn(50) * 0.01
        result_no_rf = rolling_sharpe_numba(returns, 20, risk_free_rate=0.0)
        result_with_rf = rolling_sharpe_numba(returns, 20, risk_free_rate=0.05)
        # Results should differ
        assert not np.allclose(result_no_rf[19:], result_with_rf[19:], equal_nan=True)

    @given(
        size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20)
    def test_ic_bounds(self, size):
        """Property: IC should always be in [-1, 1]."""
        np.random.seed(42)
        predictions = np.random.randn(size)
        returns = np.random.randn(size)
        ic = calculate_ic_vectorized(predictions, returns, 0)
        if np.isfinite(ic):
            assert -1.0 <= ic <= 1.0
