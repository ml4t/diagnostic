"""Tests for Probability of Backtest Overfitting (PBO) function."""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.stats import compute_pbo


class TestComputePBO:
    """Tests for compute_pbo function."""

    def test_basic_output_structure(self):
        """Test that compute_pbo returns PBOResult with expected attributes."""
        # Create synthetic IS/OOS performance data
        # 100 CV combinations, 10 strategies
        np.random.seed(42)
        is_performance = np.random.randn(100, 10)
        oos_performance = np.random.randn(100, 10)

        result = compute_pbo(is_performance, oos_performance)

        # Check required attributes exist
        assert hasattr(result, "pbo")
        assert hasattr(result, "pbo_pct")
        assert hasattr(result, "n_combinations")
        assert hasattr(result, "n_strategies")
        assert hasattr(result, "is_best_rank_oos_median")
        assert hasattr(result, "is_best_rank_oos_mean")
        assert hasattr(result, "degradation_mean")
        assert hasattr(result, "degradation_std")

    def test_pbo_bounds(self):
        """Test that PBO is between 0 and 1."""
        np.random.seed(42)
        is_performance = np.random.randn(50, 5)
        oos_performance = np.random.randn(50, 5)

        result = compute_pbo(is_performance, oos_performance)

        assert 0.0 <= result.pbo <= 1.0
        assert 0.0 <= result.pbo_pct <= 100.0
        assert result.pbo_pct == result.pbo * 100

    def test_perfect_no_overfitting(self):
        """Test PBO with perfectly correlated IS/OOS performance."""
        np.random.seed(42)
        # IS and OOS are identical - no overfitting
        performance = np.random.randn(100, 10)
        is_performance = performance.copy()
        oos_performance = performance.copy()

        result = compute_pbo(is_performance, oos_performance)

        # With identical IS/OOS, the IS-best should also be OOS-best
        # PBO should be low (close to 0)
        assert result.pbo < 0.2
        # Degradation should be near zero
        assert abs(result.degradation_mean) < 0.01

    def test_high_overfitting(self):
        """Test PBO with anti-correlated IS/OOS (simulating overfitting)."""
        np.random.seed(42)
        n_combinations = 100
        n_strategies = 10

        # IS and OOS are anti-correlated
        is_performance = np.random.randn(n_combinations, n_strategies)
        oos_performance = -is_performance + 0.1 * np.random.randn(n_combinations, n_strategies)

        result = compute_pbo(is_performance, oos_performance)

        # With anti-correlation, IS-best is often OOS-worst
        # PBO should be high (close to 1)
        assert result.pbo > 0.5
        # Degradation should be positive (IS better than OOS)
        assert result.degradation_mean > 0

    def test_n_combinations_and_strategies(self):
        """Test that n_combinations and n_strategies are correctly reported."""
        np.random.seed(42)
        n_combinations = 75
        n_strategies = 8

        is_performance = np.random.randn(n_combinations, n_strategies)
        oos_performance = np.random.randn(n_combinations, n_strategies)

        result = compute_pbo(is_performance, oos_performance)

        assert result.n_combinations == n_combinations
        assert result.n_strategies == n_strategies

    def test_rank_statistics(self):
        """Test that rank statistics are reasonable."""
        np.random.seed(42)
        n_strategies = 10
        is_performance = np.random.randn(100, n_strategies)
        oos_performance = np.random.randn(100, n_strategies)

        result = compute_pbo(is_performance, oos_performance)

        # Ranks should be between 1 and n_strategies
        assert 1 <= result.is_best_rank_oos_median <= n_strategies
        assert 1 <= result.is_best_rank_oos_mean <= n_strategies

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        is_performance = np.random.randn(100, 10)
        oos_performance = np.random.randn(100, 5)  # Different number of strategies

        with pytest.raises(ValueError, match="same shape"):
            compute_pbo(is_performance, oos_performance)

    def test_insufficient_strategies_error(self):
        """Test that single strategy raises ValueError."""
        is_performance = np.random.randn(100, 1)
        oos_performance = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="at least 2 strategies"):
            compute_pbo(is_performance, oos_performance)

    def test_1d_input_handling(self):
        """Test that 1D input is handled (pre-computed combinations)."""
        np.random.seed(42)
        # 1D input - treated as single combination with multiple strategies
        is_performance = np.random.randn(10)
        oos_performance = np.random.randn(10)

        result = compute_pbo(is_performance, oos_performance)

        assert result.n_combinations == 1
        assert result.n_strategies == 10

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        is_performance = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        oos_performance = np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])

        result1 = compute_pbo(is_performance, oos_performance)
        result2 = compute_pbo(is_performance, oos_performance)

        assert result1 == result2

    def test_all_same_strategy_best(self):
        """Test when same strategy is best in all combinations."""
        n_combinations = 50
        n_strategies = 5

        # Strategy 0 is always best in both IS and OOS
        is_performance = np.random.randn(n_combinations, n_strategies)
        is_performance[:, 0] = 10.0  # Strategy 0 always best IS

        oos_performance = np.random.randn(n_combinations, n_strategies)
        oos_performance[:, 0] = 10.0  # Strategy 0 always best OOS

        result = compute_pbo(is_performance, oos_performance)

        # PBO should be 0 (IS-best is always OOS-best = rank 1)
        assert result.pbo == 0.0
        assert result.is_best_rank_oos_mean == 1.0
        assert result.is_best_rank_oos_median == 1.0
