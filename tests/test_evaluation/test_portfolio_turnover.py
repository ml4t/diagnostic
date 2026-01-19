"""Tests for portfolio turnover calculation."""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.portfolio_analysis import compute_portfolio_turnover


class TestComputePortfolioTurnover:
    """Tests for compute_portfolio_turnover function."""

    def test_basic_output_structure(self):
        """Test that compute_portfolio_turnover returns expected keys."""
        np.random.seed(42)
        # 252 days, 10 assets with valid portfolio weights
        weights = np.random.dirichlet(np.ones(10), size=252)

        result = compute_portfolio_turnover(weights)

        # Check required keys
        assert "turnover_mean" in result
        assert "turnover_median" in result
        assert "turnover_std" in result
        assert "turnover_max" in result
        assert "turnover_total" in result
        assert "n_periods" in result
        assert "is_annualized" in result
        assert "periods_per_year" in result

    def test_zero_turnover(self):
        """Test turnover is zero when weights don't change."""
        # Same weights every day
        weights = np.tile([0.5, 0.3, 0.2], (100, 1))

        result = compute_portfolio_turnover(weights)

        assert result["turnover_mean"] == 0.0
        assert result["turnover_median"] == 0.0
        assert result["turnover_std"] == 0.0
        assert result["turnover_max"] == 0.0
        assert result["turnover_total"] == 0.0

    def test_full_turnover(self):
        """Test turnover calculation with complete portfolio flip."""
        # Day 1: 100% asset A
        # Day 2: 100% asset B
        weights = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        result = compute_portfolio_turnover(weights, annualize=False)

        # Full flip: |1-0| + |0-1| = 2, turnover = 2/2 = 1 (100%)
        assert result["turnover_mean"] == 100.0  # 100%
        assert result["turnover_total"] == 100.0

    def test_partial_turnover(self):
        """Test turnover calculation with partial rebalance."""
        weights = np.array(
            [
                [0.6, 0.4],
                [0.5, 0.5],
            ]
        )

        result = compute_portfolio_turnover(weights, annualize=False)

        # Change: |0.6-0.5| + |0.4-0.5| = 0.2, turnover = 0.2/2 = 0.1 (10%)
        assert result["turnover_mean"] == pytest.approx(10.0, abs=0.01)

    def test_annualization(self):
        """Test that annualization works correctly."""
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(5), size=252)

        result_annual = compute_portfolio_turnover(weights, annualize=True, periods_per_year=252)
        result_raw = compute_portfolio_turnover(weights, annualize=False)

        # Annualized mean should be ~252x the raw mean (in percentage terms)
        assert result_annual["is_annualized"] is True
        assert result_raw["is_annualized"] is False
        assert result_annual["turnover_mean"] == pytest.approx(
            result_raw["turnover_mean"] * 252, rel=0.01
        )

    def test_different_periods_per_year(self):
        """Test with different periods_per_year settings."""
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(5), size=52)  # Weekly for 1 year

        result = compute_portfolio_turnover(weights, annualize=True, periods_per_year=52)

        assert result["periods_per_year"] == 52
        assert result["n_periods"] == 52

    def test_n_periods_count(self):
        """Test that n_periods is correctly reported."""
        weights = np.random.dirichlet(np.ones(5), size=100)

        result = compute_portfolio_turnover(weights)

        assert result["n_periods"] == 100

    def test_invalid_1d_input(self):
        """Test that 1D input raises ValueError."""
        weights = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError, match="2D array"):
            compute_portfolio_turnover(weights)

    def test_insufficient_periods(self):
        """Test that single period raises ValueError."""
        weights = np.array([[0.5, 0.3, 0.2]])

        with pytest.raises(ValueError, match="at least 2 periods"):
            compute_portfolio_turnover(weights)

    def test_list_input(self):
        """Test that list input is converted to array."""
        weights = [
            [0.5, 0.5],
            [0.4, 0.6],
            [0.5, 0.5],
        ]

        result = compute_portfolio_turnover(weights, annualize=False)

        # Should work without error
        assert result["n_periods"] == 3
        assert result["turnover_total"] > 0

    def test_single_asset(self):
        """Test turnover with single asset (always 100% weight)."""
        weights = np.ones((100, 1))

        result = compute_portfolio_turnover(weights)

        # No turnover possible with single asset at 100%
        assert result["turnover_total"] == 0.0

    def test_many_assets(self):
        """Test turnover with many assets."""
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(100), size=252)

        result = compute_portfolio_turnover(weights)

        # Should compute without error
        assert result["n_periods"] == 252
        assert result["turnover_total"] > 0

    def test_turnover_statistics(self):
        """Test that turnover statistics are consistent."""
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(5), size=100)

        result = compute_portfolio_turnover(weights, annualize=False)

        # Max should be >= mean
        assert result["turnover_max"] >= result["turnover_mean"]
        # Std should be non-negative
        assert result["turnover_std"] >= 0

    def test_daily_rebalance_realistic(self):
        """Test with realistic daily rebalancing scenario."""
        np.random.seed(42)
        n_days = 252
        n_assets = 10

        # Start with equal weights, add small random changes
        weights = np.zeros((n_days, n_assets))
        weights[0] = np.ones(n_assets) / n_assets

        for t in range(1, n_days):
            # Small random changes, then renormalize
            new_weights = weights[t - 1] + 0.02 * np.random.randn(n_assets)
            new_weights = np.maximum(new_weights, 0)  # No negative weights
            weights[t] = new_weights / new_weights.sum()

        result = compute_portfolio_turnover(weights, annualize=True)

        # Realistic turnover should be moderate
        assert result["turnover_mean"] > 0
        assert result["turnover_mean"] < 10000  # Sanity check

    def test_dates_parameter_optional(self):
        """Test that dates parameter is optional."""
        weights = np.random.dirichlet(np.ones(5), size=50)

        # Should work without dates
        result1 = compute_portfolio_turnover(weights, dates=None)

        # Should work with dates
        dates = np.arange(50)
        result2 = compute_portfolio_turnover(weights, dates=dates)

        # Results should be the same (dates don't affect calculation currently)
        assert result1["turnover_total"] == result2["turnover_total"]
