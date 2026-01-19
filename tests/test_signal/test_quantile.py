"""Tests for quantile analysis functions.

Tests quantile.py: compute_quantile_returns, compute_spread, compute_monotonicity.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.signal.quantile import (
    compute_monotonicity,
    compute_quantile_returns,
    compute_spread,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def quantile_data():
    """Data with quantile assignments and forward returns."""
    # 5 quantiles, with returns increasing by quantile (monotonic)
    np.random.seed(42)

    data_rows = []
    for d in [date(2024, 1, i) for i in range(1, 11)]:  # 10 dates
        for q in range(1, 6):  # 5 quantiles
            for asset_idx in range(10):  # 10 assets per quantile
                # Higher quantiles have higher returns on average
                base_return = q * 0.005  # 0.5%, 1.0%, 1.5%, 2.0%, 2.5%
                fwd_return = base_return + np.random.randn() * 0.002
                data_rows.append(
                    {
                        "date": d,
                        "asset": f"A{q}_{asset_idx}",
                        "factor": q + np.random.randn() * 0.1,
                        "quantile": q,
                        "1D_fwd_return": fwd_return,
                    }
                )

    return pl.DataFrame(data_rows)


@pytest.fixture
def nonmonotonic_data():
    """Data where returns don't increase with quantiles (V-shape)."""
    data_rows = []
    # Returns: Q1=high, Q2=low, Q3=high (V-shape)
    returns_by_q = {1: 0.02, 2: 0.005, 3: 0.02}

    for d in [date(2024, 1, i) for i in range(1, 6)]:
        for q in [1, 2, 3]:
            for asset_idx in range(10):
                data_rows.append(
                    {
                        "date": d,
                        "asset": f"A{q}_{asset_idx}",
                        "quantile": q,
                        "1D_fwd_return": returns_by_q[q],
                    }
                )

    return pl.DataFrame(data_rows)


# =============================================================================
# Tests: compute_quantile_returns
# =============================================================================


class TestComputeQuantileReturns:
    """Tests for compute_quantile_returns function."""

    def test_basic(self, quantile_data):
        """Test basic quantile returns computation."""
        q_returns = compute_quantile_returns(quantile_data, period=1, n_quantiles=5)

        assert len(q_returns) == 5
        assert all(q in q_returns for q in range(1, 6))
        # Higher quantiles should have higher returns
        assert q_returns[5] > q_returns[1]

    def test_missing_return_column(self, quantile_data):
        """Test when return column doesn't exist."""
        q_returns = compute_quantile_returns(quantile_data, period=99, n_quantiles=5)

        # All values should be NaN
        assert len(q_returns) == 5
        assert all(np.isnan(q_returns[q]) for q in range(1, 6))

    def test_missing_quantile(self):
        """Test when some quantiles have no data."""
        # Data only for quantiles 1, 3, 5 (missing 2, 4)
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 6,
                "asset": [f"A{i}" for i in range(6)],
                "quantile": [1, 1, 3, 3, 5, 5],
                "1D_fwd_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )

        q_returns = compute_quantile_returns(data, period=1, n_quantiles=5)

        # Missing quantiles should be filled with NaN
        assert len(q_returns) == 5
        assert not np.isnan(q_returns[1])
        assert np.isnan(q_returns[2])  # Missing
        assert not np.isnan(q_returns[3])
        assert np.isnan(q_returns[4])  # Missing
        assert not np.isnan(q_returns[5])

    def test_all_nan_returns(self):
        """Test when all returns are NaN."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5,
                "asset": [f"A{i}" for i in range(5)],
                "quantile": [1, 2, 3, 4, 5],
                "1D_fwd_return": [float("nan")] * 5,
            }
        )

        q_returns = compute_quantile_returns(data, period=1, n_quantiles=5)

        # All quantile returns should be NaN
        assert all(np.isnan(q_returns[q]) for q in range(1, 6))

    def test_single_obs_per_quantile(self):
        """Test when each quantile has single observation."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3,
                "asset": ["A", "B", "C"],
                "quantile": [1, 2, 3],
                "1D_fwd_return": [0.01, 0.02, 0.03],
            }
        )

        q_returns = compute_quantile_returns(data, period=1, n_quantiles=3)

        # Single observation mean = that value
        assert q_returns[1] == 0.01
        assert q_returns[2] == 0.02
        assert q_returns[3] == 0.03


# =============================================================================
# Tests: compute_spread
# =============================================================================


class TestComputeSpread:
    """Tests for compute_spread function."""

    def test_basic(self, quantile_data):
        """Test basic spread computation."""
        spread_stats = compute_spread(quantile_data, period=1, n_quantiles=5)

        assert "spread" in spread_stats
        assert "t_stat" in spread_stats
        assert "p_value" in spread_stats
        # Spread should be positive (Q5 > Q1)
        assert spread_stats["spread"] > 0

    def test_missing_return_column(self, quantile_data):
        """Test when return column doesn't exist."""
        spread_stats = compute_spread(quantile_data, period=99, n_quantiles=5)

        assert np.isnan(spread_stats["spread"])
        assert np.isnan(spread_stats["t_stat"])
        assert np.isnan(spread_stats["p_value"])

    def test_insufficient_top_returns(self):
        """Test when top quantile has insufficient data."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5,
                "asset": [f"A{i}" for i in range(5)],
                "quantile": [1, 1, 1, 1, 5],  # Only 1 in Q5, 4 in Q1
                "1D_fwd_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )

        spread_stats = compute_spread(data, period=1, n_quantiles=5)

        # Insufficient top returns (n < 2)
        assert np.isnan(spread_stats["spread"])

    def test_insufficient_bottom_returns(self):
        """Test when bottom quantile has insufficient data."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5,
                "asset": [f"A{i}" for i in range(5)],
                "quantile": [1, 5, 5, 5, 5],  # Only 1 in Q1, 4 in Q5
                "1D_fwd_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )

        spread_stats = compute_spread(data, period=1, n_quantiles=5)

        # Insufficient bottom returns (n < 2)
        assert np.isnan(spread_stats["spread"])

    def test_zero_spread(self):
        """Test when top and bottom have same average return."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 6,
                "asset": [f"A{i}" for i in range(6)],
                "quantile": [1, 1, 1, 5, 5, 5],
                "1D_fwd_return": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],  # Same avg
            }
        )

        spread_stats = compute_spread(data, period=1, n_quantiles=5)

        # Spread should be zero
        assert abs(spread_stats["spread"]) < 1e-10
        # T-stat should be 0
        assert abs(spread_stats["t_stat"]) < 1e-10

    def test_nan_values_removed(self):
        """Test that NaN values are properly removed before t-test."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 6,
                "asset": [f"A{i}" for i in range(6)],
                "quantile": [1, 1, 1, 5, 5, 5],
                "1D_fwd_return": [
                    0.01,
                    0.02,
                    float("nan"),  # NaN in Q1
                    0.05,
                    0.06,
                    float("nan"),
                ],  # NaN in Q5
            }
        )

        spread_stats = compute_spread(data, period=1, n_quantiles=5)

        # Should still compute with 2 values per quantile after NaN removal
        assert not np.isnan(spread_stats["spread"])
        # Spread = (0.05+0.06)/2 - (0.01+0.02)/2 = 0.055 - 0.015 = 0.04
        assert abs(spread_stats["spread"] - 0.04) < 1e-10


# =============================================================================
# Tests: compute_monotonicity
# =============================================================================


class TestComputeMonotonicity:
    """Tests for compute_monotonicity function."""

    def test_perfectly_monotonic_increasing(self):
        """Test perfectly monotonically increasing returns."""
        q_returns = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}

        mono = compute_monotonicity(q_returns)

        # Perfect positive correlation (allow floating-point tolerance)
        assert abs(mono - 1.0) < 1e-10

    def test_perfectly_monotonic_decreasing(self):
        """Test perfectly monotonically decreasing returns."""
        q_returns = {1: 0.05, 2: 0.04, 3: 0.03, 4: 0.02, 5: 0.01}

        mono = compute_monotonicity(q_returns)

        # Perfect negative correlation (allow floating-point tolerance)
        assert abs(mono - (-1.0)) < 1e-10

    def test_nonmonotonic(self):
        """Test non-monotonic returns (V-shape)."""
        q_returns = {1: 0.02, 2: 0.01, 3: 0.02}  # V-shape

        mono = compute_monotonicity(q_returns)

        # Low monotonicity (close to 0)
        assert -0.5 < mono < 0.5

    def test_fewer_than_3_quantiles(self):
        """Test when fewer than 3 quantiles."""
        q_returns = {1: 0.01, 2: 0.02}

        mono = compute_monotonicity(q_returns)

        # Need at least 3 points for rank correlation
        assert np.isnan(mono)

    def test_all_nan_returns(self):
        """Test when all returns are NaN."""
        q_returns = {1: float("nan"), 2: float("nan"), 3: float("nan")}

        mono = compute_monotonicity(q_returns)

        # Empty after filtering NaN
        assert np.isnan(mono)

    def test_mixed_nan_returns(self):
        """Test when some returns are NaN."""
        q_returns = {
            1: 0.01,
            2: float("nan"),
            3: 0.02,
            4: float("nan"),
            5: 0.03,
        }

        mono = compute_monotonicity(q_returns)

        # 3 valid values remain: 1->0.01, 3->0.02, 5->0.03
        # Should be perfectly monotonic
        assert abs(mono - 1.0) < 1e-10
