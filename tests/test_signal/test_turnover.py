"""Tests for turnover and autocorrelation functions.

Tests turnover.py: compute_turnover, compute_autocorrelation, estimate_half_life.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.signal.turnover import (
    compute_autocorrelation,
    compute_turnover,
    estimate_half_life,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def turnover_data():
    """Data with stable quantile membership (low turnover)."""
    data_rows = []
    # 10 dates, 20 assets across 5 quantiles
    # Assets stay in same quantiles across dates (stable)
    for d in [date(2024, 1, i) for i in range(1, 11)]:
        for q in range(1, 6):  # 5 quantiles
            for asset_idx in range(4):  # 4 assets per quantile
                data_rows.append(
                    {
                        "date": d,
                        "asset": f"Q{q}A{asset_idx}",
                        "quantile": q,
                        "factor": q + asset_idx * 0.1,
                    }
                )

    return pl.DataFrame(data_rows)


@pytest.fixture
def high_turnover_data():
    """Data with rotating quantile membership (high turnover)."""
    data_rows = []
    # Assets rotate between quantiles each day
    for d_idx, d in enumerate([date(2024, 1, i) for i in range(1, 6)]):
        for a_idx in range(20):
            # Rotate quantile assignment each day
            q = ((a_idx + d_idx) % 5) + 1
            data_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "quantile": q,
                    "factor": float(a_idx),
                }
            )

    return pl.DataFrame(data_rows)


@pytest.fixture
def autocorrelation_data():
    """Data with high factor autocorrelation (persistence)."""
    np.random.seed(42)
    data_rows = []

    # Create persistent factor values for each asset
    n_assets = 30
    n_dates = 15
    dates = [date(2024, 1, i) for i in range(1, n_dates + 1)]

    # Initial factor values
    base_factors = np.random.randn(n_assets)

    for d_idx, d in enumerate(dates):
        # Add small noise to factors each day (high autocorrelation)
        noise = np.random.randn(n_assets) * 0.1
        current_factors = base_factors + noise * d_idx * 0.1

        for a_idx in range(n_assets):
            data_rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "factor": current_factors[a_idx],
                }
            )

    return pl.DataFrame(data_rows)


# =============================================================================
# Tests: compute_turnover
# =============================================================================


class TestComputeTurnover:
    """Tests for compute_turnover function."""

    def test_low_turnover(self, turnover_data):
        """Test turnover with stable quantile membership."""
        turnover = compute_turnover(turnover_data, n_quantiles=5)

        # Assets stay in same quantiles, so turnover should be 0
        assert turnover == 0.0

    def test_high_turnover(self, high_turnover_data):
        """Test turnover with rotating membership."""
        turnover = compute_turnover(high_turnover_data, n_quantiles=5)

        # Assets rotate, so turnover should be high (close to 1.0)
        assert turnover > 0.5

    def test_single_date(self):
        """Test when only one date (insufficient for turnover)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5,
                "asset": ["A", "B", "C", "D", "E"],
                "quantile": [1, 2, 3, 4, 5],
            }
        )

        turnover = compute_turnover(data, n_quantiles=5)

        # Need at least 2 dates
        assert np.isnan(turnover)

    def test_no_asset_overlap(self):
        """Test when assets don't overlap between dates (turnover = 1.0)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3 + [date(2024, 1, 2)] * 3,
                "asset": ["A", "B", "C", "X", "Y", "Z"],  # Different assets each day
                "quantile": [1, 2, 3, 1, 2, 3],
            }
        )

        turnover = compute_turnover(data, n_quantiles=3)

        # No overlap = 100% turnover
        assert turnover == 1.0

    def test_perfect_overlap(self):
        """Test when all assets stay in same quantiles (turnover = 0)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3 + [date(2024, 1, 2)] * 3,
                "asset": ["A", "B", "C", "A", "B", "C"],  # Same assets
                "quantile": [1, 2, 3, 1, 2, 3],  # Same quantiles
            }
        )

        turnover = compute_turnover(data, n_quantiles=3)

        # Perfect overlap = 0% turnover
        assert turnover == 0.0

    def test_empty_quantile_some_dates(self):
        """Test when a quantile is empty for some dates."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3 + [date(2024, 1, 2)] * 2,  # Q3 missing on day 2
                "asset": ["A", "B", "C", "A", "B"],
                "quantile": [1, 2, 3, 1, 2],  # No Q3 on day 2
            }
        )

        turnover = compute_turnover(data, n_quantiles=3)

        # Should still compute for quantiles that exist on both days
        # Q1: A->A = 0% turnover, Q2: B->B = 0% turnover
        # Q3 has no transition to compute
        assert turnover == 0.0


# =============================================================================
# Tests: compute_autocorrelation
# =============================================================================


class TestComputeAutocorrelation:
    """Tests for compute_autocorrelation function."""

    def test_high_autocorrelation(self, autocorrelation_data):
        """Test autocorrelation with persistent factors."""
        autocorr = compute_autocorrelation(autocorrelation_data, lags=[1, 2, 3], min_obs=10)

        assert len(autocorr) == 3
        # High persistence means high lag-1 autocorrelation
        assert autocorr[0] > 0.8  # Lag 1
        # Autocorrelation should decay with lag
        assert autocorr[0] >= autocorr[1] >= autocorr[2]

    def test_insufficient_dates(self):
        """Test when dates < max(lags) + 1."""
        # Only 3 dates, but want lag 5
        # The function checks: len(dates) < max(lags) + 1, i.e., 3 < 6 = True
        # So it returns all NaN for all lags early
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5 + [date(2024, 1, 3)] * 5,
                "asset": ["A", "B", "C", "D", "E"] * 3,
                "factor": list(range(15)),
            }
        )

        autocorr = compute_autocorrelation(data, lags=[1, 2, 3, 4, 5])

        # All should be NaN because len(dates)=3 < max(lags)+1=6
        assert all(np.isnan(ac) for ac in autocorr)

    def test_no_overlapping_assets(self):
        """Test when assets don't overlap between dates."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 3 + [date(2024, 1, 2)] * 3,
                "asset": ["A", "B", "C", "X", "Y", "Z"],  # No overlap
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        autocorr = compute_autocorrelation(data, lags=[1], min_obs=3)

        # No overlapping assets means empty merge
        assert np.isnan(autocorr[0])

    def test_constant_factors(self):
        """Test when factors are constant (undefined correlation)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 10 + [date(2024, 1, 2)] * 10,
                "asset": [f"A{i}" for i in range(10)] * 2,
                "factor": [5.0] * 20,  # All constant
            }
        )

        autocorr = compute_autocorrelation(data, lags=[1])

        # Constant factors means correlation is undefined
        assert np.isnan(autocorr[0])

    def test_min_obs_filtering(self):
        """Test that pairs below min_obs are excluded."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 3,
                "asset": ["A", "B", "C", "D", "E", "A", "B", "C"],  # Only 3 overlap
                "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1],
            }
        )

        autocorr_with_filter = compute_autocorrelation(data, lags=[1], min_obs=5)
        autocorr_no_filter = compute_autocorrelation(data, lags=[1], min_obs=3)

        # With min_obs=5, the 3 overlapping assets should be filtered
        assert np.isnan(autocorr_with_filter[0])
        # With min_obs=3, should compute
        assert not np.isnan(autocorr_no_filter[0])

    def test_single_lag(self):
        """Test autocorrelation with single lag."""
        np.random.seed(42)
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 20 + [date(2024, 1, 2)] * 20,
                "asset": [f"A{i}" for i in range(20)] * 2,
                "factor": list(np.random.randn(20)) + list(np.random.randn(20) + 0.5),
            }
        )

        autocorr = compute_autocorrelation(data, lags=[1])

        assert len(autocorr) == 1
        # Should be a correlation value between -1 and 1
        assert -1 <= autocorr[0] <= 1


# =============================================================================
# Tests: estimate_half_life
# =============================================================================


class TestEstimateHalfLife:
    """Tests for estimate_half_life function."""

    def test_normal_decay(self):
        """Test half-life with normal decay pattern."""
        # Autocorrelation decays: 0.8, 0.6, 0.4, 0.3, 0.2
        # 50% of 0.8 = 0.4, which occurs between lag 2 and 3
        autocorr = [0.8, 0.6, 0.4, 0.3, 0.2]

        half_life = estimate_half_life(autocorr)

        # Should be between 2 and 3 (exact value depends on interpolation)
        assert half_life is not None
        assert 2 < half_life < 4

    def test_never_decays(self):
        """Test when autocorrelation never falls below 50%."""
        # All values above 50% of first value
        autocorr = [0.8, 0.7, 0.6, 0.55, 0.5]  # Never goes below 0.4 (50% of 0.8)

        half_life = estimate_half_life(autocorr)

        # Can't estimate half-life if never decays
        assert half_life is None

    def test_immediate_decay(self):
        """Test when first lag is already below 50%."""
        # Actually this tests when valid_ac[0] <= 0
        autocorr_negative = [-0.5, -0.3, -0.1]

        half_life = estimate_half_life(autocorr_negative)

        # Negative first value means can't estimate
        assert half_life is None

    def test_all_nan(self):
        """Test when all autocorrelations are NaN."""
        autocorr = [float("nan"), float("nan"), float("nan")]

        half_life = estimate_half_life(autocorr)

        # Empty valid list
        assert half_life is None

    def test_single_value(self):
        """Test with single autocorrelation value."""
        autocorr = [0.8]

        half_life = estimate_half_life(autocorr)

        # Need at least 2 values
        assert half_life is None

    def test_rapid_decay_first_lag(self):
        """Test when decay happens at exactly first lag."""
        # First value 0.8, second value 0.2 (below 0.4 threshold)
        autocorr = [0.8, 0.2, 0.1]

        half_life = estimate_half_life(autocorr)

        # Should interpolate between lag 1 and 2
        assert half_life is not None
        assert 1 < half_life < 2

    def test_exact_threshold_match(self):
        """Test when a value exactly equals threshold."""
        # 50% of 1.0 is 0.5, which is exactly lag 2 value
        autocorr = [1.0, 0.7, 0.5, 0.3]

        half_life = estimate_half_life(autocorr)

        # At lag 3 (index 2), value equals threshold
        # But it's detected at index 2 because 0.5 < 0.5 is False
        # The loop finds 0.3 < 0.5 at index 3
        assert half_life is not None
        assert 2 < half_life < 4
