"""Tests for IC (Information Coefficient) computation.

Tests ic.py: compute_ic_series, compute_ic_summary.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.signal.signal_ic import compute_ic_series, compute_ic_summary

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def prepared_data():
    """Data prepared for IC computation with factor, returns, and quantiles."""
    # Create data where factor and returns are correlated
    # Higher factor -> higher return (positive IC)
    np.random.seed(42)

    data_rows = []
    for d in [date(2024, 1, i) for i in range(1, 21)]:  # 20 dates
        for asset_idx in range(50):  # 50 assets per date
            factor = asset_idx / 50  # 0 to 0.98
            # Return correlates with factor + noise
            fwd_return = factor * 0.01 + np.random.randn() * 0.005
            data_rows.append(
                {
                    "date": d,
                    "asset": f"A{asset_idx}",
                    "factor": factor,
                    "1D_fwd_return": fwd_return,
                    "5D_fwd_return": fwd_return * 5,
                }
            )

    return pl.DataFrame(data_rows)


@pytest.fixture
def uncorrelated_data():
    """Data with no correlation between factor and returns (IC ~ 0)."""
    np.random.seed(123)

    data_rows = []
    for d in [date(2024, 1, i) for i in range(1, 11)]:
        for asset_idx in range(30):
            data_rows.append(
                {
                    "date": d,
                    "asset": f"A{asset_idx}",
                    "factor": np.random.randn(),
                    "1D_fwd_return": np.random.randn() * 0.01,
                }
            )

    return pl.DataFrame(data_rows)


# =============================================================================
# Tests: compute_ic_series
# =============================================================================


class TestComputeICSeries:
    """Tests for compute_ic_series function."""

    def test_spearman_method(self, prepared_data):
        """Test IC with default Spearman method."""
        dates, ic_vals = compute_ic_series(prepared_data, period=1, method="spearman")

        assert len(dates) > 0
        assert len(ic_vals) == len(dates)
        # Factor correlates with returns, so IC should be positive on average
        assert np.mean(ic_vals) > 0

    def test_pearson_method(self, prepared_data):
        """Test IC with Pearson correlation method."""
        dates, ic_vals = compute_ic_series(prepared_data, period=1, method="pearson")

        assert len(dates) > 0
        assert len(ic_vals) == len(dates)
        # Pearson should also be positive for correlated data
        assert np.mean(ic_vals) > 0

    def test_min_obs_filters_dates(self):
        """Test that dates below min_obs threshold are excluded."""
        # Create data with varying asset counts per date
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 20,
                "asset": [f"A{i}" for i in range(5)] + [f"A{i}" for i in range(20)],
                "factor": list(range(5)) + list(range(20)),
                "1D_fwd_return": [i * 0.01 for i in range(5)] + [i * 0.01 for i in range(20)],
            }
        )

        # With min_obs=10, day 1 (5 assets) should be excluded
        dates, ic_vals = compute_ic_series(data, period=1, min_obs=10)

        assert len(dates) == 1  # Only day 2
        assert dates[0] == date(2024, 1, 2)

    def test_all_dates_below_min_obs(self):
        """Test when all dates have insufficient observations."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 5 + [date(2024, 1, 2)] * 5,
                "asset": [f"A{i}" for i in range(5)] + [f"A{i}" for i in range(5)],
                "factor": list(range(5)) + list(range(5)),
                "1D_fwd_return": [i * 0.01 for i in range(5)] + [i * 0.01 for i in range(5)],
            }
        )

        dates, ic_vals = compute_ic_series(data, period=1, min_obs=10)

        assert len(dates) == 0
        assert len(ic_vals) == 0

    def test_nan_in_returns(self, prepared_data):
        """Test that NaN in returns are handled properly."""
        # Add some NaN values
        data = prepared_data.with_columns(
            pl.when(pl.col("asset") == "A0")
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("1D_fwd_return"))
            .alias("1D_fwd_return")
        )

        dates, ic_vals = compute_ic_series(data, period=1)

        # Should still compute IC after removing NaN pairs
        assert len(dates) > 0
        assert all(not np.isnan(ic) for ic in ic_vals)

    def test_identical_factors(self):
        """Test when all factors are identical (undefined correlation)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 20,
                "asset": [f"A{i}" for i in range(20)],
                "factor": [5.0] * 20,  # All identical
                "1D_fwd_return": [i * 0.01 for i in range(20)],
            }
        )

        dates, ic_vals = compute_ic_series(data, period=1)

        # Date should be skipped (correlation undefined)
        assert len(dates) == 0

    def test_identical_returns(self):
        """Test when all returns are identical (undefined correlation)."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)] * 20,
                "asset": [f"A{i}" for i in range(20)],
                "factor": list(range(20)),
                "1D_fwd_return": [0.01] * 20,  # All identical
            }
        )

        dates, ic_vals = compute_ic_series(data, period=1)

        # Date should be skipped (correlation undefined for identical values)
        assert len(dates) == 0

    def test_missing_return_column(self, prepared_data):
        """Test behavior when return column doesn't exist."""
        # The function raises ColumnNotFoundError for missing columns
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            compute_ic_series(prepared_data, period=99)


# =============================================================================
# Tests: compute_ic_summary
# =============================================================================


class TestComputeICSummary:
    """Tests for compute_ic_summary function."""

    def test_empty_series(self):
        """Test summary of empty IC series."""
        summary = compute_ic_summary([])

        assert np.isnan(summary["mean"])
        assert np.isnan(summary["std"])
        assert np.isnan(summary["t_stat"])
        assert np.isnan(summary["p_value"])

    def test_single_value(self):
        """Test summary with single IC value (n < 2)."""
        summary = compute_ic_summary([0.05])

        # n < 2, so all should be NaN
        assert np.isnan(summary["mean"])
        assert np.isnan(summary["std"])
        assert np.isnan(summary["t_stat"])
        assert np.isnan(summary["p_value"])

    def test_zero_std(self):
        """Test summary when all IC values are identical."""
        summary = compute_ic_summary([0.05, 0.05, 0.05, 0.05])

        assert summary["mean"] == 0.05
        assert summary["std"] == 0.0
        assert np.isnan(summary["t_stat"])  # Division by zero
        assert np.isnan(summary["p_value"])

    def test_significant_ic(self):
        """Test summary with statistically significant IC."""
        # Strong positive ICs with low variance
        ic_series = [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.08, 0.09, 0.08, 0.09]
        summary = compute_ic_summary(ic_series)

        assert 0.07 < summary["mean"] < 0.10
        assert summary["std"] > 0
        assert summary["t_stat"] > 0
        assert summary["p_value"] < 0.05  # Should be significant

    def test_nonsignificant_ic(self, uncorrelated_data):
        """Test summary with non-significant IC (random data)."""
        dates, ic_vals = compute_ic_series(uncorrelated_data, period=1)

        if len(ic_vals) >= 2:
            summary = compute_ic_summary(ic_vals)

            # Random data should have IC near zero
            assert -0.2 < summary["mean"] < 0.2

    def test_pct_positive(self):
        """Test pct_positive calculation."""
        ic_series = [0.1, 0.05, -0.03, 0.08, -0.02]  # 3/5 positive
        summary = compute_ic_summary(ic_series)

        assert summary["pct_positive"] == 0.6  # 60%
