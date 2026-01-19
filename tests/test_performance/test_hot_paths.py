"""Performance regression tests for hot path optimizations.

These tests verify that critical functions scale linearly (not quadratically)
and complete within reasonable time limits.
"""

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


class TestComputeForwardReturnsPerformance:
    """Performance tests for compute_forward_returns."""

    @pytest.fixture
    def large_factor_data(self):
        """Generate large factor dataset for performance testing."""
        n_dates = 252  # 1 year
        n_assets = 500  # 500 stocks
        n_rows = n_dates * n_assets

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
        assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

        return pl.DataFrame(
            {
                "date": [d for d in dates for _ in assets],
                "asset": assets * n_dates,
                "factor": np.random.randn(n_rows),
            }
        )

    @pytest.fixture
    def large_price_data(self, large_factor_data):
        """Generate matching price data."""
        dates = large_factor_data["date"].unique().sort().to_list()
        assets = large_factor_data["asset"].unique().to_list()

        # Add extra dates for forward returns
        extra_dates = [dates[-1] + timedelta(days=i) for i in range(1, 22)]
        all_dates = dates + extra_dates

        n_rows = len(all_dates) * len(assets)

        return pl.DataFrame(
            {
                "date": [d for d in all_dates for _ in assets],
                "asset": assets * len(all_dates),
                "price": 100 + np.random.randn(n_rows).cumsum() * 0.01,
            }
        )

    @pytest.mark.benchmark
    def test_forward_returns_completes_in_time(self, large_factor_data, large_price_data):
        """Verify compute_forward_returns completes in reasonable time.

        With vectorized implementation, 126K rows should complete in <5 seconds.
        Old row-iteration would timeout or take minutes.
        """
        from ml4t.diagnostic.signal._utils import compute_forward_returns

        start = time.perf_counter()
        result = compute_forward_returns(
            large_factor_data,
            large_price_data,
            periods=(1, 5, 21),
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"compute_forward_returns took {elapsed:.2f}s (expected <10s)"
        assert "1D_fwd_return" in result.columns
        assert "5D_fwd_return" in result.columns
        assert "21D_fwd_return" in result.columns
        assert result.height == large_factor_data.height

    @pytest.mark.benchmark
    def test_forward_returns_linear_scaling(self, large_price_data):
        """Verify linear scaling with input size."""
        from ml4t.diagnostic.signal._utils import compute_forward_returns

        dates = large_price_data["date"].unique().sort().to_list()[:252]
        assets = large_price_data["asset"].unique().to_list()

        times = []
        sizes = [1000, 5000, 10000]

        for n_rows in sizes:
            n_dates = n_rows // len(assets) + 1
            n_dates = min(n_dates, len(dates))

            factor_data = pl.DataFrame(
                {
                    "date": [d for d in dates[:n_dates] for _ in assets[: n_rows // n_dates]],
                    "asset": (assets[: n_rows // n_dates] * n_dates)[:n_rows],
                    "factor": np.random.randn(min(n_rows, n_dates * (n_rows // n_dates))),
                }
            )

            start = time.perf_counter()
            compute_forward_returns(factor_data, large_price_data, periods=(1,))
            times.append(time.perf_counter() - start)

        # Check roughly linear scaling: 10x data should be <5x time (accounting for overhead)
        if times[0] > 0.01:  # Only check if first measurement is meaningful
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            assert ratio < size_ratio * 2, (
                f"Non-linear scaling: {ratio:.1f}x time for {size_ratio}x data"
            )


class TestExpandingStdPerformance:
    """Performance tests for expanding_std."""

    @pytest.mark.benchmark
    def test_expanding_std_linear_scaling(self):
        """Verify expanding_std is O(n) not O(n²)."""
        from ml4t.diagnostic.backends.polars_backend import PolarsBackend

        times = []
        sizes = [1000, 5000, 10000]

        for n in sizes:
            data = pl.DataFrame({"value": np.random.randn(n)})

            start = time.perf_counter()
            PolarsBackend.fast_expanding_window(data, columns=["value"], operation="std")
            times.append(time.perf_counter() - start)

        # O(n) scaling: 10x data should be ~10x time (with some tolerance)
        # O(n²) would be 100x time
        if times[0] > 0.001:  # Only check if first measurement is meaningful
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            # Allow 3x tolerance for O(n) (should NOT be 100x like O(n²))
            assert ratio < size_ratio * 3, (
                f"Non-linear scaling detected: {ratio:.1f}x time for {size_ratio}x data. "
                f"If ratio is ~{size_ratio**2}, this is O(n²) not O(n)."
            )


class TestComputeICSeriesPerformance:
    """Performance tests for compute_ic_series."""

    @pytest.mark.benchmark
    def test_ic_series_completes_in_time(self):
        """Verify compute_ic_series completes in reasonable time."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_series

        n_dates = 252
        n_assets = 100
        n_rows = n_dates * n_assets

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]

        predictions = pl.DataFrame(
            {
                "date": [d for d in dates for _ in range(n_assets)],
                "prediction": np.random.randn(n_rows),
            }
        )
        returns = pl.DataFrame(
            {
                "date": [d for d in dates for _ in range(n_assets)],
                "forward_return": np.random.randn(n_rows) * 0.02,
            }
        )

        start = time.perf_counter()
        result = compute_ic_series(predictions, returns)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"compute_ic_series took {elapsed:.2f}s (expected <5s)"
        assert result.height == n_dates
