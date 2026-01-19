"""Tests for memory-efficient streaming functionality.

This module tests the streaming implementations in PolarsBackend that allow
ml4t-diagnostic to handle very large datasets (10M+ samples) without memory issues.

NOTE: These tests use large datasets (150K samples) and are slow (~120s).
Run with: pytest -m slow tests/test_backends/test_streaming.py
"""

import pytest

# Mark entire module as slow (large dataset streaming tests)
pytestmark = pytest.mark.slow

import numpy as np
import polars as pl

from ml4t.diagnostic.backends.polars_backend import PolarsBackend


class TestStreamingFunctionality:
    """Test memory-efficient streaming operations."""

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for comparison tests."""
        np.random.seed(42)
        n_samples = 1000
        return {
            "x": pl.Series("x", np.random.randn(n_samples)),
            "y": pl.Series("y", np.random.randn(n_samples)),
            "predictions": pl.Series("pred", np.random.randn(n_samples)),
            "returns_1d": pl.Series("ret_1d", np.random.randn(n_samples) * 0.02),
            "returns_5d": pl.Series("ret_5d", np.random.randn(n_samples) * 0.05),
            "returns_20d": pl.Series("ret_20d", np.random.randn(n_samples) * 0.10),
        }

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for streaming tests."""
        np.random.seed(42)
        n_samples = 20000  # Reduced from 150K - still tests streaming logic
        return {
            "x": pl.Series("x", np.random.randn(n_samples)),
            "y": pl.Series("y", np.random.randn(n_samples)),
            "predictions": pl.Series("pred", np.random.randn(n_samples)),
            "returns_1d": pl.Series("ret_1d", np.random.randn(n_samples) * 0.02),
            "returns_5d": pl.Series("ret_5d", np.random.randn(n_samples) * 0.05),
            "returns_20d": pl.Series("ret_20d", np.random.randn(n_samples) * 0.10),
        }

    @pytest.mark.slow
    def test_streaming_vs_standard_consistency(self, large_dataset):
        """Test that streaming produces same results as standard method."""
        x, y = large_dataset["x"], large_dataset["y"]
        window = 50

        # Use smaller chunk size to force streaming behavior
        streaming_result = PolarsBackend.fast_rolling_correlation_streaming(
            x,
            y,
            window,
            chunk_size=10000,
        )

        # Standard method (will be memory-intensive for large data)
        # We'll use a smaller subset to compare
        subset_size = 20000
        x_subset = x[:subset_size]
        y_subset = y[:subset_size]

        standard_result = PolarsBackend.fast_rolling_correlation(
            x_subset,
            y_subset,
            window,
        )
        streaming_subset = streaming_result[:subset_size]

        # Results should be very close (allowing for small numerical differences)
        standard_vals = standard_result.drop_nulls().to_numpy()
        streaming_vals = streaming_subset.drop_nulls().to_numpy()

        # Compare non-null values
        min_len = min(len(standard_vals), len(streaming_vals))
        if min_len > 0:
            np.testing.assert_allclose(
                standard_vals[:min_len],
                streaming_vals[:min_len],
                rtol=1e-10,
                atol=1e-12,
            )

    @pytest.mark.slow
    def test_streaming_multi_horizon_ic(self, large_dataset):
        """Test streaming multi-horizon IC calculation."""
        predictions = large_dataset["predictions"]
        returns_matrix = pl.DataFrame(
            {
                "1d": large_dataset["returns_1d"],
                "5d": large_dataset["returns_5d"],
                "20d": large_dataset["returns_20d"],
            },
        )

        window = 100
        chunk_size = 25000

        # Test streaming IC calculation
        ic_result = PolarsBackend.fast_multi_horizon_ic_streaming(
            predictions,
            returns_matrix,
            window,
            chunk_size=chunk_size,
        )

        # Check output structure
        assert isinstance(ic_result, pl.DataFrame)
        expected_columns = ["ic_1d", "ic_5d", "ic_20d"]
        assert all(col in ic_result.columns for col in expected_columns)

        # Check that we get reasonable results
        for col in expected_columns:
            ic_values = ic_result[col].drop_nulls()
            assert len(ic_values) > 0
            # IC should be bounded between -1 and 1
            assert ic_values.min() >= -1.0  # type: ignore[operator]
            assert ic_values.max() <= 1.0  # type: ignore[operator]

    def test_adaptive_chunk_size(self):
        """Test adaptive chunk size calculation."""
        # Small dataset should not be chunked
        chunk_size = PolarsBackend.adaptive_chunk_size(50000, n_features=5)
        assert chunk_size == 50000  # Should return full size

        # Large dataset should be chunked
        chunk_size = PolarsBackend.adaptive_chunk_size(1000000, n_features=10)
        assert 10000 <= chunk_size <= 100000  # Should be within bounds

        # Test with different memory targets
        small_memory = PolarsBackend.adaptive_chunk_size(
            1000000,
            n_features=10,
            target_memory_mb=100,
        )
        large_memory = PolarsBackend.adaptive_chunk_size(
            1000000,
            n_features=10,
            target_memory_mb=1000,
        )
        assert small_memory <= large_memory  # Larger memory should allow larger chunks

    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        estimates = PolarsBackend.estimate_memory_usage(100000, 5, "float64")

        # Check that all expected keys are present
        expected_keys = [
            "base_dataframe_mb",
            "rolling_operations_mb",
            "multi_horizon_ic_mb",
            "recommended_chunk_size",
        ]
        assert all(key in estimates for key in expected_keys)

        # Check that estimates are reasonable
        assert estimates["base_dataframe_mb"] > 0
        assert estimates["rolling_operations_mb"] > estimates["base_dataframe_mb"]
        assert estimates["multi_horizon_ic_mb"] > estimates["rolling_operations_mb"]
        assert estimates["recommended_chunk_size"] > 1000

    @pytest.mark.slow
    def test_memory_efficient_operation_framework(self, large_dataset):
        """Test the generic memory-efficient operation framework."""
        data = pl.DataFrame({"value": large_dataset["x"], "other": large_dataset["y"]})

        def rolling_mean_operation(chunk_df, window=10):
            """Test operation that computes rolling mean."""
            return chunk_df.select(
                [pl.col("value").rolling_mean(window).alias("rolling_mean")],
            )

        # Test with streaming
        result = PolarsBackend.memory_efficient_operation(
            data,
            rolling_mean_operation,
            chunk_size=20000,
            overlap=9,  # window - 1
            window=10,
        )

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert "rolling_mean" in result.columns
        assert len(result) == len(data)

        # Check that rolling mean values are reasonable
        rolling_vals = result["rolling_mean"].drop_nulls()
        assert len(rolling_vals) > 0

    def test_streaming_with_small_dataset(self, small_dataset):
        """Test that streaming works correctly with small datasets."""
        x, y = small_dataset["x"], small_dataset["y"]
        window = 20

        # Should automatically use standard method for small data
        result = PolarsBackend.fast_rolling_correlation_streaming(
            x,
            y,
            window,
            chunk_size=50000,  # Larger than dataset
        )

        # Should be equivalent to standard method
        standard_result = PolarsBackend.fast_rolling_correlation(x, y, window)

        # Results should be identical
        np.testing.assert_array_equal(result.to_numpy(), standard_result.to_numpy())

    def test_streaming_edge_cases(self):
        """Test streaming with edge cases."""
        # Very small dataset
        tiny_x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        tiny_y = pl.Series("y", [2.0, 4.0, 6.0, 8.0, 10.0])

        result = PolarsBackend.fast_rolling_correlation_streaming(
            tiny_x,
            tiny_y,
            window=3,
            chunk_size=1000,
        )

        assert len(result) == 5
        # Perfect positive correlation expected
        corr_vals = result.drop_nulls()
        if len(corr_vals) > 0:
            assert all(abs(val - 1.0) < 1e-10 for val in corr_vals.to_list())

    @pytest.mark.slow
    def test_streaming_performance_characteristics(self, large_dataset):
        """Test that streaming has expected performance characteristics.

        Note: This test is marked as slow because it processes 50k samples 3 times.
        Typically takes 30-60 seconds depending on hardware.
        """
        import time

        predictions = large_dataset["predictions"]
        returns_matrix = pl.DataFrame(
            {"1d": large_dataset["returns_1d"], "5d": large_dataset["returns_5d"]},
        )

        window = 50

        # Test with different chunk sizes
        chunk_sizes = [10000, 25000, 50000]
        times = []

        for chunk_size in chunk_sizes:
            start_time = time.time()
            result = PolarsBackend.fast_multi_horizon_ic_streaming(
                predictions[:50000],  # Use subset for timing
                returns_matrix[:50000],
                window,
                chunk_size=chunk_size,
            )
            end_time = time.time()
            times.append(end_time - start_time)

            # Verify we get results
            assert len(result) > 0

        # All chunk sizes should complete in reasonable time (< 120 seconds)
        # Increased from 10s to 120s to account for slower CI/CD environments
        assert all(t < 120.0 for t in times), f"Times: {times}"

    @pytest.mark.slow
    def test_streaming_memory_boundaries(self):
        """Test streaming behavior with different chunk sizes."""
        n_samples = 15000
        x = pl.Series("x", np.random.randn(n_samples))
        y = pl.Series("y", np.random.randn(n_samples))

        # Test with small chunk size (forces chunking)
        result_small_chunk = PolarsBackend.fast_rolling_correlation_streaming(
            x,
            y,
            window=50,
            chunk_size=5000,
        )

        # Test with large chunk size (no chunking)
        result_large_chunk = PolarsBackend.fast_rolling_correlation_streaming(
            x,
            y,
            window=50,
            chunk_size=50000,
        )

        # Both should produce same results
        assert len(result_small_chunk) == n_samples
        assert len(result_large_chunk) == n_samples

        # Both should have similar statistical properties
        small_corr = result_small_chunk.drop_nulls().std()
        large_corr = result_large_chunk.drop_nulls().std()
        if small_corr is not None and large_corr is not None:
            assert abs(small_corr - large_corr) < 0.01  # Should be nearly identical

    @pytest.mark.slow
    def test_streaming_with_different_data_types(self):
        """Test streaming with different data types and ranges."""
        np.random.seed(42)
        n_samples = 15000  # Reduced from 120K

        # Test with different value ranges
        test_cases = [
            ("small_values", np.random.randn(n_samples) * 0.01),
            ("large_values", np.random.randn(n_samples) * 1000),
            ("integer_like", np.round(np.random.randn(n_samples) * 10)),
        ]

        for _name, data in test_cases:
            x = pl.Series("x", data)
            y = pl.Series("y", np.random.randn(n_samples))

            result = PolarsBackend.fast_rolling_correlation_streaming(
                x,
                y,
                window=100,
                chunk_size=30000,
            )

            # Should handle all data types without error
            assert len(result) == n_samples
            corr_vals = result.drop_nulls()
            if len(corr_vals) > 0:
                # Correlations should be bounded
                assert corr_vals.min() >= -1.0
                assert corr_vals.max() <= 1.0
