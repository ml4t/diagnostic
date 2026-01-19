"""Tests for PolarsBackend optimized operations."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.backends.polars_backend import PolarsBackend


class TestFastRollingCorrelation:
    """Tests for fast_rolling_correlation method."""

    def test_basic_correlation(self):
        """Test basic rolling correlation calculation."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = pl.Series("y", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = PolarsBackend.fast_rolling_correlation(x, y, window=5)

        assert len(result) == len(x)
        # Perfect positive correlation for identical series
        assert result[-1] == pytest.approx(1.0, rel=1e-5)

    def test_negative_correlation(self):
        """Test negative rolling correlation."""
        x = pl.Series("x", list(range(1, 11)))
        y = pl.Series("y", list(range(10, 0, -1)))

        result = PolarsBackend.fast_rolling_correlation(x, y, window=5)

        # Perfect negative correlation
        assert result[-1] == pytest.approx(-1.0, rel=1e-5)

    def test_min_periods(self):
        """Test min_periods parameter."""
        x = pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = pl.Series("y", [1.0, 2.0, 3.0, 4.0, 5.0])

        result = PolarsBackend.fast_rolling_correlation(x, y, window=5, min_periods=3)

        assert len(result) == len(x)
        # First 2 values should be NaN (less than min_periods)
        assert result.is_null()[0]
        assert result.is_null()[1]
        # Third value should be valid (3 periods available)
        assert not result.is_null()[2]


class TestFastRollingSpearmanCorrelation:
    """Tests for fast_rolling_spearman_correlation method."""

    def test_basic_spearman(self):
        """Test basic rolling Spearman correlation."""
        np.random.seed(42)
        x = pl.Series("x", np.random.randn(50))
        y = pl.Series("y", np.random.randn(50))

        result = PolarsBackend.fast_rolling_spearman_correlation(x, y, window=10)

        assert len(result) == len(x)

    def test_perfect_monotonic(self):
        """Test with perfect monotonic relationship."""
        x = pl.Series("x", list(range(1, 21)))
        y = pl.Series("y", list(range(1, 21)))

        result = PolarsBackend.fast_rolling_spearman_correlation(x, y, window=10)

        # Perfect Spearman correlation
        assert result[-1] == pytest.approx(1.0, rel=1e-5)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        x = pl.Series("x", [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = pl.Series("y", [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = PolarsBackend.fast_rolling_spearman_correlation(x, y, window=5)

        assert len(result) == len(x)


class TestFastMultiHorizonIC:
    """Tests for fast_multi_horizon_ic method."""

    def test_basic_multi_horizon(self):
        """Test multi-horizon IC calculation."""
        np.random.seed(42)
        n = 100

        predictions = pl.Series("pred", np.random.randn(n))
        returns_matrix = pl.DataFrame(
            {
                "1d": np.random.randn(n),
                "5d": np.random.randn(n),
                "20d": np.random.randn(n),
            }
        )

        result = PolarsBackend.fast_multi_horizon_ic(predictions, returns_matrix, window=20)

        assert result.shape[1] == 3  # Three horizons
        assert "ic_1d" in result.columns
        assert "ic_5d" in result.columns
        assert "ic_20d" in result.columns

    def test_min_periods_default(self):
        """Test default min_periods calculation."""
        predictions = pl.Series("pred", range(50))
        returns_matrix = pl.DataFrame({"1d": range(50)})

        result = PolarsBackend.fast_multi_horizon_ic(predictions, returns_matrix, window=20)

        # First 9 values should be NaN (min_periods = 20 // 2 = 10)
        assert np.isnan(result["ic_1d"][0])


class TestFastQuantileAssignment:
    """Tests for fast_quantile_assignment method."""

    def test_basic_quantiles(self):
        """Test basic quantile assignment."""
        df = pl.DataFrame(
            {
                "value": list(range(100)),
            }
        )

        result = PolarsBackend.fast_quantile_assignment(df, "value", n_quantiles=5)

        assert "value_quantile" in result.columns
        # Should have 5 unique quantiles
        unique_quantiles = result["value_quantile"].unique()
        assert len(unique_quantiles) == 5

    def test_grouped_quantiles(self):
        """Test quantile assignment within groups."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "value": list(range(50)) * 2,
            }
        )

        result = PolarsBackend.fast_quantile_assignment(
            df, "value", n_quantiles=5, by_group="group"
        )

        assert "value_quantile" in result.columns


class TestFastTimeAwareSplit:
    """Tests for fast_time_aware_split method."""

    def test_basic_split(self):
        """Test basic time-aware splitting."""
        dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True)
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(len(dates)),
            }
        )

        train, test, buffer = PolarsBackend.fast_time_aware_split(
            df,
            time_column="date",
            test_start=pl.date(2024, 7, 1),
            test_end=pl.date(2024, 9, 1),
        )

        # Test data should be in the specified range
        assert len(test) > 0
        assert len(train) > 0

    def test_split_with_buffers(self):
        """Test split with buffer zones."""
        from datetime import timedelta

        dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True)
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(len(dates)),
            }
        )

        train, test, buffer = PolarsBackend.fast_time_aware_split(
            df,
            time_column="date",
            test_start=pl.date(2024, 7, 1),
            test_end=pl.date(2024, 9, 1),
            buffer_before=timedelta(days=30),
            buffer_after=timedelta(days=30),
        )

        assert len(test) > 0
        assert len(train) > 0
        assert len(buffer) > 0


class TestFastGroupStatistics:
    """Tests for fast_group_statistics method."""

    def test_basic_statistics(self):
        """Test basic group statistics."""
        df = pl.DataFrame(
            {
                "group": ["A", "A", "B", "B", "B"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = PolarsBackend.fast_group_statistics(
            df,
            group_column="group",
            value_column="value",
            statistics=["mean", "std", "count"],
        )

        assert "value_mean" in result.columns
        assert "value_std" in result.columns
        assert "value_count" in result.columns

    def test_all_statistics(self):
        """Test all available statistics."""
        df = pl.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "value": list(range(20)),
            }
        )

        result = PolarsBackend.fast_group_statistics(
            df,
            group_column="group",
            value_column="value",
            statistics=["mean", "std", "min", "max", "count", "sum", "median"],
        )

        expected_cols = [
            "value_mean",
            "value_std",
            "value_min",
            "value_max",
            "value_count",
            "value_sum",
            "value_median",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_invalid_statistic(self):
        """Test error for invalid statistic."""
        df = pl.DataFrame(
            {
                "group": ["A", "B"],
                "value": [1.0, 2.0],
            }
        )

        with pytest.raises(ValueError, match="Unknown statistic"):
            PolarsBackend.fast_group_statistics(df, "group", "value", statistics=["invalid_stat"])


class TestFastExpandingWindow:
    """Tests for fast_expanding_window method."""

    def test_expanding_mean(self):
        """Test expanding mean calculation."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = PolarsBackend.fast_expanding_window(df, columns=["value"], operation="mean")

        assert "value_expanding_mean" in result.columns
        # First value should be 1, last should be 3 (mean of 1-5)
        assert result["value_expanding_mean"][0] == pytest.approx(1.0)
        assert result["value_expanding_mean"][-1] == pytest.approx(3.0)

    def test_expanding_sum(self):
        """Test expanding sum calculation."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = PolarsBackend.fast_expanding_window(df, columns=["value"], operation="sum")

        assert "value_expanding_sum" in result.columns
        assert result["value_expanding_sum"][-1] == pytest.approx(15.0)

    def test_invalid_operation(self):
        """Test error for invalid operation."""
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Unknown operation"):
            PolarsBackend.fast_expanding_window(df, columns=["value"], operation="invalid")


class TestToNumpyBatch:
    """Tests for to_numpy_batch method."""

    def test_small_data_direct_conversion(self):
        """Test direct conversion for small data."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )

        result = PolarsBackend.to_numpy_batch(df, batch_size=10000)

        assert result.shape == (3, 2)
        assert result[0, 0] == 1.0

    def test_batched_conversion(self):
        """Test batched conversion for larger data."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "a": np.random.randn(1000),
                "b": np.random.randn(1000),
            }
        )

        result = PolarsBackend.to_numpy_batch(df, batch_size=100)

        assert result.shape == (1000, 2)

    def test_column_selection(self):
        """Test conversion with column selection."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            }
        )

        result = PolarsBackend.to_numpy_batch(df, columns=["a", "c"])

        assert result.shape == (3, 2)


class TestStreamingCorrelation:
    """Tests for streaming rolling correlation."""

    def test_small_data_uses_standard(self):
        """Test that small data uses standard method."""
        x = pl.Series("x", list(range(100)))
        y = pl.Series("y", list(range(100)))

        result = PolarsBackend.fast_rolling_correlation_streaming(x, y, window=10, chunk_size=50000)

        assert len(result) == 100

    def test_large_data_streaming(self):
        """Test streaming for larger data."""
        np.random.seed(42)
        n = 1000
        x = pl.Series("x", np.random.randn(n))
        y = pl.Series("y", np.random.randn(n))

        result = PolarsBackend.fast_rolling_correlation_streaming(x, y, window=50, chunk_size=200)

        assert len(result) == n


class TestMemoryEstimation:
    """Tests for memory estimation utilities."""

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        estimates = PolarsBackend.estimate_memory_usage(
            n_samples=100000,
            n_features=50,
            data_type="float64",
        )

        assert "base_dataframe_mb" in estimates
        assert "rolling_operations_mb" in estimates
        assert "multi_horizon_ic_mb" in estimates
        assert "recommended_chunk_size" in estimates
        assert estimates["base_dataframe_mb"] > 0

    def test_adaptive_chunk_size_small_data(self):
        """Test chunk size for small data (returns full size)."""
        chunk = PolarsBackend.adaptive_chunk_size(
            total_samples=1000,
            n_features=10,
        )

        # Small data should return full dataset
        assert chunk == 1000

    def test_adaptive_chunk_size_large_data(self):
        """Test chunk size for large data."""
        chunk = PolarsBackend.adaptive_chunk_size(
            total_samples=1000000,
            n_features=100,
            target_memory_mb=500,
        )

        # Should be within bounds
        assert 10000 <= chunk <= 100000


class TestMemoryEfficientOperation:
    """Tests for generic memory-efficient operation."""

    def test_small_data_direct(self):
        """Test direct processing for small data."""
        df = pl.DataFrame(
            {
                "value": list(range(100)),
            }
        )

        def double_values(chunk):
            return chunk.with_columns((pl.col("value") * 2).alias("doubled"))

        result = PolarsBackend.memory_efficient_operation(df, double_values, chunk_size=50000)

        assert "doubled" in result.columns
        assert result["doubled"][-1] == 198

    def test_chunked_processing(self):
        """Test chunked processing for larger data."""
        df = pl.DataFrame(
            {
                "value": list(range(500)),
            }
        )

        def process_chunk(chunk):
            return chunk.select(
                [
                    pl.col("value"),
                    (pl.col("value") + 1).alias("plus_one"),
                ]
            )

        result = PolarsBackend.memory_efficient_operation(
            df, process_chunk, chunk_size=100, overlap=0
        )

        assert len(result) == 500
        assert result["plus_one"][-1] == 500


class TestRollingCorrelationExpr:
    """Tests for _rolling_correlation_expr method."""

    def test_expression_generation(self):
        """Test that expression can be used in DataFrame context."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )

        expr = PolarsBackend._rolling_correlation_expr("x", "y", window=5, min_periods=3)
        result = df.select(expr.alias("corr"))

        assert "corr" in result.columns
        # Perfect correlation for identical series
        assert result["corr"][-1] == pytest.approx(1.0, rel=1e-5)
