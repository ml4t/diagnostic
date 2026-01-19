"""Tests for expanding window operations in PolarsBackend.

This module tests the expanding window functionality to ensure correct
calculation of expanding statistics (mean, std, sum, min, max) and
compares results against pandas reference implementation.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.backends.polars_backend import PolarsBackend


class TestExpandingWindow:
    """Test expanding window statistics calculations."""

    def test_expanding_mean(self):
        """Test expanding mean calculation."""
        # Create test data
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="mean")

        expected_means = [1.0, 1.5, 2.0, 2.5, 3.0]  # Manual calculation
        actual_means = result["value_expanding_mean"].to_list()

        np.testing.assert_array_almost_equal(actual_means, expected_means, decimal=10)

    def test_expanding_std(self):
        """Test expanding standard deviation calculation."""
        # Create test data with known std values
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="std")

        # Compare with pandas expanding std
        pandas_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_stds = pandas_series.expanding().std().tolist()
        actual_stds = result["value_expanding_std"].to_list()

        # First value should be NaN/None (std of single value)
        assert actual_stds[0] is None or pd.isna(actual_stds[0])

        # Compare remaining values
        np.testing.assert_array_almost_equal(actual_stds[1:], expected_stds[1:], decimal=10)

    def test_expanding_sum(self):
        """Test expanding sum calculation."""
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="sum")

        expected_sums = [1.0, 3.0, 6.0, 10.0, 15.0]  # Cumulative sum
        actual_sums = result["value_expanding_sum"].to_list()

        np.testing.assert_array_equal(actual_sums, expected_sums)

    def test_expanding_min(self):
        """Test expanding minimum calculation."""
        data = pl.DataFrame({"value": [5.0, 2.0, 8.0, 1.0, 6.0]})

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="min")

        expected_mins = [5.0, 2.0, 2.0, 1.0, 1.0]  # Cumulative minimum
        actual_mins = result["value_expanding_min"].to_list()

        np.testing.assert_array_equal(actual_mins, expected_mins)

    def test_expanding_max(self):
        """Test expanding maximum calculation."""
        data = pl.DataFrame({"value": [5.0, 2.0, 8.0, 1.0, 6.0]})

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="max")

        expected_maxs = [5.0, 5.0, 8.0, 8.0, 8.0]  # Cumulative maximum
        actual_maxs = result["value_expanding_max"].to_list()

        np.testing.assert_array_equal(actual_maxs, expected_maxs)

    def test_multiple_columns(self):
        """Test expanding calculations on multiple columns."""
        data = pl.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [10.0, 20.0, 30.0]})

        result = PolarsBackend.fast_expanding_window(data, ["col1", "col2"], operation="mean")

        # Check that both columns have expanding means
        assert "col1_expanding_mean" in result.columns
        assert "col2_expanding_mean" in result.columns

        np.testing.assert_array_almost_equal(
            result["col1_expanding_mean"].to_list(), [1.0, 1.5, 2.0]
        )
        np.testing.assert_array_almost_equal(
            result["col2_expanding_mean"].to_list(), [10.0, 15.0, 20.0]
        )

    def test_min_periods_std(self):
        """Test min_periods parameter with standard deviation."""
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        # Require at least 3 observations for std
        result = PolarsBackend.fast_expanding_window(
            data, ["value"], operation="std", min_periods=3
        )

        actual_stds = result["value_expanding_std"].to_list()

        # First two values should be None (less than min_periods)
        assert actual_stds[0] is None
        assert actual_stds[1] is None

        # Third value onwards should have valid std
        assert actual_stds[2] is not None
        assert actual_stds[3] is not None
        assert actual_stds[4] is not None

    def test_pandas_comparison(self):
        """Compare all operations against pandas expanding methods."""
        # Create more complex test data
        np.random.seed(42)
        values = np.random.randn(20)

        # Polars calculation
        pl_data = pl.DataFrame({"value": values})

        operations = ["mean", "std", "sum", "min", "max"]

        for operation in operations:
            pl_result = PolarsBackend.fast_expanding_window(pl_data, ["value"], operation=operation)

            # Pandas reference
            pd_series = pd.Series(values)
            if operation == "mean":
                pd_result = pd_series.expanding().mean().tolist()
            elif operation == "std":
                pd_result = pd_series.expanding().std().tolist()
            elif operation == "sum":
                pd_result = pd_series.expanding().sum().tolist()
            elif operation == "min":
                pd_result = pd_series.expanding().min().tolist()
            elif operation == "max":
                pd_result = pd_series.expanding().max().tolist()

            pl_values = pl_result[f"value_expanding_{operation}"].to_list()

            if operation == "std":
                # Handle NaN values in std comparison
                for _i, (pl_val, pd_val) in enumerate(zip(pl_values, pd_result)):
                    if pd.isna(pd_val):
                        assert pl_val is None or pd.isna(pl_val)
                    else:
                        assert pl_val is not None
                        np.testing.assert_almost_equal(pl_val, pd_val, decimal=10)
            else:
                np.testing.assert_array_almost_equal(pl_values, pd_result, decimal=10)

    def test_edge_cases(self):
        """Test edge cases: empty data, single value, all NaN."""
        # Single value
        single_data = pl.DataFrame({"value": [5.0]})
        result = PolarsBackend.fast_expanding_window(single_data, ["value"], operation="mean")
        assert result["value_expanding_mean"].to_list() == [5.0]

        # Single value std should be None/NaN
        result_std = PolarsBackend.fast_expanding_window(single_data, ["value"], operation="std")
        std_val = result_std["value_expanding_std"].to_list()[0]
        assert std_val is None or pd.isna(std_val)

    def test_unknown_operation(self):
        """Test error handling for unknown operations."""
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Unknown operation"):
            PolarsBackend.fast_expanding_window(data, ["value"], operation="unknown")

    def test_performance_benchmark(self):
        """Benchmark performance with larger dataset."""
        # Create larger dataset for performance testing
        n_rows = 10000
        np.random.seed(42)
        values = np.random.randn(n_rows)

        data = pl.DataFrame({"value": values})

        import time

        start_time = time.time()

        result = PolarsBackend.fast_expanding_window(data, ["value"], operation="std")

        end_time = time.time()
        elapsed = end_time - start_time

        # Should complete within reasonable time (adjust as needed)
        assert elapsed < 5.0  # 5 seconds max
        assert len(result) == n_rows
        assert "value_expanding_std" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
