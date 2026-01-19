"""Test for cumulative returns visualization fix.

This test verifies that plot_quantile_returns correctly aligns
positions to time index before calculating cumulative returns.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation.visualization import plot_quantile_returns


class TestCumulativeReturnsVisualization:
    """Test cumulative returns calculation with proper time alignment."""

    def test_time_aligned_cumulative_returns(self):
        """Test that cumulative returns are properly aligned to time index."""
        # Create test data with explicit time index
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create predictions and returns
        np.random.seed(42)
        predictions = pd.Series(np.random.randn(100), index=dates)
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)

        # Call the function
        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=5, show_cumulative=True
        )

        # Verify the figure was created
        assert fig is not None

        # Check that cumulative returns subplot exists
        assert len(fig.data) > 5  # Bar chart + 5 quantile lines

        # Verify that cumulative returns traces use time index
        for trace_idx in range(1, 6):  # Skip the bar chart
            trace = fig.data[trace_idx]

            # Check that x-axis data contains datetime values
            if hasattr(trace, "x") and trace.x is not None and len(trace.x) > 0:
                # Should be datetime objects from the time index
                assert isinstance(trace.x[0], datetime | pd.Timestamp | np.datetime64)

                # Verify that x values are sorted (time-aligned)
                x_values = pd.to_datetime(trace.x)
                assert x_values.is_monotonic_increasing

    def test_cumulative_returns_without_time_index(self):
        """Test fallback behavior when no time index is available."""
        # Create test data without time index
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100) * 0.02

        # Call the function
        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=5, show_cumulative=True
        )

        # Verify the figure was created
        assert fig is not None

        # Check that cumulative returns subplot exists
        assert len(fig.data) > 5  # Bar chart + 5 quantile lines

        # Verify that cumulative returns traces use position index
        for trace_idx in range(1, 6):  # Skip the bar chart
            trace = fig.data[trace_idx]

            # Check that x-axis data contains position indices
            if hasattr(trace, "x") and trace.x is not None and len(trace.x) > 0:
                # Should be integer positions
                assert isinstance(trace.x[0], int | np.integer | float)

                # Verify that positions are sequential
                x_array = np.array(trace.x)
                expected = np.arange(len(x_array))
                np.testing.assert_array_almost_equal(x_array, expected)

    def test_cumulative_returns_ordering(self):
        """Test that cumulative returns maintain proper temporal ordering."""
        # Create test data with shuffled time index
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Shuffle the dates to test proper sorting
        np.random.seed(42)
        shuffle_idx = np.random.permutation(50)
        dates_shuffled = dates[shuffle_idx]

        predictions = pd.Series(np.random.randn(50), index=dates_shuffled)
        returns = pd.Series(np.ones(50) * 0.01, index=dates_shuffled)  # Constant positive returns

        # Call the function
        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=3, show_cumulative=True
        )

        # For each quantile trace
        for trace_idx in range(1, 4):  # 3 quantiles
            trace = fig.data[trace_idx]

            if hasattr(trace, "y") and trace.y is not None and len(trace.y) > 1:
                # Cumulative returns should be monotonically increasing for constant positive returns
                y_values = np.array(trace.y)
                # Check that cumulative returns are non-decreasing (allowing for floating point errors)
                differences = np.diff(y_values)
                assert np.all(differences >= -1e-10), (
                    "Cumulative returns should be non-decreasing for positive returns"
                )

    def test_quantile_separation(self):
        """Test that different quantiles show different cumulative patterns."""
        # Create test data with clear prediction-return relationship
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create predictions with clear ranking
        predictions = pd.Series(np.linspace(-2, 2, 100), index=dates)

        # Returns correlated with predictions (higher predictions = higher returns)
        returns = pd.Series(predictions.values * 0.01 + np.random.randn(100) * 0.001, index=dates)

        # Call the function
        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=5, show_cumulative=True
        )

        # Extract final cumulative returns for each quantile
        final_cumulative = []
        for trace_idx in range(1, 6):  # 5 quantiles
            trace = fig.data[trace_idx]
            if hasattr(trace, "y") and trace.y is not None and len(trace.y) > 0:
                final_cumulative.append(trace.y[-1])

        # Higher quantiles should have higher final cumulative returns
        # (allowing some tolerance for randomness)
        if len(final_cumulative) == 5:
            # Check general trend (Q5 should be higher than Q1)
            assert final_cumulative[-1] > final_cumulative[0], (
                "Highest quantile should have higher cumulative return than lowest"
            )

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Empty series
        predictions = pd.Series([])
        returns = pd.Series([])

        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=5, show_cumulative=True
        )

        # Should return a figure without error
        assert fig is not None

    def test_nan_data_handling(self):
        """Test handling of NaN data."""
        # All NaN data
        predictions = pd.Series([np.nan] * 10)
        returns = pd.Series([np.nan] * 10)

        fig = plot_quantile_returns(
            predictions=predictions, returns=returns, n_quantiles=5, show_cumulative=True
        )

        # Should return a figure with annotation about no valid data
        assert fig is not None
        # Check for the "No valid data" annotation
        assert len(fig.layout.annotations) > 0

    def test_mixed_series_array_inputs(self):
        """Test that function handles mixed Series and array inputs."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # Test with Series predictions and array returns
        predictions_series = pd.Series(np.random.randn(50), index=dates)
        returns_array = np.random.randn(50) * 0.02

        fig = plot_quantile_returns(
            predictions=predictions_series,
            returns=returns_array,
            n_quantiles=3,
            show_cumulative=True,
        )
        assert fig is not None

        # Test with array predictions and Series returns
        predictions_array = np.random.randn(50)
        returns_series = pd.Series(np.random.randn(50) * 0.02, index=dates)

        fig = plot_quantile_returns(
            predictions=predictions_array,
            returns=returns_series,
            n_quantiles=3,
            show_cumulative=True,
        )
        assert fig is not None

        # The second case should use the time index from returns
        for trace_idx in range(1, 4):  # 3 quantiles
            trace = fig.data[trace_idx]
            if hasattr(trace, "x") and trace.x is not None and len(trace.x) > 0:
                # Should use datetime from returns Series
                assert isinstance(trace.x[0], datetime | pd.Timestamp | np.datetime64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
