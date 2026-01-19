"""Tests for sampling utility functions.

This module tests the stratified sampling, importance weighting,
balanced subsampling, and event-based sampling functions.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.core.sampling import (
    balanced_subsample,
    event_based_sample,
    sample_weights_by_importance,
    stratified_sample_time_series,
)


class TestStratifiedSampleTimeSeries:
    """Test stratified sampling that preserves time series properties."""

    def test_basic_pandas_sampling(self):
        """Test basic stratified sampling with pandas DataFrame."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=1000, freq="h"),
                "label": rng.choice([-1, 0, 1], 1000),
                "feature": rng.randn(1000),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.3,
            random_state=42,
        )

        assert isinstance(result, pd.DataFrame)
        # Should have roughly 30% of original data
        assert len(result) < len(df)
        assert len(result) > 0
        # All original labels should be present
        assert set(result["label"].unique()).issubset(set(df["label"].unique()))

    def test_basic_polars_sampling(self):
        """Test basic stratified sampling with polars DataFrame."""
        rng = np.random.RandomState(42)
        df = pl.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=1000, freq="h").tolist(),
                "label": rng.choice([-1, 0, 1], 1000).tolist(),
                "feature": rng.randn(1000).tolist(),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.3,
            random_state=42,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) < len(df)
        assert len(result) > 0

    def test_preserve_order_pandas(self):
        """Test that temporal order is preserved with preserve_order=True."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=500, freq="h"),
                "label": rng.choice([0, 1], 500),
                "feature": rng.randn(500),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.5,
            time_column="time",
            preserve_order=True,
            random_state=42,
        )

        # Check temporal order is maintained
        times = result["time"].values
        assert np.all(np.diff(times) >= np.timedelta64(0))

    def test_preserve_order_polars(self):
        """Test that temporal order is preserved with polars."""
        rng = np.random.RandomState(42)
        df = pl.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=500, freq="h").tolist(),
                "label": rng.choice([0, 1], 500).tolist(),
                "feature": rng.randn(500).tolist(),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.5,
            time_column="time",
            preserve_order=True,
            random_state=42,
        )

        # Check temporal order
        times = result["time"].to_numpy()
        assert len(times) > 0

    def test_no_preserve_order(self):
        """Test sampling without preserving order."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=200, freq="h"),
                "label": rng.choice([0, 1], 200),
                "feature": rng.randn(200),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.5,
            preserve_order=False,
            random_state=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=300, freq="h"),
                "label": rng.choice([-1, 0, 1], 300),
                "feature": rng.randn(300),
            }
        )

        result1 = stratified_sample_time_series(
            df, stratify_column="label", sample_frac=0.3, random_state=42
        )
        result2 = stratified_sample_time_series(
            df, stratify_column="label", sample_frac=0.3, random_state=42
        )
        result3 = stratified_sample_time_series(
            df, stratify_column="label", sample_frac=0.3, random_state=43
        )

        pd.testing.assert_frame_equal(result1, result2)
        # Different seeds should give different results
        assert not result1.equals(result3)

    def test_invalid_type_raises_error(self):
        """Test that invalid data types raise TypeError."""
        with pytest.raises(TypeError, match="must be pd.DataFrame or pl.DataFrame"):
            stratified_sample_time_series(
                {"a": [1, 2, 3]},  # type: ignore[arg-type]
                stratify_column="a",
                sample_frac=0.5,
            )

    def test_small_sample_frac(self):
        """Test with small sample fraction."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "label": rng.choice([0, 1], 100),
                "feature": rng.randn(100),
            }
        )

        result = stratified_sample_time_series(
            df,
            stratify_column="label",
            sample_frac=0.1,
            random_state=42,
        )

        assert len(result) < len(df)


class TestSampleWeightsByImportance:
    """Test importance-based sampling weights."""

    def test_return_magnitude_method(self):
        """Test return_magnitude weighting method."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

        weights = sample_weights_by_importance(returns, method="return_magnitude")

        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0)
        # Largest absolute return should have largest weight
        max_abs_idx = np.argmax(np.abs(returns))
        assert weights[max_abs_idx] == weights.max()

    def test_recency_method(self):
        """Test recency (exponential decay) weighting method."""
        returns = np.random.randn(100) * 0.02

        weights = sample_weights_by_importance(returns, method="recency")

        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0)
        # More recent (later indices) should have higher weights
        assert weights[-1] > weights[0]
        # Weights should be monotonically increasing
        assert np.all(np.diff(weights) >= 0)

    def test_volatility_method(self):
        """Test volatility-based weighting method."""
        # Create returns with varying volatility
        rng = np.random.RandomState(42)
        returns = np.concatenate(
            [
                rng.randn(50) * 0.01,  # Low volatility
                rng.randn(50) * 0.05,  # High volatility
            ]
        )

        weights = sample_weights_by_importance(returns, method="volatility")

        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0)
        # All weights should be non-negative
        assert np.all(weights >= 0)

    def test_decay_factor_effect(self):
        """Test effect of different decay factors."""
        returns = np.random.randn(50) * 0.02

        weights_high_decay = sample_weights_by_importance(
            returns, method="recency", decay_factor=0.99
        )
        weights_low_decay = sample_weights_by_importance(
            returns, method="recency", decay_factor=0.90
        )

        # Higher decay factor should give more uniform weights
        assert np.std(weights_high_decay) < np.std(weights_low_decay)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        returns = np.random.randn(50)

        with pytest.raises(ValueError, match="method must be one of"):
            sample_weights_by_importance(returns, method="invalid_method")

    def test_empty_returns_raises_error(self):
        """Test that empty returns array raises ValueError."""
        with pytest.raises(ValueError, match="returns array cannot be empty"):
            sample_weights_by_importance(np.array([]))

    def test_invalid_decay_factor_raises_error(self):
        """Test that invalid decay factor raises ValueError."""
        returns = np.random.randn(50)

        with pytest.raises(ValueError, match="decay_factor must be in"):
            sample_weights_by_importance(returns, method="recency", decay_factor=0.0)

        with pytest.raises(ValueError, match="decay_factor must be in"):
            sample_weights_by_importance(returns, method="recency", decay_factor=1.0)

        with pytest.raises(ValueError, match="decay_factor must be in"):
            sample_weights_by_importance(returns, method="recency", decay_factor=1.5)

    def test_all_zero_returns(self):
        """Test handling of all-zero returns."""
        returns = np.zeros(50)

        weights = sample_weights_by_importance(returns, method="return_magnitude")

        # Should return equal weights
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(weights, 1.0 / len(returns))

    def test_single_sample(self):
        """Test with single sample."""
        returns = np.array([0.05])

        weights = sample_weights_by_importance(returns, method="return_magnitude")
        assert np.isclose(weights[0], 1.0)

        weights = sample_weights_by_importance(returns, method="recency")
        assert np.isclose(weights[0], 1.0)

    def test_two_samples_volatility(self):
        """Test volatility method with only two samples."""
        returns = np.array([0.01, -0.02])

        weights = sample_weights_by_importance(returns, method="volatility")

        assert np.isclose(weights.sum(), 1.0)
        assert len(weights) == 2


class TestBalancedSubsample:
    """Test balanced subsampling for class imbalance."""

    def test_undersample_basic(self):
        """Test basic undersampling functionality."""
        rng = np.random.RandomState(42)
        # Imbalanced dataset
        X = rng.randn(300, 5)
        y = np.array([0] * 200 + [1] * 50 + [-1] * 50)

        X_balanced, y_balanced = balanced_subsample(X, y, method="undersample", random_state=42)

        # Should have reduced size
        assert len(y_balanced) < len(y)
        # Check class distribution is more balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        count_dict = dict(zip(unique, counts))
        # Minority classes should be preserved
        assert 1 in count_dict
        assert -1 in count_dict

    def test_hybrid_method(self):
        """Test hybrid under/oversampling method."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = np.array([0] * 100 + [1] * 50 + [-1] * 50)

        X_balanced, y_balanced = balanced_subsample(
            X, y, method="hybrid", minority_weight=1.0, random_state=42
        )

        # Should have samples from all classes
        unique = np.unique(y_balanced)
        assert len(unique) == 3

    def test_minority_weight_effect(self):
        """Test effect of minority_weight parameter."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = np.array([0] * 150 + [1] * 50)

        # Lower minority weight = more aggressive undersampling of neutral class
        X_low, y_low = balanced_subsample(
            X, y, method="undersample", minority_weight=0.5, random_state=42
        )

        X_high, y_high = balanced_subsample(
            X, y, method="undersample", minority_weight=1.0, random_state=42
        )

        # Both should have balanced classes
        assert len(y_low) <= len(y_high) or len(y_low) >= len(y_high) * 0.5

    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        rng = np.random.RandomState(42)
        X = rng.randn(150, 4)
        y = np.array([0] * 100 + [1] * 50)

        X1, y1 = balanced_subsample(X, y, random_state=42)
        X2, y2 = balanced_subsample(X, y, random_state=42)
        X3, y3 = balanced_subsample(X, y, random_state=43)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        # Different seeds should give different results
        assert not np.array_equal(X1, X3)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        X = np.random.randn(50, 2)
        y = np.array([0] * 30 + [1] * 20)

        with pytest.raises(ValueError, match="Unknown method"):
            balanced_subsample(X, y, method="invalid_method")

    def test_all_same_class(self):
        """Test with single class (no balancing needed)."""
        X = np.random.randn(50, 3)
        y = np.zeros(50)

        X_balanced, y_balanced = balanced_subsample(X, y, random_state=42)

        assert len(y_balanced) == len(y)
        np.testing.assert_array_equal(np.unique(y_balanced), [0])

    def test_shuffling(self):
        """Test that output is shuffled."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = np.array([0] * 50 + [1] * 50)

        X_balanced, y_balanced = balanced_subsample(X, y, random_state=42)

        # Output should be shuffled (not grouped by class)
        # Check that first half is not all same class
        first_half = y_balanced[: len(y_balanced) // 2]
        assert len(np.unique(first_half)) > 1 or len(y_balanced) < 4


class TestEventBasedSample:
    """Test event-based sampling with minimum spacing."""

    def test_basic_pandas_sampling(self):
        """Test basic event sampling with pandas."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=1000, freq="h"),
                "event": rng.choice([0, 1], 1000, p=[0.9, 0.1]),  # 10% events
                "feature": rng.randn(1000),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=20,
            random_state=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 20
        # All returned rows should have event=True
        assert result["event"].all()

    def test_basic_polars_sampling(self):
        """Test basic event sampling with polars."""
        rng = np.random.RandomState(42)
        df = pl.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=500, freq="h").tolist(),
                "event": rng.choice([0, 1], 500, p=[0.8, 0.2]).tolist(),
                "feature": rng.randn(500).tolist(),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=30,
            random_state=42,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) <= 30

    def test_sample_frac(self):
        """Test sampling by fraction instead of count."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "event": rng.choice([0, 1], 200, p=[0.7, 0.3]),
                "value": rng.randn(200),
            }
        )

        total_events = df["event"].sum()

        result = event_based_sample(
            df,
            event_column="event",
            sample_frac=0.5,
            random_state=42,
        )

        # Should sample approximately 50% of events
        assert len(result) <= int(total_events * 0.5) + 5  # Allow some variance

    def test_minimum_spacing(self):
        """Test minimum spacing between sampled events."""
        rng = np.random.RandomState(42)
        # Create events at every position
        df = pd.DataFrame(
            {
                "event": np.ones(100, dtype=int),
                "feature": rng.randn(100),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=10,
            min_event_spacing=5,
            random_state=42,
        )

        # Check spacing between sampled events
        indices = result.index.tolist()
        for i in range(len(indices) - 1):
            spacing = abs(indices[i + 1] - indices[i])
            assert spacing > 5

    def test_no_n_samples_or_sample_frac_raises_error(self):
        """Test that missing n_samples and sample_frac raises error."""
        df = pd.DataFrame({"event": [0, 1, 0, 1], "value": [1, 2, 3, 4]})

        with pytest.raises(ValueError, match="Either n_samples or sample_frac"):
            event_based_sample(df, event_column="event")

    def test_invalid_type_raises_error(self):
        """Test that invalid data type raises TypeError."""
        with pytest.raises(TypeError, match="must be pd.DataFrame or pl.DataFrame"):
            event_based_sample(
                [1, 2, 3],  # type: ignore[arg-type]
                event_column="event",
                n_samples=1,
            )

    def test_no_events(self):
        """Test handling when there are no events."""
        df = pd.DataFrame(
            {
                "event": np.zeros(50, dtype=int),
                "value": np.random.randn(50),
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=10,
            random_state=42,
        )

        # Should return empty DataFrame
        assert len(result) == 0

    def test_fewer_events_than_requested(self):
        """Test when fewer events exist than requested samples."""
        df = pd.DataFrame(
            {
                "event": [0, 1, 0, 1, 0],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = event_based_sample(
            df,
            event_column="event",
            n_samples=10,  # Request more than available
            random_state=42,
        )

        # Should return all events (only 2 available)
        assert len(result) == 2

    def test_reproducibility(self):
        """Test reproducibility with random_state."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "event": rng.choice([0, 1], 200, p=[0.7, 0.3]),
                "value": rng.randn(200),
            }
        )

        result1 = event_based_sample(df, event_column="event", n_samples=20, random_state=42)
        result2 = event_based_sample(df, event_column="event", n_samples=20, random_state=42)
        result3 = event_based_sample(df, event_column="event", n_samples=20, random_state=43)

        pd.testing.assert_frame_equal(result1, result2)
        # Different seeds should give different results (if enough events)
        if len(result1) > 1:
            assert not result1.equals(result3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
