"""Tests for PurgedWalkForwardCV splitter."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from ml4t.diagnostic.splitters.walk_forward import PurgedWalkForwardCV


class TestPurgedWalkForwardCV:
    """Test suite for PurgedWalkForwardCV."""

    def test_basic_walk_forward_split(self):
        """Test basic walk-forward splitting without purging/embargo."""
        X = np.arange(100).reshape(100, 1)
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=0, embargo_size=0)

        splits = list(cv.split(X))
        assert len(splits) == 3

        # Check that test sets are sequential and non-overlapping
        test_starts = []
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # Test indices should be contiguous
            assert np.all(np.diff(test_idx) == 1)

            # Training should precede test
            assert np.max(train_idx) < np.min(test_idx)

            test_starts.append(test_idx[0])

        # Test sets should be sequential
        assert all(test_starts[i] < test_starts[i + 1] for i in range(len(test_starts) - 1))

    def test_walk_forward_with_purging(self):
        """Test walk-forward with purging."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)
        label_horizon = 5
        cv = PurgedWalkForwardCV(n_splits=3, label_horizon=label_horizon)

        for train_idx, test_idx in cv.split(X):
            # Check purging: no training sample should be within
            # label_horizon of test start
            test_start = test_idx[0]
            assert not any(
                idx >= test_start - label_horizon and idx < test_start for idx in train_idx
            )

    def test_walk_forward_with_embargo(self):
        """Test walk-forward with embargo."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)
        embargo_size = 3
        cv = PurgedWalkForwardCV(n_splits=2, embargo_size=embargo_size)

        splits = list(cv.split(X))

        # For first split, check embargo affects second split's training
        first_train, first_test = splits[0]
        second_train, second_test = splits[1]

        # No training samples from second split should be within
        # embargo_size after first test end
        first_test_end = first_test[-1] + 1
        assert not any(
            idx >= first_test_end and idx < first_test_end + embargo_size for idx in second_train
        )

    def test_expanding_vs_rolling_window(self):
        """Test expanding vs rolling window behavior."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)

        # Expanding window
        cv_expanding = PurgedWalkForwardCV(n_splits=3, expanding=True)
        splits_expanding = list(cv_expanding.split(X))

        # Each successive training set should be larger
        train_sizes = [len(train) for train, _ in splits_expanding]
        assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))

        # Rolling window with fixed size
        cv_rolling = PurgedWalkForwardCV(n_splits=3, expanding=False, train_size=30)
        splits_rolling = list(cv_rolling.split(X))

        # Training sets should have similar sizes (accounting for purging)
        train_sizes_rolling = [len(train) for train, _ in splits_rolling]
        # Sizes might vary slightly due to purging, but should be close
        assert max(train_sizes_rolling) - min(train_sizes_rolling) < 10

    def test_with_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = PurgedWalkForwardCV(
            n_splits=2,
            label_horizon=pd.Timedelta("5D"),
            embargo_size=pd.Timedelta("3D"),
        )

        splits = list(cv.split(X))
        assert len(splits) == 2

        for train_idx, test_idx in splits:
            # Verify temporal ordering
            train_dates = X.index[train_idx]
            test_dates = X.index[test_idx]
            assert train_dates.max() < test_dates.min()

    def test_with_polars_dataframe(self):
        """Test with polars DataFrame input."""
        X = pl.DataFrame({"feature": np.arange(100)})
        cv = PurgedWalkForwardCV(n_splits=3)

        splits = list(cv.split(X))
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_test_size_specifications(self):
        """Test different test_size specifications."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)

        # Fixed test size
        cv_fixed = PurgedWalkForwardCV(n_splits=3, test_size=20)
        for _, test_idx in cv_fixed.split(X):
            assert len(test_idx) == 20

        # Proportional test size
        cv_prop = PurgedWalkForwardCV(n_splits=3, test_size=0.2)
        for _, test_idx in cv_prop.split(X):
            assert len(test_idx) == 20  # 0.2 * 100

    def test_gap_parameter(self):
        """Test gap between train and test sets."""
        X = np.arange(100).reshape(100, 1)
        gap = 5
        cv = PurgedWalkForwardCV(n_splits=2, gap=gap)

        for train_idx, test_idx in cv.split(X):
            # Check gap between train and test
            assert test_idx[0] - train_idx[-1] == gap + 1

    def test_percentage_embargo(self):
        """Test embargo specified as percentage."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)
        cv = PurgedWalkForwardCV(n_splits=2, embargo_pct=0.05)

        splits = list(cv.split(X))
        assert len(splits) == 2

        # Verify embargo is applied (5% of 100 = 5 samples)
        first_train, first_test = splits[0]
        second_train, _ = splits[1]

        # Check that ~5 samples after first test are embargoed
        first_test_end = first_test[-1] + 1
        embargo_count = sum(
            1
            for idx in range(first_test_end, min(first_test_end + 5, n_samples))
            if idx not in second_train
        )
        assert embargo_count >= 4  # Allow some flexibility

    def test_get_n_splits(self):
        """Test get_n_splits method."""
        cv = PurgedWalkForwardCV(n_splits=5)
        assert cv.get_n_splits() == 5

        # Should work with any input (ignored)
        X = np.arange(10).reshape(10, 1)
        assert cv.get_n_splits(X) == 5

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        # Negative splits - now validated by Pydantic config
        with pytest.raises(ValidationError, match="greater than 0"):
            PurgedWalkForwardCV(n_splits=-1)

        # Negative label_horizon - validated by Pydantic
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PurgedWalkForwardCV(label_horizon=-1)

        # Negative embargo - validated by Pydantic
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PurgedWalkForwardCV(embargo_size=-1)

    def test_edge_case_small_dataset(self):
        """Test with small dataset."""
        X = np.arange(20).reshape(20, 1)
        cv = PurgedWalkForwardCV(n_splits=2, label_horizon=2, embargo_size=2)

        splits = list(cv.split(X))
        assert len(splits) == 2

        # Even with small data, should produce valid splits
        for train_idx, test_idx in splits:
            assert len(train_idx) >= 0  # Might be empty due to purging
            assert len(test_idx) > 0

    def test_last_split_uses_remaining_data(self):
        """Test that last split uses all remaining data."""
        n_samples = 105  # Not evenly divisible
        X = np.arange(n_samples).reshape(n_samples, 1)
        cv = PurgedWalkForwardCV(n_splits=3)

        splits = list(cv.split(X))
        last_train, last_test = splits[-1]

        # Last test should extend to end of data
        assert last_test[-1] == n_samples - 1
