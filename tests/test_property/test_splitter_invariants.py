"""Property-based tests for splitter invariants using Hypothesis.

This module tests mathematical properties that must hold for all valid
cross-validation splitters, regardless of the specific implementation.
"""

import numpy as np
import pandas as pd
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.diagnostic.splitters.base import BaseSplitter


class SimpleSplitter(BaseSplitter):
    """Simple splitter for testing invariants."""

    def __init__(self, n_splits: int = 3):
        """Initialize the splitter with specified number of splits."""
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """Generate train/test splits."""
        n_samples = self._validate_data(X, y, groups)
        indices = np.arange(n_samples)

        # Handle case where n_splits > n_samples
        actual_splits = min(self.n_splits, n_samples)

        # Simple non-overlapping consecutive splits
        fold_sizes = np.full(actual_splits, n_samples // actual_splits)
        fold_sizes[: n_samples % actual_splits] += 1

        current = 0
        for fold_idx in range(actual_splits):
            start = current
            stop = current + fold_sizes[fold_idx]

            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            current = stop
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class TestSplitterInvariants:
    """Property-based tests for cross-validation splitter invariants."""

    @given(
        n_samples=st.integers(min_value=10, max_value=1000),
        n_splits=st.integers(min_value=2, max_value=20),
        n_features=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_all_indices_used_numpy(self, n_samples, n_splits, n_features):
        """Test that all indices are used at least once across splits."""
        # Ensure n_splits doesn't exceed n_samples
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = np.random.rand(n_samples, n_features)

        all_test_indices = []
        for _train, test in splitter.split(X):
            all_test_indices.extend(test)

        # Every index should appear in test set exactly once
        assert sorted(all_test_indices) == list(range(n_samples))

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_train_test_disjoint(self, n_samples, n_splits):
        """Test that train and test sets are disjoint for each split."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = pl.DataFrame({"x": np.random.rand(n_samples)})

        for train, test in splitter.split(X):
            # No overlap between train and test
            assert len(np.intersect1d(train, test)) == 0

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_indices_in_valid_range(self, n_samples, n_splits):
        """Test that all indices are valid (0 to n_samples-1)."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = pd.DataFrame(np.random.rand(n_samples, 3))

        for train, test in splitter.split(X):
            # All indices should be in valid range
            assert np.all(train >= 0)
            assert np.all(train < n_samples)
            assert np.all(test >= 0)
            assert np.all(test < n_samples)

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_train_test_union_complete(self, n_samples, n_splits):
        """Test that union of train and test equals all indices for each split."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = np.random.rand(n_samples, 5)

        for train, test in splitter.split(X):
            # Union should give us all indices
            union = np.union1d(train, test)
            assert len(union) == n_samples
            assert np.array_equal(sorted(union), list(range(n_samples)))

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_consistent_n_splits(self, n_samples, n_splits):
        """Test that the number of splits matches get_n_splits()."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = np.random.rand(n_samples, 3)

        # Count actual splits
        actual_splits = sum(1 for _ in splitter.split(X))

        # Should match get_n_splits()
        assert actual_splits == splitter.get_n_splits(X)
        assert actual_splits == n_splits

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=20)
    def test_deterministic_splits(self, n_samples, n_splits):
        """Test that splits are deterministic for the same input."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = np.random.rand(n_samples, 3)

        # Get splits twice
        splits1 = list(splitter.split(X))
        splits2 = list(splitter.split(X))

        # Should be identical
        assert len(splits1) == len(splits2)

        for (train1, test1), (train2, test2) in zip(splits1, splits2, strict=False):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    @given(
        n_samples=st.integers(min_value=20, max_value=100),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=20)
    def test_minimum_train_test_size(self, n_samples, n_splits):
        """Test that train and test sets have minimum reasonable sizes."""
        n_splits = min(n_splits, n_samples)

        splitter = SimpleSplitter(n_splits=n_splits)
        X = np.random.rand(n_samples, 3)

        for train, test in splitter.split(X):
            # Each set should have at least 1 sample
            assert len(train) >= 1
            assert len(test) >= 1

            # Train set should typically be larger than test set
            # (this is a soft requirement, may not hold for all splitters)
            assert len(train) >= len(test) * 0.5  # At least half the size

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=1, max_value=10),
        input_type=st.sampled_from(["numpy", "pandas", "polars"]),
    )
    @settings(max_examples=30)
    def test_invariants_across_types(self, n_samples, n_features, input_type):
        """Test that invariants hold regardless of input type."""
        splitter = SimpleSplitter(n_splits=3)

        # Create data in the requested format
        data = np.random.rand(n_samples, n_features)

        if input_type == "numpy":
            X = data
        elif input_type == "pandas":
            X = pd.DataFrame(data)
        else:  # polars
            X = pl.DataFrame(data)  # type: ignore[assignment]

        # Basic invariants should hold
        all_test = []
        n_actual_splits = 0

        for train, test in splitter.split(X):
            n_actual_splits += 1
            all_test.extend(test)

            # Disjoint
            assert len(np.intersect1d(train, test)) == 0

            # Valid range
            assert np.all(train >= 0) and np.all(train < n_samples)
            assert np.all(test >= 0) and np.all(test < n_samples)

            # Complete union
            assert len(np.union1d(train, test)) == n_samples

        # All indices used
        assert sorted(all_test) == list(range(n_samples))

        # Correct number of splits
        assert n_actual_splits == splitter.get_n_splits()


class TestSplitterEdgeCases:
    """Test edge cases that might break invariants."""

    def test_minimum_samples(self):
        """Test with minimum number of samples."""
        splitter = SimpleSplitter(n_splits=2)
        X = np.array([[1], [2]])  # Just 2 samples

        splits = list(splitter.split(X))
        assert len(splits) == 2

        # Each split should have exactly 1 train and 1 test sample
        for train, test in splits:
            assert len(train) == 1
            assert len(test) == 1

    def test_n_splits_exceeds_samples_error(self):
        """Test that appropriate behavior when n_splits > n_samples."""
        # This is implementation-specific, but we test the simple splitter
        splitter = SimpleSplitter(n_splits=10)
        X = np.array([[1], [2], [3]])  # Only 3 samples

        # The simple splitter should handle this gracefully
        splits = list(splitter.split(X))

        # Should have at most 3 splits (one per sample)
        assert len(splits) <= 3
