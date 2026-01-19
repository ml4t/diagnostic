"""Tests for group isolation utilities."""

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.splitters.group_isolation import (
    count_samples_per_group,
    get_group_boundaries,
    isolate_groups_from_train,
    split_by_groups,
    validate_group_isolation,
)


class TestValidateGroupIsolation:
    """Tests for validate_group_isolation function."""

    def test_valid_isolation_numpy(self):
        """Test that non-overlapping groups are detected as valid."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5, 6, 7])
        groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])

        is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)

        assert is_valid
        assert len(overlap) == 0

    def test_invalid_isolation_numpy(self):
        """Test that overlapping groups are detected."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([2, 3, 4, 5])  # Overlaps with train
        groups = np.array(["A", "A", "B", "B", "C", "C"])

        is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)

        assert not is_valid
        assert overlap == {"B"}

    def test_with_pandas_series(self):
        """Test with pandas Series groups."""
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])
        groups = pd.Series(["A", "A", "B", "B"])

        is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)

        assert is_valid
        assert len(overlap) == 0

    def test_with_polars_series(self):
        """Test with polars Series groups."""
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])
        groups = pl.Series(["A", "A", "B", "B"])

        is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)

        assert is_valid
        assert len(overlap) == 0

    def test_multiple_overlapping_groups(self):
        """Test detection of multiple overlapping groups."""
        train_idx = np.array([0, 1, 2, 3, 4, 5])
        test_idx = np.array([2, 3, 6, 7])
        groups = np.array(["A", "A", "B", "B", "C", "C", "B", "B"])

        is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)

        assert not is_valid
        assert overlap == {"B"}


class TestIsolateGroupsFromTrain:
    """Tests for isolate_groups_from_train function."""

    def test_basic_isolation(self):
        """Test basic group isolation from training set."""
        train_idx = np.array([0, 1, 2, 3, 4, 5])
        test_idx = np.array([6, 7])
        groups = np.array(["A", "A", "B", "B", "C", "C", "C", "C"])

        clean_train = isolate_groups_from_train(train_idx, test_idx, groups)

        # Should remove indices 4,5 (group C)
        assert len(clean_train) == 4
        assert set(clean_train) == {0, 1, 2, 3}
        assert all(groups[clean_train] != "C")

    def test_no_overlap_no_removal(self):
        """Test that no samples are removed when groups don't overlap."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        groups = np.array(["A", "A", "B", "B", "C", "C"])

        clean_train = isolate_groups_from_train(train_idx, test_idx, groups)

        # Should keep all training samples
        assert len(clean_train) == len(train_idx)
        assert set(clean_train) == set(train_idx)

    def test_complete_overlap_removal(self):
        """Test when all training samples must be removed."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        groups = np.array(["A", "A", "A", "A", "A", "A"])

        clean_train = isolate_groups_from_train(train_idx, test_idx, groups)

        # Should remove all training samples
        assert len(clean_train) == 0

    def test_with_pandas_series(self):
        """Test isolation with pandas Series."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        groups = pd.Series(["A", "A", "B", "B", "B", "B"])

        clean_train = isolate_groups_from_train(train_idx, test_idx, groups)

        # Should remove indices 2,3 (group B)
        assert len(clean_train) == 2
        assert set(clean_train) == {0, 1}

    def test_with_polars_series(self):
        """Test isolation with polars Series."""
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        groups = pl.Series(["A", "A", "B", "B", "B", "B"])

        clean_train = isolate_groups_from_train(train_idx, test_idx, groups)

        # Should remove indices 2,3 (group B)
        assert len(clean_train) == 2
        assert set(clean_train) == {0, 1}


class TestGetGroupBoundaries:
    """Tests for get_group_boundaries function."""

    def test_contiguous_groups(self):
        """Test with contiguous groups."""
        groups = np.array(["A", "A", "A", "B", "B", "C"])

        boundaries = get_group_boundaries(groups)

        assert boundaries["A"] == (0, 3)
        assert boundaries["B"] == (3, 5)
        assert boundaries["C"] == (5, 6)

    def test_single_sample_groups(self):
        """Test with single-sample groups."""
        groups = np.array(["A", "B", "C"])

        boundaries = get_group_boundaries(groups)

        assert boundaries["A"] == (0, 1)
        assert boundaries["B"] == (1, 2)
        assert boundaries["C"] == (2, 3)

    def test_with_sorted_indices(self):
        """Test with explicit sorted indices."""
        groups = np.array(["B", "A", "C", "A", "B", "C"])
        sorted_idx = np.array([1, 3, 0, 4, 2, 5])  # Sort to: A,A,B,B,C,C

        boundaries = get_group_boundaries(groups, sorted_indices=sorted_idx)

        assert boundaries["A"] == (0, 2)
        assert boundaries["B"] == (2, 4)
        assert boundaries["C"] == (4, 6)

    def test_with_pandas_series(self):
        """Test with pandas Series."""
        groups = pd.Series(["A", "A", "B", "B"])

        boundaries = get_group_boundaries(groups)

        assert boundaries["A"] == (0, 2)
        assert boundaries["B"] == (2, 4)

    def test_with_polars_series(self):
        """Test with polars Series."""
        groups = pl.Series(["A", "A", "B", "B"])

        boundaries = get_group_boundaries(groups)

        assert boundaries["A"] == (0, 2)
        assert boundaries["B"] == (2, 4)


class TestSplitByGroups:
    """Tests for split_by_groups function."""

    def test_basic_split(self):
        """Test basic group-based splitting."""
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        all_groups = ["A", "B", "C"]

        train_idx, test_idx = split_by_groups(
            n_samples=6,
            groups=groups,
            test_group_indices=[2],  # Group 'C'
            all_group_ids=all_groups,
        )

        assert set(groups[train_idx]) == {"A", "B"}
        assert set(groups[test_idx]) == {"C"}
        assert len(train_idx) == 4
        assert len(test_idx) == 2

    def test_multiple_test_groups(self):
        """Test with multiple groups in test set."""
        groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
        all_groups = ["A", "B", "C", "D"]

        train_idx, test_idx = split_by_groups(
            n_samples=8,
            groups=groups,
            test_group_indices=[1, 3],  # Groups 'B' and 'D'
            all_group_ids=all_groups,
        )

        assert set(groups[train_idx]) == {"A", "C"}
        assert set(groups[test_idx]) == {"B", "D"}
        assert len(train_idx) == 4
        assert len(test_idx) == 4

    def test_all_groups_in_test(self):
        """Test when all groups go to test set."""
        groups = np.array(["A", "A", "B", "B"])
        all_groups = ["A", "B"]

        train_idx, test_idx = split_by_groups(
            n_samples=4,
            groups=groups,
            test_group_indices=[0, 1],  # All groups
            all_group_ids=all_groups,
        )

        assert len(train_idx) == 0
        assert len(test_idx) == 4

    def test_no_groups_in_test(self):
        """Test when no groups go to test set."""
        groups = np.array(["A", "A", "B", "B"])
        all_groups = ["A", "B"]

        train_idx, test_idx = split_by_groups(
            n_samples=4, groups=groups, test_group_indices=[], all_group_ids=all_groups
        )

        assert len(train_idx) == 4
        assert len(test_idx) == 0

    def test_with_pandas_series(self):
        """Test with pandas Series."""
        groups = pd.Series(["A", "A", "B", "B"])
        all_groups = ["A", "B"]

        train_idx, test_idx = split_by_groups(
            n_samples=4,
            groups=groups,
            test_group_indices=[1],  # Group 'B'
            all_group_ids=all_groups,
        )

        assert set(groups[train_idx]) == {"A"}
        assert set(groups[test_idx]) == {"B"}

    def test_with_polars_series(self):
        """Test with polars Series."""
        groups = pl.Series(["A", "A", "B", "B"])
        all_groups = ["A", "B"]

        train_idx, test_idx = split_by_groups(
            n_samples=4,
            groups=groups,
            test_group_indices=[1],  # Group 'B'
            all_group_ids=all_groups,
        )

        assert set(groups[train_idx]) == {"A"}
        assert set(groups[test_idx]) == {"B"}


class TestCountSamplesPerGroup:
    """Tests for count_samples_per_group function."""

    def test_basic_counting(self):
        """Test basic sample counting per group."""
        groups = np.array(["A", "A", "A", "B", "B", "C"])

        counts = count_samples_per_group(groups)

        assert counts == {"A": 3, "B": 2, "C": 1}

    def test_uniform_groups(self):
        """Test with uniform group sizes."""
        groups = np.array(["A", "A", "B", "B", "C", "C"])

        counts = count_samples_per_group(groups)

        assert counts == {"A": 2, "B": 2, "C": 2}

    def test_single_group(self):
        """Test with single group."""
        groups = np.array(["A", "A", "A", "A"])

        counts = count_samples_per_group(groups)

        assert counts == {"A": 4}

    def test_with_pandas_series(self):
        """Test with pandas Series."""
        groups = pd.Series(["A", "A", "B"])

        counts = count_samples_per_group(groups)

        assert counts == {"A": 2, "B": 1}

    def test_with_polars_series(self):
        """Test with polars Series."""
        groups = pl.Series(["A", "A", "B"])

        counts = count_samples_per_group(groups)

        assert counts == {"A": 2, "B": 1}

    def test_numeric_groups(self):
        """Test with numeric group identifiers."""
        groups = np.array([1, 1, 2, 2, 2, 3])

        counts = count_samples_per_group(groups)

        assert counts == {1: 2, 2: 3, 3: 1}
