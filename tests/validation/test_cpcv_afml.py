"""Golden tests for Combinatorial Purged Cross-Validation (CPCV) against AFML Chapter 7.

These tests verify the mathematical properties of CPCV as described in
López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 7.

Key Properties (AFML Chapter 7):
================================

1. Number of Combinations: C(N, k) = N! / (k! * (N-k)!)
   The number of ways to choose k test groups from N total groups.

2. Test Group Appearance: Each group appears in the test set exactly C(N-1, k-1) times.
   This is because we fix one group in test and choose the remaining k-1 from N-1.

3. Training Group Appearance: Each group appears in training exactly C(N-1, k) times.
   This is C(N, k) - C(N-1, k-1) = C(N-1, k).

4. Backtest Symmetry: All groups participate equally in testing.
   Total test slots = C(N, k) * k = N * C(N-1, k-1), divided equally among N groups.

5. Expected Number of Backtests per Observation: φ[N,k] = k * C(N, k) / N = C(N-1, k-1)
   Each observation appears in the test set this many times on average.

References:
-----------
- López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7.
- Bailey, D., Borwein, J., López de Prado, M., and Zhu, Q. (2017).
  "The Probability of Backtest Overfitting". Journal of Computational Finance.
"""

import math
from collections import Counter

import numpy as np
import pytest

from ml4t.diagnostic.splitters import CombinatorialPurgedCV


class TestCPCVAFMLFormulas:
    """Golden tests for CPCV mathematical properties from AFML Chapter 7."""

    # Test cases: (n_groups, n_test_groups)
    AFML_TEST_CASES = [
        (5, 1),  # Simple case: C(5,1) = 5
        (5, 2),  # AFML example: C(5,2) = 10
        (6, 2),  # Common case: C(6,2) = 15
        (8, 2),  # C(8,2) = 28
        (6, 3),  # C(6,3) = 20
        (10, 3),  # Large: C(10,3) = 120
        (8, 4),  # Symmetric: C(8,4) = 70
    ]

    @pytest.mark.parametrize("n_groups,n_test_groups", AFML_TEST_CASES)
    def test_number_of_combinations(self, n_groups: int, n_test_groups: int):
        """Verify C(N, k) combinations are generated.

        AFML Chapter 7, Section 7.4.1:
        "The number of paths is the binomial coefficient C(N, k)."
        """
        expected_combinations = math.comb(n_groups, n_test_groups)

        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,  # No purging for mathematical properties
            embargo_size=0,  # No embargo for mathematical properties
        )

        assert cv.get_n_splits() == expected_combinations

        # Verify actual splits
        X = np.arange(n_groups * 50).reshape(-1, 1)  # Enough samples
        splits = list(cv.split(X))
        assert len(splits) == expected_combinations

    @pytest.mark.parametrize("n_groups,n_test_groups", AFML_TEST_CASES)
    def test_test_group_appearance_count(self, n_groups: int, n_test_groups: int):
        """Each group appears in test set exactly C(N-1, k-1) times.

        AFML Chapter 7, Section 7.4.1:
        "Each of the N groups appears in the test set exactly C(N-1, k-1) times."

        Derivation: Fix group i in test. Need to choose remaining k-1 test groups
        from the other N-1 groups. Number of ways = C(N-1, k-1).
        """
        expected_test_appearances = math.comb(n_groups - 1, n_test_groups - 1)

        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,
            embargo_size=0,
        )

        # Track which groups appear in each test set
        X = np.arange(n_groups * 100).reshape(-1, 1)
        group_size = len(X) // n_groups

        group_test_counts = Counter()

        for _train_idx, test_idx in cv.split(X):
            # Determine which groups are in test
            for g in range(n_groups):
                group_start = g * group_size
                group_end = (g + 1) * group_size if g < n_groups - 1 else len(X)
                group_indices = set(range(group_start, group_end))

                # Check if this group is in test
                if group_indices & set(test_idx):
                    group_test_counts[g] += 1

        # Each group should appear exactly C(N-1, k-1) times
        for g in range(n_groups):
            assert group_test_counts[g] == expected_test_appearances, (
                f"Group {g} appeared {group_test_counts[g]} times, expected {expected_test_appearances}"
            )

    @pytest.mark.parametrize("n_groups,n_test_groups", AFML_TEST_CASES)
    def test_training_group_appearance_count(self, n_groups: int, n_test_groups: int):
        """Each group appears in training set exactly C(N-1, k) times.

        Derivation: Group i is in training when NOT in test.
        Combinations where i is NOT in test = C(N, k) - C(N-1, k-1) = C(N-1, k)
        """
        expected_train_appearances = math.comb(n_groups - 1, n_test_groups)

        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,
            embargo_size=0,
        )

        X = np.arange(n_groups * 100).reshape(-1, 1)
        group_size = len(X) // n_groups

        group_train_counts = Counter()

        for train_idx, _test_idx in cv.split(X):
            for g in range(n_groups):
                group_start = g * group_size
                group_end = (g + 1) * group_size if g < n_groups - 1 else len(X)
                group_indices = set(range(group_start, group_end))

                # Check if this group is in train (has any overlap)
                if group_indices & set(train_idx):
                    group_train_counts[g] += 1

        # Each group should appear exactly C(N-1, k) times in training
        for g in range(n_groups):
            assert group_train_counts[g] == expected_train_appearances, (
                f"Group {g} in training {group_train_counts[g]} times, expected {expected_train_appearances}"
            )

    @pytest.mark.parametrize("n_groups,n_test_groups", AFML_TEST_CASES)
    def test_total_test_slots(self, n_groups: int, n_test_groups: int):
        """Total test slots = C(N, k) * k = N * C(N-1, k-1).

        AFML Chapter 7, Section 7.4.1:
        "The total number of test slots across all paths is C(N, k) * k."

        Equivalently: N * C(N-1, k-1) (each of N groups appears C(N-1, k-1) times).
        """
        n_combinations = math.comb(n_groups, n_test_groups)
        expected_total_slots = n_combinations * n_test_groups

        # Verify identity: C(N, k) * k = N * C(N-1, k-1)
        alternative = n_groups * math.comb(n_groups - 1, n_test_groups - 1)
        assert expected_total_slots == alternative

        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,
            embargo_size=0,
        )

        X = np.arange(n_groups * 100).reshape(-1, 1)
        group_size = len(X) // n_groups

        total_test_slots = 0
        for _train_idx, test_idx in cv.split(X):
            # Count how many groups are in test
            groups_in_test = 0
            for g in range(n_groups):
                group_start = g * group_size
                group_end = (g + 1) * group_size if g < n_groups - 1 else len(X)
                group_indices = set(range(group_start, group_end))
                if group_indices & set(test_idx):
                    groups_in_test += 1
            total_test_slots += groups_in_test

        assert total_test_slots == expected_total_slots

    @pytest.mark.parametrize("n_groups,n_test_groups", AFML_TEST_CASES)
    def test_backtest_symmetry(self, n_groups: int, n_test_groups: int):
        """All groups participate equally in testing (symmetry property).

        AFML Chapter 7 emphasizes that CPCV treats all time periods equally,
        unlike walk-forward CV which has early periods tested less frequently.
        """
        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,
            embargo_size=0,
        )

        X = np.arange(n_groups * 100).reshape(-1, 1)
        group_size = len(X) // n_groups

        group_test_counts = Counter()

        for _train_idx, test_idx in cv.split(X):
            for g in range(n_groups):
                group_start = g * group_size
                group_end = (g + 1) * group_size if g < n_groups - 1 else len(X)
                group_indices = set(range(group_start, group_end))
                if group_indices & set(test_idx):
                    group_test_counts[g] += 1

        # All groups should have the same count (symmetry)
        counts = list(group_test_counts.values())
        assert len(set(counts)) == 1, f"Asymmetric test counts: {group_test_counts}"

    @pytest.mark.parametrize(
        "n_groups,n_test_groups",
        [
            (5, 2),
            (6, 2),
            (6, 3),
        ],
    )
    def test_complementary_property(self, n_groups: int, n_test_groups: int):
        """Verify C(N, k) = C(N, N-k) symmetry.

        Mathematical property: The number of ways to choose k items equals
        the number of ways to choose the remaining N-k items.
        """
        n_train_groups = n_groups - n_test_groups

        expected_combinations = math.comb(n_groups, n_test_groups)
        complement = math.comb(n_groups, n_train_groups)

        assert expected_combinations == complement


class TestCPCVPurgingProperties:
    """Test purging properties that don't affect combination counts."""

    def test_purging_reduces_training_size(self):
        """Purging should remove samples from training, not change combinations.

        AFML Chapter 7.4.2: Purging removes samples within label_horizon of test set.

        Note: Not every split will have reduced training - when test groups are at
        the data boundary (groups 0-1), there's nothing before them to purge.
        We verify that TOTAL training size across all splits is smaller.
        """
        n_samples = 600
        X = np.arange(n_samples).reshape(-1, 1)

        cv_no_purge = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, label_horizon=0, embargo_size=0
        )
        cv_with_purge = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, label_horizon=10, embargo_size=0
        )

        # Same number of combinations
        assert cv_no_purge.get_n_splits() == cv_with_purge.get_n_splits()

        # Count total training samples across all splits
        total_train_no_purge = sum(len(t) for t, _ in cv_no_purge.split(X))
        total_train_with_purge = sum(len(t) for t, _ in cv_with_purge.split(X))

        # Total training should be smaller with purging
        assert total_train_with_purge < total_train_no_purge, (
            f"Expected less training with purging: {total_train_with_purge} < {total_train_no_purge}"
        )

        # At least some splits should have reduced training
        reduced_count = 0
        for (train_np, _), (train_p, _) in zip(cv_no_purge.split(X), cv_with_purge.split(X)):
            if len(train_p) < len(train_np):
                reduced_count += 1

        assert reduced_count > 0, "At least some splits should have reduced training"

    def test_embargo_reduces_training_size(self):
        """Embargo should remove samples after test set.

        AFML Chapter 7.4.2: Embargo removes samples within embargo_size after test.

        Note: Not every split will have reduced training - when test groups are at
        the data end (groups 4-5), there's nothing after them to embargo.
        We verify that TOTAL training size across all splits is smaller.
        """
        n_samples = 600
        X = np.arange(n_samples).reshape(-1, 1)

        cv_no_embargo = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, label_horizon=0, embargo_size=0
        )
        cv_with_embargo = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, label_horizon=0, embargo_size=10
        )

        # Same number of combinations
        assert cv_no_embargo.get_n_splits() == cv_with_embargo.get_n_splits()

        # Count total training samples across all splits
        total_train_no_embargo = sum(len(t) for t, _ in cv_no_embargo.split(X))
        total_train_with_embargo = sum(len(t) for t, _ in cv_with_embargo.split(X))

        # Total training should be smaller with embargo
        assert total_train_with_embargo < total_train_no_embargo, (
            f"Expected less training with embargo: {total_train_with_embargo} < {total_train_no_embargo}"
        )

        # At least some splits should have reduced training
        reduced_count = 0
        for (train_ne, _), (train_e, _) in zip(cv_no_embargo.split(X), cv_with_embargo.split(X)):
            if len(train_e) < len(train_ne):
                reduced_count += 1

        assert reduced_count > 0, "At least some splits should have reduced training"


class TestCPCVSpecificExamples:
    """Test specific examples from AFML for verification."""

    def test_afml_example_5_groups_2_test(self):
        """Verify the N=5, k=2 example from AFML Chapter 7.

        With 5 groups and 2 test groups:
        - C(5, 2) = 10 combinations
        - Each group in test C(4, 1) = 4 times
        - Each group in train C(4, 2) = 6 times
        """
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)

        # 10 combinations
        assert cv.get_n_splits() == 10

        # Each group in test 4 times
        X = np.arange(500).reshape(-1, 1)
        group_test_counts = Counter()
        group_size = 100  # 500 / 5

        for _train_idx, test_idx in cv.split(X):
            for g in range(5):
                group_start = g * group_size
                group_end = group_start + group_size
                if set(range(group_start, group_end)) & set(test_idx):
                    group_test_counts[g] += 1

        assert all(c == 4 for c in group_test_counts.values())

    def test_afml_example_6_groups_2_test(self):
        """Verify N=6, k=2: common example in financial backtesting.

        With 6 groups and 2 test groups:
        - C(6, 2) = 15 combinations
        - Each group in test C(5, 1) = 5 times
        - Each group in train C(5, 2) = 10 times
        """
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)

        assert cv.get_n_splits() == 15

        X = np.arange(600).reshape(-1, 1)
        group_size = 100

        group_test_counts = Counter()
        group_train_counts = Counter()

        for train_idx, test_idx in cv.split(X):
            for g in range(6):
                group_start = g * group_size
                group_end = group_start + group_size
                group_indices = set(range(group_start, group_end))

                if group_indices & set(test_idx):
                    group_test_counts[g] += 1
                if group_indices & set(train_idx):
                    group_train_counts[g] += 1

        # Each group in test 5 times
        assert all(c == 5 for c in group_test_counts.values())
        # Each group in train 10 times
        assert all(c == 10 for c in group_train_counts.values())

    def test_symmetric_case_n8_k4(self):
        """Test symmetric case where k = N/2.

        With N=8, k=4:
        - C(8, 4) = 70 combinations
        - Each group in test C(7, 3) = 35 times (half of combinations)
        - Each group in train C(7, 4) = 35 times (also half)

        This is the most "balanced" CPCV configuration.
        """
        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=4)

        assert cv.get_n_splits() == 70

        # In symmetric case, each group is in test exactly half the time
        X = np.arange(800).reshape(-1, 1)
        group_size = 100

        group_test_counts = Counter()

        for _train_idx, test_idx in cv.split(X):
            for g in range(8):
                group_start = g * group_size
                group_end = group_start + group_size
                if set(range(group_start, group_end)) & set(test_idx):
                    group_test_counts[g] += 1

        # Each group appears in test 35 times (C(7,3) = 35)
        assert all(c == 35 for c in group_test_counts.values())


class TestCPCVExpectedBacktestsPhi:
    """Test the φ[N,k] expected number of backtests formula.

    From AFML: φ[N,k] = expected number of times each observation appears in test
              = k * C(N, k) / N = C(N-1, k-1)
    """

    @pytest.mark.parametrize(
        "n_groups,n_test_groups",
        [
            (5, 1),
            (5, 2),
            (6, 2),
            (6, 3),
            (8, 2),
            (10, 3),
        ],
    )
    def test_phi_formula(self, n_groups: int, n_test_groups: int):
        """Verify φ[N,k] = C(N-1, k-1).

        This formula gives the expected number of times each observation
        appears in the test set across all CPCV combinations.
        """
        phi = math.comb(n_groups - 1, n_test_groups - 1)

        # Alternative formula: k * C(N, k) / N
        n_combinations = math.comb(n_groups, n_test_groups)
        phi_alt = n_test_groups * n_combinations / n_groups

        assert phi == phi_alt

        # Verify empirically
        cv = CombinatorialPurgedCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            label_horizon=0,
            embargo_size=0,
        )

        # Use samples that divide evenly
        n_samples = n_groups * 100
        X = np.arange(n_samples).reshape(-1, 1)

        # Count test appearances for each sample
        sample_test_counts = np.zeros(n_samples)

        for _train_idx, test_idx in cv.split(X):
            sample_test_counts[test_idx] += 1

        # Each sample should appear exactly phi times
        # (with perfectly divisible groups)
        unique_counts = np.unique(sample_test_counts)

        # All samples appear phi times (since groups are equal size)
        assert np.all(sample_test_counts == phi), (
            f"Expected all samples to appear {phi} times, got {unique_counts}"
        )
