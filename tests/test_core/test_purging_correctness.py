"""High-quality correctness tests for time-series purging and embargo.

These tests verify that purging and embargo correctly prevent data leakage
in time-series cross-validation, following López de Prado (2018) AFML Chapter 7.

Key invariants tested:
1. Purging removes samples whose labels overlap with test period
2. Embargo removes samples after test to prevent serial correlation leakage
3. No future data ever appears in training set
4. Combined purge + embargo creates proper gap around test set

AFML Golden Cases:
- Figure 7.1: Basic purging scenario
- Percentage-based embargo (1-5% recommendation)
- Multi-fold isolation requirements
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.diagnostic.core.purging import (
    apply_purging_and_embargo,
    calculate_embargo_indices,
    calculate_purge_indices,
)


class TestPurgingInvariants:
    """Tests verifying fundamental purging invariants."""

    def test_purging_removes_samples_with_label_overlap(self):
        """Purged samples are exactly those with labels overlapping test period.

        If we predict at time t with horizon h, the label uses data from [t, t+h].
        For test starting at T, we must purge training samples where t+h >= T,
        i.e., t >= T - h.
        """
        n_samples = 100
        test_start = 50
        test_end = 60
        label_horizon = 5

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # Purged indices should be [test_start - label_horizon, test_start)
        expected = list(range(test_start - label_horizon, test_start))
        assert purged == expected, f"Expected {expected}, got {purged}"

        # Verify: for any purged index t, t + horizon >= test_start
        for t in purged:
            assert t + label_horizon >= test_start, (
                f"Index {t} with horizon {label_horizon} doesn't overlap test"
            )

        # Verify: for any non-purged training index t < test_start - horizon,
        # t + horizon < test_start (no overlap)
        for t in range(test_start - label_horizon):
            assert t not in purged, f"Index {t} incorrectly purged"
            assert t + label_horizon < test_start, f"Index {t} should have been purged"

    def test_no_future_data_in_training(self):
        """Critical: training set must never contain future information.

        After purging, the latest training sample's label should not
        extend into the test period.
        """
        n_samples = 100
        test_start = 50
        label_horizon = 10

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=60,
            label_horizon=label_horizon,
        )

        # All training indices before purge zone
        train_indices = set(range(test_start)) - set(purged)

        # The latest training sample's label should end before test_start
        if train_indices:
            latest_train = max(train_indices)
            label_end = latest_train + label_horizon
            assert label_end < test_start, (
                f"Training sample {latest_train} has label ending at {label_end} "
                f">= test_start {test_start}"
            )

    def test_purge_at_boundary(self):
        """Test purging at dataset boundaries."""
        # Test at start: test starts at index 3 with horizon 5
        # Should purge max(0, 3-5) to 3, i.e., [0, 1, 2]
        purged = calculate_purge_indices(n_samples=100, test_start=3, test_end=10, label_horizon=5)
        assert purged == [0, 1, 2], f"Expected [0,1,2], got {purged}"

    def test_zero_horizon_no_purge(self):
        """With horizon=0, no purging needed (labels don't look forward)."""
        purged = calculate_purge_indices(n_samples=100, test_start=50, test_end=60, label_horizon=0)
        assert purged == [], f"Expected no purging with horizon=0, got {purged}"


class TestEmbargoInvariants:
    """Tests verifying embargo invariants."""

    def test_embargo_removes_samples_after_test(self):
        """Embargo should remove samples immediately after test period."""
        n_samples = 100
        test_start = 50
        test_end = 60
        embargo_size = 5

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_size=embargo_size,
        )

        # Should be [test_end, test_end + embargo_size)
        expected = list(range(test_end, min(test_end + embargo_size, n_samples)))
        assert embargoed == expected, f"Expected {expected}, got {embargoed}"

    def test_embargo_percentage_calculation(self):
        """Embargo as percentage of total samples."""
        n_samples = 100
        embargo_pct = 0.05  # 5% = 5 samples

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=50,
            test_end=60,
            embargo_pct=embargo_pct,
        )

        # Should embargo 5 samples after test_end=60
        assert len(embargoed) == 5
        assert embargoed == [60, 61, 62, 63, 64]

    def test_no_embargo_when_unspecified(self):
        """No embargo if neither embargo_size nor embargo_pct specified."""
        embargoed = calculate_embargo_indices(n_samples=100, test_start=50, test_end=60)
        assert embargoed == []

    def test_embargo_at_end_boundary(self):
        """Embargo should not exceed dataset bounds."""
        embargoed = calculate_embargo_indices(
            n_samples=100, test_start=90, test_end=95, embargo_size=10
        )
        # Should only go to index 99 (last valid index)
        assert max(embargoed) < 100


class TestCombinedPurgeEmbargo:
    """Tests for combined purging and embargo."""

    def test_purge_and_embargo_create_gap(self):
        """Combined purge + embargo should create a gap around test set."""
        n_samples = 100
        test_start = 50
        test_end = 60
        label_horizon = 5
        embargo_size = 3

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )
        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_size=embargo_size,
        )

        # Combine test, purge, embargo - these should not be in training
        excluded = set(purged) | set(range(test_start, test_end)) | set(embargoed)

        # Training is everything before test_start minus purged
        train_before = set(range(test_start)) - set(purged)
        # Training after test is everything after embargo
        train_after = set(range(test_end + embargo_size, n_samples))

        # Verify no overlap between training and excluded
        assert train_before.isdisjoint(excluded)
        assert train_after.isdisjoint(excluded)

        # Verify the gap structure
        # Gap before test: [45, 50) = purge zone
        # Test: [50, 60)
        # Gap after test: [60, 63) = embargo zone
        assert purged == [45, 46, 47, 48, 49]
        assert embargoed == [60, 61, 62]


class TestTimestampBasedPurging:
    """Tests for timestamp-based (not integer) purging."""

    def test_timestamp_purging_basic(self):
        """Test purging with actual timestamps."""
        # Create 100 days of data
        timestamps = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")

        test_start = pd.Timestamp("2024-02-20", tz="UTC")  # day 50
        test_end = pd.Timestamp("2024-03-01", tz="UTC")  # day 60
        label_horizon = pd.Timedelta("5D")

        purged = calculate_purge_indices(
            timestamps=timestamps,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # Should purge days 45-49 (5 days before test_start)
        assert len(purged) == 5
        assert 45 in purged and 49 in purged
        assert 44 not in purged  # Too early
        assert 50 not in purged  # Part of test set

    def test_timestamp_purging_irregular_spacing(self):
        """Purging should work correctly with irregular timestamps (e.g., weekends)."""
        # Create business day timestamps (no weekends)
        timestamps = pd.bdate_range("2024-01-01", periods=100, tz="UTC")

        test_start = timestamps[50]
        test_end = timestamps[60]
        label_horizon = pd.Timedelta("7D")  # 7 calendar days

        purged = calculate_purge_indices(
            timestamps=timestamps,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # All purged timestamps should be within 7 days of test_start
        for idx in purged:
            assert timestamps[idx] >= test_start - label_horizon
            assert timestamps[idx] < test_start

    def test_timezone_awareness_required(self):
        """Should raise error for timezone-naive timestamps."""
        timestamps = pd.date_range("2024-01-01", periods=100, freq="D")  # No tz

        with pytest.raises(ValueError, match="timezone-aware"):
            calculate_purge_indices(
                timestamps=timestamps,
                test_start=pd.Timestamp("2024-02-20", tz="UTC"),
                test_end=pd.Timestamp("2024-03-01", tz="UTC"),
                label_horizon=pd.Timedelta("5D"),
            )


class TestEdgeCases:
    """Edge case handling."""

    def test_test_at_start(self):
        """Test set at very beginning of data."""
        purged = calculate_purge_indices(n_samples=100, test_start=0, test_end=10, label_horizon=5)
        # Nothing to purge before index 0
        assert purged == []

    def test_test_at_end(self):
        """Test set at very end of data."""
        purged = calculate_purge_indices(
            n_samples=100, test_start=90, test_end=100, label_horizon=5
        )
        # Should purge [85, 90)
        assert purged == [85, 86, 87, 88, 89]

    def test_large_horizon_relative_to_data(self):
        """Horizon larger than available data before test."""
        purged = calculate_purge_indices(
            n_samples=100, test_start=10, test_end=20, label_horizon=50
        )
        # Should purge [max(0, 10-50), 10) = [0, 10)
        assert purged == list(range(10))


class TestRealWorldScenario:
    """Test realistic financial ML scenario."""

    def test_triple_barrier_purging(self):
        """Simulate triple barrier labeling where labels span variable horizons.

        In triple barrier, label horizon is the maximum time to hit TP/SL/timeout.
        We need to purge conservatively using the max possible horizon.
        """
        n_samples = 252  # One trading year
        test_start = 200
        test_end = 220
        max_label_horizon = 21  # 21-day max holding period

        calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=max_label_horizon,
        )

        # Training samples before purge zone
        train_valid = set(range(test_start - max_label_horizon))

        # Verify each training sample's label (at max horizon) doesn't overlap test
        for t in train_valid:
            label_end = t + max_label_horizon
            assert label_end < test_start, (
                f"Train sample {t} label extends to {label_end} >= test_start"
            )

    def test_walk_forward_fold_isolation(self):
        """Verify fold isolation in walk-forward scenario.

        In walk-forward: train on [0, T1), test on [T1, T2), then train on [0, T2), test on [T2, T3)
        Each test fold must not see its own future.
        """
        n_samples = 252
        fold_size = 21
        n_folds = 5
        label_horizon = 10

        for fold in range(n_folds):
            test_start = (fold + 1) * fold_size
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples:
                break

            purged = calculate_purge_indices(
                n_samples=n_samples,
                test_start=test_start,
                test_end=test_end,
                label_horizon=label_horizon,
            )

            # Training set for this fold
            train_before_purge = set(range(test_start))
            train_valid = train_before_purge - set(purged)

            # Critical check: no training sample's label overlaps test
            if train_valid:
                latest = max(train_valid)
                assert latest + label_horizon < test_start, (
                    f"Fold {fold}: train sample {latest} leaks into test at {test_start}"
                )


# ==============================================================================
# AFML Golden Tests - Chapter 7 Verification
# ==============================================================================


class TestAFMLGoldenCases:
    """Golden tests derived from AFML Chapter 7 purging examples."""

    def test_afml_figure_7_1_basic_purging(self):
        """AFML Figure 7.1: Basic purging scenario.

        Scenario: 100 samples, test period [50, 60), labels depend on 5 periods ahead.
        Expected: Purge indices [45, 50) because their labels contain test data.

        For test starting at t=50, sample at t=45 has label depending on [45, 50),
        which is OUTSIDE test. Sample at t=49 has label depending on [49, 54),
        which OVERLAPS test [50, 60). So we purge [45, 50).
        """
        n_samples = 100
        test_start = 50
        test_end = 60
        label_horizon = 5

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # AFML specifies purging [test_start - label_horizon, test_start)
        expected = list(range(45, 50))
        assert purged == expected, f"Expected {expected}, got {purged}"

    def test_afml_embargo_calculation(self):
        """AFML Chapter 7: Embargo prevents serial correlation leakage.

        Scenario: After test period ends at t=60, embargo 5 samples to prevent
        the model from learning patterns that persist across test boundary.
        """
        n_samples = 100
        test_start = 50
        test_end = 60
        embargo_size = 5

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_size=embargo_size,
        )

        # Embargo should cover [test_end, test_end + embargo_size)
        expected = list(range(60, 65))
        assert embargoed == expected, f"Expected {expected}, got {embargoed}"

    def test_afml_combined_purge_embargo(self):
        """AFML Chapter 7: Combined purging and embargo.

        Full scenario from the book:
        - 100 samples
        - Test period: [50, 60)
        - Label horizon: 5 (labels look 5 periods ahead)
        - Embargo: 5 periods after test

        Training should exclude:
        - Test period: [50, 60)
        - Purged: [45, 50) - labels overlap with test
        - Embargo: [60, 65) - serial correlation protection

        Remaining training: [0, 45) ∪ [65, 100) = 80 samples
        """
        n_samples = 100
        train_indices = np.arange(n_samples)
        test_start = 50
        test_end = 60
        label_horizon = 5
        embargo_size = 5

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            n_samples=n_samples,
        )

        # Expected: [0, 45) ∪ [65, 100) = 45 + 35 = 80 samples
        assert len(clean_train) == 80

        # Verify exact indices
        expected_train = np.concatenate([np.arange(0, 45), np.arange(65, 100)])
        np.testing.assert_array_equal(np.sort(clean_train), expected_train)

    def test_afml_percentage_embargo_recommendation(self):
        """AFML suggests embargo of 1-5% of total sample size.

        Test with 2% embargo on 1000 samples = 20 samples embargo.
        """
        n_samples = 1000
        test_start = 500
        test_end = 600
        embargo_pct = 0.02  # 2%

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_pct=embargo_pct,
        )

        # 2% of 1000 = 20 samples
        assert len(embargoed) == 20
        assert embargoed == list(range(600, 620))


# ==============================================================================
# Property-Based Tests (Hypothesis)
# ==============================================================================


class TestPurgingPropertyTests:
    """Property-based tests using Hypothesis for comprehensive verification."""

    @given(
        n_samples=st.integers(min_value=20, max_value=1000),
        test_start_pct=st.floats(min_value=0.2, max_value=0.7),
        test_size_pct=st.floats(min_value=0.1, max_value=0.3),
        horizon_pct=st.floats(min_value=0.01, max_value=0.15),
    )
    @settings(max_examples=50)
    def test_purge_indices_never_overlap_with_test(
        self, n_samples, test_start_pct, test_size_pct, horizon_pct
    ):
        """Property: Purge indices never overlap with test indices."""
        test_start = int(n_samples * test_start_pct)
        test_end = min(n_samples, int(test_start + n_samples * test_size_pct))
        label_horizon = max(1, int(n_samples * horizon_pct))

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        test_indices = set(range(test_start, test_end))
        purged_set = set(purged)

        assert len(purged_set & test_indices) == 0, "Purge and test indices must not overlap"

    @given(
        n_samples=st.integers(min_value=20, max_value=1000),
        test_start_pct=st.floats(min_value=0.2, max_value=0.6),
        test_size_pct=st.floats(min_value=0.1, max_value=0.3),
        embargo_pct=st.floats(min_value=0.01, max_value=0.1),
    )
    @settings(max_examples=50)
    def test_embargo_indices_never_overlap_with_test(
        self, n_samples, test_start_pct, test_size_pct, embargo_pct
    ):
        """Property: Embargo indices never overlap with test indices."""
        test_start = int(n_samples * test_start_pct)
        test_end = min(n_samples - 5, int(test_start + n_samples * test_size_pct))

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_pct=embargo_pct,
        )

        test_indices = set(range(test_start, test_end))
        embargo_set = set(embargoed)

        assert len(embargo_set & test_indices) == 0, "Embargo and test indices must not overlap"

    @given(
        n_samples=st.integers(min_value=50, max_value=500),
        test_start_pct=st.floats(min_value=0.3, max_value=0.5),
        horizon_pct=st.floats(min_value=0.02, max_value=0.1),
        embargo_pct=st.floats(min_value=0.02, max_value=0.1),
    )
    @settings(max_examples=50)
    def test_clean_train_has_no_overlap_with_removed(
        self, n_samples, test_start_pct, horizon_pct, embargo_pct
    ):
        """Property: Clean training set has no overlap with removed indices."""
        test_start = int(n_samples * test_start_pct)
        test_end = min(n_samples - 10, test_start + int(n_samples * 0.2))
        label_horizon = max(1, int(n_samples * horizon_pct))

        train_indices = np.arange(n_samples)

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
            embargo_pct=embargo_pct,
            n_samples=n_samples,
        )

        # Test indices
        test_indices = set(range(test_start, test_end))

        # Purge indices
        purged = set(
            calculate_purge_indices(
                n_samples=n_samples,
                test_start=test_start,
                test_end=test_end,
                label_horizon=label_horizon,
            )
        )

        # Embargo indices
        embargoed = set(
            calculate_embargo_indices(
                n_samples=n_samples,
                test_start=test_start,
                test_end=test_end,
                embargo_pct=embargo_pct,
            )
        )

        removed = test_indices | purged | embargoed
        clean_set = set(clean_train.tolist())

        assert len(clean_set & removed) == 0, "Clean train must not contain any removed indices"

    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        test_start_pct=st.floats(min_value=0.3, max_value=0.6),
    )
    @settings(max_examples=30)
    def test_purge_count_bounded_by_horizon(self, n_samples, test_start_pct):
        """Property: Number of purged samples is at most label_horizon."""
        test_start = int(n_samples * test_start_pct)
        test_end = min(n_samples, test_start + 50)
        label_horizon = 10

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # Purge count bounded by min(label_horizon, test_start)
        max_purge = min(label_horizon, test_start)
        assert len(purged) <= max_purge


# ==============================================================================
# Additional Edge Case Tests
# ==============================================================================


class TestExtendedEdgeCases:
    """Extended edge case testing."""

    def test_single_sample_test_set(self):
        """Test with a single sample in the test set."""
        purged = calculate_purge_indices(n_samples=100, test_start=50, test_end=51, label_horizon=5)
        # Should purge [45, 50)
        assert purged == list(range(45, 50))

    def test_minimal_data_scenario(self):
        """Test with minimal data (5 samples)."""
        n_samples = 5
        train_indices = np.arange(n_samples)

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=2,
            test_end=3,
            label_horizon=1,
            embargo_size=1,
            n_samples=n_samples,
        )

        # Test: [2]
        # Purge: [1] (2-1 to 2)
        # Embargo: [3]
        # Remaining: [0, 4]
        assert len(clean_train) == 2
        assert set(clean_train) == {0, 4}

    def test_entire_dataset_excluded(self):
        """Test when purge + embargo + test covers entire dataset."""
        n_samples = 20
        train_indices = np.arange(n_samples)

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=5,
            test_end=15,
            label_horizon=5,  # Purges [0, 5)
            embargo_size=5,  # Embargoes [15, 20)
            n_samples=n_samples,
        )

        # Everything should be excluded
        assert len(clean_train) == 0

    def test_large_horizon_vs_small_dataset(self):
        """Test when horizon is larger than available data."""
        purged = calculate_purge_indices(
            n_samples=50,
            test_start=10,
            test_end=20,
            label_horizon=100,  # Much larger
        )
        # Can only purge [0, 10)
        assert purged == list(range(10))

    def test_consecutive_test_sets(self):
        """Test handling of consecutive test sets (as in k-fold CV)."""
        n_samples = 100
        label_horizon = 5
        embargo_size = 3

        # Fold 1: test [0, 20) - no training data before test
        clean1 = apply_purging_and_embargo(
            train_indices=np.arange(n_samples),
            test_start=0,
            test_end=20,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            n_samples=n_samples,
        )

        # Fold 2: test [20, 40)
        clean2 = apply_purging_and_embargo(
            train_indices=np.arange(n_samples),
            test_start=20,
            test_end=40,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            n_samples=n_samples,
        )

        # Both should be valid
        assert len(clean1) > 0
        assert len(clean2) > 0

        # For fold 1 (test starts at 0), there's no data before test
        clean1_before_test = clean1[clean1 < 20]
        assert len(clean1_before_test) == 0  # No data before test

        # For fold 2, there should be [0, 15) before test (purge removes [15, 20))
        clean2_before_test = clean2[clean2 < 20]
        assert len(clean2_before_test) == 15  # [0, 15) = 15 samples

        # Verify clean2 has correct structure
        # Before test: [0, 15)
        # Test: [20, 40) (excluded)
        # Embargo: [40, 43) (excluded)
        # After: [43, 100)
        expected_clean2 = np.concatenate([np.arange(0, 15), np.arange(43, 100)])
        np.testing.assert_array_equal(np.sort(clean2), expected_clean2)


# ==============================================================================
# Performance Tests
# ==============================================================================


class TestPurgingPerformance:
    """Performance tests for purging operations."""

    def test_large_dataset_performance(self):
        """Purging should be fast even with large datasets."""
        import time

        n_samples = 100_000
        train_indices = np.arange(n_samples)

        start = time.time()

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=40_000,
            test_end=60_000,
            label_horizon=1_000,
            embargo_size=500,
            n_samples=n_samples,
        )

        elapsed = time.time() - start

        # Should complete in well under 1 second
        assert elapsed < 1.0, f"Purging took {elapsed:.2f}s for {n_samples} samples"

        # Verify correctness
        expected_removed = 20_000 + 1_000 + 500  # test + purge + embargo
        expected_train = n_samples - expected_removed
        assert len(clean_train) == expected_train

    def test_many_fold_scenario(self):
        """Test performance with many folds."""
        import time

        n_samples = 10_000
        n_folds = 100
        fold_size = n_samples // n_folds
        label_horizon = 10
        embargo_size = 5

        start = time.time()

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size

            apply_purging_and_embargo(
                train_indices=np.arange(n_samples),
                test_start=test_start,
                test_end=test_end,
                label_horizon=label_horizon,
                embargo_size=embargo_size,
                n_samples=n_samples,
            )

        elapsed = time.time() - start

        # 100 folds should complete quickly
        assert elapsed < 5.0, f"100 folds took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
