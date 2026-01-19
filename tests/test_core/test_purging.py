"""Tests for purging and embargo functionality."""

import numpy as np
import pandas as pd

from ml4t.diagnostic.core.purging import (
    apply_purging_and_embargo,
    calculate_embargo_indices,
    calculate_purge_indices,
)


class TestPurging:
    """Test suite for purging functionality."""

    def test_basic_purging_with_integer_indices(self):
        """Test purging with simple integer indices."""
        n_samples = 100
        test_start = 50
        test_end = 60
        label_horizon = 5  # Labels depend on 5 periods ahead

        # Calculate indices to purge
        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        # Should purge [45, 50) - samples whose labels overlap with test
        expected_purged = list(range(45, 50))
        assert purged == expected_purged

    def test_purging_at_boundaries(self):
        """Test purging behavior at data boundaries."""
        n_samples = 50

        # Test at start boundary
        purged_start = calculate_purge_indices(
            n_samples=n_samples,
            test_start=0,
            test_end=10,
            label_horizon=5,
        )
        # No samples before 0 to purge
        assert purged_start == []

        # Test at end boundary
        purged_end = calculate_purge_indices(
            n_samples=n_samples,
            test_start=45,
            test_end=50,
            label_horizon=5,
        )
        # Should purge [40, 45)
        assert purged_end == list(range(40, 45))

    def test_purging_with_timestamps(self):
        """Test purging with datetime indices."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start_time = timestamps[50]
        test_end_time = timestamps[60]
        label_horizon = pd.Timedelta("5D")

        purged = calculate_purge_indices(
            timestamps=timestamps,
            test_start=test_start_time,
            test_end=test_end_time,
            label_horizon=label_horizon,
        )

        # Should purge 5 days before test start
        expected_purged = list(range(45, 50))
        assert purged == expected_purged

    def test_no_purging_when_no_overlap(self):
        """Test that no purging occurs when there's no overlap."""
        n_samples = 100
        test_start = 50
        test_end = 60
        label_horizon = 0  # No forward dependency

        purged = calculate_purge_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
        )

        assert purged == []


class TestEmbargo:
    """Test suite for embargo functionality."""

    def test_basic_embargo_with_integer_indices(self):
        """Test embargo with simple integer indices."""
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

        # Should embargo [60, 65)
        expected_embargo = list(range(60, 65))
        assert embargoed == expected_embargo

    def test_embargo_at_end_boundary(self):
        """Test embargo behavior at data end."""
        n_samples = 100
        test_start = 90
        test_end = 95
        embargo_size = 10  # Would go beyond data

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_size=embargo_size,
        )

        # Should only embargo up to end of data
        expected_embargo = list(range(95, 100))
        assert embargoed == expected_embargo

    def test_percentage_embargo(self):
        """Test embargo specified as percentage."""
        n_samples = 100
        test_start = 50
        test_end = 60
        embargo_pct = 0.05  # 5% of data

        embargoed = calculate_embargo_indices(
            n_samples=n_samples,
            test_start=test_start,
            test_end=test_end,
            embargo_pct=embargo_pct,
        )

        # 5% of 100 = 5 samples
        expected_embargo = list(range(60, 65))
        assert embargoed == expected_embargo

    def test_embargo_with_timestamps(self):
        """Test embargo with datetime indices."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start_time = timestamps[50]
        test_end_time = timestamps[60]
        embargo_size = pd.Timedelta("5D")

        embargoed = calculate_embargo_indices(
            timestamps=timestamps,
            test_start=test_start_time,
            test_end=test_end_time,
            embargo_size=embargo_size,
        )

        # Should embargo 5 days after test end
        expected_embargo = list(range(60, 65))
        assert embargoed == expected_embargo


class TestCombinedPurgingAndEmbargo:
    """Test suite for combined purging and embargo."""

    def test_apply_purging_and_embargo(self):
        """Test applying both purging and embargo to indices."""
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

        # Should remove:
        # - Test indices [50, 60)
        # - Purged indices [45, 50)
        # - Embargo indices [60, 65)
        # Remaining: [0, 45) and [65, 100)
        expected_train = np.concatenate([np.arange(0, 45), np.arange(65, 100)])

        np.testing.assert_array_equal(clean_train, expected_train)

    def test_no_training_data_edge_case(self):
        """Test when purging and embargo remove all training data."""
        n_samples = 20
        train_indices = np.arange(n_samples)
        test_start = 5
        test_end = 15
        label_horizon = 10  # Large horizon
        embargo_size = 10  # Large embargo

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start,
            test_end=test_end,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            n_samples=n_samples,
        )

        # All data should be removed
        assert len(clean_train) == 0

    def test_multi_fold_scenario(self):
        """Test realistic scenario with multiple folds."""
        n_samples = 100

        # Simulate 3 folds
        folds = [
            (20, 40),  # Fold 1: test [20, 40)
            (40, 60),  # Fold 2: test [40, 60)
            (60, 80),  # Fold 3: test [60, 80)
        ]

        label_horizon = 5
        embargo_size = 3

        for _i, (test_start, test_end) in enumerate(folds):
            # Get all indices except current test fold
            train_indices = np.concatenate(
                [np.arange(0, test_start), np.arange(test_end, n_samples)],
            )

            clean_train = apply_purging_and_embargo(
                train_indices=train_indices,
                test_start=test_start,
                test_end=test_end,
                label_horizon=label_horizon,
                embargo_size=embargo_size,
                n_samples=n_samples,
            )

            # Verify no overlap with test set
            assert not np.any(np.isin(clean_train, np.arange(test_start, test_end)))

            # Verify purging
            purge_start = max(0, test_start - label_horizon)
            assert not np.any(np.isin(clean_train, np.arange(purge_start, test_start)))

            # Verify embargo
            embargo_end = min(n_samples, test_end + embargo_size)
            assert not np.any(np.isin(clean_train, np.arange(test_end, embargo_end)))


class TestPurgingValidation:
    """Test validation in purging functions."""

    def test_purge_requires_n_samples_for_integer(self):
        """Test that n_samples is required for integer-based purging."""
        import pytest

        with pytest.raises(ValueError, match="n_samples required"):
            calculate_purge_indices(
                test_start=50,
                test_end=60,
                label_horizon=5,
                # Missing n_samples
            )

    def test_purge_timestamp_requires_timestamp_inputs(self):
        """Test that timestamps mode requires Timestamp inputs."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

        with pytest.raises(TypeError, match="must be Timestamps"):
            calculate_purge_indices(
                timestamps=timestamps,
                test_start=50,  # Integer, not Timestamp
                test_end=60,
                label_horizon=5,
            )

    def test_purge_timestamps_require_tz_awareness(self):
        """Test that timestamps must be timezone-aware."""
        import pytest

        # Naive timestamps (no timezone)
        naive_timestamps = pd.date_range("2020-01-01", periods=100, freq="D")
        test_start = pd.Timestamp("2020-02-20", tz="UTC")
        test_end = pd.Timestamp("2020-03-01", tz="UTC")

        with pytest.raises(ValueError, match="timezone-aware"):
            calculate_purge_indices(
                timestamps=naive_timestamps,
                test_start=test_start,
                test_end=test_end,
                label_horizon=pd.Timedelta("5D"),
            )

    def test_purge_test_start_requires_tz_awareness(self):
        """Test that test_start must be timezone-aware."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = pd.Timestamp("2020-02-20")  # Naive
        test_end = pd.Timestamp("2020-03-01", tz="UTC")

        with pytest.raises(ValueError, match="test_start must be timezone-aware"):
            calculate_purge_indices(
                timestamps=timestamps,
                test_start=test_start,
                test_end=test_end,
                label_horizon=pd.Timedelta("5D"),
            )

    def test_purge_test_end_requires_tz_awareness(self):
        """Test that test_end must be timezone-aware."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = pd.Timestamp("2020-02-20", tz="UTC")
        test_end = pd.Timestamp("2020-03-01")  # Naive

        with pytest.raises(ValueError, match="test_end must be timezone-aware"):
            calculate_purge_indices(
                timestamps=timestamps,
                test_start=test_start,
                test_end=test_end,
                label_horizon=pd.Timedelta("5D"),
            )

    def test_purge_label_horizon_int_converted_to_timedelta(self):
        """Test that integer label_horizon is converted to Timedelta."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = timestamps[50]
        test_end = timestamps[60]

        # Pass integer for label_horizon - should be converted to days
        purged = calculate_purge_indices(
            timestamps=timestamps,
            test_start=test_start,
            test_end=test_end,
            label_horizon=5,  # Integer, should become 5 days
        )

        # Should purge 5 days before test start
        expected_purged = list(range(45, 50))
        assert purged == expected_purged


class TestEmbargoValidation:
    """Test validation in embargo functions."""

    def test_embargo_no_params_returns_empty(self):
        """Test that no embargo params returns empty list."""
        result = calculate_embargo_indices(
            n_samples=100,
            test_start=50,
            test_end=60,
            # No embargo_size or embargo_pct
        )
        assert result == []

    def test_embargo_both_params_raises_error(self):
        """Test that specifying both params raises error."""
        import pytest

        with pytest.raises(ValueError, match="Specify either.*not both"):
            calculate_embargo_indices(
                n_samples=100,
                test_start=50,
                test_end=60,
                embargo_size=5,
                embargo_pct=0.05,  # Both specified
            )

    def test_embargo_timestamp_requires_timestamp_inputs(self):
        """Test that timestamps mode requires Timestamp inputs."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

        with pytest.raises(TypeError, match="must be Timestamps"):
            calculate_embargo_indices(
                timestamps=timestamps,
                test_start=50,  # Integer, not Timestamp
                test_end=60,
                embargo_size=5,
            )

    def test_embargo_timestamps_require_tz_awareness(self):
        """Test that timestamps must be timezone-aware."""
        import pytest

        naive_timestamps = pd.date_range("2020-01-01", periods=100, freq="D")
        test_start = pd.Timestamp("2020-02-20", tz="UTC")
        test_end = pd.Timestamp("2020-03-01", tz="UTC")

        with pytest.raises(ValueError, match="timezone-aware"):
            calculate_embargo_indices(
                timestamps=naive_timestamps,
                test_start=test_start,
                test_end=test_end,
                embargo_size=pd.Timedelta("5D"),
            )

    def test_embargo_test_start_requires_tz_awareness(self):
        """Test that test_start must be timezone-aware."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = pd.Timestamp("2020-02-20")  # Naive
        test_end = pd.Timestamp("2020-03-01", tz="UTC")

        with pytest.raises(ValueError, match="test_start must be timezone-aware"):
            calculate_embargo_indices(
                timestamps=timestamps,
                test_start=test_start,
                test_end=test_end,
                embargo_size=pd.Timedelta("5D"),
            )

    def test_embargo_test_end_requires_tz_awareness(self):
        """Test that test_end must be timezone-aware."""
        import pytest

        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = pd.Timestamp("2020-02-20", tz="UTC")
        test_end = pd.Timestamp("2020-03-01")  # Naive

        with pytest.raises(ValueError, match="test_end must be timezone-aware"):
            calculate_embargo_indices(
                timestamps=timestamps,
                test_start=test_start,
                test_end=test_end,
                embargo_size=pd.Timedelta("5D"),
            )

    def test_embargo_pct_with_timestamps(self):
        """Test percentage embargo with timestamps."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = timestamps[50]
        test_end = timestamps[60]

        embargoed = calculate_embargo_indices(
            timestamps=timestamps,
            test_start=test_start,
            test_end=test_end,
            embargo_pct=0.05,  # 5% of duration
        )

        # Should have some embargo indices
        assert len(embargoed) > 0
        assert all(idx >= 60 for idx in embargoed)

    def test_embargo_int_converted_to_timedelta(self):
        """Test that integer embargo_size is converted to Timedelta."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        test_start = timestamps[50]
        test_end = timestamps[60]

        embargoed = calculate_embargo_indices(
            timestamps=timestamps,
            test_start=test_start,
            test_end=test_end,
            embargo_size=5,  # Integer, should become 5 days
        )

        expected = list(range(60, 65))
        assert embargoed == expected

    def test_embargo_requires_n_samples_for_integer(self):
        """Test that n_samples is required for integer-based embargo."""
        import pytest

        with pytest.raises(ValueError, match="n_samples required"):
            calculate_embargo_indices(
                test_start=50,
                test_end=60,
                embargo_size=5,
                # Missing n_samples
            )


class TestApplyPurgingWithTimestamps:
    """Test apply_purging_and_embargo with timestamps."""

    def test_apply_with_timestamps(self):
        """Test combined purging and embargo with timestamps."""
        n_samples = 100
        timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="D", tz="UTC")
        train_indices = np.arange(n_samples)
        test_start = timestamps[50]
        test_end = timestamps[60]

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start,
            test_end=test_end,
            label_horizon=pd.Timedelta("5D"),
            embargo_size=pd.Timedelta("5D"),
            timestamps=timestamps,
        )

        # Should remove test, purged, and embargoed indices
        # Test [50, 60), Purge [45, 50), Embargo [60, 65)
        expected_removed = set(range(45, 65))
        for idx in clean_train:
            assert idx not in expected_removed

    def test_apply_with_embargo_pct_and_timestamps(self):
        """Test combined with embargo_pct and timestamps."""
        n_samples = 100
        timestamps = pd.date_range("2020-01-01", periods=n_samples, freq="D", tz="UTC")
        train_indices = np.arange(n_samples)
        test_start = timestamps[50]
        test_end = timestamps[60]

        clean_train = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start,
            test_end=test_end,
            label_horizon=pd.Timedelta("5D"),
            embargo_pct=0.05,
            timestamps=timestamps,
        )

        # Test indices should be removed
        test_range = set(range(50, 60))
        for idx in clean_train:
            assert idx not in test_range
