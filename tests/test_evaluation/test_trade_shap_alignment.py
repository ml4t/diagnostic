"""Tests for Trade-SHAP timestamp alignment.

Tests cover O(log n) timestamp lookup with exact and nearest-match scenarios,
including edge cases like duplicate timestamps and boundary conditions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation.trade_shap.alignment import (
    AlignmentResult,
    TimestampAligner,
)


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_exact_match_result(self):
        """AlignmentResult for exact match."""
        result = AlignmentResult(index=5, exact=True, distance_seconds=0.0)
        assert result.index == 5
        assert result.exact is True
        assert result.distance_seconds == 0.0

    def test_nearest_match_result(self):
        """AlignmentResult for nearest match."""
        result = AlignmentResult(index=5, exact=False, distance_seconds=30.0)
        assert result.index == 5
        assert result.exact is False
        assert result.distance_seconds == 30.0

    def test_no_match_result(self):
        """AlignmentResult when no match found."""
        result = AlignmentResult(index=None, exact=False, distance_seconds=float("inf"))
        assert result.index is None
        assert result.exact is False
        assert result.distance_seconds == float("inf")


class TestTimestampAlignerInit:
    """Tests for TimestampAligner initialization."""

    def test_empty_timestamps_raises_value_error(self):
        """Empty timestamp array should raise ValueError."""
        with pytest.raises(ValueError, match="empty timestamp array"):
            TimestampAligner.from_datetime_index([])

    def test_valid_timestamps_creates_aligner(self):
        """Valid timestamps should create aligner correctly."""
        timestamps = [datetime(2024, 1, i) for i in range(1, 4)]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        assert len(aligner) == 3
        assert aligner.tolerance_seconds == 0.0

    def test_lookup_dict_built_correctly(self):
        """O(1) lookup dict should be built correctly."""
        timestamps = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        # Each timestamp should map to its index
        assert datetime(2024, 1, 1) in aligner.index_by_ts
        assert datetime(2024, 1, 2) in aligner.index_by_ts
        assert datetime(2024, 1, 3) in aligner.index_by_ts


class TestFromDatetimeIndex:
    """Tests for from_datetime_index factory method."""

    def test_pandas_datetimeindex_input(self):
        """Pandas DatetimeIndex should be handled."""
        timestamps = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        aligner = TimestampAligner.from_datetime_index(timestamps)

        assert len(aligner) == 3

    def test_numpy_datetime64_array_input(self):
        """Numpy datetime64 array should be handled."""
        timestamps = np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            dtype="datetime64[D]",
        )
        aligner = TimestampAligner.from_datetime_index(timestamps)

        assert len(aligner) == 3

    def test_python_datetime_list_input(self):
        """Python datetime list should be handled."""
        timestamps = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        assert len(aligner) == 3

    def test_duplicate_timestamps_keeps_first(self):
        """Duplicate timestamps should keep first occurrence."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),  # index 0
            datetime(2024, 1, 1, 12, 0),  # index 1
            datetime(2024, 1, 1, 10, 0),  # index 2 (duplicate of index 0)
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        # Should map to first occurrence (index 0)
        assert aligner.index_by_ts[datetime(2024, 1, 1, 10, 0)] == 0

    def test_tolerance_parameter(self):
        """Tolerance parameter should be stored."""
        timestamps = [datetime(2024, 1, 1)]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=60.0)

        assert aligner.tolerance_seconds == 60.0


class TestAlign:
    """Tests for align() method."""

    def test_exact_match_returns_correct_index(self):
        """Exact match (tolerance=0) returns correct index."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=0.0)

        result = aligner.align(datetime(2024, 1, 1, 11, 0))

        assert result.index == 1
        assert result.exact is True
        assert result.distance_seconds == 0.0

    def test_no_exact_match_no_tolerance_returns_none(self):
        """No exact match with tolerance=0 returns None."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 12, 0),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=0.0)

        result = aligner.align(datetime(2024, 1, 1, 11, 0))

        assert result.index is None
        assert result.exact is False
        assert result.distance_seconds == float("inf")

    def test_nearest_match_within_tolerance(self):
        """Nearest match within tolerance returns closest index."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 1, 10, 5, 0),  # 5 min = 300 sec
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=600.0)

        # Target is 2 min after first timestamp (120 sec)
        target = datetime(2024, 1, 1, 10, 2, 0)
        result = aligner.align(target)

        assert result.index == 0  # Closer to first timestamp
        assert result.exact is False
        assert result.distance_seconds == 120.0  # 2 min = 120 sec

    def test_nearest_match_beyond_tolerance_returns_none(self):
        """Nearest match beyond tolerance returns None."""
        timestamps = [datetime(2024, 1, 1, 10, 0, 0)]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=60.0)

        # Target is 5 min (300 sec) away - beyond 60 sec tolerance
        target = datetime(2024, 1, 1, 10, 5, 0)
        result = aligner.align(target)

        assert result.index is None
        assert result.exact is False
        assert result.distance_seconds == 300.0  # Reports actual distance

    def test_target_before_all_timestamps(self):
        """Target before all timestamps (boundary)."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=7200.0)

        # Target is 1 hour before first timestamp
        target = datetime(2024, 1, 1, 9, 0)
        result = aligner.align(target)

        assert result.index == 0  # Closest is first
        assert result.exact is False
        assert result.distance_seconds == 3600.0  # 1 hour

    def test_target_after_all_timestamps(self):
        """Target after all timestamps (boundary)."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=7200.0)

        # Target is 1 hour after last timestamp
        target = datetime(2024, 1, 1, 12, 0)
        result = aligner.align(target)

        assert result.index == 1  # Closest is last
        assert result.exact is False
        assert result.distance_seconds == 3600.0  # 1 hour

    def test_target_equidistant_from_neighbors(self):
        """Target equidistant from two timestamps picks one consistently."""
        timestamps = [
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 1, 10, 2, 0),  # 2 min apart
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=120.0)

        # Target is exactly in the middle
        target = datetime(2024, 1, 1, 10, 1, 0)
        result = aligner.align(target)

        # Should return one of them (implementation-dependent which)
        assert result.index in [0, 1]
        assert result.exact is False
        assert result.distance_seconds == 60.0  # 1 min


class TestAlignMany:
    """Tests for align_many() method."""

    def test_empty_targets_returns_empty_list(self):
        """Empty targets list returns empty results."""
        timestamps = [datetime(2024, 1, 1)]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        result = aligner.align_many([])

        assert result == []

    def test_multiple_targets_returns_list(self):
        """Multiple targets returns list of AlignmentResults."""
        timestamps = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        targets = [datetime(2024, 1, 1), datetime(2024, 1, 3)]
        results = aligner.align_many(targets)

        assert len(results) == 2
        assert results[0].index == 0
        assert results[1].index == 2

    def test_mixed_success_failure_results(self):
        """Mixed success and failure results."""
        timestamps = [datetime(2024, 1, 1)]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=0.0)

        targets = [
            datetime(2024, 1, 1),  # Exact match
            datetime(2024, 1, 2),  # No match
        ]
        results = aligner.align_many(targets)

        assert len(results) == 2
        assert results[0].index == 0
        assert results[0].exact is True
        assert results[1].index is None
        assert results[1].exact is False


class TestLen:
    """Tests for __len__ method."""

    def test_len_returns_timestamp_count(self):
        """__len__ returns number of timestamps."""
        timestamps = [datetime(2024, 1, i) for i in range(1, 11)]
        aligner = TimestampAligner.from_datetime_index(timestamps)

        assert len(aligner) == 10


class TestEdgeCases:
    """Additional edge case tests."""

    def test_single_timestamp(self):
        """Single timestamp works correctly."""
        timestamps = [datetime(2024, 1, 1, 12, 0)]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=3600.0)

        # Exact match
        result = aligner.align(datetime(2024, 1, 1, 12, 0))
        assert result.index == 0
        assert result.exact is True

        # Within tolerance
        result = aligner.align(datetime(2024, 1, 1, 12, 30))
        assert result.index == 0
        assert result.exact is False
        assert result.distance_seconds == 1800.0

    def test_unsorted_input_handled(self):
        """Unsorted input is sorted internally."""
        timestamps = [
            datetime(2024, 1, 3),
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=86400.0)

        # Should find each timestamp correctly
        assert aligner.align(datetime(2024, 1, 1)).index == 1  # Original index
        assert aligner.align(datetime(2024, 1, 2)).index == 2
        assert aligner.align(datetime(2024, 1, 3)).index == 0

    def test_millisecond_precision(self):
        """Millisecond precision is preserved."""
        timestamps = [
            datetime(2024, 1, 1, 12, 0, 0, 0),
            datetime(2024, 1, 1, 12, 0, 0, 500000),  # +500ms
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=1.0)

        # Target is 200ms, closer to first
        target = datetime(2024, 1, 1, 12, 0, 0, 200000)
        result = aligner.align(target)

        assert result.index == 0
        assert result.distance_seconds == pytest.approx(0.2, abs=0.001)

    def test_large_number_of_timestamps(self):
        """Large number of timestamps performs reasonably."""
        n = 10000
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(minutes=i) for i in range(n)]
        aligner = TimestampAligner.from_datetime_index(timestamps, tolerance_seconds=60.0)

        # Find middle timestamp
        target = base + timedelta(minutes=n // 2)
        result = aligner.align(target)

        assert result.index == n // 2
        assert result.exact is True
