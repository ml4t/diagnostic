"""Tests for splitter utility functions."""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.utils import (
    convert_indices_to_timestamps,
    get_time_boundaries,
    validate_timestamp_array,
)


class TestConvertIndicesToTimestamps:
    """Test timestamp conversion utility."""

    def test_with_none_timestamps(self):
        """Test conversion with no timestamps returns indices."""
        start_idx, end_idx = 10, 20
        start_time, end_time = convert_indices_to_timestamps(start_idx, end_idx, None)

        assert start_time == start_idx
        assert end_time == end_idx

    def test_with_regular_datetimeindex(self):
        """Test conversion with regular pandas DatetimeIndex."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D")
        start_idx, end_idx = 10, 20

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[10]
        assert end_time == timestamps[20]

    def test_with_end_beyond_data(self):
        """Test estimation when end index is beyond available data."""
        timestamps = pd.date_range("2020-01-01", periods=50, freq="D")
        start_idx, end_idx = 10, 60  # end_idx > len(timestamps)

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[10]
        # Should estimate end_time based on frequency
        expected_end = timestamps[-1] + pd.Timedelta(
            days=11,
        )  # 60 - 49 = 11 days beyond
        assert end_time == expected_end

    def test_with_irregular_timestamps(self):
        """Test with irregular timestamp spacing."""
        # Create irregular timestamps
        base_dates = pd.date_range("2020-01-01", periods=30, freq="D")
        # Make some gaps irregular
        irregular_dates = base_dates.tolist()
        irregular_dates[10:15] = pd.date_range("2020-01-11", periods=5, freq="2D")
        timestamps = pd.DatetimeIndex(irregular_dates)

        start_idx, end_idx = 5, 35  # end beyond data

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[5]
        # Should still provide a reasonable estimate
        assert isinstance(end_time, pd.Timestamp)

    def test_with_numpy_array_timestamps(self):
        """Test with numpy array of timestamps."""
        timestamps = np.array(
            [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(50)],
        )

        start_idx, end_idx = 10, 60

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[10]
        # Should estimate beyond the array
        expected_days_beyond = 60 - 50 + 1  # 11 days
        expected_end = timestamps[-1] + np.timedelta64(expected_days_beyond, "D")
        assert end_time == expected_end

    def test_edge_case_single_timestamp(self):
        """Test with only one timestamp."""
        timestamps = pd.DatetimeIndex(["2020-01-01"])

        start_idx, end_idx = 0, 5

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[0]
        # Should return last timestamp when can't estimate frequency
        assert end_time == timestamps[-1]

    def test_edge_case_two_timestamps(self):
        """Test with only two timestamps."""
        timestamps = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])

        start_idx, end_idx = 0, 5

        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )

        assert start_time == timestamps[0]
        # Should estimate based on the single difference
        expected_end = timestamps[-1] + pd.Timedelta(days=4)  # 5-2+1 = 4 days beyond
        assert end_time == expected_end


class TestValidateTimestampArray:
    """Test timestamp validation utility."""

    def test_valid_none_timestamps(self):
        """Test that None timestamps are valid."""
        validate_timestamp_array(None, 100)  # Should not raise

    def test_valid_matching_length(self):
        """Test valid timestamps with matching length."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D")
        validate_timestamp_array(timestamps, 100)  # Should not raise

    def test_invalid_length_mismatch(self):
        """Test error when timestamp length doesn't match samples."""
        timestamps = pd.date_range("2020-01-01", periods=50, freq="D")

        with pytest.raises(ValueError, match="Timestamp array length"):
            validate_timestamp_array(timestamps, 100)

    def test_invalid_non_monotonic_pandas(self):
        """Test error with non-monotonic pandas timestamps."""
        timestamps = pd.DatetimeIndex(
            [
                "2020-01-01",
                "2020-01-03",
                "2020-01-02",  # Not monotonic
            ],
        )

        with pytest.raises(ValueError, match="non-decreasing order"):
            validate_timestamp_array(timestamps, 3)

    def test_invalid_non_monotonic_numpy(self):
        """Test error with non-monotonic numpy timestamps."""
        timestamps = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-02"),  # Not monotonic
            ],
        )

        with pytest.raises(ValueError, match="non-decreasing order"):
            validate_timestamp_array(timestamps, 3)

    def test_valid_non_strictly_increasing(self):
        """Test that duplicate timestamps are allowed."""
        timestamps = pd.DatetimeIndex(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",  # Duplicates allowed
            ],
        )

        validate_timestamp_array(timestamps, 3)  # Should not raise

    def test_valid_non_strictly_increasing_numpy(self):
        """Test that duplicate numpy timestamps are allowed."""
        timestamps = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-02"),
            ],
        )

        validate_timestamp_array(timestamps, 3)  # Should not raise


class TestGetTimeBoundaries:
    """Test batch timestamp conversion utility."""

    def test_multiple_group_conversion(self):
        """Test converting multiple group boundaries."""
        timestamps = pd.date_range("2020-01-01", periods=100, freq="D")
        group_boundaries = [(0, 20), (20, 40), (40, 60), (60, 80)]
        group_indices = [0, 2]  # Select 1st and 3rd groups

        time_boundaries = get_time_boundaries(
            group_boundaries,
            group_indices,
            timestamps,
        )

        assert len(time_boundaries) == 2

        # First group
        start_time1, end_time1 = time_boundaries[0]
        assert start_time1 == timestamps[0]
        assert end_time1 == timestamps[20]

        # Third group
        start_time2, end_time2 = time_boundaries[1]
        assert start_time2 == timestamps[40]
        assert end_time2 == timestamps[60]

    def test_with_none_timestamps(self):
        """Test batch conversion with no timestamps."""
        group_boundaries = [(0, 20), (20, 40)]
        group_indices = [0, 1]

        time_boundaries = get_time_boundaries(group_boundaries, group_indices, None)

        assert len(time_boundaries) == 2
        assert time_boundaries[0] == (0, 20)
        assert time_boundaries[1] == (20, 40)

    def test_with_beyond_data_indices(self):
        """Test batch conversion with some indices beyond data."""
        timestamps = pd.date_range("2020-01-01", periods=50, freq="D")
        group_boundaries = [(0, 20), (30, 60)]  # Second group extends beyond
        group_indices = [0, 1]

        time_boundaries = get_time_boundaries(
            group_boundaries,
            group_indices,
            timestamps,
        )

        assert len(time_boundaries) == 2

        # First group (within data)
        start_time1, end_time1 = time_boundaries[0]
        assert start_time1 == timestamps[0]
        assert end_time1 == timestamps[20]

        # Second group (extends beyond)
        start_time2, end_time2 = time_boundaries[1]
        assert start_time2 == timestamps[30]
        # Should be estimated
        assert isinstance(end_time2, pd.Timestamp)
        assert end_time2 > timestamps[-1]
