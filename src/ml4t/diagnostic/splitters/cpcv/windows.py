"""Time window computation for CPCV purging.

This module handles computing purge windows from test indices:
- Timestamp windows from exact indices
- Contiguous segment detection
- Window merging for efficient purging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    """A time window for purging, with exclusive end bound.

    Attributes
    ----------
    start : pd.Timestamp
        Start of the window (inclusive).
    end_exclusive : pd.Timestamp
        End of the window (exclusive).
    """

    start: pd.Timestamp
    end_exclusive: pd.Timestamp


def timestamp_window_from_indices(
    indices: NDArray[np.intp],
    timestamps: pd.DatetimeIndex,
) -> TimeWindow | None:
    """Compute timestamp window from actual indices.

    This is critical for correct purging in session-aligned mode. Instead of
    using (min_row_idx, max_row_idx) boundaries which can span unrelated rows
    in interleaved data, we compute the actual timestamp bounds from the test
    indices.

    Parameters
    ----------
    indices : ndarray
        Row indices of test samples.
    timestamps : pd.DatetimeIndex
        Timestamps for all samples.

    Returns
    -------
    TimeWindow or None
        Window with (start_time, end_time_exclusive) if indices non-empty.
        None if indices is empty (signals caller to skip purging).

    Notes
    -----
    The end is made exclusive by adding 1 nanosecond. This handles the case
    of duplicate timestamps at the boundary.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> timestamps = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    >>> indices = np.array([2, 3, 4])
    >>> window = timestamp_window_from_indices(indices, timestamps)
    >>> window.start
    Timestamp('2020-01-03 00:00:00+0000', tz='UTC')
    """
    import pandas as pd

    if len(indices) == 0:
        # Empty indices - return None to signal callers to skip purging
        return None

    test_timestamps = timestamps.take(indices)
    start_time = test_timestamps.min()
    # Add 1 nanosecond to make end exclusive (handles duplicate timestamps)
    end_time_exclusive = test_timestamps.max() + pd.Timedelta(1, "ns")
    return TimeWindow(start=start_time, end_exclusive=end_time_exclusive)


def find_contiguous_segments(
    test_groups_data: list[tuple[int, int, int, NDArray[np.intp] | None]],
    asset_indices: NDArray[np.intp],
) -> list[list[tuple[int, int, int, NDArray[np.intp]]]]:
    """Find contiguous segments of test groups for a given asset.

    Groups test data into contiguous segments based on temporal adjacency.
    This allows applying one purge window per segment instead of per group,
    which is more efficient and statistically correct.

    Parameters
    ----------
    test_groups_data : list of tuple
        Each tuple contains (group_idx, group_start, group_end, exact_indices).
        exact_indices is non-None for session-aligned mode.
    asset_indices : ndarray
        Indices belonging to this asset.

    Returns
    -------
    segments : list of list of tuple
        Each segment is a list of (group_idx, start, end, asset_test_indices).
        Segments are separated by gaps in the test groups.

    Notes
    -----
    In session-aligned mode, exact_indices should be used instead of
    generating indices via np.arange (which is wrong for interleaved data).
    """
    contiguous_segments: list[list[tuple[int, int, int, NDArray[np.intp]]]] = []
    current_segment: list[tuple[int, int, int, NDArray[np.intp]]] = []

    for group_idx, group_start, group_end, exact_indices in test_groups_data:
        # Get test indices for this asset in this group
        if exact_indices is not None:
            # Session-aligned mode: use exact indices
            group_test_indices = exact_indices
        else:
            # Standard mode: generate from boundaries
            group_test_indices = np.arange(group_start, group_end)
        asset_group_test_indices = np.intersect1d(group_test_indices, asset_indices)

        if len(asset_group_test_indices) == 0:
            # No test data for this asset in this group
            if current_segment:
                contiguous_segments.append(current_segment)
                current_segment = []
            continue

        # Check if this group is contiguous with the previous segment
        # current_segment[-1][2] is group_end (exclusive), gap exists if group_start > group_end
        if current_segment and group_start > current_segment[-1][2]:  # Gap detected
            # Finish current segment and start new one
            contiguous_segments.append(current_segment)
            current_segment = [(group_idx, group_start, group_end, asset_group_test_indices)]
        else:
            # Add to current segment
            current_segment.append((group_idx, group_start, group_end, asset_group_test_indices))

    # Don't forget the last segment
    if current_segment:
        contiguous_segments.append(current_segment)

    return contiguous_segments


def merge_windows(windows: list[TimeWindow]) -> list[TimeWindow]:
    """Merge overlapping time windows.

    This can reduce the number of purge operations when windows overlap,
    and provides clearer semantics about what's being purged.

    Parameters
    ----------
    windows : list of TimeWindow
        Windows to merge.

    Returns
    -------
    merged : list of TimeWindow
        Non-overlapping windows covering the same time ranges.
    """
    if not windows:
        return []

    # Sort by start time
    sorted_windows = sorted(windows, key=lambda w: w.start)
    merged = [sorted_windows[0]]

    for window in sorted_windows[1:]:
        last = merged[-1]
        if window.start <= last.end_exclusive:
            # Overlapping - merge by extending end
            merged[-1] = TimeWindow(
                start=last.start,
                end_exclusive=max(last.end_exclusive, window.end_exclusive),
            )
        else:
            # Non-overlapping - add new window
            merged.append(window)

    return merged
