"""Utility functions for cross-validation splitters.

This module contains shared functionality used across different splitter
implementations, particularly for handling timestamp conversions and
boundary calculations.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from pandas import Timedelta


def convert_indices_to_timestamps(
    start_idx: int,
    end_idx: int,
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
) -> tuple[int | Any, int | Any]:
    """Convert indices to timestamps with robust boundary handling.

    This function handles the conversion of array indices to timestamp values,
    with robust estimation when the end index extends beyond available data.
    It's designed to handle both regular and irregular time series frequencies.

    Parameters
    ----------
    start_idx : int
        Starting index
    end_idx : int
        Ending index (exclusive)
    timestamps : pd.DatetimeIndex or np.ndarray, optional
        Array of timestamps. If None, returns original indices.

    Returns:
    -------
    tuple[Union[int, Any], Union[int, Any]]
        (start_time, end_time) where times are either timestamps or indices

    Examples:
    --------
    >>> import pandas as pd
    >>> timestamps = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> start_time, end_time = convert_indices_to_timestamps(10, 20, timestamps)
    >>> print(start_time, end_time)
    2020-01-11 00:00:00 2020-01-21 00:00:00

    >>> # Handle end index beyond data
    >>> start_time, end_time = convert_indices_to_timestamps(90, 105, timestamps)
    >>> print(end_time)  # Estimated based on frequency
    2020-04-15 00:00:00
    """
    if timestamps is None:
        return start_idx, end_idx

    # Convert start index (always available)
    start_time = timestamps[start_idx]

    # Handle end index with robust boundary checking
    if end_idx < len(timestamps):
        # Direct lookup when index is within bounds
        end_time = timestamps[end_idx]
    else:
        # Estimate end time when beyond available data
        end_time = _estimate_timestamp_beyond_data(end_idx, timestamps)

    return start_time, end_time


def _estimate_timestamp_beyond_data(
    target_idx: int,
    timestamps: pd.DatetimeIndex | np.ndarray,
) -> Any:
    """Estimate timestamp for an index beyond available data.

    This function provides robust timestamp estimation for irregular
    time series by using multiple frequency estimation methods.

    Parameters
    ----------
    target_idx : int
        Target index beyond the timestamp array
    timestamps : pd.DatetimeIndex or np.ndarray
        Available timestamps

    Returns:
    -------
    Any
        Estimated timestamp
    """
    if len(timestamps) < 2:
        # Can't estimate frequency with fewer than 2 points
        return timestamps[-1]

    # Calculate how many steps beyond the data we need
    steps_beyond = target_idx - len(timestamps) + 1

    if isinstance(timestamps, pd.DatetimeIndex):
        # Use pandas DatetimeIndex inference for better frequency handling
        try:
            # Try to infer frequency from the index
            freq = timestamps.freq or pd.infer_freq(timestamps)
            if freq is not None:
                # freq is DateOffset or str - arithmetic works at runtime
                return cast(
                    Any, timestamps[-1] + steps_beyond * pd.tseries.frequencies.to_offset(freq)
                )
        except (ValueError, TypeError):
            # Fall back to simple difference calculation
            pass

    # Robust frequency estimation using multiple methods
    # estimated_freq can be Timedelta or np.timedelta64 depending on input type
    estimated_freq: Timedelta | np.timedelta64 | Any
    if len(timestamps) >= 10:
        # Use median of recent differences for more robust estimation
        recent_diffs = np.diff(timestamps[-10:])
        # Sort and take middle value to preserve timedelta type
        sorted_diffs = np.sort(recent_diffs)
        mid_idx = len(sorted_diffs) // 2
        estimated_freq = sorted_diffs[mid_idx]
    elif len(timestamps) >= 3:
        # Use median of all differences
        all_diffs = np.diff(timestamps)
        # Sort and take middle value to preserve timedelta type
        sorted_diffs = np.sort(all_diffs)
        mid_idx = len(sorted_diffs) // 2
        estimated_freq = sorted_diffs[mid_idx]
    else:
        # Simple two-point difference
        estimated_freq = timestamps[-1] - timestamps[-2]

    # Estimate the target timestamp - cast needed for mixed datetime arithmetic
    estimated_time: Any = timestamps[-1] + steps_beyond * estimated_freq

    return estimated_time


def validate_timestamp_array(
    timestamps: pd.DatetimeIndex | np.ndarray | None,
    n_samples: int,
) -> None:
    """Validate timestamp array for use in cross-validation.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex or np.ndarray, optional
        Timestamp array to validate
    n_samples : int
        Expected number of samples

    Raises:
    ------
    ValueError
        If timestamps are invalid or mismatched with sample count
    """
    if timestamps is None:
        return

    if len(timestamps) != n_samples:
        raise ValueError(
            f"Timestamp array length ({len(timestamps)}) does not match number of samples ({n_samples})",
        )

    if len(timestamps) > 1:
        # Check for non-decreasing order (allows for duplicate timestamps)
        if isinstance(timestamps, pd.DatetimeIndex):
            if not timestamps.is_monotonic_increasing:
                raise ValueError("Timestamps must be in non-decreasing order")
        else:
            diffs = np.diff(timestamps)
            if np.issubdtype(diffs.dtype, np.timedelta64):
                non_decreasing = np.all(diffs >= np.timedelta64(0, "ns"))
            else:
                non_decreasing = np.all(diffs >= 0)

            if not non_decreasing:
                raise ValueError("Timestamps must be in non-decreasing order")


def get_time_boundaries(
    group_boundaries: list[tuple[int, int]],
    group_indices: list[int],
    timestamps: pd.DatetimeIndex | np.ndarray | None = None,
) -> list[tuple[int | Any, int | Any]]:
    """Convert multiple group boundaries from indices to timestamps.

    Parameters
    ----------
    group_boundaries : list[tuple[int, int]]
        List of (start_idx, end_idx) boundaries
    group_indices : list[int]
        Indices of groups to convert
    timestamps : pd.DatetimeIndex or np.ndarray, optional
        Timestamp array

    Returns:
    -------
    list[tuple[Union[int, Any], Union[int, Any]]]
        List of (start_time, end_time) boundaries
    """
    time_boundaries = []

    for group_idx in group_indices:
        start_idx, end_idx = group_boundaries[group_idx]
        start_time, end_time = convert_indices_to_timestamps(
            start_idx,
            end_idx,
            timestamps,
        )
        time_boundaries.append((start_time, end_time))

    return time_boundaries
