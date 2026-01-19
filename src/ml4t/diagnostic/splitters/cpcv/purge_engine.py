"""Purging engine for CPCV.

This module implements the core purging and embargo logic:
- Mask-based purging (efficient for large datasets)
- Single-asset and multi-asset purging strategies
- Segment-based purging for temporal coherence
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ml4t.diagnostic.core.purging import apply_purging_and_embargo
from ml4t.diagnostic.splitters.cpcv.windows import (
    find_contiguous_segments,
    timestamp_window_from_indices,
)
from ml4t.diagnostic.splitters.utils import convert_indices_to_timestamps

if TYPE_CHECKING:
    import pandas as pd


def apply_single_asset_purging(
    train_indices: NDArray[np.intp],
    test_group_indices: tuple[int, ...],
    group_boundaries: list[tuple[int, int]],
    n_samples: int,
    timestamps: pd.DatetimeIndex | None,
    label_horizon: int | pd.Timedelta,
    embargo_size: int | pd.Timedelta | None,
    embargo_pct: float | None,
    group_indices_list: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.intp]:
    """Apply purging for single-asset data.

    For each test group, removes training samples that would cause
    look-ahead bias due to label overlap or temporal proximity.

    Parameters
    ----------
    train_indices : ndarray
        Initial training indices.
    test_group_indices : tuple of int
        Indices of groups used for testing.
    group_boundaries : list of tuple
        Boundaries (start, end) for each group.
    n_samples : int
        Total number of samples.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for time-based purging.
    label_horizon : int or pd.Timedelta
        Forward-looking period of labels.
    embargo_size : int or pd.Timedelta, optional
        Buffer period after test set.
    embargo_pct : float, optional
        Embargo as percentage of samples.
    group_indices_list : list of ndarray, optional
        Exact indices per group (for session-aligned mode).

    Returns
    -------
    clean_indices : ndarray
        Training indices after purging.
    """
    for test_group_idx in test_group_indices:
        # Compute purge window bounds
        if group_indices_list is not None and timestamps is not None:
            # Session-aligned mode: use actual timestamps from test indices
            test_indices = group_indices_list[test_group_idx]
            window = timestamp_window_from_indices(test_indices, timestamps)
            if window is None:
                # Empty test group - skip purging for this group
                continue
            test_start_time = window.start
            test_end_time = window.end_exclusive
        else:
            # Standard mode: use boundaries
            test_start_idx, test_end_idx = group_boundaries[test_group_idx]
            test_start_time, test_end_time = convert_indices_to_timestamps(
                test_start_idx,
                test_end_idx,
                timestamps,
            )

        # Apply purging and embargo for this test group
        train_indices = apply_purging_and_embargo(
            train_indices=train_indices,
            test_start=test_start_time,
            test_end=test_end_time,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            embargo_pct=embargo_pct,
            n_samples=n_samples,
            timestamps=timestamps,
        )

    return train_indices


def apply_multi_asset_purging(
    train_indices: NDArray[np.intp],
    test_group_indices: tuple[int, ...],
    group_boundaries: list[tuple[int, int]],
    n_samples: int,
    timestamps: pd.DatetimeIndex | None,
    groups_array: NDArray[Any],
    label_horizon: int | pd.Timedelta,
    embargo_size: int | pd.Timedelta | None,
    embargo_pct: float | None,
    group_indices_list: list[NDArray[np.intp]] | None = None,
) -> NDArray[np.intp]:
    """Apply purging for multi-asset data with per-asset isolation.

    This method correctly handles non-contiguous test groups by applying
    purging for each contiguous segment of test data separately per asset.

    Parameters
    ----------
    train_indices : ndarray
        Initial training indices.
    test_group_indices : tuple of int
        Indices of groups used for testing.
    group_boundaries : list of tuple
        Boundaries (start, end) for each group.
    n_samples : int
        Total number of samples.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for time-based purging.
    groups_array : ndarray
        Asset labels for each sample.
    label_horizon : int or pd.Timedelta
        Forward-looking period of labels.
    embargo_size : int or pd.Timedelta, optional
        Buffer period after test set.
    embargo_pct : float, optional
        Embargo as percentage of samples.
    group_indices_list : list of ndarray, optional
        Exact indices per group (for session-aligned mode).

    Returns
    -------
    clean_indices : ndarray
        Training indices after per-asset purging.
    """
    if len(groups_array) != n_samples:
        raise ValueError(
            f"groups length ({len(groups_array)}) must match number of samples ({n_samples})",
        )

    # Prepare test groups data for contiguous segment detection
    test_groups_data = prepare_test_groups_data(
        test_group_indices, group_boundaries, group_indices_list
    )

    # Apply purging per asset
    final_train_indices: list[int] = []
    unique_assets = np.unique(groups_array)

    for asset_id in unique_assets:
        # Process this asset's training data with purging
        asset_train = process_asset_purging(
            asset_id=asset_id,
            groups_array=groups_array,
            train_indices=train_indices,
            test_groups_data=test_groups_data,
            n_samples=n_samples,
            timestamps=timestamps,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            embargo_pct=embargo_pct,
            group_indices_list=group_indices_list,
        )
        final_train_indices.extend(asset_train)

    # Sort for deterministic output
    return np.sort(np.array(final_train_indices, dtype=np.intp))


def prepare_test_groups_data(
    test_group_indices: tuple[int, ...],
    group_boundaries: list[tuple[int, int]],
    group_indices_list: list[NDArray[np.intp]] | None = None,
) -> list[tuple[int, int, int, NDArray[np.intp] | None]]:
    """Prepare and sort test groups data for contiguous segment detection.

    Parameters
    ----------
    test_group_indices : tuple of int
        Which groups are used for testing.
    group_boundaries : list of tuple
        Boundaries (start, end) for each group.
    group_indices_list : list of ndarray, optional
        Exact indices per group (for session-aligned mode).

    Returns
    -------
    test_groups_data : list of tuple
        Sorted list of (group_idx, start_idx, end_idx, exact_indices).
        In session-aligned mode, exact_indices contains the actual row indices;
        otherwise it's None.
    """
    test_groups_data: list[tuple[int, int, int, NDArray[np.intp] | None]] = []
    for test_group_idx in test_group_indices:
        test_start_idx, test_end_idx = group_boundaries[test_group_idx]
        exact_indices = (
            group_indices_list[test_group_idx] if group_indices_list is not None else None
        )
        test_groups_data.append((test_group_idx, test_start_idx, test_end_idx, exact_indices))

    # Sort test groups by start index to identify contiguous segments
    test_groups_data.sort(key=lambda x: x[1])
    return test_groups_data


def process_asset_purging(
    asset_id: Any,
    groups_array: NDArray[Any],
    train_indices: NDArray[np.intp],
    test_groups_data: list[tuple[int, int, int, NDArray[np.intp] | None]],
    n_samples: int,
    timestamps: pd.DatetimeIndex | None,
    label_horizon: int | pd.Timedelta,
    embargo_size: int | pd.Timedelta | None,
    embargo_pct: float | None,
    group_indices_list: list[NDArray[np.intp]] | None = None,
) -> list[int]:
    """Process purging for a single asset across all test segments.

    Parameters
    ----------
    asset_id : any
        Identifier for this asset.
    groups_array : ndarray
        Asset labels for all samples.
    train_indices : ndarray
        Candidate training indices.
    test_groups_data : list of tuple
        Test group information from prepare_test_groups_data.
    n_samples : int
        Total number of samples.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for time-based purging.
    label_horizon : int or pd.Timedelta
        Forward-looking period of labels.
    embargo_size : int or pd.Timedelta, optional
        Buffer period after test set.
    embargo_pct : float, optional
        Embargo as percentage of samples.
    group_indices_list : list of ndarray, optional
        Exact indices per group (for session-aligned mode).

    Returns
    -------
    clean_indices : list of int
        Training indices for this asset after purging.
    """
    # Find indices for this asset
    asset_mask = groups_array == asset_id
    asset_indices = np.where(asset_mask)[0]

    # Get train indices for this asset
    asset_train_indices = np.intersect1d(train_indices, asset_indices)

    if len(asset_train_indices) == 0:
        return []

    # Find contiguous segments of test groups for this asset
    contiguous_segments = find_contiguous_segments(
        test_groups_data,
        asset_indices,
    )

    # If no test data for this asset, keep all training data
    if not contiguous_segments:
        return asset_train_indices.tolist()

    # Apply purging for each contiguous segment
    return apply_segment_purging(
        asset_train_indices=asset_train_indices,
        contiguous_segments=contiguous_segments,
        n_samples=n_samples,
        timestamps=timestamps,
        label_horizon=label_horizon,
        embargo_size=embargo_size,
        embargo_pct=embargo_pct,
        group_indices_list=group_indices_list,
    )


def apply_segment_purging(
    asset_train_indices: NDArray[np.intp],
    contiguous_segments: list[list[tuple[int, int, int, NDArray[np.intp]]]],
    n_samples: int,
    timestamps: pd.DatetimeIndex | None,
    label_horizon: int | pd.Timedelta,
    embargo_size: int | pd.Timedelta | None,
    embargo_pct: float | None,
    group_indices_list: list[NDArray[np.intp]] | None = None,
) -> list[int]:
    """Apply purging across all contiguous segments for an asset.

    Uses a set-based approach for tracking remaining indices, which is
    efficient for the iterative purging across segments.

    Parameters
    ----------
    asset_train_indices : ndarray
        Training indices for this asset.
    contiguous_segments : list of list of tuple
        Segments from find_contiguous_segments.
    n_samples : int
        Total number of samples.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for time-based purging.
    label_horizon : int or pd.Timedelta
        Forward-looking period of labels.
    embargo_size : int or pd.Timedelta, optional
        Buffer period after test set.
    embargo_pct : float, optional
        Embargo as percentage of samples.
    group_indices_list : list of ndarray, optional
        Exact indices per group (for session-aligned mode).

    Returns
    -------
    clean_indices : list of int
        Sorted training indices after purging all segments.
    """
    remaining_train_indices = set(asset_train_indices)

    for segment in contiguous_segments:
        if not segment:
            continue

        # Compute purge window bounds
        if group_indices_list is not None and timestamps is not None:
            # Session-aligned mode: compute timestamp bounds from actual test indices
            segment_test_indices = np.concatenate([item[3] for item in segment])
            window = timestamp_window_from_indices(segment_test_indices, timestamps)
            if window is None:
                # Empty test segment - skip purging for this segment
                continue
            segment_start_time = window.start
            segment_end_time = window.end_exclusive
        else:
            # Standard mode: use boundaries
            segment_start_idx = segment[0][1]  # Start of first group in segment
            segment_end_idx = segment[-1][2]  # End of last group in segment
            segment_start_time, segment_end_time = convert_indices_to_timestamps(
                segment_start_idx,
                segment_end_idx,
                timestamps,
            )

        # Apply purging for this contiguous segment
        remaining_array = np.array(list(remaining_train_indices), dtype=np.intp)

        if len(remaining_array) == 0:
            break

        clean_segment_train = apply_purging_and_embargo(
            train_indices=remaining_array,
            test_start=segment_start_time,
            test_end=segment_end_time,
            label_horizon=label_horizon,
            embargo_size=embargo_size,
            embargo_pct=embargo_pct,
            n_samples=n_samples,
            timestamps=timestamps,
        )

        # Update remaining indices (remove those that were purged)
        remaining_train_indices = set(clean_segment_train)

    return sorted(remaining_train_indices)
