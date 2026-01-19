"""Group partitioning strategies for CPCV.

This module handles partitioning the timeline into groups:
- Contiguous partitioning (equal-sized time slices)
- Session-aligned partitioning (respects trading session boundaries)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def create_contiguous_partitions(
    n_samples: int,
    n_groups: int,
) -> list[tuple[int, int]]:
    """Create boundaries for contiguous groups.

    Partitions n_samples into n_groups approximately equal-sized groups.
    Earlier groups get extra samples when n_samples is not evenly divisible.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_groups : int
        Number of groups to create.

    Returns
    -------
    boundaries : list of tuple
        List of (start_idx, end_idx) for each group.
        end_idx is exclusive (standard Python convention).

    Raises
    ------
    ValueError
        If boundaries don't satisfy CPCV invariants.

    Examples
    --------
    >>> create_contiguous_partitions(100, 5)
    [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    >>> create_contiguous_partitions(103, 5)
    [(0, 21), (21, 42), (42, 62), (62, 82), (82, 103)]
    """
    base_size = n_samples // n_groups
    remainder = n_samples % n_groups

    boundaries = []
    current_start = 0

    for i in range(n_groups):
        # Add extra sample to first 'remainder' groups
        group_size = base_size + (1 if i < remainder else 0)
        group_end = current_start + group_size

        boundaries.append((current_start, group_end))
        current_start = group_end

    # Validate invariants
    validate_contiguous_partitions(boundaries, n_samples)

    return boundaries


def validate_contiguous_partitions(
    boundaries: list[tuple[int, int]],
    n_samples: int,
) -> None:
    """Validate CPCV group boundary invariants.

    Ensures:
    1. All samples are covered (no gaps)
    2. No overlap between groups
    3. Groups are contiguous

    Parameters
    ----------
    boundaries : list of tuple
        List of (start_idx, end_idx) for each group.
    n_samples : int
        Total number of samples.

    Raises
    ------
    ValueError
        If any invariant is violated.
    """
    if not boundaries:
        raise ValueError("CPCV invariant violated: no group boundaries created")

    # Check first boundary starts at 0
    if boundaries[0][0] != 0:
        raise ValueError(
            f"CPCV invariant violated: first group must start at 0, got {boundaries[0][0]}"
        )

    # Check last boundary ends at n_samples
    if boundaries[-1][1] != n_samples:
        raise ValueError(
            f"CPCV invariant violated: last group must end at {n_samples}, got {boundaries[-1][1]}"
        )

    # Check contiguity (each group starts where previous ended)
    for i in range(1, len(boundaries)):
        prev_end = boundaries[i - 1][1]
        curr_start = boundaries[i][0]
        if curr_start != prev_end:
            raise ValueError(
                f"CPCV invariant violated: gap between group {i - 1} (ends at {prev_end}) "
                f"and group {i} (starts at {curr_start})"
            )

    # Check each group is non-empty
    for i, (start, end) in enumerate(boundaries):
        if end <= start:
            raise ValueError(
                f"CPCV invariant violated: group {i} is empty or invalid (start={start}, end={end})"
            )


def create_session_partitions(
    X: pl.DataFrame | pd.DataFrame,
    session_col: str,
    n_groups: int,
    session_to_indices_fn: Callable[
        [pl.DataFrame | pd.DataFrame, str],
        tuple[list[Any], dict[Any, NDArray[np.intp]]],
    ],
) -> list[NDArray[np.intp]]:
    """Create exact index arrays per group, aligned to session boundaries.

    Unlike contiguous partitioning which returns (start, end) ranges,
    this method returns EXACT index arrays for each group. This is critical
    for correct behavior with non-contiguous or interleaved data.

    Parameters
    ----------
    X : DataFrame
        Data with session column.
    session_col : str
        Name of column containing session identifiers.
    n_groups : int
        Number of groups to create.
    session_to_indices_fn : callable
        Function that returns (ordered_sessions, session_to_indices_dict).
        Typically from BaseSplitter._session_to_indices.

    Returns
    -------
    group_indices : list of np.ndarray
        List of numpy arrays containing exact row indices for each group.
        Each array contains the indices for all rows belonging to sessions
        in that group.

    Raises
    ------
    ValueError
        If not enough sessions for the requested number of groups.

    Notes
    -----
    The key difference from contiguous partitioning is that we track
    exact indices rather than (start, end) boundaries. This prevents
    incorrect index ranges when data is interleaved by asset within sessions.
    """
    # Get session -> indices mapping
    ordered_sessions, session_to_indices = session_to_indices_fn(X, session_col)
    n_sessions = len(ordered_sessions)

    if n_sessions < n_groups:
        raise ValueError(
            f"Not enough sessions ({n_sessions}) for {n_groups} groups. "
            f"Need at least {n_groups} sessions."
        )

    # Partition sessions into groups
    base_sessions_per_group = n_sessions // n_groups
    remainder = n_sessions % n_groups

    group_indices_list = []
    current_session_idx = 0

    for i in range(n_groups):
        # Add extra session to first 'remainder' groups
        sessions_in_group = base_sessions_per_group + (1 if i < remainder else 0)
        session_group_end = current_session_idx + sessions_in_group

        # Get sessions for this group
        group_sessions = ordered_sessions[current_session_idx:session_group_end]

        # Collect EXACT indices for sessions in this group
        indices_arrays = [session_to_indices[s] for s in group_sessions]
        if indices_arrays:
            group_indices = np.concatenate(indices_arrays)
            # Sort for predictable ordering
            group_indices = np.sort(group_indices)
        else:
            group_indices = np.array([], dtype=np.intp)

        group_indices_list.append(group_indices)
        current_session_idx = session_group_end

    return group_indices_list


def boundaries_to_indices(
    boundaries: list[tuple[int, int]],
    groups: tuple[int, ...],
) -> NDArray[np.intp]:
    """Convert group boundaries to flat index array for selected groups.

    Parameters
    ----------
    boundaries : list of tuple
        List of (start_idx, end_idx) for each group.
    groups : tuple of int
        Which groups to include.

    Returns
    -------
    indices : np.ndarray
        Sorted array of indices for selected groups.
    """
    # Use numpy concatenation instead of Python list extend for performance
    ranges = [np.arange(boundaries[g][0], boundaries[g][1], dtype=np.intp) for g in groups]
    if not ranges:
        return np.array([], dtype=np.intp)
    return np.concatenate(ranges)


def exact_indices_to_array(
    group_indices_list: list[NDArray[np.intp]],
    groups: tuple[int, ...],
) -> NDArray[np.intp]:
    """Concatenate exact index arrays for selected groups.

    Parameters
    ----------
    group_indices_list : list of np.ndarray
        List of exact index arrays for each group.
    groups : tuple of int
        Which groups to include.

    Returns
    -------
    indices : np.ndarray
        Sorted array of indices for selected groups.
    """
    arrays = [group_indices_list[g] for g in groups]
    if not arrays or all(len(a) == 0 for a in arrays):
        return np.array([], dtype=np.intp)
    return np.sort(np.concatenate(arrays))
