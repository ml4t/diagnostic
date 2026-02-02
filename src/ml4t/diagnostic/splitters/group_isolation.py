"""Group isolation utilities for multi-asset cross-validation.

This module provides utilities to prevent the same asset (e.g., contract, symbol)
from appearing in both training and test sets during cross-validation. This is
critical for avoiding data leakage in multi-asset strategies.

Example Use Cases
-----------------
1. **Futures contracts**: Prevent ES_202312 from being in both train and test
2. **Multiple symbols**: Ensure AAPL data doesn't leak between folds
3. **Multi-strategy**: Isolate strategies to prevent cross-contamination

Integration with qdata
----------------------
The `groups` parameter should contain asset identifiers that come from your
data pipeline. Typically this would be a column like 'symbol', 'contract',
or 'asset_id' from your DataFrame.

Example::

    import polars as pl
    from ml4t.diagnostic.splitters import WalkForwardCV

    # Data with asset identifiers
    df = pl.DataFrame({
        'timestamp': [...],
        'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', ...],
        'returns': [...]
    })

    # Cross-validate with group isolation
    cv = WalkForwardCV(n_splits=5, isolate_groups=True)

    for train_idx, test_idx in cv.split(df, groups=df['symbol']):
        # Groups in test_idx will NEVER appear in train_idx
        train_symbols = df[train_idx]['symbol'].unique()
        test_symbols = df[test_idx]['symbol'].unique()
        assert len(set(train_symbols) & set(test_symbols)) == 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def validate_group_isolation(
    train_indices: NDArray[np.intp],
    test_indices: NDArray[np.intp],
    groups: pl.Series | pd.Series | NDArray[Any],
) -> tuple[bool, set]:
    """Validate that train and test sets have no overlapping groups.

    Parameters
    ----------
    train_indices : ndarray
        Training set indices.

    test_indices : ndarray
        Test set indices.

    groups : array-like
        Group labels for each sample.

    Returns
    -------
    is_valid : bool
        True if no groups overlap between train and test.

    overlapping_groups : set
        Set of group IDs that appear in both train and test.
        Empty if is_valid=True.

    Examples
    --------
    >>> import numpy as np
    >>> train_idx = np.array([0, 1, 2, 3])
    >>> test_idx = np.array([4, 5, 6, 7])
    >>> groups = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])
    >>> is_valid, overlap = validate_group_isolation(train_idx, test_idx, groups)
    >>> assert is_valid  # Groups don't overlap
    >>> assert len(overlap) == 0
    """
    # Convert groups to numpy array
    groups_array = DataFrameAdapter.to_numpy(groups).flatten()

    # Get unique groups in train and test
    train_groups = set(groups_array[train_indices])
    test_groups = set(groups_array[test_indices])

    # Find overlap
    overlapping_groups = train_groups & test_groups

    return len(overlapping_groups) == 0, overlapping_groups


def isolate_groups_from_train(
    train_indices: NDArray[np.intp],
    test_indices: NDArray[np.intp],
    groups: pl.Series | pd.Series | NDArray[Any],
) -> NDArray[np.intp]:
    """Remove samples from training set that share groups with test set.

    This function ensures strict group isolation by removing all training
    samples whose group appears anywhere in the test set.

    Parameters
    ----------
    train_indices : ndarray
        Initial training set indices.

    test_indices : ndarray
        Test set indices.

    groups : array-like
        Group labels for each sample.

    Returns
    -------
    clean_train_indices : ndarray
        Training indices with test groups removed.

    Examples
    --------
    >>> import numpy as np
    >>> train_idx = np.array([0, 1, 2, 3, 4, 5])
    >>> test_idx = np.array([6, 7])
    >>> groups = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'])
    >>> clean_train = isolate_groups_from_train(train_idx, test_idx, groups)
    >>> # Removes indices 4,5 because they share group 'C' with test indices 6,7
    >>> assert all(groups[clean_train] != 'C')

    Notes
    -----
    This can significantly reduce training set size if groups are imbalanced.
    Consider using group-aware splitting strategies to maintain balanced folds.
    """
    # Convert groups to numpy array
    groups_array = DataFrameAdapter.to_numpy(groups).flatten()

    # Get unique groups in test set
    test_groups = set(groups_array[test_indices])

    # Filter train indices to exclude any samples from test groups
    clean_train_mask = np.array([groups_array[idx] not in test_groups for idx in train_indices])

    return train_indices[clean_train_mask]


def get_group_boundaries(
    groups: pl.Series | pd.Series | NDArray[Any],
    sorted_indices: NDArray[np.intp] | None = None,
) -> dict[Any, tuple[int, int]]:
    """Get start and end indices for each unique group in sorted data.

    This is useful for group-aware splitting where you want to keep groups
    contiguous and avoid splitting a group across train/test boundaries.

    Parameters
    ----------
    groups : array-like
        Group labels for each sample.

    sorted_indices : ndarray, optional
        Pre-sorted indices. If None, assumes data is already sorted by group.

    Returns
    -------
    boundaries : dict
        Mapping from group ID to (start_idx, end_idx) tuple.

    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['A', 'A', 'A', 'B', 'B', 'C'])
    >>> boundaries = get_group_boundaries(groups)
    >>> assert boundaries['A'] == (0, 3)
    >>> assert boundaries['B'] == (3, 5)
    >>> assert boundaries['C'] == (5, 6)

    Notes
    -----
    This assumes groups are contiguous in the data. If groups are interleaved,
    provide `sorted_indices` to ensure correct boundary detection.
    """
    # Convert groups to numpy array
    groups_array = DataFrameAdapter.to_numpy(groups).flatten()

    # Apply sorting if provided
    if sorted_indices is not None:
        groups_array = groups_array[sorted_indices]

    # Find boundaries using change detection
    boundaries = {}
    unique_groups = []
    current_group = None
    start_idx = 0

    for i, group_id in enumerate(groups_array):
        if group_id != current_group:
            # Group changed - record previous group's boundary
            if current_group is not None:
                boundaries[current_group] = (start_idx, i)

            # Start new group
            current_group = group_id
            start_idx = i
            unique_groups.append(group_id)

    # Don't forget the last group
    if current_group is not None:
        boundaries[current_group] = (start_idx, len(groups_array))

    return boundaries


def split_by_groups(
    n_samples: int,
    groups: pl.Series | pd.Series | NDArray[Any],
    test_group_indices: list[int],
    all_group_ids: list[Any],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Split samples into train/test based on group assignments.

    This creates a complete split where all samples from specified test groups
    go to the test set, and all other samples go to the training set.

    Parameters
    ----------
    n_samples : int
        Total number of samples.

    groups : array-like
        Group labels for each sample.

    test_group_indices : list of int
        Indices into `all_group_ids` specifying which groups go to test.

    all_group_ids : list
        Sorted list of all unique group IDs in the dataset.

    Returns
    -------
    train_indices : ndarray
        Indices of samples in training set.

    test_indices : ndarray
        Indices of samples in test set.

    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    >>> all_groups = ['A', 'B', 'C']
    >>> train_idx, test_idx = split_by_groups(
    ...     n_samples=6,
    ...     groups=groups,
    ...     test_group_indices=[2],  # Group 'C'
    ...     all_group_ids=all_groups
    ... )
    >>> assert set(groups[train_idx]) == {'A', 'B'}
    >>> assert set(groups[test_idx]) == {'C'}
    """
    # Convert groups to numpy array
    groups_array = DataFrameAdapter.to_numpy(groups).flatten()

    # Get test group IDs
    test_group_ids = {all_group_ids[i] for i in test_group_indices}

    # Create masks
    test_mask = np.isin(groups_array, list(test_group_ids))
    train_mask = ~test_mask

    # Get indices
    train_indices = np.where(train_mask)[0].astype(np.intp)
    test_indices = np.where(test_mask)[0].astype(np.intp)

    return train_indices, test_indices


def count_samples_per_group(
    groups: pl.Series | pd.Series | NDArray[Any],
) -> dict[Any, int]:
    """Count number of samples for each unique group.

    Useful for understanding group distribution and detecting imbalanced groups.

    Parameters
    ----------
    groups : array-like
        Group labels for each sample.

    Returns
    -------
    counts : dict
        Mapping from group ID to sample count.

    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['A', 'A', 'A', 'B', 'B', 'C'])
    >>> counts = count_samples_per_group(groups)
    >>> assert counts == {'A': 3, 'B': 2, 'C': 1}
    """
    # Convert groups to numpy array
    groups_array = DataFrameAdapter.to_numpy(groups).flatten()

    # Count using numpy unique
    unique_groups, counts = np.unique(groups_array, return_counts=True)

    return dict(zip(unique_groups, counts, strict=False))


# Make functions available at module level
__all__ = [
    "validate_group_isolation",
    "isolate_groups_from_train",
    "get_group_boundaries",
    "split_by_groups",
    "count_samples_per_group",
]
