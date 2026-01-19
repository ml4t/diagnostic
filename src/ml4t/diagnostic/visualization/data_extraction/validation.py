"""Validation helpers for data extraction.

Provides length and dimension validation for extracted visualization data.
"""

from __future__ import annotations

import numpy as np


def _validate_lengths_match(
    *arrays: tuple[str, list | np.ndarray],
) -> None:
    """Validate that all provided arrays have matching lengths.

    Parameters
    ----------
    *arrays : tuple[str, list | np.ndarray]
        Tuples of (name, array) to validate.

    Raises
    ------
    ValueError
        If arrays have different lengths.
    """
    if not arrays:
        return

    lengths = [(name, len(arr)) for name, arr in arrays]
    unique_lengths = {length for _, length in lengths}

    if len(unique_lengths) > 1:
        length_info = ", ".join(f"{name}={length}" for name, length in lengths)
        raise ValueError(
            f"Length mismatch in data extraction: {length_info}. "
            "All arrays must have the same length for consistent visualization."
        )


def _validate_matrix_feature_alignment(matrix: np.ndarray, feature_names: list[str]) -> None:
    """Validate that interaction matrix dimensions match feature names.

    Parameters
    ----------
    matrix : np.ndarray
        Square interaction matrix.
    feature_names : list[str]
        Feature names for matrix axes.

    Raises
    ------
    ValueError
        If matrix is not square or dimensions don't match feature count.
    """
    n_features = len(feature_names)
    if matrix.ndim != 2:
        raise ValueError(
            f"Interaction matrix must be 2D, got {matrix.ndim}D with shape {matrix.shape}"
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Interaction matrix must be square, got shape {matrix.shape}")
    if matrix.shape[0] != n_features:
        raise ValueError(
            f"Interaction matrix size ({matrix.shape[0]}) does not match "
            f"number of features ({n_features})"
        )
