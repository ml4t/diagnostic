"""Normalization functions for SHAP vector clustering.

Provides L1, L2, and standardization normalization with proper
handling of edge cases (zero vectors, zero variance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


NormalizationType = Literal["l1", "l2", "standardize", "none"]


def normalize_l1(vectors: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """L1 normalization: Scale each row by sum of absolute values.

    Args:
        vectors: Input vectors of shape (n_samples, n_features)

    Returns:
        L1-normalized vectors where each row sums to 1.0 (in absolute terms)

    Note:
        Zero vectors are returned unchanged (no division by zero)
    """
    l1_norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
    l1_norms = np.where(l1_norms == 0, 1.0, l1_norms)
    return vectors / l1_norms


def normalize_l2(vectors: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """L2 normalization: Scale each row to unit Euclidean norm.

    Args:
        vectors: Input vectors of shape (n_samples, n_features)

    Returns:
        L2-normalized unit vectors (norm = 1.0 per row)

    Note:
        Zero vectors are returned unchanged (no division by zero)
    """
    l2_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    l2_norms = np.where(l2_norms == 0, 1.0, l2_norms)
    return vectors / l2_norms


def standardize(vectors: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Z-score standardization: (x - mean) / std per feature.

    Args:
        vectors: Input vectors of shape (n_samples, n_features)

    Returns:
        Standardized vectors (mean=0, std=1 per feature column)

    Note:
        Zero-variance features are returned unchanged
    """
    mean = np.mean(vectors, axis=0, keepdims=True)
    std = np.std(vectors, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (vectors - mean) / std


def normalize(
    vectors: NDArray[np.floating[Any]],
    method: NormalizationType | None = None,
) -> NDArray[np.floating[Any]]:
    """Apply normalization to vectors.

    Args:
        vectors: Input vectors of shape (n_samples, n_features)
        method: Normalization method: 'l1', 'l2', 'standardize', 'none', or None

    Returns:
        Normalized vectors

    Raises:
        ValueError: If normalization produces NaN/Inf or method is unknown

    Example:
        >>> vectors = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalize(vectors, method='l2')
        array([[0.267, 0.535, 0.802],
               [0.456, 0.570, 0.684]])
    """
    if method is None or method == "none":
        return vectors.copy()
    elif method == "l1":
        normalized = normalize_l1(vectors)
    elif method == "l2":
        normalized = normalize_l2(vectors)
    elif method == "standardize":
        normalized = standardize(vectors)
    else:
        raise ValueError(
            f"Invalid normalization method: '{method}'. "
            "Valid options: 'l1', 'l2', 'standardize', 'none', None"
        )

    # Validate output
    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalization produced NaN or Inf values. "
            "This may indicate zero-variance features or numerical instability. "
            f"Normalization method: {method}"
        )

    return normalized
