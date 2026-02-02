"""Fold persistence for cross-validation reproducibility.

This module provides utilities for saving and loading cross-validation fold
configurations, enabling reproducible research and efficient caching of expensive
split computations (especially for CPCV with many combinations).

Examples
--------
>>> from ml4t.diagnostic.splitters import WalkForwardCV
>>> from ml4t.diagnostic.splitters.persistence import save_folds, load_folds
>>>
>>> # Save fold configuration
>>> cv = WalkForwardCV(n_splits=5, test_size=100)
>>> folds = list(cv.split(X))
>>> save_folds(folds, X, "my_folds.json", metadata={"strategy": "walk_forward"})
>>>
>>> # Load and reuse fold configuration
>>> loaded_folds, metadata = load_folds("my_folds.json")
>>> for train_idx, test_idx in loaded_folds:
>>>     # Use same splits as original
>>>     pass
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import NDArray

from ml4t.diagnostic.config.base import BaseConfig


def save_folds(
    folds: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    X: NDArray[np.floating] | pd.DataFrame | pl.DataFrame,
    filepath: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    include_timestamps: bool = True,
) -> None:
    """Save cross-validation folds to disk.

    Parameters
    ----------
    folds : list[tuple[NDArray, NDArray]]
        List of (train_indices, test_indices) tuples from CV splitter.
    X : array-like or DataFrame
        Original data used for splitting (for timestamp extraction if DataFrame).
    filepath : str or Path
        Path to save fold configuration (JSON format).
    metadata : dict, optional
        Additional metadata to store (e.g., splitter config, data info).
    include_timestamps : bool, default=True
        If True and X is a DataFrame with DatetimeIndex, save timestamps
        alongside indices for better human readability.

    Examples
    --------
    >>> from ml4t.diagnostic.splitters import WalkForwardCV
    >>> cv = WalkForwardCV(n_splits=5, test_size=100)
    >>> folds = list(cv.split(X))
    >>> save_folds(folds, X, "cv_folds.json", metadata={"n_splits": 5})
    """
    filepath = Path(filepath)

    # Extract timestamps if available
    timestamps = None
    if include_timestamps and isinstance(X, pd.DataFrame | pd.Series):
        if isinstance(X.index, pd.DatetimeIndex):
            timestamps = X.index.astype(str).tolist()
    elif include_timestamps and isinstance(X, pl.DataFrame):
        # Polars doesn't have index, check if first column is datetime
        first_col = X.columns[0]
        if X[first_col].dtype == pl.Datetime:
            timestamps = X[first_col].cast(pl.Utf8).to_list()

    # Build fold data structure
    fold_data: dict[str, Any] = {
        "version": "1.0",
        "n_folds": len(folds),
        "n_samples": len(X),
        "folds": [],
        "metadata": metadata or {},
    }

    if timestamps:
        fold_data["timestamps"] = timestamps

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_info = {
            "fold_id": fold_idx,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        }

        # Add timestamp ranges if available (handle empty folds)
        if timestamps:
            if len(train_idx) > 0:
                fold_info["train_start"] = timestamps[train_idx[0]]
                fold_info["train_end"] = timestamps[train_idx[-1]]
            else:
                fold_info["train_start"] = None
                fold_info["train_end"] = None

            if len(test_idx) > 0:
                fold_info["test_start"] = timestamps[test_idx[0]]
                fold_info["test_end"] = timestamps[test_idx[-1]]
            else:
                fold_info["test_start"] = None
                fold_info["test_end"] = None

        fold_data["folds"].append(fold_info)

    # Save to JSON
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        json.dump(fold_data, f, indent=2)


def load_folds(
    filepath: str | Path,
) -> tuple[list[tuple[NDArray[np.int_], NDArray[np.int_]]], dict[str, Any]]:
    """Load cross-validation folds from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to saved fold configuration (JSON format).

    Returns
    -------
    folds : list[tuple[NDArray, NDArray]]
        List of (train_indices, test_indices) tuples.
    metadata : dict
        Metadata dictionary stored with folds.

    Examples
    --------
    >>> folds, metadata = load_folds("cv_folds.json")
    >>> print(f"Loaded {len(folds)} folds")
    >>> print(f"Metadata: {metadata}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Fold file not found: {filepath}")

    with filepath.open("r") as f:
        fold_data = json.load(f)

    # Validate version
    if fold_data.get("version") != "1.0":
        raise ValueError(f"Unsupported fold file version: {fold_data.get('version')}")

    # Reconstruct folds
    folds = []
    for fold_info in fold_data["folds"]:
        train_idx = np.array(fold_info["train_indices"], dtype=np.int_)
        test_idx = np.array(fold_info["test_indices"], dtype=np.int_)
        folds.append((train_idx, test_idx))

    metadata = fold_data.get("metadata", {})

    return folds, metadata


def save_config(
    config: Any,  # SplitterConfig or subclass
    filepath: str | Path,
) -> None:
    """Save splitter configuration to disk.

    This is a convenience wrapper around config.to_json() for consistency
    with the persistence API.

    Parameters
    ----------
    config : SplitterConfig
        Configuration object to save.
    filepath : str or Path
        Path to save configuration (JSON format).

    Examples
    --------
    >>> from ml4t.diagnostic.splitters.config import WalkForwardConfig
    >>> config = WalkForwardConfig(n_splits=5, test_size=100)
    >>> save_config(config, "cv_config.json")
    """
    filepath = Path(filepath)
    config.to_json(filepath)


def load_config(
    filepath: str | Path,
    config_class: type[BaseConfig],
) -> BaseConfig:
    """Load splitter configuration from disk.

    This is a convenience wrapper around config_class.from_json() for consistency
    with the persistence API.

    Parameters
    ----------
    filepath : str or Path
        Path to saved configuration (JSON format).
    config_class : type
        Configuration class to instantiate (e.g., WalkForwardConfig).

    Returns
    -------
    config : SplitterConfig
        Loaded configuration object.

    Examples
    --------
    >>> from ml4t.diagnostic.splitters.config import WalkForwardConfig
    >>> config = load_config("cv_config.json", WalkForwardConfig)
    >>> print(config.n_splits)
    """
    filepath = Path(filepath)
    return config_class.from_json(filepath)


def verify_folds(
    folds: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    n_samples: int,
) -> dict[str, Any]:
    """Verify fold integrity and compute statistics.

    Parameters
    ----------
    folds : list[tuple[NDArray, NDArray]]
        List of (train_indices, test_indices) tuples.
    n_samples : int
        Total number of samples in dataset.

    Returns
    -------
    stats : dict
        Dictionary containing fold statistics and validation results.

    Examples
    --------
    >>> folds, _ = load_folds("cv_folds.json")
    >>> stats = verify_folds(folds, n_samples=1000)
    >>> print(f"Valid: {stats['valid']}")
    >>> print(f"Coverage: {stats['coverage']:.1%}")
    """
    stats: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "n_folds": len(folds),
        "n_samples": n_samples,
        "train_sizes": [],
        "test_sizes": [],
    }

    all_train_indices: set[int] = set()
    all_test_indices: set[int] = set()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        stats["train_sizes"].append(len(train_idx))
        stats["test_sizes"].append(len(test_idx))

        # Check for index overlap within fold
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            stats["valid"] = False
            stats["errors"].append(
                f"Fold {fold_idx}: {len(overlap)} overlapping indices between train and test"
            )

        # Check for out-of-range indices
        if np.any(train_idx < 0) or np.any(train_idx >= n_samples):
            stats["valid"] = False
            stats["errors"].append(f"Fold {fold_idx}: Train indices out of range")

        if np.any(test_idx < 0) or np.any(test_idx >= n_samples):
            stats["valid"] = False
            stats["errors"].append(f"Fold {fold_idx}: Test indices out of range")

        all_train_indices.update(train_idx)
        all_test_indices.update(test_idx)

    # Compute coverage statistics
    all_indices = all_train_indices | all_test_indices
    stats["coverage"] = len(all_indices) / n_samples
    stats["train_coverage"] = len(all_train_indices) / n_samples
    stats["test_coverage"] = len(all_test_indices) / n_samples

    # Compute size statistics
    if stats["train_sizes"]:
        train_sizes: list[int] = stats["train_sizes"]
        test_sizes: list[int] = stats["test_sizes"]
        stats["avg_train_size"] = np.mean(train_sizes)
        stats["std_train_size"] = np.std(train_sizes)
        stats["avg_test_size"] = np.mean(test_sizes)
        stats["std_test_size"] = np.std(test_sizes)

    return stats


__all__ = [
    "save_folds",
    "load_folds",
    "save_config",
    "load_config",
    "verify_folds",
]
