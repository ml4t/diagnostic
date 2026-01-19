"""Validation helper functions for barrier analysis results.

This module provides validation utilities for quantile-keyed dictionaries
used across all barrier result classes.
"""

from __future__ import annotations

from typing import Any


def _validate_quantile_dict_keys(
    quantile_labels: list[str],
    dicts: list[tuple[str, dict[str, Any]]],
) -> None:
    """Validate that all quantile-keyed dicts have the expected keys.

    Parameters
    ----------
    quantile_labels : list[str]
        Expected quantile labels (keys).
    dicts : list[tuple[str, dict]]
        List of (field_name, dict) tuples to validate.

    Raises
    ------
    ValueError
        If any dict has different keys than quantile_labels.
    """
    expected_keys = set(quantile_labels)
    for name, d in dicts:
        actual_keys = set(d.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            raise ValueError(
                f"Key mismatch in '{name}': missing={missing or 'none'}, extra={extra or 'none'}"
            )
