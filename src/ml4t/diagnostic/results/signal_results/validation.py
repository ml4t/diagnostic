"""Validation helper functions for signal result classes.

This module provides utility functions for validating dictionary key consistency
and normalizing period strings used in signal analysis results.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

from typing import Any


def _validate_dict_keys_match(
    data: dict[str, Any],
    required_fields: list[str],
    optional_fields: list[str] | None = None,
    reference_field: str | None = None,
) -> None:
    """Validate that all dict fields share the same keys.

    Parameters
    ----------
    data : dict
        Model data dictionary.
    required_fields : list[str]
        Required dict field names that must all share the same keys.
    optional_fields : list[str] | None
        Optional dict field names that, if present and not None, must also share the same keys.
    reference_field : str | None
        Field to use as reference for key set. If None, uses first required field.

    Raises
    ------
    ValueError
        If any dict field has different keys than the reference.
    """
    if not required_fields:
        return

    ref_field = reference_field or required_fields[0]
    ref_keys = set(data.get(ref_field, {}).keys())

    if not ref_keys:
        return  # Empty reference, nothing to validate

    # Check required fields
    for field in required_fields:
        if field == ref_field:
            continue
        field_data = data.get(field)
        if field_data is None:
            raise ValueError(
                f"Required field '{field}' is None but '{ref_field}' has keys: {ref_keys}"
            )
        field_keys = set(field_data.keys())
        if field_keys != ref_keys:
            missing = ref_keys - field_keys
            extra = field_keys - ref_keys
            raise ValueError(
                f"Key mismatch in '{field}': "
                f"missing={missing or 'none'}, extra={extra or 'none'} "
                f"(reference: '{ref_field}')"
            )

    # Check optional fields (only if they exist and are not None)
    for field in optional_fields or []:
        field_data = data.get(field)
        if field_data is None:
            continue
        field_keys = set(field_data.keys())
        if field_keys != ref_keys:
            missing = ref_keys - field_keys
            extra = field_keys - ref_keys
            raise ValueError(
                f"Key mismatch in '{field}': "
                f"missing={missing or 'none'}, extra={extra or 'none'} "
                f"(reference: '{ref_field}')"
            )


def _normalize_period(period: int | str) -> str:
    """Normalize period to canonical string format used internally.

    Accepts:
    - int: 21 -> "21D"
    - str without suffix: "21" -> "21D"
    - str with suffix: "21D" -> "21D"

    Parameters
    ----------
    period : int | str
        Period as integer or string, with or without 'D' suffix.

    Returns
    -------
    str
        Canonical period key with 'D' suffix (e.g., "21D").

    Examples
    --------
    >>> _normalize_period(21)
    '21D'
    >>> _normalize_period('21')
    '21D'
    >>> _normalize_period('21D')
    '21D'
    """
    if isinstance(period, int):
        return f"{period}D"
    period_str = str(period).strip()
    if period_str.endswith("D"):
        return period_str
    return f"{period_str}D"


def _figure_from_data(data: dict | str) -> Any:
    """Convert figure data to Plotly Figure.

    Handles both dict (direct) and JSON string formats transparently.
    This fixes the type ambiguity where figures may be stored as either
    Python dicts or JSON strings.

    Parameters
    ----------
    data : dict | str
        Figure data as Python dict or JSON string.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly Figure object.
    """
    import plotly.io as pio

    if isinstance(data, str):
        # Already JSON string
        return pio.from_json(data)
    elif isinstance(data, dict):
        # Python dict - convert directly to Figure
        import plotly.graph_objects as go

        return go.Figure(data)
    else:
        raise TypeError(f"Expected dict or str for figure data, got {type(data)}")
