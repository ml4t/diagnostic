"""Shared formatting helpers."""

from __future__ import annotations

import math


def format_finite(value: float, format_spec: str, na: str = "N/A") -> str:
    """Format a finite number and use a sentinel for unavailable values."""
    return format(value, format_spec) if math.isfinite(value) else na
