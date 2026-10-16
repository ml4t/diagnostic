"""Shared formatting helpers."""

from __future__ import annotations

import math


def format_finite(value: float | None, format_spec: str) -> str:
    """Format a finite number and use a sentinel for unavailable values."""
    return format(value, format_spec) if value is not None and math.isfinite(value) else "N/A"
