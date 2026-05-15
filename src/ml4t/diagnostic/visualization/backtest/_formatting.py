"""Shared display-formatting helpers for backtest visualizations."""

from __future__ import annotations


def format_currency_adaptive(value: float) -> str:
    """Format currency with magnitude-sensitive precision."""
    abs_value = abs(value)
    if abs_value >= 100:
        return f"${value:,.0f}"
    if abs_value >= 1:
        return f"${value:,.2f}"
    if abs_value >= 0.01:
        return f"${value:,.4f}"
    return f"${value:.2e}"


def format_percent_adaptive(value: float) -> str:
    """Format percentages with enough precision for small non-zero values."""
    abs_value = abs(value)
    if abs_value >= 1:
        return f"{value:.1f}%"
    if abs_value >= 0.01:
        return f"{value:.2f}%"
    if abs_value > 0:
        return f"{value:.2g}%"
    return "0%"
