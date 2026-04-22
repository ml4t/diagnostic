"""Information Coefficient metrics.

This module provides the short public import path for IC helpers. The existing
``ml4t.diagnostic.evaluation.metrics`` path remains available for compatibility.
"""

from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_by_horizon,
    compute_ic_ir,
    compute_ic_series,
    cross_sectional_ic,
    cross_sectional_ic_series,
    information_coefficient,
    pooled_ic,
)

__all__ = [
    "pooled_ic",
    "information_coefficient",
    "cross_sectional_ic",
    "cross_sectional_ic_series",
    "compute_ic_series",
    "compute_ic_by_horizon",
    "compute_ic_ir",
]
