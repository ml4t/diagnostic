"""Trade SHAP diagnostics dashboard package.

This package provides an interactive Streamlit dashboard for visualizing
Trade-SHAP analysis results, including statistical validation, worst trades,
SHAP explanations, and error patterns.

Usage:
    >>> from ml4t.diagnostic.evaluation.trade_dashboard import run_diagnostics_dashboard
    >>> run_diagnostics_dashboard(result)
"""

from __future__ import annotations

from ml4t.diagnostic.evaluation.trade_dashboard.app import run_dashboard
from ml4t.diagnostic.evaluation.trade_dashboard.normalize import normalize_result
from ml4t.diagnostic.evaluation.trade_dashboard.types import (
    DashboardBundle,
    DashboardConfig,
    ReturnSummary,
)

# Backward compatibility alias
run_diagnostics_dashboard = run_dashboard

__all__ = [
    "DashboardBundle",
    "DashboardConfig",
    "ReturnSummary",
    "normalize_result",
    "run_dashboard",
    "run_diagnostics_dashboard",  # Backward compat alias
]
