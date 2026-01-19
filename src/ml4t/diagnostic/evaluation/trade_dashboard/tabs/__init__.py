"""Dashboard tab modules.

Each tab module provides a `render_tab(st, bundle)` function that renders
the tab content using the Streamlit instance and DashboardBundle.
"""

from __future__ import annotations

from ml4t.diagnostic.evaluation.trade_dashboard.tabs import (
    patterns,
    shap_analysis,
    stat_validation,
    worst_trades,
)

__all__ = [
    "patterns",
    "shap_analysis",
    "stat_validation",
    "worst_trades",
]
