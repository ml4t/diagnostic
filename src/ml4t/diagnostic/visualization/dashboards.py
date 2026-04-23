"""Interactive dashboard components for rich visualization.

Architecture:
    - BaseDashboard: Abstract base class defining dashboard interface
    - FeatureImportanceDashboard: Multi-tab importance analysis
    - FeatureInteractionDashboard: Dedicated interaction report shell

Design Principles:
    - Progressive disclosure: Summary → Detail → Deep-dive
    - Modular composition: Dashboards work standalone or compose
    - Interactive controls: Tabs, dropdowns, filters, drill-down
    - Professional output: Publication-quality HTML with embedded JS
"""

from __future__ import annotations

# Re-export everything from the decomposed package
from ml4t.diagnostic.visualization.dashboards import (
    THEMES,
    BaseDashboard,
    DashboardSection,
    FeatureImportanceDashboard,
    FeatureInteractionDashboard,
    get_theme,
)

__all__ = [
    # Theme utilities
    "THEMES",
    "get_theme",
    # Base classes
    "BaseDashboard",
    "DashboardSection",
    # Dashboard implementations
    "FeatureImportanceDashboard",
    "FeatureInteractionDashboard",
]
