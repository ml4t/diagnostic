"""Dashboard components for rich visualization.

This package provides the dashboard architecture for creating interactive,
multi-tab analytical dashboards. Dashboards compose multiple visualizations
with interactive controls (tabs, dropdowns, filters) into cohesive analytical
experiences.

Architecture:
    - BaseDashboard: Abstract base class defining dashboard interface
    - DashboardSection: Container for a single dashboard section (tab)
    - FeatureImportanceDashboard: Multi-tab importance analysis
    - FeatureInteractionDashboard: Network and matrix interaction views

Design Principles:
    - Progressive disclosure: Summary → Detail → Deep-dive
    - Modular composition: Dashboards work standalone or compose
    - Interactive controls: Tabs, dropdowns, filters, drill-down
    - LLM-ready: Structured data enables future interpretation
    - Professional output: Publication-quality HTML with embedded JS
"""

from .base import THEMES, BaseDashboard, DashboardSection, get_theme
from .importance import FeatureImportanceDashboard
from .interaction import FeatureInteractionDashboard

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
