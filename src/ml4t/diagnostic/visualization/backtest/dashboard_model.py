"""Typed content model for backtest dashboard rendering.

Provides a lightweight presentation model so the integrated dashboard and
focused exports can share one content graph instead of building HTML directly
from section dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .template_system import TearsheetSection


@dataclass(frozen=True)
class BacktestWorkspaceSpec:
    """Definition of a dashboard workspace."""

    id: str
    title: str
    description: str = ""


@dataclass
class BacktestSectionContent:
    """Rendered content for a tearsheet section."""

    section: TearsheetSection
    html: str


@dataclass
class BacktestWorkspaceContent:
    """Rendered workspace made up of multiple tearsheet sections."""

    spec: BacktestWorkspaceSpec
    sections: list[BacktestSectionContent] = field(default_factory=list)

    @property
    def is_visible(self) -> bool:
        """Return whether the workspace has any rendered content."""
        return bool(self.sections)


@dataclass
class BacktestDashboardModel:
    """Presentation-ready dashboard model for a backtest report."""

    workspaces: list[BacktestWorkspaceContent] = field(default_factory=list)

    def visible_workspaces(self) -> list[BacktestWorkspaceContent]:
        """Return visible workspaces in display order."""
        return [workspace for workspace in self.workspaces if workspace.is_visible]
