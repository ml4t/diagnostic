"""Base classes and utilities for dashboard components.

This module provides the foundation for creating interactive,
multi-tab analytical dashboards with consistent styling.

Classes:
    BaseDashboard: Abstract base class defining dashboard interface
    DashboardSection: Container for a single dashboard section (tab)

Functions:
    get_theme: Get theme configuration by name
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import plotly.graph_objects as go

from ...evaluation.themes import DARK_TEMPLATE, DEFAULT_TEMPLATE

if TYPE_CHECKING:
    pass

# Theme mapping for dashboard use
THEMES = {
    "light": {"template": DEFAULT_TEMPLATE, "plot_bgcolor": "#ffffff", "font_color": "#000000"},
    "dark": {"template": DARK_TEMPLATE, "plot_bgcolor": "#1e1e1e", "font_color": "#ffffff"},
}


def get_theme(theme_name: str) -> dict:
    """Get theme configuration.

    Parameters
    ----------
    theme_name : str
        Theme name ('light' or 'dark')

    Returns
    -------
    dict
        Theme configuration with template, plot_bgcolor, font_color
    """
    return THEMES.get(theme_name, THEMES["light"])


class DashboardSection:
    """Container for a single dashboard section (tab).

    Each section represents one view or perspective on the analysis,
    typically containing multiple plots, tables, or text content.
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        content: str = "",
        plots: list[go.Figure] | None = None,
    ):
        """Initialize dashboard section.

        Parameters
        ----------
        title : str
            Section title (shown in tab or header)
        description : str, default=""
            Optional description text (HTML supported)
        content : str, default=""
            Section content (HTML)
        plots : list of go.Figure, optional
            Plotly figures to include in section
        """
        self.title = title
        self.description = (
            f'<div class="section-description">{description}</div>' if description else ""
        )
        self.content = content
        self.plots = plots or []

    def add_plot(self, fig: go.Figure, container_id: str | None = None) -> None:
        """Add a Plotly figure to this section.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add
        container_id : str, optional
            HTML id for plot container. Auto-generated if None.
        """
        self.plots.append(fig)

        # Generate plot HTML and append to content
        plot_id = container_id or f"plot-{len(self.plots)}"
        plot_html = fig.to_html(include_plotlyjs=False, div_id=plot_id, config={"responsive": True})

        self.content += f'<div class="plot-container">{plot_html}</div>'

    def add_html(self, html: str) -> None:
        """Add custom HTML content to section.

        Parameters
        ----------
        html : str
            HTML content to append
        """
        self.content += html


class BaseDashboard(ABC):
    """Abstract base class for interactive dashboards.

    All dashboards follow a common pattern:
    1. Extract structured data from analysis results
    2. Create multiple visualizations (plots, tables)
    3. Compose into interactive HTML with tabs/controls
    4. Optionally export to PDF or JSON

    Subclasses must implement:
        - generate(): Transform raw results into complete HTML
    """

    def __init__(
        self,
        title: str,
        theme: Literal["light", "dark"] = "light",
        width: int | None = None,
        height: int | None = None,
    ):
        """Initialize dashboard.

        Parameters
        ----------
        title : str
            Dashboard title (displayed at top)
        theme : {'light', 'dark'}, default='light'
            Visual theme for all plots and styling
        width : int, optional
            Dashboard width in pixels. If None, uses responsive width.
        height : int, optional
            Dashboard height in pixels. If None, uses auto height.
        """
        self.title = title
        self.theme = theme
        self.width = width
        self.height = height
        self.theme_config = THEMES[theme]
        self.created_at = datetime.now()

        # Sections storage (populated by subclasses)
        self.sections: list[DashboardSection] = []

    @abstractmethod
    def generate(self, analysis_results: Any, **kwargs: Any) -> str:
        """Generate complete dashboard HTML from analysis results.

        Parameters
        ----------
        analysis_results : Any
            Raw analysis results (format depends on dashboard type)
        **kwargs
            Additional dashboard-specific parameters

        Returns
        -------
        str
            Complete HTML document with embedded CSS/JS
        """
        pass

    def save(self, output_path: str, analysis_results: Any, **kwargs: Any) -> str:
        """Generate and save dashboard to file.

        Parameters
        ----------
        output_path : str
            Path for output HTML file
        analysis_results : Any
            Raw analysis results (format depends on dashboard type)
        **kwargs
            Additional dashboard-specific parameters

        Returns
        -------
        str
            Path to saved file
        """
        html = self.generate(analysis_results, **kwargs)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def _build_header(self) -> str:
        """Build dashboard header HTML."""
        return f"""
        <div class="dashboard-header">
            <h1>{self.title}</h1>
            <p class="timestamp">Generated: {self.created_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        """

    def _build_navigation(self) -> str:
        """Build tab navigation HTML."""
        if len(self.sections) <= 1:
            return ""  # No navigation needed for single-section dashboards

        tabs_html = []
        for i, section in enumerate(self.sections):
            active_class = "active" if i == 0 else ""
            tabs_html.append(
                f'<button class="tab-button {active_class}" onclick="switchTab(event, \'section-{i}\')">'
                f"{section.title}</button>"
            )

        return f"""
        <div class="tab-navigation">
            {"".join(tabs_html)}
        </div>
        """

    def _build_sections(self) -> str:
        """Build all dashboard sections HTML."""
        sections_html = []

        for i, section in enumerate(self.sections):
            active_class = "active" if i == 0 else ""
            sections_html.append(f"""
            <div id="section-{i}" class="tab-content {active_class}">
                <h2>{section.title}</h2>
                {section.description}
                {section.content}
            </div>
            """)

        return "".join(sections_html)

    def _get_base_styles(self) -> str:
        """Get base CSS styles for dashboard."""
        bg_color = self.theme_config["plot_bgcolor"]
        text_color = self.theme_config["font_color"]
        border_color = "#555" if self.theme == "dark" else "#ddd"

        return f"""
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: {bg_color};
                color: {text_color};
            }}

            .dashboard-header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid {border_color};
            }}

            .dashboard-header h1 {{
                margin: 0;
                font-size: 2em;
                font-weight: 600;
            }}

            .timestamp {{
                margin: 10px 0 0 0;
                font-size: 0.9em;
                opacity: 0.7;
            }}

            .tab-navigation {{
                display: flex;
                gap: 5px;
                margin-bottom: 20px;
                border-bottom: 2px solid {border_color};
            }}

            .tab-button {{
                padding: 12px 24px;
                background: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                cursor: pointer;
                font-size: 1em;
                color: {text_color};
                transition: all 0.3s ease;
            }}

            .tab-button:hover {{
                background-color: {"rgba(255,255,255,0.05)" if self.theme == "dark" else "rgba(0,0,0,0.05)"};
            }}

            .tab-button.active {{
                border-bottom-color: #1f77b4;
                font-weight: 600;
            }}

            .tab-content {{
                display: none;
                animation: fadeIn 0.3s;
            }}

            .tab-content.active {{
                display: block;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            .section-description {{
                margin: 10px 0 20px 0;
                padding: 15px;
                background-color: {"rgba(255,255,255,0.05)" if self.theme == "dark" else "rgba(0,0,0,0.05)"};
                border-left: 4px solid #1f77b4;
                border-radius: 4px;
            }}

            .plot-container {{
                margin: 20px 0;
            }}

            .insights-panel {{
                margin: 30px 0;
                padding: 20px;
                background-color: {"rgba(100,150,255,0.1)" if self.theme == "dark" else "rgba(100,150,255,0.05)"};
                border-radius: 8px;
                border: 1px solid {border_color};
            }}

            .insights-panel h3 {{
                margin-top: 0;
                color: #1f77b4;
            }}

            .insights-panel ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}

            .insights-panel li {{
                margin: 8px 0;
                line-height: 1.5;
            }}

            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}

            .metric-card {{
                padding: 15px;
                background-color: {"rgba(255,255,255,0.05)" if self.theme == "dark" else "rgba(0,0,0,0.05)"};
                border-radius: 6px;
                border: 1px solid {border_color};
            }}

            .metric-label {{
                font-size: 0.85em;
                opacity: 0.7;
                margin-bottom: 5px;
            }}

            .metric-value {{
                font-size: 1.5em;
                font-weight: 600;
            }}

            .metric-sublabel {{
                font-size: 0.75em;
                opacity: 0.6;
                margin-top: 5px;
            }}

            .feature-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.95em;
            }}

            .feature-table thead {{
                background-color: {"rgba(255,255,255,0.1)" if self.theme == "dark" else "rgba(0,0,0,0.1)"};
            }}

            .feature-table th {{
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                border-bottom: 2px solid {border_color};
            }}

            .feature-table td {{
                padding: 10px 15px;
                border-bottom: 1px solid {border_color};
            }}

            .feature-table tbody tr:hover {{
                background-color: {"rgba(255,255,255,0.05)" if self.theme == "dark" else "rgba(0,0,0,0.02)"};
            }}

            .badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 600;
            }}

            .badge-high {{
                background-color: rgba(40, 167, 69, 0.2);
                color: #28a745;
            }}

            .badge-medium {{
                background-color: rgba(255, 193, 7, 0.2);
                color: #ffc107;
            }}

            .badge-low {{
                background-color: rgba(220, 53, 69, 0.2);
                color: #dc3545;
            }}

            .badge-n\\/a {{
                background-color: rgba(108, 117, 125, 0.2);
                color: #6c757d;
            }}

            /* Search box styling */
            #feature-search {{
                width: 100%;
                padding: 10px;
                font-size: 16px;
                border: 1px solid {border_color};
                border-radius: 4px;
                margin-bottom: 15px;
                background-color: {bg_color};
                color: {text_color};
            }}

            #feature-search:focus {{
                outline: none;
                border-color: #1f77b4;
                box-shadow: 0 0 5px rgba(31, 119, 180, 0.3);
            }}

            /* Table zebra striping */
            .feature-table tbody tr:nth-child(even) {{
                background-color: {"rgba(255,255,255,0.02)" if self.theme == "dark" else "rgba(0,0,0,0.02)"};
            }}

            .feature-table tbody tr:nth-child(odd) {{
                background-color: transparent;
            }}

            /* Low agreement highlighting */
            .feature-table tbody tr.low-agreement {{
                background-color: {"rgba(255,200,100,0.1)" if self.theme == "dark" else "rgba(255,200,100,0.15)"} !important;
                border-left: 3px solid #ff9800;
            }}

            /* Info icon tooltips */
            .info-icon {{
                display: inline-block;
                width: 16px;
                height: 16px;
                line-height: 16px;
                text-align: center;
                border-radius: 50%;
                background-color: #1f77b4;
                color: white;
                font-size: 12px;
                cursor: help;
                margin-left: 5px;
            }}

            /* Improved tab navigation */
            .tab-navigation {{
                border-bottom: 2px solid {border_color};
                margin-bottom: 30px;
            }}
        </style>
        """

    def _get_base_scripts(self) -> str:
        """Get base JavaScript for interactivity."""
        return """
        <script>
            function switchTab(event, sectionId) {
                // Hide all tab contents
                const contents = document.getElementsByClassName('tab-content');
                for (let content of contents) {
                    content.classList.remove('active');
                }

                // Deactivate all tab buttons
                const buttons = document.getElementsByClassName('tab-button');
                for (let button of buttons) {
                    button.classList.remove('active');
                }

                // Show selected tab
                document.getElementById(sectionId).classList.add('active');
                event.currentTarget.classList.add('active');
            }

            // Plotly responsive resizing
            window.addEventListener('resize', function() {
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(plot => {
                    Plotly.Plots.resize(plot);
                });
            });

            // Table sorting functionality
            document.addEventListener('DOMContentLoaded', function() {
                const table = document.getElementById('feature-importance-table');
                if (!table) return;

                const headers = table.querySelectorAll('thead th');
                let sortDirection = {};  // Track sort direction for each column

                headers.forEach((header, colIndex) => {
                    header.style.cursor = 'pointer';
                    header.style.userSelect = 'none';
                    sortDirection[colIndex] = 1;  // 1 for ascending, -1 for descending

                    header.addEventListener('click', function() {
                        const tbody = table.querySelector('tbody');
                        const rows = Array.from(tbody.querySelectorAll('tr'));

                        // Sort rows
                        rows.sort((a, b) => {
                            let aValue = a.cells[colIndex].innerText.trim();
                            let bValue = b.cells[colIndex].innerText.trim();

                            // Handle different data types
                            // Try parsing as number first
                            const aNum = parseFloat(aValue.replace('%', '').replace(',', ''));
                            const bNum = parseFloat(bValue.replace('%', '').replace(',', ''));

                            if (!isNaN(aNum) && !isNaN(bNum)) {
                                return (aNum - bNum) * sortDirection[colIndex];
                            } else {
                                // String comparison
                                return aValue.localeCompare(bValue) * sortDirection[colIndex];
                            }
                        });

                        // Clear and repopulate tbody
                        tbody.innerHTML = '';
                        rows.forEach(row => tbody.appendChild(row));

                        // Toggle sort direction for next click
                        sortDirection[colIndex] *= -1;

                        // Visual indicator
                        headers.forEach(h => h.style.opacity = '0.7');
                        header.style.opacity = '1.0';
                    });
                });
            });
        </script>
        """


__all__ = [
    "THEMES",
    "get_theme",
    "BaseDashboard",
    "DashboardSection",
]
