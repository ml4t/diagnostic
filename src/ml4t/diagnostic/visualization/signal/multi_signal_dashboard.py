"""Multi-Signal Analysis Dashboard - Multi-tab interactive HTML dashboard.

This module provides the MultiSignalDashboard class for creating comprehensive,
self-contained HTML dashboards for multi-signal comparison and analysis.

The dashboard follows the Focus+Context pattern with 5 tabs:
1. Summary - Key metrics cards, searchable/sortable table of all signals
2. Distribution - IC ridge plot, ranking bar chart
3. Correlation - Signal correlation cluster heatmap
4. Efficiency - Pareto frontier scatter (IC IR vs Turnover)
5. Comparison (optional) - Side-by-side tear sheets for selected signals

References
----------
Tufte, E. (1983). "The Visual Display of Quantitative Information"
Few, S. (2012). "Show Me the Numbers"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from ml4t.diagnostic.visualization.dashboards import (
    BaseDashboard,
    DashboardSection,
)
from ml4t.diagnostic.visualization.signal.multi_signal_plots import (
    plot_ic_ridge,
    plot_pareto_frontier,
    plot_signal_correlation_heatmap,
    plot_signal_ranking_bar,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.multi_signal_results import (
        ComparisonResult,
        MultiSignalSummary,
    )


class MultiSignalDashboard(BaseDashboard):
    """Interactive multi-tab dashboard for multi-signal analysis results.

    Creates a self-contained HTML dashboard with comprehensive visualizations
    for analyzing and comparing 50-200 signals simultaneously.

    The dashboard includes 5 tabs:

    1. **Summary**: Metric cards with FDR/FWER counts, searchable signal table
    2. **Distribution**: IC ridge plot showing IC ranges, ranking bar chart
    3. **Correlation**: Hierarchical cluster heatmap revealing redundant signals
    4. **Efficiency**: Pareto frontier scatter (IC IR vs Turnover trade-off)
    5. **Comparison** (optional): Side-by-side metrics for selected signals

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import MultiSignalAnalysis
    >>> from ml4t.diagnostic.visualization.signal import MultiSignalDashboard
    >>>
    >>> # Run multi-signal analysis
    >>> analyzer = MultiSignalAnalysis(signals_dict, price_data)
    >>> summary = analyzer.compute_summary()
    >>> corr_matrix = analyzer.correlation_matrix()
    >>>
    >>> # Create and save dashboard
    >>> dashboard = MultiSignalDashboard(title="Alpha Signal Comparison")
    >>> dashboard.save(
    ...     "multi_signal_dashboard.html",
    ...     summary=summary,
    ...     correlation_matrix=corr_matrix
    ... )

    >>> # Dark theme with comparison
    >>> comparison = analyzer.compare(selection="uncorrelated", n=5)
    >>> dashboard = MultiSignalDashboard(title="Top Signals", theme="dark")
    >>> html = dashboard.generate(
    ...     summary=summary,
    ...     correlation_matrix=corr_matrix,
    ...     comparison=comparison
    ... )

    Notes
    -----
    - Dashboard is self-contained HTML with embedded Plotly.js (via CDN)
    - All visualizations are interactive (zoom, pan, hover)
    - Works offline once loaded (all data embedded)

    See Also
    --------
    MultiSignalAnalysis : Main multi-signal analysis class
    MultiSignalSummary : Result container for multi-signal summary
    ComparisonResult : Result container for signal comparison
    """

    def __init__(
        self,
        title: str = "Multi-Signal Analysis Dashboard",
        theme: Literal["light", "dark"] = "light",
        width: int | None = None,
        height: int | None = None,
    ):
        """Initialize Multi-Signal Analysis Dashboard.

        Parameters
        ----------
        title : str, default="Multi-Signal Analysis Dashboard"
            Dashboard title displayed at top
        theme : {'light', 'dark'}, default='light'
            Visual theme for all plots and styling
        width : int, optional
            Dashboard width in pixels. If None, uses responsive width.
        height : int, optional
            Dashboard height in pixels. If None, uses auto height.
        """
        super().__init__(title, theme, width, height)

    def generate(
        self,
        analysis_results: MultiSignalSummary,
        correlation_matrix: pl.DataFrame | None = None,
        comparison: ComparisonResult | None = None,
        **_kwargs: Any,
    ) -> str:
        """Generate complete dashboard HTML.

        Parameters
        ----------
        analysis_results : MultiSignalSummary
            Results from MultiSignalAnalysis.compute_summary()
        correlation_matrix : pl.DataFrame | None
            Signal correlation matrix from MultiSignalAnalysis.correlation_matrix()
        comparison : ComparisonResult | None
            Optional comparison results for selected signals
        **kwargs
            Additional parameters (currently unused)

        Returns
        -------
        str
            Complete HTML document
        """
        # Clear any previous sections
        self.sections: list[DashboardSection] = []

        # Create tabbed layout
        self._create_tabbed_layout(analysis_results, correlation_matrix, comparison)

        # Compose final HTML
        return self._compose_html()

    def save(
        self,
        output_path: str,
        analysis_results: MultiSignalSummary,
        correlation_matrix: pl.DataFrame | None = None,
        comparison: ComparisonResult | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate and save dashboard to file.

        Parameters
        ----------
        output_path : str
            Path for output HTML file
        analysis_results : MultiSignalSummary
            Results from MultiSignalAnalysis.compute_summary()
        correlation_matrix : pl.DataFrame | None
            Signal correlation matrix
        comparison : ComparisonResult | None
            Optional comparison results
        **kwargs
            Additional parameters passed to generate()

        Returns
        -------
        str
            Path to saved file
        """
        html = self.generate(
            analysis_results,
            correlation_matrix=correlation_matrix,
            comparison=comparison,
            **kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    # =========================================================================
    # Tabbed Layout Methods
    # =========================================================================

    def _create_tabbed_layout(
        self,
        summary: MultiSignalSummary,
        correlation_matrix: pl.DataFrame | None = None,
        comparison: ComparisonResult | None = None,
    ) -> None:
        """Create tabbed dashboard layout."""
        # Tab 1: Summary
        self.sections.append(self._create_summary_tab(summary))

        # Tab 2: Distribution
        self.sections.append(self._create_distribution_tab(summary))

        # Tab 3: Correlation (if matrix provided)
        if correlation_matrix is not None:
            self.sections.append(self._create_correlation_tab(correlation_matrix))

        # Tab 4: Efficiency
        self.sections.append(self._create_efficiency_tab(summary))

        # Tab 5: Comparison (if provided)
        if comparison is not None:
            self.sections.append(self._create_comparison_tab(comparison, summary))

    def _create_summary_tab(self, summary: MultiSignalSummary) -> DashboardSection:
        """Create Summary tab with metric cards and signal table."""
        content_parts = []

        # Metric cards
        content_parts.append(self._build_metric_cards(summary))

        # Signal table (searchable/sortable)
        content_parts.append(self._build_signal_table(summary))

        return DashboardSection(
            title="Summary",
            description=(
                "<p>Overview of all analyzed signals with multiple testing corrections. "
                f"Analyzed <strong>{summary.n_signals}</strong> signals across periods "
                f"<strong>{summary.periods}</strong>.</p>"
            ),
            content="\n".join(content_parts),
        )

    def _create_distribution_tab(self, summary: MultiSignalSummary) -> DashboardSection:
        """Create Distribution tab with IC ridge and ranking plots."""
        content_parts = []

        # IC Ridge Plot
        try:
            fig_ridge = plot_ic_ridge(
                summary,
                max_signals=50,
                sort_by="ic_mean" if "ic_mean" in summary.summary_data else "ic_ir",
                theme=self.theme,
            )
            content_parts.append(
                f'<div class="plot-container">'
                f"{fig_ridge.to_html(full_html=False, include_plotlyjs='cdn')}"
                f"</div>"
            )
        except Exception as e:
            content_parts.append(f'<p class="error">Could not create IC ridge plot: {e}</p>')

        # Ranking Bar Chart
        try:
            metric = "ic_ir" if "ic_ir" in summary.summary_data else "ic_mean"
            fig_bar = plot_signal_ranking_bar(
                summary,
                metric=metric,
                top_n=20,
                theme=self.theme,
            )
            content_parts.append(
                f'<div class="plot-container">'
                f"{fig_bar.to_html(full_html=False, include_plotlyjs=False)}"
                f"</div>"
            )
        except Exception as e:
            content_parts.append(f'<p class="error">Could not create ranking plot: {e}</p>')

        return DashboardSection(
            title="Distribution",
            description=(
                "<p>IC distribution across signals. The ridge plot shows IC range "
                "(5th-95th percentile) with mean indicated. Green indicates FDR-significant.</p>"
            ),
            content="\n".join(content_parts),
        )

    def _create_correlation_tab(self, correlation_matrix: pl.DataFrame) -> DashboardSection:
        """Create Correlation tab with cluster heatmap."""
        content_parts = []

        # Insights about correlation structure
        try:
            corr_values = correlation_matrix.to_numpy()
            n_signals = len(correlation_matrix.columns)

            # Count high correlations (excluding diagonal)
            high_corr_count = 0
            for i in range(n_signals):
                for j in range(i + 1, n_signals):
                    if abs(corr_values[i, j]) > 0.7:
                        high_corr_count += 1

            total_pairs = n_signals * (n_signals - 1) // 2
            pct_redundant = high_corr_count / total_pairs * 100 if total_pairs > 0 else 0

            content_parts.append(
                f'<div class="insights-panel">'
                f"<h3>Correlation Analysis</h3>"
                f"<ul>"
                f"<li><strong>{n_signals}</strong> signals analyzed</li>"
                f"<li><strong>{high_corr_count}</strong> pairs ({pct_redundant:.1f}%) have |correlation| > 0.7</li>"
                f"<li>Highly correlated signals may represent the same underlying alpha</li>"
                f"</ul>"
                f"</div>"
            )
        except Exception:
            pass

        # Correlation heatmap
        try:
            fig_heatmap = plot_signal_correlation_heatmap(
                correlation_matrix,
                cluster=True,
                max_signals=100,
                theme=self.theme,
            )
            content_parts.append(
                f'<div class="plot-container">'
                f"{fig_heatmap.to_html(full_html=False, include_plotlyjs=False)}"
                f"</div>"
            )
        except Exception as e:
            content_parts.append(f'<p class="error">Could not create heatmap: {e}</p>')

        return DashboardSection(
            title="Correlation",
            description=(
                "<p>Signal correlation matrix with hierarchical clustering. "
                "Clustered ordering reveals groups of similar signals - "
                "the '100 signals = 3 unique bets' pattern.</p>"
            ),
            content="\n".join(content_parts),
        )

    def _create_efficiency_tab(self, summary: MultiSignalSummary) -> DashboardSection:
        """Create Efficiency tab with Pareto frontier plot."""
        content_parts = []

        # Check required metrics
        has_turnover = "turnover_mean" in summary.summary_data
        has_ic_ir = "ic_ir" in summary.summary_data

        if has_turnover and has_ic_ir:
            try:
                fig_pareto = plot_pareto_frontier(
                    summary,
                    x_metric="turnover_mean",
                    y_metric="ic_ir",
                    theme=self.theme,
                )
                content_parts.append(
                    f'<div class="plot-container">'
                    f"{fig_pareto.to_html(full_html=False, include_plotlyjs=False)}"
                    f"</div>"
                )
            except Exception as e:
                content_parts.append(f'<p class="error">Could not create Pareto plot: {e}</p>')
        else:
            missing = []
            if not has_turnover:
                missing.append("turnover_mean")
            if not has_ic_ir:
                missing.append("ic_ir")
            content_parts.append(
                f'<p class="warning">Pareto frontier plot requires: {", ".join(missing)}</p>'
            )

        return DashboardSection(
            title="Efficiency",
            description=(
                "<p>Pareto frontier showing trade-off between signal quality (IC IR) and "
                "implementation cost (turnover). Signals on the frontier offer the best "
                "risk-adjusted returns for their turnover level.</p>"
            ),
            content="\n".join(content_parts),
        )

    def _create_comparison_tab(
        self,
        comparison: ComparisonResult,
        summary: MultiSignalSummary,
    ) -> DashboardSection:
        """Create Comparison tab for selected signals."""
        content_parts = []

        # Selection info
        content_parts.append(
            f'<div class="insights-panel">'
            f"<h3>Selection Details</h3>"
            f"<ul>"
            f"<li>Method: <strong>{comparison.selection_method}</strong></li>"
            f"<li>Signals selected: <strong>{len(comparison.signals)}</strong></li>"
            f"</ul>"
            f"<p>Selected signals: {', '.join(comparison.signals)}</p>"
            f"</div>"
        )

        # Comparison table
        content_parts.append(self._build_comparison_table(comparison, summary))

        return DashboardSection(
            title="Comparison",
            description=(
                "<p>Side-by-side comparison of selected signals. Signals were selected "
                f'using the "<strong>{comparison.selection_method}</strong>" method.</p>'
            ),
            content="\n".join(content_parts),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_metric_cards(self, summary: MultiSignalSummary) -> str:
        """Build metric cards HTML."""
        fdr_pct = summary.n_fdr_significant / summary.n_signals * 100
        fwer_pct = summary.n_fwer_significant / summary.n_signals * 100

        return f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Signals</div>
                <div class="metric-value">{summary.n_signals}</div>
                <div class="metric-sublabel">Periods: {summary.periods}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">FDR Significant (α={summary.fdr_alpha:.0%})</div>
                <div class="metric-value">{summary.n_fdr_significant}</div>
                <div class="metric-sublabel">{fdr_pct:.1f}% of signals</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">FWER Significant (α={summary.fwer_alpha:.0%})</div>
                <div class="metric-value">{summary.n_fwer_significant}</div>
                <div class="metric-sublabel">{fwer_pct:.1f}% of signals</div>
            </div>
        </div>
        """

    def _build_signal_table(self, summary: MultiSignalSummary) -> str:
        """Build searchable/sortable signal table HTML."""
        df = summary.get_dataframe()

        # Define columns to display (in order)
        display_cols = ["signal_name"]

        # Add metrics columns if available
        for col in ["ic_mean", "ic_std", "ic_ir", "ic_t_stat", "ic_p_value"]:
            if col in df.columns:
                display_cols.append(col)

        # Add turnover if available
        if "turnover_mean" in df.columns:
            display_cols.append("turnover_mean")

        # Add significance flags
        for col in ["fdr_significant", "fwer_significant"]:
            if col in df.columns:
                display_cols.append(col)

        # Build header
        header_cells = []
        for col in display_cols:
            nice_name = col.replace("_", " ").title()
            header_cells.append(f"<th>{nice_name}</th>")

        # Build rows
        rows = []
        for i in range(len(df)):
            cells = []
            for col in display_cols:
                value = df[col][i]

                # Format based on column type
                if col == "signal_name":
                    cell_html = f"<td><strong>{value}</strong></td>"
                elif "significant" in col:
                    badge_class = "badge-high" if value else "badge-low"
                    badge_text = "Yes" if value else "No"
                    cell_html = f'<td><span class="badge {badge_class}">{badge_text}</span></td>'
                elif col == "ic_p_value":
                    cell_html = f"<td>{value:.4f}</td>"
                elif isinstance(value, float):
                    cell_html = f"<td>{value:.4f}</td>"
                else:
                    cell_html = f"<td>{value}</td>"

                cells.append(cell_html)

            rows.append(f"<tr>{''.join(cells)}</tr>")

        return f"""
        <input type="text" id="signal-search" placeholder="Search signals..."
               onkeyup="filterSignalTable()">
        <table class="feature-table" id="signal-table">
            <thead>
                <tr>{"".join(header_cells)}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    def _build_comparison_table(
        self,
        comparison: ComparisonResult,
        summary: MultiSignalSummary,
    ) -> str:
        """Build comparison table for selected signals."""
        summary_df = summary.get_dataframe()
        rows = []

        # Define columns for comparison
        display_cols = ["signal_name"]
        for col in ["ic_mean", "ic_ir", "turnover_mean", "fdr_significant"]:
            if col in summary_df.columns:
                display_cols.append(col)

        # Header
        header_cells = [f"<th>{col.replace('_', ' ').title()}</th>" for col in display_cols]

        # Build rows for selected signals (maintaining order)
        for signal_name in comparison.signals:
            # Find row in summary
            mask = summary_df["signal_name"] == signal_name
            if not mask.any():
                continue

            signal_df = summary_df.filter(mask)

            cells = []
            for col in display_cols:
                value = signal_df[col][0]

                if col == "signal_name":
                    cell_html = f"<td><strong>{value}</strong></td>"
                elif "significant" in col:
                    badge_class = "badge-high" if value else "badge-low"
                    badge_text = "Yes" if value else "No"
                    cell_html = f'<td><span class="badge {badge_class}">{badge_text}</span></td>'
                elif isinstance(value, float):
                    cell_html = f"<td>{value:.4f}</td>"
                else:
                    cell_html = f"<td>{value}</td>"

                cells.append(cell_html)

            rows.append(f"<tr>{''.join(cells)}</tr>")

        return f"""
        <h3>Selected Signals Comparison</h3>
        <table class="feature-table">
            <thead>
                <tr>{"".join(header_cells)}</tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    # =========================================================================
    # HTML Composition
    # =========================================================================

    def _compose_html(self) -> str:
        """Compose final HTML document."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {self._get_base_styles()}
</head>
<body>
    {self._build_header()}
    {self._build_navigation()}
    {self._build_sections()}
    {self._get_base_scripts()}
</body>
</html>
        """

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
            return ""

        tabs_html = []
        for i, section in enumerate(self.sections):
            active_class = "active" if i == 0 else ""
            tabs_html.append(
                f'<button class="tab-button {active_class}" '
                f"onclick=\"switchTab(event, 'section-{i}')\">"
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
                <div class="section-description">{section.description}</div>
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
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
                cursor: pointer;
            }}

            .feature-table th:hover {{
                background-color: {"rgba(255,255,255,0.05)" if self.theme == "dark" else "rgba(0,0,0,0.05)"};
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

            .badge-low {{
                background-color: rgba(220, 53, 69, 0.2);
                color: #dc3545;
            }}

            #signal-search {{
                width: 100%;
                padding: 10px;
                font-size: 16px;
                border: 1px solid {border_color};
                border-radius: 4px;
                margin-bottom: 15px;
                background-color: {bg_color};
                color: {text_color};
            }}

            #signal-search:focus {{
                outline: none;
                border-color: #1f77b4;
                box-shadow: 0 0 5px rgba(31, 119, 180, 0.3);
            }}

            .error {{
                color: #dc3545;
                padding: 10px;
                background-color: rgba(220, 53, 69, 0.1);
                border-radius: 4px;
            }}

            .warning {{
                color: #ffc107;
                padding: 10px;
                background-color: rgba(255, 193, 7, 0.1);
                border-radius: 4px;
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

            // Signal table filtering
            function filterSignalTable() {
                const input = document.getElementById('signal-search');
                if (!input) return;

                const filter = input.value.toLowerCase();
                const table = document.getElementById('signal-table');
                if (!table) return;

                const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

                for (let row of rows) {
                    const signalName = row.cells[0].textContent.toLowerCase();
                    if (signalName.includes(filter)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                }
            }

            // Table sorting functionality
            document.addEventListener('DOMContentLoaded', function() {
                const tables = document.querySelectorAll('.feature-table');

                tables.forEach(table => {
                    const headers = table.querySelectorAll('thead th');
                    let sortDirection = {};

                    headers.forEach((header, colIndex) => {
                        header.addEventListener('click', function() {
                            const tbody = table.querySelector('tbody');
                            const rows = Array.from(tbody.querySelectorAll('tr'));

                            // Toggle sort direction
                            sortDirection[colIndex] = !sortDirection[colIndex];
                            const ascending = sortDirection[colIndex];

                            // Sort rows
                            rows.sort((a, b) => {
                                const aVal = a.cells[colIndex].textContent;
                                const bVal = b.cells[colIndex].textContent;

                                // Try numeric comparison first
                                const aNum = parseFloat(aVal);
                                const bNum = parseFloat(bVal);

                                if (!isNaN(aNum) && !isNaN(bNum)) {
                                    return ascending ? aNum - bNum : bNum - aNum;
                                }

                                // Fall back to string comparison
                                return ascending
                                    ? aVal.localeCompare(bVal)
                                    : bVal.localeCompare(aVal);
                            });

                            // Re-append sorted rows
                            rows.forEach(row => tbody.appendChild(row));

                            // Update header indicators
                            headers.forEach(h => h.textContent = h.textContent.replace(/ ▲| ▼/g, ''));
                            header.textContent += ascending ? ' ▲' : ' ▼';
                        });
                    });
                });
            });

            // Plotly responsive resizing
            window.addEventListener('resize', function() {
                const plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(plot => {
                    Plotly.Plots.resize(plot);
                });
            });
        </script>
        """


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "MultiSignalDashboard",
]
