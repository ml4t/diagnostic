"""Signal Analysis Dashboard - Multi-tab interactive HTML dashboard.

This module provides the SignalDashboard class for creating comprehensive,
self-contained HTML dashboards for signal/factor analysis results.

The dashboard follows the BaseDashboard pattern with 5 tabs:
1. Summary - Key metrics cards, signal quality assessment
2. IC Analysis - Information coefficient time series, distribution, heatmap
3. Quantile Analysis - Returns by quantile, cumulative performance, spread
4. Turnover - Signal stability, autocorrelation
5. Events (optional) - Event study results if provided
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ml4t.diagnostic.visualization.dashboards import (
    BaseDashboard,
    DashboardSection,
)
from ml4t.diagnostic.visualization.signal.event_plots import (
    plot_ar_distribution,
    plot_caar,
    plot_car_by_event,
    plot_event_heatmap,
)
from ml4t.diagnostic.visualization.signal.ic_plots import (
    plot_ic_heatmap,
    plot_ic_histogram,
    plot_ic_qq,
    plot_ic_ts,
)
from ml4t.diagnostic.visualization.signal.quantile_plots import (
    plot_cumulative_returns,
    plot_quantile_returns_bar,
    plot_quantile_returns_violin,
    plot_spread_timeseries,
)
from ml4t.diagnostic.visualization.signal.turnover_plots import (
    plot_autocorrelation,
    plot_top_bottom_turnover,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.event_results import EventStudyResult
    from ml4t.diagnostic.results.signal_results import SignalTearSheet


class SignalDashboard(BaseDashboard):
    """Interactive multi-tab dashboard for signal analysis results.

    Creates a self-contained HTML dashboard with comprehensive visualizations
    of signal/factor analysis results. The dashboard includes 5 tabs:

    1. **Summary**: Key metrics at a glance, signal quality badges, insights
    2. **IC Analysis**: IC time series, histogram, Q-Q plot, monthly heatmap
    3. **Quantile Analysis**: Returns by quantile, cumulative, spread analysis
    4. **Turnover**: Signal stability, autocorrelation by lag
    5. **Events** (optional): Event study results if provided

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import SignalAnalysis
    >>> from ml4t.diagnostic.visualization.signal import SignalDashboard
    >>>
    >>> # Run signal analysis
    >>> analyzer = SignalAnalysis(factor_data, price_data)
    >>> tear_sheet = analyzer.create_tear_sheet()
    >>>
    >>> # Create and save dashboard
    >>> dashboard = SignalDashboard(title="Momentum Factor Analysis")
    >>> dashboard.save("momentum_dashboard.html", tear_sheet)

    >>> # Dark theme with custom title
    >>> dashboard = SignalDashboard(
    ...     title="Value Factor Analysis",
    ...     theme="dark"
    ... )
    >>> html = dashboard.generate(tear_sheet)

    Notes
    -----
    - Dashboard is self-contained HTML with embedded Plotly.js (via CDN)
    - All visualizations are interactive (zoom, pan, hover)
    - Works offline once loaded (all data embedded)

    See Also
    --------
    SignalAnalysis : Main signal analysis class
    SignalTearSheet : Result container for signal analysis
    """

    def __init__(
        self,
        title: str = "Signal Analysis Dashboard",
        theme: Literal["light", "dark"] = "light",
        width: int | None = None,
        height: int | None = None,
    ):
        """Initialize Signal Analysis Dashboard.

        Parameters
        ----------
        title : str, default="Signal Analysis Dashboard"
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
        analysis_results: SignalTearSheet,
        include_events: bool = False,
        event_analysis: Any | None = None,
        **_kwargs: Any,
    ) -> str:
        """Generate complete dashboard HTML.

        Parameters
        ----------
        analysis_results : SignalTearSheet
            Results from SignalAnalysis.create_tear_sheet()
        include_events : bool, default=False
            Whether to include Events tab (requires event_analysis)
        event_analysis : EventStudyResult, optional
            Event study results to include in Events tab
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
        self._create_tabbed_layout(analysis_results, include_events, event_analysis)

        # Compose final HTML
        return self._compose_html()

    def save(
        self,
        output_path: str,
        analysis_results: SignalTearSheet,
        include_events: bool = False,
        event_analysis: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate and save dashboard to file.

        Parameters
        ----------
        output_path : str
            Path for output HTML file
        analysis_results : SignalTearSheet
            Results from SignalAnalysis.create_tear_sheet()
        include_events : bool, default=False
            Whether to include Events tab
        event_analysis : EventStudyResult, optional
            Event study results for Events tab
        **kwargs
            Additional parameters passed to generate()

        Returns
        -------
        str
            Path to saved file
        """
        html = self.generate(
            analysis_results,
            include_events=include_events,
            event_analysis=event_analysis,
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
        tear_sheet: SignalTearSheet,
        include_events: bool = False,
        event_analysis: Any | None = None,
    ) -> None:
        """Create tabbed dashboard layout."""
        # Define tabs
        tabs = [
            ("summary", "Summary"),
            ("ic", "IC Analysis"),
            ("quantile", "Quantile Analysis"),
            ("turnover", "Turnover"),
        ]

        # Add events tab if requested
        if include_events and event_analysis is not None:
            tabs.append(("events", "Events"))

        # Build tab content
        tab_contents = {
            "summary": self._create_summary_tab(tear_sheet),
            "ic": self._create_ic_tab(tear_sheet),
            "quantile": self._create_quantile_tab(tear_sheet),
            "turnover": self._create_turnover_tab(tear_sheet),
        }

        if include_events and event_analysis is not None:
            tab_contents["events"] = self._create_events_tab(event_analysis)

        # Build tab navigation buttons
        tab_buttons = "".join(
            [
                f'<button class="tab-button{" active" if i == 0 else ""}" '
                f"onclick=\"switchTab(event, '{tab_id}')\">{tab_name}</button>"
                for i, (tab_id, tab_name) in enumerate(tabs)
            ]
        )

        # Build tab content divs
        tab_divs = "".join(
            [
                f'<div id="{tab_id}" class="tab-content{" active" if i == 0 else ""}">'
                f"{tab_contents[tab_id]}</div>"
                for i, (tab_id, _) in enumerate(tabs)
            ]
        )

        # Compose tabbed layout
        html_content = f"""
        <div class="tab-navigation">
            {tab_buttons}
        </div>
        {tab_divs}
        """

        # Create single section with all tabbed content
        section = DashboardSection(
            title="Signal Analysis",
            description="",
            content=html_content,
        )
        self.sections.append(section)

    # =========================================================================
    # Summary Tab
    # =========================================================================

    def _create_summary_tab(self, tear_sheet: SignalTearSheet) -> str:
        """Create Summary tab with key metrics and insights."""
        html_parts = ["<h2>Summary</h2>"]

        # Metadata section
        html_parts.append(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Signal Name</div>
                <div class="metric-value">{tear_sheet.signal_name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Assets</div>
                <div class="metric-value">{tear_sheet.n_assets:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dates</div>
                <div class="metric-value">{tear_sheet.n_dates:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Date Range</div>
                <div class="metric-value" style="font-size: 1em;">
                    {tear_sheet.date_range[0][:10]}<br>to {tear_sheet.date_range[1][:10]}
                </div>
            </div>
        </div>
        """)

        # IC Summary metrics
        if tear_sheet.ic_analysis is not None:
            ic = tear_sheet.ic_analysis
            periods = list(ic.ic_mean.keys())
            first_period = periods[0] if periods else "1D"

            ic_mean = ic.ic_mean.get(first_period, 0)
            ic_ir = ic.ic_ir.get(first_period, 0)
            ic_positive = ic.ic_positive_pct.get(first_period, 0)
            ic_t = ic.ic_t_stat.get(first_period, 0)

            # Quality badge based on IC
            if abs(ic_mean) > 0.05:
                quality_badge = '<span class="badge badge-high">Strong</span>'
            elif abs(ic_mean) > 0.02:
                quality_badge = '<span class="badge badge-medium">Moderate</span>'
            else:
                quality_badge = '<span class="badge badge-low">Weak</span>'

            html_parts.append(f"""
            <h3>IC Metrics ({first_period})</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Mean IC</div>
                    <div class="metric-value">{ic_mean:.4f}</div>
                    <div class="metric-sublabel">{quality_badge}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">IC IR</div>
                    <div class="metric-value">{ic_ir:.3f}</div>
                    <div class="metric-sublabel">
                        {"Good" if ic_ir > 0.5 else "Moderate" if ic_ir > 0.2 else "Low"}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">IC Positive %</div>
                    <div class="metric-value">{ic_positive:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">t-statistic</div>
                    <div class="metric-value">{ic_t:.2f}</div>
                    <div class="metric-sublabel">
                        {"Significant" if abs(ic_t) > 2 else "Not significant"}
                    </div>
                </div>
            </div>
            """)

            # RAS-adjusted IC if available
            if ic.ras_adjusted_ic is not None and ic.ras_significant is not None:
                ras_ic = ic.ras_adjusted_ic.get(first_period, 0)
                ras_sig = ic.ras_significant.get(first_period, False)
                sig_icon = "✓" if ras_sig else "✗"
                sig_color = "#28a745" if ras_sig else "#dc3545"

                html_parts.append(f"""
                <div class="insights-panel">
                    <h3>RAS-Adjusted IC (Multiple Testing Correction)</h3>
                    <p><strong>Adjusted IC:</strong> {ras_ic:.4f}</p>
                    <p><strong>Significant:</strong>
                        <span style="color: {sig_color}; font-weight: bold;">{sig_icon}</span>
                        {"Signal passes multiple testing correction" if ras_sig else "Signal may be spurious"}
                    </p>
                </div>
                """)

        # Quantile spread summary
        if tear_sheet.quantile_analysis is not None:
            qa = tear_sheet.quantile_analysis
            periods = qa.periods
            first_period = periods[0] if periods else "1D"

            spread = qa.spread_mean.get(first_period, 0)
            spread_t = qa.spread_t_stat.get(first_period, 0)
            monotonic = qa.is_monotonic.get(first_period, False)

            html_parts.append(f"""
            <h3>Quantile Analysis ({first_period})</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Spread (Top-Bottom)</div>
                    <div class="metric-value">{spread:.4%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Spread t-stat</div>
                    <div class="metric-value">{spread_t:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Monotonic</div>
                    <div class="metric-value">{"Yes" if monotonic else "No"}</div>
                </div>
            </div>
            """)

        # Turnover summary
        if tear_sheet.turnover_analysis is not None:
            ta = tear_sheet.turnover_analysis
            periods = list(ta.mean_turnover.keys())
            first_period = periods[0] if periods else "1D"

            turnover = ta.mean_turnover.get(first_period, 0)
            half_life = ta.half_life.get(first_period)

            html_parts.append(f"""
            <h3>Turnover ({first_period})</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Mean Turnover</div>
                    <div class="metric-value">{turnover:.1%}</div>
                    <div class="metric-sublabel">
                        {"High" if turnover > 0.3 else "Moderate" if turnover > 0.15 else "Low"}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Signal Half-Life</div>
                    <div class="metric-value">
                        {f"{half_life:.1f}" if half_life else "N/A"}
                    </div>
                    <div class="metric-sublabel">periods</div>
                </div>
            </div>
            """)

        # IR_tc summary
        if tear_sheet.ir_tc_analysis is not None:
            ir_tc = tear_sheet.ir_tc_analysis
            periods = list(ir_tc.ir_gross.keys())
            first_period = periods[0] if periods else "1D"

            ir_gross = ir_tc.ir_gross.get(first_period, 0)
            ir_net = ir_tc.ir_tc.get(first_period, 0)
            cost_drag = ir_tc.cost_drag.get(first_period, 0)

            html_parts.append(f"""
            <h3>Transaction Cost Impact ({first_period})</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Gross IR</div>
                    <div class="metric-value">{ir_gross:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Net IR (after costs)</div>
                    <div class="metric-value">{ir_net:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cost Drag</div>
                    <div class="metric-value">{cost_drag:.1%}</div>
                </div>
            </div>
            """)

        return "\n".join(html_parts)

    # =========================================================================
    # IC Analysis Tab
    # =========================================================================

    def _create_ic_tab(self, tear_sheet: SignalTearSheet) -> str:
        """Create IC Analysis tab with all IC visualizations."""
        html_parts = ["<h2>Information Coefficient Analysis</h2>"]

        if tear_sheet.ic_analysis is None:
            html_parts.append("<p>IC analysis not available.</p>")
            return "\n".join(html_parts)

        ic = tear_sheet.ic_analysis
        theme_name = "dark" if self.theme == "dark" else "default"

        # Period selector
        periods = list(ic.ic_mean.keys())
        html_parts.append(self._create_period_selector("ic", periods))

        # IC Time Series
        try:
            fig_ts = plot_ic_ts(ic, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_ts.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>IC time series plot unavailable.</p>")

        # Two-column layout for histogram and Q-Q
        html_parts.append('<div style="display: flex; gap: 20px; flex-wrap: wrap;">')

        # IC Histogram
        try:
            fig_hist = plot_ic_histogram(ic, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
            html_parts.append(fig_hist.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>IC histogram unavailable.</p>")

        # IC Q-Q Plot
        try:
            fig_qq = plot_ic_qq(ic, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
            html_parts.append(fig_qq.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>IC Q-Q plot unavailable.</p>")

        html_parts.append("</div>")

        # IC Heatmap (monthly)
        try:
            fig_heatmap = plot_ic_heatmap(ic, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_heatmap.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>IC heatmap unavailable.</p>")

        return "\n".join(html_parts)

    # =========================================================================
    # Quantile Analysis Tab
    # =========================================================================

    def _create_quantile_tab(self, tear_sheet: SignalTearSheet) -> str:
        """Create Quantile Analysis tab."""
        html_parts = ["<h2>Quantile Returns Analysis</h2>"]

        if tear_sheet.quantile_analysis is None:
            html_parts.append("<p>Quantile analysis not available.</p>")
            return "\n".join(html_parts)

        qa = tear_sheet.quantile_analysis
        theme_name = "dark" if self.theme == "dark" else "default"

        # Period selector
        periods = qa.periods
        html_parts.append(self._create_period_selector("quantile", periods))

        # Quantile Returns Bar
        try:
            fig_bar = plot_quantile_returns_bar(qa, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_bar.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>Quantile returns bar chart unavailable.</p>")

        # Two-column layout
        html_parts.append('<div style="display: flex; gap: 20px; flex-wrap: wrap;">')

        # Quantile Returns Violin
        try:
            fig_violin = plot_quantile_returns_violin(qa, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
            html_parts.append(fig_violin.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>Quantile violin plot unavailable.</p>")

        # Spread Time Series
        try:
            fig_spread = plot_spread_timeseries(qa, period=periods[0], theme=theme_name)
            html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
            html_parts.append(fig_spread.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>Spread time series unavailable.</p>")

        html_parts.append("</div>")

        # Cumulative Returns
        if qa.cumulative_returns is not None:
            try:
                fig_cum = plot_cumulative_returns(qa, period=periods[0], theme=theme_name)
                html_parts.append('<div class="plot-container">')
                html_parts.append(fig_cum.to_html(include_plotlyjs=False, full_html=False))
                html_parts.append("</div>")
            except Exception:
                html_parts.append("<p>Cumulative returns plot unavailable.</p>")

        return "\n".join(html_parts)

    # =========================================================================
    # Turnover Tab
    # =========================================================================

    def _create_turnover_tab(self, tear_sheet: SignalTearSheet) -> str:
        """Create Turnover tab."""
        html_parts = ["<h2>Signal Turnover Analysis</h2>"]

        if tear_sheet.turnover_analysis is None:
            html_parts.append("<p>Turnover analysis not available.</p>")
            return "\n".join(html_parts)

        ta = tear_sheet.turnover_analysis
        theme_name = "dark" if self.theme == "dark" else "default"

        # Two-column layout
        html_parts.append('<div style="display: flex; gap: 20px; flex-wrap: wrap;">')

        # Top/Bottom Turnover
        try:
            fig_turnover = plot_top_bottom_turnover(ta, theme=theme_name)
            html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
            html_parts.append(fig_turnover.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception:
            html_parts.append("<p>Turnover chart unavailable.</p>")

        # Autocorrelation
        periods = list(ta.autocorrelation.keys())
        if periods:
            try:
                fig_ac = plot_autocorrelation(ta, period=periods[0], theme=theme_name)
                html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
                html_parts.append(fig_ac.to_html(include_plotlyjs=False, full_html=False))
                html_parts.append("</div>")
            except Exception:
                html_parts.append("<p>Autocorrelation plot unavailable.</p>")

        html_parts.append("</div>")

        # Summary table
        html_parts.append(self._create_turnover_summary_table(ta))

        return "\n".join(html_parts)

    def _create_turnover_summary_table(self, ta: Any) -> str:
        """Create turnover summary table."""
        periods = list(ta.mean_turnover.keys())

        rows = []
        for period in periods:
            mean_to = ta.mean_turnover.get(period, 0)
            top_to = ta.top_quantile_turnover.get(period, 0)
            bottom_to = ta.bottom_quantile_turnover.get(period, 0)
            mean_ac = ta.mean_autocorrelation.get(period, 0)
            half_life = ta.half_life.get(period)

            rows.append(f"""
            <tr>
                <td>{period}</td>
                <td>{mean_to:.1%}</td>
                <td>{top_to:.1%}</td>
                <td>{bottom_to:.1%}</td>
                <td>{mean_ac:.4f}</td>
                <td>{f"{half_life:.1f}" if half_life else "N/A"}</td>
            </tr>
            """)

        return f"""
        <h3>Turnover Summary</h3>
        <table class="feature-table">
            <thead>
                <tr>
                    <th>Period</th>
                    <th>Mean Turnover</th>
                    <th>Top Quantile</th>
                    <th>Bottom Quantile</th>
                    <th>Mean AC</th>
                    <th>Half-Life</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    # =========================================================================
    # Events Tab - Event Study Analysis
    # =========================================================================

    def _create_events_tab(self, event_analysis: EventStudyResult) -> str:
        """Create Events tab for event study results.

        Displays comprehensive event study analysis including:
        - Summary metrics (CAAR, significance, n events)
        - CAAR time series with confidence bands
        - Event drift heatmap
        - AR distribution on event day
        - CAR by event bar chart

        Parameters
        ----------
        event_analysis : EventStudyResult
            Complete event study results.

        Returns
        -------
        str
            HTML content for the Events tab.
        """
        html_parts = ["<h2>Event Study Analysis</h2>"]
        theme_name = "dark" if self.theme == "dark" else "default"

        # Summary metrics section
        sig_status = "Significant" if event_analysis.is_significant else "Not Significant"
        sig_color = "#28a745" if event_analysis.is_significant else "#dc3545"

        html_parts.append(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Events Analyzed</div>
                <div class="metric-value">{event_analysis.n_events}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Event Window</div>
                <div class="metric-value">
                    [{event_analysis.event_window[0]}, {event_analysis.event_window[1]}]
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">{event_analysis.model_name.replace("_", " ").title()}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final CAAR</div>
                <div class="metric-value">{event_analysis.final_caar:+.4f}</div>
                <div class="metric-sublabel">{event_analysis.final_caar * 100:+.2f}%</div>
            </div>
        </div>
        """)

        # Event day AAR and significance
        html_parts.append(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Event Day AAR (t=0)</div>
                <div class="metric-value">{event_analysis.event_day_aar:+.4f}</div>
                <div class="metric-sublabel">{event_analysis.event_day_aar * 100:+.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test</div>
                <div class="metric-value">{event_analysis.test_name.replace("_", " ").title()}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Statistic</div>
                <div class="metric-value">{event_analysis.test_statistic:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P-value</div>
                <div class="metric-value">{event_analysis.p_value:.4f}</div>
                <div class="metric-sublabel" style="color: {sig_color};">{sig_status}</div>
            </div>
        </div>
        """)

        # CAAR plot with confidence bands
        try:
            fig_caar = plot_caar(
                event_analysis,
                show_confidence=True,
                show_aar_bars=True,
                theme=theme_name,
            )
            html_parts.append('<div class="plot-container">')
            html_parts.append(fig_caar.to_html(include_plotlyjs=False, full_html=False))
            html_parts.append("</div>")
        except Exception as e:
            html_parts.append(f"<p>CAAR plot unavailable: {e}</p>")

        # Two-column layout for heatmap and distribution
        html_parts.append('<div style="display: flex; gap: 20px; flex-wrap: wrap;">')

        # Event heatmap (if individual results available)
        if (
            event_analysis.individual_results is not None
            and len(event_analysis.individual_results) > 0
        ):
            try:
                # Limit to 30 events for readability
                ar_results = event_analysis.individual_results[:30]
                fig_heatmap = plot_event_heatmap(ar_results, theme=theme_name)
                html_parts.append('<div class="plot-container" style="flex: 1; min-width: 500px;">')
                html_parts.append(fig_heatmap.to_html(include_plotlyjs=False, full_html=False))
                html_parts.append("</div>")
            except Exception as e:
                html_parts.append(f"<p>Event heatmap unavailable: {e}</p>")

            # AR distribution on event day
            try:
                fig_dist = plot_ar_distribution(
                    event_analysis,
                    day=0,
                    show_kde=True,
                    theme=theme_name,
                )
                html_parts.append('<div class="plot-container" style="flex: 1; min-width: 400px;">')
                html_parts.append(fig_dist.to_html(include_plotlyjs=False, full_html=False))
                html_parts.append("</div>")
            except Exception as e:
                html_parts.append(f"<p>AR distribution plot unavailable: {e}</p>")

        html_parts.append("</div>")  # Close flex container

        # CAR by event bar chart (top 20 by magnitude)
        if (
            event_analysis.individual_results is not None
            and len(event_analysis.individual_results) > 0
        ):
            try:
                fig_car = plot_car_by_event(
                    event_analysis.individual_results,
                    sort_by="car",
                    top_n=min(20, len(event_analysis.individual_results)),
                    theme=theme_name,
                )
                html_parts.append('<div class="plot-container">')
                html_parts.append(fig_car.to_html(include_plotlyjs=False, full_html=False))
                html_parts.append("</div>")
            except Exception as e:
                html_parts.append(f"<p>CAR by event chart unavailable: {e}</p>")

        # Events table
        if (
            event_analysis.individual_results is not None
            and len(event_analysis.individual_results) > 0
        ):
            html_parts.append(self._create_events_table(event_analysis))

        return "\n".join(html_parts)

    def _create_events_table(self, event_analysis: EventStudyResult) -> str:
        """Create table summarizing individual event results.

        Parameters
        ----------
        event_analysis : EventStudyResult
            Event study results with individual event data.

        Returns
        -------
        str
            HTML table of event results.
        """
        if event_analysis.individual_results is None:
            return ""

        # Sort by CAR magnitude (descending)
        sorted_results = sorted(
            event_analysis.individual_results,
            key=lambda x: abs(x.car),
            reverse=True,
        )

        rows = []
        for r in sorted_results[:20]:  # Limit to top 20
            car_color = "#28a745" if r.car >= 0 else "#dc3545"
            ar_day0 = r.ar_by_day.get(0, 0.0)
            beta_str = f"{r.estimation_beta:.2f}" if r.estimation_beta is not None else "N/A"
            rows.append(f"""
            <tr>
                <td>{r.event_id}</td>
                <td>{r.asset}</td>
                <td>{r.event_date[:10] if len(r.event_date) >= 10 else r.event_date}</td>
                <td style="color: {car_color};">{r.car:+.4f}</td>
                <td>{ar_day0:+.4f}</td>
                <td>{beta_str}</td>
            </tr>
            """)

        n_shown = min(20, len(event_analysis.individual_results))
        n_total = len(event_analysis.individual_results)
        table_title = f"Individual Event Results (Top {n_shown} of {n_total} by |CAR|)"

        return f"""
        <h3>{table_title}</h3>
        <table class="feature-table">
            <thead>
                <tr>
                    <th>Event ID</th>
                    <th>Asset</th>
                    <th>Event Date</th>
                    <th>CAR</th>
                    <th>AR (t=0)</th>
                    <th>Beta</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_period_selector(self, tab_id: str, periods: list[str]) -> str:
        """Create period selector dropdown (for future JS interactivity)."""
        if len(periods) <= 1:
            return ""

        options = "".join([f'<option value="{p}">{p}</option>' for p in periods])
        return f"""
        <div class="period-selector" style="margin-bottom: 15px;">
            <label for="{tab_id}-period">Period: </label>
            <select id="{tab_id}-period" style="padding: 5px;">
                {options}
            </select>
            <span style="font-size: 0.85em; color: #666; margin-left: 10px;">
                (Changing period will update in future version)
            </span>
        </div>
        """

    def _compose_html(self) -> str:
        """Compose complete HTML document."""
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
    {self._build_sections()}
    {self._get_base_scripts()}
</body>
</html>
"""
