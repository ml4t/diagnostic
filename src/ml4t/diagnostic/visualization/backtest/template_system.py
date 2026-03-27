"""Template system for backtest tearsheets.

Provides persona-based templates that customize the tearsheet content
for different user types:
- quant_trader: Trade-level analysis, MFE/MAE, exit optimization
- hedge_fund: Risk-adjusted returns, cost attribution, drawdowns
- risk_manager: Statistical validity, DSR, confidence intervals
- full: Everything included
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


@dataclass
class TearsheetSection:
    """Definition of a tearsheet section."""

    name: str
    title: str
    band: str = "appendix"
    enabled: bool = True
    priority: int = 0  # Lower = higher priority (shown first)
    description: str = ""


@dataclass
class TearsheetBand:
    """Definition of a narrative band in the tearsheet layout."""

    name: str
    title: str
    description: str
    tone: Literal["hero", "primary", "secondary", "appendix"] = "primary"


@dataclass
class TearsheetTemplate:
    """Template configuration for a tearsheet."""

    name: str
    description: str
    bands: list[TearsheetBand] = field(default_factory=list)
    sections: list[TearsheetSection] = field(default_factory=list)

    @classmethod
    def quant_trader(cls) -> TearsheetTemplate:
        """Template focused on trade-level analysis for quantitative traders.

        Emphasizes:
        - Trade execution efficiency (MFE/MAE)
        - Exit reason analysis
        - Trade-by-trade waterfall
        - Duration and timing analysis
        """
        return cls(
            name="quant_trader",
            description="Trade-level deep dive for strategy optimization",
            bands=[
                TearsheetBand(
                    "hero",
                    "Summary",
                    "",
                    tone="hero",
                ),
                TearsheetBand(
                    "implementation",
                    "Trading",
                    "",
                ),
                TearsheetBand(
                    "burden",
                    "Detail",
                    "",
                    tone="secondary",
                ),
                TearsheetBand(
                    "appendix",
                    "Appendix",
                    "",
                    tone="appendix",
                ),
            ],
            sections=[
                TearsheetSection("executive_summary", "Summary", band="hero", priority=0),
                TearsheetSection("overview_snapshot", "Performance Snapshot", band="hero", priority=1),
                TearsheetSection(
                    "activity_strip", "Activity", band="implementation", priority=3
                ),
                TearsheetSection(
                    "occupancy_overview",
                    "Exposure",
                    band="implementation",
                    priority=4,
                ),
                TearsheetSection("attribution_overview", "Attribution", band="burden", priority=5),
                TearsheetSection("drawdown_anatomy", "Drawdown Anatomy", band="burden", priority=6),
                TearsheetSection("mfe_mae", "Exit Efficiency (MFE/MAE)", band="burden", priority=7),
                TearsheetSection(
                    "exit_reasons", "Exit Reason Breakdown", band="burden", priority=8
                ),
                TearsheetSection(
                    "trade_waterfall", "Trade-by-Trade PnL", band="burden", priority=9
                ),
                TearsheetSection("duration", "Trade Duration Analysis", band="burden", priority=10),
                TearsheetSection(
                    "ic_time_series",
                    "IC Time Series",
                    band="appendix",
                    priority=13,
                    enabled=False,
                ),
                TearsheetSection(
                    "quintile_returns",
                    "Quintile Returns",
                    band="appendix",
                    priority=14,
                    enabled=False,
                ),
                TearsheetSection(
                    "prediction_trade_alignment",
                    "Prediction vs Trade Outcomes",
                    band="appendix",
                    priority=15,
                    enabled=False,
                ),
                TearsheetSection(
                    "shap_errors",
                    "SHAP Error Patterns",
                    band="appendix",
                    priority=16,
                    enabled=False,
                ),
                # Disabled by default
                TearsheetSection(
                    "equity_curve", "Equity Curve", band="appendix", priority=17, enabled=False
                ),
                TearsheetSection(
                    "drawdowns", "Drawdowns", band="appendix", priority=18, enabled=False
                ),
            ],
        )

    @classmethod
    def hedge_fund(cls) -> TearsheetTemplate:
        """Template focused on risk-adjusted returns for hedge fund managers.

        Emphasizes:
        - Portfolio performance metrics
        - Drawdown analysis
        - Cost attribution
        - Risk metrics
        """
        return cls(
            name="hedge_fund",
            description="Risk-adjusted performance for portfolio managers",
            bands=[
                TearsheetBand(
                    "hero",
                    "Summary",
                    "",
                    tone="hero",
                ),
                TearsheetBand(
                    "implementation",
                    "Performance",
                    "",
                ),
                TearsheetBand(
                    "burden",
                    "Trading",
                    "",
                    tone="secondary",
                ),
                TearsheetBand(
                    "appendix",
                    "Appendix",
                    "",
                    tone="appendix",
                ),
            ],
            sections=[
                TearsheetSection("executive_summary", "Summary", band="hero", priority=0),
                TearsheetSection("overview_snapshot", "Performance Snapshot", band="hero", priority=1),
                TearsheetSection("equity_curve", "Equity Curve", band="implementation", priority=3),
                TearsheetSection(
                    "rolling_metrics", "Rolling Performance", band="implementation", priority=4
                ),
                TearsheetSection(
                    "drawdowns", "Drawdown Analysis", band="implementation", priority=5
                ),
                TearsheetSection(
                    "drawdown_anatomy", "Drawdown Anatomy", band="implementation", priority=6
                ),
                TearsheetSection(
                    "monthly_heatmap_overview", "Monthly Returns", band="implementation", priority=7
                ),
                TearsheetSection(
                    "annual_returns", "Annual Returns", band="implementation", priority=8
                ),
                TearsheetSection("activity_strip", "Activity", band="burden", priority=10),
                TearsheetSection("occupancy_overview", "Exposure", band="burden", priority=11),
                TearsheetSection("attribution_overview", "Attribution", band="burden", priority=12),
                TearsheetSection("cost_waterfall", "Cost Attribution", band="burden", priority=13),
                TearsheetSection(
                    "cost_sensitivity", "Cost Sensitivity", band="burden", priority=14
                ),
                # Disabled by default
                TearsheetSection(
                    "mfe_mae", "Exit Efficiency", band="appendix", priority=22, enabled=False
                ),
                TearsheetSection(
                    "trade_waterfall", "Trade Details", band="appendix", priority=23, enabled=False
                ),
                # ML (disabled by default, enabled when predictions passed)
                TearsheetSection(
                    "ml_summary_strip",
                    "ML Summary",
                    band="appendix",
                    priority=19,
                    enabled=False,
                ),
                TearsheetSection(
                    "ic_time_series",
                    "IC Time Series",
                    band="appendix",
                    priority=20,
                    enabled=False,
                ),
                TearsheetSection(
                    "quintile_returns",
                    "Quintile Returns",
                    band="appendix",
                    priority=21,
                    enabled=False,
                ),
                # Factor (disabled by default, enabled when factor_data passed)
                TearsheetSection(
                    "factor_exposure",
                    "Factor Exposures",
                    band="appendix",
                    priority=24,
                    enabled=False,
                ),
                TearsheetSection(
                    "factor_attribution",
                    "Factor Attribution",
                    band="appendix",
                    priority=25,
                    enabled=False,
                ),
            ],
        )

    @classmethod
    def risk_manager(cls) -> TearsheetTemplate:
        """Template focused on statistical validity for risk managers.

        Emphasizes:
        - Deflated Sharpe Ratio
        - Confidence intervals
        - Minimum track record length
        - Tail risk metrics
        """
        return cls(
            name="risk_manager",
            description="Statistical rigor for risk oversight",
            bands=[
                TearsheetBand(
                    "hero",
                    "Summary",
                    "",
                    tone="hero",
                ),
                TearsheetBand(
                    "implementation",
                    "Performance",
                    "",
                ),
                TearsheetBand(
                    "burden",
                    "Risk",
                    "",
                    tone="secondary",
                ),
                TearsheetBand(
                    "appendix",
                    "Appendix",
                    "",
                    tone="appendix",
                ),
            ],
            sections=[
                TearsheetSection("executive_summary", "Summary", band="hero", priority=0),
                TearsheetSection("overview_snapshot", "Performance Snapshot", band="hero", priority=1),
                TearsheetSection("equity_curve", "Equity Curve", band="implementation", priority=3),
                TearsheetSection(
                    "rolling_metrics", "Rolling Performance", band="implementation", priority=4
                ),
                TearsheetSection("drawdowns", "Drawdowns", band="burden", priority=5),
                TearsheetSection("drawdown_anatomy", "Drawdown Anatomy", band="burden", priority=6),
                TearsheetSection(
                    "validity_card", "Statistical Validity", band="burden", priority=7
                ),
                TearsheetSection(
                    "confidence_intervals",
                    "Confidence Intervals",
                    band="burden",
                    priority=9,
                ),
                TearsheetSection("min_trl", "Minimum Track Record", band="burden", priority=10),
                TearsheetSection("sharpe_bootstrap", "Sharpe Bootstrap", band="burden", priority=11),
                TearsheetSection(
                    "distribution", "Returns Distribution", band="appendix", priority=12
                ),
                # Disabled by default
                TearsheetSection(
                    "mfe_mae", "Exit Efficiency", band="appendix", priority=13, enabled=False
                ),
                TearsheetSection(
                    "cost_waterfall",
                    "Cost Attribution",
                    band="appendix",
                    priority=14,
                    enabled=False,
                ),
                # ML (disabled by default, enabled when predictions passed)
                TearsheetSection(
                    "ml_summary_strip",
                    "ML Summary",
                    band="appendix",
                    priority=16,
                    enabled=False,
                ),
                TearsheetSection(
                    "ic_time_series",
                    "IC Time Series",
                    band="appendix",
                    priority=17,
                    enabled=False,
                ),
                TearsheetSection(
                    "quintile_returns",
                    "Quintile Returns",
                    band="appendix",
                    priority=18,
                    enabled=False,
                ),
                # Factor (disabled by default, enabled when factor_data passed)
                TearsheetSection(
                    "factor_risk", "Factor Risk", band="appendix", priority=20, enabled=False
                ),
            ],
        )

    @classmethod
    def full(cls) -> TearsheetTemplate:
        """Complete template per layout-spec.md (2026-03-26)."""
        return cls(
            name="full",
            description="Comprehensive analysis — structured argument, not analytics dump",
            bands=[
                TearsheetBand("hero", "Overview", "", tone="hero"),
                TearsheetBand("performance", "Performance", ""),
                TearsheetBand("trading", "Trading", "", tone="secondary"),
                TearsheetBand("validation", "Validation", "", tone="appendix"),
            ],
            sections=[
                # === Overview ===
                TearsheetSection("executive_summary", "Summary", band="hero", priority=0),
                TearsheetSection("overview_snapshot", "Performance Snapshot", band="hero", priority=1),
                TearsheetSection("metrics_sidebar", "Key Metrics", band="hero", priority=2),
                TearsheetSection("credibility_box", "Credibility", band="hero", priority=3),
                TearsheetSection("cost_summary_line", "Cost Summary", band="hero", priority=4),
                TearsheetSection("top_contributors", "Contributors", band="hero", priority=5),
                TearsheetSection("monthly_heatmap_overview", "Monthly Returns", band="hero", priority=6),
                # === Performance ===
                TearsheetSection("equity_curve", "Cumulative Return", band="performance", priority=10),
                TearsheetSection("top_drawdowns_table", "Top Drawdowns", band="performance", priority=11),
                TearsheetSection("drawdowns", "Underwater", band="performance", priority=12),
                TearsheetSection("rolling_metrics", "Rolling Metrics", band="performance", priority=13),
                TearsheetSection("annual_returns", "Annual Returns", band="performance", priority=14),
                TearsheetSection("distribution", "Return Distribution", band="performance", priority=15),
                TearsheetSection("stock_attribution", "Stock Attribution", band="performance", priority=16),
                # === Trading ===
                TearsheetSection("activity_strip", "Activity", band="trading", priority=20),
                TearsheetSection("occupancy_overview", "Exposure", band="trading", priority=21),
                TearsheetSection("cost_waterfall", "Cost Bridge", band="trading", priority=22),
                TearsheetSection("cost_sensitivity", "Cost Sensitivity", band="trading", priority=23),
                TearsheetSection("attribution_overview", "Attribution", band="trading", priority=24),
                TearsheetSection("mfe_mae", "MFE/MAE", band="trading", priority=25),
                TearsheetSection("duration", "Trade Duration", band="trading", priority=26),
                TearsheetSection("worst_trades_table", "Worst Trades", band="trading", priority=27),
                TearsheetSection("trade_waterfall", "Trade PnL Waterfall", band="trading", priority=28, enabled=False),
                TearsheetSection("exit_reasons", "Exit Reasons", band="trading", priority=29),
                # === Validation ===
                TearsheetSection("validity_card", "Statistical Validity", band="validation", priority=30),
                TearsheetSection("confidence_intervals", "Confidence Intervals", band="validation", priority=31),
                TearsheetSection("min_trl", "Minimum Track Record", band="validation", priority=32),
                TearsheetSection("sharpe_bootstrap", "Sharpe Bootstrap Distribution", band="validation", priority=33),
                TearsheetSection("drawdown_anatomy", "Drawdown Anatomy", band="validation", priority=34),
                # === ML (conditional) ===
                TearsheetSection("ml_summary_strip", "ML Summary", band="validation", priority=60, enabled=False),
                TearsheetSection("signal_diagnostics", "Signal Diagnostics", band="validation", priority=61, enabled=False),
                TearsheetSection("ic_time_series", "IC Time Series", band="validation", priority=61, enabled=False),
                TearsheetSection("quintile_returns", "Quintile Returns", band="validation", priority=61, enabled=False),
                TearsheetSection("prediction_trade_alignment", "Prediction vs Outcomes", band="validation", priority=62, enabled=False),
                TearsheetSection("prediction_calibration", "Prediction Calibration", band="validation", priority=63, enabled=False),
                TearsheetSection("shap_errors", "SHAP Error Patterns", band="validation", priority=64, enabled=False),
                # === Factors (conditional) ===
                TearsheetSection("factor_exposure", "Factor Exposures", band="validation", priority=70, enabled=False),
                TearsheetSection("factor_legend", "Factor Legend", band="validation", priority=70, enabled=False),
                TearsheetSection("factor_attribution", "Factor Attribution", band="validation", priority=71, enabled=False),
                TearsheetSection("factor_risk", "Factor Risk Decomposition", band="validation", priority=72, enabled=False),
            ],
        )

    def get_enabled_sections(self) -> list[TearsheetSection]:
        """Return only enabled sections, sorted by priority."""
        return sorted(
            [s for s in self.sections if s.enabled],
            key=lambda s: s.priority,
        )

    def get_band(self, name: str) -> TearsheetBand | None:
        """Return narrative band metadata by name."""
        for band in self.bands:
            if band.name == name:
                return band
        return None

    def enable_section(self, name: str) -> None:
        """Enable a section by name."""
        for section in self.sections:
            if section.name == name:
                section.enabled = True
                return
        raise ValueError(f"Section '{name}' not found in template")

    def disable_section(self, name: str) -> None:
        """Disable a section by name."""
        for section in self.sections:
            if section.name == name:
                section.enabled = False
                return
        raise ValueError(f"Section '{name}' not found in template")


def get_template(
    name: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
) -> TearsheetTemplate:
    """Get a tearsheet template by name.

    Parameters
    ----------
    name : {"quant_trader", "hedge_fund", "risk_manager", "full"}
        Template name

    Returns
    -------
    TearsheetTemplate
        The requested template

    Examples
    --------
    >>> template = get_template("quant_trader")
    >>> for section in template.get_enabled_sections():
    ...     print(section.title)
    """
    templates = {
        "quant_trader": TearsheetTemplate.quant_trader,
        "hedge_fund": TearsheetTemplate.hedge_fund,
        "risk_manager": TearsheetTemplate.risk_manager,
        "full": TearsheetTemplate.full,
    }

    if name not in templates:
        raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")

    return templates[name]()


# CSS styles for HTML tearsheet
TEARSHEET_CSS = """
<style>
    :root {
        /* Typography */
        --c-text: #1a1a2e;
        --c-text-secondary: #475569;
        --c-text-muted: #64748b;
        /* Surfaces */
        --c-bg: #ffffff;
        --c-surface: #fafaf9;
        --c-border: #e2e8f0;
        --c-border-light: #f1f5f9;
        /* Brand */
        --c-accent: #2563eb;
        --c-accent-light: #eff6ff;
        --c-header-bg: #0a1628;
        --c-header-text: #f8f8f6;
        /* Semantic: status */
        --c-positive: #10b981;
        --c-negative: #ef4444;
        --c-warning: #f59e0b;
        --c-neutral: #64748b;
        /* Semantic: data */
        --c-series-primary: #0a1628;
        --c-series-secondary: #d4a84b;
        --c-series-tertiary: #1a2d4a;
        --c-benchmark: #d4a84b;
        --c-fill-light: rgba(10,22,40,0.06);
        /* Quantile ramp (red→green, 5 steps) */
        --c-q1: #ef4444;
        --c-q2: #c87533;
        --c-q3: #e4b85b;
        --c-q4: #6b9f9e;
        --c-q5: #10b981;
        /* Fonts */
        --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
        --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    }

    [data-theme="dark"] {
        --c-text: #e8e8ec;
        --c-text-secondary: #a0a0b0;
        --c-text-muted: #707080;
        --c-bg: #0a1628;
        --c-surface: #152238;
        --c-border: #1a2d4a;
        --c-border-light: #152238;
        --c-accent: #60a5fa;
        --c-accent-light: #1e293b;
        --c-header-bg: #0a1628;
        --c-header-text: #e8e8ec;
        --c-positive: #34d399;
        --c-negative: #f87171;
        --c-warning: #fbbf24;
        --c-neutral: #94a3b8;
        --c-series-primary: #e8e8ec;
        --c-series-secondary: #d4a84b;
        --c-series-tertiary: #60a5fa;
        --c-benchmark: #d4a84b;
        --c-fill-light: rgba(255,255,255,0.06);
    }

    *, *::before, *::after { box-sizing: border-box; }

    body {
        font-family: var(--font-sans);
        font-size: 13px;
        line-height: 1.5;
        color: var(--c-text);
        background: var(--c-bg);
        margin: 0;
        padding: 0;
        -webkit-font-smoothing: antialiased;
    }

    .tearsheet-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0;
    }

    .tearsheet-shell {
        display: flex;
        flex-direction: column;
    }

    /* -- Masthead ---------------------------------------- */
    .report-masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: var(--c-header-bg);
        color: var(--c-header-text);
        border-bottom: 3px solid var(--c-accent);
    }

    .report-title {
        margin: 0;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .report-title:empty { display: none; }

    .report-subtitle {
        margin: 2px 0 0;
        font-size: 12px;
        opacity: 0.7;
    }

    .report-subtitle:empty { display: none; }

    .report-meta {
        display: flex;
        gap: 16px;
        font-size: 11px;
        opacity: 0.8;
    }

    .report-meta-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 10px;
        opacity: 0.6;
    }

    .report-meta-value {
        font-variant-numeric: tabular-nums;
    }

    /* -- Workspace Tabs ---------------------------------- */
    .report-main {
        display: flex;
        flex-direction: column;
    }

    .workspace-shell {
        display: flex;
        flex-direction: column;
    }

    .workspace-tabs {
        display: flex;
        gap: 0;
        padding: 0 24px;
        background: var(--c-surface);
        border-bottom: 1px solid var(--c-border);
        overflow-x: auto;
    }

    .workspace-tab {
        appearance: none;
        border: none;
        border-bottom: 3px solid transparent;
        padding: 10px 16px;
        background: transparent;
        color: var(--c-text-muted);
        font-family: var(--font-sans);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        cursor: pointer;
        white-space: nowrap;
        transition: color 120ms, border-color 120ms;
    }

    .workspace-tab:hover {
        color: var(--c-text);
    }

    .workspace-tab.is-active,
    .workspace-tab[aria-selected="true"] {
        color: var(--c-accent);
        border-bottom-color: var(--c-accent);
        background: rgba(99, 110, 250, 0.06);
    }

    /* -- Workspace Panels -------------------------------- */
    .workspace-panels {
        display: flex;
        flex-direction: column;
    }

    .workspace-panel {
        display: none;
        flex-direction: column;
    }

    .workspace-panel.is-active {
        display: flex;
    }

    .workspace-header {
        display: none;
    }

    /* -- Section Flow ------------------------------------ */
    .report-section-flow {
        display: flex;
        flex-direction: column;
    }

    .report-section {
        display: flex;
        flex-direction: column;
        padding: 16px 24px;
        border-bottom: 1px solid var(--c-border-light);
        min-width: 0;
    }

    .report-section:last-child {
        border-bottom: none;
    }

    /* Compact strip sections — reduce vertical padding */
    .report-section[data-section="credibility_box"],
    .report-section[data-section="cost_summary_line"],
    .report-section[data-section="top_contributors"] {
        padding: 8px 24px;
    }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: var(--c-text);
        margin: 0 0 10px;
        padding: 0;
        border: none;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .section-title--hidden {
        display: none;
    }

    /* -- Chart Containers -------------------------------- */
    .chart-container {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        min-width: 0;
        width: 100%;
        overflow: hidden;
    }

    .chart-container .plotly-graph-div,
    .chart-container .js-plotly-plot {
        width: 100% !important;
    }

    /* -- KPI Strip --------------------------------------- */
    .executive-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0;
        border: 1px solid var(--c-border);
        border-radius: 6px;
        overflow: hidden;
        background: var(--c-surface);
    }

    .executive-kpi {
        flex: 1 1 140px;
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: 12px 16px;
        border-right: 1px solid var(--c-border-light);
        min-width: 0;
    }

    .executive-kpi:last-child {
        border-right: none;
    }

    .executive-kpi-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--c-text-muted);
    }

    .executive-kpi-value {
        font-size: 18px;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
        color: var(--c-text);
    }

    .executive-strip-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 2px 8px;
        font-size: 10px;
        color: var(--c-text-muted);
        font-variant-numeric: tabular-nums;
    }

    /* -- Section Pairs (side-by-side) -------------------- */
    .report-section-pair {
        display: flex;
        gap: 0;
        border-bottom: 1px solid var(--c-border-light);
        align-items: stretch;
    }

    .report-section-pair > .report-section {
        flex: 1 1 50%;
        border-bottom: none;
        min-width: 0;
        display: flex;
        flex-direction: column;
    }

    .report-section-pair > .report-section:first-child {
        border-right: 1px solid var(--c-border-light);
    }

    @media (max-width: 900px) {
        .report-section-pair {
            flex-direction: column;
        }
        .report-section-pair > .report-section:first-child {
            border-right: none;
            border-bottom: 1px solid var(--c-border-light);
        }
    }

    /* -- Credibility Box --------------------------------- */
    .credibility-box {
        display: flex;
        flex-wrap: wrap;
        gap: 0;
        border: 1px solid var(--c-border);
        border-left: 3px solid var(--c-accent);
        border-radius: 6px;
        overflow: hidden;
        background: var(--c-surface);
        padding: 0;
        min-height: 40px;
        align-items: center;
    }

    .credibility-item {
        flex: 1 1 180px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-right: 1px solid var(--c-border-light);
    }

    .credibility-item:last-child {
        border-right: none;
    }

    .credibility-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .credibility-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--c-text-muted);
    }

    .credibility-value {
        font-size: 13px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }

    /* -- Cost Summary Line ------------------------------- */
    .cost-summary-line {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 20px;
        padding: 8px 14px;
        border: 1px solid var(--c-border);
        border-radius: 6px;
        background: var(--c-surface);
        font-size: 12px;
        font-variant-numeric: tabular-nums;
        color: var(--c-text-secondary);
        min-height: 40px;
        align-items: center;
    }

    .cost-summary-item {
        display: flex;
        gap: 6px;
    }

    .cost-summary-label {
        font-weight: 400;
    }

    .cost-summary-value {
        font-weight: 700;
        color: var(--c-text);
    }

    /* -- Section Footnotes ------------------------------- */
    .section-footnote {
        font-size: 11px;
        line-height: 1.5;
        color: var(--c-text-muted);
        margin-top: 8px;
        padding: 6px 0 0;
        border-top: 1px solid var(--c-border-light);
    }

    .section-footnote strong {
        color: var(--c-text-secondary);
        font-weight: 600;
    }

    /* -- Metrics Table ----------------------------------- */
    .metrics-table-wrap {
        overflow-x: auto;
        border: 1px solid var(--c-border);
        border-radius: 6px;
        background: var(--c-bg);
    }

    .metrics-table-intro {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 12px;
        border-bottom: 1px solid var(--c-border);
        background: var(--c-surface);
    }

    .metrics-table-intro-chip {
        padding: 0;
        border: 0;
        background: transparent;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--c-text-muted);
    }

    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        font-variant-numeric: tabular-nums;
    }

    .metrics-table thead th {
        padding: 8px 12px;
        text-align: right;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--c-text-muted);
        background: var(--c-surface);
        border-bottom: 2px solid var(--c-border);
        white-space: nowrap;
    }

    .metrics-table thead th:first-child,
    .metrics-table thead th:nth-child(3),
    .metrics-table tbody td:first-child,
    .metrics-table tbody td:nth-child(3),
    .metrics-table tbody th:first-child {
        text-align: left;
    }

    .metrics-table tbody td {
        padding: 5px 12px;
        border-bottom: 1px solid var(--c-border-light);
        text-align: right;
        vertical-align: baseline;
    }

    .metrics-table-row:hover td {
        background: var(--c-accent-light);
    }

    .metrics-table tbody tr:last-child td,
    .metrics-table tbody tr:last-child th {
        border-bottom: none;
    }

    .metrics-table-group th {
        padding: 10px 12px 4px;
        background: transparent;
        border-bottom: 1px solid var(--c-border);
    }

    .metrics-group-heading {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--c-accent);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: var(--c-text);
    }

    .metric-primary {
        font-weight: 700;
        color: var(--c-text);
    }

    .metric-benchmark {
        color: var(--c-text-secondary);
    }

    .metric-spread {
        display: inline-flex;
        align-items: center;
        font-variant-numeric: tabular-nums;
    }

    .metric-spread-value {
        font-weight: 700;
        color: var(--c-text);
    }

    .metric-spread--na {
        color: var(--c-text-muted);
    }

    .metrics-table-empty {
        margin: 0;
        padding: 12px;
        color: var(--c-text-muted);
    }

    /* -- Wide + Narrow (sidebar layout) ------------------ */
    .report-section-sidebar {
        display: flex;
        gap: 0;
        border-bottom: 1px solid var(--c-border-light);
    }

    .report-section-sidebar > .report-section:first-child {
        flex: 2 1 66%;
        border-bottom: none;
        min-width: 0;
    }

    .report-section-sidebar > .report-section:last-child {
        flex: 1 1 34%;
        border-bottom: none;
        border-left: 1px solid var(--c-border-light);
        min-width: 0;
    }

    @media (max-width: 900px) {
        .report-section-sidebar {
            flex-direction: column;
        }
        .report-section-sidebar > .report-section:last-child {
            border-left: none;
            border-top: 1px solid var(--c-border-light);
        }
    }

    /* -- Metrics Sidebar --------------------------------- */
    .metrics-sidebar {
        padding: 12px 16px;
        font-size: 12px;
        line-height: 1.6;
        overflow-y: auto;
    }

    .metrics-sidebar-group {
        margin-bottom: 10px;
    }

    .metrics-sidebar-group:last-child {
        margin-bottom: 0;
    }

    .metrics-sidebar-header {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--c-text-muted);
        padding-bottom: 4px;
        border-bottom: 1px solid var(--c-border-light);
        margin-bottom: 4px;
    }

    .metrics-sidebar-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 1px 0;
    }

    .metrics-sidebar-label {
        color: var(--c-text-secondary);
        white-space: nowrap;
    }

    .metrics-sidebar-value {
        font-weight: 500;
        font-variant-numeric: tabular-nums;
        text-align: right;
        padding-left: 8px;
    }

    .metrics-sidebar-value.negative {
        color: var(--c-negative);
    }

    /* -- Validity Card ----------------------------------- */
    .validity-card {
        border: 1px solid var(--c-border);
        border-radius: 6px;
        overflow: hidden;
        background: var(--c-surface);
    }

    .validity-card-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--c-text-muted);
        padding: 10px 16px;
        border-bottom: 1px solid var(--c-border-light);
    }

    .validity-card-body {
        display: flex;
    }

    .validity-column {
        flex: 1;
        padding: 12px 16px;
        border-right: 1px solid var(--c-border-light);
    }

    .validity-column:last-child {
        border-right: none;
    }

    .validity-column-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        font-weight: 600;
        color: var(--c-text);
        margin-bottom: 8px;
    }

    .validity-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .validity-metric {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 2px 0;
        font-size: 12px;
    }

    .validity-metric-label {
        color: var(--c-text-secondary);
    }

    .validity-metric-value {
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    .validity-context {
        font-size: 11px;
        color: var(--c-text-muted);
        margin-top: 6px;
        font-style: italic;
    }

    @media (max-width: 900px) {
        .validity-card-body {
            flex-direction: column;
        }
        .validity-column {
            border-right: none;
            border-bottom: 1px solid var(--c-border-light);
        }
        .validity-column:last-child {
            border-bottom: none;
        }
    }

    /* -- Data Tables (dense, financial) ------------------ */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
    }

    .data-table th {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        color: var(--c-text-muted);
        padding: 8px 12px;
        border-bottom: 2px solid var(--c-border);
        text-align: left;
    }

    .data-table th.num {
        text-align: right;
    }

    .data-table td {
        padding: 6px 12px;
        border-bottom: 1px solid var(--c-border-light);
    }

    .data-table tr:nth-child(even) {
        background: var(--c-surface);
    }

    .data-table .num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .data-table .negative {
        color: var(--c-negative);
    }

    .data-table .positive {
        color: var(--c-positive);
    }

    /* -- Collapsible Detail ------------------------------ */
    .section-detail {
        border-bottom: 1px solid var(--c-border-light);
    }

    .section-detail:last-child {
        border-bottom: none;
    }

    .section-detail-summary {
        display: flex;
        align-items: center;
        padding: 10px 24px;
        font-size: 12px;
        font-weight: 600;
        color: var(--c-text-secondary);
        cursor: pointer;
        user-select: none;
        list-style: none;
    }

    .section-detail-summary::-webkit-details-marker {
        display: none;
    }

    .section-detail-summary::before {
        content: '\25b8';
        display: inline-block;
        margin-right: 8px;
        font-size: 11px;
        transition: transform 120ms;
    }

    .section-detail[open] > .section-detail-summary::before {
        transform: rotate(90deg);
    }

    .section-detail-summary:hover {
        color: var(--c-text);
        background: var(--c-accent-light);
    }

    /* -- Responsive -------------------------------------- */
    @media (max-width: 900px) {
        .report-masthead {
            flex-direction: column;
            gap: 8px;
            padding: 12px 16px;
        }

        .workspace-tabs {
            padding: 0 16px;
        }

        .report-section {
            padding: 12px 16px;
        }

        .executive-kpi {
            flex-basis: 120px;
        }
    }

    @media (max-width: 600px) {
        .executive-strip {
            flex-direction: column;
        }

        .executive-kpi {
            border-right: none;
            border-bottom: 1px solid var(--c-border-light);
        }

        .executive-kpi:last-child {
            border-bottom: none;
        }

        .workspace-tab {
            padding: 8px 12px;
            font-size: 11px;
        }
    }

    /* -- Print ------------------------------------------- */
    @media print {
        .workspace-tabs { display: none; }
        .workspace-panel { display: flex !important; }
        .report-section { break-inside: avoid; }
        body { font-size: 11px; }
    }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{document_title}</title>
    {css}
    {plotly_js}
</head>
<body>
    <div class="tearsheet-container">
        <div class="tearsheet-shell">
        <header class="report-masthead">
            <div>
                <h1 class="report-title">{title}</h1>
                <p class="report-subtitle">{subtitle}</p>
            </div>
            <aside class="report-meta">
                <div>
                    <div class="report-meta-label">Report Date</div>
                    <div class="report-meta-value">{timestamp}</div>
                </div>
            </aside>
        </header>

        <main class="report-main">
            {sections_html}
        </main>
        </div>
    </div>
    <script>
        (() => {{
            const tabs = Array.from(document.querySelectorAll(".workspace-tab"));
            const panels = Array.from(document.querySelectorAll(".workspace-panel"));
            if (!tabs.length || !panels.length) return;

            const activate = (targetId) => {{
                tabs.forEach((tab) => {{
                    const active = tab.dataset.target === targetId;
                    tab.classList.toggle("is-active", active);
                    tab.setAttribute("aria-selected", active ? "true" : "false");
                }});
                panels.forEach((panel) => {{
                    const active = panel.dataset.workspace === targetId;
                    panel.classList.toggle("is-active", active);
                    panel.setAttribute("aria-hidden", active ? "false" : "true");
                    if (active) {{
                        panel.querySelectorAll(".js-plotly-plot").forEach((plot) => {{
                            if (window.Plotly) window.Plotly.Plots.resize(plot);
                        }});
                    }}
                }});
            }};

            tabs.forEach((tab) => {{
                tab.addEventListener("click", () => activate(tab.dataset.target));
            }});

            activate(tabs[0].dataset.target);
        }})();
    </script>
</body>
</html>
"""
