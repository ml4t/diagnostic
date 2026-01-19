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
    enabled: bool = True
    priority: int = 0  # Lower = higher priority (shown first)
    description: str = ""


@dataclass
class TearsheetTemplate:
    """Template configuration for a tearsheet."""

    name: str
    description: str
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
            sections=[
                TearsheetSection("executive_summary", "Executive Summary", priority=0),
                TearsheetSection("key_insights", "Key Insights", priority=1),
                TearsheetSection("mfe_mae", "Exit Efficiency (MFE/MAE)", priority=2),
                TearsheetSection("exit_reasons", "Exit Reason Breakdown", priority=3),
                TearsheetSection("trade_waterfall", "Trade-by-Trade PnL", priority=4),
                TearsheetSection("duration", "Trade Duration Analysis", priority=5),
                TearsheetSection("consecutive", "Win/Loss Streaks", priority=6),
                TearsheetSection("size_return", "Position Size Analysis", priority=7),
                TearsheetSection("shap_errors", "SHAP Error Patterns", priority=8, enabled=False),
                # Disabled by default
                TearsheetSection("equity_curve", "Equity Curve", priority=10, enabled=False),
                TearsheetSection("drawdowns", "Drawdowns", priority=11, enabled=False),
                TearsheetSection("dsr", "Statistical Validity", priority=12, enabled=False),
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
            sections=[
                TearsheetSection("executive_summary", "Executive Summary", priority=0),
                TearsheetSection("key_insights", "Key Insights", priority=1),
                TearsheetSection("equity_curve", "Equity Curve", priority=2),
                TearsheetSection("drawdowns", "Drawdown Analysis", priority=3),
                TearsheetSection("cost_waterfall", "Cost Attribution", priority=4),
                TearsheetSection("cost_sensitivity", "Cost Sensitivity", priority=5),
                TearsheetSection("rolling_metrics", "Rolling Performance", priority=6),
                TearsheetSection("monthly_returns", "Monthly Returns Heatmap", priority=7),
                TearsheetSection("annual_returns", "Annual Returns", priority=8),
                # Disabled by default
                TearsheetSection("mfe_mae", "Exit Efficiency", priority=10, enabled=False),
                TearsheetSection("trade_waterfall", "Trade Details", priority=11, enabled=False),
                TearsheetSection("dsr", "Statistical Tests", priority=12, enabled=False),
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
            sections=[
                TearsheetSection("executive_summary", "Executive Summary", priority=0),
                TearsheetSection(
                    "statistical_summary", "Statistical Validity Overview", priority=1
                ),
                TearsheetSection("dsr_gauge", "Deflated Sharpe Ratio", priority=2),
                TearsheetSection("confidence_intervals", "Metric Confidence Intervals", priority=3),
                TearsheetSection("min_trl", "Minimum Track Record", priority=4),
                TearsheetSection(
                    "ras_analysis", "RAS Overfitting Check", priority=5, enabled=False
                ),
                TearsheetSection("drawdowns", "Drawdown Analysis", priority=6),
                TearsheetSection("tail_risk", "Tail Risk (VaR/CVaR)", priority=7),
                TearsheetSection("distribution", "Returns Distribution", priority=8),
                # Disabled by default
                TearsheetSection("mfe_mae", "Exit Efficiency", priority=10, enabled=False),
                TearsheetSection("cost_waterfall", "Cost Attribution", priority=11, enabled=False),
            ],
        )

    @classmethod
    def full(cls) -> TearsheetTemplate:
        """Complete template with all available sections."""
        return cls(
            name="full",
            description="Comprehensive analysis with all available visualizations",
            sections=[
                # Executive
                TearsheetSection("executive_summary", "Executive Summary", priority=0),
                TearsheetSection("key_insights", "Key Insights", priority=1),
                # Performance
                TearsheetSection("equity_curve", "Equity Curve", priority=10),
                TearsheetSection("drawdowns", "Drawdown Analysis", priority=11),
                TearsheetSection("monthly_returns", "Monthly Returns", priority=12),
                TearsheetSection("annual_returns", "Annual Returns", priority=13),
                TearsheetSection("rolling_metrics", "Rolling Performance", priority=14),
                # Trade Analysis
                TearsheetSection("mfe_mae", "Exit Efficiency (MFE/MAE)", priority=20),
                TearsheetSection("exit_reasons", "Exit Reason Breakdown", priority=21),
                TearsheetSection("trade_waterfall", "Trade-by-Trade PnL", priority=22),
                TearsheetSection("duration", "Trade Duration Analysis", priority=23),
                TearsheetSection("consecutive", "Win/Loss Streaks", priority=24),
                TearsheetSection("size_return", "Position Size Analysis", priority=25),
                # Cost Attribution
                TearsheetSection("cost_waterfall", "Cost Attribution", priority=30),
                TearsheetSection("cost_sensitivity", "Cost Sensitivity", priority=31),
                TearsheetSection("cost_by_asset", "Costs by Asset", priority=32),
                # Statistical Validity
                TearsheetSection("statistical_summary", "Statistical Validity", priority=40),
                TearsheetSection("dsr_gauge", "Deflated Sharpe Ratio", priority=41),
                TearsheetSection("confidence_intervals", "Confidence Intervals", priority=42),
                TearsheetSection("min_trl", "Minimum Track Record", priority=43),
                TearsheetSection("ras_analysis", "RAS Analysis", priority=44, enabled=False),
                # Distribution & Risk
                TearsheetSection("distribution", "Returns Distribution", priority=50),
                TearsheetSection("tail_risk", "Tail Risk", priority=51),
                # SHAP (optional, requires model)
                TearsheetSection("shap_errors", "SHAP Error Patterns", priority=60, enabled=False),
            ],
        )

    def get_enabled_sections(self) -> list[TearsheetSection]:
        """Return only enabled sections, sorted by priority."""
        return sorted(
            [s for s in self.sections if s.enabled],
            key=lambda s: s.priority,
        )

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
        --primary-color: #636EFA;
        --success-color: #00CC96;
        --warning-color: #FECB52;
        --danger-color: #EF553B;
        --text-color: #2E2E2E;
        --bg-color: #FFFFFF;
        --card-bg: #F8F9FA;
        --border-color: #DEE2E6;
    }

    [data-theme="dark"] {
        --primary-color: #636EFA;
        --success-color: #00CC96;
        --warning-color: #FECB52;
        --danger-color: #EF553B;
        --text-color: #E0E0E0;
        --bg-color: #1E1E1E;
        --card-bg: #2D2D2D;
        --border-color: #404040;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        color: var(--text-color);
        background-color: var(--bg-color);
        margin: 0;
        padding: 20px;
        line-height: 1.6;
    }

    .tearsheet-container {
        max-width: 1400px;
        margin: 0 auto;
    }

    .tearsheet-header {
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid var(--border-color);
    }

    .tearsheet-header h1 {
        margin: 0 0 10px 0;
        font-size: 2em;
    }

    .tearsheet-header .subtitle {
        color: #666;
        font-size: 1.1em;
    }

    .section {
        margin-bottom: 40px;
    }

    .section-title {
        font-size: 1.4em;
        font-weight: 600;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color);
    }

    .chart-container {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }

    .row {
        display: flex;
        flex-wrap: wrap;
        margin: -10px;
    }

    .col-6 {
        flex: 0 0 50%;
        padding: 10px;
        box-sizing: border-box;
    }

    .col-12 {
        flex: 0 0 100%;
        padding: 10px;
        box-sizing: border-box;
    }

    @media (max-width: 900px) {
        .col-6 {
            flex: 0 0 100%;
        }
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
        color: #666;
        font-size: 0.9em;
    }

    .timestamp {
        font-size: 0.85em;
        color: #888;
    }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {css}
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <div class="tearsheet-container">
        <header class="tearsheet-header">
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        {sections_html}

        <footer class="footer">
            <p>Generated by ml4t-diagnostic | State-of-the-art backtest analysis</p>
        </footer>
    </div>
</body>
</html>
"""
