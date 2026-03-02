"""Unified backtest tearsheet generation.

The main entry point for generating comprehensive backtest reports.
Combines all visualization modules into a single, publication-quality
HTML document.

This is the primary interface users should use:
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    html = generate_backtest_tearsheet(
        backtest_result,
        template="full",
        theme="default",
        output_path="report.html",
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .template_system import (
    HTML_TEMPLATE,
    TEARSHEET_CSS,
    TearsheetTemplate,
    get_template,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go
    import polars as pl


def generate_backtest_tearsheet(
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
    theme: Literal["default", "dark", "print", "presentation"] = "default",
    title: str = "Backtest Analysis Report",
    subtitle: str = "",
    benchmark_returns: pl.Series | np.ndarray | None = None,
    n_trials: int | None = None,
    interactive: bool = True,
    include_plotlyjs: bool = True,
) -> str:
    """Generate a comprehensive backtest tearsheet.

    This is the main entry point for creating publication-quality backtest
    reports. It combines all visualization modules into a single HTML document.

    Parameters
    ----------
    trades : pl.DataFrame, optional
        Trade records with columns like: symbol, entry_time, exit_time,
        pnl, gross_pnl, net_pnl, mfe, mae, exit_reason, duration, size
    returns : pl.Series or np.ndarray, optional
        Daily returns series for portfolio-level analysis
    equity_curve : pl.DataFrame, optional
        Equity curve with date and equity columns
    metrics : dict, optional
        Pre-computed metrics dict with keys like:
        - sharpe, cagr, max_drawdown, win_rate, profit_factor
        - dsr_probability, min_trl, etc. for statistical validity
    output_path : str or Path, optional
        If provided, save HTML to this path
    template : {"quant_trader", "hedge_fund", "risk_manager", "full"}
        Template persona to use (determines which sections are shown)
    theme : {"default", "dark", "print", "presentation"}
        Visual theme for the charts
    title : str
        Report title
    subtitle : str
        Report subtitle (e.g., strategy name, date range)
    benchmark_returns : pl.Series or np.ndarray, optional
        Benchmark returns for comparison
    n_trials : int, optional
        Number of trials for DSR calculation
    interactive : bool
        Whether charts should be interactive (vs static images)
    include_plotlyjs : bool
        Whether to include Plotly.js (set False if already loaded)

    Returns
    -------
    str
        HTML string of the complete tearsheet

    Examples
    --------
    >>> from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet
    >>>
    >>> # From trades DataFrame
    >>> html = generate_backtest_tearsheet(
    ...     trades=my_trades,
    ...     metrics={"sharpe": 1.5, "max_drawdown": -0.15},
    ...     template="quant_trader",
    ...     output_path="strategy_report.html",
    ... )
    >>>
    >>> # From returns series
    >>> html = generate_backtest_tearsheet(
    ...     returns=daily_returns,
    ...     template="risk_manager",
    ...     n_trials=100,  # For DSR
    ... )
    """
    # Get template
    tmpl = get_template(template)

    # Generate sections HTML
    sections_html = _generate_sections(
        tmpl,
        trades=trades,
        returns=returns,
        equity_curve=equity_curve,
        metrics=metrics,
        benchmark_returns=benchmark_returns,
        n_trials=n_trials,
        theme=theme,
        interactive=interactive,
    )

    # Generate full HTML - conditionally include Plotly JS
    if include_plotlyjs:
        css = TEARSHEET_CSS
    else:
        css = TEARSHEET_CSS.replace(
            '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
            "",
        )

    html = HTML_TEMPLATE.format(
        theme=theme if theme == "dark" else "light",
        title=title,
        subtitle=subtitle or f"Template: {template}",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        css=css,
        sections_html=sections_html,
    )

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

    return html


def _generate_sections(
    template: TearsheetTemplate,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    n_trials: int | None = None,
    theme: str = "default",
    interactive: bool = True,
) -> str:
    """Generate HTML for all enabled sections."""
    sections_html = []

    for section in template.get_enabled_sections():
        section_html = _generate_section(
            section.name,
            section.title,
            trades=trades,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            n_trials=n_trials,
            theme=theme,
            interactive=interactive,
        )
        if section_html:
            sections_html.append(section_html)

    return "\n".join(sections_html)


def _generate_section(
    section_name: str,
    section_title: str,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    n_trials: int | None = None,
    theme: str = "default",
    interactive: bool = True,
) -> str | None:
    """Generate HTML for a single section."""
    try:
        fig = _create_section_figure(
            section_name,
            trades=trades,
            returns=returns,
            equity_curve=equity_curve,
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            n_trials=n_trials,
            theme=theme,
        )

        if fig is None:
            return None

        # Convert figure to HTML
        if interactive:
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        else:
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)

        return f"""
        <section class="section">
            <h2 class="section-title">{section_title}</h2>
            <div class="chart-container">
                {fig_html}
            </div>
        </section>
        """

    except Exception as e:
        # Log error but don't fail the whole report
        return f"""
        <section class="section">
            <h2 class="section-title">{section_title}</h2>
            <div class="chart-container">
                <p style="color: #999;">Section unavailable: {str(e)}</p>
            </div>
        </section>
        """


def _create_section_figure(
    section_name: str,
    trades: pl.DataFrame | None = None,
    returns: pl.Series | np.ndarray | None = None,
    equity_curve: pl.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    benchmark_returns: pl.Series | np.ndarray | None = None,
    n_trials: int | None = None,
    theme: str = "default",
) -> go.Figure | None:
    """Create the Plotly figure for a specific section."""

    metrics = metrics or {}

    # Executive Summary sections
    if section_name == "executive_summary":
        if not metrics:
            return None
        from .executive_summary import create_executive_summary

        return create_executive_summary(metrics, theme=theme)

    if section_name == "key_insights":
        if not metrics:
            return None
        from .executive_summary import create_key_insights

        insights = create_key_insights(metrics)
        # Create a simple text figure for insights
        import plotly.graph_objects as go

        fig = go.Figure()
        insight_text = "<br>".join([f"• [{i.category.upper()}] {i.message}" for i in insights])
        fig.add_annotation(
            text=insight_text or "No insights available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
            align="left",
        )
        fig.update_layout(
            height=max(150, len(insights) * 40 + 50),
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig

    # Trade Analysis sections
    if section_name == "mfe_mae":
        if trades is None:
            return None
        from .trade_plots import plot_mfe_mae_scatter

        return plot_mfe_mae_scatter(trades, theme=theme)

    if section_name == "exit_reasons":
        if trades is None:
            return None
        from .trade_plots import plot_exit_reason_breakdown

        return plot_exit_reason_breakdown(trades, theme=theme)

    if section_name == "trade_waterfall":
        if trades is None:
            return None
        from .trade_plots import plot_trade_waterfall

        return plot_trade_waterfall(trades, theme=theme)

    if section_name == "duration":
        if trades is None:
            return None
        from .trade_plots import plot_trade_duration_distribution

        return plot_trade_duration_distribution(trades, theme=theme)

    if section_name == "consecutive":
        if trades is None:
            return None
        from .trade_plots import plot_consecutive_analysis

        return plot_consecutive_analysis(trades, theme=theme)

    if section_name == "size_return":
        if trades is None:
            return None
        from .trade_plots import plot_trade_size_vs_return

        return plot_trade_size_vs_return(trades, theme=theme)

    # Cost Attribution sections
    if section_name == "cost_waterfall":
        gross_pnl = metrics.get("gross_pnl")
        commission = metrics.get("commission", 0)
        slippage = metrics.get("slippage", 0)
        if gross_pnl is None:
            return None
        from .cost_attribution import plot_cost_waterfall

        return plot_cost_waterfall(
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            theme=theme,
        )

    if section_name == "cost_sensitivity":
        if returns is None:
            return None
        from .cost_attribution import plot_cost_sensitivity

        return plot_cost_sensitivity(returns, theme=theme)

    if section_name == "cost_by_asset":
        if trades is None:
            return None
        from .cost_attribution import plot_cost_by_asset

        return plot_cost_by_asset(trades, theme=theme)

    # Statistical Validity sections
    if section_name == "statistical_summary":
        from .statistical_validity import plot_statistical_summary_card

        return plot_statistical_summary_card(metrics, theme=theme)

    if section_name == "dsr_gauge":
        dsr_prob = metrics.get("dsr_probability")
        sharpe = metrics.get("sharpe")
        if dsr_prob is None or sharpe is None:
            return None
        from .statistical_validity import plot_dsr_gauge

        return plot_dsr_gauge(
            dsr_probability=dsr_prob,
            observed_sharpe=sharpe,
            expected_max_sharpe=metrics.get("expected_max_sharpe"),
            n_trials=n_trials,
            theme=theme,
        )

    if section_name == "confidence_intervals":
        # Build CI dict from metrics
        ci_metrics = {}
        for key in ["sharpe", "cagr", "max_drawdown"]:
            if key in metrics:
                ci_metrics[key] = {
                    "point": metrics[key],
                    "lower_95": metrics.get(f"{key}_lower_95", metrics[key] * 0.7),
                    "upper_95": metrics.get(f"{key}_upper_95", metrics[key] * 1.3),
                }
        if not ci_metrics:
            return None
        from .statistical_validity import plot_confidence_intervals

        return plot_confidence_intervals(ci_metrics, theme=theme)

    if section_name == "min_trl":
        sharpe = metrics.get("sharpe")
        periods = metrics.get("n_periods", metrics.get("n_observations"))
        if sharpe is None or periods is None:
            return None
        from .statistical_validity import plot_minimum_track_record

        return plot_minimum_track_record(
            observed_sharpe=sharpe,
            current_periods=periods,
            theme=theme,
        )

    if section_name == "ras_analysis":
        original_ic = metrics.get("original_ic")
        adjusted_ic = metrics.get("ras_adjusted_ic")
        rademacher = metrics.get("rademacher_complexity")
        if original_ic is None or adjusted_ic is None or rademacher is None:
            return None
        from .statistical_validity import plot_ras_analysis

        return plot_ras_analysis(
            original_ic=float(original_ic),
            adjusted_ic=float(adjusted_ic),
            rademacher_complexity=float(rademacher),
            theme=theme,
        )

    # Portfolio-level sections — delegate to portfolio visualization functions
    # These require a returns series; create a PortfolioAnalysis on demand
    if section_name in (
        "equity_curve",
        "drawdowns",
        "monthly_returns",
        "annual_returns",
        "rolling_metrics",
        "distribution",
    ):
        if returns is None:
            return None

        import polars as pl

        from ml4t.diagnostic.evaluation import PortfolioAnalysis

        # Build PortfolioAnalysis from returns array
        if isinstance(returns, np.ndarray):
            ret_series = pl.Series("returns", returns)
        else:
            ret_series = returns

        # Infer periods_per_year from data length and date range (heuristic)
        n = len(ret_series)
        periods_per_year = 252  # default daily
        if n < 100:
            periods_per_year = 12  # likely monthly

        bench = None
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, np.ndarray):
                bench = pl.Series("benchmark", benchmark_returns)
            else:
                bench = benchmark_returns

        pa = PortfolioAnalysis(ret_series, benchmark=bench, periods_per_year=periods_per_year)

        if section_name == "equity_curve":
            from ml4t.diagnostic.visualization.portfolio import plot_cumulative_returns

            return plot_cumulative_returns(pa, theme=theme)

        if section_name == "drawdowns":
            from ml4t.diagnostic.visualization.portfolio import plot_drawdown_underwater

            return plot_drawdown_underwater(pa, theme=theme)

        if section_name == "monthly_returns":
            from ml4t.diagnostic.visualization.portfolio import plot_monthly_returns_heatmap

            return plot_monthly_returns_heatmap(pa, theme=theme)

        if section_name == "annual_returns":
            from ml4t.diagnostic.visualization.portfolio import plot_annual_returns_bar

            return plot_annual_returns_bar(pa, theme=theme)

        if section_name == "rolling_metrics":
            from ml4t.diagnostic.visualization.portfolio import plot_rolling_sharpe

            rolling = pa.compute_rolling_metrics(windows=[63, 252], metrics=["sharpe"])
            return plot_rolling_sharpe(rolling_result=rolling, windows=[63, 252], theme=theme)

        if section_name == "distribution":
            from ml4t.diagnostic.visualization.portfolio import plot_returns_distribution

            return plot_returns_distribution(pa, theme=theme)

    if section_name == "tail_risk":
        # Tail risk analysis not yet implemented
        return None

    # Unknown section
    return None


class BacktestTearsheet:
    """Object-oriented interface for building tearsheets incrementally.

    Provides a fluent API for customizing tearsheet content before generation.

    Examples
    --------
    >>> tearsheet = BacktestTearsheet(template="quant_trader")
    >>> tearsheet.add_trades(my_trades)
    >>> tearsheet.add_metrics({"sharpe": 1.5, "max_drawdown": -0.15})
    >>> tearsheet.enable_section("dsr_gauge")
    >>> html = tearsheet.generate()
    """

    def __init__(
        self,
        template: Literal["quant_trader", "hedge_fund", "risk_manager", "full"] = "full",
        theme: Literal["default", "dark", "print", "presentation"] = "default",
        title: str = "Backtest Analysis Report",
    ):
        """Initialize tearsheet builder."""
        self.template = get_template(template)
        self.theme = theme
        self.title = title
        self.subtitle = ""

        # Data
        self.trades: pl.DataFrame | None = None
        self.returns: pl.Series | np.ndarray | None = None
        self.equity_curve: pl.DataFrame | None = None
        self.metrics: dict[str, Any] = {}
        self.benchmark_returns: pl.Series | np.ndarray | None = None
        self.n_trials: int | None = None

    def add_trades(self, trades: pl.DataFrame) -> BacktestTearsheet:
        """Add trade records to the tearsheet."""
        self.trades = trades
        return self

    def add_returns(self, returns: pl.Series | np.ndarray) -> BacktestTearsheet:
        """Add daily returns series."""
        self.returns = returns
        return self

    def add_equity_curve(self, equity: pl.DataFrame) -> BacktestTearsheet:
        """Add equity curve DataFrame."""
        self.equity_curve = equity
        return self

    def add_metrics(self, metrics: dict[str, Any]) -> BacktestTearsheet:
        """Add or update metrics dictionary."""
        self.metrics.update(metrics)
        return self

    def add_benchmark(self, returns: pl.Series | np.ndarray) -> BacktestTearsheet:
        """Add benchmark returns for comparison."""
        self.benchmark_returns = returns
        return self

    def set_n_trials(self, n: int) -> BacktestTearsheet:
        """Set number of trials for DSR calculation."""
        self.n_trials = n
        return self

    def set_title(self, title: str, subtitle: str = "") -> BacktestTearsheet:
        """Set report title and subtitle."""
        self.title = title
        self.subtitle = subtitle
        return self

    def enable_section(self, name: str) -> BacktestTearsheet:
        """Enable a section by name."""
        self.template.enable_section(name)
        return self

    def disable_section(self, name: str) -> BacktestTearsheet:
        """Disable a section by name."""
        self.template.disable_section(name)
        return self

    def generate(
        self,
        output_path: str | Path | None = None,
        interactive: bool = True,
    ) -> str:
        """Generate the tearsheet HTML.

        Parameters
        ----------
        output_path : str or Path, optional
            If provided, save HTML to this path
        interactive : bool
            Whether charts should be interactive

        Returns
        -------
        str
            HTML string of the complete tearsheet
        """
        return generate_backtest_tearsheet(
            trades=self.trades,
            returns=self.returns,
            equity_curve=self.equity_curve,
            metrics=self.metrics,
            output_path=output_path,
            template=self.template.name,  # type: ignore
            theme=self.theme,
            title=self.title,
            subtitle=self.subtitle,
            benchmark_returns=self.benchmark_returns,
            n_trials=self.n_trials,
            interactive=interactive,
        )
