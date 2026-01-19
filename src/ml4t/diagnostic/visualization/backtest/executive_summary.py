"""Executive summary visualizations for backtest analysis.

Provides KPI cards with traffic lights (red/yellow/green) and
automated insight generation for backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import (
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Default Thresholds for Traffic Lights
# =============================================================================

DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "sharpe_ratio": {
        "red": (-float("inf"), 0.5),
        "yellow": (0.5, 1.5),
        "green": (1.5, float("inf")),
        "format": "{:.2f}",
        "label": "Sharpe Ratio",
        "higher_is_better": True,
    },
    "sortino_ratio": {
        "red": (-float("inf"), 0.5),
        "yellow": (0.5, 1.5),
        "green": (1.5, float("inf")),
        "format": "{:.2f}",
        "label": "Sortino Ratio",
        "higher_is_better": True,
    },
    "calmar_ratio": {
        "red": (-float("inf"), 0.5),
        "yellow": (0.5, 1.0),
        "green": (1.0, float("inf")),
        "format": "{:.2f}",
        "label": "Calmar Ratio",
        "higher_is_better": True,
    },
    "cagr": {
        "red": (-float("inf"), 0.05),
        "yellow": (0.05, 0.15),
        "green": (0.15, float("inf")),
        "format": "{:.1%}",
        "label": "CAGR",
        "higher_is_better": True,
    },
    "total_return": {
        "red": (-float("inf"), 0.0),
        "yellow": (0.0, 0.20),
        "green": (0.20, float("inf")),
        "format": "{:.1%}",
        "label": "Total Return",
        "higher_is_better": True,
    },
    "max_drawdown": {
        "red": (0.30, float("inf")),
        "yellow": (0.15, 0.30),
        "green": (-float("inf"), 0.15),
        "format": "{:.1%}",
        "label": "Max Drawdown",
        "higher_is_better": False,
    },
    "win_rate": {
        "red": (-float("inf"), 0.40),
        "yellow": (0.40, 0.55),
        "green": (0.55, float("inf")),
        "format": "{:.1%}",
        "label": "Win Rate",
        "higher_is_better": True,
    },
    "profit_factor": {
        "red": (-float("inf"), 1.0),
        "yellow": (1.0, 1.5),
        "green": (1.5, float("inf")),
        "format": "{:.2f}",
        "label": "Profit Factor",
        "higher_is_better": True,
    },
    "expectancy": {
        "red": (-float("inf"), 0.0),
        "yellow": (0.0, 50.0),
        "green": (50.0, float("inf")),
        "format": "${:.2f}",
        "label": "Expectancy",
        "higher_is_better": True,
    },
    "avg_trade": {
        "red": (-float("inf"), 0.0),
        "yellow": (0.0, 25.0),
        "green": (25.0, float("inf")),
        "format": "${:.2f}",
        "label": "Avg Trade",
        "higher_is_better": True,
    },
    "n_trades": {
        "red": (-float("inf"), 30),
        "yellow": (30, 100),
        "green": (100, float("inf")),
        "format": "{:.0f}",
        "label": "Total Trades",
        "higher_is_better": True,
    },
    "volatility": {
        "red": (0.30, float("inf")),
        "yellow": (0.15, 0.30),
        "green": (-float("inf"), 0.15),
        "format": "{:.1%}",
        "label": "Volatility",
        "higher_is_better": False,
    },
}

# Color definitions
TRAFFIC_LIGHT_COLORS = {
    "green": "#28A745",
    "yellow": "#FFC107",
    "red": "#DC3545",
    "neutral": "#6C757D",
}


# =============================================================================
# Traffic Light Functions
# =============================================================================


def get_traffic_light_color(
    value: float,
    metric_name: str,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Determine traffic light color for a metric value.

    Parameters
    ----------
    value : float
        The metric value to evaluate
    metric_name : str
        Name of the metric (must be in thresholds)
    thresholds : dict, optional
        Custom thresholds. Uses DEFAULT_THRESHOLDS if None.

    Returns
    -------
    str
        Color code: "green", "yellow", "red", or "neutral"
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if metric_name not in thresholds:
        return "neutral"

    config = thresholds[metric_name]

    # Handle NaN
    if np.isnan(value):
        return "neutral"

    # Check which range the value falls into
    for color in ["green", "yellow", "red"]:
        low, high = config[color]
        if low <= value < high:
            return color

    return "neutral"


def _format_metric_value(
    value: float,
    metric_name: str,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format a metric value for display.

    Parameters
    ----------
    value : float
        The metric value
    metric_name : str
        Name of the metric
    thresholds : dict, optional
        Thresholds containing format strings

    Returns
    -------
    str
        Formatted value string
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if np.isnan(value):
        return "N/A"

    if metric_name in thresholds:
        fmt = thresholds[metric_name].get("format", "{:.2f}")
        return fmt.format(value)

    return f"{value:.2f}"


def _get_metric_label(
    metric_name: str,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Get display label for a metric.

    Parameters
    ----------
    metric_name : str
        Internal metric name
    thresholds : dict, optional
        Thresholds containing labels

    Returns
    -------
    str
        Human-readable label
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if metric_name in thresholds:
        return thresholds[metric_name].get("label", metric_name.replace("_", " ").title())

    return metric_name.replace("_", " ").title()


# =============================================================================
# Metric Card Creation
# =============================================================================


def create_metric_card(
    metric_name: str,
    value: float,
    *,
    delta: float | None = None,
    delta_reference: str | None = None,
    sparkline_data: list[float] | None = None,
    thresholds: dict[str, dict[str, Any]] | None = None,
    theme: str | None = None,
) -> go.Figure:
    """Create a single KPI metric card with traffic light indicator.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., "sharpe_ratio", "max_drawdown")
    value : float
        Current metric value
    delta : float, optional
        Change from reference (e.g., vs benchmark or previous period)
    delta_reference : str, optional
        Label for delta reference (e.g., "vs Benchmark", "vs YTD")
    sparkline_data : list[float], optional
        Rolling values for mini sparkline
    thresholds : dict, optional
        Custom thresholds for traffic light
    theme : str, optional
        Plot theme

    Returns
    -------
    go.Figure
        Single metric card as Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get traffic light color
    color_name = get_traffic_light_color(value, metric_name, thresholds)
    color = TRAFFIC_LIGHT_COLORS.get(color_name, TRAFFIC_LIGHT_COLORS["neutral"])

    # Get label for metric
    label = _get_metric_label(metric_name, thresholds)

    # Create figure
    fig = go.Figure()

    # Add indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta" if delta is not None else "number",
            value=value,
            number={
                "font": {"size": 48, "color": color},
                "valueformat": _get_plotly_format(metric_name, thresholds),
            },
            delta={
                "reference": value - delta if delta is not None else 0,
                "relative": False,
                "valueformat": ".2%",
            }
            if delta is not None
            else None,
            title={
                "text": f"<b>{label}</b>"
                + (
                    f"<br><span style='font-size:12px'>{delta_reference}</span>"
                    if delta_reference
                    else ""
                ),
                "font": {"size": 16},
            },
            domain={"x": [0, 1], "y": [0.3, 1]},
        )
    )

    # Add sparkline if provided
    if sparkline_data is not None and len(sparkline_data) > 2:
        fig.add_trace(
            go.Scatter(
                y=sparkline_data,
                mode="lines",
                line={"color": color, "width": 2},
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)",
                showlegend=False,
                xaxis="x2",
                yaxis="y2",
            )
        )

        # Add second axis for sparkline
        fig.update_layout(
            xaxis2={
                "domain": [0.1, 0.9],
                "anchor": "y2",
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
            yaxis2={
                "domain": [0.05, 0.25],
                "anchor": "x2",
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
        )

    # Add traffic light circle
    fig.add_shape(
        type="circle",
        x0=0.85,
        y0=0.85,
        x1=0.95,
        y1=0.95,
        xref="paper",
        yref="paper",
        fillcolor=color,
        line={"color": color},
    )

    fig.update_layout(
        height=200,
        width=250,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        **theme_config["layout"],
    )

    return fig


def _get_plotly_format(
    metric_name: str,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Convert Python format string to Plotly d3 format.

    Parameters
    ----------
    metric_name : str
        Metric name
    thresholds : dict, optional
        Thresholds with format strings

    Returns
    -------
    str
        Plotly d3 format string
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if metric_name not in thresholds:
        return ".2f"

    py_fmt = thresholds[metric_name].get("format", "{:.2f}")

    # Convert Python format to d3
    if "%" in py_fmt:
        return ".1%"
    elif "$" in py_fmt:
        return "$.2f"
    elif ".0f" in py_fmt:
        return ".0f"
    elif ".1f" in py_fmt:
        return ".1f"
    else:
        return ".2f"


# =============================================================================
# Executive Summary Grid
# =============================================================================


def create_executive_summary(
    metrics: dict[str, float],
    *,
    selected_metrics: list[str] | None = None,
    thresholds: dict[str, dict[str, Any]] | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    rolling_metrics: dict[str, list[float]] | None = None,
    title: str = "Executive Summary",
    theme: str | None = None,
    cols: int = 3,
    height: int | None = None,
    width: int | None = None,
) -> go.Figure:
    """Create executive summary grid with KPI cards and traffic lights.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary of metric name to value
    selected_metrics : list[str], optional
        Specific metrics to display. If None, uses sensible defaults.
    thresholds : dict, optional
        Custom thresholds for traffic lights
    benchmark_metrics : dict[str, float], optional
        Benchmark values for delta display
    rolling_metrics : dict[str, list[float]], optional
        Rolling values for sparklines
    title : str, default "Executive Summary"
        Dashboard title
    theme : str, optional
        Plot theme ("default", "dark", "print", "presentation")
    cols : int, default 3
        Number of columns in the grid
    height : int, optional
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Executive summary dashboard with KPI cards

    Examples
    --------
    >>> from ml4t.diagnostic.visualization.backtest import create_executive_summary
    >>> metrics = {
    ...     "sharpe_ratio": 1.85,
    ...     "max_drawdown": 0.12,
    ...     "win_rate": 0.58,
    ...     "profit_factor": 1.75,
    ...     "cagr": 0.22,
    ...     "n_trades": 156,
    ... }
    >>> fig = create_executive_summary(metrics)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Default metrics selection
    if selected_metrics is None:
        selected_metrics = [
            "sharpe_ratio",
            "cagr",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "n_trades",
        ]

    # Filter to available metrics
    available_metrics = [m for m in selected_metrics if m in metrics]

    if not available_metrics:
        # Fallback to any available
        available_metrics = list(metrics.keys())[:6]

    n_metrics = len(available_metrics)
    rows = (n_metrics + cols - 1) // cols

    # Create subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    for idx, metric_name in enumerate(available_metrics):
        row = idx // cols + 1
        col = idx % cols + 1

        value = metrics.get(metric_name, np.nan)

        # Get traffic light color
        color_name = get_traffic_light_color(value, metric_name, thresholds)
        color = TRAFFIC_LIGHT_COLORS.get(color_name, TRAFFIC_LIGHT_COLORS["neutral"])

        # Format label
        label = _get_metric_label(metric_name, thresholds)

        # Compute delta if benchmark available
        delta = None
        if benchmark_metrics and metric_name in benchmark_metrics:
            delta = value - benchmark_metrics[metric_name]

        # Add indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta" if delta is not None else "number",
                value=value,
                number={
                    "font": {"size": 36, "color": color},
                    "valueformat": _get_plotly_format(metric_name, thresholds),
                },
                delta={
                    "reference": value - delta if delta is not None else 0,
                    "relative": False,
                    "valueformat": ".2f",
                    "increasing": {"color": "#28A745"},
                    "decreasing": {"color": "#DC3545"},
                }
                if delta is not None
                else None,
                title={"text": f"<b>{label}</b>", "font": {"size": 14}},
            ),
            row=row,
            col=col,
        )

    # Calculate dimensions
    card_height = 180
    if height is None:
        height = rows * card_height + 100

    if width is None:
        width = cols * 280 + 100

    # Build layout without conflicting with theme_config margin
    layout_updates = {
        "title": {
            "text": f"<b>{title}</b>",
            "font": {"size": 20},
            "x": 0.5,
            "xanchor": "center",
        },
        "height": height,
        "width": width,
        "margin": {"l": 40, "r": 40, "t": 80, "b": 40},
    }

    # Apply theme layout (without overwriting our explicit settings)
    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig


# =============================================================================
# Automated Insights Generation
# =============================================================================


@dataclass
class Insight:
    """A single automated insight from backtest analysis."""

    category: Literal["strength", "weakness", "warning", "info"]
    metric: str
    message: str
    severity: int  # 1-5 scale
    value: float | None = None
    threshold: float | None = None


def create_key_insights(
    metrics: dict[str, float],
    *,
    trades_df: pl.DataFrame | None = None,
    equity_df: pl.DataFrame | None = None,
    max_insights: int = 5,
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> list[Insight]:
    """Generate automated insights from backtest metrics.

    Analyzes metrics and generates human-readable insights about
    strengths, weaknesses, and warnings.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary of metric name to value
    trades_df : pl.DataFrame, optional
        Trade-level data for deeper analysis
    equity_df : pl.DataFrame, optional
        Equity curve data for time-based analysis
    max_insights : int, default 5
        Maximum number of insights to return
    thresholds : dict, optional
        Custom thresholds for evaluation

    Returns
    -------
    list[Insight]
        List of insights sorted by severity

    Examples
    --------
    >>> insights = create_key_insights({"sharpe_ratio": 2.1, "max_drawdown": 0.35})
    >>> for insight in insights:
    ...     print(f"[{insight.category}] {insight.message}")
    [strength] Sharpe ratio of 2.10 is excellent (top 10% of strategies)
    [warning] Maximum drawdown of 35.0% exceeds typical institutional tolerance (20%)
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    insights: list[Insight] = []

    # --- Sharpe Ratio Insights ---
    if "sharpe_ratio" in metrics:
        sharpe = metrics["sharpe_ratio"]
        if sharpe >= 2.0:
            insights.append(
                Insight(
                    category="strength",
                    metric="sharpe_ratio",
                    message=f"Sharpe ratio of {sharpe:.2f} is excellent (top 10% of strategies)",
                    severity=5,
                    value=sharpe,
                    threshold=2.0,
                )
            )
        elif sharpe >= 1.5:
            insights.append(
                Insight(
                    category="strength",
                    metric="sharpe_ratio",
                    message=f"Sharpe ratio of {sharpe:.2f} indicates strong risk-adjusted performance",
                    severity=4,
                    value=sharpe,
                    threshold=1.5,
                )
            )
        elif sharpe < 0.5:
            insights.append(
                Insight(
                    category="weakness",
                    metric="sharpe_ratio",
                    message=f"Sharpe ratio of {sharpe:.2f} suggests poor risk-adjusted returns",
                    severity=4,
                    value=sharpe,
                    threshold=0.5,
                )
            )

    # --- Maximum Drawdown Insights ---
    if "max_drawdown" in metrics:
        dd = metrics["max_drawdown"]
        if dd > 0.30:
            insights.append(
                Insight(
                    category="warning",
                    metric="max_drawdown",
                    message=f"Maximum drawdown of {dd:.1%} exceeds typical institutional tolerance (20%)",
                    severity=5,
                    value=dd,
                    threshold=0.20,
                )
            )
        elif dd > 0.20:
            insights.append(
                Insight(
                    category="warning",
                    metric="max_drawdown",
                    message=f"Maximum drawdown of {dd:.1%} is elevated - consider risk controls",
                    severity=3,
                    value=dd,
                    threshold=0.20,
                )
            )
        elif dd < 0.10:
            insights.append(
                Insight(
                    category="strength",
                    metric="max_drawdown",
                    message=f"Maximum drawdown of {dd:.1%} shows excellent capital preservation",
                    severity=4,
                    value=dd,
                    threshold=0.10,
                )
            )

    # --- Win Rate + Profit Factor Combination ---
    if "win_rate" in metrics and "profit_factor" in metrics:
        wr = metrics["win_rate"]
        pf = metrics["profit_factor"]

        if wr < 0.50 and pf > 1.5:
            insights.append(
                Insight(
                    category="info",
                    metric="win_rate",
                    message=f"Win rate of {wr:.1%} with profit factor {pf:.2f} suggests effective 'let winners run' approach",
                    severity=3,
                    value=wr,
                )
            )
        elif wr > 0.60 and pf < 1.2:
            insights.append(
                Insight(
                    category="warning",
                    metric="profit_factor",
                    message=f"High win rate ({wr:.1%}) but low profit factor ({pf:.2f}) - winners may be too small",
                    severity=3,
                    value=pf,
                )
            )

    # --- Trade Count Insights ---
    if "n_trades" in metrics:
        n = metrics["n_trades"]
        if n < 30:
            insights.append(
                Insight(
                    category="warning",
                    metric="n_trades",
                    message=f"Only {n:.0f} trades - insufficient for statistical significance",
                    severity=4,
                    value=n,
                    threshold=30,
                )
            )
        elif n > 500:
            insights.append(
                Insight(
                    category="strength",
                    metric="n_trades",
                    message=f"{n:.0f} trades provides strong statistical validity",
                    severity=3,
                    value=n,
                    threshold=100,
                )
            )

    # --- CAGR vs Volatility (Risk-adjusted) ---
    if "cagr" in metrics and "volatility" in metrics:
        cagr = metrics["cagr"]
        vol = metrics["volatility"]
        if cagr > 0 and vol > 0:
            return_per_risk = cagr / vol
            if return_per_risk > 1.0:
                insights.append(
                    Insight(
                        category="strength",
                        metric="cagr",
                        message=f"Return/risk ratio of {return_per_risk:.2f} indicates efficient risk utilization",
                        severity=3,
                        value=return_per_risk,
                    )
                )

    # --- Profit Factor Insights ---
    if "profit_factor" in metrics:
        pf = metrics["profit_factor"]
        if pf < 1.0:
            insights.append(
                Insight(
                    category="weakness",
                    metric="profit_factor",
                    message=f"Profit factor of {pf:.2f} indicates net losing strategy",
                    severity=5,
                    value=pf,
                    threshold=1.0,
                )
            )
        elif pf > 2.0:
            insights.append(
                Insight(
                    category="strength",
                    metric="profit_factor",
                    message=f"Profit factor of {pf:.2f} shows strong edge in winner/loser ratio",
                    severity=4,
                    value=pf,
                    threshold=2.0,
                )
            )

    # --- Expectancy Insights ---
    if "expectancy" in metrics:
        exp = metrics["expectancy"]
        if exp < 0:
            insights.append(
                Insight(
                    category="weakness",
                    metric="expectancy",
                    message=f"Negative expectancy (${exp:.2f}) - strategy loses money on average per trade",
                    severity=5,
                    value=exp,
                    threshold=0,
                )
            )
        elif exp > 100:
            insights.append(
                Insight(
                    category="strength",
                    metric="expectancy",
                    message=f"Strong expectancy of ${exp:.2f} per trade provides robust edge",
                    severity=4,
                    value=exp,
                    threshold=50,
                )
            )

    # Sort by severity and limit
    insights.sort(key=lambda x: x.severity, reverse=True)
    return insights[:max_insights]


def format_insights_html(insights: list[Insight]) -> str:
    """Format insights as HTML for embedding in reports.

    Parameters
    ----------
    insights : list[Insight]
        List of insights to format

    Returns
    -------
    str
        HTML string with styled insight cards
    """
    category_icons = {
        "strength": '<span style="color: #28A745; font-size: 18px;">&#10004;</span>',  # Checkmark
        "weakness": '<span style="color: #DC3545; font-size: 18px;">&#10006;</span>',  # X
        "warning": '<span style="color: #FFC107; font-size: 18px;">&#9888;</span>',  # Warning
        "info": '<span style="color: #17A2B8; font-size: 18px;">&#8505;</span>',  # Info
    }

    category_colors = {
        "strength": "#d4edda",
        "weakness": "#f8d7da",
        "warning": "#fff3cd",
        "info": "#d1ecf1",
    }

    html_parts = ['<div style="margin: 20px 0;">']

    for insight in insights:
        icon = category_icons.get(insight.category, "")
        bg_color = category_colors.get(insight.category, "#f8f9fa")

        html_parts.append(f"""
        <div style="background-color: {bg_color}; padding: 12px 16px; margin: 8px 0;
                    border-radius: 6px; display: flex; align-items: center;">
            <span style="margin-right: 12px;">{icon}</span>
            <span style="flex: 1;">{insight.message}</span>
        </div>
        """)

    html_parts.append("</div>")
    return "\n".join(html_parts)
