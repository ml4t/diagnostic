"""Executive summary visualizations for backtest analysis."""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS as _ML4T_COLORS
from ml4t.diagnostic.visualization.core import (
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    import polars as pl

    from ml4t.diagnostic.integration.backtest_profile import BacktestProfile


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
        "format": "${:,.2f}",
        "label": "Expectancy",
        "higher_is_better": True,
    },
    "avg_trade": {
        "red": (-float("inf"), 0.0),
        "yellow": (0.0, 25.0),
        "green": (25.0, float("inf")),
        "format": "${:,.2f}",
        "label": "Avg Trade",
        "higher_is_better": True,
    },
    "n_trades": {
        "red": (-float("inf"), 30),
        "yellow": (30, 100),
        "green": (100, float("inf")),
        "format": "{:,.0f}",
        "label": "Trade Count",
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
    "avg_turnover": {
        "red": (0.75, float("inf")),
        "yellow": (0.25, 0.75),
        "green": (-float("inf"), 0.25),
        "format": "{:.2f}",
        "label": "Avg Turnover",
        "higher_is_better": False,
    },
    "num_rebalance_events": {
        "red": (250, float("inf")),
        "yellow": (50, 250),
        "green": (-float("inf"), 50),
        "format": "{:,.0f}",
        "label": "Rebalances",
        "higher_is_better": False,
    },
    "avg_open_positions": {
        "red": (-float("inf"), 3),
        "yellow": (3, 10),
        "green": (10, float("inf")),
        "format": "{:.1f}",
        "label": "Avg Open Positions",
        "higher_is_better": True,
    },
    "time_in_market": {
        "red": (-float("inf"), 0.2),
        "yellow": (0.2, 0.6),
        "green": (0.6, float("inf")),
        "format": "{:.1%}",
        "label": "Time In Market",
        "higher_is_better": True,
    },
    "total_implementation_cost": {
        "red": (10000.0, float("inf")),
        "yellow": (2500.0, 10000.0),
        "green": (-float("inf"), 2500.0),
        "format": "${:,.0f}",
        "label": "Implementation Cost",
        "higher_is_better": False,
    },
    "dsr_probability": {
        "red": (-float("inf"), 0.8),
        "yellow": (0.8, 0.95),
        "green": (0.95, float("inf")),
        "format": "{:.1%}",
        "label": "Deflated Sharpe Ratio",
        "higher_is_better": True,
    },
    "min_trl": {
        "red": (252.0, float("inf")),
        "yellow": (126.0, 252.0),
        "green": (-float("inf"), 126.0),
        "format": "{:,.0f}",
        "label": "MinTRL",
        "higher_is_better": False,
    },
    "omega_ratio": {
        "format": "{:.2f}",
        "label": "Omega Ratio",
        "higher_is_better": True,
    },
    "var_95": {
        "format": "{:.2%}",
        "label": "VaR (95%)",
        "higher_is_better": False,
    },
    "cvar_95": {
        "format": "{:.2%}",
        "label": "CVaR (95%)",
        "higher_is_better": False,
    },
    "skewness": {
        "format": "{:.2f}",
        "label": "Skewness",
    },
    "kurtosis": {
        "format": "{:.1f}",
        "label": "Kurtosis",
    },
    "tail_ratio": {
        "format": "{:.2f}",
        "label": "Tail Ratio",
        "higher_is_better": True,
    },
    "stability": {
        "format": "{:.2f}",
        "label": "Stability",
        "higher_is_better": True,
    },
    "avg_win_loss_ratio": {
        "format": "{:.2f}",
        "label": "Avg Win/Loss",
        "higher_is_better": True,
    },
    "n_symbols": {
        "format": "{:,.0f}",
        "label": "Symbols Traded",
    },
    "avg_bars_held": {
        "format": "{:.0f}",
        "label": "Avg Holding Period",
    },
    "best_trade": {
        "format": "${:,.0f}",
        "label": "Best Trade",
    },
    "worst_trade": {
        "format": "${:,.0f}",
        "label": "Worst Trade",
    },
    "total_commission": {
        "format": "${:,.0f}",
        "label": "Total Commission",
    },
    "total_slippage": {
        "format": "${:,.0f}",
        "label": "Total Slippage",
    },
}

TRAFFIC_LIGHT_COLORS = {
    "green": _ML4T_COLORS["positive"],
    "yellow": _ML4T_COLORS["amber"],
    "red": _ML4T_COLORS["negative"],
    "neutral": _ML4T_COLORS["neutral"],
}


@dataclass(frozen=True)
class MetricTableSpec:
    """Presentation metadata for a tearsheet metric row."""

    benchmark_comparable: bool = False


SUMMARY_TABLE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Performance",
        (
            "total_return",
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "omega_ratio",
            "max_drawdown",
            "volatility",
        ),
    ),
    (
        "Risk",
        (
            "var_95",
            "cvar_95",
            "skewness",
            "kurtosis",
            "tail_ratio",
            "stability",
        ),
    ),
    (
        "Trading",
        (
            "win_rate",
            "profit_factor",
            "avg_trade",
            "avg_win_loss_ratio",
            "n_trades",
            "n_symbols",
            "avg_bars_held",
            "best_trade",
            "worst_trade",
        ),
    ),
    (
        "Costs",
        (
            "total_implementation_cost",
            "total_commission",
            "total_slippage",
            "avg_turnover",
        ),
    ),
    (
        "Statistical",
        ("dsr_probability", "min_trl"),
    ),
)

METRIC_TABLE_SPECS: dict[str, MetricTableSpec] = {
    "total_return": MetricTableSpec(benchmark_comparable=True),
    "cagr": MetricTableSpec(benchmark_comparable=True),
    "sharpe_ratio": MetricTableSpec(benchmark_comparable=True),
    "sortino_ratio": MetricTableSpec(benchmark_comparable=True),
    "calmar_ratio": MetricTableSpec(benchmark_comparable=True),
    "max_drawdown": MetricTableSpec(benchmark_comparable=True),
    "volatility": MetricTableSpec(benchmark_comparable=True),
    "win_rate": MetricTableSpec(),
    "profit_factor": MetricTableSpec(),
    "avg_trade": MetricTableSpec(),
    "expectancy": MetricTableSpec(),
    "n_trades": MetricTableSpec(),
    "avg_turnover": MetricTableSpec(),
    "num_rebalance_events": MetricTableSpec(),
    "avg_open_positions": MetricTableSpec(),
    "time_in_market": MetricTableSpec(),
    "total_implementation_cost": MetricTableSpec(),
    "dsr_probability": MetricTableSpec(),
    "min_trl": MetricTableSpec(),
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

    numeric_value = _coerce_numeric_value(value)
    if numeric_value is None or np.isnan(numeric_value):
        return "neutral"

    # Check which range the value falls into
    for color in ["green", "yellow", "red"]:
        low, high = config[color]
        if low <= numeric_value < high:
            return color

    return "neutral"


def _format_metric_value(
    value: Any,
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

    numeric_value = _coerce_numeric_value(value)
    if numeric_value is None:
        return html_mod.escape(str(value))
    if np.isnan(numeric_value):
        return "N/A"
    if np.isposinf(numeric_value):
        return "∞"
    if np.isneginf(numeric_value):
        return "-∞"

    if metric_name in thresholds:
        fmt = thresholds[metric_name].get("format", "{:.2f}")
        return fmt.format(numeric_value)

    return f"{numeric_value:.2f}"


def _coerce_numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_numeric_value(value: Any) -> bool:
    numeric_value = _coerce_numeric_value(value)
    return numeric_value is not None and not np.isnan(numeric_value)


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

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        height=200,
        width=250,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
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
        return ",.1%"
    elif "$" in py_fmt:
        if ",.0f" in py_fmt:
            return "$,.0f"
        if ",.2f" in py_fmt:
            return "$,.2f"
        return "$.2f"
    elif ",.0f" in py_fmt:
        return ",.0f"
    elif ",.1f" in py_fmt:
        return ",.1f"
    elif ",.2f" in py_fmt:
        return ",.2f"
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
        higher_is_better = thresholds.get(metric_name, {}).get("higher_is_better", True)
        increasing_color = (
            _ML4T_COLORS["positive"] if higher_is_better else _ML4T_COLORS["negative"]
        )
        decreasing_color = (
            _ML4T_COLORS["negative"] if higher_is_better else _ML4T_COLORS["positive"]
        )

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
                    "valueformat": _get_plotly_format(metric_name, thresholds),
                    "increasing": {"color": increasing_color},
                    "decreasing": {"color": decreasing_color},
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


def create_executive_summary_html(
    metrics: dict[str, float],
    *,
    selected_metrics: list[str] | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    benchmark_label: str = "Benchmark",
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create a dense HTML executive strip for first-page tearsheet use."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if selected_metrics is None:
        selected_metrics = [
            "total_return",
            "cagr",
            "sharpe_ratio",
            "max_drawdown",
            "volatility",
            "dsr_probability",
        ]

    # Resolve common aliases so preset names like "sharpe_ratio" find "sharpe"
    metric_aliases: dict[str, str] = {
        "sharpe_ratio": "sharpe",
        "sharpe": "sharpe_ratio",
    }

    def _resolve_metric(name: str) -> str | None:
        if name in metrics:
            return name
        alias = metric_aliases.get(name)
        if alias and alias in metrics:
            return alias
        return None

    metric_order = [
        _resolve_metric(m) for m in selected_metrics if _resolve_metric(m) is not None
    ]
    if not metric_order:
        metric_order = list(metrics.keys())[:6]

    cards: list[str] = []
    for metric_name in metric_order[:6]:
        label = _get_metric_label(metric_name, thresholds)
        value_text = _format_metric_value(metrics[metric_name], metric_name, thresholds)
        footer = ""
        benchmark_value = benchmark_metrics.get(metric_name) if benchmark_metrics else None
        spec = METRIC_TABLE_SPECS.get(metric_name)
        if (
            spec is not None
            and spec.benchmark_comparable
            and _has_numeric_value(benchmark_value)
        ):
            benchmark_text = _format_metric_value(benchmark_value, metric_name, thresholds)
            spread_text = _format_metric_spread_text(
                metric_name, metrics[metric_name], benchmark_value, thresholds
            )
            footer = (
                f'<div class="executive-strip-meta">'
                f"<span>{html_mod.escape(benchmark_label)} {html_mod.escape(benchmark_text)}</span>"
                f"<span>{html_mod.escape(spread_text)}</span>"
                f"</div>"
            )

        cards.append(
            f"""
            <div class="executive-kpi">
                <div class="executive-kpi-label">{html_mod.escape(label)}</div>
                <div class="executive-kpi-value">{html_mod.escape(value_text)}</div>
                {footer}
            </div>
            """
        )

    return f"""
    <div class="executive-strip">
        {"".join(cards)}
    </div>
    """


def create_key_metrics_table_html(
    metrics: dict[str, float],
    *,
    selected_metrics: list[str] | None = None,
    benchmark_metrics: dict[str, float] | None = None,
    benchmark_label: str = "Benchmark",
    thresholds: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create a compact HTML table of key metrics for tearsheet display."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    has_benchmark = bool(benchmark_metrics)
    column_count = 4

    rendered_rows: list[str] = []
    used_metrics: set[str] = set()

    for group_name, metric_names in SUMMARY_TABLE_GROUPS:
        group_metric_names = [metric_name for metric_name in metric_names if metric_name in metrics]
        rows = _render_metric_group_rows(
            group_metric_names,
            metrics,
            thresholds,
            has_benchmark=has_benchmark,
            benchmark_metrics=benchmark_metrics,
            benchmark_label=benchmark_label,
        )
        used_metrics.update(group_metric_names)

        if rows:
            rendered_rows.append(
                f"""
                <tr class="metrics-table-group">
                    <th colspan="{column_count}">
                        <div class="metrics-group-heading">{html_mod.escape(group_name)}</div>
                    </th>
                </tr>
                {"".join(rows)}
                """
            )

    fallback_metrics = selected_metrics or list(metrics.keys())
    remaining = [
        metric for metric in fallback_metrics if metric in metrics and metric not in used_metrics
    ]
    if remaining:
        fallback_rows = _render_metric_group_rows(
            remaining[:8],
            metrics,
            thresholds,
            has_benchmark=has_benchmark,
            benchmark_metrics=benchmark_metrics,
            benchmark_label=benchmark_label,
        )
        rendered_rows.append(
            f"""
            <tr class="metrics-table-group">
                <th colspan="{column_count}">
                    <div class="metrics-group-heading">Additional Metrics</div>
                </th>
            </tr>
            {"".join(fallback_rows)}
            """
        )

    if not rendered_rows:
        return '<p class="metrics-table-empty">No summary metrics available.</p>'

    benchmark_context = (
        f"""
        <div class="metrics-table-intro">
            <div class="metrics-table-intro-chip">
                Benchmark: {html_mod.escape(benchmark_label)} | Spread = Strategy - {html_mod.escape(benchmark_label)}
            </div>
        </div>
        """
        if has_benchmark
        else ""
    )

    header_html = (
        f"""
        <thead>
            <tr>
                <th>Metric</th>
                <th>Strategy</th>
                <th>{html_mod.escape(benchmark_label)}</th>
                <th>Spread</th>
            </tr>
        </thead>
        """
        if has_benchmark
        else """
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        """
    )

    return f"""
    <div class="metrics-table-wrap">
        {benchmark_context}
        <table class="metrics-table">
            {header_html}
            <tbody>
                {"".join(rendered_rows)}
            </tbody>
        </table>
    </div>
    """


def _render_metric_group_rows(
    metric_names: list[str],
    metrics: dict[str, float],
    thresholds: dict[str, dict[str, Any]],
    *,
    has_benchmark: bool,
    benchmark_metrics: dict[str, float] | None,
    benchmark_label: str,
) -> list[str]:
    if has_benchmark:
        return [
            _render_metric_table_row(
                metric_name,
                metrics[metric_name],
                thresholds,
                has_benchmark=True,
                benchmark_value=benchmark_metrics.get(metric_name) if benchmark_metrics else None,
                benchmark_label=benchmark_label,
            )
            for metric_name in metric_names
        ]

    rows: list[str] = []
    for index in range(0, len(metric_names), 2):
        left_name = metric_names[index]
        right_name = metric_names[index + 1] if index + 1 < len(metric_names) else None
        rows.append(
            _render_metric_pair_row(
                left_name,
                metrics[left_name],
                thresholds,
                right_name=right_name,
                right_value=metrics[right_name] if right_name is not None else None,
            )
        )
    return rows


def _render_metric_table_row(
    metric_name: str,
    value: float,
    thresholds: dict[str, dict[str, Any]],
    *,
    has_benchmark: bool,
    benchmark_value: float | None,
    benchmark_label: str,
) -> str:
    spec = METRIC_TABLE_SPECS.get(metric_name, MetricTableSpec())
    label = _get_metric_label(metric_name, thresholds)
    value_text = _format_metric_value(value, metric_name, thresholds)
    if not has_benchmark:
        return f"""
        <tr class="metrics-table-row">
            <td data-label="Metric"><div class="metric-label">{html_mod.escape(label)}</div></td>
            <td data-label="Value">
                <div class="metric-primary">{html_mod.escape(value_text)}</div>
            </td>
        </tr>
        """

    benchmark_text = "—"
    spread_html = '<div class="metric-spread metric-spread--na">—</div>'
    if spec.benchmark_comparable and _has_numeric_value(benchmark_value):
        benchmark_text = _format_metric_value(benchmark_value, metric_name, thresholds)
        spread_html = _format_metric_spread(
            metric_name,
            value,
            benchmark_value,
            thresholds,
        )

    return f"""
    <tr class="metrics-table-row">
        <td data-label="Metric"><div class="metric-label">{html_mod.escape(label)}</div></td>
        <td data-label="Strategy">
            <div class="metric-primary">{html_mod.escape(value_text)}</div>
        </td>
        <td data-label="{html_mod.escape(benchmark_label)}">
            <div class="metric-benchmark">{html_mod.escape(benchmark_text)}</div>
        </td>
        <td data-label="Spread">{spread_html}</td>
    </tr>
    """


def _render_metric_pair_row(
    metric_name: str,
    value: float,
    thresholds: dict[str, dict[str, Any]],
    *,
    right_name: str | None,
    right_value: float | None,
) -> str:
    left_label = _get_metric_label(metric_name, thresholds)
    left_value = _format_metric_value(value, metric_name, thresholds)

    right_label_html = ""
    right_value_html = ""
    if right_name is not None and right_value is not None:
        right_label = _get_metric_label(right_name, thresholds)
        right_text = _format_metric_value(right_value, right_name, thresholds)
        right_label_html = f'<div class="metric-label">{html_mod.escape(right_label)}</div>'
        right_value_html = f'<div class="metric-primary">{html_mod.escape(right_text)}</div>'

    return f"""
    <tr class="metrics-table-row">
        <td data-label="Metric"><div class="metric-label">{html_mod.escape(left_label)}</div></td>
        <td data-label="Value">
            <div class="metric-primary">{html_mod.escape(left_value)}</div>
        </td>
        <td data-label="Metric">{right_label_html}</td>
        <td data-label="Value">{right_value_html}</td>
    </tr>
    """


def _format_metric_spread(
    metric_name: str,
    strategy_value: float,
    benchmark_value: float,
    thresholds: dict[str, dict[str, Any]],
) -> str:
    delta = strategy_value - benchmark_value
    prefix = "+" if delta > 0 else ""
    delta_text = f"{prefix}{_format_metric_value(delta, metric_name, thresholds)}"
    return (
        '<div class="metric-spread">'
        f'<span class="metric-spread-value">{html_mod.escape(delta_text)}</span>'
        "</div>"
    )


def _format_metric_spread_text(
    metric_name: str,
    strategy_value: float,
    benchmark_value: float,
    thresholds: dict[str, dict[str, Any]],
) -> str:
    delta = strategy_value - benchmark_value
    prefix = "+" if delta > 0 else ""
    delta_text = f"{prefix}{_format_metric_value(delta, metric_name, thresholds)}"
    return f"Spread {delta_text}"


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
    profile: BacktestProfile | None = None,
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
    profile : BacktestProfile, optional
        Backtest profile with availability, burden, concentration, and other
        profile-native analytics
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

    # --- Profile-native burden / concentration / availability insights ---
    if profile is not None:
        cost_share = profile.attribution["metrics"].get("top_5_cost_share")
        if cost_share is not None and cost_share > 0.75:
            insights.append(
                Insight(
                    category="warning",
                    metric="cost_burden",
                    message=(
                        f"Top symbols account for {cost_share:.1%} of implementation cost - "
                        "burden is highly concentrated"
                    ),
                    severity=4,
                    value=cost_share,
                    threshold=0.75,
                )
            )

        pnl_share = profile.attribution["metrics"].get("top_5_pnl_share")
        if pnl_share is not None and pnl_share > 0.80:
            insights.append(
                Insight(
                    category="warning",
                    metric="pnl_concentration",
                    message=(
                        f"Top contributors explain {pnl_share:.1%} of net PnL - "
                        "returns may be fragile to symbol concentration"
                    ),
                    severity=4,
                    value=pnl_share,
                    threshold=0.80,
                )
            )

        avg_turnover = profile.activity["metrics"].get("avg_turnover")
        if avg_turnover is not None and avg_turnover > 0.50:
            insights.append(
                Insight(
                    category="warning",
                    metric="avg_turnover",
                    message=(
                        f"Average turnover of {avg_turnover:.2f} per period suggests substantial "
                        "implementation burden"
                    ),
                    severity=3,
                    value=avg_turnover,
                    threshold=0.50,
                )
            )

        execution_availability = profile.availability.families["execution"]
        if execution_availability.status.value == "partial":
            coverage = execution_availability.coverage or 0.0
            insights.append(
                Insight(
                    category="warning",
                    metric="execution_availability",
                    message=(
                        f"Quote-aware execution audit is only partially available "
                        f"({coverage:.1%} coverage)"
                    ),
                    severity=4,
                    value=coverage,
                    threshold=profile.quote_coverage_threshold,
                )
            )
        elif execution_availability.status.value == "unavailable":
            insights.append(
                Insight(
                    category="info",
                    metric="execution_availability",
                    message="Quote-aware execution audit is unavailable for this backtest",
                    severity=1,
                )
            )

    # Sort by severity and limit
    insights.sort(key=lambda x: x.severity, reverse=True)
    selected = insights[:max_insights]

    if profile is not None:
        essential_metrics = ["cost_burden", "execution_availability"]
        for metric_name in essential_metrics:
            essential_insight = next(
                (insight for insight in insights if insight.metric == metric_name),
                None,
            )
            if essential_insight is not None and all(
                insight.metric != metric_name for insight in selected
            ):
                selected.append(essential_insight)

    return selected


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
        "strength": f'<span style="color: {_ML4T_COLORS["positive"]}; font-size: 18px;">&#10004;</span>',
        "weakness": f'<span style="color: {_ML4T_COLORS["negative"]}; font-size: 18px;">&#10006;</span>',
        "warning": f'<span style="color: {_ML4T_COLORS["amber"]}; font-size: 18px;">&#9888;</span>',
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
        bg_color = category_colors.get(insight.category, _ML4T_COLORS["bg_light"])

        html_parts.append(f"""
        <div style="background-color: {bg_color}; padding: 12px 16px; margin: 8px 0;
                    border-radius: 6px; display: flex; align-items: center;">
            <span style="margin-right: 12px;">{icon}</span>
            <span style="flex: 1;">{insight.message}</span>
        </div>
        """)

    html_parts.append("</div>")
    return "\n".join(html_parts)
