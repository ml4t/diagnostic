"""Tail risk analysis visualization for backtest tearsheets.

Provides VaR (Value at Risk) and CVaR (Conditional VaR / Expected Shortfall)
analysis with both historical and parametric estimates. Includes fat-tail
detection via kurtosis and skewness metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS as _ML4T_COLORS
from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    pass


def plot_tail_risk_analysis(
    returns: np.ndarray,
    *,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    title: str = "Tail Risk Analysis (VaR / CVaR)",
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Create a tail risk analysis figure with VaR/CVaR visualization.

    Layout: 1x2 subplot -- left: returns histogram with VaR/CVaR lines;
    right: metrics summary table.

    Parameters
    ----------
    returns : np.ndarray
        Array of portfolio returns (e.g. daily returns).
    confidence_levels : tuple[float, ...]
        Confidence levels for VaR/CVaR computation (default: 95%, 99%).
    title : str
        Chart title.
    theme : str, optional
        Theme name (default, dark, print, presentation).
    height : int
        Figure height in pixels.
    width : int, optional
        Figure width in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with histogram and metrics table.

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.normal(0.001, 0.02, 252)
    >>> fig = plot_tail_risk_analysis(returns)
    >>> fig.show()
    """
    from scipy import stats

    theme_config = get_theme_config(theme)

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for tail risk analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(height=height, xaxis={"visible": False}, yaxis={"visible": False})
        return fig

    # Compute statistics
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis

    # Sortino ratio (annualized, assuming daily returns)
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else sigma
    sortino = (mu * np.sqrt(252)) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Compute VaR/CVaR at each level
    var_results: dict[float, dict[str, float]] = {}
    for level in confidence_levels:
        alpha = 1 - level
        # Historical
        hist_var = float(np.percentile(returns, alpha * 100))
        tail = returns[returns <= hist_var]
        hist_cvar = float(np.mean(tail)) if len(tail) > 0 else hist_var
        # Parametric (normal)
        z = stats.norm.ppf(alpha)
        param_var = mu + z * sigma
        param_cvar = mu - sigma * float(stats.norm.pdf(z)) / alpha

        var_results[level] = {
            "hist_var": hist_var,
            "hist_cvar": hist_cvar,
            "param_var": param_var,
            "param_cvar": param_cvar,
        }

    # Build figure: histogram (left) + table (right)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.08,
    )

    # --- Left panel: returns histogram ---
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=60,
            name="Returns",
            marker_color=theme_config["colorway"][0],
            opacity=0.7,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # VaR/CVaR vertical lines
    line_colors = {
        0.95: _ML4T_COLORS["amber"],
        0.99: _ML4T_COLORS["negative"],
    }
    for level in confidence_levels:
        color = line_colors.get(level, _ML4T_COLORS["copper"])
        vr = var_results[level]
        pct = int(level * 100)

        # VaR line (solid)
        fig.add_vline(
            x=vr["hist_var"],
            line_color=color,
            line_width=2,
            line_dash="solid",
            annotation_text=f"VaR {pct}%",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color=color,
            row=1,
            col=1,
        )
        # CVaR line (dashed)
        fig.add_vline(
            x=vr["hist_cvar"],
            line_color=color,
            line_width=2,
            line_dash="dash",
            annotation_text=f"CVaR {pct}%",
            annotation_position="bottom left",
            annotation_font_size=10,
            annotation_font_color=color,
            row=1,
            col=1,
        )

    # --- Right panel: metrics table ---
    # Build table rows
    header_values = ["Metric", "Value"]
    cell_metric: list[str] = []
    cell_value: list[str] = []

    # Distribution stats
    cell_metric.append("Mean Return")
    cell_value.append(f"{mu:.4%}")
    cell_metric.append("Std Dev")
    cell_value.append(f"{sigma:.4%}")
    cell_metric.append("Skewness")
    cell_value.append(f"{skew:.3f}")
    cell_metric.append("Excess Kurtosis")
    cell_value.append(f"{kurt:.3f}")
    cell_metric.append("Sortino Ratio")
    cell_value.append(f"{sortino:.2f}")

    if kurt > 3:
        cell_metric.append("Fat Tails")
        cell_value.append("Yes (kurtosis > 3)")

    # VaR/CVaR rows
    for level in confidence_levels:
        pct = int(level * 100)
        vr = var_results[level]
        cell_metric.append(f"VaR {pct}% (Hist)")
        cell_value.append(f"{vr['hist_var']:.4%}")
        cell_metric.append(f"CVaR {pct}% (Hist)")
        cell_value.append(f"{vr['hist_cvar']:.4%}")
        cell_metric.append(f"VaR {pct}% (Param)")
        cell_value.append(f"{vr['param_var']:.4%}")
        cell_metric.append(f"CVaR {pct}% (Param)")
        cell_value.append(f"{vr['param_cvar']:.4%}")

    # Table colors
    table_header_fill = _ML4T_COLORS["slate"]
    table_header_font = "#FFFFFF"
    table_cell_fill = theme_config["layout"].get("paper_bgcolor", "#FFFFFF")
    table_font_color = theme_config["layout"].get("font", {}).get("color", "#333333")

    fig.add_trace(
        go.Table(
            header={
                "values": [f"<b>{h}</b>" for h in header_values],
                "fill_color": table_header_fill,
                "font": {"color": table_header_font, "size": 12},
                "align": "left",
                "height": 30,
            },
            cells={
                "values": [cell_metric, cell_value],
                "fill_color": table_cell_fill,
                "font": {"color": table_font_color, "size": 11},
                "align": ["left", "right"],
                "height": 25,
            },
        ),
        row=1,
        col=2,
    )

    # Layout
    layout_updates: dict = {
        "title": {"text": title, "font": {"size": 18}},
        "height": height,
        "xaxis": {"title": "Return"},
        "yaxis": {"title": "Frequency"},
        "showlegend": False,
        "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
    }
    if width:
        layout_updates["width"] = width

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig
