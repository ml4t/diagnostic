"""Return attribution visualizations: waterfall and stacked area charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization._colors import get_factor_color
from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.results import AttributionResult


def plot_return_attribution_waterfall(
    result: AttributionResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Waterfall chart of cumulative return attribution.

    Shows how each factor, alpha, and residual contribute to total return.
    Uses per-factor colors from the canonical factor palette.

    Parameters
    ----------
    result : AttributionResult
        Attribution result with cumulative returns.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly waterfall figure.
    """
    theme_config = get_theme_config(theme)

    # Use additive (sum of daily) contributions — these are exactly additive:
    # sum(factor_contributions[f]) + sum(alpha) + sum(residual) == sum(total_returns)
    all_labels = list(result.factor_names) + ["Alpha", "Residual", "Total"]
    values: list[float] = []
    for f in result.factor_names:
        values.append(float(np.sum(result.factor_contributions[f])))
    alpha_sum = float(np.sum(result.alpha_contribution))
    residual_sum = float(np.sum(result.residual))
    values.append(alpha_sum)
    values.append(residual_sum)
    total_val = sum(values)  # exact sum of additive components
    values.append(total_val)

    measures = ["relative"] * (len(result.factor_names) + 2) + ["total"]

    # Text labels
    text = [f"{v:.2%}" for v in values]

    # Per-bar colors
    bar_colors = [get_factor_color(f) for f in result.factor_names]
    bar_colors.append(get_factor_color("Alpha"))
    bar_colors.append(get_factor_color("Residual"))
    bar_colors.append(get_factor_color("Total"))

    # Hover with CI where available
    hover_texts = []
    for f in result.factor_names:
        val = float(np.sum(result.factor_contributions[f]))
        ci = result.attribution_ci.get(f)
        if ci:
            hover_texts.append(
                f"<b>{f}</b><br>Contribution: {val:.4f}<br>CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
        else:
            hover_texts.append(f"<b>{f}</b><br>Contribution: {val:.4f}")
    hover_texts.append(f"<b>Alpha</b><br>Contribution: {alpha_sum:.4f}")
    hover_texts.append(f"<b>Residual</b><br>Contribution: {residual_sum:.4f}")
    hover_texts.append(f"<b>Total</b><br>Sum of contributions: {total_val:.4f}")

    # Build waterfall with per-bar coloring via individual Bar traces
    fig = go.Figure()

    base = 0.0
    for i, (label, val) in enumerate(zip(all_labels, values)):
        is_total = measures[i] == "total"
        bar_base = 0.0 if is_total else base

        fig.add_trace(
            go.Bar(
                x=[label],
                y=[abs(val)],
                base=[bar_base if val >= 0 else bar_base + val],
                marker_color=bar_colors[i],
                text=[text[i]],
                textposition="outside",
                hovertext=[hover_texts[i]],
                hoverinfo="text",
                showlegend=False,
            )
        )

        if not is_total:
            base += val

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Return Attribution (Additive)",
        yaxis_title="Contribution (sum of daily \u03b2 \u00d7 factor return)",
        yaxis_tickformat=".2%",
        height=height,
        width=width or theme_config["defaults"]["width"],
        barmode="overlay",
        bargap=0.3,
    )

    # Footnote
    fig.add_annotation(
        text=(
            "Additive decomposition: \u03a3 \u03b2[t\u22121] \u00d7 f[t]. "
            "Components sum exactly to Total. Hover for confidence intervals."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.15,
        showarrow=False,
        font={"size": 10, "color": "gray"},
        align="left",
    )

    return fig


def plot_return_attribution_area(
    result: AttributionResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Stacked area chart of cumulative factor contributions over time.

    Parameters
    ----------
    result : AttributionResult
        Attribution result with cumulative contributions.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly stacked area figure.
    """
    theme_config = get_theme_config(theme)

    fig = go.Figure()

    timestamps = result.timestamps

    # Factors
    for _i, f in enumerate(result.factor_names):
        color = get_factor_color(f)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=result.cumulative_factor[f],
                mode="lines",
                name=f,
                stackgroup="one",
                line={"width": 0.5, "color": color},
                hovertemplate=(
                    f"<b>{f}</b><br>Date: %{{x}}<br>Cumulative: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    # Alpha
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.cumulative_alpha,
            mode="lines",
            name="Alpha",
            stackgroup="one",
            line={"width": 0.5, "color": get_factor_color("Alpha")},
        )
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.cumulative_residual,
            mode="lines",
            name="Residual",
            stackgroup="one",
            line={"width": 0.5, "color": get_factor_color("Residual")},
        )
    )

    # Total return line (not stacked)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.cumulative_total,
            mode="lines",
            name="Total",
            line={"color": "black", "width": 2, "dash": "dash"},
            hovertemplate=("<b>Total</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>"),
        )
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Cumulative Return Attribution",
        yaxis_title="Cumulative Return",
        height=height,
        width=width or theme_config["defaults"]["width"],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig
