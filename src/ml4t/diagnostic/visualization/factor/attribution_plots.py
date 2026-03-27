"""Return attribution visualizations: waterfall and stacked area charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    # Build waterfall data
    all_labels = list(result.factor_names) + ["Alpha", "Residual", "Total"]
    values = []
    for f in result.factor_names:
        values.append(result.cumulative_factor[f][-1])
    values.append(result.cumulative_alpha[-1])
    values.append(result.cumulative_residual[-1])
    total_val = result.cumulative_total[-1]
    values.append(total_val)

    measures = ["relative"] * (len(result.factor_names) + 2) + ["total"]

    # Clean text labels (value only, CI in hover)
    text = [f"{v:.2%}" for v in values]

    # Per-bar colors from factor palette
    bar_colors = [get_factor_color(f) for f in result.factor_names]
    bar_colors.append(get_factor_color("Alpha"))
    bar_colors.append(get_factor_color("Residual"))
    bar_colors.append(get_factor_color("Total"))

    # Hover with CI where available
    hover_texts = []
    for f in result.factor_names:
        val = result.cumulative_factor[f][-1]
        ci = result.attribution_ci.get(f)
        if ci:
            hover_texts.append(
                f"<b>{f}</b><br>Return: {val:.4f}<br>"
                f"CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
        else:
            hover_texts.append(f"<b>{f}</b><br>Return: {val:.4f}")
    hover_texts.append(f"<b>Alpha</b><br>Return: {result.cumulative_alpha[-1]:.4f}")
    hover_texts.append(f"<b>Residual</b><br>Return: {result.cumulative_residual[-1]:.4f}")
    hover_texts.append(f"<b>Total</b><br>Return: {total_val:.4f}")

    # Plotly Waterfall doesn't support per-bar marker_color directly,
    # so we use individual Bar traces for per-factor coloring
    fig = go.Figure()

    # Build cumulative base for waterfall effect
    base = 0.0
    for i, (label, val) in enumerate(zip(all_labels, values)):
        is_total = measures[i] == "total"
        bar_base = 0.0 if is_total else base
        bar_val = val if is_total else val

        fig.add_trace(go.Bar(
            x=[label],
            y=[abs(bar_val)],
            base=[bar_base if bar_val >= 0 else bar_base + bar_val],
            marker_color=bar_colors[i],
            text=[text[i]],
            textposition="outside",
            hovertext=[hover_texts[i]],
            hoverinfo="text",
            showlegend=False,
        ))

        if not is_total:
            base += val

    # Connector lines between bars
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Return Attribution (Cumulative)",
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".1%",
        height=height,
        width=width or theme_config["defaults"]["width"],
        barmode="overlay",
        bargap=0.3,
    )

    # Footnote
    fig.add_annotation(
        text=(
            "Additive attribution: cumulative sum of daily \u03b2 \u00d7 factor return. "
            "Total may differ from compound strategy return due to compounding effects. "
            "Hover for confidence intervals."
        ),
        xref="paper", yref="paper",
        x=0, y=-0.15,
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
                    f"<b>{f}</b><br>Date: %{{x}}<br>"
                    "Cumulative: %{y:.4f}<extra></extra>"
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
            hovertemplate=(
                "<b>Total</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Cumulative Return Attribution",
        yaxis_title="Cumulative Return",
        height=height,
        width=width or theme_config["defaults"]["width"],
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
        },
    )

    return fig
