"""Return attribution visualizations: waterfall and stacked area charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

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
    labels = list(result.factor_names) + ["Alpha", "Residual", "Total"]
    values = []
    for f in result.factor_names:
        values.append(result.cumulative_factor[f][-1])
    values.append(result.cumulative_alpha[-1])
    values.append(result.cumulative_residual[-1])
    total_val = result.cumulative_total[-1]

    measures = ["relative"] * (len(result.factor_names) + 2) + ["total"]
    values.append(total_val)

    # Hover text with CI where available
    text = []
    for f in result.factor_names:
        val = result.cumulative_factor[f][-1]
        ci = result.attribution_ci.get(f)
        if ci:
            text.append(f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            text.append(f"{val:.4f}")
    text.append(f"{result.cumulative_alpha[-1]:.4f}")
    text.append(f"{result.cumulative_residual[-1]:.4f}")
    text.append(f"{total_val:.4f}")

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values[:-1] + [values[-1]],
            measure=measures,
            text=text,
            textposition="outside",
            increasing={"marker_color": theme_config["colorway"][0]},
            decreasing={
                "marker_color": theme_config["colorway"][2]
                if len(theme_config["colorway"]) > 2
                else "#ef4444"
            },
            totals={
                "marker_color": theme_config["colorway"][1]
                if len(theme_config["colorway"]) > 1
                else "#D4A84B"
            },
            connector={"line": {"color": "gray", "width": 0.5, "dash": "dot"}},
        )
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Return Attribution (Cumulative)",
        yaxis_title="Cumulative Return",
        height=height,
        width=width or theme_config["defaults"]["width"],
        showlegend=False,
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
    colorway = theme_config["colorway"]

    fig = go.Figure()

    timestamps = result.timestamps

    # Factors
    for i, f in enumerate(result.factor_names):
        color = colorway[i % len(colorway)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=result.cumulative_factor[f],
                mode="lines",
                name=f,
                stackgroup="one",
                line={"width": 0.5, "color": color},
                hovertemplate=f"<b>{f}</b><br>Date: %{{x}}<br>Cumulative: %{{y:.4f}}<extra></extra>",
            )
        )

    # Alpha
    alpha_color = colorway[len(result.factor_names) % len(colorway)]
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.cumulative_alpha,
            mode="lines",
            name="Alpha",
            stackgroup="one",
            line={"width": 0.5, "color": alpha_color},
        )
    )

    # Residual
    resid_color = "gray"
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=result.cumulative_residual,
            mode="lines",
            name="Residual",
            stackgroup="one",
            line={"width": 0.5, "color": resid_color},
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
            hovertemplate="<b>Total</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Cumulative Return Attribution",
        yaxis_title="Cumulative Return",
        height=height,
        width=width or theme_config["defaults"]["width"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig
