"""Risk attribution visualizations: donut chart and MCTR bar chart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.results import RiskAttributionResult


def plot_risk_attribution_pie(
    result: RiskAttributionResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Donut chart of variance decomposition (factor + idiosyncratic).

    Parameters
    ----------
    result : RiskAttributionResult
        Risk attribution result.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly donut chart.
    """
    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    labels = list(result.factor_names) + ["Idiosyncratic"]
    values = [result.factor_contributions[f] for f in result.factor_names]
    values.append(result.idiosyncratic_variance)

    # Use absolute values for pie (can have negative factor contributions)
    abs_values = [abs(v) for v in values]

    colors = [colorway[i % len(colorway)] for i in range(len(result.factor_names))]
    colors.append("gray")

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=abs_values,
            hole=0.4,
            marker={"colors": colors},
            textinfo="label+percent",
            textposition="outside",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Variance: %{customdata[0]:.6f}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            customdata=[[v] for v in values],
        )
    )

    import numpy as np

    total_vol = np.sqrt(result.total_variance) if result.total_variance > 0 else 0.0
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Risk Decomposition",
        height=height,
        width=width or theme_config["defaults"]["width"],
        annotations=[
            {
                "text": f"Vol: {total_vol:.2%}",
                "x": 0.5,
                "y": 0.5,
                "font_size": 14,
                "showarrow": False,
            }
        ],
    )

    return fig


def plot_risk_attribution_bar(
    result: RiskAttributionResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Bar chart of marginal contribution to risk (MCTR) per factor.

    Parameters
    ----------
    result : RiskAttributionResult
        Risk attribution result.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.

    Returns
    -------
    go.Figure
        Plotly bar chart.
    """
    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    factor_names = result.factor_names
    mctr_values = [result.mctr[f] for f in factor_names]
    colors = [colorway[i % len(colorway)] for i in range(len(factor_names))]

    fig = go.Figure(
        go.Bar(
            x=factor_names,
            y=mctr_values,
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>MCTR: %{y:.4f}<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Marginal Contribution to Risk (MCTR)",
        yaxis_title="MCTR",
        height=height,
        width=width or theme_config["defaults"]["width"],
    )

    return fig
