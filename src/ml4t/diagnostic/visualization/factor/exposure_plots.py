"""Factor exposure visualizations: beta bars and rolling betas.

All plots follow the library theme system with the safe two-call pattern:
    fig.update_layout(theme_config["layout"])   # Theme defaults first
    fig.update_layout(title=..., legend=...)     # Overrides second
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import get_theme_config

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.factor.results import (
        FactorModelResult,
        RollingExposureResult,
    )


def plot_factor_betas_bar(
    result: FactorModelResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of factor betas with CI error bars.

    Parameters
    ----------
    result : FactorModelResult
        Static factor model result.
    theme : str | None
        Plot theme name.
    height : int
        Figure height in pixels.
    width : int | None
        Figure width in pixels.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    factor_names = result.factor_names
    betas = [result.betas[f] for f in factor_names]
    ci_lower = [result.beta_cis[f][0] for f in factor_names]
    ci_upper = [result.beta_cis[f][1] for f in factor_names]

    # Error bar lengths (asymmetric)
    error_minus = [b - cl for b, cl in zip(betas, ci_lower)]
    error_plus = [cu - b for b, cu in zip(betas, ci_upper)]

    # Color by sign
    colors = [colorway[0] if b >= 0 else colorway[2] for b in betas]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=factor_names,
            x=betas,
            orientation="h",
            marker_color=colors,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": error_plus,
                "arrayminus": error_minus,
                "color": "gray",
                "thickness": 1.5,
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Beta: %{x:.4f}<br>"
                f"CI ({result.confidence_level:.0%}): [%{{customdata[0]:.4f}}, %{{customdata[1]:.4f}}]"
                "<extra></extra>"
            ),
            customdata=list(zip(ci_lower, ci_upper)),
        )
    )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Factor Exposures (Betas)",
        xaxis_title="Beta",
        yaxis_title=None,
        height=height,
        width=width or theme_config["defaults"]["width"],
        yaxis={"autorange": "reversed"},
    )

    return fig


def plot_rolling_betas(
    result: RollingExposureResult,
    *,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
    show_r_squared: bool = True,
) -> go.Figure:
    """Multi-line time series of rolling factor betas.

    Parameters
    ----------
    result : RollingExposureResult
        Rolling exposure result.
    theme : str | None
        Plot theme name.
    height : int
        Figure height.
    width : int | None
        Figure width.
    show_r_squared : bool
        Show rolling R² on secondary y-axis.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    from plotly.subplots import make_subplots

    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    if show_r_squared:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=["Rolling Factor Betas", "Rolling R²"],
        )
    else:
        fig = go.Figure()

    timestamps = result.timestamps

    for i, f in enumerate(result.factor_names):
        color = colorway[i % len(colorway)]
        betas = result.rolling_betas[f]

        trace = go.Scatter(
            x=timestamps,
            y=betas,
            mode="lines",
            name=f,
            line={"color": color, "width": 1.5},
            hovertemplate=f"<b>{f}</b><br>Date: %{{x}}<br>Beta: %{{y:.4f}}<extra></extra>",
        )

        if show_r_squared:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    # Zero line for betas
    if show_r_squared:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=1, col=1)

        # R² subplot
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=result.rolling_r_squared,
                mode="lines",
                name="R²",
                line={"color": "gray", "width": 1},
                showlegend=False,
                hovertemplate="R²: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[0, 1], row=2, col=1)
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

    fig.update_layout(theme_config["layout"])

    title = "Rolling Factor Betas" if not show_r_squared else None
    fig.update_layout(
        title=title,
        height=height,
        width=width or theme_config["defaults"]["width"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig
