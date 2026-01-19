"""Drawdown visualization functions for portfolio analysis.

Interactive Plotly plots for drawdown analysis including:
- Underwater (drawdown) curve
- Top drawdown periods
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import (
        DrawdownResult,
        PortfolioAnalysis,
    )


def plot_drawdown_underwater(
    analysis: PortfolioAnalysis | None = None,
    drawdown_result: DrawdownResult | None = None,
    theme: str | None = None,
    height: int = 300,
    width: int | None = None,
) -> go.Figure:
    """Plot underwater (drawdown) curve over time.

    Parameters
    ----------
    analysis : PortfolioAnalysis, optional
        Portfolio analysis object (used if drawdown_result not provided)
    drawdown_result : DrawdownResult, optional
        Pre-computed drawdown analysis
    theme : str, optional
        Plot theme
    height : int, default 300
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get drawdown data
    if drawdown_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or drawdown_result")
        drawdown_result = analysis.compute_drawdown_analysis()

    fig = create_base_figure(
        title="Underwater (Drawdown) Curve",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        height=height,
        width=width,
        theme=theme,
    )

    dates = drawdown_result.dates.to_list()
    underwater = drawdown_result.underwater_curve.to_numpy()

    # Fill area below zero
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=underwater,
            mode="lines",
            name="Drawdown",
            line={"color": theme_config["colorway"][1], "width": 1},
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.3)",  # Semi-transparent red
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
        )
    )

    # Mark max drawdown
    max_dd_idx = np.argmin(underwater)
    fig.add_trace(
        go.Scatter(
            x=[dates[max_dd_idx]],
            y=[underwater[max_dd_idx]],
            mode="markers",
            name=f"Max DD: {underwater[max_dd_idx]:.1%}",
            marker={"color": "darkred", "size": 10, "symbol": "circle"},
            hovertemplate="Max Drawdown<br>Date: %{x}<br>DD: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
        hovermode="x unified",
        yaxis={"tickformat": ".0%"},
    )

    return fig


def plot_drawdown_periods(
    analysis: PortfolioAnalysis | None = None,
    drawdown_result: DrawdownResult | None = None,
    top_n: int = 5,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot top N drawdown periods with details.

    Shows drawdown depth, duration, and recovery time for the
    worst drawdown periods.

    Parameters
    ----------
    analysis : PortfolioAnalysis, optional
        Portfolio analysis object (used if drawdown_result not provided)
    drawdown_result : DrawdownResult, optional
        Pre-computed drawdown analysis
    top_n : int, default 5
        Number of top drawdowns to show
    theme : str, optional
        Plot theme
    height : int, default 400
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get drawdown data
    if drawdown_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or drawdown_result")
        drawdown_result = analysis.compute_drawdown_analysis(top_n=top_n)

    top_drawdowns = drawdown_result.top_drawdowns[:top_n]

    if not top_drawdowns:
        # No drawdowns found
        fig = create_base_figure(
            title="Top Drawdown Periods",
            height=height,
            width=width,
            theme=theme,
        )
        fig.add_annotation(
            text="No significant drawdowns found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        return fig

    # Create horizontal bar chart
    depths = [abs(d.depth) for d in top_drawdowns]
    labels = [f"#{i + 1}" for i in range(len(top_drawdowns))]

    fig = go.Figure()

    # Drawdown depth bars
    fig.add_trace(
        go.Bar(
            y=labels,
            x=depths,
            orientation="h",
            name="Drawdown Depth",
            marker_color=theme_config["colorway"][1],
            text=[f"{d:.1%}" for d in depths],
            textposition="outside",
            hovertemplate=(
                "Drawdown: %{x:.2%}<br>"
                "Peak: %{customdata[0]}<br>"
                "Valley: %{customdata[1]}<br>"
                "Recovery: %{customdata[2]}<br>"
                "Duration: %{customdata[3]} days<extra></extra>"
            ),
            customdata=[
                [
                    str(d.peak_date)[:10],
                    str(d.valley_date)[:10],
                    str(d.recovery_date)[:10] if d.recovery_date else "Not recovered",
                    d.duration_days,
                ]
                for d in top_drawdowns
            ],
        )
    )

    fig.update_layout(
        title=f"Top {len(top_drawdowns)} Drawdown Periods",
        xaxis_title="Drawdown Depth",
        yaxis_title="",
        height=height,
        width=width,
        xaxis={"tickformat": ".0%"},
        yaxis={"autorange": "reversed"},  # #1 at top
        showlegend=False,
    )

    return fig


def plot_drawdown_summary(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    height: int = 600,
    width: int | None = None,
) -> go.Figure:
    """Create combined drawdown summary with underwater curve and top periods.

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object
    theme : str, optional
        Plot theme
    height : int, default 600
        Figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Combined subplot figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    drawdown_result = analysis.compute_drawdown_analysis(top_n=5)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=["Underwater Curve", "Top 5 Drawdowns"],
        vertical_spacing=0.12,
    )

    # === Row 1: Underwater curve ===
    dates = drawdown_result.dates.to_list()
    underwater = drawdown_result.underwater_curve.to_numpy()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=underwater,
            mode="lines",
            name="Drawdown",
            line={"color": theme_config["colorway"][1], "width": 1},
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.3)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Mark top drawdowns on the curve
    for i, dd in enumerate(drawdown_result.top_drawdowns[:5]):
        # Find valley date in dates list
        try:
            valley_idx = dates.index(dd.valley_date)
            fig.add_trace(
                go.Scatter(
                    x=[dates[valley_idx]],
                    y=[dd.depth],
                    mode="markers",
                    name=f"DD #{i + 1}",
                    marker={
                        "color": theme_config["colorway"][i % len(theme_config["colorway"])],
                        "size": 10,
                        "symbol": "circle",
                    },
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        except (ValueError, IndexError):
            pass

    # === Row 2: Top drawdowns bar chart ===
    top_drawdowns = drawdown_result.top_drawdowns[:5]
    if top_drawdowns:
        depths = [abs(d.depth) for d in top_drawdowns]
        labels = [f"#{i + 1}: {str(d.valley_date)[:10]}" for i, d in enumerate(top_drawdowns)]

        fig.add_trace(
            go.Bar(
                y=labels,
                x=depths,
                orientation="h",
                marker_color=[
                    theme_config["colorway"][i % len(theme_config["colorway"])]
                    for i in range(len(top_drawdowns))
                ],
                text=[f"{d:.1%}" for d in depths],
                textposition="outside",
                showlegend=False,
                hovertemplate="Drawdown: %{x:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title="Drawdown Analysis",
        height=height,
        width=width,
        **theme_config["layout"],
    )

    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    return fig
