"""Turnover and signal stability visualization plots.

This module provides interactive Plotly visualizations for turnover analysis:
- plot_top_bottom_turnover: Turnover rates for extreme quantiles
- plot_autocorrelation: Signal rank autocorrelation by lag
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    format_percentage,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult


def plot_top_bottom_turnover(
    turnover_result: TurnoverAnalysisResult,
    show_all_quantiles: bool = False,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot turnover rates for extreme (top/bottom) quantiles.

    Parameters
    ----------
    turnover_result : TurnoverAnalysisResult
        Turnover analysis result from a signal analysis workflow.
    show_all_quantiles : bool, default False
        Show all quantiles instead of just top and bottom
    theme : str | None
        Plot theme (default, dark, print, presentation)
    width : int | None
        Figure width in pixels
    height : int | None
        Figure height in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> turnover_result = analyzer.compute_turnover_analysis()
    >>> fig = plot_top_bottom_turnover(turnover_result)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get periods and quantiles
    periods = list(turnover_result.quantile_turnover.keys())

    # Create figure
    fig = create_base_figure(
        title="Quantile Turnover Rates",
        xaxis_title="Period",
        yaxis_title="Turnover Rate",
        width=width or theme_config["defaults"]["bar_height"],
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    if show_all_quantiles:
        # Show all quantiles
        first_period = periods[0]
        quantile_labels = list(turnover_result.quantile_turnover[first_period].keys())

        # Use colorway for all quantiles
        n_quantiles = len(quantile_labels)
        colors = (theme_config["colorway"] * ((n_quantiles // len(theme_config["colorway"])) + 1))[
            :n_quantiles
        ]

        for i, q_label in enumerate(quantile_labels):
            turnover_values = [
                turnover_result.quantile_turnover[p].get(q_label, 0) for p in periods
            ]

            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=turnover_values,
                    name=q_label,
                    marker_color=colors[i],
                    hovertemplate=f"{q_label}<br>Period: %{{x}}<br>Turnover: %{{y:.2%}}<extra></extra>",
                )
            )

        fig.update_layout(barmode="group")
    else:
        # Just top and bottom
        top_turnover = [turnover_result.top_quantile_turnover.get(p, 0) for p in periods]
        bottom_turnover = [turnover_result.bottom_quantile_turnover.get(p, 0) for p in periods]
        mean_turnover = [turnover_result.mean_turnover.get(p, 0) for p in periods]

        # Top quantile
        fig.add_trace(
            go.Bar(
                x=periods,
                y=top_turnover,
                name="Top Quantile",
                marker_color=theme_config["colorway"][2],  # Green
                hovertemplate="Top Quantile<br>Period: %{x}<br>Turnover: %{y:.2%}<extra></extra>",
            )
        )

        # Bottom quantile
        fig.add_trace(
            go.Bar(
                x=periods,
                y=bottom_turnover,
                name="Bottom Quantile",
                marker_color=theme_config["colorway"][1],  # Red
                hovertemplate="Bottom Quantile<br>Period: %{x}<br>Turnover: %{y:.2%}<extra></extra>",
            )
        )

        # Mean turnover line
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=mean_turnover,
                mode="lines+markers",
                name="Mean Turnover",
                line={"color": theme_config["colorway"][0], "width": 2, "dash": "dash"},
                marker={"size": 8},
                hovertemplate="Mean<br>Period: %{x}<br>Turnover: %{y:.2%}<extra></extra>",
            )
        )

        fig.update_layout(barmode="group")

    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".0%")

    # Half-life annotation
    half_lives = turnover_result.half_life
    if half_lives:
        hl_text = "<b>Signal Half-Life:</b><br>"
        for period, hl in half_lives.items():
            if hl is not None:
                hl_text += f"{period}: {hl:.1f} periods<br>"
            else:
                hl_text += f"{period}: N/A<br>"

        fig.add_annotation(
            text=hl_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)" if theme != "dark" else "rgba(50,50,50,0.8)",
            bordercolor="gray",
            borderwidth=1,
            align="left",
            xanchor="right",
            yanchor="top",
        )

    return fig


def plot_autocorrelation(
    turnover_result: TurnoverAnalysisResult,
    period: str | None = None,
    max_lags: int | None = None,
    show_significance: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot signal rank autocorrelation by lag.

    Parameters
    ----------
    turnover_result : TurnoverAnalysisResult
        Turnover analysis result from a signal analysis workflow.
    period : str | None
        Period to plot. If None, uses first period.
    max_lags : int | None
        Maximum number of lags to show. If None, shows all available.
    show_significance : bool, default True
        Show significance bands (±1.96/√n for 95% CI)
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> turnover_result = analyzer.compute_turnover_analysis()
    >>> fig = plot_autocorrelation(turnover_result, period="5D")
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get period data
    periods = list(turnover_result.autocorrelation.keys())
    if period is None:
        period = periods[0]
    elif period not in periods:
        raise ValueError(f"Period '{period}' not found. Available: {periods}")

    ac_values = turnover_result.autocorrelation[period]
    lags = turnover_result.autocorrelation_lags

    if max_lags is not None:
        lags = lags[:max_lags]
        ac_values = ac_values[:max_lags]

    # Create figure
    fig = create_base_figure(
        title=f"Signal Rank Autocorrelation ({period})",
        xaxis_title="Lag (Periods)",
        yaxis_title="Autocorrelation",
        width=width or theme_config["defaults"]["bar_height"],
        height=height or theme_config["defaults"]["bar_height"],
        theme=theme,
    )

    # Bar chart for autocorrelation
    colors = [
        theme_config["colorway"][0] if ac >= 0 else theme_config["colorway"][1] for ac in ac_values
    ]

    fig.add_trace(
        go.Bar(
            x=lags,
            y=ac_values,
            marker_color=colors,
            hovertemplate="Lag %{x}<br>AC: %{y:.4f}<extra></extra>",
            name="Autocorrelation",
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

    # Significance bands
    if show_significance:
        # Approximate CI: ±1.96/√n where n is number of observations
        # Using a reasonable default if we don't have exact n
        n_obs = 252  # Approximate trading days
        ci = 1.96 / np.sqrt(n_obs)

        fig.add_hline(
            y=ci,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="95% CI",
            annotation_position="right",
        )
        fig.add_hline(
            y=-ci,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
        )

    # Half-life annotation
    half_life = turnover_result.half_life.get(period)
    mean_ac = turnover_result.mean_autocorrelation.get(period, 0)

    summary_text = (
        f"<b>Signal Persistence:</b><br>"
        f"Mean AC (Lag 1-5): {mean_ac:.4f}<br>"
        f"Half-life: {half_life:.1f} periods"
        if half_life
        else "Half-life: N/A"
    )

    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)" if theme != "dark" else "rgba(50,50,50,0.8)",
        bordercolor="gray",
        borderwidth=1,
        align="left",
        xanchor="right",
        yanchor="top",
    )

    # Decay interpretation
    if len(ac_values) > 0:
        ac0 = ac_values[0]
        decay_rate = "fast" if ac0 < 0.5 else "moderate" if ac0 < 0.8 else "slow"

        fig.add_annotation(
            text=f"Signal decay: {decay_rate}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font={"size": 12},
        )

    return fig


def plot_turnover_heatmap(
    turnover_result: TurnoverAnalysisResult,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot turnover rates as a heatmap (quantile × period).

    Parameters
    ----------
    turnover_result : TurnoverAnalysisResult
        Turnover analysis result from a signal analysis workflow.
    theme : str | None
        Plot theme
    width : int | None
        Figure width
    height : int | None
        Figure height

    Returns
    -------
    go.Figure
        Interactive Plotly heatmap
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Build matrix
    periods = list(turnover_result.quantile_turnover.keys())
    first_period = periods[0]
    quantile_labels = list(turnover_result.quantile_turnover[first_period].keys())

    z_rows = []
    for q_label in quantile_labels:
        row = [turnover_result.quantile_turnover[p].get(q_label, 0) for p in periods]
        z_rows.append(row)

    z_matrix = np.array(z_rows)

    # Create figure
    fig = create_base_figure(
        title="Turnover Rates by Quantile and Period",
        xaxis_title="Period",
        yaxis_title="Quantile",
        width=width or theme_config["defaults"]["heatmap_height"] - 200,
        height=height or 300 + 30 * len(quantile_labels),
        theme=theme,
    )

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=periods,
            y=quantile_labels,
            colorscale="Oranges",
            colorbar={"title": "Turnover", "tickformat": ".0%"},
            hovertemplate="Quantile: %{y}<br>Period: %{x}<br>Turnover: %{z:.2%}<extra></extra>",
        )
    )

    # Add text annotations
    for i, q_label in enumerate(quantile_labels):
        for j, period in enumerate(periods):
            val = z_matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            fig.add_annotation(
                x=period,
                y=q_label,
                text=format_percentage(val),
                showarrow=False,
                font={"size": 10, "color": text_color},
            )

    return fig
