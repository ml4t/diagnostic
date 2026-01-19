"""Returns visualization functions for portfolio analysis.

Interactive Plotly plots for return analysis including:
- Cumulative returns
- Rolling returns
- Annual returns bar charts
- Monthly returns heatmap
- Returns distribution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import (
        PortfolioAnalysis,
        RollingMetricsResult,
    )


def plot_cumulative_returns(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    show_benchmark: bool = True,
    log_scale: bool = False,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Plot cumulative returns over time.

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object with returns data
    theme : str, optional
        Plot theme ("default", "dark", "print", "presentation")
    show_benchmark : bool, default True
        Show benchmark returns if available
    log_scale : bool, default False
        Use log scale for y-axis
    height : int, default 500
        Figure height in pixels
    width : int, optional
        Figure width in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    fig = create_base_figure(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=height,
        width=width,
        theme=theme,
    )

    # Compute cumulative returns
    cum_returns = (1 + analysis.returns).cumprod()
    dates = analysis.dates.to_list()

    # Strategy line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_returns,
            mode="lines",
            name="Strategy",
            line={"color": theme_config["colorway"][0], "width": 2},
            hovertemplate="Date: %{x}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # Benchmark line
    if show_benchmark and analysis.has_benchmark and analysis.benchmark is not None:
        bench_cum = (1 + analysis.benchmark).cumprod()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bench_cum,
                mode="lines",
                name="Benchmark",
                line={"color": theme_config["colorway"][1], "width": 2, "dash": "dash"},
                hovertemplate="Date: %{x}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    if log_scale:
        fig.update_yaxes(type="log")

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    return fig


def plot_rolling_returns(
    analysis: PortfolioAnalysis | None = None,
    rolling_result: RollingMetricsResult | None = None,
    windows: list[int] | None = None,
    theme: str | None = None,
    height: int = 500,
    width: int | None = None,
) -> go.Figure:
    """Plot rolling returns for multiple windows.

    Parameters
    ----------
    analysis : PortfolioAnalysis, optional
        Portfolio analysis object (used if rolling_result not provided)
    rolling_result : RollingMetricsResult, optional
        Pre-computed rolling metrics
    windows : list[int], optional
        Rolling windows to plot. Default [21, 63, 252].
    theme : str, optional
        Plot theme
    height : int, default 500
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

    if windows is None:
        windows = [21, 63, 252]

    # Get rolling metrics
    if rolling_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or rolling_result")
        rolling_result = analysis.compute_rolling_metrics(
            windows=windows,
            metrics=["returns"],
        )

    fig = create_base_figure(
        title="Rolling Returns",
        xaxis_title="Date",
        yaxis_title="Rolling Return",
        height=height,
        width=width,
        theme=theme,
    )

    dates = rolling_result.dates.to_list()

    for i, window in enumerate(windows):
        if window in rolling_result.returns:
            returns = rolling_result.returns[window].to_numpy()
            color = theme_config["colorway"][i % len(theme_config["colorway"])]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=returns,
                    mode="lines",
                    name=f"{window}d",
                    line={"color": color, "width": 1.5},
                    hovertemplate=f"{window}d Return: %{{y:.2%}}<extra></extra>",
                )
            )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        hovermode="x unified",
    )

    return fig


def plot_annual_returns_bar(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    show_benchmark: bool = True,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot annual returns as bar chart.

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object
    theme : str, optional
        Plot theme
    show_benchmark : bool, default True
        Show benchmark returns if available
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

    # Compute annual returns
    annual = analysis.compute_annual_returns()
    years = annual["year"].to_list()
    returns = annual["annual_return"].to_numpy()

    fig = create_base_figure(
        title="Annual Returns",
        xaxis_title="Year",
        yaxis_title="Return",
        height=height,
        width=width,
        theme=theme,
    )

    # Color bars based on positive/negative
    colors = [
        theme_config["colorway"][2] if r > 0 else theme_config["colorway"][1] for r in returns
    ]

    fig.add_trace(
        go.Bar(
            x=years,
            y=returns,
            name="Strategy",
            marker_color=colors,
            hovertemplate="Year: %{x}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # Add benchmark if available
    if show_benchmark and analysis.has_benchmark:
        # Compute benchmark annual returns
        import polars as pl

        bench_df = (
            pl.DataFrame(
                {
                    "date": analysis.dates,
                    "return": analysis.benchmark,
                }
            )
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                ]
            )
            .group_by("year")
            .agg((1 + pl.col("return")).product().alias("annual_return") - 1)
            .sort("year")
        )

        fig.add_trace(
            go.Scatter(
                x=bench_df["year"].to_list(),
                y=bench_df["annual_return"].to_numpy(),
                mode="lines+markers",
                name="Benchmark",
                line={"color": theme_config["colorway"][1], "width": 2},
                marker={"size": 8},
                hovertemplate="Year: %{x}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        bargap=0.3,
    )

    return fig


def plot_monthly_returns_heatmap(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot monthly returns as a year x month heatmap.

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object
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

    # Get monthly returns matrix
    matrix = analysis.get_monthly_returns_matrix()

    years = matrix["year"].to_list()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Extract values (columns 1-12 are the months)
    z_values = []
    for row in matrix.iter_rows():
        z_values.append([row[i] if row[i] is not None else np.nan for i in range(1, 13)])

    z_array = np.array(z_values)

    # Color scale: red for negative, green for positive
    colorscale = [
        [0.0, "#d73027"],
        [0.25, "#fc8d59"],
        [0.5, "#ffffff"],
        [0.75, "#91cf60"],
        [1.0, "#1a9850"],
    ]

    # Find symmetric range for color scale
    max_abs = np.nanmax(np.abs(z_array))
    if np.isnan(max_abs):
        max_abs = 0.1

    fig = go.Figure(
        data=go.Heatmap(
            z=z_array,
            x=months,
            y=years,
            colorscale=colorscale,
            zmin=-max_abs,
            zmax=max_abs,
            text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in z_array],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
            colorbar={
                "title": "Return",
                "tickformat": ".0%",
            },
        )
    )

    fig.update_layout(
        title="Monthly Returns",
        xaxis_title="Month",
        yaxis_title="Year",
        height=height,
        width=width,
        yaxis={"autorange": "reversed"},  # Most recent year at top
    )

    return fig


def plot_returns_distribution(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    bins: int = 50,
    show_normal: bool = True,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot returns distribution histogram with optional normal fit.

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object
    theme : str, optional
        Plot theme
    bins : int, default 50
        Number of histogram bins
    show_normal : bool, default True
        Overlay normal distribution fit
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

    returns = analysis.returns
    clean_returns = returns[~np.isnan(returns)]

    fig = create_base_figure(
        title="Returns Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=height,
        width=width,
        theme=theme,
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=clean_returns,
            nbinsx=bins,
            name="Returns",
            marker_color=theme_config["colorway"][0],
            opacity=0.7,
            histnorm="probability density",
            hovertemplate="Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # Normal fit
    if show_normal:
        from scipy import stats as sp_stats

        mean = np.mean(clean_returns)
        std = np.std(clean_returns, ddof=1)

        x_range = np.linspace(min(clean_returns), max(clean_returns), 100)
        y_normal = sp_stats.norm.pdf(x_range, mean, std)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode="lines",
                name=f"Normal (μ={mean:.4f}, σ={std:.4f})",
                line={"color": theme_config["colorway"][1], "width": 2, "dash": "dash"},
            )
        )

    # Add VaR lines
    var_95 = np.percentile(clean_returns, 5)
    var_99 = np.percentile(clean_returns, 1)

    fig.add_vline(
        x=var_95,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.2%}",
        annotation_position="top",
    )

    fig.add_vline(
        x=var_99,
        line_dash="dot",
        line_color="red",
        annotation_text=f"VaR 99%: {var_99:.2%}",
        annotation_position="bottom",
    )

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        bargap=0.05,
    )

    return fig
