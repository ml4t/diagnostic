"""Risk visualization functions for portfolio analysis.

Interactive Plotly plots for risk metrics including:
- Rolling volatility
- Rolling Sharpe ratio
- Rolling beta
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def plot_rolling_volatility(
    analysis: PortfolioAnalysis | None = None,
    rolling_result: RollingMetricsResult | None = None,
    windows: list[int] | None = None,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot rolling annualized volatility.

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

    if windows is None:
        windows = [21, 63, 252]

    # Get rolling metrics
    if rolling_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or rolling_result")
        rolling_result = analysis.compute_rolling_metrics(
            windows=windows,
            metrics=["volatility"],
        )

    fig = create_base_figure(
        title="Rolling Volatility (Annualized)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        height=height,
        width=width,
        theme=theme,
    )

    dates = rolling_result.dates.to_list()

    for i, window in enumerate(windows):
        if window in rolling_result.volatility:
            vol = rolling_result.volatility[window].to_numpy()
            color = theme_config["colorway"][i % len(theme_config["colorway"])]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=vol,
                    mode="lines",
                    name=f"{window}d",
                    line={"color": color, "width": 1.5},
                    hovertemplate=f"{window}d Vol: %{{y:.2%}}<extra></extra>",
                )
            )

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        hovermode="x unified",
        yaxis={"tickformat": ".0%"},
    )

    return fig


def plot_rolling_sharpe(
    analysis: PortfolioAnalysis | None = None,
    rolling_result: RollingMetricsResult | None = None,
    windows: list[int] | None = None,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot rolling Sharpe ratio.

    Parameters
    ----------
    analysis : PortfolioAnalysis, optional
        Portfolio analysis object (used if rolling_result not provided)
    rolling_result : RollingMetricsResult, optional
        Pre-computed rolling metrics
    windows : list[int], optional
        Rolling windows to plot. Default [63, 126, 252].
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

    if windows is None:
        windows = [63, 126, 252]

    # Get rolling metrics
    if rolling_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or rolling_result")
        rolling_result = analysis.compute_rolling_metrics(
            windows=windows,
            metrics=["sharpe"],
        )

    fig = create_base_figure(
        title="Rolling Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=height,
        width=width,
        theme=theme,
    )

    dates = rolling_result.dates.to_list()

    for i, window in enumerate(windows):
        if window in rolling_result.sharpe:
            sharpe = rolling_result.sharpe[window].to_numpy()
            color = theme_config["colorway"][i % len(theme_config["colorway"])]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=sharpe,
                    mode="lines",
                    name=f"{window}d",
                    line={"color": color, "width": 1.5},
                    hovertemplate=f"{window}d Sharpe: %{{y:.2f}}<extra></extra>",
                )
            )

    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="green",
        line_width=1,
        annotation_text="Good (1.0)",
        annotation_position="right",
    )
    fig.add_hline(
        y=2,
        line_dash="dash",
        line_color="darkgreen",
        line_width=1,
        annotation_text="Excellent (2.0)",
        annotation_position="right",
    )

    fig.update_layout(
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    return fig


def plot_rolling_beta(
    analysis: PortfolioAnalysis | None = None,
    rolling_result: RollingMetricsResult | None = None,
    window: int = 126,
    theme: str | None = None,
    height: int = 400,
    width: int | None = None,
) -> go.Figure:
    """Plot rolling beta vs benchmark.

    Parameters
    ----------
    analysis : PortfolioAnalysis, optional
        Portfolio analysis object (must have benchmark)
    rolling_result : RollingMetricsResult, optional
        Pre-computed rolling metrics
    window : int, default 126
        Rolling window size (6 months)
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

    Raises
    ------
    ValueError
        If no benchmark data is available
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get rolling metrics
    if rolling_result is None:
        if analysis is None:
            raise ValueError("Must provide either analysis or rolling_result")
        if not analysis.has_benchmark:
            raise ValueError("Benchmark data required for beta calculation")

        rolling_result = analysis.compute_rolling_metrics(
            windows=[window],
            metrics=["beta"],
        )

    if not rolling_result.beta:
        raise ValueError("No beta data available")

    fig = create_base_figure(
        title=f"Rolling Beta ({window}d)",
        xaxis_title="Date",
        yaxis_title="Beta",
        height=height,
        width=width,
        theme=theme,
    )

    dates = rolling_result.dates.to_list()
    beta = rolling_result.beta[window].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=beta,
            mode="lines",
            name="Beta",
            line={"color": theme_config["colorway"][0], "width": 2},
            fill="tozeroy",
            fillcolor=f"rgba({int(theme_config['colorway'][0][1:3], 16)}, "
            f"{int(theme_config['colorway'][0][3:5], 16)}, "
            f"{int(theme_config['colorway'][0][5:7], 16)}, 0.2)",
            hovertemplate="Date: %{x}<br>Beta: %{y:.2f}<extra></extra>",
        )
    )

    # Reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="orange",
        line_width=1,
        annotation_text="Market (1.0)",
        annotation_position="right",
    )

    fig.update_layout(hovermode="x unified")

    return fig
