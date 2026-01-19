"""Multi-Signal Analysis Visualization Plots.

This module provides interactive Plotly visualizations for multi-signal analysis:
- plot_ic_ridge: IC density ridge plot showing distribution per signal
- plot_signal_ranking_bar: Horizontal bar chart of signals by metric
- plot_signal_correlation_heatmap: Signal correlation heatmap with clustering
- plot_pareto_frontier: Scatter plot with Pareto frontier highlighted

All plots follow the Focus+Context pattern for analyzing 50-200 signals:
- Focus: Selected/significant signals highlighted
- Context: All signals shown in background for comparison

References
----------
Tufte, E. (1983). "The Visual Display of Quantitative Information"
Few, S. (2012). "Show Me the Numbers"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_colorscale,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    import polars as pl

    from ml4t.diagnostic.results.multi_signal_results import MultiSignalSummary


# =============================================================================
# IC Ridge Plot
# =============================================================================


def plot_ic_ridge(
    summary: MultiSignalSummary,
    max_signals: int = 50,
    sort_by: str = "ic_mean",
    show_significance: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """IC density ridge plot showing IC distribution per signal.

    Displays horizontal bars from IC 5th to 95th percentile with point at mean.
    Color indicates FDR significance (green=significant, gray=not significant).

    Parameters
    ----------
    summary : MultiSignalSummary
        Summary metrics from MultiSignalAnalysis.compute_summary()
    max_signals : int, default 50
        Maximum number of signals to display
    sort_by : str, default "ic_mean"
        Metric to sort signals by. Options: "ic_mean", "ic_ir", "ic_t_stat"
    show_significance : bool, default True
        Color by FDR significance
    theme : str | None
        Plot theme (default, dark, print, presentation)
    width : int | None
        Figure width in pixels
    height : int | None
        Figure height in pixels (auto-scaled by n_signals if None)

    Returns
    -------
    go.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> summary = analyzer.compute_summary()
    >>> fig = plot_ic_ridge(summary, max_signals=30, sort_by="ic_ir")
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get DataFrame and sort
    df = summary.get_dataframe()

    if sort_by not in df.columns:
        available = [c for c in df.columns if "ic" in c.lower()]
        raise ValueError(f"Sort metric '{sort_by}' not found. Available: {available}")

    # Sort and limit
    df = df.sort(sort_by, descending=True).head(max_signals)
    n_signals = len(df)

    # Extract data
    signal_names = df["signal_name"].to_list()
    ic_means = df["ic_mean"].to_list() if "ic_mean" in df.columns else [0] * n_signals

    # Get percentiles if available, otherwise use std
    if "ic_p5" in df.columns and "ic_p95" in df.columns:
        ic_lower = df["ic_p5"].to_list()
        ic_upper = df["ic_p95"].to_list()
    elif "ic_std" in df.columns:
        ic_stds = df["ic_std"].to_list()
        ic_lower = [m - 1.96 * s for m, s in zip(ic_means, ic_stds)]
        ic_upper = [m + 1.96 * s for m, s in zip(ic_means, ic_stds)]
    else:
        ic_lower = ic_means
        ic_upper = ic_means

    # Get significance flags
    if show_significance and "fdr_significant" in df.columns:
        fdr_significant = df["fdr_significant"].to_list()
    else:
        fdr_significant = [False] * n_signals

    # Colors: significant=green, not significant=gray
    colors = [theme_config["colorway"][0] if sig else "#888888" for sig in fdr_significant]

    # Calculate height based on number of signals
    if height is None:
        height = max(400, min(1200, n_signals * 25 + 100))

    # Create figure
    fig = create_base_figure(
        title=f"IC Distribution by Signal (Top {n_signals} by {sort_by})",
        xaxis_title="Information Coefficient",
        yaxis_title="",
        width=width or 800,
        height=height,
        theme=theme,
    )

    # Add range bars (5th to 95th percentile)
    for i, (name, lower, upper, mean, color) in enumerate(
        zip(signal_names, ic_lower, ic_upper, ic_means, colors)
    ):
        # Range line
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[name, name],
                mode="lines",
                line={"color": color, "width": 4},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean point
        fig.add_trace(
            go.Scatter(
                x=[mean],
                y=[name],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "diamond"},
                name=name if i == 0 else None,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"IC Mean: {mean:.4f}<br>"
                    f"IC Range: [{lower:.4f}, {upper:.4f}]"
                    "<extra></extra>"
                ),
            )
        )

    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Update layout for horizontal bar style
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": signal_names[::-1]},
        showlegend=False,
        margin={"l": 200, "r": 50, "t": 60, "b": 50},
    )

    return fig


# =============================================================================
# Signal Ranking Bar Chart
# =============================================================================


def plot_signal_ranking_bar(
    summary: MultiSignalSummary,
    metric: str = "ic_ir",
    top_n: int = 20,
    color_by: str = "fdr_significant",
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of top signals by metric.

    Parameters
    ----------
    summary : MultiSignalSummary
        Summary metrics from MultiSignalAnalysis.compute_summary()
    metric : str, default "ic_ir"
        Metric to rank by. Options: "ic_ir", "ic_mean", "ic_t_stat"
    top_n : int, default 20
        Number of top signals to display
    color_by : str, default "fdr_significant"
        How to color bars: "fdr_significant", "fwer_significant", or None
    theme : str | None
        Plot theme
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
    >>> fig = plot_signal_ranking_bar(summary, metric="ic_ir", top_n=15)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Get DataFrame and sort
    df = summary.get_dataframe()

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {df.columns}")

    df = df.sort(metric, descending=True).head(top_n)

    signal_names = df["signal_name"].to_list()
    values = df[metric].to_list()

    # Determine colors
    if color_by and color_by in df.columns:
        significant = df[color_by].to_list()
        colors = [theme_config["colorway"][0] if sig else "#CCCCCC" for sig in significant]
    else:
        colors = [theme_config["colorway"][0]] * len(signal_names)

    # Calculate height
    if height is None:
        height = max(400, min(800, top_n * 30 + 100))

    # Create figure
    fig = create_base_figure(
        title=f"Top {top_n} Signals by {metric.upper().replace('_', ' ')}",
        xaxis_title=metric.upper().replace("_", " "),
        yaxis_title="",
        width=width or 700,
        height=height,
        theme=theme,
    )

    # Add horizontal bars
    fig.add_trace(
        go.Bar(
            x=values,
            y=signal_names,
            orientation="h",
            marker={"color": colors},
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": signal_names[::-1]},
        margin={"l": 200, "r": 80, "t": 60, "b": 50},
    )

    return fig


# =============================================================================
# Signal Correlation Heatmap
# =============================================================================


def plot_signal_correlation_heatmap(
    correlation_matrix: pl.DataFrame,
    cluster: bool = True,
    max_signals: int = 100,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Signal correlation heatmap with optional hierarchical clustering.

    Reveals "100 signals = 3 unique bets" pattern through correlation analysis.
    When clustering is enabled, reorders signals by dendrogram to show clusters.

    Parameters
    ----------
    correlation_matrix : pl.DataFrame
        Signal correlation matrix from MultiSignalAnalysis.correlation_matrix()
    cluster : bool, default True
        Apply hierarchical clustering to reorder signals
    max_signals : int, default 100
        Maximum signals to display (limits browser memory)
    theme : str | None
        Plot theme
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
    >>> corr_matrix = analyzer.correlation_matrix()
    >>> fig = plot_signal_correlation_heatmap(corr_matrix, cluster=True)
    >>> fig.show()
    """
    theme = validate_theme(theme)
    get_theme_config(theme)

    # Get signal names and correlation values
    signal_names = correlation_matrix.columns

    # Limit to max_signals (take first N)
    if len(signal_names) > max_signals:
        signal_names = signal_names[:max_signals]
        correlation_matrix = correlation_matrix.select(signal_names)
        # Filter rows as well
        correlation_matrix = correlation_matrix.head(max_signals)

    n_signals = len(signal_names)

    # Convert to numpy for clustering
    corr_values = correlation_matrix.to_numpy()

    # Hierarchical clustering to reorder
    if cluster and n_signals > 2:
        # Convert correlation to distance (1 - abs(corr))
        # Handle any NaN values
        corr_clean = np.nan_to_num(corr_values, nan=0.0)
        distance_matrix = 1 - np.abs(corr_clean)

        # Ensure symmetry and proper diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Clip to valid range
        distance_matrix = np.clip(distance_matrix, 0, 2)

        # Convert to condensed form and perform clustering
        condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed, method="average")

        # Get leaf order from dendrogram
        dend = dendrogram(linkage_matrix, no_plot=True)
        order = dend["leaves"]

        # Reorder signals and correlation matrix
        signal_names = [signal_names[i] for i in order]
        corr_values = corr_values[np.ix_(order, order)]

    # Determine size
    if width is None:
        width = max(600, min(1000, n_signals * 10 + 200))
    if height is None:
        height = max(600, min(1000, n_signals * 10 + 200))

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_values,
            x=signal_names,
            y=signal_names,
            colorscale=get_colorscale("rdbu"),
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"title": "Correlation", "tickformat": ".2f"},
            hovertemplate=("<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"),
        )
    )

    # Update layout
    title = f"Signal Correlation Matrix ({n_signals} signals)"
    if cluster:
        title += " - Clustered"

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis={"tickangle": 45, "side": "bottom"},
        yaxis={"autorange": "reversed"},
        margin={"l": 150, "r": 50, "t": 60, "b": 150},
    )

    return fig


# =============================================================================
# Pareto Frontier Plot
# =============================================================================


def plot_pareto_frontier(
    summary: MultiSignalSummary,
    x_metric: str = "turnover_mean",
    y_metric: str = "ic_ir",
    minimize_x: bool = True,
    maximize_y: bool = True,
    highlight_pareto: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Scatter plot with Pareto frontier highlighted.

    Shows all signals as points with Pareto-optimal signals connected by line.
    Useful for identifying signals that offer best trade-offs between metrics.

    Parameters
    ----------
    summary : MultiSignalSummary
        Summary metrics from MultiSignalAnalysis.compute_summary()
    x_metric : str, default "turnover_mean"
        Metric for x-axis (typically something to minimize)
    y_metric : str, default "ic_ir"
        Metric for y-axis (typically something to maximize)
    minimize_x : bool, default True
        If True, lower x values are better
    maximize_y : bool, default True
        If True, higher y values are better
    highlight_pareto : bool, default True
        Highlight Pareto-optimal signals
    theme : str | None
        Plot theme
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
    >>> fig = plot_pareto_frontier(summary, x_metric="turnover_mean", y_metric="ic_ir")
    >>> fig.show()
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    df = summary.get_dataframe()

    # Validate metrics
    for m in [x_metric, y_metric]:
        if m not in df.columns:
            raise ValueError(f"Metric '{m}' not found. Available: {df.columns}")

    signal_names = df["signal_name"].to_list()
    x_values = df[x_metric].to_list()
    y_values = df[y_metric].to_list()

    # Identify Pareto frontier
    pareto_mask = _compute_pareto_mask(x_values, y_values, minimize_x, maximize_y)
    pareto_signals = [n for n, p in zip(signal_names, pareto_mask) if p]

    # Colors: Pareto=primary color, others=gray
    colors = [theme_config["colorway"][0] if p else "#CCCCCC" for p in pareto_mask]

    # Create figure
    fig = create_base_figure(
        title=f"Signal Efficiency: {y_metric} vs {x_metric}",
        xaxis_title=x_metric.upper().replace("_", " "),
        yaxis_title=y_metric.upper().replace("_", " "),
        width=width or 800,
        height=height or 600,
        theme=theme,
    )

    # Add all signals as scatter
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker={
                "size": 10,
                "color": colors,
                "line": {"width": 1, "color": "white"},
            },
            text=signal_names,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{x_metric}: %{{x:.4f}}<br>"
                f"{y_metric}: %{{y:.4f}}"
                "<extra></extra>"
            ),
            name="All Signals",
        )
    )

    # Add Pareto frontier line
    if highlight_pareto and len(pareto_signals) > 1:
        # Get Pareto points and sort for line
        pareto_x = [x for x, p in zip(x_values, pareto_mask) if p]
        pareto_y = [y for y, p in zip(y_values, pareto_mask) if p]

        # Sort by x for nice line
        sorted_pairs = sorted(zip(pareto_x, pareto_y))
        pareto_x_sorted = [p[0] for p in sorted_pairs]
        pareto_y_sorted = [p[1] for p in sorted_pairs]

        fig.add_trace(
            go.Scatter(
                x=pareto_x_sorted,
                y=pareto_y_sorted,
                mode="lines",
                line={"color": theme_config["colorway"][1], "width": 2, "dash": "dot"},
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

    # Add annotation for number of Pareto signals
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Pareto optimal: {len(pareto_signals)} / {len(signal_names)}",
        showarrow=False,
        font={"size": 12},
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=theme_config["colorway"][0],
        borderwidth=1,
    )

    return fig


def _compute_pareto_mask(
    x_values: list[float],
    y_values: list[float],
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> list[bool]:
    """Compute Pareto frontier mask.

    Returns True for points on the Pareto frontier (non-dominated).
    """
    n = len(x_values)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            x_better = x_values[j] <= x_values[i] if minimize_x else x_values[j] >= x_values[i]
            y_better = y_values[j] >= y_values[i] if maximize_y else y_values[j] <= y_values[i]

            x_strictly = x_values[j] < x_values[i] if minimize_x else x_values[j] > x_values[i]
            y_strictly = y_values[j] > y_values[i] if maximize_y else y_values[j] < y_values[i]

            # j dominates i if j is at least as good in both and strictly better in one
            if x_better and y_better and (x_strictly or y_strictly):
                is_pareto[i] = False
                break

    return is_pareto


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "plot_ic_ridge",
    "plot_signal_ranking_bar",
    "plot_signal_correlation_heatmap",
    "plot_pareto_frontier",
]
