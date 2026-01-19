"""Event Study Visualization Module.

Provides interactive Plotly visualizations for event study analysis:
- CAAR time series with confidence bands
- Event drift heatmap
- AR distribution plots

All plots follow ML4T Diagnostic visualization standards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import get_theme_config as get_theme

if TYPE_CHECKING:
    from ml4t.diagnostic.results.event_results import (
        AbnormalReturnResult,
        EventStudyResult,
    )


def plot_caar(
    result: EventStudyResult,
    show_confidence: bool = True,
    show_aar_bars: bool = False,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot Cumulative Average Abnormal Returns (CAAR) time series.

    Creates a line plot of CAAR over the event window with optional
    confidence interval bands and vertical line at t=0.

    Parameters
    ----------
    result : EventStudyResult
        Event study results containing CAAR data.
    show_confidence : bool, default True
        Show confidence interval as shaded band.
    show_aar_bars : bool, default False
        Show AAR as bars in secondary y-axis.
    theme : str | None
        Visualization theme (None uses default).
    width : int | None
        Figure width in pixels.
    height : int | None
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure with CAAR visualization.

    Examples
    --------
    >>> result = analysis.run()
    >>> fig = plot_caar(result, show_confidence=True)
    >>> fig.show()
    """
    theme_config = get_theme(theme)
    colors = theme_config.get("colors", {})

    # Create figure
    if show_aar_bars:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Cumulative Average Abnormal Return (CAAR)", "Daily AAR"),
        )
    else:
        fig = go.Figure()

    # Get data
    x = result.caar_dates
    y = result.caar

    # Add confidence band
    if show_confidence:
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=result.caar_ci_upper + result.caar_ci_lower[::-1],
                fill="toself",
                fillcolor=colors.get("ci_fill", "rgba(31, 119, 180, 0.2)"),
                line={"color": "rgba(0,0,0,0)"},
                name=f"{int(result.confidence_level * 100)}% CI",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1 if show_aar_bars else None,
            col=1 if show_aar_bars else None,
        )

    # Add CAAR line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="CAAR",
            line={"color": colors.get("primary", "#1f77b4"), "width": 2},
            marker={"size": 6},
            hovertemplate="Day %{x}<br>CAAR: %{y:.4f}<extra></extra>",
        ),
        row=1 if show_aar_bars else None,
        col=1 if show_aar_bars else None,
    )

    # Add vertical line at t=0
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=colors.get("event_line", "red"),
        annotation_text="Event",
        annotation_position="top",
        row=1 if show_aar_bars else None,
        col=1 if show_aar_bars else None,
    )

    # Add horizontal line at y=0
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        row=1 if show_aar_bars else None,
        col=1 if show_aar_bars else None,
    )

    # Add AAR bars if requested
    if show_aar_bars:
        aar_x = list(result.aar_by_day.keys())
        aar_y = list(result.aar_by_day.values())

        bar_colors = [
            colors.get("positive", "#2ca02c") if v >= 0 else colors.get("negative", "#d62728")
            for v in aar_y
        ]

        fig.add_trace(
            go.Bar(
                x=aar_x,
                y=aar_y,
                marker_color=bar_colors,
                name="AAR",
                hovertemplate="Day %{x}<br>AAR: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    # Update layout
    title = f"Event Study: CAAR (n={result.n_events} events)"
    if result.is_significant:
        title += f" - Significant at {result.confidence_level:.0%}"

    fig.update_layout(
        title=title,
        xaxis_title="Days Relative to Event",
        yaxis_title="CAAR",
        width=width or 800,
        height=height or (600 if show_aar_bars else 450),
        template=theme_config.get("template", "plotly_white"),
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Add statistical info annotation
    annotation_text = (
        f"Test: {result.test_name}<br>"
        f"Stat: {result.test_statistic:.3f}<br>"
        f"p-value: {result.p_value:.4f}"
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=annotation_text,
        showarrow=False,
        font={"size": 10},
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def plot_event_heatmap(
    ar_results: list[AbnormalReturnResult],
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot event drift heatmap showing AR for each event by day.

    Creates a heatmap with events on Y-axis and relative days on X-axis,
    colored by abnormal return magnitude.

    Parameters
    ----------
    ar_results : list[AbnormalReturnResult]
        Individual event abnormal return results.
    theme : str | None
        Visualization theme.
    width : int | None
        Figure width.
    height : int | None
        Figure height.

    Returns
    -------
    go.Figure
        Plotly figure with heatmap.

    Examples
    --------
    >>> ar_results = analysis.compute_abnormal_returns()
    >>> fig = plot_event_heatmap(ar_results)
    >>> fig.show()
    """
    if not ar_results:
        raise ValueError("No abnormal return results provided")

    theme_config = get_theme(theme)

    # Collect all days and build matrix
    all_days: set[int] = set()
    for r in ar_results:
        all_days.update(r.ar_by_day.keys())
    sorted_days = sorted(all_days)

    # Build heatmap matrix
    z_matrix = []
    y_labels = []
    hover_texts = []

    for r in ar_results:
        row = []
        hover_row = []
        for day in sorted_days:
            ar = r.ar_by_day.get(day, float("nan"))
            row.append(ar)
            hover_row.append(
                f"Event: {r.event_id}<br>Asset: {r.asset}<br>Day: {day}<br>AR: {ar:.4f}"
                if not (ar != ar)  # not nan check
                else f"Day {day}: No data"
            )
        z_matrix.append(row)
        hover_texts.append(hover_row)
        y_labels.append(f"{r.event_id} ({r.asset})")

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=sorted_days,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": "AR"},
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
        )
    )

    # Add vertical line at event day
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)

    fig.update_layout(
        title="Abnormal Returns by Event and Day",
        xaxis_title="Days Relative to Event",
        yaxis_title="Event",
        width=width or 900,
        height=height or max(400, len(ar_results) * 25 + 100),
        template=theme_config.get("template", "plotly_white"),
    )

    return fig


def plot_ar_distribution(
    result: EventStudyResult,
    day: int = 0,
    show_kde: bool = True,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot distribution of abnormal returns for a specific day.

    Creates a histogram with optional KDE overlay showing the
    cross-sectional distribution of ARs on a given day.

    Parameters
    ----------
    result : EventStudyResult
        Event study results with individual event data.
    day : int, default 0
        Relative day to plot (0 = event day).
    show_kde : bool, default True
        Overlay kernel density estimate.
    theme : str | None
        Visualization theme.
    width : int | None
        Figure width.
    height : int | None
        Figure height.

    Returns
    -------
    go.Figure
        Plotly figure with AR distribution.
    """
    if result.individual_results is None:
        raise ValueError("EventStudyResult must include individual_results")

    theme_config = get_theme(theme)
    colors = theme_config.get("colors", {})

    # Collect ARs for the specified day
    ars = []
    for r in result.individual_results:
        if day in r.ar_by_day:
            ars.append(r.ar_by_day[day])

    if not ars:
        raise ValueError(f"No AR data available for day {day}")

    import numpy as np

    ars_array = np.array(ars)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=ars_array,
            nbinsx=min(20, len(ars)),
            name="AR Distribution",
            marker_color=colors.get("primary", "#1f77b4"),
            opacity=0.7,
            histnorm="probability density" if show_kde else None,
        )
    )

    # Add KDE if requested
    if show_kde and len(ars) >= 5:
        from scipy import stats as sp_stats

        kde = sp_stats.gaussian_kde(ars_array)
        x_range = np.linspace(min(ars_array), max(ars_array), 100)
        kde_y = kde(x_range)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_y,
                mode="lines",
                name="KDE",
                line={"color": colors.get("secondary", "#ff7f0e"), "width": 2},
            )
        )

    # Add vertical line at mean
    mean_ar = float(np.mean(ars_array))
    fig.add_vline(
        x=mean_ar,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_ar:.4f}",
        annotation_position="top right",
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    # Calculate statistics
    std_ar = float(np.std(ars_array, ddof=1))
    t_stat = mean_ar / (std_ar / np.sqrt(len(ars))) if std_ar > 0 else 0
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(ars) - 1)) if len(ars) > 1 else 1.0

    day_label = "Event Day" if day == 0 else f"Day {day:+d}"
    fig.update_layout(
        title=f"Abnormal Return Distribution - {day_label}",
        xaxis_title="Abnormal Return",
        yaxis_title="Density" if show_kde else "Count",
        width=width or 600,
        height=height or 400,
        template=theme_config.get("template", "plotly_white"),
    )

    # Add statistics annotation
    annotation_text = (
        f"n = {len(ars)}<br>"
        f"Mean = {mean_ar:.4f}<br>"
        f"Std = {std_ar:.4f}<br>"
        f"t-stat = {t_stat:.3f}<br>"
        f"p-value = {p_val:.4f}"
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        text=annotation_text,
        showarrow=False,
        font={"size": 10},
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def plot_car_by_event(
    ar_results: list[AbnormalReturnResult],
    sort_by: str = "car",
    top_n: int | None = None,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot cumulative abnormal returns by event as horizontal bar chart.

    Shows CAR for each event, sorted by magnitude, useful for
    identifying outliers and event heterogeneity.

    Parameters
    ----------
    ar_results : list[AbnormalReturnResult]
        Individual event results.
    sort_by : str, default "car"
        Sort by "car" (magnitude) or "date".
    top_n : int | None
        Show only top N events by magnitude.
    theme : str | None
        Visualization theme.
    width : int | None
        Figure width.
    height : int | None
        Figure height.

    Returns
    -------
    go.Figure
        Plotly figure with CAR bar chart.
    """
    if not ar_results:
        raise ValueError("No abnormal return results provided")

    theme_config = get_theme(theme)
    colors = theme_config.get("colors", {})

    # Sort results
    if sort_by == "car":
        sorted_results = sorted(ar_results, key=lambda x: abs(x.car), reverse=True)
    else:
        sorted_results = sorted(ar_results, key=lambda x: x.event_date)

    # Limit to top_n if specified
    if top_n is not None:
        sorted_results = sorted_results[:top_n]

    # Prepare data
    labels = [f"{r.event_id} ({r.asset})" for r in sorted_results]
    cars = [r.car for r in sorted_results]
    bar_colors = [
        colors.get("positive", "#2ca02c") if c >= 0 else colors.get("negative", "#d62728")
        for c in cars
    ]

    fig = go.Figure(
        data=go.Bar(
            y=labels,
            x=cars,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="Event: %{y}<br>CAR: %{x:.4f}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="gray")

    title = "Cumulative Abnormal Return by Event"
    if top_n is not None:
        title += f" (Top {top_n})"

    fig.update_layout(
        title=title,
        xaxis_title="CAR",
        yaxis_title="Event",
        width=width or 700,
        height=height or max(400, len(sorted_results) * 25 + 100),
        template=theme_config.get("template", "plotly_white"),
        yaxis={"autorange": "reversed"},  # Largest at top
    )

    return fig
