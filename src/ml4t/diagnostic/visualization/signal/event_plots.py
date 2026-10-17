"""Event Study Visualization Module.

Provides interactive Plotly visualizations for event study analysis:
- CAAR time series with confidence bands
- Event drift heatmap
- AR distribution plots

All plots follow ML4T Diagnostic visualization standards.
"""

from __future__ import annotations

import math
import tempfile
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization._colors import COLORS as _ML4T_COLORS
from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme

if TYPE_CHECKING:
    from ml4t.diagnostic.results.event_results import (
        AbnormalReturnResult,
        EventStudyResult,
    )
    from ml4t.diagnostic.signal.event_study import SignalEventStudyResult
    from ml4t.diagnostic.signal.events import AlignedEventResult, EventReturnsResult


def plot_aligned_event_paths(
    result: AlignedEventResult,
    *,
    std_multiplier: float = 1.0,
    quantile: int | None = None,
    group: str | int | None = None,
    theme: str | None = None,
) -> go.Figure:
    """Plot actual bucket/group path means and optional sample-std error bars.

    These are supplied levels or full-clock wealth, not event-zero CAR.
    Undefined values/deviations stay gaps; no standard error or CI is inferred.
    """
    if isinstance(std_multiplier, bool) or not np.isfinite(std_multiplier) or std_multiplier < 0:
        raise ValueError("std_multiplier must be finite and nonnegative")
    config = get_theme_config(validate_theme(theme))
    figure = go.Figure()
    data = result.statistics
    if quantile is not None:
        data = data.filter(pl.col("quantile") == quantile)
    if group is not None:
        if "group" not in data.columns:
            raise ValueError("event paths have no group summaries")
        data = data.filter(pl.col("group") == group)
    if data.is_empty():
        raise ValueError("no event paths match the selected quantile/group")
    keys = ["quantile", *(["group"] if "group" in data.columns else [])]
    for index, part in enumerate(data.partition_by(keys, maintain_order=True)):
        part = part.sort("offset")
        label = f"Q{part['quantile'][0]}"
        if "group" in part.columns:
            label += f" / {part['group'][0]}"
        figure.add_trace(
            go.Scatter(
                x=part["offset"].to_list(),
                y=part["mean"].to_list(),
                mode="lines+markers",
                name=label,
                line={"color": config["colorway"][index % len(config["colorway"])]},
                connectgaps=False,
                error_y={
                    "type": "data",
                    "array": [
                        None if value is None else value * std_multiplier for value in part["std"]
                    ],
                    "visible": std_multiplier > 0,
                },
                customdata=list(
                    zip(part["n_obs"].to_list(), part["n_requested"].to_list(), strict=True)
                ),
                hovertemplate="Offset %{x}<br>Mean %{y}<br>Finite paths %{customdata[0]} / %{customdata[1]}<extra>%{fullData.name}</extra>",
                meta={
                    "analysis": "aligned_event_paths",
                    "uncertainty": "sample_std",
                    "std_multiplier": std_multiplier,
                },
            )
        )
    units = {"supplied_levels": "Supplied levels", "wealth_starting_at_one": "Wealth (start = 1)"}[
        result.metadata["units"]
    ]
    if result.metadata["demeaned"] or result.metadata["group_adjust"]:
        units = (
            "Level difference" if result.metadata["input_kind"] == "levels" else "Wealth difference"
        )
    figure.update_layout(**config["layout"])
    figure.update_layout(
        title={"text": "Aligned event paths", "font": {"size": 16}},
        font={"size": 14},
        height=500,
        autosize=True,
        showlegend=True,
        legend={"orientation": "h", "y": -0.25, "x": 0, "font": {"size": 14}},
        margin={"l": 65, "r": 20, "t": 45, "b": 120},
        xaxis={"title": "Observation offset", "tickfont": {"size": 14}},
        yaxis={"title": units, "tickfont": {"size": 14}},
        meta={"analysis": "aligned_event_paths", **result.metadata},
    )
    figure.add_vline(x=0, line_dash="dot", line_color=_ML4T_COLORS["slate"])
    return figure


def plot_event_returns(
    result: EventReturnsResult, *, std_bar: bool = True, theme: str | None = None
) -> list[go.Figure]:
    """Render the overall path, optional quantile deviations and group overviews."""
    if not isinstance(std_bar, bool):
        raise TypeError("std_bar must be bool")
    overview = plot_aligned_event_paths(result.overall, std_multiplier=0, theme=theme)
    overview.update_layout(title_text="All quantiles")
    figures = [overview]
    if std_bar:
        for quantile in result.overall.statistics["quantile"].unique().sort():
            figure = plot_aligned_event_paths(result.overall, quantile=quantile, theme=theme)
            figure.update_layout(title_text=f"Quantile {quantile}")
            figures.append(figure)
    if result.groups is not None:
        for group in result.groups.statistics["group"].unique().sort():
            figure = plot_aligned_event_paths(
                result.groups, group=group, std_multiplier=0, theme=theme
            )
            figure.update_layout(title_text=f"Group: {escape(str(group))}")
            for trace in figure.data:
                trace.name = trace.name.split(" / ", 1)[0]
            figures.append(figure)
    return figures


def generate_event_returns_report(
    result: EventReturnsResult,
    *,
    output_file: str | Path | None = None,
    std_bar: bool = True,
    theme: str | None = None,
) -> str:
    """Compose all event-returns panels without replacing the global sample.

    Export the inspectable data separately with ``result.to_json()``. Display
    units remain supplied levels or full-clock wealth, not basis points or CAR.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    figures = plot_event_returns(result, std_bar=std_bar, theme=theme)
    metadata = result.overall.metadata
    adjustment = (
        "within each event-date group"
        if metadata["group_adjust"]
        else "across the event-date universe"
        if metadata["demeaned"]
        else "none"
    )
    sections = [
        {
            "title": str(figure.layout.title.text),
            "figure_index": index,
            "text": (
                f"Window: {metadata['before']} rows before and {metadata['after']} after each event. Demeaning: {adjustment}. Values use level or wealth units; adjusted values are differences, not wealth starting at one or event-zero returns. Undefined observations remain missing."
                if index == 0
                else "Error bars show sample standard deviation, not SE or confidence intervals."
                if str(figure.layout.title.text).startswith("Quantile ")
                else "Group-specific means supplement the global overview; deviations are not displayed."
            ),
        }
        for index, figure in enumerate(figures)
    ]
    html = combine_figures_to_html(
        figures, title="Signal event returns", theme=theme, sections=sections
    )
    return _publish_event_html(html, output_file)


def plot_signal_event_study(
    result: SignalEventStudyResult, *, theme: str | None = None
) -> list[go.Figure]:
    """Plot elapsed event counts, optional paths and the actual quantile samples."""
    config = get_theme_config(validate_theme(theme))
    counts = result.event_distribution
    distribution = go.Figure(
        go.Bar(x=counts["date"].to_list(), y=counts["n_events"].to_list(), name="Events")
    )
    distribution.update_layout(
        meta={
            "analysis": "elapsed_event_distribution",
            "bin_width_ns": result.metadata["bin_width_ns"],
        }
    )
    figures = [distribution]
    titles = [("Event distribution", "Events", "Elapsed bin start")]
    if result.event_returns is not None:
        paths = plot_event_returns(result.event_returns, std_bar=True, theme=theme)
        figures.extend(paths)
        titles.extend(
            (
                str(figure.layout.title.text),
                str(figure.layout.yaxis.title.text),
                "Observation offset",
            )
            for figure in paths
        )
    bars = go.Figure()
    violin = go.Figure()
    columns = result.metadata["return_columns"]
    for index, column in enumerate(columns):
        color = config["colorway"][index % len(config["colorway"])]
        means = result.mean_returns.filter(pl.col("return_column") == column).sort("quantile")
        sample = result.return_observations.filter(
            (pl.col("return_column") == column) & pl.col("display_mean").is_not_null()
        ).sort("quantile", "date")
        bars.add_trace(
            go.Bar(
                x=[f"Q{quantile}" for quantile in means["quantile"]],
                y=means["display_mean"].to_list(),
                name=column,
                marker_color=color,
                customdata=means["n_obs"].to_list(),
                hovertemplate="%{x}<br>Mean %{y:.4%}<br>Dates %{customdata}<extra>%{fullData.name}</extra>",
            )
        )
        violin.add_trace(
            go.Violin(
                x=[f"Q{quantile}" for quantile in sample["quantile"]],
                y=sample["display_mean"].to_list(),
                name=column,
                fillcolor=color,
                line_color=color,
                points="all",
                spanmode="hard",
                box_visible=True,
                meanline_visible=True,
                customdata=list(
                    zip(
                        sample["date"].to_pandas().astype(str).to_list(),
                        sample["n_obs"].to_list(),
                        strict=True,
                    )
                ),
                hovertemplate="%{x}<br>Date %{customdata[0]}<br>Return %{y:.4%}<br>Assets %{customdata[1]}<extra>%{fullData.name}</extra>",
            )
        )
    bars.update_layout(
        barmode="group",
        meta={
            "analysis": "event_quantile_means",
            "sample_unit": "date",
            "rate_of_return": result.metadata["rate_of_return"],
        },
    )
    violin.update_layout(
        violinmode="group",
        meta={
            "analysis": "event_quantile_observations",
            "sample_unit": "date",
            "visible_percentiles": [1, 99],
            "sample_policy": "all defined date/bucket means; no synthetic samples",
        },
    )
    values = result.return_observations["display_mean"].drop_nulls().to_numpy()
    if len(values):
        lower, upper = np.percentile(values, [1, 99])
        if lower < upper:
            violin.update_yaxes(range=[float(lower), float(upper)])
    figures.extend([bars, violin])
    unit = (
        f"Return per {result.metadata['base_period']}"
        if result.metadata["rate_of_return"]
        else "Supplied-horizon return"
    )
    titles.extend(
        [
            ("Quantile return means", unit, "Quantile"),
            ("Observed date/bucket returns", unit, "Quantile"),
        ]
    )
    for figure, (title, y_title, x_title) in zip(figures, titles, strict=True):
        figure.update_layout(**config["layout"])
        figure.update_layout(
            title={"text": title, "font": {"size": 16}},
            font={"size": 14},
            height=500,
            autosize=True,
            margin={"l": 70, "r": 20, "t": 45, "b": 120},
            legend={"orientation": "h", "y": -0.25, "x": 0, "font": {"size": 14}},
        )
        figure.update_xaxes(title_text=x_title, tickfont={"size": 14})
        figure.update_yaxes(title_text=y_title, tickfont={"size": 14})
    distribution.update_xaxes(tickangle=0, nticks=3, automargin=True)
    distribution.update_yaxes(dtick=1)
    bars.update_yaxes(tickformat=".2%")
    violin.update_yaxes(tickformat=".2%")
    return figures


def generate_signal_event_study_report(
    result: SignalEventStudyResult,
    *,
    output_file: str | Path | None = None,
    theme: str | None = None,
) -> str:
    """Compose factor stats, elapsed bins, optional paths and empirical return panels."""
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    table = (
        result.factor_statistics.rename(
            {
                "quantile": "Quantile",
                "min": "Minimum",
                "max": "Maximum",
                "mean": "Mean",
                "std": "Sample deviation",
                "count": "Events",
                "count_percent": "Event share (%)",
            }
        )
        .to_pandas()
        .to_html(
            index=False,
            escape=True,
            na_rep="Undefined",
            float_format=lambda value: format_finite(value, ".6g"),
            border=0,
        )
    )
    table = table.replace("<table ", '<table style="font-size:14px;border-collapse:collapse" ', 1)
    table = table.replace("<th>", '<th style="padding:6px">').replace(
        "<td>", '<td style="padding:6px">'
    )
    sections = [
        {"title": "Factor statistics", "text": f'<div style="overflow-x:auto">{table}</div>'}
    ]
    figures = plot_signal_event_study(result, theme=theme)
    for index, figure in enumerate(figures):
        analysis = (figure.layout.meta or {}).get("analysis")
        if analysis == "elapsed_event_distribution":
            text = "Counts include every prepared event and empty elapsed-time bins. Bin count can include an endpoint bin beyond the requested count; singleton event dates use one observed bin. These are not calendar sessions or expected bars."
        elif analysis in ("event_quantile_means", "event_quantile_observations"):
            text = f"Unadjusted date/bucket means give dates equal weight. Rate conversion: {result.metadata['rate_of_return']}; base horizon: {escape(result.metadata['base_period'])}. Raw horizon means, deviations and counts remain separately inspectable. The violin retains all defined samples; its visible range uses the 1st/99th percentiles, not sample trimming or simulated data."
        else:
            text = "Unadjusted supplied levels or full-clock wealth, not event-zero CAR. Offsets count observations. Separate quantile bars show sample deviation, not SE or confidence intervals."
        sections.append(
            {"title": str(figure.layout.title.text), "text": text, "figure_index": index}
        )
    if result.event_returns is None:
        sections.insert(
            2,
            {
                "title": "Aligned path availability",
                "text": f"Not displayed: {escape(result.metadata['event_paths_availability'])}. No path or uncertainty is synthesized.",
            },
        )
    html = combine_figures_to_html(
        figures, title="Signal event study", sections=sections, theme=theme
    )
    return _publish_event_html(html, output_file)


def plot_event_count_distribution(
    result: AlignedEventResult, *, by_group: bool = False, theme: str | None = None
) -> go.Figure:
    """Plot actual event counts, not the number of complete or estimated paths."""
    if not isinstance(by_group, bool):
        raise TypeError("by_group must be bool")
    data = result.event_counts
    if by_group and "group" not in data.columns:
        raise ValueError("event counts have no group labels")
    keys = ["quantile", *(["group"] if by_group else [])]
    data = data.group_by(["date", *keys]).agg(pl.col("n_events").sum()).sort([*keys, "date"])
    config = get_theme_config(validate_theme(theme))
    figure = go.Figure()
    for index, part in enumerate(data.partition_by(keys, maintain_order=True)):
        label = f"Q{part['quantile'][0]}"
        if by_group:
            label += f" / {part['group'][0]}"
        figure.add_trace(
            go.Bar(
                x=part["date"].to_list(),
                y=part["n_events"].to_list(),
                name=label,
                marker_color=config["colorway"][index % len(config["colorway"])],
                meta={
                    "analysis": "event_counts",
                    "sample": "all declared events, before window clipping",
                },
            )
        )
    figure.update_layout(**config["layout"])
    figure.update_layout(
        title={"text": "Event counts", "font": {"size": 16}},
        font={"size": 14},
        height=500,
        autosize=True,
        showlegend=True,
        barmode="stack",
        legend={"orientation": "h", "y": -0.25, "x": 0, "font": {"size": 14}},
        margin={"l": 65, "r": 20, "t": 45, "b": 120},
        xaxis={
            "title": "Event timestamp",
            "tickfont": {"size": 14},
            "tickangle": 0,
            "nticks": 3,
            "automargin": True,
        },
        yaxis={"title": "Events", "tickfont": {"size": 14}, "dtick": 1},
        meta={"analysis": "event_counts", "calendar_validation": "not performed"},
    )
    return figure


def generate_aligned_event_report(
    result: AlignedEventResult,
    *,
    output_file: str | Path | None = None,
    std_multiplier: float = 1.0,
    theme: str | None = None,
) -> str:
    """Use the existing HTML composer; publish a file only after complete rendering.

    Return HTML when output_file is omitted, otherwise return the absolute path.
    The two panels show actual aligned means/sample deviations and event counts.
    This is not an abnormal-return report or the complete legacy event tear sheet.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    figures = [
        plot_aligned_event_paths(result, std_multiplier=std_multiplier, theme=theme),
        plot_event_count_distribution(result, by_group=result.metadata["by_group"], theme=theme),
    ]
    html = combine_figures_to_html(
        figures,
        title="Aligned signal events",
        theme=theme,
        sections=[
            {
                "title": "Aligned event paths",
                "text": "Offsets count observed rows. Supplied levels are not rebased at event zero. Error bars show sample standard deviation, not SE or confidence intervals. Undefined paths remain missing.",
                "figure_index": 0,
            },
            {
                "title": "Event counts",
                "text": "Counts retain declared events before window clipping. Calendar-grid validation and abnormal-return estimation are not performed.",
                "figure_index": 1,
            },
        ],
    )
    return _publish_event_html(html, output_file)


def _publish_event_html(html: str, output_file: str | Path | None) -> str:
    if output_file is None:
        return html
    target = Path(output_file).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ml4t-events-", dir=target.parent) as directory:
        temporary = Path(directory) / "report.html"
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(target)
    return str(target)


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
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

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
    ci_available = any(
        np.isfinite(lower) and np.isfinite(upper)
        for lower, upper in zip(result.caar_ci_lower, result.caar_ci_upper, strict=False)
    )
    if show_confidence and ci_available:
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=result.caar_ci_upper + result.caar_ci_lower[::-1],
                fill="toself",
                fillcolor="rgba(10, 22, 40, 0.15)",  # _ML4T_COLORS["blue"] at 15%
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
            line={"color": theme_config["colorway"][0], "width": 2},
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
        line_color=_ML4T_COLORS["negative"],
        annotation_text="Event",
        annotation_position="top",
        row=1 if show_aar_bars else None,
        col=1 if show_aar_bars else None,
    )

    # Add horizontal line at y=0
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=_ML4T_COLORS["silver_muted"],
        row=1 if show_aar_bars else None,
        col=1 if show_aar_bars else None,
    )

    # Add AAR bars if requested
    if show_aar_bars:
        aar_x = list(result.aar_by_day.keys())
        aar_y = list(result.aar_by_day.values())

        bar_colors = [
            _ML4T_COLORS["positive"] if v >= 0 else _ML4T_COLORS["negative"] for v in aar_y
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

        fig.add_vline(x=0, line_dash="dash", line_color=_ML4T_COLORS["negative"], row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=_ML4T_COLORS["silver_muted"], row=2, col=1)

    # Update layout
    title = f"Event Study: CAAR (n={result.n_events} events)"
    if result.is_significant:
        title += f" - Significant at {result.confidence_level:.0%}"

    layout_updates = {
        "title": title,
        "xaxis_title": "Days Relative to Event",
        "yaxis_title": "CAAR",
        "width": width or 800,
        "height": height or (600 if show_aar_bars else 450),
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    # Add statistical info annotation
    annotation_text = (
        f"Test: {result.test_name}<br>"
        f"Stat: {format_finite(result.test_statistic, '.3f')}<br>"
        f"p-value: {format_finite(result.p_value, '.4f')}"
    )
    if show_confidence and not ci_available:
        annotation_text += "<br>CI unavailable"
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
        bordercolor=_ML4T_COLORS["silver_muted"],
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

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

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
            has_observation = day in r.ar_by_day
            ar = r.ar_by_day.get(day, float("nan"))
            row.append(ar)
            if has_observation:
                hover_row.append(
                    f"Event: {r.event_id}<br>Asset: {r.asset}<br>Day: {day}<br>"
                    f"AR: {format_finite(ar, '.4f')}"
                )
            else:
                hover_row.append(f"Day {day}: No data")
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
    fig.add_vline(x=0, line_dash="dash", line_color=_ML4T_COLORS["neutral"], line_width=2)

    layout_updates = {
        "title": "Abnormal Returns by Event and Day",
        "xaxis_title": "Days Relative to Event",
        "yaxis_title": "Event",
        "width": width or 900,
        "height": height or max(400, len(ar_results) * 25 + 100),
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

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

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)
    colorway = theme_config["colorway"]

    # Collect ARs for the specified day
    ars = []
    for r in result.individual_results:
        if day in r.ar_by_day:
            ars.append(r.ar_by_day[day])

    if not ars:
        raise ValueError(f"No AR data available for day {day}")

    ars_array = np.array(ars)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=ars_array,
            nbinsx=min(20, len(ars)),
            name="AR Distribution",
            marker_color=colorway[0],
            opacity=0.7,
            histnorm="probability density" if show_kde else None,
        )
    )

    # Add KDE if requested
    all_finite = bool(np.isfinite(ars_array).all())
    if show_kde and len(ars) >= 5 and all_finite:
        kde = sp_stats.gaussian_kde(ars_array)
        x_range = np.linspace(min(ars_array), max(ars_array), 100)
        kde_y = kde(x_range)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_y,
                mode="lines",
                name="KDE",
                line={
                    "color": colorway[1] if len(colorway) > 1 else _ML4T_COLORS["amber"],
                    "width": 2,
                },
            )
        )

    # Add vertical line at mean
    mean_ar = float(np.mean(ars_array))
    if np.isfinite(mean_ar):
        fig.add_vline(
            x=mean_ar,
            line_dash="dash",
            line_color=_ML4T_COLORS["negative"],
            annotation_text=f"Mean: {format_finite(mean_ar, '.4f')}",
            annotation_position="top right",
        )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dot", line_color=_ML4T_COLORS["silver_muted"])

    # Calculate statistics
    std_ar = float(np.std(ars_array, ddof=1))
    if not np.isfinite(mean_ar) or not np.isfinite(std_ar):
        t_stat = p_val = float("nan")
    elif std_ar > 0:
        t_stat = mean_ar / (std_ar / np.sqrt(len(ars)))
        p_val = 2 * sp_stats.t.sf(abs(t_stat), df=len(ars) - 1)
    elif mean_ar == 0:
        t_stat, p_val = 0.0, 1.0
    else:
        t_stat = p_val = float("nan")

    day_label = "Event Day" if day == 0 else f"Day {day:+d}"

    layout_updates = {
        "title": f"Abnormal Return Distribution - {day_label}",
        "xaxis_title": "Abnormal Return",
        "yaxis_title": "Density" if show_kde else "Count",
        "width": width or 600,
        "height": height or 400,
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    # Add statistics annotation
    annotation_text = (
        f"n = {len(ars)}<br>"
        f"Mean = {format_finite(mean_ar, '.4f')}<br>"
        f"Std = {format_finite(std_ar, '.4f')}<br>"
        f"t-stat = {format_finite(t_stat, '.3f')}<br>"
        f"p-value = {format_finite(p_val, '.4f')}"
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
        bordercolor=_ML4T_COLORS["silver_muted"],
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

    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Sort results
    if sort_by == "car":
        sorted_results = sorted(
            ar_results,
            key=lambda result: abs(result.car) if math.isfinite(result.car) else -math.inf,
            reverse=True,
        )
    else:
        sorted_results = sorted(ar_results, key=lambda x: x.event_date)

    # Limit to top_n if specified
    if top_n is not None:
        sorted_results = sorted_results[:top_n]
    unavailable_count = sum(not math.isfinite(result.car) for result in sorted_results)

    # Prepare data
    labels = [f"{r.event_id} ({r.asset})" for r in sorted_results]
    cars = [r.car for r in sorted_results]
    plotted_cars = [car if math.isfinite(car) else 0.0 for car in cars]
    bar_colors = [
        _ML4T_COLORS["silver_muted"]
        if not math.isfinite(car)
        else _ML4T_COLORS["positive"]
        if car >= 0
        else _ML4T_COLORS["negative"]
        for car in cars
    ]
    car_labels = [format_finite(car, ".4f") for car in cars]
    bar_text = ["" if math.isfinite(car) else "N/A" for car in cars]

    fig = go.Figure(
        data=go.Bar(
            y=labels,
            x=plotted_cars,
            orientation="h",
            marker_color=bar_colors,
            customdata=car_labels,
            text=bar_text,
            textposition="outside",
            hovertemplate="Event: %{y}<br>CAR: %{customdata}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color=_ML4T_COLORS["silver_muted"])

    title = "Cumulative Abnormal Return by Event"
    if top_n is not None:
        title += f" (Top {top_n})"
    if unavailable_count:
        title += f" - {unavailable_count} unavailable"

    layout_updates = {
        "title": title,
        "xaxis_title": "CAR",
        "yaxis_title": "Event",
        "width": width or 700,
        "height": height or max(400, len(sorted_results) * 25 + 100),
        "yaxis": {"autorange": "reversed"},
    }

    for key, value in theme_config["layout"].items():
        if key not in layout_updates:
            layout_updates[key] = value

    fig.update_layout(**layout_updates)

    return fig
