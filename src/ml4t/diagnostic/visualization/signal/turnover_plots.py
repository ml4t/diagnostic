"""Turnover and signal stability visualization plots.

This module provides interactive Plotly visualizations for turnover analysis:
- plot_top_bottom_turnover: Turnover rates for extreme quantiles
- plot_autocorrelation: Signal rank autocorrelation by lag
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization.core import (
    create_base_figure,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult


def _plot_dated_persistence(
    result: TurnoverAnalysisResult,
    *,
    rank: bool,
    lag: int | None,
    show_all_quantiles: bool,
    show_significance: bool,
    theme: str,
    width: int | None,
    height: int | None,
) -> go.Figure:
    import pandas as pd

    from ml4t.diagnostic.signal._prepared import positive_count

    observations = result.get_dataframe("dated_autocorrelation" if rank else "dated_turnover")
    available = observations["lag"].unique().sort().to_list()
    if not available and lag is None:
        figure = create_base_figure(
            title="Dated Rank Autocorrelation" if rank else "Dated Quantile Turnover",
            theme=theme,
            width=width,
            height=height,
        )
        figure.add_annotation(
            text="Rank autocorrelation unavailable" if rank else "Turnover unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure
    selected = (
        positive_count(lag, "lag") if lag is not None else available[0] if available else None
    )
    if selected not in available:
        raise ValueError(
            f"lag {selected!r} not found in dated observations. Available: {available}"
        )
    observations = observations.filter(pl.col("lag") == selected).sort("date")
    config = get_theme_config(theme)
    figure = create_base_figure(
        title=f"Dated {'Rank Autocorrelation' if rank else 'Quantile Turnover'}<br>Lag {selected} Observations",
        xaxis_title="Observed date",
        yaxis_title="Rank autocorrelation" if rank else "Entrants / current membership",
        theme=theme,
        width=width or config["defaults"]["line_height"] + 300,
        height=height or config["defaults"]["line_height"],
        margin={"t": 90},
    )
    metadata = {"lag": selected, "lag_unit": "observations"}
    for column in (
        ["clock_source", "rank_scope", "correlation_method"]
        if rank
        else ["clock_source", "turnover_method"]
    ):
        values = observations[column].unique().to_list()
        if len(values) != 1:
            raise ValueError(f"dated observations mix {column} values")
        metadata[column] = values[0]
    buckets = [None] if rank else observations["quantile"].unique().sort().to_list()
    if not rank and not show_all_quantiles:
        buckets = list(dict.fromkeys([buckets[-1], buckets[0]])) if buckets else []
    defined = observations["autocorrelation" if rank else "turnover"].drop_nulls()
    category_clock = not len(defined)
    for index, bucket in enumerate(buckets):
        rows = observations if rank else observations.filter(pl.col("quantile") == bucket)
        if rows["date"].dtype.is_temporal():
            timestamps = [pd.Timestamp(value) for value in rows["date"].to_pandas()]
            dates = [value.isoformat() for value in timestamps]
            category_clock |= any(value.value % 1_000_000 for value in timestamps)
            category_clock |= len(timestamps) == 1
        else:
            dates = [str(value) for value in rows["date"]]
            category_clock = True
        fields = (
            ["n_obs", "n_current", "n_previous", "status"]
            if rank
            else ["n_current", "n_previous", "n_overlap", "n_entered", "status"]
        )
        custom = [
            [day, *[row.get(field) for field in fields]]
            for day, row in zip(dates, rows.iter_rows(named=True), strict=True)
        ]
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=rows["autocorrelation" if rank else "turnover"].to_list(),
                name="Full-date rank correlation" if rank else f"Q{bucket}",
                meta={"quantile": bucket} if not rank else None,
                mode="lines+markers",
                connectgaps=False,
                customdata=custom,
                line={"color": config["colorway"][index % len(config["colorway"])]},
                marker={"size": 4},
                hovertemplate="Date: %{customdata[0]}<br>"
                + ("Correlation: %{y:.6f}" if rank else "Turnover: %{y:.2%}")
                + "".join(
                    f"<br>{field}: %{{customdata[{i + 1}]}}" for i, field in enumerate(fields)
                )
                + "<extra>%{fullData.name}</extra>",
            )
        )
    metadata["x_clock"] = "observation_order" if category_clock else "dated"
    figure.update_layout(
        meta=metadata, showlegend=True, legend={"orientation": "h", "x": 0, "y": -0.2}
    )
    figure.update_xaxes(type="category" if category_clock else "date")
    if category_clock and figure.data:
        dates = list(figure.data[0].x)
        figure.update_xaxes(
            title_text="Observed date order",
            tickmode="array",
            tickvals=dates,
            ticktext=[str(index + 1) for index in range(len(dates))],
        )
    if not rank:
        figure.update_yaxes(tickformat=".0%", range=[-0.05, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1])
    else:
        figure.update_yaxes(range=[-1.05, 1.05], tickvals=[-1, -0.5, 0, 0.5, 1])
    if not len(defined):
        figure.add_annotation(
            text="Rank autocorrelation unavailable" if rank else "Turnover unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    if rank and show_significance:
        figure.add_annotation(
            text="CI unavailable",
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            yshift=8,
            showarrow=False,
        )
    return figure


def _plot_dated_turnover_heatmap(
    result: TurnoverAnalysisResult, theme: str, width: int | None, height: int | None
) -> go.Figure:
    values = result.get_dataframe("turnover")
    lags = values["lag"].unique().sort().to_list()
    buckets = values["quantile"].unique().sort().to_list()
    cells = {(row["quantile"], row["lag"]): row for row in values.to_dicts()}
    z = [[cells[(bucket, lag)]["turnover"] for lag in lags] for bucket in buckets]
    counts = [[cells[(bucket, lag)]["n_dates"] for lag in lags] for bucket in buckets]
    figure = create_base_figure(
        title="Dated Mean Turnover by Quantile and Observation Lag",
        xaxis_title="Lag (Observations)",
        yaxis_title="Quantile",
        width=width,
        height=height or 300 + 30 * len(buckets),
        theme=theme,
    )
    if not values.height:
        figure.add_annotation(
            text="Turnover unavailable",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure
    figure.add_trace(
        go.Heatmap(
            x=lags,
            y=[f"Q{bucket}" for bucket in buckets],
            z=z,
            customdata=counts,
            text=[
                [
                    f"{value:.1%}" if value is not None and np.isfinite(value) else ""
                    for value in row
                ]
                for row in z
            ],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Oranges",
            zmin=0,
            zmax=1,
            colorbar={"title": "Turnover", "tickformat": ".0%"},
            hovertemplate="%{y}<br>Lag: %{x} observations<br>Mean turnover: %{z:.2%}<br>Defined dates: %{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        meta={
            "lag_unit": "observations",
            "turnover_method": "entrants_over_current",
            "reduction": "mean_of_defined_dates",
        }
    )
    if not values["turnover"].drop_nulls().len():
        figure.add_annotation(
            text="Turnover unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    return figure


def plot_top_bottom_turnover(
    turnover_result: TurnoverAnalysisResult,
    show_all_quantiles: bool = False,
    theme: str | None = None,
    width: int | None = None,
    height: int | None = None,
    *,
    lag: int | None = None,
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
    lag : int | None
        Observation lag for dated results; defaults to the first recorded lag.
        Dated inputs render actual time series, while legacy inputs retain bars.

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
    if turnover_result.dated_turnover_ipc_base64 is not None:
        return _plot_dated_persistence(
            turnover_result,
            rank=False,
            lag=lag,
            show_all_quantiles=show_all_quantiles,
            show_significance=False,
            theme=theme,
            width=width,
            height=height,
        )
    if lag is not None:
        raise ValueError("lag selects dated observations, not legacy horizon summaries")

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
            hl_display = format_finite(hl, ".1f") if hl is not None else "N/A"
            unit = " periods" if hl is not None and np.isfinite(hl) else ""
            hl_text += f"{period}: {hl_display}{unit}<br>"

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
    *,
    lag: int | None = None,
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
    lag : int | None
        Observation lag for dated results; defaults to the first recorded lag.
        Dated inputs render full-date-rank time series, without inferred confidence
        bands. Their period and max_lags selectors do not apply.

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
    if turnover_result.dated_turnover_ipc_base64 is not None:
        if period is not None:
            raise ValueError("period selects legacy horizons; use lag for dated observations")
        if max_lags is not None:
            raise ValueError(
                "max_lags selects legacy lag summaries; use lag for dated observations"
            )
        return _plot_dated_persistence(
            turnover_result,
            rank=True,
            lag=lag,
            show_all_quantiles=False,
            show_significance=show_significance,
            theme=theme,
            width=width,
            height=height,
        )
    if lag is not None:
        raise ValueError("lag selects dated observations, not legacy horizon summaries")

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
    half_life_display = format_finite(half_life, ".1f") if half_life is not None else "N/A"
    half_life_unit = " periods" if half_life is not None and np.isfinite(half_life) else ""

    summary_text = (
        f"<b>Signal Persistence:</b><br>"
        f"Mean AC (Lag 1-5): {format_finite(mean_ac, '.4f')}<br>"
        f"Half-life: {half_life_display}{half_life_unit}"
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

    Dated components use their existing mean-of-defined-dates reducer by bucket
    and observation lag, with defined-date counts on hover and blank missing
    cells. Legacy components retain their original horizon summary inputs.

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
    if turnover_result.dated_turnover_ipc_base64 is not None:
        return _plot_dated_turnover_heatmap(turnover_result, theme, width, height)

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
                text=format_finite(val, ".1%"),
                showarrow=False,
                font={"size": 10, "color": text_color},
            )

    return fig


def plot_signal_turnover(
    turnover_result: TurnoverAnalysisResult,
    *,
    show_all_quantiles: bool = False,
    theme: str | None = None,
) -> list[go.Figure]:
    """Render every recorded membership and rank lag using the existing plots.

    Each component retains its own recorded lag order and clock. Empty results
    render availability notices, without inventing a lag or uncertainty estimate.
    Historical scalar maps remain supported by the original individual plots.
    """
    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult

    if not isinstance(turnover_result, TurnoverAnalysisResult):
        raise TypeError("turnover_result must be a TurnoverAnalysisResult")
    if turnover_result.dated_turnover_ipc_base64 is None:
        raise ValueError("complete turnover plots require recorded dated observations")
    if not isinstance(show_all_quantiles, bool):
        raise TypeError("show_all_quantiles must be bool")
    figures = []
    for table, rank in [("dated_turnover", False), ("dated_autocorrelation", True)]:
        lags = turnover_result.get_dataframe(table)["lag"].unique(maintain_order=True).to_list()
        for lag in lags or [None]:
            if rank:
                figure = plot_autocorrelation(turnover_result, lag=lag, theme=theme)
            else:
                figure = plot_top_bottom_turnover(
                    turnover_result, lag=lag, show_all_quantiles=show_all_quantiles, theme=theme
                )
            figure.update_layout(
                width=None,
                autosize=True,
                font={"size": 14},
                title_font={"size": 16},
                legend={"font": {"size": 14}},
            )
            figure.update_xaxes(tickfont_size=14, title_font_size=14)
            figure.update_yaxes(tickfont_size=14, title_font_size=14)
            figure.update_annotations(font_size=14)
            for annotation in figure.layout.annotations:
                if annotation.text == "Rank autocorrelation unavailable":
                    annotation.text = "Rank autocorrelation<br>unavailable"
            if (figure.layout.meta or {}).get("clock_source") == "supplied":
                figure.update_xaxes(
                    title_text="Supplied clock order"
                    if figure.layout.xaxis.type == "category"
                    else "Supplied clock date"
                )
            figures.append(figure)
    return figures


def generate_signal_turnover_report(
    turnover_result: TurnoverAnalysisResult,
    *,
    output_file: str | Path | None = None,
    show_all_quantiles: bool = False,
    theme: str | None = None,
) -> str:
    """Compose owned persistence tables and every lag, then publish atomically.

    The result's existing JSON/Arrow views retain full-precision observations,
    counts and method metadata. Display rounding does not change those tables.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html
    from ml4t.diagnostic.visualization.signal.event_plots import _publish_event_html

    figures = plot_signal_turnover(
        turnover_result, show_all_quantiles=show_all_quantiles, theme=theme
    )
    sections = [
        {
            "title": "Methods and availability",
            "text": (
                "<p>Turnover is entrants / current membership. Rank autocorrelation uses "
                "full-date average ranks and paired Pearson correlation. Lags count "
                "observations on each recorded clock, not inferred days or sessions. "
                "Undefined estimates remain missing. No confidence interval or half-life "
                "is inferred. Export full-precision observations with the result's JSON "
                "and get_dataframe methods.</p>"
            ),
        }
    ]
    for name, title in [
        ("turnover", "All-quantile mean turnover"),
        ("autocorrelation", "Mean rank autocorrelation"),
        ("dated_turnover", "Dated membership observations"),
        ("dated_autocorrelation", "Dated rank observations"),
    ]:
        table = (
            turnover_result.get_dataframe(name)
            .to_pandas()
            .to_html(index=False, na_rep="N/A", float_format=lambda value: f"{value:.6g}", border=0)
        )
        sections.append(
            {
                "title": title,
                "text": f'<div style="overflow-x: auto; white-space: nowrap;">{table}</div>',
            }
        )
    for index, figure in enumerate(figures):
        clock = escape(str((figure.layout.meta or {}).get("clock_source", "unavailable")))
        sections.append(
            {
                "title": str(figure.layout.title.text),
                "figure_index": index,
                "text": f"<p>Recorded observation clock: {clock}. Gaps are not zero estimates.</p>",
            }
        )
    html = combine_figures_to_html(
        figures, title="Signal turnover and rank persistence", sections=sections, theme=theme
    )
    return _publish_event_html(html, output_file)
