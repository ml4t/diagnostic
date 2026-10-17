"""Compose actual information-analysis tables and existing IC charts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme
from ml4t.diagnostic.visualization.signal.event_plots import _publish_event_html
from ml4t.diagnostic.visualization.signal.ic_plots import (
    plot_ic_histogram,
    plot_ic_qq,
    plot_ic_ts,
    plot_monthly_ic_heatmap,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.information import SignalInformationResult


def plot_signal_information(
    result: SignalInformationResult,
    *,
    theme: str | None = None,
    theoretical_dist: Any = None,
) -> list[go.Figure]:
    """Render every supplied horizon, with actual group or civil-month aggregates.

    The Q-Q panel explicitly selects the fitted/standardized convention. Existing
    plot_ic_qq keeps its native default for existing callers. No KDE or fitted
    quantiles are synthesized for empty or degenerate samples.
    """
    model = result.to_ic_result()
    figures = []
    for column in result.metadata["return_columns"]:
        figures.extend(
            [
                plot_ic_ts(
                    model,
                    period=column,
                    rolling_window=22,
                    min_periods=22,
                    show_significance=False,
                    theme=theme,
                ),
                plot_ic_histogram(model, period=column, bins=25, theme=theme),
                plot_ic_qq(
                    model,
                    period=column,
                    fitted=True,
                    theoretical_dist=theoretical_dist,
                    theme=theme,
                ),
            ]
        )
    if result.metadata["by_group"]:
        groups = result.get_table("mean_group_ic")
        figure = go.Figure()
        for column in result.metadata["return_columns"]:
            sample = groups.filter(pl.col("return_column") == column)
            figure.add_trace(
                go.Bar(
                    x=sample["group"].to_list(),
                    y=sample["mean_ic"].to_list(),
                    name=column,
                    customdata=sample.select("n_defined", "n_dates").rows(),
                    hovertemplate="Group %{x}<br>Mean IC %{y}<br>Defined dates %{customdata[0]} / %{customdata[1]}<extra>%{fullData.name}</extra>",
                )
            )
        figure.update_layout(**get_theme_config(validate_theme(theme))["layout"])
        figure.update_layout(
            title="Mean IC by group",
            barmode="group",
            xaxis_title="Group",
            yaxis_title="Mean IC",
            meta={"analysis": "group_ic_means"},
        )
        figures.append(figure)
    else:
        figures.extend(
            plot_monthly_ic_heatmap(model, period=column, theme=theme)
            for column in result.metadata["return_columns"]
        )
    for figure in figures:
        figure.update_layout(
            width=None,
            autosize=True,
            height=500,
            font={"size": 14},
            title_font_size=16,
            showlegend=True,
            legend={"orientation": "h", "y": -0.25, "x": 0, "font": {"size": 14}},
            margin={"l": 65, "r": 25, "t": 45, "b": 120},
        )
        figure.update_xaxes(tickfont_size=14, title_font_size=17)
        figure.update_yaxes(tickfont_size=14, title_font_size=17)
        figure.update_annotations(font_size=14)
    return figures


def generate_signal_information_report(
    result: SignalInformationResult,
    *,
    output_file: str | Path | None = None,
    theme: str | None = None,
    theoretical_dist: Any = None,
) -> str:
    """Publish the owned information workflow, validating all panels before writing."""
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    table = (
        result.get_table("summary")
        .rename(
            {
                "return_column": "Horizon",
                "n_dates": "Dates",
                "n_defined": "Defined ICs",
                "mean_ic": "Mean IC",
                "std_ic": "Sample deviation",
                "risk_adjusted_ic": "Mean / deviation",
                "t_stat": "T statistic",
                "p_value": "p value",
                "pct_positive": "Positive fraction",
                "skew": "Skew",
                "kurtosis": "Excess kurtosis",
                "status": "Status",
            }
        )
        .to_pandas()
        .to_html(
            index=False,
            escape=True,
            na_rep="Undefined",
            border=0,
            float_format=lambda value: format_finite(value, ".6g"),
        )
    )
    table = table.replace("<table ", '<table style="font-size:14px;border-collapse:collapse" ', 1)
    table = table.replace("<th>", '<th style="padding:6px">').replace(
        "<td>", '<td style="padding:6px">'
    )
    sections = [
        {
            "title": "Information analysis",
            "text": f"Method: {result.metadata['method']}; date/group outcome demeaning: {result.metadata['group_adjust']}; group summary: {result.metadata['by_group']}; minimum paired assets: {result.metadata['min_obs']}. "
            "IC and mean/deviation are dimensionless. Inference is the naive two-sided one-sample t-test, not HAC. Constant or insufficient samples retain point values/counts but have undefined inference. "
            f'<div style="overflow-x:auto">{table}</div>',
        }
    ]
    figures = plot_signal_information(result, theme=theme, theoretical_dist=theoretical_dist)
    for index, figure in enumerate(figures):
        if index < 3 * len(result.metadata["return_columns"]):
            text = [
                "Points are actual dated ICs. Rolling means require all22finite ICs in each complete22-observation window. This is not a calendar month or a confidence band.",
                "The histogram retains actual defined ICs. KDE is unavailable for degenerate or insufficient samples; missing ICs are not replaced by zeros.",
                "Q-Q points use the explicitly fitted/standardized distribution. Empty, singleton or constant samples retain observed values but have no fitted quantiles.",
            ][index % 3]
        elif result.metadata["by_group"]:
            text = "Group means supplement the global analysis and give each defined date equal weight. Actual defined/total date counts are in the export and hover data. These are not group-neutral portfolio weights."
        else:
            text = "Means use defined ICs within each supplied civil year/month, not exchange sessions or expected bars. Missing observations are not zero ICs; actual counts remain in the export."
        sections.append(
            {
                "title": str(figure.layout.title.text),
                "figure_index": index,
                "text": text,
            }
        )
    html = combine_figures_to_html(
        figures, title="Signal information analysis", sections=sections, theme=theme
    )
    return _publish_event_html(html, output_file)
