"""Render retained signal returns using the existing empirical quantile charts."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.utils.formatting import format_finite
from ml4t.diagnostic.visualization.core import create_base_figure
from ml4t.diagnostic.visualization.signal.event_plots import _publish_event_html
from ml4t.diagnostic.visualization.signal.quantile_plots import (
    plot_cumulative_returns,
    plot_quantile_returns_bar,
    plot_quantile_returns_violin,
    plot_spread_timeseries,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.returns_report import SignalReturnsResult


def plot_signal_returns(
    result: SignalReturnsResult,
    *,
    theme: str | None = None,
    rolling_window: int = 21,
    bandwidth: float = 1.0,
    ylim_percentiles: tuple[float, float] | None = None,
) -> list[go.Figure]:
    """Render all retained horizons and optional group means without resampling.

    Bars aggregate equally weighted date means; violins retain those actual
    dated means. Spread bands use propagated sample standard error, not a
    confidence interval. Wealth compounds raw outcomes, never display rates.
    """
    from ml4t.diagnostic.signal._prepared import positive_count

    positive_count(rolling_window, "rolling_window")
    figures = []
    metadata = result.metadata
    for column in metadata["return_columns"]:
        model = result.to_quantile_result(column)
        selected = [
            ("quantile_mean", plot_quantile_returns_bar(model, show_spread=False, theme=theme)),
            (
                "quantile_distribution",
                plot_quantile_returns_violin(model, theme=theme, ylim_percentiles=ylim_percentiles),
            ),
        ]
        if model.n_quantiles >= 2:
            spread = plot_spread_timeseries(
                model,
                rolling_window=rolling_window,
                bandwidth=bandwidth,
                theme=theme,
                upper_quantile=metadata["upper_quantile"],
                lower_quantile=metadata["lower_quantile"],
                exposure_scale=metadata["spread_exposure_scale"],
            )
            spread.layout.annotations = ()
            selected.append(("quantile_spread", spread))
        path = (
            result.get_table("factor_paths").filter(pl.col("return_column") == column).sort("date")
        )
        if not path.is_empty():
            timestamps = [pd.Timestamp(value) for value in path["date"].to_pandas()]
            dates = [value.isoformat() for value in timestamps]
            factor = create_base_figure(
                title="Factor-weighted wealth",
                xaxis_title="Date",
                yaxis_title="Wealth multiple",
                theme=theme,
            )
            factor.add_trace(
                go.Scatter(
                    x=dates,
                    y=path["wealth"].to_list(),
                    mode="lines",
                    name=column,
                    customdata=path.select("weighted_return", "n_obs").rows(),
                    hovertemplate="Date: %{x}<br>Wealth: %{y}<br>Supplied return: %{customdata[0]}<br>Finite assets: %{customdata[1]}<extra>%{fullData.name}</extra>",
                )
            )
            if any(value.value % 1_000_000 for value in timestamps):
                factor.update_xaxes(
                    type="category",
                    tickmode="array",
                    tickvals=dates,
                    ticktext=[str(index + 1) for index in range(len(dates))],
                    title_text="Observed date order",
                )
            selected.extend(
                [
                    ("factor_wealth", factor),
                    (
                        "quantile_cumulative_returns",
                        plot_cumulative_returns(model, show_spread=False, theme=theme),
                    ),
                ]
            )
        for analysis, figure in selected:
            extra = dict(figure.layout.meta or {})
            extra.update(analysis=analysis, return_column=column)
            figure.update_layout(meta=extra)
            if analysis != "quantile_mean":
                figure.update_layout(
                    title_text=escape(str(figure.layout.title.text).split(" (")[0])
                    + "<br>("
                    + "<br>".join(
                        escape(column[index : index + 24]) for index in range(0, len(column), 24)
                    )
                    + ")"
                )
            figures.append(figure)
        groups = result.get_table("group_quantile_statistics").filter(
            pl.col("return_column") == column
        )
        for group in groups["group"].unique(maintain_order=True):
            view = result.to_quantile_result(column, group=group)
            figure = plot_quantile_returns_bar(view, show_spread=False, theme=theme)
            figure.update_layout(
                title_text="Group quantile means<br>" + escape(str(group)) + " / " + escape(column),
                meta={
                    **dict(figure.layout.meta or {}),
                    "analysis": "group_quantile_mean",
                    "group": group,
                    "return_column": column,
                },
            )
            figures.append(figure)
    for figure in figures:
        figure.update_layout(
            width=None,
            autosize=True,
            height=520,
            font_size=14,
            title_font_size=16,
            legend={"orientation": "h", "x": 0, "y": -0.25, "font_size": 14},
        )
        if figure.layout.meta["analysis"] not in {"quantile_mean", "group_quantile_mean"}:
            figure.update_layout(margin={"l": 65, "r": 25, "t": 90, "b": 145})
        figure.update_xaxes(tickfont_size=14, title_font_size=14)
        figure.update_yaxes(tickfont_size=14, title_font_size=14)
        figure.update_annotations(font_size=14)
    return figures


def generate_signal_returns_report(
    result: SignalReturnsResult,
    *,
    output_file: str | Path | None = None,
    theme: str | None = None,
    rolling_window: int = 21,
    bandwidth: float = 1.0,
    ylim_percentiles: tuple[float, float] | None = None,
) -> str:
    """Publish the validated native returns workflow with owned regression tables."""
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    figures = plot_signal_returns(
        result,
        theme=theme,
        rolling_window=rolling_window,
        bandwidth=bandwidth,
        ylim_percentiles=ylim_percentiles,
    )
    sections = []
    for name in ("regression_statistics", "quantile_statistics", "group_quantile_statistics"):
        frame = result.get_table(name)
        if frame.is_empty():
            continue
        table = frame.to_pandas().to_html(
            index=False,
            escape=True,
            na_rep="Undefined",
            border=0,
            float_format=lambda value: format_finite(value, ".6g"),
        )
        sections.append(
            {
                "title": name.replace("_", " ").capitalize(),
                "text": '<div style="overflow-x:auto;font-size:14px">' + table + "</div>",
            }
        )
    for index, figure in enumerate(figures):
        analysis = figure.layout.meta["analysis"]
        descriptions = {
            "quantile_mean": "Bars show equal-date mean returns and sample standard error, not pooled asset inference. Raw and display statistics remain separately available in the tables. Missing errors are not zero errors.",
            "group_quantile_mean": "Group bars supplement the global sample. Each defined date has equal weight within its group and bucket; actual date counts remain in the table and hover. No dated group paths are inferred.",
            "quantile_distribution": "Violins retain the actual defined date/bucket means. No synthetic observations are drawn. Plotly density presentation is not a claim of Matplotlib kernel-density equivalence; percentile limits only change the visible range.",
            "quantile_spread": f"The spread uses {result.metadata['spread_exposure']} exposure and the retained upper/lower buckets. The {rolling_window}-observation rolling mean requires complete finite windows. Bands are {bandwidth:g} times independently propagated sample errors, not confidence coverage.",
            "factor_wealth": "The dimensionless wealth multiple compounds raw supplied-horizon returns and is not self-financing account P&L. Missing outcomes use the declared flat-wealth policy, not observed zero returns. No calendar or executed holdings are inferred.",
            "quantile_cumulative_returns": "Cumulative returns are wealth minus one, not wealth multiples. Paths compound raw date/bucket means, never display-converted rates. Exact timestamp keys remain attached; sub-millisecond keys use observation-order axes rather than browser date rounding.",
        }
        sections.append(
            {
                "title": str(analysis).replace("_", " ").capitalize(),
                "figure_index": index,
                "text": descriptions[analysis],
            }
        )
    sections.insert(
        0,
        {
            "title": "Methods and units",
            "text": "Returns are decimal fractions; beta and wealth multiples are dimensionless. Annual alpha compounds the intercept using each explicitly supplied horizon scale. Retained tables expose the exact joint samples, counts and undefined statuses. Uncomputed inference is not fabricated.",
        },
    )
    html = combine_figures_to_html(
        figures,
        title="Signal returns analysis",
        sections=sections,
        theme=theme,
        include_plotlyjs=True,
    )
    return _publish_event_html(html, output_file)
