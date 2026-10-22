"""Familiar analytical report inputs with explicit Plotly/HTML presentation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import pandas as pd

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

from . import perf_attrib

if TYPE_CHECKING:
    import plotly.graph_objects as go

FACTOR_PARTITIONS = MappingProxyType(
    {
        "style": ("momentum", "size", "value", "reversal_short_term", "volatility"),
        "sector": (
            "basic_materials",
            "consumer_cyclical",
            "financial_services",
            "real_estate",
            "consumer_defensive",
            "health_care",
            "utilities",
            "communication_services",
            "energy",
            "industrials",
            "technology",
        ),
    }
)


def create_perf_attrib_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    pos_in_dollars: bool = True,
    factor_partitions: Mapping[str, Iterable[str]] | None = FACTOR_PARTITIONS,
    return_fig: bool = False,
    *,
    output_file: str | Path | None = None,
    theme: str | None = None,
    periods_per_year: int = 252,
) -> str | list[go.Figure]:
    """Render all attribution tables and partitioned component/exposure panels.

    The default returns offline HTML, or a saved path when output_file is set,
    instead of triggering a notebook display. return_fig=True returns the
    ordered list of Plotly figures, not one Matplotlib Figure. Tables remain
    in HTML output. An explicit output_file is honored in either return mode.
    None selects all factors; an empty partition mapping selects no factor
    panels. Unknown factors yield a visibly empty panel, not invented exposure.
    """
    from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html

    scale = positive_integer(periods_per_year, "periods_per_year")
    if not isinstance(return_fig, bool):
        raise TypeError("return_fig must be a boolean")
    partitions = None
    if factor_partitions is not None:
        if not isinstance(factor_partitions, Mapping):
            raise TypeError("factor_partitions must map panel names to factor sequences or be None")
        partitions = {}
        for name, values in factor_partitions.items():
            if isinstance(values, str | bytes) or not isinstance(values, Iterable):
                raise TypeError("factor_partitions values must be factor sequences, not text")
            partitions[str(name)] = tuple(values)
    exposures, table = perf_attrib.perf_attrib(
        returns, positions, factor_returns, factor_loadings, transactions, pos_in_dollars
    )
    summary, factors = perf_attrib.create_perf_attrib_stats(
        table, exposures, periods_per_year=scale
    )
    figures = [perf_attrib.plot_returns(table, theme=theme)]
    sections = [
        {
            "title": "Performance relative to supplied common risk factors",
            "text": f"<p>Annualized statistics use {scale} observations per year. "
            "Specific returns are unexplained returns, not evidence of skill. "
            "Component growth paths compound separately and are not an additive "
            "wealth-linked decomposition.</p>"
            '<div class="attribution-statistics" style="overflow-x:auto">'
            + summary.to_frame("Value").to_html(float_format=lambda value: f"{value:.6g}")
            + "</div><p>Scroll the factor statistics horizontally on narrow displays.</p>"
            '<div class="attribution-statistics" style="overflow-x:auto">'
            + factors.to_html(
                classes="attribution-factor-statistics", float_format=lambda value: f"{value:.6g}"
            )
            + "</div>",
            "figure_index": 0,
        }
    ]
    selected = [(None, None)] if partitions is None else list(partitions.items())
    for exposure_panel in (False, True):
        for group, members in selected:
            source = exposures if exposure_panel else table
            columns = source.columns if members is None else source.columns.intersection(members)
            title = (
                (
                    f"Daily {group} factor exposures"
                    if exposure_panel
                    else f"Cumulative common {group} returns attribution"
                )
                if group is not None
                else (
                    "Daily risk factor exposures"
                    if exposure_panel
                    else "Cumulative common returns attribution"
                )
            )
            plot = (
                perf_attrib.plot_risk_exposures
                if exposure_panel
                else perf_attrib.plot_factor_contribution_to_perf
            )
            figure = plot(source[columns], title=title, theme=theme)
            if not len(figure.data):
                figure.add_annotation(
                    text="No matching supplied factors",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font_size=14,
                )
            sections.append({"title": escape(title), "figure_index": len(figures)})
            figures.append(figure)
    html = combine_figures_to_html(
        figures,
        title="Supplied holdings factor attribution",
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_plotlyjs=True,
    )
    return figures if return_fig else html
