"""Familiar analytical report inputs composed from owned native analytics."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import positive_integer

from . import perf_attrib
from .capacity_sheet import create_capacity_tear_sheet

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization.portfolio import PortfolioTearSheet


def create_round_trip_tear_sheet(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    sector_mappings: Mapping[Any, str] | pd.Series | None = None,
    estimate_intraday: bool | np.bool_ | str | None = "infer",
    return_fig: bool = False,
    *,
    as_of: Any = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
) -> PortfolioTearSheet | None:
    """Compose the original daily gross cash-price, flat-opening round-trip route.

    Returns a native Plotly PortfolioTearSheet when return_fig=True; otherwise
    displays it and returns None. There is no invented portfolio-return summary.
    Intraday estimation is retrospective and may select the legacy NAV profile,
    but only supplied ending positions can request hypothetical terminal closes.
    These closes stay separately flagged, never actual executions or realization.
    Sector panels preserve the explicit legacy whole-row complete-case mask.
    """
    from ml4t.diagnostic.visualization.portfolio import create_round_trip_dashboard

    from . import round_trips, utils
    from .txn import _dated

    _dated(returns, "returns")
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(return_fig, bool | np.bool_):
        raise TypeError("return_fig must be a boolean")
    selected = utils.check_intraday(estimate_intraday, returns, positions, transactions)
    closed = round_trips.add_closing_transactions(
        positions, transactions, as_of=as_of, feed_spec=feed_spec, calendar=calendar
    )
    native, labels = round_trips._native_round_trips(closed)
    synthetic = bool(native.metadata.get("legacy_synthetic_rows"))
    # This is the reference's explicit daily denominator, not an inferred native NAV.
    beginning_nav = selected.sum(axis="columns") / (1 + returns)
    original_nav_rows = len(beginning_nav)
    if not beginning_nav.index.is_unique:
        # Equal peak candidates are exposure observations, not extra closing trades.
        grouped = beginning_nav.groupby(level=0, sort=False)
        if (grouped.nunique(dropna=False) > 1).any():
            raise ValueError("tied daily estimates disagree on the selected NAV denominator")
        beginning_nav = grouped.agg(lambda values: values.iloc[0])
    display_labels = {
        index: str(getattr(label, "symbol", label)) for index, label in enumerate(labels)
    }
    native.metadata["compatibility_profile"] = {
        "method": "legacy_pyfolio_round_trip_sheet",
        "asset_labels": list(labels),
        "initial_state": "flat opening plus complete actual transactions",
        "costs": "gross cash-price profile; legacy ignores commissions/slippage fields",
        "closing_inventory": "supplied original ending positions, never an estimated peak",
        "nav_method": "selected daily position+cash sum divided by1+supplied daily return; no fabricated returns",
        "selected_positions_kind": "estimated_intraday"
        if selected is not positions
        else "supplied",
    }
    report = create_round_trip_dashboard(
        None,
        native,
        pnl="gross",
        return_basis="nav",
        portfolio_value=beginning_nav,
        calendar=None,
        alignment="civil_date",
        include_synthetic=synthetic,
        statistics_method="legacy_pyfolio",
        probability_method="legacy_zero_prior",
        lifetime_sample=16,
        lifetime_linewidth=18,
        asset_labels=display_labels,
        holding_duration_method="legacy_whole_days",
        return_display="percent",
    )
    statistics = report.round_trip_statistics
    assert statistics is not None
    statistics.metadata["intraday_estimation"] = dict(selected.attrs.get("diagnostic", {}))
    statistics.metadata["denominator"] = {
        "method": native.metadata["compatibility_profile"]["nav_method"],
        "date": [value.isoformat() for value in beginning_nav.index],
        "values": [None if pd.isna(value) else float(value) for value in beginning_nav],
        "source": native.metadata["compatibility_profile"]["selected_positions_kind"],
        "equal_daily_candidate_duplicates_collapsed": original_nav_rows - len(beginning_nav),
    }
    if statistics.observations.height < 5:
        warnings.warn(
            "Fewer than 5 round-trip observations; retained with explicit sample/uncertainty coverage.",
            UserWarning,
            stacklevel=2,
        )
    if sector_mappings is not None:
        _sector_panel(
            report, round_trips._project_round_trips(native, labels, beginning_nav), sector_mappings
        )
    if return_fig:
        return report
    report.show()
    return None


def _sector_panel(
    report: PortfolioTearSheet, trades: pd.DataFrame, sector_mappings: Mapping[Any, str] | pd.Series
) -> None:
    import plotly.graph_objects as go

    from .round_trip_plotting import _attribution
    from .round_trips import apply_sector_mappings_to_round_trips

    selected = apply_sector_mappings_to_round_trips(trades, sector_mappings)
    summary, labels = _attribution(selected)
    rows = [
        {
            "scope": "sector",
            "asset": None,
            "sector": str(labels[int(row["asset"])]),
            "pnl": row["pnl"],
            "pnl_fraction": row["pnl_fraction"],
            "n_observations": row["n_observations"],
        }
        for row in summary.to_dicts()
    ]
    attribution = report.round_trip_attribution
    assert attribution is not None
    sectors = pl.DataFrame(rows, schema=attribution.schema)
    report.round_trip_attribution = pl.concat([attribution, sectors])
    figure = go.Figure(report.figures["Round-trip Asset Attribution"])
    figure.data = []
    figure.add_trace(
        go.Bar(
            x=sectors["pnl_fraction"].to_list(),
            y=sectors["sector"].to_list(),
            orientation="h",
            customdata=sectors["pnl"].to_list(),
            name="Gross cash share",
            hovertemplate="%{y}: %{x:.3f}; cash %{customdata}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Sector")
    figure.update_layout(
        height=max(360, 120 + len(rows) * 28),
        meta={
            "analysis": "prepared_round_trips",
            "description": "Explicit legacy sector complete-case projection: all missing fields remove the row, even when cash PnL is known. Signed sector shares divide by selected total cash, not NAV. Native label-only attribution retains known cash independently.",
            "sector_projection": dict(selected.attrs["diagnostic_sector_projection"]),
        },
    )
    if not rows:
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        figure.add_annotation(
            text="Unavailable for this sample.<br>See sector complete-case coverage.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font_size=13,
        )
    report.figures["Round-trip Sector Attribution"] = figure
    assert report.round_trip_statistics is not None
    report.round_trip_statistics.metadata["sector_projection"] = dict(
        selected.attrs["diagnostic_sector_projection"]
    )


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


__all__ = [
    "create_capacity_tear_sheet",
    "create_perf_attrib_tear_sheet",
    "create_round_trip_tear_sheet",
]
