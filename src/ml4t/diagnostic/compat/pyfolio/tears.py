"""Complete familiar round-trip workflow composed from owned native analytics."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

if TYPE_CHECKING:
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


__all__ = ["create_round_trip_tear_sheet"]
