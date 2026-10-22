"""Observed intraday holdings on their own supplied clock, without return invention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from .frame_inputs import _frame_declarations, _merge_cash, _positions, _valuation_snapshot
from .holdings import HoldingsAnalysisResult, analyze_holdings
from .transactions import _clock


def analyze_intraday_exposure(
    positions: pl.DataFrame | pd.DataFrame,
    valuations: pl.DataFrame | pd.DataFrame | None = None,
    *,
    complete_snapshots: bool,
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    timestamp_col: str | None = None,
    entity_col: str | None = None,
    positions_format: Literal["auto", "long", "wide"] = "auto",
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
) -> HoldingsAnalysisResult:
    """Analyze complete observed marked holdings independently of return observations.

    Shared position/account normalization and holdings/occupancy calculations are
    reused. Values are reported signed cash marks, not guessed quantities or
    interpolated quotes. Missing NAV remains unknown even when cash and positions
    are available. Dates/session labels cannot establish intraday observations.
    Calendar resolution uses explicit argument, metadata, then NYSE; explicit
    None disables schedules. Validation maps sessions without replacing original
    instants or inferring a missing-bar grid. Means/time-in-market describe the
    supplied sample, not elapsed-time weighting or a continuous exposure maximum.
    """
    if complete_snapshots is not True:
        raise ValueError(
            "complete_snapshots=True must explicitly declare complete intraday holdings"
        )
    if positions_format not in ("auto", "long", "wide"):
        raise ValueError("positions_format must be auto, long or wide")
    feed, timestamp, entity = _frame_declarations(feed_spec, timestamp_col, entity_col)
    owned, cash = _positions(positions, timestamp, entity, positions_format)
    dates = owned["date"].unique().sort()
    if not isinstance(dates.dtype, pl.Datetime) or feed.timestamp_semantics == "session_label":
        raise ValueError("intraday snapshots require actual Datetime instants, not session labels")
    if dates.is_empty():
        raise ValueError("at least one complete observed intraday snapshot is required")
    effective_calendar = feed.resolve_calendar(calendar=calendar)
    sessions = _clock(dates, effective_calendar, feed.timestamp_semantics, timezone=feed.timezone)
    account = _merge_cash(_valuation_snapshot(valuations, timestamp), cash)
    if account is None:
        account = pl.DataFrame({"date": dates})
    result = analyze_holdings(
        owned, account, dates, complete_snapshots=True, snapshot_kind="intraday"
    )
    if effective_calendar is not None:
        result.timeline = result.timeline.with_columns(sessions.rename("session"))
    wire = feed.to_dict()
    declared = "calendar" in wire and (
        feed.time_contract_version == 2 or wire["calendar"] is not None
    )
    result.metadata.update(
        method="observed_complete_snapshots",
        feed_spec=wire,
        calendar=effective_calendar,
        calendar_source="argument"
        if calendar is not UNSET_CALENDAR
        else "feed_spec"
        if declared
        else "default",
        peak_semantics="maximum over supplied observations, not a continuous-time maximum",
        average_semantics="equal-weight supplied observations, not elapsed-time weighting",
        input_clock="original intraday instants; independent of any return-series clock",
        calendar_validation="session membership only; no expected-grid inference",
        positions_unit="reported signed marked cash value; no quantity or mark inference",
        unreported_NAV="unknown, never inferred from holdings/cash",
    )
    if calendar is not UNSET_CALENDAR:
        result.metadata["calendar_argument"] = calendar
    return result
