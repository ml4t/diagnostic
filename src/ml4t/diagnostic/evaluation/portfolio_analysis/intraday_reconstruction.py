"""Explicit cash-price account replay, reusing owned fills and holdings calculations."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from .fifo_inputs import clock, instant
from .holdings import HoldingsAnalysisResult, analyze_holdings
from .transactions import _clock

if TYPE_CHECKING:
    from .fifo import RoundTripResult


def _number(value: Any) -> Fraction:
    if isinstance(value, bool) or not isfinite(value):
        raise ValueError("cash-price amounts must be finite real numbers")
    return Fraction(str(value))


def _float(value: Fraction | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not isfinite(result):
        raise ValueError("cash-price account value cannot be represented as finite Float64")
    return result


def reconstruct_intraday_exposure(
    result: RoundTripResult,
    *,
    opening_cash: float | int | None,
    cash_model: Literal["cash_price"],
    opening_as_of: Any,
    calendar: str | CalendarUnset | None,
) -> HoldingsAnalysisResult:
    if cash_model != "cash_price":
        raise ValueError("cash_model='cash_price' explicitly selects full-notional cash accounting")
    cash = None if opening_cash is None else _number(opening_cash)
    feed = FeedSpec.from_any(result.metadata.get("feed_spec", {"time_contract_version": 2}))
    times = clock(result.executions["timestamp"], "execution timestamp", feed.timezone)
    aware = times.tz is not None
    declared_start = result.metadata["opening_as_of"]
    if opening_as_of is not None and declared_start is not None:
        supplied = instant(opening_as_of, "opening_as_of", feed.timezone, aware)
        declared = instant(declared_start, "opening_as_of", feed.timezone, aware)
        if supplied != declared:
            raise ValueError("opening_as_of must equal the complete opening inventory anchor")
    start_value = declared_start if declared_start is not None else opening_as_of
    if start_value is None or result.metadata["as_of"] is None:
        raise ValueError("cash-price reconstruction requires explicit opening_as_of and as_of")
    start = instant(start_value, "opening_as_of", feed.timezone, aware)
    end = instant(result.metadata["as_of"], "as_of", feed.timezone, aware)
    if end < start:
        raise ValueError("as_of cannot precede the cash anchor")
    selection = calendar
    if selection is UNSET_CALENDAR and result.metadata.get("calendar_source") == "argument":
        selection = result.metadata["calendar_argument"]
    resolved = feed.resolve_calendar(calendar=selection)
    quantities: dict[Any, Fraction] = defaultdict(Fraction)
    multipliers: dict[Any, Fraction] = {}

    def multiplier(row: dict[str, Any]) -> Fraction:
        value = _number(row["multiplier"])
        asset = row["asset"]
        if asset in multipliers and multipliers[asset] != value:
            raise ValueError("cash-price replay requires a consistent multiplier per asset")
        multipliers[asset] = value
        return value

    if result.opening_lots is not None:
        for row in result.opening_lots.to_dicts():
            multiplier(row)
            quantities[row["asset"]] += _number(row["quantity"])
    fills: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for timestamp, row in zip(times, result.executions.to_dicts(), strict=True):
        if start <= timestamp <= end:
            multiplier(row)
            fills[timestamp.value].append(row)
    quotes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if result.marks is not None:
        mark_times = clock(result.marks["timestamp"], "mark timestamp", feed.timezone, aware)
        for timestamp, row in zip(mark_times, result.marks.to_dicts(), strict=True):
            if timestamp <= end and row["asset"] in multipliers:
                quotes[timestamp.value].append(row)
    latest: dict[Any, tuple[Fraction, int]] = {}
    for timestamp in sorted(value for value in quotes if value <= start.value):
        for row in quotes[timestamp]:
            latest[row["asset"]] = (_number(row["price"]), timestamp)
    states, holdings = [], []

    def snapshot(timestamp: int, kind: str, record_id: Any = None) -> None:
        observation = len(states)
        marked = Fraction()
        for asset, quantity in quantities.items():
            if not quantity:
                continue
            if asset not in latest:
                raise ValueError(f"held asset {asset!r} requires a known-at supplied mark")
            price, quote_time = latest[asset]
            value = quantity * price * multipliers[asset]
            marked += value
            holdings.append(
                {
                    "date": timestamp,
                    "observation_id": observation,
                    "asset": asset,
                    "value": _float(value),
                    "value_exact": str(value),
                    "quantity": _float(quantity),
                    "quantity_exact": str(quantity),
                    "quantity_sign": 1 if quantity > 0 else -1,
                    "price": _float(price),
                    "price_exact": str(price),
                    "multiplier_exact": str(multipliers[asset]),
                    "mark_timestamp": quote_time,
                    "mark_age_ns": timestamp - quote_time,
                }
            )
        equity = None if cash is None else cash + marked
        states.append(
            {
                "date": timestamp,
                "observation_id": observation,
                "cash": _float(cash),
                "equity": _float(equity),
                "cash_exact": None if cash is None else str(cash),
                "equity_exact": None if equity is None else str(equity),
                "event_kind": kind,
                "execution_record_id": record_id,
            }
        )

    snapshot(start.value, "opening")
    events = sorted({start.value, end.value, *fills, *(t for t in quotes if t > start.value)})
    for timestamp in events:
        if timestamp != start.value:
            for row in quotes.get(timestamp, []):
                latest[row["asset"]] = (_number(row["price"]), timestamp)
        for row in fills.get(timestamp, []):
            amount = _number(row["quantity"])
            quantities[row["asset"]] += amount
            fee = row["commission"]
            cash = (
                None
                if cash is None or fee is None
                else (
                    cash - amount * _number(row["price"]) * multipliers[row["asset"]] - _number(fee)
                )
            )
            snapshot(timestamp, "execution", row["record_id"])
        changed_mark = any(quantities[row["asset"]] for row in quotes.get(timestamp, []))
        if (
            timestamp != start.value
            and not fills.get(timestamp)
            and (timestamp == end.value or changed_mark)
        ):
            snapshot(timestamp, "terminal" if timestamp == end.value else "mark")
    dtype = pl.Datetime("ns", str(start.tz) if aware else None)
    account = pl.DataFrame(
        states, schema_overrides={"execution_record_id": result.executions.schema["record_id"]}
    ).with_columns(pl.col("date").cast(dtype))
    schema = {
        "date": pl.Int64,
        "observation_id": pl.Int64,
        "asset": result.executions.schema["asset"],
        "value": pl.Float64,
        "value_exact": pl.String,
        "quantity": pl.Float64,
        "quantity_exact": pl.String,
        "quantity_sign": pl.Int8,
        "price": pl.Float64,
        "price_exact": pl.String,
        "multiplier_exact": pl.String,
        "mark_timestamp": pl.Int64,
        "mark_age_ns": pl.Int64,
    }
    positions = pl.DataFrame(holdings, schema=schema).with_columns(
        pl.col("date", "mark_timestamp").cast(dtype)
    )
    dates = account["date"]
    sessions = _clock(dates, resolved, None, executions=True, timezone=feed.timezone)
    prepared = analyze_holdings(
        positions,
        account,
        dates,
        complete_snapshots=True,
        snapshot_kind="intraday",
        observation_ids=account["observation_id"],
    )
    prepared.timeline = prepared.timeline.join(
        account.select(
            "observation_id", "cash_exact", "equity_exact", "event_kind", "execution_record_id"
        ),
        on="observation_id",
        how="left",
        maintain_order="left",
    )
    prepared.positions = prepared.positions.join(
        positions.select(
            "date",
            "observation_id",
            "asset",
            "quantity_exact",
            "quantity_sign",
            "price",
            "value_exact",
            "price_exact",
            "multiplier_exact",
            "mark_timestamp",
            "mark_age_ns",
        ),
        on=["date", "observation_id", "asset"],
        how="left",
        maintain_order="left",
    )
    if resolved is not None:
        prepared.timeline = prepared.timeline.with_columns(sessions.rename("session"))
    prepared.metadata.update(
        method="explicit_cash_price_reconstruction",
        cash_model=cash_model,
        feed_spec=feed.to_dict(),
        calendar=resolved,
        opening_as_of=start.isoformat(),
        as_of=end.isoformat(),
        same_instant_order="original execution row order; observation_id identifies each state",
        quotes="last supplied mark at/before each instant; same-instant marks known before fills; fill prices are not substituted for marks",
        sampling="opening, after each actual fill, mark-only instants, terminal if not already sampled",
        peak_semantics="maximum over reconstructed event observations, not continuous-time maximum",
        average_semantics="equal-weight event observations, not elapsed-time weighting",
        costs="opening cash already includes historical fees; actual commission once; no extra slippage deduction",
        unknown_cash="missing opening cash or any missing commission leaves subsequent cash/NAV unknown",
        unsupported_accounting="cash-price only; no inferred futures margin, FX, dividends, deposits or corporate actions",
        position_counts="nonzero exact quantity, not nonzero marked value; long/short counts follow quantity sign even for zero/negative prices",
        exposure_signs="marked cash value determines long/short exposure; this need not match quantity direction at negative prices",
        numeric_projection="numeric account/mark fields are Float64; *_exact fields retain decimal-input rational quantities, prices, multipliers and cash values",
        calendar_source="argument"
        if calendar is not UNSET_CALENDAR
        else result.metadata.get("calendar_source", "default"),
    )
    return prepared
