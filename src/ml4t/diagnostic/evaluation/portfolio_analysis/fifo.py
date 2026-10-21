"""Exact-unit FIFO lot matching over actual owned executions, not broker simulation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

import pandas as pd
import polars as pl

from .fifo_inputs import clock, instant, mark_inventory, opening_inventory, price_dtype


@dataclass
class RoundTripResult:
    """Matched lot segments and remaining unmarked lots, with original executions.

    Cash calculations are Float64. Matching units use exact rational arithmetic
    on decimal input representations, avoiding both integer truncation and
    phantom residual lots. quantity_exact retains the exact rational units when
    the numeric quantity cannot represent a large integer or tiny fraction.
    Commission is allocated once; execution slippage is already in actual prices.
    """

    segments: pl.DataFrame
    open_lots: pl.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    executions: pl.DataFrame
    opening_lots: pl.DataFrame | None = None
    marks: pl.DataFrame | None = None
    synthetic_segments: pl.DataFrame | None = None


@dataclass
class _Lot:
    identifier: Any
    timestamp: int | None
    direction: int
    quantity: Fraction
    price: Fraction | None
    price_input: Any
    multiplier: Fraction
    commission: Fraction | None
    opening_id: Any = None


def _cash(value: Fraction | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("FIFO cash calculation exceeds finite Float64 account currency") from None
    if not isfinite(result):
        raise ValueError("FIFO cash calculation exceeds finite Float64 account currency")
    return result


def _known_sum(values: list[Fraction | None]) -> float | None:
    total = Fraction(0)
    for value in values:
        if value is None:
            return None
        total += value
    return _cash(total)


def _frame(
    records: list[dict[str, Any]], schema: dict[str, Any], timestamps: list[str], zone: str | None
) -> pl.DataFrame:
    frame = pl.DataFrame(records, schema=schema)
    return frame.with_columns(pl.col(column).cast(pl.Datetime("ns", zone)) for column in timestamps)


def reconstruct_round_trips(
    fills: pl.DataFrame | None,
    *,
    initial_state: str | None,
    opening_lots: pl.DataFrame | pd.DataFrame | None = None,
    opening_as_of: Any = None,
    marks: pl.DataFrame | pd.DataFrame | None = None,
    as_of: Any = None,
    synthetic_close: bool = False,
    timezone: str | None = None,
) -> RoundTripResult:
    """Match actual fills from a declared complete opening book, with optional marks."""
    if initial_state not in ("flat", "supplied"):
        raise ValueError("initial_state='flat' or 'supplied' must declare a complete opening book")
    if not isinstance(synthetic_close, bool):
        raise TypeError("synthetic_close must be a boolean")
    if initial_state == "flat" and (opening_lots is not None or opening_as_of is not None):
        raise ValueError("opening_lots/opening_as_of require initial_state='supplied'")
    if initial_state == "supplied" and (opening_lots is None or opening_as_of is None):
        raise ValueError("initial_state='supplied' requires opening_lots and opening_as_of")
    if (marks is not None or synthetic_close) and as_of is None:
        raise ValueError("marks and synthetic_close require an explicit as_of valuation instant")
    if fills is None:
        raise ValueError(
            "FIFO requires supplied transactions; absent fills are not an empty ledger"
        )
    clocks = clock(fills["timestamp"], "execution timestamp", timezone)
    aware = clocks.tz is not None
    cutoff = instant(as_of, "as_of", timezone, aware) if as_of is not None else None
    start = (
        instant(opening_as_of, "opening_as_of", timezone, aware)
        if opening_as_of is not None
        else None
    )
    if cutoff is not None and start is not None and cutoff < start:
        raise ValueError("as_of cannot precede opening_as_of")
    owned_opening, opening_clocks = None, None
    if opening_lots is not None:
        assert start is not None
        owned_opening, opening_clocks = opening_inventory(
            opening_lots,
            asset_dtype=fills["asset"].dtype,
            timezone=timezone,
            aware=aware,
            opening_as_of=start,
        )
    owned_marks = None
    selected_marks = {}
    if marks is not None:
        owned_marks, mark_clocks = mark_inventory(
            marks, asset_dtype=fills["asset"].dtype, timezone=timezone, aware=aware
        )
        assert cutoff is not None
        for index, record in enumerate(owned_marks.to_dicts()):
            stamp = mark_clocks[index]
            previous = selected_marks.get(record["asset"])
            if stamp <= cutoff and (previous is None or stamp > previous[0]):
                selected_marks[record["asset"]] = (stamp, record["price"], index)
    before = [index for index in range(fills.height) if start is not None and clocks[index] < start]
    future = [
        index for index in range(fills.height) if cutoff is not None and clocks[index] > cutoff
    ]
    excluded = set(before + future)
    order = sorted(
        (index for index in range(fills.height) if index not in excluded),
        key=lambda index: (clocks[index].value, index),
    )
    raw = fills.to_dicts()
    books: dict[Any, deque[_Lot]] = defaultdict(deque)
    multipliers: dict[Any, Fraction] = {}
    if owned_opening is not None:
        assert opening_clocks is not None
        # Caller row order declares FIFO order when entry time is unknown or tied.
        for index, record in enumerate(owned_opening.to_dicts()):
            quantity = Fraction(str(record["quantity"]))
            direction = 1 if quantity > 0 else -1
            multiplier = Fraction(str(record["multiplier"]))
            asset = record["asset"]
            if multipliers.setdefault(asset, multiplier) != multiplier:
                raise ValueError(
                    "an asset's execution multiplier cannot change across matched lots"
                )
            if books[asset] and books[asset][0].direction != direction:
                raise ValueError(
                    "opening inventory cannot contain opposing lots for the same asset"
                )
            stamp = opening_clocks[index]
            books[asset].append(
                _Lot(
                    None,
                    stamp.value if pd.notna(stamp) else None,
                    direction,
                    abs(quantity),
                    None if record["entry_price"] is None else Fraction(str(record["entry_price"])),
                    record["entry_price"],
                    multiplier,
                    None
                    if record["remaining_commission"] is None
                    else Fraction(str(record["remaining_commission"])),
                    record["lot_id"],
                )
            )
    segments = []
    remaining = []
    realized = []
    realized_net = []
    allocated_fees = []
    remaining_fees = []
    unallocated_fees = []
    for index in order:
        record = raw[index]
        quantity = Fraction(str(record["quantity"]))
        fee = None if record["commission"] is None else Fraction(str(record["commission"]))
        if quantity == 0:
            unallocated_fees.append(fee)
            continue
        direction = 1 if quantity > 0 else -1
        total = abs(quantity)
        units = total
        price = Fraction(str(record["price"]))
        multiplier = Fraction(str(record["multiplier"]))
        if multipliers.setdefault(record["asset"], multiplier) != multiplier:
            raise ValueError("an asset's execution multiplier cannot change across matched lots")
        book = books[record["asset"]]
        while units and book and book[0].direction != direction:
            lot = book[0]
            matched = min(units, lot.quantity)
            entry_fee = None if lot.commission is None else lot.commission * matched / lot.quantity
            exit_fee = None if fee is None else fee * matched / total
            commission = None if entry_fee is None or exit_fee is None else entry_fee + exit_fee
            pnl = (
                None
                if lot.price is None
                else lot.direction * matched * (price - lot.price) * multiplier
            )
            net = None if commission is None or pnl is None else pnl - commission
            invested = None if lot.price is None else matched * abs(lot.price) * multiplier
            segments.append(
                {
                    "asset": record["asset"],
                    "entry_record_id": lot.identifier,
                    "opening_lot_id": lot.opening_id,
                    "exit_record_id": record["record_id"],
                    "entry_timestamp": lot.timestamp,
                    "exit_timestamp": clocks[index].value,
                    "direction": "long" if lot.direction > 0 else "short",
                    "quantity": float(matched),
                    "quantity_exact": str(matched),
                    "entry_price": lot.price_input,
                    "exit_price": record["price"],
                    "multiplier": _cash(multiplier),
                    "gross_pnl": _cash(pnl),
                    "commission": _cash(commission),
                    "net_pnl": _cash(net),
                    "gross_lot_return": _cash(pnl / invested)
                    if invested and pnl is not None
                    else None,
                    "net_lot_return": _cash(net / invested)
                    if invested and net is not None
                    else None,
                }
            )
            realized.append(pnl)
            realized_net.append(net)
            allocated_fees.append(commission)
            lot.quantity -= matched
            if lot.commission is not None:
                assert entry_fee is not None
                lot.commission -= entry_fee
            units -= matched
            if not lot.quantity:
                book.popleft()
        if units:
            book.append(
                _Lot(
                    record["record_id"],
                    clocks[index].value,
                    direction,
                    units,
                    price,
                    record["price"],
                    multiplier,
                    None if fee is None else fee * units / total,
                )
            )
    for asset, book in books.items():
        for lot in book:
            remaining.append(
                {
                    "asset": asset,
                    "entry_record_id": lot.identifier,
                    "opening_lot_id": lot.opening_id,
                    "entry_timestamp": lot.timestamp,
                    "direction": "long" if lot.direction > 0 else "short",
                    "quantity": float(lot.quantity),
                    "quantity_exact": str(lot.quantity),
                    "entry_price": lot.price_input,
                    "multiplier": _cash(lot.multiplier),
                    "remaining_commission": _cash(lot.commission),
                }
            )
            remaining_fees.append(lot.commission)
    common = {
        "asset": fills["asset"].dtype,
        "entry_record_id": fills["record_id"].dtype,
        "entry_timestamp": pl.Int64,
        "direction": pl.String,
        "quantity": pl.Float64,
        "quantity_exact": pl.String,
        "entry_price": price_dtype(fills, owned_opening),
        "opening_lot_id": owned_opening["lot_id"].dtype if owned_opening is not None else pl.Null,
        "multiplier": pl.Float64,
    }
    closed_schema = {
        **common,
        "exit_record_id": fills["record_id"].dtype,
        "exit_timestamp": pl.Int64,
        "exit_price": fills["price"].dtype,
        "gross_pnl": pl.Float64,
        "commission": pl.Float64,
        "net_pnl": pl.Float64,
        "gross_lot_return": pl.Float64,
        "net_lot_return": pl.Float64,
    }
    open_schema = {**common, "remaining_commission": pl.Float64}
    zone = str(clocks.tz) if aware else None
    result = RoundTripResult(
        _frame(segments, closed_schema, ["entry_timestamp", "exit_timestamp"], zone),
        _frame(remaining, open_schema, ["entry_timestamp"], zone),
        {
            "n_segments": len(segments),
            "n_open_lots": len(remaining),
            "n_known_gross_segments": sum(value is not None for value in realized),
            "n_known_net_segments": sum(value is not None for value in realized_net),
            "realized_gross_pnl": _known_sum(realized),
            "realized_net_pnl": _known_sum(realized_net),
            "allocated_commission": _known_sum(allocated_fees),
            "remaining_commission": _known_sum(remaining_fees),
            "unallocated_zero_quantity_commission": _known_sum(unallocated_fees),
        },
        {
            "matching": "FIFO actual ordered executions, not consecutive-fill VWAP aggregation",
            "opening_state": "explicit " + initial_state,
            "opening_as_of": start.isoformat() if start is not None else None,
            "as_of": cutoff.isoformat() if cutoff is not None else None,
            "excluded_before_opening": len(before),
            "excluded_future_executions": len(future),
            "processed_executions": len(order),
            "n_opening_lots": owned_opening.height if owned_opening is not None else 0,
            "opening_lot_order": "supplied FIFO row order; unknown entry times remain unknown",
            "same_instant_order": "source row order",
            "open_lots": "actual remaining lots, unmarked, no inferred terminal closes",
            "cash_precision": "Float64; original observations remain owned",
            "quantity_precision": "exact rational matching from decimal inputs; quantity_exact is authoritative for nonrepresentable Float64 units",
            "costs": "cash commission allocated once; actual execution prices already include price slippage",
            "returns": "lot PnL / absolute entry notional, not a portfolio NAV contribution",
        },
        fills.clone(),
        owned_opening,
        owned_marks,
    )
    if cutoff is not None:
        _mark_open_lots(
            result,
            books,
            selected_marks,
            cutoff,
            zone,
            closed_schema,
            synthetic_close,
            realized,
            realized_net,
        )
    return result


def _mark_open_lots(
    result: RoundTripResult,
    books: dict[Any, deque[_Lot]],
    marks: dict[Any, tuple[pd.Timestamp, Any, int]],
    cutoff: pd.Timestamp,
    zone: str | None,
    closed_schema: dict[str, Any],
    synthetic_close: bool,
    realized: list[Fraction | None],
    realized_net: list[Fraction | None],
) -> None:
    values = []
    synthetic = []
    gross_values, net_values = [], []
    for asset, book in books.items():
        mark = marks.get(asset)
        if book and synthetic_close and mark is None:
            raise ValueError(f"synthetic_close requires a known-at mark for open asset {asset!r}")
        for lot in book:
            stamp, price, row = mark if mark is not None else (None, None, None)
            numeric_price = Fraction(str(price)) if price is not None else None
            pnl = (
                lot.direction * lot.quantity * (numeric_price - lot.price) * lot.multiplier
                if numeric_price is not None and lot.price is not None
                else None
            )
            net = pnl - lot.commission if pnl is not None and lot.commission is not None else None
            gross_values.append(pnl)
            net_values.append(net)
            values.append(
                {
                    "mark": price,
                    "mark_timestamp": stamp.value if stamp is not None else None,
                    "mark_source_row": row,
                    "mark_age_seconds": (cutoff.value - stamp.value) / 1e9
                    if stamp is not None
                    else None,
                    "marked_notional": _cash(
                        lot.direction * lot.quantity * numeric_price * lot.multiplier
                    )
                    if numeric_price is not None
                    else None,
                    "unrealized_gross_pnl": _cash(pnl),
                    "unrealized_net_pnl": _cash(net),
                }
            )
            if synthetic_close:
                invested = (
                    abs(lot.price) * lot.quantity * lot.multiplier
                    if lot.price is not None
                    else None
                )
                synthetic.append(
                    {
                        "asset": asset,
                        "entry_record_id": lot.identifier,
                        "opening_lot_id": lot.opening_id,
                        "entry_timestamp": lot.timestamp,
                        "exit_timestamp": cutoff.value,
                        "exit_record_id": None,
                        "direction": "long" if lot.direction > 0 else "short",
                        "quantity": float(lot.quantity),
                        "quantity_exact": str(lot.quantity),
                        "entry_price": lot.price_input,
                        "exit_price": price,
                        "multiplier": _cash(lot.multiplier),
                        "gross_pnl": _cash(pnl),
                        "net_pnl": _cash(net),
                        "commission": _cash(lot.commission),
                        "gross_lot_return": _cash(pnl / invested)
                        if invested and pnl is not None
                        else None,
                        "net_lot_return": _cash(net / invested)
                        if invested and net is not None
                        else None,
                        "is_synthetic": True,
                    }
                )
    mark_schema = {
        "mark": result.marks["price"].dtype if result.marks is not None else pl.Float64,
        "mark_timestamp": pl.Int64,
        "mark_source_row": pl.Int64,
        "mark_age_seconds": pl.Float64,
        "marked_notional": pl.Float64,
        "unrealized_gross_pnl": pl.Float64,
        "unrealized_net_pnl": pl.Float64,
    }
    valuation = _frame(values, mark_schema, ["mark_timestamp"], zone)
    result.open_lots = result.open_lots.hstack(valuation)
    for label, observed, actual in (
        ("gross", gross_values, realized),
        ("net", net_values, realized_net),
    ):
        result.metrics[f"unrealized_{label}_pnl"] = _known_sum(observed)
        result.metrics[f"marked_{label}_pnl"] = _known_sum(actual + observed)
    result.metrics["n_marked_open_lots"] = sum(value["mark"] is not None for value in values)
    result.metrics["n_known_basis_open_lots"] = (
        result.open_lots["entry_price"].len() - result.open_lots["entry_price"].null_count()
    )
    result.metrics["n_known_unrealized_gross_lots"] = sum(
        value is not None for value in gross_values
    )
    result.metrics["n_known_unrealized_net_lots"] = sum(value is not None for value in net_values)
    result.metadata.update(
        marks="last supplied timestamp at/before as_of; quote age is explicit, not guaranteed current",
        marked_notional="signed units * price * multiplier; not inferred cash or NAV",
        synthetic_close="separate hypothetical closes, zero additional closing fee; actual inventory/realized PnL unchanged"
        if synthetic_close
        else "not requested",
    )
    if synthetic_close:
        result.synthetic_segments = _frame(
            synthetic,
            {**closed_schema, "exit_price": mark_schema["mark"], "is_synthetic": pl.Boolean},
            ["entry_timestamp", "exit_timestamp"],
            zone,
        )
