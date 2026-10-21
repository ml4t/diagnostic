"""Exact-unit FIFO lot matching over actual owned executions, not broker simulation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

import pandas as pd
import polars as pl


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


@dataclass
class _Lot:
    identifier: Any
    timestamp: int
    direction: int
    quantity: Fraction
    price: Fraction
    price_input: Any
    multiplier: Fraction
    commission: Fraction | None


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
    fills: pl.DataFrame | None, *, initial_state: str | None
) -> RoundTripResult:
    """Match observed fills from an explicitly flat opening state.

    No implicit position is inferred from a first sale or an EOD portfolio.
    Non-flat opening lots, marking and synthetic terminal closes are separate
    inputs to the remaining reconstruction workflow, not fabricated here.
    """
    if initial_state != "flat":
        raise ValueError("initial_state='flat' must declare the opening book explicitly")
    if fills is None:
        raise ValueError(
            "FIFO requires supplied transactions; absent fills are not an empty ledger"
        )
    clocks = pd.DatetimeIndex(fills["timestamp"].to_pandas())
    order = sorted(range(fills.height), key=lambda index: (clocks[index].value, index))
    raw = fills.to_dicts()
    books: dict[Any, deque[_Lot]] = defaultdict(deque)
    multipliers: dict[Any, Fraction] = {}
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
            pnl = lot.direction * matched * (price - lot.price) * multiplier
            net = None if commission is None else pnl - commission
            invested = matched * abs(lot.price) * multiplier
            segments.append(
                {
                    "asset": record["asset"],
                    "entry_record_id": lot.identifier,
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
                    "gross_lot_return": _cash(pnl / invested) if invested else None,
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
        "entry_price": fills["price"].dtype,
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
    zone = fills["timestamp"].dtype.time_zone
    return RoundTripResult(
        _frame(segments, closed_schema, ["entry_timestamp", "exit_timestamp"], zone),
        _frame(remaining, open_schema, ["entry_timestamp"], zone),
        {
            "n_segments": len(segments),
            "n_open_lots": len(remaining),
            "realized_gross_pnl": _known_sum(realized),
            "realized_net_pnl": _known_sum(realized_net),
            "allocated_commission": _known_sum(allocated_fees),
            "remaining_commission": _known_sum(remaining_fees),
            "unallocated_zero_quantity_commission": _known_sum(unallocated_fees),
        },
        {
            "matching": "FIFO actual ordered executions, not consecutive-fill VWAP aggregation",
            "opening_state": "explicit flat",
            "same_instant_order": "source row order",
            "open_lots": "actual remaining lots, unmarked, no inferred terminal closes",
            "cash_precision": "Float64; original observations remain owned",
            "quantity_precision": "exact rational matching from decimal inputs; quantity_exact is authoritative for nonrepresentable Float64 units",
            "costs": "cash commission allocated once; actual execution prices already include price slippage",
            "returns": "lot PnL / absolute entry notional, not a portfolio NAV contribution",
        },
        fills.clone(),
    )
