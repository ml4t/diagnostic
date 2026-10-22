"""Project explicit hypothetical closes from a complete legacy cash-price book."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from math import isfinite
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis.fifo import (
    RoundTripResult,
    reconstruct_round_trips,
)
from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_inputs import clock, instant
from ml4t.diagnostic.splitters.calendar import TradingCalendar


def closing_transactions(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    as_of: Any,
    feed_spec: FeedSpec | Mapping[str, Any] | None,
    calendar: str | CalendarUnset | None,
) -> pd.DataFrame:
    from .round_trips import _inputs
    from .txn import _dated

    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame")
    _dated(positions, "positions")
    if (
        positions.empty
        or not positions.index.is_unique
        or not positions.index.is_monotonic_increasing
    ):
        raise ValueError("positions require a nonempty unique ordered observation index")
    if positions.columns.has_duplicates or "cash" not in positions:
        raise ValueError("positions require distinct asset columns and separate cash")
    if "is_synthetic" in transactions and transactions["is_synthetic"].any():
        raise ValueError("pass actual transactions, not an already synthetically closed projection")
    owned, labels = _inputs(transactions)
    native = reconstruct_round_trips(owned, initial_state="flat")
    quantities: dict[Any, Fraction] = defaultdict(Fraction)
    for row in native.open_lots.to_dicts():
        quantities[labels[int(row["asset"])]] += Fraction(row["quantity_exact"]) * (
            1 if row["direction"] == "long" else -1
        )
    for label in positions.columns:
        if label == "cash":
            continue
        value = positions[label].iloc[-1]
        if quantities.get(label, 0) == 0 and pd.notna(value) and value != 0:
            raise ValueError(
                "ending marked holdings disagree with the declared flat opening execution book"
            )
    projected = transactions[["symbol", "amount", "price"]].copy(deep=True)
    projected["is_synthetic"] = False
    if not quantities:
        projected.attrs["diagnostic_closing"] = {
            "method": "explicit_hypothetical_closes",
            "n_synthetic": 0,
        }
        return projected
    feed = (
        FeedSpec.from_any(feed_spec) if feed_spec is not None else FeedSpec(time_contract_version=2)
    )
    times = clock(owned["timestamp"], "transaction timestamp", feed.timezone)
    aware = times.tz is not None
    resolved = feed.resolve_calendar(calendar=calendar)
    last = positions.index[-1]
    if as_of is not None:
        terminal = instant(as_of, "as_of", feed.timezone, aware)
    elif resolved is not None and last == last.normalize():
        schedule = TradingCalendar(resolved)._schedule(start_date=last.date(), end_date=last.date())
        if schedule.empty:
            raise ValueError("last position label is not a session; supply an explicit as_of")
        terminal = pd.Timestamp(schedule["market_close"].iloc[0])
        if not aware:
            raise ValueError("calendar closing requires aware executions or a declared timezone")
        terminal = terminal.tz_convert(times.tz)
    else:
        terminal = instant(last, "last position observation", feed.timezone, aware)
    if aware:
        terminal = terminal.tz_convert(times.tz)
    if transactions.index.tz is None and feed.timezone is not None:
        projected.index = times.rename(transactions.index.name)
    if times.max() > terminal:
        raise ValueError(
            "closing as_of precedes actual executions; provide the actual valuation instant"
        )
    additions = []
    for label, quantity in quantities.items():
        if label not in positions:
            raise ValueError(
                f"open asset {label!r} needs an explicitly supplied ending marked value"
            )
        value = positions[label].iloc[-1]
        if isinstance(value, bool) or pd.isna(value) or not isfinite(value):
            raise ValueError(
                "open holdings require finite ending marked values, not unknown-to-zero filling"
            )
        implied = Fraction(str(value)) / quantity
        amount = -quantity
        additions.append(
            {
                "symbol": label,
                "amount": int(amount) if amount.denominator == 1 else float(amount),
                "amount_exact": str(amount),
                "price": int(implied) if implied.denominator == 1 else float(implied),
                "is_synthetic": True,
            }
        )
    closing = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "amount"} for row in additions],
        index=pd.DatetimeIndex([terminal] * len(additions)),
    )
    closing["amount"] = pd.Series(
        [row["amount"] for row in additions], index=closing.index, dtype=object
    )
    output = pd.concat([projected, closing])
    output.index.name = transactions.index.name
    wire = feed.to_dict()
    declared_calendar = "calendar" in wire and (
        feed.time_contract_version == 2 or wire["calendar"] is not None
    )
    output.attrs["diagnostic_closing"] = {
        "method": "explicit_hypothetical_closes",
        "n_synthetic": len(closing),
        "as_of": terminal.isoformat(),
        "timezone": feed.timezone,
        "calendar": resolved,
        "calendar_source": "argument"
        if calendar is not UNSET_CALENDAR
        else "feed_spec"
        if declared_calendar
        else "default",
        "feed_spec": wire,
        "book": "flat opening plus complete actual transactions; carried inventory requires native supplied openings",
        "price": "implied ending marked value / exact net units, not an observed quote",
        "precision": "amount_exact is authoritative for hypothetical units; implied fractional prices project to Float64",
        "clock": "explicit as_of, or declared session close for daily labels; no invented plus-one-second fill",
        "costs": "legacy gross cash-price profile; no inferred commission or slippage",
    }
    return output


def native_with_hypothetical_closes(transactions: pd.DataFrame) -> tuple[RoundTripResult, pd.Index]:
    from .round_trips import _inputs
    from .txn import _dated

    _dated(transactions, "transactions")
    flags = transactions["is_synthetic"]
    if not pd.api.types.is_bool_dtype(flags) or flags.isna().any():
        raise ValueError("is_synthetic must contain explicit nonmissing booleans")
    actual, labels = _inputs(transactions.loc[~flags])
    synthetic = transactions.loc[flags]
    if synthetic.index.nunique() != 1:
        raise ValueError("hypothetical close rows require one explicit terminal instant")
    terminal = synthetic.index[0]
    profile = dict(transactions.attrs.get("diagnostic_closing", {}))
    actual_times = clock(actual["timestamp"], "transaction timestamp", profile.get("timezone"))
    if len(actual_times) and actual_times.max() > terminal:
        raise ValueError("hypothetical close cannot precede actual transactions")
    expected: dict[Any, Fraction] = defaultdict(Fraction)
    for asset, amount in zip(actual["asset"].to_list(), actual["quantity"].to_list(), strict=True):
        expected[asset] += Fraction(str(amount))
    codes = {label: index for index, label in enumerate(labels)}
    requested = {}
    marks = []
    for index in range(len(synthetic)):
        symbol = synthetic["symbol"].iloc[index]
        if symbol not in codes:
            raise ValueError("hypothetical close references an asset outside actual transactions")
        code = codes[symbol]
        if code in requested:
            raise ValueError("one hypothetical close per asset is required")
        visible_amount = synthetic["amount"].iloc[index]
        exact = synthetic["amount_exact"].iloc[index] if "amount_exact" in synthetic else None
        amount = Fraction(str(visible_amount)) if pd.isna(exact) else Fraction(str(exact))
        if Fraction(str(float(visible_amount))) != Fraction(str(float(amount))):
            raise ValueError("hypothetical amount disagrees with its exact projection")
        requested[code] = amount
        marks.append(
            {"timestamp": terminal, "asset": code, "price": synthetic["price"].iloc[index]}
        )
    if {asset: -quantity for asset, quantity in expected.items() if quantity} != requested:
        raise ValueError(
            "hypothetical quantities must exactly close the complete actual execution book"
        )
    result = reconstruct_round_trips(
        actual,
        initial_state="flat",
        matching="consecutive_vwap",
        marks=pl.from_pandas(pd.DataFrame(marks)),
        as_of=terminal,
        synthetic_close=True,
        timezone=profile.get("timezone"),
    )
    profile.update(
        n_synthetic=len(synthetic),
        actual_fills=len(actual),
        mark_source="explicitly requested hypothetical prices, not observed market quotes",
        costs="legacy gross cash-price profile; actual commission/slippage fields ignored",
    )
    result.metadata.update(legacy_synthetic_rows=len(synthetic), legacy_closing_profile=profile)
    return result, labels
