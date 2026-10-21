"""Pure Zipline performance extraction into the existing owned portfolio boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.evaluation import PortfolioAnalysis


def _asset(
    value: Any, declarations: Mapping[int, Mapping[str, Any]]
) -> tuple[int, str | None, float]:
    if isinstance(value, Mapping):
        identifier = value.get("sid")
        details = value
    elif isinstance(value, Integral) and not isinstance(value, bool):
        identifier = value
        details = declarations.get(int(identifier), {})
    else:
        identifier = getattr(value, "sid", None)
        details = {
            "symbol": getattr(value, "symbol", None),
            "multiplier": getattr(value, "price_multiplier", None),
        }
    if isinstance(identifier, bool) or not isinstance(identifier, Integral):
        raise ValueError("Zipline asset sid must be a stable integer, not a symbol")
    multiplier = details.get("multiplier", details.get("price_multiplier"))
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, Real)
        or not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("Zipline raw asset IDs require declared finite positive multipliers")
    symbol = details.get("symbol")
    return int(identifier), str(symbol) if symbol is not None else None, float(multiplier)


def _records(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(
            f"Zipline {name} must contain actual nested record lists; missing is not empty"
        )
    return list(value)


def portfolio_analysis_from_zipline(
    performance: pd.DataFrame,
    *,
    benchmark: pd.Series | None = None,
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    calendar: str | CalendarUnset | None = UNSET_CALENDAR,
    asset_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    periods_per_year: int = 252,
) -> PortfolioAnalysis:
    """Consume actual run_algorithm output without importing Zipline or Pyfolio.

    Required returns remain authoritative. Optional portfolio_value/ending_cash,
    complete nested positions and actual transactions enable only their dependent
    analyses. Asset objects are read by stable sid/price_multiplier attributes;
    portable dictionaries or integer IDs with asset_metadata are also accepted.
    Futures marked notional uses its multiplier; NAV and cash stay producer values.
    Unknown execution commission/slippage is never apportioned from account costs.
    """
    if not isinstance(performance, pd.DataFrame) or not isinstance(
        performance.index, pd.DatetimeIndex
    ):
        raise TypeError("Zipline performance requires a pandas DataFrame with a DatetimeIndex")
    if "returns" not in performance.columns:
        raise ValueError("Zipline performance requires reported returns")
    if performance.columns.has_duplicates:
        raise ValueError("Zipline performance columns must be unique")
    feed = (
        FeedSpec.from_any(feed_spec)
        if feed_spec is not None
        else FeedSpec(
            timestamp_col="date",
            entity_col="asset",
            time_contract_version=2,
            timestamp_semantics="event_time",
        )
    )
    dates = performance.index.copy(deep=True)
    declarations = asset_metadata or {}
    labels = []
    books = []
    positions = None
    if "positions" in performance:
        for index, value in enumerate(performance["positions"]):
            book = {}
            for record in _records(value, "positions"):
                identifier, symbol, multiplier = _asset(record.get("sid"), declarations)
                if identifier in book:
                    raise ValueError("Zipline positions repeat an asset within a snapshot")
                amount, price = record.get("amount"), record.get("last_sale_price")
                if any(
                    isinstance(number, bool)
                    or not isinstance(number, Real)
                    or not np.isfinite(number)
                    for number in (amount, price)
                ):
                    raise ValueError("Zipline positions require finite amount and last_sale_price")
                book[identifier] = {
                    "value": float(amount) * float(price) * multiplier,
                    "quantity": amount,
                    "mark": price,
                    "multiplier": multiplier,
                    "symbol": symbol,
                    "cost_basis": record.get("cost_basis"),
                }
                labels.append(
                    {
                        "observation": index,
                        "sid": identifier,
                        "symbol": symbol,
                        "multiplier": multiplier,
                    }
                )
            books.append(book)
        assets = sorted({identifier for book in books for identifier in book})
        # Zipline emits complete books. Empty lists explicitly prove all-cash dates;
        # only those known complete snapshots permit zeroing absent assets.
        positions = (
            pd.DataFrame(
                [
                    {
                        "date": date,
                        "asset": identifier,
                        **book.get(identifier, {"value": 0.0, "quantity": 0}),
                    }
                    for date, book in zip(dates, books, strict=True)
                    for identifier in assets
                ]
            )
            if assets
            else pd.DataFrame(index=dates)
        )
    account = None
    if "portfolio_value" in performance or "ending_cash" in performance:
        account = pd.DataFrame({"date": dates})
        for source, target in (("portfolio_value", "equity"), ("ending_cash", "cash")):
            if source in performance:
                account[target] = performance[source].to_numpy(copy=True)
    executions = None
    if "transactions" in performance:
        records = []
        for value in performance["transactions"]:
            for record in _records(value, "transactions"):
                identifier, symbol, multiplier = _asset(record.get("sid"), declarations)
                timestamp_value = record.get("dt")
                if timestamp_value is None:
                    raise ValueError("Zipline transactions require actual execution dt instants")
                records.append(
                    {
                        "timestamp": timestamp_value,
                        "asset": identifier,
                        "quantity": record.get("amount"),
                        "price": record.get("price"),
                        "commission": record.get("commission"),
                        "slippage": record.get("slippage"),
                        "multiplier": multiplier,
                        "symbol": symbol,
                        "order_id": record.get("order_id"),
                    }
                )
        executions = (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(
                {
                    "timestamp": pd.Series([], dtype=dates.dtype),
                    "asset": pd.Series([], dtype="int64"),
                    "quantity": pd.Series([], dtype="float64"),
                    "price": pd.Series([], dtype="float64"),
                    "commission": pd.Series([], dtype="float64"),
                    "slippage": pd.Series([], dtype="float64"),
                    "multiplier": pd.Series([], dtype="float64"),
                }
            )
        )
        if records:
            executions["timestamp"] = pd.to_datetime(executions["timestamp"])
    analysis = PortfolioAnalysis.from_frames(
        performance["returns"].copy(deep=True),
        benchmark=benchmark,
        positions=positions,
        valuations=account,
        transactions=executions,
        complete_snapshots=positions is not None,
        snapshot_kind="end_of_day",
        feed_spec=feed,
        timestamp_col="date",
        entity_col="asset",
        calendar=calendar,
        periods_per_year=periods_per_year,
        transaction_timestamp_col="timestamp",
        positions_format="auto",
    )
    inputs = analysis.input_data
    assert inputs is not None
    metadata = inputs.metadata
    metadata["source"] = {
        "format": "Zipline run_algorithm performance",
        "reported_NAV": "portfolio_value" in performance,
        "reported_cash": "ending_cash" in performance,
        "asset_labels": labels,
        "execution_costs": "per-record values only; portfolio commissions are not apportioned",
        "positions_value": "amount * last_sale_price * declared multiplier; not inferred NAV",
    }
    analysis._input_data = replace(inputs, _metadata=json.dumps(metadata, allow_nan=False))
    return analysis


__all__ = ["portfolio_analysis_from_zipline"]
