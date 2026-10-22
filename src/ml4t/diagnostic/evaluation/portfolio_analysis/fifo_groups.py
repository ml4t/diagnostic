"""Explicit consecutive-VWAP analytical groups over actual retained executions."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from .fifo import RoundTripResult


def group_executions(
    fills: pl.DataFrame, clocks: pd.DatetimeIndex, order: list[int], max_gap: str | timedelta | None
) -> pl.DataFrame:
    from .fifo import _cash, _known_sum

    if max_gap is None:
        max_gap = "8h"
    if (
        not isinstance(max_gap, str | timedelta)
        or isinstance(max_gap, str)
        and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", max_gap.strip())
    ):
        raise ValueError("max_gap requires an explicit elapsed unit, such as '8h'")
    try:
        gap = pd.Timedelta(max_gap)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("max_gap must be a finite elapsed duration with an explicit unit") from exc
    if pd.isna(gap) or gap < pd.Timedelta(0):
        raise ValueError("max_gap must be a nonnegative finite elapsed duration")
    raw = fills.to_dicts()
    by_asset: dict[Any, list[list[int]]] = defaultdict(list)
    multipliers = {}
    for index in order:
        record = raw[index]
        asset = record["asset"]
        multiplier = Fraction(str(record["multiplier"]))
        if multipliers.setdefault(asset, multiplier) != multiplier:
            raise ValueError("an asset's execution multiplier cannot change across matched lots")
        previous = by_asset[asset][-1][-1] if by_asset[asset] else None
        quantity = Fraction(str(record["quantity"]))
        same_direction = (
            previous is not None and quantity * Fraction(str(raw[previous]["quantity"])) > 0
        )
        if same_direction and clocks[index].value - clocks[previous].value <= gap.value:
            by_asset[asset][-1].append(index)
        else:
            by_asset[asset].append([index])
    membership = sorted(
        (indices for groups in by_asset.values() for indices in groups),
        key=lambda indices: (clocks[indices[0]].value, indices[0]),
    )
    records = []
    for identifier, indices in enumerate(membership):
        first = raw[indices[0]]
        quantities = [Fraction(str(raw[index]["quantity"])) for index in indices]
        quantity = sum(quantities, Fraction(0))
        price = (
            sum(
                (
                    units * Fraction(str(raw[index]["price"]))
                    for index, units in zip(indices, quantities, strict=True)
                ),
                Fraction(0),
            )
            / quantity
            if quantity
            else Fraction(str(first["price"]))
        )
        fees = [
            None if raw[index]["commission"] is None else Fraction(str(raw[index]["commission"]))
            for index in indices
        ]
        fee = (
            sum((value for value in fees if value is not None), Fraction(0))
            if all(value is not None for value in fees)
            else None
        )
        records.append(
            {
                "record_id": identifier,
                "asset": first["asset"],
                "timestamp": clocks[indices[0]].value,
                "available_at": clocks[indices[-1]].value,
                "quantity": _cash(quantity),
                "quantity_exact": str(quantity),
                "price": _cash(price),
                "price_exact": str(price),
                "multiplier": first["multiplier"],
                "multiplier_exact": str(Fraction(str(first["multiplier"]))),
                "commission": _known_sum(fees),
                "commission_exact": str(fee) if fee is not None else None,
                "member_record_ids": [raw[index]["record_id"] for index in indices],
                "source_record_id": first["record_id"] if len(indices) == 1 else None,
                "first_record_id": first["record_id"],
                "n_source_executions": len(indices),
            }
        )
    frame = pl.DataFrame(
        records,
        schema={
            "record_id": pl.UInt32,
            "asset": fills["asset"].dtype,
            "timestamp": pl.Int64,
            "available_at": pl.Int64,
            "quantity": pl.Float64,
            "quantity_exact": pl.String,
            "price": pl.Float64,
            "price_exact": pl.String,
            "multiplier": pl.Float64,
            "multiplier_exact": pl.String,
            "commission": pl.Float64,
            "commission_exact": pl.String,
            "member_record_ids": pl.List(fills["record_id"].dtype),
            "source_record_id": fills["record_id"].dtype,
            "first_record_id": fills["record_id"].dtype,
            "n_source_executions": pl.UInt32,
        },
    )
    zone = str(clocks.tz) if clocks.tz is not None else None
    return frame.with_columns(pl.col("timestamp", "available_at").cast(pl.Datetime("ns", zone)))


def restore_sources(result: RoundTripResult, groups: pl.DataFrame, source: pl.DataFrame) -> None:
    for name in ("segments", "open_lots", "synthetic_segments"):
        frame = getattr(result, name)
        if frame is None:
            continue
        for role in ("entry", "exit"):
            column = role + "_record_id"
            if column not in frame.columns:
                continue
            key = role + "_group_id"
            frame = frame.rename({column: key}).join(
                groups.select(
                    pl.col("record_id").alias(key),
                    pl.col("source_record_id").alias(column),
                    pl.col("available_at").alias(role + "_available_at"),
                    pl.col("member_record_ids").alias(role + "_source_record_ids"),
                ),
                on=key,
                how="left",
                maintain_order="left",
            )
            if role == "entry" and result.opening_lots is not None:
                start = pd.Timestamp(result.metadata["opening_as_of"]).value
                frame = frame.with_columns(
                    pl.when(pl.col("opening_lot_id").is_not_null())
                    .then(pl.lit(start).cast(frame[role + "_available_at"].dtype))
                    .otherwise(pl.col(role + "_available_at"))
                    .alias(role + "_available_at")
                )
            if role == "exit" and name == "synthetic_segments":
                frame = frame.with_columns(pl.col("exit_timestamp").alias("exit_available_at"))
        setattr(result, name, frame)
    result.executions = source.clone()
    result.matching_groups = groups.clone()
    result.metadata.update(
        matching="consecutive_vwap",
        open_lots="remaining analytical group inventory; actual net units, grouped rather than actual-FIFO cost basis",
        intraday_reconstruction="not an execution-state reconstruction; use actual_fifo and original fills",
        group_prices="analytical VWAP projections, not actual execution prices; exact price/quantity retained in matching_groups",
        group_time="first source instant is the representative timestamp; group price is only available_at the last member",
        group_identity="separate group IDs; actual record ID only when group has exactly one source execution; source lists retained",
        zero_quantity="separate unallocated cash events, not VWAP inventory",
    )
