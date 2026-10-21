"""Closing-fill observations and existing trade-statistics integration over FIFO results."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis

    from .fifo import RoundTripResult


def completed_trades(result: RoundTripResult, *, include_synthetic: bool) -> pl.DataFrame:
    from .fifo import _cash, _frame

    if not isinstance(include_synthetic, bool):
        raise TypeError("include_synthetic must be a boolean")

    groups = defaultdict(list)
    grouped = result.matching_groups is not None
    sources = [(result.segments, False)]
    if include_synthetic:
        if result.synthetic_segments is None:
            raise ValueError("synthetic segments were not requested during reconstruction")
        sources.append((result.synthetic_segments, True))
    for frame, synthetic in sources:
        temporal = ["entry_timestamp", "exit_timestamp"]
        if grouped:
            temporal += ["entry_available_at", "exit_available_at"]
        rows = frame.with_columns(pl.col(temporal).cast(pl.Int64)).to_dicts()
        for row in rows:
            closing_id = row["exit_group_id"] if grouped else row["exit_record_id"]
            groups[(row["asset"], closing_id, row["direction"], synthetic)].append(row)
    records = []
    for (asset, exit_id, direction, synthetic), rows in groups.items():
        quantities = [Fraction(row["quantity_exact"]) for row in rows]
        quantity = sum(quantities, Fraction(0))
        exact_cash = {
            name: sum((Fraction(row[name + "_exact"]) for row in rows), Fraction(0))
            if all(row[name + "_exact"] is not None for row in rows)
            else None
            for name in ("gross_pnl", "net_pnl", "commission", "entry_notional")
        }
        cash = {name: _cash(value) for name, value in exact_cash.items()}
        known_entry = all(row["entry_price"] is not None for row in rows)
        average_entry = (
            _cash(
                sum(
                    (
                        Fraction(str(row["entry_price"])) * units
                        for row, units in zip(rows, quantities, strict=True)
                    ),
                    Fraction(0),
                )
                / quantity
            )
            if known_entry
            else None
        )
        first_entry = (
            min(row["entry_timestamp"] for row in rows)
            if all(row["entry_timestamp"] is not None for row in rows)
            else None
        )
        exit_timestamp = rows[0]["exit_timestamp"]
        record = {
            "asset": asset,
            "exit_record_id": rows[0]["exit_record_id"],
            "direction": direction,
            "entry_timestamp": first_entry,
            "exit_timestamp": exit_timestamp,
            "duration_seconds": (exit_timestamp - first_entry) / 1e9
            if first_entry is not None
            else None,
            "quantity": _cash(quantity),
            "quantity_exact": str(quantity),
            "entry_price": average_entry,
            "exit_price": rows[0]["exit_price"],
            "multiplier": rows[0]["multiplier"],
            "n_segments": len(rows),
            "is_synthetic": synthetic,
            **cash,
            **{
                name + "_exact": str(value) if value is not None else None
                for name, value in exact_cash.items()
            },
        }
        if grouped:
            record["exit_group_id"] = exit_id
            record["entry_available_at"] = (
                max(row["entry_available_at"] for row in rows)
                if all(row["entry_available_at"] is not None for row in rows)
                else None
            )
            record["exit_available_at"] = rows[0]["exit_available_at"]
        for name in ("gross", "net"):
            # Sum exact cash first; never average the entry/exit-pair returns.
            pnl = (
                sum((Fraction(row[name + "_pnl_exact"]) for row in rows), Fraction(0))
                if cash[name + "_pnl"] is not None
                else None
            )
            invested = (
                sum((Fraction(row["entry_notional_exact"]) for row in rows), Fraction(0))
                if cash["entry_notional"] is not None
                else None
            )
            record[name + "_lot_return"] = (
                _cash(pnl / invested) if invested and pnl is not None else None
            )
        records.append(record)
    schema = {
        "asset": result.segments["asset"].dtype,
        "exit_record_id": result.segments["exit_record_id"].dtype,
        "direction": pl.String,
        "entry_timestamp": pl.Int64,
        "exit_timestamp": pl.Int64,
        "duration_seconds": pl.Float64,
        "quantity": pl.Float64,
        "quantity_exact": pl.String,
        "entry_price": pl.Float64,
        "exit_price": result.segments["exit_price"].dtype,
        "multiplier": pl.Float64,
        "n_segments": pl.UInt32,
        "is_synthetic": pl.Boolean,
        **dict.fromkeys(
            ("gross_pnl_exact", "net_pnl_exact", "commission_exact", "entry_notional_exact"),
            pl.String,
        ),
        **dict.fromkeys(
            (
                "gross_pnl",
                "net_pnl",
                "commission",
                "entry_notional",
                "gross_lot_return",
                "net_lot_return",
            ),
            pl.Float64,
        ),
    }
    if grouped:
        schema["exit_group_id"] = result.segments["exit_group_id"].dtype
        schema.update(entry_available_at=pl.Int64, exit_available_at=pl.Int64)
    # Synthetic quotes can have a different lossless input dtype from actual exits.
    if include_synthetic and result.synthetic_segments is not None:
        from .fifo_inputs import price_dtype

        schema["exit_price"] = price_dtype(
            result.segments.rename({"exit_price": "price"}),
            result.synthetic_segments.rename(
                {"exit_price": "entry_price", "entry_price": "original_entry_price"}
            ),
        )
    frame = _frame(
        records,
        schema,
        ["entry_timestamp", "exit_timestamp"]
        + (["entry_available_at", "exit_available_at"] if grouped else []),
        result.segments["exit_timestamp"].dtype.time_zone,
    )
    duration = frame["exit_timestamp"] - frame["entry_timestamp"]
    if (duration.drop_nulls().cast(pl.Int64) < 0).any():
        raise ValueError("completed duration cannot be represented in nanoseconds")
    return frame.with_columns(duration.alias("duration"))


def selected_trades(
    result: RoundTripResult, *, pnl: str, missing: str, include_synthetic: bool
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if pnl not in ("gross", "net"):
        raise ValueError("pnl must explicitly select 'gross' or 'net'")
    if missing not in ("raise", "exclude"):
        raise ValueError("missing must be 'raise' or explicit 'exclude'")
    frame = completed_trades(result, include_synthetic=include_synthetic)
    column = pnl + "_pnl"
    excluded = frame[column].null_count()
    if excluded and missing == "raise":
        raise ValueError(
            f"{excluded} closing-fill observations have unknown {pnl} PnL; use explicit missing='exclude' or a known PnL view"
        )
    return frame.filter(pl.col(column).is_not_null()), {
        "observation": "one closing analytical consecutive group per asset/direction"
        if result.matching_groups is not None
        else "one closing fill per asset/direction, not one matched entry lot",
        "pnl": pnl,
        "missing": missing,
        "n_total_observations": frame.height,
        "n_excluded_unknown_pnl": excluded,
        "include_synthetic": include_synthetic,
        "return_denominator": "summed absolute matched entry notional, not portfolio NAV",
        "timestamp_convention": result.metadata.get("group_time", "actual closing event instants"),
    }


def to_trade_analysis(
    result: RoundTripResult, *, pnl: Literal["gross", "net"], missing: str, include_synthetic: bool
) -> TradeAnalysis:
    from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis, TradeMetrics

    frame, coverage = selected_trades(
        result, pnl=pnl, missing=missing, include_synthetic=include_synthetic
    )
    if frame.is_empty():
        raise ValueError(
            "no known completed-trade PnL observations; inspect coverage before analysis"
        )
    zone = frame["exit_timestamp"].dtype.time_zone
    metrics = []
    for row in frame.with_columns(
        pl.col("entry_timestamp", "exit_timestamp").cast(pl.Int64)
    ).to_dicts():
        entry = (
            pd.Timestamp(row["entry_timestamp"], tz=zone)
            if row["entry_timestamp"] is not None
            else None
        )
        exit_time = pd.Timestamp(row["exit_timestamp"], tz=zone)
        metrics.append(
            TradeMetrics(
                timestamp=exit_time,
                symbol=str(row["asset"]),
                entry_price=row["entry_price"],
                exit_price=row["exit_price"],
                pnl=row[pnl + "_pnl"],
                duration=exit_time - entry if entry is not None else None,
                entry_timestamp=entry,
                direction=row["direction"],
                quantity=row["quantity"],
                multiplier=row["multiplier"],
                fees=row["commission"],
                return_basis="entry_notional",
                entry_notional=row["entry_notional"],
                metadata={
                    "asset": row["asset"],
                    "exit_record_id": row["exit_record_id"],
                    "exit_group_id": row.get("exit_group_id"),
                    "n_segments": row["n_segments"],
                    "is_synthetic": row["is_synthetic"],
                    "entry_price": "analytical quantity-weighted entry projection; original executions remain in FIFO result",
                    "pnl_view": pnl,
                },
            )
        )
    return TradeAnalysis(metrics, source_metadata=coverage)


def win_probability(
    result: RoundTripResult,
    *,
    pnl: Literal["gross", "net"],
    missing: str,
    include_synthetic: bool,
    confidence: float,
) -> dict[str, Any]:
    from scipy.stats import beta

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be a finite real number strictly between 0 and 1")
    frame, coverage = selected_trades(
        result, pnl=pnl, missing=missing, include_synthetic=include_synthetic
    )
    winners = frame.filter(pl.col(pnl + "_pnl") > 0).height
    alpha, beta_parameter = 1 + winners, 1 + frame.height - winners
    tail = (1 - confidence) / 2
    return {
        **coverage,
        "method": "Beta-binomial probability of strictly positive PnL",
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "posterior_alpha": float(alpha),
        "posterior_beta": float(beta_parameter),
        "n_observations": frame.height,
        "n_winners": winners,
        "n_nonwinners": frame.height - winners,
        "n_even": frame.filter(pl.col(pnl + "_pnl") == 0).height,
        "mean": alpha / (alpha + beta_parameter),
        "confidence": float(confidence),
        "lower": float(beta.ppf(tail, alpha, beta_parameter)),
        "upper": float(beta.ppf(1 - tail, alpha, beta_parameter)),
        "sample_state": "observed" if frame.height else "prior_only",
    }


def portfolio_contributions(
    result: RoundTripResult,
    portfolio_value: pl.DataFrame | pd.DataFrame | pd.Series,
    *,
    timestamp_col: str,
    value_col: str,
    calendar: str | CalendarUnset | None,
    alignment: str,
    nav_method: str,
    include_synthetic: bool,
) -> pl.DataFrame:
    from ml4t.diagnostic.utils.validation import (
        _align_datetime_key_resolution,
        _validate_observation_keys,
    )

    from .fifo_inputs import clock
    from .frame_inputs import _frame, _rename, _temporal
    from .holdings import _numeric
    from .transaction_metrics import cash_fraction_expr
    from .transactions import _clock

    if alignment not in ("declared", "civil_date"):
        raise ValueError("alignment must be 'declared' or explicit 'civil_date'")
    if nav_method not in ("positive_nav", "legacy_ieee"):
        raise ValueError("nav_method must be 'positive_nav' or explicit 'legacy_ieee'")
    if not all(isinstance(name, str) and name.strip() for name in (timestamp_col, value_col)):
        raise ValueError("timestamp_col and value_col must name nonempty columns")
    if timestamp_col == value_col:
        raise ValueError("timestamp_col and value_col must name distinct columns")
    if isinstance(portfolio_value, pd.Series):
        if not isinstance(portfolio_value.index, pd.DatetimeIndex):
            raise ValueError("portfolio_value Series requires a dated index")
        nav = pl.DataFrame(
            {
                "date": pl.from_pandas(portfolio_value.index),
                "equity": pl.from_pandas(portfolio_value.reset_index(drop=True)),
            }
        )
    else:
        nav = _rename(_frame(portfolio_value, timestamp_col), value_col, "equity")
    _temporal(nav)
    _validate_observation_keys(nav, ["date"], "beginning portfolio NAV")
    nav = _numeric(nav, "equity", nullable=True)
    feed = (
        FeedSpec.from_any(result.metadata["feed_spec"])
        if "feed_spec" in result.metadata
        else FeedSpec(timestamp_col="date", entity_col="asset", time_contract_version=2)
    )
    selection = calendar
    if selection is UNSET_CALENDAR and result.metadata.get("calendar_source") == "argument":
        selection = result.metadata["calendar_argument"]
    resolved = feed.resolve_calendar(calendar=selection)
    if alignment == "civil_date" and resolved is not None:
        raise ValueError(
            "civil_date is a separate legacy date profile; explicitly set calendar=None"
        )
    trips = completed_trades(result, include_synthetic=include_synthetic)
    if alignment == "civil_date":
        nav_keys = nav["date"].dt.date()
        trip_keys = trips["exit_timestamp"].dt.date().alias("date")
    else:
        nav_keys = _clock(nav["date"], resolved, feed.timestamp_semantics, timezone=feed.timezone)
        trip_keys = (
            _clock(trips["exit_timestamp"], resolved, None, executions=True, timezone=feed.timezone)
            if trips.height
            else pl.Series("date", [], dtype=nav_keys.dtype)
        )
        if resolved is None:
            if isinstance(nav_keys.dtype, pl.Datetime):
                nav_keys = pl.from_pandas(clock(nav_keys, "NAV timestamp", feed.timezone)).rename(
                    "date"
                )
            if isinstance(trip_keys.dtype, pl.Datetime) and isinstance(nav_keys.dtype, pl.Datetime):
                if trip_keys.dtype.time_zone is not None and nav_keys.dtype.time_zone is not None:
                    nav_keys = nav_keys.dt.convert_time_zone(trip_keys.dtype.time_zone)
    nav = nav.select("equity").with_columns(nav_keys.alias("date"))
    _validate_observation_keys(nav, ["date"], "aligned beginning portfolio NAV")
    trips = trips.with_columns(trip_keys.alias("date"))
    trips, nav = _align_datetime_key_resolution(trips, nav, "date")
    if trips["date"].dtype != nav["date"].dtype:
        raise ValueError(
            "NAV and closing-fill keys must match; do not manufacture date labels into event instants"
        )
    joined = trips.join(
        nav.rename({"equity": "portfolio_nav"}), on="date", how="left", maintain_order="left"
    )
    contributions = joined.with_columns(
        cash_fraction_expr(
            pl.col("gross_pnl"),
            pl.col("portfolio_nav"),
            positive_finite=nav_method != "legacy_ieee",
        ).alias("gross_nav_contribution"),
        cash_fraction_expr(
            pl.col("net_pnl"), pl.col("portfolio_nav"), positive_finite=nav_method != "legacy_ieee"
        ).alias("net_nav_contribution"),
        pl.when(pl.col("portfolio_nav").is_finite() & (pl.col("portfolio_nav") > 0))
        .then(pl.lit("defined_nav"))
        .otherwise(pl.lit("unavailable_nav"))
        .alias("normalization_status"),
        pl.lit("supplied beginning-period portfolio NAV including cash").alias("nav_basis"),
        pl.lit(alignment).alias("nav_alignment"),
        pl.lit(nav_method).alias("nav_method"),
        pl.lit(resolved, dtype=pl.String).alias("calendar"),
    ).rename({"date": "nav_observation"})
    for name in ("gross_nav_contribution", "net_nav_contribution"):
        if nav_method != "legacy_ieee" and not contributions[name].drop_nulls().is_finite().all():
            raise ValueError(
                "NAV contribution exceeds finite Float64; inspect PnL and supplied NAV"
            )
    return contributions
