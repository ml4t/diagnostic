"""Closing-fill observations and existing trade-statistics integration over FIFO results."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis

    from .fifo import RoundTripResult


def completed_trades(result: RoundTripResult, *, include_synthetic: bool) -> pl.DataFrame:
    from .fifo import _cash, _frame, _known_sum

    if not isinstance(include_synthetic, bool):
        raise TypeError("include_synthetic must be a boolean")

    groups = defaultdict(list)
    sources = [(result.segments, False)]
    if include_synthetic:
        if result.synthetic_segments is None:
            raise ValueError("synthetic segments were not requested during reconstruction")
        sources.append((result.synthetic_segments, True))
    for frame, synthetic in sources:
        rows = frame.with_columns(
            pl.col("entry_timestamp", "exit_timestamp").cast(pl.Int64)
        ).to_dicts()
        for row in rows:
            groups[(row["asset"], row["exit_record_id"], row["direction"], synthetic)].append(row)
    records = []
    for (asset, exit_id, direction, synthetic), rows in groups.items():
        quantities = [Fraction(row["quantity_exact"]) for row in rows]
        quantity = sum(quantities, Fraction(0))
        cash = {
            name: _known_sum(
                [
                    Fraction(row[name + "_exact"]) if row[name + "_exact"] is not None else None
                    for row in rows
                ]
            )
            for name in ("gross_pnl", "net_pnl", "commission", "entry_notional")
        }
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
            "exit_record_id": exit_id,
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
        }
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
    # Synthetic quotes can have a different lossless input dtype from actual exits.
    if include_synthetic and result.synthetic_segments is not None:
        from .fifo_inputs import price_dtype

        schema["exit_price"] = price_dtype(
            result.segments.rename({"exit_price": "price"}),
            result.synthetic_segments.rename(
                {"exit_price": "entry_price", "entry_price": "original_entry_price"}
            ),
        )
    return _frame(
        records,
        schema,
        ["entry_timestamp", "exit_timestamp"],
        result.segments["exit_timestamp"].dtype.time_zone,
    )


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
        "observation": "one closing fill per asset/direction, not one matched entry lot",
        "pnl": pnl,
        "missing": missing,
        "n_total_observations": frame.height,
        "n_excluded_unknown_pnl": excluded,
        "include_synthetic": include_synthetic,
        "return_denominator": "summed absolute matched entry notional, not portfolio NAV",
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
        raise ValueError("confidence must be a finite real number strictly between0and1")
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
