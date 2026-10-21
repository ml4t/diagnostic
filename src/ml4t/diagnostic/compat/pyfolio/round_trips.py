"""Pyfolio-shaped round trips over native owned executions and explicit VWAP/FIFO."""

from __future__ import annotations

from datetime import timedelta
from fractions import Fraction

import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.fifo import (
    RoundTripResult,
    reconstruct_round_trips,
)
from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_groups import group_executions
from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_inputs import clock
from ml4t.diagnostic.evaluation.portfolio_analysis.fifo_statistics import (
    DURATION_METRICS,
    PNL_METRICS,
    RETURN_METRICS,
    SUMMARY_METRICS,
    summarize_round_trips,
)
from ml4t.diagnostic.evaluation.portfolio_analysis.frame_inputs import normalize_executions

from .txn import _dated


def _inputs(transactions: pd.DataFrame) -> tuple[pl.DataFrame, pd.Index]:
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame")
    _dated(transactions, "transactions")
    if transactions.columns.has_duplicates:
        raise ValueError("transaction columns must be distinct")
    codes, labels = pd.factorize(transactions["symbol"], sort=True)
    if (codes < 0).any():
        raise ValueError("transaction symbols must be nonmissing")
    # The reference consumes these fields only and ignores commission/slippage.
    # Keep labels separately rather than coercing Asset objects or integer IDs to floats.
    owned = (
        transactions[["amount", "price"]]
        .reset_index(drop=True)
        .rename(columns={"amount": "quantity"})
    )
    owned["asset"] = codes
    owned["timestamp"] = transactions.index
    return normalize_executions(owned), labels


def _native_round_trips(transactions: pd.DataFrame) -> tuple[RoundTripResult, pd.Index]:
    observed, labels = _inputs(transactions)
    result = reconstruct_round_trips(observed, initial_state="flat", matching="consecutive_vwap")
    return result, labels


def _groupby_consecutive(
    txn: pd.DataFrame, max_delta: str | timedelta = timedelta(hours=8)
) -> pd.DataFrame:
    """Project explicit adjacent-gap analytical groups, preserving original labels.

    Exact units are retained without integer share expansion. Empty input returns
    an empty typed projection rather than failing concatenation. These VWAP rows
    are analytical groups, not actual fills; use the native result for membership
    and last-member availability before treating any value as known at an instant.
    """
    observed, labels = _inputs(txn)
    clocks = clock(observed["timestamp"], "transaction timestamp", None)
    order = sorted(range(observed.height), key=lambda index: (clocks[index].value, index))
    groups = group_executions(observed, clocks, order, max_delta)
    groups = groups.sort(["asset", "timestamp", "record_id"])
    quantities = [Fraction(value) for value in groups["quantity_exact"].to_list()]
    integral = pd.api.types.is_integer_dtype(txn["amount"].dtype)
    values = [
        int(value) if integral and value.denominator == 1 else float(value) for value in quantities
    ]
    table = pd.DataFrame({"amount": values, "price": groups["price"].to_list()})
    table["symbol"] = pd.Series(
        [labels[int(value)] for value in groups["asset"].to_list()], dtype=txn["symbol"].dtype
    )
    table.index = pd.DatetimeIndex(groups["timestamp"].to_pandas()).rename("dt")
    return table[["amount", "symbol", "price"]]


def extract_round_trips(
    transactions: pd.DataFrame, portfolio_value: pd.Series | None = None
) -> pd.DataFrame:
    """Return familiar gross-PnL closing-group rows using the single native matcher.

    The legacy profile declares an initially flat book and eight-hour consecutive
    VWAP grouping; it ignores commissions as the reference does. Use native FIFO
    for supplied openings, net costs, exact actual fills or source provenance.
    Profile prices are cash per unit; optional multiplier fields are ignored as
    in the reference. Native inputs retain actual point-price multipliers.
    portfolio_value is supplied beginning-period NAV, including cash, aligned by
    the explicit legacy civil-date convention. It is not the lot denominator.
    Legacy NAV division retains zero/negative-denominator NaN/Inf array behavior;
    native positive-NAV defaults remain separate. Zero entry-notional lot return
    stays undefined instead of reproducing the reference's division-by-zero crash.
    Fractional/negative-price reference defects are corrected, not reproduced.
    """
    result, labels = _native_round_trips(transactions)
    closed = result.completed_trades().sort(["asset", "exit_timestamp", "exit_group_id"])
    frame = (
        closed.select("gross_pnl", "entry_timestamp", "exit_timestamp", "gross_lot_return")
        .to_pandas()
        .rename(
            columns={
                "gross_pnl": "pnl",
                "entry_timestamp": "open_dt",
                "exit_timestamp": "close_dt",
                "gross_lot_return": "rt_returns",
            }
        )
    )
    frame["long"] = closed["direction"].to_numpy() == "long"
    frame["symbol"] = pd.Series(
        [labels[int(value)] for value in closed["asset"].to_list()], dtype=object
    )
    frame["duration"] = frame["close_dt"] - frame["open_dt"]
    frame = frame[["pnl", "open_dt", "close_dt", "long", "rt_returns", "symbol", "duration"]]
    if portfolio_value is not None:
        nav = result.portfolio_contributions(
            portfolio_value, calendar=None, alignment="civil_date", nav_method="legacy_ieee"
        ).sort(["asset", "exit_timestamp", "exit_group_id"])
        frame["returns"] = nav["gross_nav_contribution"].to_numpy()
    return frame


_PNL_LABELS = dict(
    zip(
        PNL_METRICS,
        (
            "Total profit",
            "Gross profit",
            "Gross loss",
            "Profit factor",
            "Avg. trade net profit",
            "Avg. winning trade",
            "Avg. losing trade",
            "Ratio Avg. Win:Avg. Loss",
            "Largest winning trade",
            "Largest losing trade",
        ),
        strict=True,
    )
)
_SUMMARY_LABELS = dict(
    zip(
        SUMMARY_METRICS,
        (
            "Total number of round_trips",
            "Percent profitable",
            "Winning round_trips",
            "Losing round_trips",
            "Even round_trips",
        ),
        strict=True,
    )
)
_DURATION_LABELS = dict(
    zip(
        DURATION_METRICS,
        ("Avg duration", "Median duration", "Longest duration", "Shortest duration"),
        strict=True,
    )
)
_RETURN_LABELS = dict(
    zip(
        RETURN_METRICS,
        (
            "Avg returns all round_trips",
            "Avg returns winning",
            "Avg returns losing",
            "Median returns all round_trips",
            "Median returns winning",
            "Median returns losing",
            "Largest winning trade",
            "Largest losing trade",
        ),
        strict=True,
    )
)


def _wide_statistics(
    table: pl.DataFrame, metric_labels: dict[str, str], asset_labels: pd.Index | None = None
) -> pd.DataFrame:
    if table.is_empty():
        result = pd.DataFrame(index=list(metric_labels))
    elif asset_labels is None:
        result = table.to_pandas().pivot(index="metric", columns="scope", values="value")
        order = [scope for scope in ("all", "short", "long") if scope in result.columns]
        result = result.reindex(index=list(metric_labels), columns=order).rename(
            columns={"all": "All trades", "short": "Short trades", "long": "Long trades"}
        )
    else:
        result = table.to_pandas().pivot(index="metric", columns="asset", values="value")
        result = result.reindex(index=list(metric_labels), columns=sorted(result.columns))
        result.columns = asset_labels.take([int(value) for value in result.columns])
    result.index = pd.Index([metric_labels[name] for name in result.index], name=None)
    result.columns.name = "symbol" if asset_labels is not None else None
    return result


def gen_round_trip_stats(round_trips: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Project all five original tables from native completed-observation statistics.

    returns must be supplied NAV-relative observations; rt_returns are not a
    substitute. Each metric retains its own missingness/sign sample. The explicit
    legacy count and all-loss profit-factor conventions do not change native
    TradeStatistics defaults. Duration remains typed, not invented zero days.
    """
    if not isinstance(round_trips, pd.DataFrame):
        raise TypeError("round_trips must be a pandas DataFrame")
    if round_trips.columns.has_duplicates:
        raise ValueError("round-trip columns must be distinct")
    if "returns" not in round_trips.columns:
        raise ValueError(
            "portfolio NAV-relative returns are required; pass portfolio_value to extraction, not rt_returns"
        )
    required = {"pnl", "duration", "symbol", "long"}
    if missing := required.difference(round_trips.columns):
        raise ValueError(f"round-trip statistics require columns: {sorted(missing)}")
    if not round_trips["long"].dropna().isin([True, False]).all():
        raise ValueError("long must contain true/false or unknown direction")
    codes, labels = pd.factorize(round_trips["symbol"], sort=True)
    projected = (
        round_trips[["pnl", "duration", "returns"]]
        .reset_index(drop=True)
        .rename(columns={"pnl": "gross_pnl", "returns": "gross_nav_contribution"})
    )
    projected["asset"] = pd.array(
        [int(code) if code >= 0 else None for code in codes], dtype="Int64"
    )
    projected["direction"] = pd.Series(
        [
            None if pd.isna(value) else "long" if bool(value) else "short"
            for value in round_trips["long"]
        ],
        dtype="string",
    )
    native = summarize_round_trips(
        pl.from_pandas(projected),
        pnl_col="gross_pnl",
        return_col="gross_nav_contribution",
        method="legacy_pyfolio",
    )
    results = {
        name: _wide_statistics(native.tables[name], mapping)
        for name, mapping in (
            ("pnl", _PNL_LABELS),
            ("summary", _SUMMARY_LABELS),
            ("duration", _DURATION_LABELS),
            ("returns", _RETURN_LABELS),
        )
    }
    results["symbols"] = _wide_statistics(native.tables["symbols"], _RETURN_LABELS, labels)
    coverage = native.coverage.to_dicts()
    for row in coverage:
        if row["asset"] is not None:
            row["asset"] = labels[int(row["asset"])]
    for table in results.values():
        table.attrs["diagnostic"] = {**native.metadata, "coverage": [dict(row) for row in coverage]}
    return results


__all__ = ["extract_round_trips", "gen_round_trip_stats"]
