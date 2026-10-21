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
from ml4t.diagnostic.evaluation.portfolio_analysis.frame_inputs import normalize_executions
from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import cash_fraction_expr

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
    Legacy IEEE division retains zero/negative-denominator NaN/Inf array behavior;
    native positive-NAV/defined-lot policies remain separate defaults.
    Fractional/negative-price reference defects are corrected, not reproduced.
    """
    result, labels = _native_round_trips(transactions)
    closed = (
        result.completed_trades()
        .sort(["asset", "exit_timestamp", "exit_group_id"])
        .with_columns(
            cash_fraction_expr(
                pl.col("gross_pnl"), pl.col("entry_notional"), positive_finite=False
            ).alias("legacy_lot_return")
        )
    )
    frame = (
        closed.select("gross_pnl", "entry_timestamp", "exit_timestamp", "legacy_lot_return")
        .to_pandas()
        .rename(
            columns={
                "gross_pnl": "pnl",
                "entry_timestamp": "open_dt",
                "exit_timestamp": "close_dt",
                "legacy_lot_return": "rt_returns",
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


__all__ = ["extract_round_trips"]
