"""Familiar Zipline tuple projection over the single native input normalizer."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import execution_value_expr
from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline


def extract_rets_pos_txn_from_zipline(
    backtest: pd.DataFrame,
    *,
    asset_metadata: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return daily pandas returns, complete marked positions/cash and actual fills.

    Native extraction validates once, preserving supplied account facts. The
    facade uses legacy civil-day labels (naive daily labels are designated UTC)
    but does not mutate the producer frame or discard complete all-cash dates.
    Actual execution instants retain timezone/ns. Raw integer IDs use the legacy
    cash-price-per-unit convention; supply asset_metadata for point multipliers.
    Asset objects are accepted without importing their producer package.
    """
    required = {"positions", "transactions", "ending_cash"}
    if missing := required.difference(backtest.columns):
        raise ValueError(f"Zipline tuple requires supplied columns: {sorted(missing)}")
    labels: dict[int, Any] = {}
    scalar_symbols: dict[int, Any] = {}
    declarations = dict(asset_metadata or {})
    for kind in ("positions", "transactions"):
        for records in backtest[kind]:
            if not isinstance(records, list | tuple):
                raise ValueError(f"Zipline {kind} must contain actual nested record lists")
            for record in records:
                source = record["sid"]
                if isinstance(source, Mapping):
                    identifier = source["sid"]
                    label = identifier
                elif isinstance(source, Integral) and not isinstance(source, bool):
                    identifier = int(source)
                    label = source
                    declarations.setdefault(identifier, {"multiplier": 1})
                else:
                    identifier = getattr(source, "sid", None)
                    label = source
                labels[identifier] = label
                if not isinstance(source, Mapping):
                    scalar_symbols[identifier] = source
    analysis = portfolio_analysis_from_zipline(backtest, calendar=None, asset_metadata=declarations)
    inputs = analysis.input_data
    assert inputs is not None
    dates = pd.DatetimeIndex(inputs.returns["date"].to_pandas()).normalize()
    if dates.tz is None:
        dates = dates.tz_localize("UTC")
    if not dates.is_unique:
        raise ValueError("Zipline daily tuple requires unique civil-day observations")
    dates = dates.rename(backtest.index.name)
    returns = pd.Series(analysis.returns, index=dates.copy(deep=True), name="returns")
    owned = inputs.positions
    assert owned is not None
    if {"asset", "value"}.issubset(owned.columns):
        positions = owned.to_pandas().pivot(index="date", columns="asset", values="value")
        positions = positions.reindex(pd.DatetimeIndex(inputs.returns["date"].to_pandas()))
        positions.index = dates.copy(deep=True)
        positions = positions.rename(columns=labels)
    else:
        positions = pd.DataFrame(index=dates.copy(deep=True))
    account = inputs.valuations
    assert account is not None
    positions["cash"] = account["cash"].to_numpy()
    positions = positions.fillna(0.0)
    positions.columns.name = "sid"
    positions.index.name = "index"
    fills = inputs.transactions
    assert fills is not None
    transactions = fills.to_pandas().rename(
        columns={"asset": "sid", "quantity": "amount", "timestamp": "dt"}
    )
    if "symbol" not in transactions:
        transactions["symbol"] = transactions["sid"]
    transactions["symbol"] = transactions["symbol"].astype(object)
    for identifier, symbol in scalar_symbols.items():
        mask = transactions["sid"] == identifier
        transactions.loc[mask, "symbol"] = pd.Series(
            [symbol] * int(mask.sum()), index=transactions.index[mask], dtype=object
        )
    transactions["symbol"] = pd.Series(transactions["symbol"].to_list())
    transactions["sid"] = transactions["sid"].map(labels)
    transactions["commission"] = pd.Series(fills["commission"].to_list())
    if "order_id" not in transactions:
        transactions["order_id"] = None
    # Legacy txn_dollars is signed price cash flow without a point multiplier.
    cash_values = fills.with_columns(pl.lit(1).alias("multiplier"))
    integral = fills["quantity"].dtype.is_integer() and fills["price"].dtype.is_integer()
    transactions["txn_dollars"] = cash_values.select(
        (-execution_value_expr(integral=integral)).alias("cash")
    )["cash"].to_list()
    transactions.index = pd.DatetimeIndex(transactions["dt"])
    if transactions.index.tz is None:
        transactions.index = transactions.index.tz_localize("UTC")
    transactions.index.name = None
    transactions = transactions.sort_index(kind="stable")
    return (
        returns,
        positions,
        transactions[
            ["sid", "symbol", "price", "order_id", "amount", "commission", "dt", "txn_dollars"]
        ],
    )


__all__ = ["extract_rets_pos_txn_from_zipline"]
