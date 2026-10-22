"""Pyfolio transaction conventions over shared native financial calculations.

These helpers use legacy civil-date grouping, closing position value and the
zero-seeded first AGB denominator. They do not change native calendar=None row
mode or native beginning-NAV/unknown-first-AGB defaults. Prices here are already
cash per execution unit; use native owned inputs for contract point multipliers.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.holdings import _exposure_aggregations
from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import (
    absolute_quantity_expr,
    adjusted_return_expr,
    average_gross_book,
    cash_fraction_expr,
    execution_value_expr,
    filled_notional_expr,
)


def _dated(frame: pd.DataFrame | pd.Series, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} requires a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError(f"{name} timestamps must be nonmissing")


def map_transaction(txn: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Project familiar transaction labels without changing the source record."""
    source = txn["sid"]
    sid, symbol = (
        (source["sid"], source["symbol"]) if isinstance(source, Mapping) else (source, source)
    )
    return {
        "sid": sid,
        "symbol": symbol,
        **{name: txn[name] for name in ("price", "order_id", "amount", "commission", "dt")},
    }


def make_transaction_frame(transactions: pd.Series) -> pd.DataFrame:
    """Flatten actual nested transactions, preserving stable same-instant order.

    Empty inputs return a typed empty ledger instead of the reference's missing
    column error. Source instants are retained rather than losing timezone or ns
    through a datetime ndarray. No record is inferred from changing positions.
    """
    _dated(transactions, "transactions")
    records = []
    for values in transactions:
        if not isinstance(values, list | tuple):
            raise ValueError("transactions must contain actual nested record lists")
        records.extend(map_transaction(value) for value in values)
    columns = ["sid", "symbol", "price", "order_id", "amount", "commission", "dt"]
    result = pd.DataFrame(records, columns=columns)
    if not records:
        result.index = pd.DatetimeIndex([], tz=transactions.index.tz)
        return result.assign(txn_dollars=pd.Series([], dtype=float))
    result = result.sort_values("dt", kind="stable")
    result.index = pd.DatetimeIndex(pd.to_datetime(result["dt"]))
    result.index.name = None
    values = (
        pl.from_pandas(result[["amount", "price"]])
        .rename({"amount": "quantity"})
        .with_columns(pl.lit(1).alias("multiplier"))
    )
    integral = values["quantity"].dtype.is_integer() and values["price"].dtype.is_integer()
    result["txn_dollars"] = values.select((-execution_value_expr(integral=integral)).alias("cash"))[
        "cash"
    ].to_list()
    return result


def get_txn_vol(transactions: pd.DataFrame) -> pd.DataFrame:
    """Civil-date dollar/share activity, with independent amount/price missingness.

    The signed-price reference convention is explicit. Native actual-fill
    notional instead takes the absolute price and honors declared multipliers.
    Integer units use a widened accumulator rather than signed64-bit overflow.
    """
    _dated(transactions, "transactions")
    values = transactions[["amount", "price"]].copy()
    values.columns = ["quantity", "price"]
    frame = pl.from_pandas(values, nan_to_null=True).with_columns(
        pl.from_pandas(transactions.index.normalize()).rename("date"),
        pl.lit(1).alias("multiplier"),
    )
    integral = frame["quantity"].dtype.is_integer() and frame["price"].dtype.is_integer()
    grouped = (
        frame.group_by("date")
        .agg(
            filled_notional_expr(absolute_price=False, integral=integral).sum().alias("txn_volume"),
            absolute_quantity_expr(frame["quantity"].dtype).sum().alias("txn_shares"),
        )
        .sort("date")
    )
    index = pd.DatetimeIndex(grouped["date"].to_pandas()).rename(transactions.index.name)
    return pd.DataFrame(
        {name: grouped[name].to_list() for name in ("txn_volume", "txn_shares")}, index=index
    )


def _books(positions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if positions.columns.has_duplicates:
        raise ValueError("position columns must identify distinct assets")
    totals = []
    risky = []
    for index, column in enumerate(positions.columns):
        values = pl.from_pandas(positions.iloc[:, index], nan_to_null=True)
        if not values.dtype.is_numeric():
            raise TypeError("position values must be numeric")
        frame = pl.DataFrame({"row": range(len(positions)), "value": values.cast(pl.Float64)})
        totals.append(frame)
        if column != "cash":
            risky.append(frame)
    clock = pl.DataFrame({"row": range(len(positions))})
    gross = clock.join(
        pl.concat(risky).group_by("row").agg(_exposure_aggregations())
        if risky
        else clock.with_columns(pl.lit(0.0).alias("gross_exposure")),
        on="row",
        how="left",
    ).sort("row")["gross_exposure"]
    net = clock.join(
        pl.concat(totals).group_by("row").agg(pl.col("value").sum())
        if totals
        else clock.with_columns(pl.lit(0.0).alias("value")),
        on="row",
        how="left",
    ).sort("row")["value"]
    return pd.Series(gross.to_numpy(), index=positions.index), pd.Series(
        net.to_numpy(), index=positions.index
    )


def get_turnover(
    positions: pd.DataFrame, transactions: pd.DataFrame, denominator: str = "AGB"
) -> pd.Series:
    """Legacy gross traded value over zero-seeded AGB or closing position value."""
    if denominator not in ("AGB", "portfolio_value"):
        raise ValueError("denominator must be AGB or portfolio_value")
    if denominator == "AGB" and "cash" not in positions.columns:
        raise KeyError("cash")
    _dated(positions, "positions")
    gross, nav = _books(positions)
    if denominator == "AGB":
        nav = pd.Series(
            average_gross_book(pl.DataFrame({"gross": gross.to_numpy()}), initial_gross=0.0)[
                "denominator"
            ].to_numpy(),
            index=positions.index,
        )
    nav.index = nav.index.normalize()
    traded, nav = get_txn_vol(transactions)["txn_volume"].astype(float).align(nav, join="outer")
    frame = pl.DataFrame(
        {"notional": traded.to_numpy(dtype=float), "denominator": nav.to_numpy(dtype=float)}
    )
    values = frame.select(
        cash_fraction_expr(pl.col("notional"), pl.col("denominator"), positive_finite=False).alias(
            "turnover"
        )
    )["turnover"].to_numpy()
    return pd.Series(values, index=traded.index).fillna(0.0)


def adjust_returns_for_slippage(
    returns: pd.Series, positions: pd.DataFrame, transactions: pd.DataFrame, slippage_bps: float
) -> pd.Series:
    """Legacy extra-bps penalty on closing position value, without observed fees."""
    _dated(returns, "returns")
    if (
        isinstance(slippage_bps, bool)
        or not isinstance(slippage_bps, Real)
        or not np.isfinite(slippage_bps)
    ):
        raise ValueError("slippage_bps must be a finite real value")
    _dated(positions, "positions")
    _, nav = _books(positions)
    if nav.index.has_duplicates:
        grouped = nav.groupby(level=0, sort=False)
        if (grouped.nunique(dropna=False) > 1).any():
            raise ValueError(
                "duplicate position dates disagree on the closing NAV cost denominator"
            )
        # Tied exposure candidates are not additional return periods or duplicate costs.
        nav = grouped.agg(lambda values: values.iloc[0])
    costs = get_txn_vol(transactions)["txn_volume"] * (0.0001 * slippage_bps)
    index = returns.index.union(nav.index).union(costs.index)
    frame = pl.DataFrame(
        {
            "return": returns.reindex(index).to_numpy(dtype=float),
            "nav": nav.reindex(index).to_numpy(dtype=float),
            "cost": costs.reindex(index).to_numpy(dtype=float),
        }
    )
    values = frame.select(
        adjusted_return_expr(
            pl.col("return"), pl.col("cost"), pl.col("nav"), method="legacy_pnl"
        ).alias("adjusted")
    )["adjusted"].to_numpy()
    result = pd.Series(values, index=index.copy(deep=True))
    result.index.freq = None
    return result


__all__ = [
    "map_transaction",
    "make_transaction_frame",
    "get_txn_vol",
    "get_turnover",
    "adjust_returns_for_slippage",
]
