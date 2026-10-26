"""Familiar supplied-input projections and explicit native intraday estimates."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.portfolio_analysis.transaction_metrics import execution_value_expr
from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline


def detect_intraday(
    positions: pd.DataFrame, transactions: pd.DataFrame, threshold: float = 0.25
) -> bool:
    """Classify the original daily nonzero-position/unique-transacted-symbol ratio.

    This is a sample-count heuristic, not marked exposure or an intraday peak.
    No transactions gives False without an undefined-division warning.
    """
    from .txn import _dated

    _dated(positions, "positions")
    _dated(transactions, "transactions")
    daily_transacted = transactions.groupby(transactions.index.date)["symbol"].nunique().sum()
    held = positions.drop(columns="cash").replace(0, np.nan).count().sum()
    return bool(held / daily_transacted < threshold) if daily_transacted else False


def estimate_intraday(
    returns: pd.Series,
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    EOD_hour: int = 23,  # noqa: N803 - Published keyword spelling.
) -> pd.DataFrame:
    """Return a familiar explicitly estimated daily transaction-notional projection.

    Uses the native estimator, never a second financial implementation. Original
    NaN-position-to-zero behavior is an explicit facade profile. EOD_hour stays
    accepted but unused, as in published Pyfolio0.10.0. Returns are not recomputed.
    This estimate ignores fees and is not observed/reconstructed account state.
    """
    from ml4t.diagnostic.evaluation.portfolio_analysis import estimate_intraday_positions

    from .txn import _dated

    for value, name in (
        (returns, "returns"),
        (positions, "positions"),
        (transactions, "transactions"),
    ):
        _dated(value, name)
    if positions.columns.has_duplicates or "cash" not in positions:
        raise ValueError("positions require distinct asset columns and separate cash")
    if (transactions["symbol"] == "cash").any():
        raise ValueError(
            "cash is reserved account cash in the wide facade; use native long positions for that asset ID"
        )
    assets = [name for name in positions.columns if name != "cash"]
    codes, labels = pd.factorize(
        pd.Index([*assets, *transactions["symbol"].to_list()], dtype=object), sort=False
    )
    if (codes < 0).any():
        raise ValueError("position/transaction symbols must be nonmissing")
    names = {label: f"asset{index}" for index, label in enumerate(labels)}
    owned_positions = positions.rename(columns=names).fillna(0.0)
    owned_transactions = (
        transactions[["symbol", "amount", "price"]]
        .copy(deep=True)
        .rename(columns={"symbol": "asset", "amount": "quantity"})
    )
    owned_transactions["asset"] = transactions["symbol"].map(names).astype("string")
    if owned_transactions.empty:
        owned_transactions = owned_transactions.astype({"quantity": float, "price": float})
    if returns.empty:
        raise ValueError("returns must supply the first EOD period's return")
    first_return = None if pd.isna(returns.iloc[0]) else float(returns.iloc[0])
    native = estimate_intraday_positions(
        owned_positions,
        owned_transactions,
        first_return=first_return,
        complete_snapshots=True,
        calendar=None,
        method="legacy_peak_notional",
        capital_method="legacy_zero_fill",
    )
    output = pd.DataFrame(index=range(native.timeline.height))
    for index, label in enumerate(labels):
        output[label] = (
            native.positions.filter(pl.col("asset") == f"asset{index}")
            .sort("observation_id")["value"]
            .to_list()
        )
    output["cash"] = pd.Series(native.timeline["cash"].to_list(), dtype=float)
    index = pd.DatetimeIndex(native.timeline["date"].to_list())
    if positions.index.tz is not None:
        index = index.tz_localize(positions.index.tz)
    output.index = index.as_unit(positions.index.unit).rename("period_close")
    output.columns.name = "sid"
    output.attrs["diagnostic"] = {
        **native.metadata,
        "snapshot_kind": native.snapshot_kind,
        "legacy_position_missing_to_zero": int(positions.isna().to_numpy().sum()),
        "EOD_hour": EOD_hour,
        "EOD_hour_semantics": "accepted but unused, matching published0.10.0",
        "candidate_timestamps": [
            value.isoformat() if pd.notna(value) else None
            for value in pd.DatetimeIndex(native.timeline["candidate_timestamp"].to_pandas())
        ],
        "member_stop": native.timeline["member_stop"].to_list(),
    }
    return output


def check_intraday(
    estimate: bool | np.bool_ | str | None,
    returns: pd.Series,
    positions: pd.DataFrame | None,
    transactions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Apply the original opt-in/infer/disabled selector over the native estimate."""
    if estimate == "infer":
        if (
            positions is None
            or transactions is None
            or not detect_intraday(positions, transactions)
        ):
            return positions
        warnings.warn(
            "Detected intraday strategy; using the explicit daily notional estimate. Set estimate_intraday=False to retain supplied positions.",
            UserWarning,
            stacklevel=2,
        )
        return estimate_intraday(returns, positions, transactions)
    if estimate is None or (isinstance(estimate, bool | np.bool_) and not estimate):
        return positions
    if not isinstance(estimate, bool | np.bool_):
        raise ValueError("estimate must be True, False, None or 'infer'")
    if positions is None or transactions is None:
        raise ValueError("Positions and transactions are needed to estimate intraday holdings")
    return estimate_intraday(returns, positions, transactions)


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
    Portable descriptors use stable SIDs in both position columns and fill symbols.
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
                scalar_symbols[identifier] = label
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


__all__ = [
    "check_intraday",
    "detect_intraday",
    "estimate_intraday",
    "extract_rets_pos_txn_from_zipline",
]
