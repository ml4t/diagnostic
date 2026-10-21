"""Owned account frames without inferring returns, executions or unreported NAV."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis.holdings import _numeric, analyze_holdings
from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import date_snapshot, return_snapshot
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)

_TABLES = ("returns", "benchmark", "positions", "valuations", "transactions")


@dataclass(frozen=True)
class PortfolioInputData:
    """Immutable Arrow snapshots of reported inputs with fresh frame accessors.

    Absent tables remain None, not observed empty/zero ledgers. Returns are
    authoritative observations, never reconstructed from NAV. Normalized long
    positions retain signed market values and stable asset IDs; an all-cash
    snapshot retains date rows without inventing a cash asset. Transactions
    retain signed quantities, multipliers, original fields and unknown fees.
    """

    _tables: tuple[bytes | None, ...] = field(repr=False)
    _metadata: str = field(repr=False)

    def _table(self, name: str) -> pl.DataFrame | None:
        payload = self._tables[_TABLES.index(name)]
        return pl.read_ipc(BytesIO(payload)) if payload is not None else None

    @property
    def returns(self) -> pl.DataFrame:
        """Return canonical date/return observations, preserving exact key types."""
        result = self._table("returns")
        assert result is not None
        return result

    @property
    def benchmark(self) -> pl.DataFrame | None:
        """Return the original dated or explicitly positional benchmark snapshot."""
        return self._table("benchmark")

    @property
    def positions(self) -> pl.DataFrame | None:
        """Return signed positions, or None when no holdings were reported."""
        return self._table("positions")

    @property
    def valuations(self) -> pl.DataFrame | None:
        """Return reported cash/NAV, including any supplied initial valuation."""
        return self._table("valuations")

    @property
    def transactions(self) -> pl.DataFrame | None:
        """Return actual supplied records without synthesizing fills from weights."""
        return self._table("transactions")

    @property
    def metadata(self) -> dict:
        """Return independent declarations, input roles and normalization provenance."""
        return json.loads(self._metadata)

    def to_json(self) -> str:
        """Serialize owned typed tables and strict JSON declarations."""
        return json.dumps(
            {
                "format_version": 1,
                "tables": {
                    name: base64.b64encode(payload).decode("ascii") if payload is not None else None
                    for name, payload in zip(_TABLES, self._tables, strict=True)
                },
                "metadata": self.metadata,
            },
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> PortfolioInputData:
        """Restore versioned Arrow snapshots, preserving absence and empty tables."""
        parsed = json.loads(value)
        if parsed.get("format_version") != 1:
            raise ValueError("unsupported portfolio input format_version")
        tables = []
        for name in _TABLES:
            encoded = parsed["tables"][name]
            payload = base64.b64decode(encoded, validate=True) if encoded is not None else None
            if payload is not None:
                pl.read_ipc(BytesIO(payload))
            elif name == "returns":
                raise ValueError("portfolio inputs require returns")
            tables.append(payload)
        return cls(tuple(tables), json.dumps(parsed["metadata"], allow_nan=False))


def _rename(frame: pl.DataFrame, source: str, target: str) -> pl.DataFrame:
    if source not in frame.columns:
        raise ValueError(f"portfolio frame is missing declared column {source!r}")
    if source != target:
        if target in frame.columns:
            raise ValueError(f"ambiguous input columns {source!r} and {target!r}")
        frame = frame.rename({source: target})
    return frame


def _frame(
    value: pl.DataFrame | pd.DataFrame, timestamp_col: str, key: str = "date"
) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        result = value.clone()
    elif isinstance(value, pd.DataFrame):
        if isinstance(value.index, pd.MultiIndex):
            value = value.reset_index()
        elif isinstance(value.index, pd.DatetimeIndex):
            if timestamp_col not in value.columns:
                value = value.rename_axis(timestamp_col).reset_index()
            else:
                keys = pd.DatetimeIndex(value[timestamp_col])
                if keys.tz != value.index.tz or not (keys == value.index).all():
                    raise ValueError("the datetime index and timestamp column must agree")
        result = pl.from_pandas(value)
    else:
        raise TypeError("portfolio frames must be pandas or Polars DataFrames")
    if timestamp_col not in result.columns:
        raise ValueError(f"portfolio frame is missing timestamp column {timestamp_col!r}")
    return _rename(result, timestamp_col, key)


def _temporal(frame: pl.DataFrame) -> None:
    dtype = frame["date"].dtype
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        raise TypeError("portfolio snapshot dates must contain Date or Datetime values")


def _execution_numeric(frame: pl.DataFrame, column: str) -> None:
    if column not in frame.columns or not frame[column].dtype.is_numeric():
        raise TypeError(f"transaction {column} must contain real numeric values")
    values = frame[column]
    if values.null_count() or values.dtype.is_float() and not values.is_finite().all():
        raise ValueError(f"transaction {column} must be finite and nonmissing")


def normalize_executions(
    transactions: pl.DataFrame | pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    entity_col: str = "asset",
    transaction_multiplier: float = 1.0,
) -> pl.DataFrame:
    """Own actual executions independently of any return series or NAV observations."""
    if (
        isinstance(transaction_multiplier, bool)
        or not isinstance(transaction_multiplier, Real)
        or not np.isfinite(transaction_multiplier)
        or transaction_multiplier <= 0
    ):
        raise ValueError("transaction_multiplier must be finite and positive")
    executions = _rename(_frame(transactions, timestamp_col, "timestamp"), entity_col, "asset")
    _validate_observation_keys(executions, ["timestamp", "asset"], "transactions", unique=False)
    if executions["asset"].dtype != pl.String and not executions["asset"].dtype.is_integer():
        raise TypeError("transaction asset IDs must be stable strings or integers")
    if executions["asset"].dtype == pl.String and (executions["asset"] == "").any():
        raise ValueError("transaction asset IDs must be nonempty")
    if not isinstance(executions["timestamp"].dtype, pl.Datetime):
        raise TypeError(
            "transaction timestamps must be Datetime values, not manufactured date instants"
        )
    for column in ("quantity", "price"):
        _execution_numeric(executions, column)
    if "side" in executions.columns:
        side = executions["side"].cast(pl.String).str.to_lowercase()
        if (
            not side.is_in(["buy", "sell"]).all()
            or ((side == "buy") & (executions["quantity"] < 0)).any()
        ):
            raise ValueError("transaction side must be buy/sell and agree with signed quantity")
        quantity = pl.col("quantity")
        if executions["quantity"].dtype.is_unsigned_integer():
            quantity = quantity.cast(pl.Int128)
            negative_quantity = pl.lit(0, dtype=pl.Int128) - quantity
        else:
            negative_quantity = -quantity
        executions = executions.with_columns(
            pl.when(side == "sell")
            .then(pl.when(quantity > 0).then(negative_quantity).otherwise(quantity))
            .otherwise(quantity)
            .alias("quantity")
        )
    if "multiplier" not in executions.columns:
        executions = executions.with_columns(
            pl.lit(float(transaction_multiplier)).alias("multiplier")
        )
    _execution_numeric(executions, "multiplier")
    if (executions["multiplier"] <= 0).any():
        raise ValueError("transaction multipliers must be strictly positive")
    for column in ("commission", "slippage"):
        if column not in executions.columns:
            executions = executions.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        executions = _numeric(executions, column, nullable=True)
    if "record_id" not in executions.columns:
        executions = executions.with_row_index("record_id")
    _validate_observation_keys(executions, ["record_id"], "transaction record IDs")
    return executions


def _observations(
    value: Any, timestamp_col: str, value_col: str, *, benchmark: bool = False
) -> pl.DataFrame:
    if benchmark and (
        isinstance(value, np.ndarray | pl.Series | list | tuple)
        or isinstance(value, pd.Series)
        and not isinstance(value.index, pd.DatetimeIndex)
    ):
        return pl.DataFrame({"return": return_snapshot(value, "benchmark")})
    if isinstance(value, pd.Series):
        if not isinstance(value.index, pd.DatetimeIndex):
            raise ValueError("dated portfolio Series require a DatetimeIndex")
        frame = pl.DataFrame(
            {
                "date": pl.from_pandas(value.index),
                "return": pl.from_pandas(value.reset_index(drop=True)),
            }
        )
    else:
        frame = _rename(_frame(value, timestamp_col), value_col, "return")
    if "return" not in frame.columns:
        raise ValueError(f"portfolio observations require return column {value_col!r}")
    dates = date_snapshot(
        frame["date"],
        frame.height,
        "benchmark dates" if benchmark else "dates",
        ordered=not benchmark,
    )
    values = return_snapshot(frame["return"], "benchmark" if benchmark else "returns")
    return pl.DataFrame({"date": dates, "return": values})


def _positions(
    value: Any, timestamp_col: str, entity_col: str, layout: str
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if (
        layout == "auto"
        and isinstance(value, pd.DataFrame | pl.DataFrame)
        and entity_col != "asset"
        and entity_col not in value.columns
        and {"asset", "value"}.issubset(value.columns)
    ):
        raise ValueError(
            f"positions are missing declared entity column {entity_col!r}; use positions_format='wide' only for actual wide asset columns"
        )
    if (
        isinstance(value, pd.DataFrame)
        and (isinstance(value.index, pd.DatetimeIndex) or timestamp_col in value.columns)
        and (
            layout == "wide"
            or layout == "auto"
            and not {entity_col, "value"}.issubset(value.columns)
        )
    ):
        assets = value
        if isinstance(value.index, pd.DatetimeIndex):
            dates = pl.from_pandas(value.index).rename("date")
            if timestamp_col in value.columns:
                declared = pl.from_pandas(value[timestamp_col])
                if declared.dtype == pl.Date or isinstance(declared.dtype, pl.Datetime):
                    labels = pd.DatetimeIndex(value[timestamp_col])
                    if labels.tz != value.index.tz or not (labels == value.index).all():
                        raise ValueError("the datetime index and timestamp column must agree")
                    assets = assets.drop(columns=timestamp_col)
        else:
            dates = pl.from_pandas(value[timestamp_col]).rename("date")
            assets = assets.drop(columns=timestamp_col)
        _temporal(pl.DataFrame({"date": dates}))
        cash = (
            pl.DataFrame(
                {
                    "date": dates,
                    "cash": pl.from_pandas(value["cash"].reset_index(drop=True)),
                }
            )
            if "cash" in value.columns
            else None
        )
        assets = assets.drop(columns="cash") if "cash" in assets.columns else assets
        if not len(assets.columns):
            frame = pl.DataFrame({"date": dates})
            _validate_observation_keys(frame, ["date"], "all-cash snapshots")
            return frame, cash
        identifiers = pl.Series("asset", assets.columns.tolist())
        count = len(identifiers)
        frame = pl.DataFrame(
            {
                "date": dates.gather(np.repeat(np.arange(len(value)), count)),
                "asset": identifiers.gather(np.tile(np.arange(count), len(value))),
                "value": assets.to_numpy().reshape(-1),
            }
        )
    else:
        frame = _frame(value, timestamp_col)
        _temporal(frame)
        if (
            layout == "wide"
            or layout == "auto"
            and not {entity_col, "value"}.issubset(frame.columns)
        ):
            cash = frame.select("date", "cash") if "cash" in frame.columns else None
            columns = [name for name in frame.columns if name not in ("date", "cash")]
            if not columns:
                _validate_observation_keys(frame, ["date"], "all-cash snapshots")
                return frame.select("date"), cash
            frame = frame.unpivot(on=columns, index="date", variable_name="asset")
        else:
            frame = _rename(frame, entity_col, "asset")
            cash = None
    _validate_observation_keys(frame, ["date", "asset"], "positions")
    if frame["asset"].dtype != pl.String and not frame["asset"].dtype.is_integer():
        raise TypeError("position asset IDs must be stable strings or integers")
    if frame["asset"].dtype == pl.String and (frame["asset"] == "").any():
        raise ValueError("position asset IDs must be nonempty")
    return _numeric(frame, "value"), cash


def _merge_cash(valuations: pl.DataFrame | None, cash: pl.DataFrame | None) -> pl.DataFrame | None:
    if cash is None:
        return valuations
    cash = _numeric(cash, "cash", nullable=True)
    _validate_observation_keys(cash, ["date"], "reported cash")
    if valuations is None:
        return cash.with_columns(pl.lit(None, dtype=pl.Float64).alias("equity"))
    valuations, cash = _align_datetime_key_resolution(valuations, cash, "date")
    combined = valuations.join(
        cash.rename({"cash": "__position_cash"}), on="date", how="full", coalesce=True
    )
    if (
        combined["cash"].is_not_null()
        & combined["__position_cash"].is_not_null()
        & (combined["cash"] != combined["__position_cash"])
    ).any():
        raise ValueError("cash in positions must agree with reported valuation cash")
    return (
        combined.with_columns(pl.coalesce("cash", "__position_cash").alias("cash"))
        .drop("__position_cash")
        .sort("date")
    )


def normalize_portfolio_frames(
    returns: Any,
    *,
    benchmark: Any,
    positions: Any,
    valuations: Any,
    transactions: Any,
    feed_spec: FeedSpec | Mapping[str, Any] | None,
    timestamp_col: str | None,
    entity_col: str | None,
    return_col: str,
    benchmark_col: str,
    transaction_timestamp_col: str | None,
    transaction_multiplier: float,
    calendar: str | CalendarUnset | None,
    complete_snapshots: bool,
    snapshot_kind: Literal["observed", "end_of_day", "intraday"],
    positions_format: str,
) -> PortfolioInputData:
    """Normalize declared roles once without provider or broker dependencies."""
    if not isinstance(complete_snapshots, bool):
        raise ValueError("complete_snapshots must be boolean")
    if snapshot_kind not in ("observed", "end_of_day", "intraday"):
        raise ValueError("snapshot_kind must be observed, end_of_day or intraday")
    if positions_format not in ("auto", "long", "wide"):
        raise ValueError("positions_format must be auto, long or wide")
    if (
        isinstance(transaction_multiplier, bool)
        or not isinstance(transaction_multiplier, Real)
        or not np.isfinite(transaction_multiplier)
        or transaction_multiplier <= 0
    ):
        raise ValueError("transaction_multiplier must be finite and positive")
    feed = (
        FeedSpec.from_any(feed_spec)
        if feed_spec is not None
        else FeedSpec(timestamp_col="date", entity_col="asset", time_contract_version=2)
    )
    timestamp = timestamp_col if timestamp_col is not None else feed.timestamp_col
    entity = entity_col if entity_col is not None else feed.entity_col or "asset"
    if not isinstance(entity, str) and isinstance(entity, Sequence) and len(entity) == 1:
        entity = entity[0]
    if (
        not isinstance(timestamp, str)
        or not isinstance(entity, str)
        or not timestamp
        or not entity
        or timestamp == entity
    ):
        raise ValueError(
            "portfolio snapshots require distinct timestamp and single-entity column names"
        )
    for name, column in (("return_col", return_col), ("benchmark_col", benchmark_col)):
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{name} must name a nonempty column")
    if transaction_timestamp_col is not None and (
        not isinstance(transaction_timestamp_col, str) or not transaction_timestamp_col.strip()
    ):
        raise ValueError("transaction_timestamp_col must name a nonempty column")
    observations = _observations(returns, timestamp, return_col)
    relative = (
        _observations(benchmark, timestamp, benchmark_col, benchmark=True)
        if benchmark is not None
        else None
    )
    holdings, cash = (
        _positions(positions, timestamp, entity, positions_format)
        if positions is not None
        else (None, None)
    )
    account = _frame(valuations, timestamp) if valuations is not None else None
    if account is not None:
        _temporal(account)
        _validate_observation_keys(account, ["date"], "valuations")
        for column in ("equity", "cash"):
            if column not in account.columns:
                account = account.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
            account = _numeric(account, column, nullable=True)
    account = _merge_cash(account, cash)
    executions = None
    if transactions is not None:
        tx_timestamp = transaction_timestamp_col or (
            feed.timestamp_col if feed_spec is not None else "timestamp"
        )
        executions = normalize_executions(
            transactions,
            timestamp_col=tx_timestamp,
            entity_col=entity,
            transaction_multiplier=transaction_multiplier,
        )
    if complete_snapshots and holdings is not None:
        selected_account = account if account is not None else observations.select("date")
        selected_account, keys = _align_datetime_key_resolution(
            selected_account, observations.select("date"), "date"
        )
        selected_account = selected_account.join(keys, on="date", how="semi").sort("date")
        analyze_holdings(
            holdings,
            selected_account,
            observations["date"],
            complete_snapshots=True,
            snapshot_kind=snapshot_kind,
        )
    wire = feed.to_dict()
    declared = "calendar" in wire and (
        feed.time_contract_version == 2 or wire["calendar"] is not None
    )
    metadata = {
        "normalization_version": 1,
        "feed_spec": wire,
        "timestamp_col": timestamp,
        "entity_col": entity,
        "calendar": feed.resolve_calendar(calendar=calendar),
        "calendar_source": "argument"
        if calendar is not UNSET_CALENDAR
        else "feed_spec"
        if declared
        else "default",
        "complete_snapshots": complete_snapshots,
        "snapshot_kind": snapshot_kind,
        "returns_kind": "net",
        "benchmark_alignment": None
        if relative is None
        else "dated keys"
        if "date" in relative.columns
        else "explicit positional",
        "positions_unit": "signed reported cash value",
        "missing_commission": "unknown, not zero",
        "unreported_NAV": "unknown, never inferred from holdings/cash",
        "default_transaction_multiplier": float(transaction_multiplier),
        "transaction_order": "supplied row order; no exchange priority inferred",
        "normalized_timestamp_col": "date",
        "normalized_entity_col": "asset",
        "calendar_validation": "not performed; supplied keys are retained",
    }
    if calendar is not UNSET_CALENDAR:
        metadata["calendar_argument"] = calendar
    tables = (observations, relative, holdings, account, executions)
    return PortfolioInputData(
        tuple(value.write_ipc(None).getvalue() if value is not None else None for value in tables),
        json.dumps(metadata, allow_nan=False),
    )
