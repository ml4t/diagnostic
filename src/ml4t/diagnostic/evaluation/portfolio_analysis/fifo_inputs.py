"""Validate supplied FIFO inventory and known-at quotes without inventing account state."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import polars as pl

from .frame_inputs import _execution_numeric, _frame


def instant(value: Any, name: str, timezone: str | None, aware: bool) -> pd.Timestamp:
    if (
        isinstance(value, (bool, int, float))
        or isinstance(value, date)
        and not isinstance(value, datetime)
    ):
        raise TypeError(f"{name} must be an explicit timestamp, not a date label or numeric value")
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError(f"{name} must be nonmissing")
        if result.tz is None and timezone is not None:
            result = result.tz_localize(timezone)
        if (result.tz is not None) != aware:
            raise ValueError(f"{name} and executions mix naive/aware instants without a timezone")
        _ = result.value  # Require representable nanosecond precision, not a truncated clock.
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def clock(
    values: pl.Series, name: str, timezone: str | None, aware: bool | None = None
) -> pd.DatetimeIndex:
    if not isinstance(values.dtype, pl.Datetime):
        raise TypeError(f"{name} must contain Datetime instants, not date labels")
    result = pd.DatetimeIndex(values.to_pandas())
    if result.tz is None and timezone is not None:
        result = result.tz_localize(timezone)
    if aware is not None and result.notna().any() and (result.tz is not None) != aware:
        raise ValueError(f"{name} and executions mix naive/aware instants without a timezone")
    return result


def _assets(frame: pl.DataFrame, dtype: pl.DataType) -> None:
    if "asset" not in frame.columns:
        raise ValueError("FIFO inputs require asset IDs")
    values = frame["asset"]
    if values.dtype != pl.String and not values.dtype.is_integer():
        raise TypeError("FIFO asset IDs must be stable strings or integers")
    if values.null_count() or values.dtype == pl.String and (values == "").any():
        raise ValueError("FIFO asset IDs must be nonmissing and nonempty")
    if (values.dtype == pl.String) != (dtype == pl.String):
        raise TypeError("FIFO asset IDs must use the same string/integer identity as executions")
    # No lossy float conversion or display-label substitution for stable integer IDs.
    if values.dtype != dtype:
        frame_values = values.cast(dtype, strict=True)
        if frame_values.to_list() != values.to_list():
            raise ValueError("FIFO asset IDs cannot be represented in the execution ID type")


def _optional_numeric(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    if name not in frame.columns or frame[name].null_count() == frame.height:
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    _execution_numeric(frame.filter(pl.col(name).is_not_null()), name)
    return frame


def opening_inventory(
    value: pl.DataFrame | pd.DataFrame,
    *,
    asset_dtype: pl.DataType,
    timezone: str | None,
    aware: bool,
    opening_as_of: pd.Timestamp,
) -> tuple[pl.DataFrame, pd.DatetimeIndex]:
    if isinstance(value, pl.DataFrame):
        frame = value.clone()
    elif isinstance(value, pd.DataFrame):
        frame = pl.from_pandas(value)
    else:
        raise TypeError("opening_lots must be a pandas or Polars DataFrame")
    _assets(frame, asset_dtype)
    _execution_numeric(frame, "quantity")
    if (frame["quantity"] == 0).any():
        raise ValueError("opening lots must have nonzero remaining quantity")
    if "lot_id" not in frame.columns:
        frame = frame.with_row_index("lot_id")
    ids = frame["lot_id"]
    if ids.dtype != pl.String and not ids.dtype.is_integer():
        raise TypeError("opening lot_id must contain stable strings or integers")
    if (
        ids.null_count()
        or ids.n_unique() != frame.height
        or ids.dtype == pl.String
        and (ids == "").any()
    ):
        raise ValueError("opening lot_id must be nonmissing and unique")
    for name in ("entry_price", "remaining_commission"):
        frame = _optional_numeric(frame, name)
    if "multiplier" not in frame.columns:
        frame = frame.with_columns(pl.lit(1.0).alias("multiplier"))
    _execution_numeric(frame, "multiplier")
    if (frame["multiplier"] <= 0).any():
        raise ValueError("opening multipliers must be strictly positive")
    if "entry_timestamp" not in frame.columns or frame["entry_timestamp"].dtype == pl.Null:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Datetime("ns", str(opening_as_of.tz) if aware else None)).alias(
                "entry_timestamp"
            )
        )
    clocks = clock(frame["entry_timestamp"], "opening entry_timestamp", timezone, aware)
    if any(value > opening_as_of for value in clocks if pd.notna(value)):
        raise ValueError("opening lots cannot have future entry timestamps after opening_as_of")
    return frame, clocks


def mark_inventory(
    value: pl.DataFrame | pd.DataFrame,
    *,
    asset_dtype: pl.DataType,
    timezone: str | None,
    aware: bool,
) -> tuple[pl.DataFrame, pd.DatetimeIndex]:
    frame = _frame(value, "timestamp", "timestamp")
    _assets(frame, asset_dtype)
    _execution_numeric(frame, "price")
    clocks = clock(frame["timestamp"], "mark timestamp", timezone, aware)
    if clocks.isna().any():
        raise ValueError("mark timestamps must be nonmissing")
    keys = list(zip(frame["asset"].to_list(), [value.value for value in clocks], strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate asset/mark timestamp is ambiguous")
    return frame, clocks


def price_dtype(fills: pl.DataFrame, opening: pl.DataFrame | None) -> pl.DataType:
    """Choose a common numeric projection without silently rounding integer prices."""
    if opening is None or not opening["entry_price"].drop_nulls().len():
        return fills["price"].dtype
    left, right = fills["price"], opening["entry_price"].drop_nulls().rename("price")
    joined = pl.concat([left.to_frame(), right.to_frame()], how="vertical_relaxed")["price"]
    if joined.dtype.is_float():
        for values in (left, right):
            if values.dtype.is_integer() and any(
                int(float(value)) != value for value in values.to_list()
            ):
                raise ValueError(
                    "mixed opening/execution price types would round integer prices; "
                    "supply a common lossless numeric dtype"
                )
    return joined.dtype
