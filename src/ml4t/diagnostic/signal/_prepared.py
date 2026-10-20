"""Strict ingestion of already-complete signal panels, without return calculation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from numbers import Integral
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.diagnostic.utils.validation import _validate_observation_keys


def positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def signal_input_contract(
    *,
    feed_spec: FeedSpec | Mapping[str, Any] | None,
    timestamp_col: str | None,
    entity_col: str | None,
    date_col: str | None,
    asset_col: str | None,
    calendar: str | CalendarUnset | None,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve shared declarations and existing column aliases without inferring a grid."""
    feed = (
        FeedSpec.from_any(feed_spec)
        if feed_spec is not None
        else FeedSpec(timestamp_col="date", entity_col="asset", time_contract_version=2)
    )
    if timestamp_col is not None and date_col is not None and timestamp_col != date_col:
        raise ValueError("timestamp_col and date_col conflict")
    if entity_col is not None and asset_col is not None and entity_col != asset_col:
        raise ValueError("entity_col and asset_col conflict")
    timestamp = timestamp_col if timestamp_col is not None else date_col
    entity = entity_col if entity_col is not None else asset_col
    timestamp = feed.timestamp_col if timestamp is None else timestamp
    entity = feed.entity_col if entity is None else entity
    if entity is None:
        entity = "asset"
    elif not isinstance(entity, str) and isinstance(entity, Sequence) and len(entity) == 1:
        entity = entity[0]
    if (
        not isinstance(timestamp, str)
        or not timestamp.strip()
        or not isinstance(entity, str)
        or not entity.strip()
        or timestamp == entity
    ):
        raise ValueError(
            "signal inputs require distinct nonempty timestamp and single-entity names"
        )
    metadata = {
        "timestamp_col": timestamp,
        "entity_col": entity,
        "feed_spec": feed.to_dict(),
        "calendar": feed.resolve_calendar(calendar=calendar),
    }
    if calendar is not UNSET_CALENDAR:
        metadata["calendar_argument"] = calendar
    return timestamp, entity, metadata


def prepared_snapshot(
    data: pl.DataFrame | pd.DataFrame,
    periods: tuple[int, ...],
    quantiles: int | None,
    factor_col: str,
    date_col: str,
    asset_col: str,
    *,
    return_columns: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, bytes, dict]:
    """Validate labels and own the exact Arrow-compatible sample used for calculations."""
    if return_columns is None:
        if not periods or len(set(periods)) != len(periods):
            raise ValueError("periods must be nonempty and unique")
        for period in periods:
            positive_count(period, "periods")
        columns = [f"{p}D_fwd_return" for p in periods]
    else:
        if isinstance(return_columns, str):
            raise ValueError("return_columns must be a nonempty sequence of distinct names")
        columns = list(return_columns)
        if (
            not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("return_columns must be a nonempty sequence of distinct names")
        if set(columns).intersection(
            {date_col, asset_col, factor_col, "quantile", "factor_quantile"}
        ):
            raise ValueError("return_columns must be distinct from signal keys and values")
    if quantiles is not None:
        positive_count(quantiles, "quantiles")
    if isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            if date_col not in data.index.names or asset_col not in data.index.names:
                raise ValueError("prepared pandas index must name its date and asset keys")
            data = data.reset_index()
        frame = pl.from_pandas(data)
    elif isinstance(data, pl.DataFrame):
        frame = data.clone()
    else:
        raise TypeError("prepared data must be a pandas or Polars DataFrame")
    if frame.is_empty():
        raise ValueError("prepared data must contain observations (a nonempty sample)")
    aliases = {"factor_quantile": "quantile"}
    if return_columns is None:
        aliases.update({f"{p}D": f"{p}D_fwd_return" for p in periods})
    renamed = {}
    for alias, canonical in aliases.items():
        if alias in frame.columns:
            if canonical in frame.columns:
                raise ValueError(f"ambiguous prepared columns: {alias} and {canonical}")
            renamed[alias] = canonical
    frame = frame.rename(renamed)
    required = [date_col, asset_col, factor_col, "quantile", *columns]
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"prepared data is missing columns: {sorted(missing)}")
    if frame[date_col].dtype != pl.Date and not isinstance(frame[date_col].dtype, pl.Datetime):
        raise TypeError("prepared date column must contain Date or Datetime values")
    _validate_observation_keys(frame, [date_col, asset_col], "prepared")
    for column in (factor_col, "quantile", *columns):
        if not frame[column].dtype.is_numeric():
            raise TypeError(f"prepared {column} must be numeric")
        values = frame[column].cast(pl.Float64)
        if values.null_count() or not values.is_finite().all():
            raise ValueError(f"prepared {column} must be complete and finite")
    buckets = frame["quantile"].cast(pl.Float64)
    valid_buckets = (buckets == buckets.floor()) & (buckets >= 1)
    if quantiles is not None:
        valid_buckets &= buckets <= quantiles
    if not valid_buckets.all():
        if quantiles is None:
            raise ValueError("prepared quantile values must be positive integers")
        raise ValueError("prepared quantile values must be integers in 1..quantiles")
    frame = frame.sort(date_col, asset_col)
    try:
        payload = frame.write_ipc(None).getvalue()
    except pl.exceptions.PolarsError as exc:
        raise ValueError("prepared data columns must be Arrow-serializable") from exc
    owned = pl.read_ipc(BytesIO(payload))
    metadata = {
        "input_rows": frame.height,
        "retained_rows": frame.height,
        "excluded_rows": 0,
        "sample_policy": "strict already-prepared input; no cleaning or filling",
        "bucket_source": "supplied, not recomputed",
        "horizon_order": list(columns)
        if return_columns is not None
        else [f"{p}D" for p in periods],
        "return_columns": list(columns),
        "bucket_ids": sorted(int(bucket) for bucket in frame["quantile"].unique()),
        "date_dtype": str(frame[date_col].dtype),
        "frame_sha256": sha256(payload).hexdigest(),
        "calendar_validation": "not performed; precomputed labels do not establish a schedule",
    }
    return owned, payload, metadata
