"""Normalize supplied long outcomes once, preserving their producing evidence."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

import polars as pl

from ml4t.diagnostic.utils.validation import _validate_observation_keys


def _nanosecond_ticks(name: str, dtype: pl.Datetime) -> pl.Expr:
    return (
        pl.col(name).cast(pl.Datetime("ns", dtype.time_zone)).cast(pl.Int64).cast(pl.Decimal(38, 0))
    )


def normalize_outcomes(
    data: pl.DataFrame,
    *,
    horizon_col: str,
    return_col: str,
    return_columns: Sequence[str] | None,
    date_col: str,
    asset_col: str,
) -> tuple[pl.DataFrame, bytes, dict[str, Any]]:
    """Pivot selected supplied outcomes, retaining exact source/end/availability keys."""
    keys = [date_col, asset_col]
    if len({*keys, horizon_col, return_col}) != 4:
        raise ValueError("outcome horizon/value names must be distinct from observation keys")
    _validate_observation_keys(data, [*keys, horizon_col], "long outcomes")
    if data.schema[horizon_col] not in (pl.String, pl.Categorical) and not isinstance(
        data.schema[horizon_col], pl.Enum
    ):
        raise TypeError("outcome horizon selectors must be strings")
    if return_col not in data.columns or not data.schema[return_col].is_numeric():
        raise TypeError("outcome values must contain numeric returns")
    if isinstance(return_columns, str):
        raise ValueError("return_columns must be a sequence, not a string")
    columns = (
        list(return_columns)
        if return_columns is not None
        else data[horizon_col].unique(maintain_order=True).to_list()
    )
    if (
        not columns
        or any(not isinstance(name, str) or not name for name in columns)
        or len(set(columns)) != len(columns)
        or set(columns).intersection(keys)
    ):
        raise ValueError("return_columns must be distinct nonempty horizon names, not keys")
    if absent := set(columns).difference(data[horizon_col].unique()):
        raise ValueError(f"requested outcome horizons are missing: {sorted(absent)}")
    selected = data.filter(pl.col(horizon_col).is_in(columns))
    if "return_method" in selected.columns and selected["return_method"].unique().to_list() != [
        "returns"
    ]:
        raise ValueError("signal outcome ingestion requires arithmetic returns, not log returns")
    metadata: dict[str, Any] = {
        "format": "long supplied outcomes",
        "horizon_col": horizon_col,
        "return_col": return_col,
        "return_columns": columns,
        "input_rows": data.height,
        "selected_rows": selected.height,
        "input_sha256": sha256(data.write_ipc(None).getvalue()).hexdigest(),
        "source_columns": selected.columns,
        "duration_unit": "ns for timezone-aware instants; unavailable for session labels/naive keys",
        "duration_encoding": "exact Decimal(38,0) integer nanoseconds, avoiding signed subtraction overflow",
    }
    if {"calendar", "calendar_declared"}.issubset(selected.columns):
        if selected.schema["calendar_declared"] != pl.Boolean:
            raise TypeError("outcome calendar_declared must be boolean")
        declarations = (
            selected.filter(pl.col("calendar_declared").fill_null(False))["calendar"]
            .unique()
            .to_list()
        )
        if len(declarations) > 1:
            raise ValueError("selected outcomes declare conflicting calendars")
        if declarations:
            metadata["calendar"] = declarations[0]
    for endpoint, duration in ((date_col, "duration_ns"), ("reference_time", "return_duration_ns")):
        left, right = selected.schema.get(endpoint), selected.schema.get("exit_time")
        if (
            isinstance(left, pl.Datetime)
            and left.time_zone is not None
            and isinstance(right, pl.Datetime)
            and right.time_zone is not None
        ):
            for name, dtype in ((endpoint, left), ("exit_time", right)):
                if selected.select(
                    pl.col(name)
                    .cast(pl.Datetime("ns", dtype.time_zone))
                    .cast(dtype)
                    .ne_missing(pl.col(name))
                    .any()
                ).item():
                    raise ValueError("outcome instants must fit the nanosecond timestamp range")
            derived = _nanosecond_ticks("exit_time", right) - _nanosecond_ticks(endpoint, left)
        else:
            derived = pl.lit(None, dtype=pl.Int64)
        if duration in selected.columns:
            dtype = selected.schema[duration]
            if not (
                dtype.is_integer()
                or dtype == pl.Null
                or isinstance(dtype, pl.Decimal)
                and dtype.scale == 0
            ):
                raise TypeError(f"supplied {duration} must use exact integer nanoseconds")
            if selected.select(pl.col(duration).ne_missing(derived).any()).item():
                raise ValueError(f"supplied {duration} disagrees with outcome endpoints")
        selected = selected.with_columns(derived.alias(duration))
    wide = selected.pivot(
        on=horizon_col, index=keys, values=return_col, maintain_order=True
    ).select(*keys, *columns)
    return wide, selected.write_ipc(None).getvalue(), metadata
