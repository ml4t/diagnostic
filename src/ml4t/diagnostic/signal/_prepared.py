"""Strict ingestion of already-complete signal panels, without return calculation."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from numbers import Integral

import pandas as pd
import polars as pl


def positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def prepared_snapshot(
    data: pl.DataFrame | pd.DataFrame,
    periods: tuple[int, ...],
    quantiles: int,
    factor_col: str,
    date_col: str,
    asset_col: str,
) -> tuple[pl.DataFrame, bytes, dict]:
    """Validate labels and own the exact Arrow-compatible sample used for calculations."""
    if not periods or len(set(periods)) != len(periods):
        raise ValueError("periods must be nonempty and unique")
    for period in periods:
        positive_count(period, "periods")
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
        raise ValueError("prepared data must contain observations")
    aliases = {"factor_quantile": "quantile", **{f"{p}D": f"{p}D_fwd_return" for p in periods}}
    renamed = {}
    for alias, canonical in aliases.items():
        if alias in frame.columns:
            if canonical in frame.columns:
                raise ValueError(f"ambiguous prepared columns: {alias} and {canonical}")
            renamed[alias] = canonical
    frame = frame.rename(renamed)
    return_columns = [f"{p}D_fwd_return" for p in periods]
    required = [date_col, asset_col, factor_col, "quantile", *return_columns]
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"prepared data is missing columns: {sorted(missing)}")
    if frame[date_col].dtype != pl.Date and not isinstance(frame[date_col].dtype, pl.Datetime):
        raise TypeError("prepared date column must contain Date or Datetime values")
    for key in (date_col, asset_col):
        if frame[key].null_count():
            raise ValueError(f"prepared {key} keys must be nonmissing")
    if frame[asset_col].dtype.is_numeric() and not frame[asset_col].is_finite().all():
        raise ValueError(f"prepared {asset_col} keys must be finite")
    if frame.select(date_col, asset_col).is_duplicated().any():
        raise ValueError("duplicate prepared date/asset keys are not allowed")
    for column in (factor_col, "quantile", *return_columns):
        if not frame[column].dtype.is_numeric():
            raise TypeError(f"prepared {column} must be numeric")
        values = frame[column].cast(pl.Float64)
        if values.null_count() or not values.is_finite().all():
            raise ValueError(f"prepared {column} must be complete and finite")
    buckets = frame["quantile"].cast(pl.Float64)
    if not ((buckets == buckets.floor()) & (buckets >= 1) & (buckets <= quantiles)).all():
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
        "horizon_order": [f"{p}D" for p in periods],
        "date_dtype": str(frame[date_col].dtype),
        "frame_sha256": sha256(payload).hexdigest(),
        "calendar_validation": "not performed; precomputed labels do not establish a schedule",
    }
    return owned, payload, metadata
