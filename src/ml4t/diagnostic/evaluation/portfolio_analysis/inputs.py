"""Validate portfolio observations before labels or input ownership are lost."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def annual_risk_free(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("risk_free must be a finite annual real rate greater than -1")
    if not np.isfinite(value) or value <= -1:
        raise ValueError("risk_free must be a finite annual real rate greater than -1")
    return float(value)


def return_snapshot(data: Any, name: str) -> np.ndarray:
    values = np.asarray(data.to_numpy() if hasattr(data, "to_numpy") else data)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional return series")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric returns, not strings, booleans or nulls")
    values = np.array(values, dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite returns; missing values are not zero returns")
    if np.any(values < -1):
        raise ValueError(f"{name} contains unsupported simple returns below -1")
    return values


def labeled_dates(data: Any) -> pd.DatetimeIndex | None:
    if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        return data.index
    return None


def date_snapshot(data: Any, length: int, name: str, *, ordered: bool = True) -> pl.Series:
    try:
        if isinstance(data, pl.Series):
            dates = data.clone().rename("date")
        elif isinstance(data, (pd.DatetimeIndex, pd.Series)):
            dates = pl.from_pandas(data).rename("date")
        else:
            dates = pl.Series("date", data)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"{name} must contain valid Date or Datetime values") from exc
    if len(dates) != length:
        raise ValueError(f"{name} length must match its return series")
    if dates.dtype != pl.Date and not isinstance(dates.dtype, pl.Datetime):
        raise TypeError(f"{name} must contain Date or Datetime values")
    if dates.null_count() or dates.n_unique() != length:
        raise ValueError(f"{name} must be nonmissing and unique")
    if ordered and not dates.is_sorted():
        raise ValueError(f"{name} must be strictly increasing")
    return dates


def date_index(dates: pl.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates.to_pandas())


def normalize_observations(
    returns: Any, benchmark: Any, dates: Any
) -> tuple[np.ndarray, np.ndarray | None, pl.Series]:
    values = return_snapshot(returns, "returns")
    labels = labeled_dates(returns)
    if dates is None:
        dates = labels
    if dates is None:
        owned_dates = pl.Series("observation", np.arange(len(values)))
    else:
        owned_dates = date_snapshot(dates, len(values), "dates")
        if labels is not None:
            supplied = date_index(owned_dates)
            if supplied.tz != labels.tz or not (supplied == labels).all():
                raise ValueError("dates must agree with the labeled returns index")

    relative = None
    if benchmark is not None:
        relative = return_snapshot(benchmark, "benchmark")
        relative_labels = labeled_dates(benchmark)
        if relative_labels is not None:
            if owned_dates.dtype == pl.Int64:
                raise ValueError("benchmark is dated but strategy returns have no dates")
            relative_dates = date_snapshot(
                relative_labels, len(relative), "benchmark dates", ordered=False
            )
            strategy_index = date_index(owned_dates)
            relative_index = date_index(relative_dates)
            if strategy_index.tz != relative_index.tz:
                raise ValueError("benchmark dates and strategy dates must use the same timezone")
            locations = relative_index.get_indexer(strategy_index)
            if (locations < 0).any():
                raise ValueError("benchmark dates must cover every strategy observation")
            relative = relative[locations]
        elif len(relative) != len(values):
            raise ValueError("benchmark length must match returns for positional input")
    return values, relative, owned_dates
