"""Turnover and autocorrelation analysis.

Simple, pure functions for factor persistence analysis.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from ml4t.diagnostic.signal._prepared import positive_count
from ml4t.diagnostic.signal._utils import _signal_frame
from ml4t.diagnostic.utils.validation import (
    _align_datetime_key_resolution,
    _validate_observation_keys,
)


def _persistence_time_keys(data: pl.DataFrame, date_col: str) -> pl.DataFrame:
    if isinstance(data.schema[date_col], pl.Datetime):
        return data.with_columns(pl.col(date_col).cast(pl.Int64))
    return data


def _membership_cache(
    data: pl.DataFrame, date_col: str, asset_col: str, quantile_col: str
) -> dict[tuple[Any, int], set[Any]]:
    values = data.select(
        pl.col(date_col).alias("__date"),
        pl.col(quantile_col).alias("__bucket"),
        pl.col(asset_col).alias("__asset"),
    )
    memberships = values.group_by(["__date", "__bucket"]).agg(pl.col("__asset").alias("__members"))
    return {(row[0], row[1]): set(row[2]) for row in memberships.rows()}


def _turnover_entries(
    memberships: dict[tuple[Any, int], set[Any]],
    dates: list[Any],
    quantiles: Sequence[int],
    lags: Sequence[int],
    *,
    current_denominator: bool,
) -> Iterator[dict[str, Any]]:
    for lag in lags:
        for bucket in quantiles:
            for index, day in enumerate(dates):
                previous_day = dates[index - lag] if index >= lag else None
                current = memberships.get((day, bucket), set())
                previous = memberships.get((previous_day, bucket), set()) if index >= lag else None
                overlap = entered = rate = None
                if current and previous:
                    overlap = len(current & previous)
                    entered = len(current) - overlap
                    rate = (
                        entered / len(current)
                        if current_denominator
                        else 1 - overlap / max(len(current), len(previous))
                    )
                yield {
                    "__clock_key": day,
                    "__previous_key": previous_day,
                    "quantile": bucket,
                    "lag": lag,
                    "turnover": rate,
                    "n_current": len(current),
                    "n_previous": len(previous) if previous is not None else None,
                    "n_overlap": overlap,
                    "n_entered": entered,
                    "status": "insufficient_history"
                    if previous is None
                    else "empty_current"
                    if not current
                    else "empty_previous"
                    if not previous
                    else "defined",
                }


def _positive_selection(values: Sequence[int], name: str) -> list[int]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a nonempty sequence of distinct positive integers")
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a nonempty sequence of distinct positive integers"
        ) from exc
    if not items:
        raise ValueError(f"{name} must be a nonempty sequence of distinct positive integers")
    selected = [positive_count(value, name) for value in items]
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} must contain distinct positive integers")
    if any(value > np.iinfo(np.int64).max for value in selected):
        raise ValueError(f"{name} must fit Int64")
    return selected


def compute_quantile_turnover(
    data: pl.DataFrame | pd.DataFrame,
    *,
    lags: Sequence[int] = (1,),
    quantiles: Sequence[int] | None = None,
    date_grid: pl.DataFrame | pd.DataFrame | None = None,
    date_col: str = "date",
    asset_col: str = "asset",
    quantile_col: str = "quantile",
) -> pl.DataFrame:
    """Inspect dated entrants/current-membership turnover for explicit row lags.

    Reuse the native scalar helper's membership calculation, with Alphalens's
    entrants/current-membership denominator. The scalar compute_turnover still
    uses the larger membership and means over defined pairs within each bucket.

    Default clocks are sorted observed dates. A supplied unique date grid must
    cover every observed date and defines lag positions, including missing
    observations. No holiday, calendar, duration or replacement date is inferred.
    Requested lag/bucket order is preserved. Missing bucket labels are excluded
    from membership but retain their observed dates/universe sizes. Empty buckets
    and insufficient history have null rates/overlap/entrants, not zero turnover.

    Inputs require unique nonmissing date/asset keys and numeric positive integer
    bucket labels. Pandas MultiIndex tables and physical column selectors are
    accepted. Labels need not be contiguous; explicitly requested absent buckets
    remain unavailable. An owned cleaned-empty input retains a typed empty table,
    or, with explicit bucket selection, unavailable observations on the supplied
    grid. Caller inputs stay unchanged.

    Returns canonical date, quantile, lag, turnover, per-leg membership/universe
    counts, overlap/entrants, status, method and explicit clock/lag-unit metadata.
    """
    selected_lags = _positive_selection(lags, "lags")
    values = _signal_frame(data)
    _validate_observation_keys(values, [date_col, asset_col], "turnover observations")
    if quantile_col not in values.columns:
        raise ValueError(f"turnover observations is missing {quantile_col}")
    if not (values[quantile_col].dtype.is_numeric() or values[quantile_col].dtype == pl.Null):
        raise TypeError("turnover bucket labels must be numeric")
    values = values.select(
        pl.col(date_col).alias("date"),
        pl.col(asset_col).alias("asset"),
        pl.col(quantile_col).alias("quantile"),
    )
    labels = values["quantile"].cast(pl.Float64)
    known = values.filter((labels.is_not_null() & ~labels.is_nan()).fill_null(False))
    observed = known["quantile"].cast(pl.Float64)
    if not ((observed >= 1) & observed.is_finite() & (observed == observed.floor())).all():
        raise ValueError("turnover bucket labels must be positive integer labels")
    buckets = [int(value) for value in known["quantile"].unique().sort().to_list()]
    selected = (
        _positive_selection(quantiles, "quantiles")
        if quantiles is not None
        else _positive_selection(buckets, "quantiles")
        if buckets
        else []
    )
    clock = values.select("date").unique().sort("date")
    if date_grid is not None:
        clock = _signal_frame(date_grid)
        _validate_observation_keys(clock, [date_col], "turnover date grid")
        clock = clock.select(pl.col(date_col).alias("date")).sort("date")
        values, clock = _align_datetime_key_resolution(values, clock, "date")
        known, clock = _align_datetime_key_resolution(known, clock, "date")
        if values.select("date").unique().join(clock, on="date", how="anti").height:
            raise ValueError("turnover date_grid must cover every observed date")
    clock = clock.with_columns(_persistence_time_keys(clock, "date")["date"].alias("__clock_key"))
    normalized = _persistence_time_keys(values, "date")
    universe = dict(normalized.group_by("date").len().rows())
    memberships = _membership_cache(
        _persistence_time_keys(known, "date"), "date", "asset", "quantile"
    )
    entries = []
    for entry in _turnover_entries(
        memberships,
        clock["__clock_key"].to_list(),
        selected,
        selected_lags,
        current_denominator=True,
    ):
        previous = entry.pop("__previous_key")
        entry["current_universe_size"] = universe.get(entry["__clock_key"], 0)
        entry["previous_universe_size"] = (
            universe.get(previous, 0) if previous is not None else None
        )
        entries.append(entry)
    schema = {
        "__clock_key": clock.schema["__clock_key"],
        "quantile": pl.Int64,
        "lag": pl.Int64,
        "turnover": pl.Float64,
        **dict.fromkeys(
            [
                "n_current",
                "n_previous",
                "n_overlap",
                "n_entered",
                "current_universe_size",
                "previous_universe_size",
            ],
            pl.UInt32,
        ),
        "status": pl.String,
    }
    return (
        pl.DataFrame(entries, schema=schema)
        .with_row_index("__row_order")
        .join(clock, on="__clock_key", how="left")
        .sort("__row_order")
        .drop("__row_order", "__clock_key")
        .with_columns(
            pl.lit("entrants_over_current").alias("turnover_method"),
            pl.lit("observations").alias("lag_unit"),
            pl.lit("supplied" if date_grid is not None else "observed").alias("clock_source"),
        )
    )


def compute_turnover(
    data: pl.DataFrame,
    n_quantiles: int,
    date_col: str = "date",
    asset_col: str = "asset",
    quantile_col: str = "quantile",
    *,
    lag: int = 1,
) -> float:
    """Compute mean turnover rate across quantiles.

    Compare quantile membership at dates separated by ``lag`` observations.
    Turnover is one minus the overlap divided by the larger membership count.

    Parameters
    ----------
    data : pl.DataFrame
        Data with date, asset, and quantile columns.
    n_quantiles : int
        Number of quantiles.
    date_col, asset_col, quantile_col : str
        Column names.
    lag : int, default 1
        Positive separation in sorted observed dates, not calendar days.

    Returns
    -------
    float
        Mean turnover rate (0-1), or NaN when no comparable date pairs exist.
    """
    if lag < 1:
        raise ValueError("lag must be positive")

    data = _persistence_time_keys(data, date_col)
    unique_dates = data.select(date_col).unique().sort(date_col).to_series().to_list()

    if len(unique_dates) <= lag:
        return float("nan")

    asset_sets = _membership_cache(data, date_col, asset_col, quantile_col)
    buckets = list(range(1, n_quantiles + 1))
    by_bucket: dict[int, list[float]] = {bucket: [] for bucket in buckets}
    for entry in _turnover_entries(
        asset_sets, unique_dates, buckets, [lag], current_denominator=False
    ):
        if entry["status"] == "defined":
            by_bucket[entry["quantile"]].append(entry["turnover"])
    all_turnovers = [float(np.mean(values)) for values in by_bucket.values() if values]

    return float(np.nanmean(all_turnovers)) if all_turnovers else float("nan")


def compute_autocorrelation(
    data: pl.DataFrame,
    lags: list[int],
    date_col: str = "date",
    asset_col: str = "asset",
    factor_col: str = "factor",
    min_obs: int = 10,
) -> list[float]:
    """Compute factor rank autocorrelation at different lags.

    Parameters
    ----------
    data : pl.DataFrame
        Data with date, asset, and factor columns.
    lags : list[int]
        Lag values (e.g., [1, 2, 3, 4, 5]).
    date_col, asset_col, factor_col : str
        Column names.
    min_obs : int, default 10
        Minimum observations per date pair.

    Returns
    -------
    list[float]
        Autocorrelation at each lag.
    """
    data = _persistence_time_keys(data, date_col)
    unique_dates = data.select(date_col).unique().sort(date_col).to_series().to_list()

    if len(unique_dates) < max(lags) + 1:
        return [float("nan")] * len(lags)

    # Cache data by date using partition_by (single pass, O(n))
    date_cache: dict[Any, pl.DataFrame] = {}
    partitions = data.select([date_col, asset_col, factor_col]).partition_by(
        date_col, as_dict=True, include_key=False
    )
    for date_key, df in partitions.items():
        # partition_by returns tuple keys when grouping by single column
        date = date_key[0] if isinstance(date_key, tuple) else date_key
        date_cache[date] = df

    autocorrelations: list[float] = []

    for lag in lags:
        correlations: list[float] = []

        for i in range(len(unique_dates) - lag):
            date_t = unique_dates[i]
            date_t_lag = unique_dates[i + lag]

            data_t = date_cache[date_t]
            data_t_lag = date_cache[date_t_lag]

            merged = data_t.join(data_t_lag, on=asset_col, how="inner", suffix="_lag")

            if merged.height < min_obs:
                continue

            rho, _ = spearmanr(
                merged[factor_col].to_numpy(), merged[f"{factor_col}_lag"].to_numpy()
            )
            if not np.isnan(rho):
                correlations.append(float(rho))

        lag_ac = float(np.mean(correlations)) if correlations else float("nan")
        autocorrelations.append(lag_ac)

    return autocorrelations


def estimate_half_life(autocorrelations: list[float]) -> float | None:
    """Estimate half-life from autocorrelation decay.

    Half-life is the lag where autocorrelation drops to 50% of lag-1 value.

    Parameters
    ----------
    autocorrelations : list[float]
        Autocorrelation at lags 1, 2, 3, ...

    Returns
    -------
    float | None
        Half-life in periods, or None if undefined.
    """
    valid_ac = [ac for ac in autocorrelations if not np.isnan(ac)]

    if len(valid_ac) < 2 or valid_ac[0] <= 0:
        return None

    threshold = 0.5 * valid_ac[0]

    for i, ac in enumerate(valid_ac):
        if ac < threshold:
            if i > 0:
                # Linear interpolation
                return i + (valid_ac[i - 1] - threshold) / (valid_ac[i - 1] - ac)
            return float(i + 1)

    return None  # Never decayed below threshold


__all__ = [
    "compute_quantile_turnover",
    "compute_turnover",
    "compute_autocorrelation",
    "estimate_half_life",
]
