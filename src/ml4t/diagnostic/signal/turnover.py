"""Turnover and autocorrelation analysis.

Simple, pure functions for factor persistence analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def compute_turnover(
    data: pl.DataFrame,
    n_quantiles: int,
    date_col: str = "date",
    asset_col: str = "asset",
    quantile_col: str = "quantile",
) -> float:
    """Compute mean turnover rate across quantiles.

    Turnover = fraction of assets that change quantile each period.

    Parameters
    ----------
    data : pl.DataFrame
        Data with date, asset, and quantile columns.
    n_quantiles : int
        Number of quantiles.
    date_col, asset_col, quantile_col : str
        Column names.

    Returns
    -------
    float
        Mean turnover rate (0-1).
    """
    unique_dates = data.select(date_col).unique().sort(date_col).to_series().to_list()

    if len(unique_dates) < 2:
        return float("nan")

    # Pre-compute asset sets per (date, quantile) using dict comprehension
    asset_lists = (
        data.group_by([date_col, quantile_col])
        .agg(pl.col(asset_col).alias("assets"))
        .sort([date_col, quantile_col])
    )
    # Use rows() for faster iteration (returns tuples)
    asset_sets: dict[tuple[Any, int], set[Any]] = {
        (row[0], row[1]): set(row[2]) for row in asset_lists.rows()
    }

    # Compute turnover for each quantile
    all_turnovers: list[float] = []

    for q in range(1, n_quantiles + 1):
        q_turnovers: list[float] = []

        for i in range(len(unique_dates) - 1):
            date_t = unique_dates[i]
            date_t1 = unique_dates[i + 1]

            assets_t = asset_sets.get((date_t, q), set())
            assets_t1 = asset_sets.get((date_t1, q), set())

            if assets_t and assets_t1:
                overlap = len(assets_t & assets_t1)
                turnover = 1 - overlap / max(len(assets_t), len(assets_t1))
                q_turnovers.append(turnover)

        if q_turnovers:
            all_turnovers.append(float(np.mean(q_turnovers)))

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


__all__ = ["compute_turnover", "compute_autocorrelation", "estimate_half_life"]
