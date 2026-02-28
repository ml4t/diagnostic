"""Information Coefficient (IC) computation.

Simple, pure functions for IC analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import t as t_dist

from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_series as compute_ic_series_core,
)


def compute_ic_series(
    data: pl.DataFrame,
    period: int,
    method: str = "spearman",
    factor_col: str = "factor",
    date_col: str = "date",
    asset_col: str = "asset",
    min_obs: int = 10,
) -> tuple[list[Any], list[float]]:
    """Compute IC time series for a single period.

    Parameters
    ----------
    data : pl.DataFrame
        Factor data with factor and forward return columns.
    period : int
        Forward return period in days.
    method : str, default "spearman"
        Correlation method ("spearman" or "pearson").
    factor_col : str, default "factor"
        Factor column name.
    date_col : str, default "date"
        Date column name.
    asset_col : str, default "asset"
        Asset/entity column used for panel joins.
    min_obs : int, default 10
        Minimum observations per date.

    Returns
    -------
    tuple[list[Any], list[float]]
        (dates, ic_values) for dates with valid IC.
    """
    return_col = f"{period}D_fwd_return"

    pred_df = data.select([date_col, asset_col, factor_col])
    ret_df = data.select([date_col, asset_col, return_col])

    ic_df = compute_ic_series_core(
        predictions=pred_df,
        returns=ret_df,
        pred_col=factor_col,
        ret_col=return_col,
        date_col=date_col,
        entity_col=asset_col,
        method=method,
        min_periods=min_obs,
    )

    if ic_df.height == 0:
        return [], []

    ic_clean = ic_df.filter(
        (pl.col("n_obs") >= min_obs) & pl.col("ic").cast(pl.Float64).is_finite()
    )
    dates = ic_clean[date_col].to_list()
    ic_values = ic_clean["ic"].cast(pl.Float64).to_list()
    return dates, ic_values


def compute_ic_summary(
    ic_series: list[float],
) -> dict[str, float]:
    """Compute summary statistics for an IC series.

    Parameters
    ----------
    ic_series : list[float]
        IC values over time.

    Returns
    -------
    dict[str, float]
        mean, std, t_stat, p_value, pct_positive
    """
    n = len(ic_series)
    if n < 2:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "pct_positive": float("nan"),
        }

    arr = np.array(ic_series)
    mean_ic = float(np.nanmean(arr))
    std_ic = float(np.nanstd(arr, ddof=1))

    if std_ic > 0:
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)))
    else:
        t_stat = float("nan")
        p_value = float("nan")

    pct_positive = float(np.mean(arr > 0))

    return {
        "mean": mean_ic,
        "std": std_ic,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "pct_positive": pct_positive,
    }


__all__ = ["compute_ic_series", "compute_ic_summary"]
