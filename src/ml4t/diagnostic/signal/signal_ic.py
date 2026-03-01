"""Information Coefficient (IC) computation.

Simple, pure functions for IC analysis.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ml4t.diagnostic.evaluation.metrics.ic_statistics import compute_ic_summary_stats
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
    summary = compute_ic_summary_stats(ic_series)
    return {
        "mean": float(summary["mean_ic"]),
        "std": float(summary["std_ic"]),
        "t_stat": float(summary["t_stat"]),
        "p_value": float(summary["p_value"]),
        "pct_positive": float(summary["pct_positive"]),
    }


__all__ = ["compute_ic_series", "compute_ic_summary"]
