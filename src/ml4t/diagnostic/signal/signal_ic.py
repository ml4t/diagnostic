"""Information Coefficient (IC) computation.

Simple, pure functions for IC analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from scipy.stats import t as t_dist


def compute_ic_series(
    data: pl.DataFrame,
    period: int,
    method: str = "spearman",
    factor_col: str = "factor",
    date_col: str = "date",
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
    min_obs : int, default 10
        Minimum observations per date.

    Returns
    -------
    tuple[list[Any], list[float]]
        (dates, ic_values) for dates with valid IC.
    """
    return_col = f"{period}D_fwd_return"

    valid_data = data.filter(pl.col(return_col).is_not_null())
    unique_dates = valid_data.select(date_col).unique().sort(date_col).to_series().to_list()

    dates: list[Any] = []
    ic_values: list[float] = []

    for date in unique_dates:
        date_data = valid_data.filter(pl.col(date_col) == date)
        if date_data.height < min_obs:
            continue

        factors = date_data[factor_col].to_numpy()
        returns = date_data[return_col].to_numpy()

        # Remove NaN pairs
        mask = ~(np.isnan(factors) | np.isnan(returns))
        if mask.sum() < min_obs:
            continue

        factors = factors[mask]
        returns = returns[mask]

        if method == "spearman":
            ic, _ = spearmanr(factors, returns)
        else:
            ic = float(np.corrcoef(factors, returns)[0, 1])

        if not np.isnan(ic):
            dates.append(date)
            ic_values.append(float(ic))

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
