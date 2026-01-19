"""Quantile analysis functions.

Simple, pure functions for analyzing returns by quantile.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr, ttest_ind


def compute_quantile_returns(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> dict[int, float]:
    """Compute mean forward returns by quantile.

    Parameters
    ----------
    data : pl.DataFrame
        Data with quantile and forward return columns.
    period : int
        Forward return period in days.
    n_quantiles : int
        Number of quantiles.
    quantile_col : str, default "quantile"
        Quantile column name.

    Returns
    -------
    dict[int, float]
        Mean return by quantile (1 = lowest factor).
    """
    return_col = f"{period}D_fwd_return"

    if return_col not in data.columns:
        return dict.fromkeys(range(1, n_quantiles + 1), float("nan"))

    result: dict[int, float] = {}

    quantile_means = (
        data.filter(pl.col(return_col).is_not_null())
        .group_by(quantile_col)
        .agg(pl.col(return_col).mean().alias("mean_return"))
        .sort(quantile_col)
    )

    for row in quantile_means.iter_rows(named=True):
        result[int(row[quantile_col])] = float(row["mean_return"])

    # Fill missing quantiles
    for q in range(1, n_quantiles + 1):
        if q not in result:
            result[q] = float("nan")

    return result


def compute_spread(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> dict[str, float]:
    """Compute long-short spread and statistics.

    Parameters
    ----------
    data : pl.DataFrame
        Data with quantile and forward return columns.
    period : int
        Forward return period in days.
    n_quantiles : int
        Number of quantiles.
    quantile_col : str, default "quantile"
        Quantile column name.

    Returns
    -------
    dict[str, float]
        spread, t_stat, p_value
    """
    return_col = f"{period}D_fwd_return"

    if return_col not in data.columns:
        return {
            "spread": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    top_returns = data.filter(pl.col(quantile_col) == n_quantiles)[return_col].to_numpy()
    bottom_returns = data.filter(pl.col(quantile_col) == 1)[return_col].to_numpy()

    top_returns = top_returns[~np.isnan(top_returns)]
    bottom_returns = bottom_returns[~np.isnan(bottom_returns)]

    if len(top_returns) < 2 or len(bottom_returns) < 2:
        return {
            "spread": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    spread = float(np.mean(top_returns) - np.mean(bottom_returns))
    t_stat, p_value = ttest_ind(top_returns, bottom_returns)

    return {
        "spread": spread,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def compute_monotonicity(
    quantile_returns: dict[int, float],
) -> float:
    """Compute monotonicity of quantile returns.

    Measures how well returns increase monotonically across quantiles.
    Uses Spearman correlation: 1.0 = perfect increase, -1.0 = perfect decrease.

    Parameters
    ----------
    quantile_returns : dict[int, float]
        Mean return by quantile.

    Returns
    -------
    float
        Monotonicity score (-1 to 1).
    """
    # Sort by quantile
    sorted_items = sorted(quantile_returns.items())
    quantiles = [q for q, r in sorted_items if not np.isnan(r)]
    returns = [r for q, r in sorted_items if not np.isnan(r)]

    if len(quantiles) < 3:
        return float("nan")

    rho, _ = spearmanr(quantiles, returns)
    return float(rho) if not np.isnan(rho) else float("nan")


__all__ = ["compute_quantile_returns", "compute_spread", "compute_monotonicity"]
