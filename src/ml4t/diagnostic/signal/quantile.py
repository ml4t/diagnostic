"""Quantile analysis functions.

Simple, pure functions for analyzing returns by quantile.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr, ttest_ind


def _compute_quantile_return_statistics(
    data: pl.DataFrame,
    period: int,
    n_quantiles: int,
    quantile_col: str = "quantile",
) -> tuple[dict[int, float], dict[int, float]]:
    return_col = f"{period}D_fwd_return"

    if return_col not in data.columns:
        missing = dict.fromkeys(range(1, n_quantiles + 1), float("nan"))
        return missing, missing.copy()

    means: dict[int, float] = {}
    standard_deviations: dict[int, float] = {}
    valid_data = data.filter(pl.col(return_col).is_not_null())

    # Polars parallel group reductions may add floats in a different order on
    # repeated calls. Reduce rows within each quantile in input order instead.
    for partition in valid_data.partition_by(quantile_col, maintain_order=True):
        quantile = partition.item(0, quantile_col)
        returns = partition.get_column(return_col).to_numpy()
        quantile_number = int(quantile)
        means[quantile_number] = float(np.mean(returns))
        standard_deviations[quantile_number] = (
            float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan")
        )

    # Fill missing quantiles
    for q in range(1, n_quantiles + 1):
        means.setdefault(q, float("nan"))
        standard_deviations.setdefault(q, float("nan"))

    return dict(sorted(means.items())), dict(sorted(standard_deviations.items()))


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
    means, _ = _compute_quantile_return_statistics(data, period, n_quantiles, quantile_col)

    return means


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


def monotonicity_score(
    quantile_returns: dict[int, float],
) -> float:
    """Compute monotonicity score from pre-computed quantile returns.

    A simple Spearman correlation between quantile ranks and their mean returns.
    For the full-featured version that accepts raw DataFrames, use
    ``ml4t.diagnostic.metrics.compute_monotonicity``.

    Parameters
    ----------
    quantile_returns : dict[int, float]
        Mean return by quantile (from ``compute_quantile_returns``).

    Returns
    -------
    float
        Spearman rho (-1 to 1). 1.0 = perfect monotonic increase.
    """
    # Sort by quantile
    sorted_items = sorted(quantile_returns.items())
    quantiles = [q for q, r in sorted_items if not np.isnan(r)]
    returns = [r for q, r in sorted_items if not np.isnan(r)]

    if len(quantiles) < 3:
        return float("nan")

    rho, _ = spearmanr(quantiles, returns)
    return float(rho) if not np.isnan(rho) else float("nan")


__all__ = [
    "compute_quantile_returns",
    "compute_spread",
    "monotonicity_score",
]
