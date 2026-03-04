"""Factor timing analysis (Tier 2).

Tests whether the strategy successfully times factor exposures by
correlating rolling betas with next-period factor returns.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from .data import FactorData
from .results import FactorTimingResult
from .rolling_model import compute_rolling_exposures
from .static_model import _align_and_prepare


def compute_factor_timing(
    returns: np.ndarray | pl.Series,
    factor_data: FactorData,
    *,
    window: int = 63,
) -> FactorTimingResult:
    """Analyze factor timing: corr(beta_k[t], F_k[t+1]).

    Positive correlation means the strategy increases exposure to a
    factor before it performs well (successful timing).

    Parameters
    ----------
    returns : np.ndarray | pl.Series
        Portfolio returns (T,).
    factor_data : FactorData
        Factor return data.
    window : int
        Rolling window for beta estimation.

    Returns
    -------
    FactorTimingResult
        Spearman rank correlations and p-values per factor.
    """
    y, X, timestamps = _align_and_prepare(returns, factor_data)
    factor_names = factor_data.factor_names

    rolling = compute_rolling_exposures(returns, factor_data, window=window)

    correlations: dict[str, float] = {}
    p_values: dict[str, float] = {}

    n_rolling = len(rolling.timestamps)

    for k, f in enumerate(factor_names):
        betas = rolling.rolling_betas[f]
        # Align: beta[t] with factor_return[t+1]
        # rolling starts at index (window-1), so factor returns at (window)..
        # We need betas[:-1] vs factor_returns[1:]
        n_pairs = min(n_rolling - 1, X.shape[0] - window)
        if n_pairs < 10:
            correlations[f] = float("nan")
            p_values[f] = float("nan")
            continue

        beta_t = betas[:n_pairs]
        factor_t1 = X[window : window + n_pairs, k]

        # Remove NaN pairs
        mask = np.isfinite(beta_t) & np.isfinite(factor_t1)
        if np.sum(mask) < 10:
            correlations[f] = float("nan")
            p_values[f] = float("nan")
            continue

        corr, p = sp_stats.spearmanr(beta_t[mask], factor_t1[mask])
        correlations[f] = float(corr)
        p_values[f] = float(p)

    return FactorTimingResult(
        correlations=correlations,
        p_values=p_values,
        factor_names=factor_names,
        window=window,
    )
