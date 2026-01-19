"""Probability of Backtest Overfitting (PBO).

PBO measures the probability that a strategy selected as best in-sample
performs below median out-of-sample. A high PBO indicates overfitting.

This module is intentionally separate from DSR/Sharpe inference because
PBO is a model selection diagnostic, not a statistical inference tool.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Probability of Backtest
Overfitting." Journal of Computational Finance, 20(4), 39-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    """Result of Probability of Backtest Overfitting calculation.

    Attributes
    ----------
    pbo : float
        Probability of Backtest Overfitting (0 to 1).
    pbo_pct : float
        PBO as percentage (0 to 100).
    n_combinations : int
        Number of IS/OOS combinations evaluated.
    n_strategies : int
        Number of strategies compared.
    is_best_rank_oos_median : float
        Median OOS rank of IS-best strategy.
    is_best_rank_oos_mean : float
        Mean OOS rank of IS-best strategy.
    degradation_mean : float
        Average OOS performance degradation vs IS.
    degradation_std : float
        Std of degradation.
    """

    pbo: float
    pbo_pct: float
    n_combinations: int
    n_strategies: int
    is_best_rank_oos_median: float
    is_best_rank_oos_mean: float
    degradation_mean: float
    degradation_std: float

    def interpret(self) -> str:
        """Generate human-readable interpretation."""
        if self.pbo < 0.10:
            risk_level = "LOW"
            assessment = "Strategy selection appears robust"
        elif self.pbo < 0.30:
            risk_level = "MODERATE"
            assessment = "Some overfitting risk - consider out-of-sample validation"
        elif self.pbo < 0.50:
            risk_level = "HIGH"
            assessment = "Significant overfitting risk - results may not generalize"
        else:
            risk_level = "SEVERE"
            assessment = "IS selection is counterproductive - consider alternative methods"

        return (
            f"Probability of Backtest Overfitting (PBO)\n"
            f"  PBO: {self.pbo_pct:.1f}%\n"
            f"  Risk Level: {risk_level}\n"
            f"  Assessment: {assessment}\n"
            f"\n"
            f"  Combinations: {self.n_combinations}\n"
            f"  Strategies: {self.n_strategies}\n"
            f"  IS-Best OOS Rank: {self.is_best_rank_oos_median:.1f} (median), "
            f"{self.is_best_rank_oos_mean:.1f} (mean)\n"
            f"  Performance Degradation: {self.degradation_mean:.4f} +/- {self.degradation_std:.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "pbo": self.pbo,
            "pbo_pct": self.pbo_pct,
            "n_combinations": self.n_combinations,
            "n_strategies": self.n_strategies,
            "is_best_rank_oos_median": self.is_best_rank_oos_median,
            "is_best_rank_oos_mean": self.is_best_rank_oos_mean,
            "degradation_mean": self.degradation_mean,
            "degradation_std": self.degradation_std,
        }


def compute_pbo(
    is_performance: np.ndarray[Any, np.dtype[Any]],
    oos_performance: np.ndarray[Any, np.dtype[Any]],
) -> PBOResult:
    """Compute Probability of Backtest Overfitting (PBO).

    PBO measures the probability that a strategy selected as best in-sample
    performs below median out-of-sample. A high PBO indicates overfitting.

    Definition
    ----------
    From Bailey & López de Prado (2014):

    .. math::

        PBO = P(rank_{OOS}(\\arg\\max_{IS}) > N/2)

    In plain English: what's the probability that the best in-sample strategy
    ranks in the bottom half out-of-sample?

    Interpretation
    --------------
    - PBO = 0%: No overfitting (best IS is also best OOS)
    - PBO = 50%: Random selection (IS performance uncorrelated with OOS)
    - PBO > 50%: Severe overfitting (IS selection is counterproductive)

    Parameters
    ----------
    is_performance : np.ndarray, shape (n_folds, n_strategies) or (n_combinations,)
        In-sample performance metrics (Sharpe, IC, returns) for each strategy.
    oos_performance : np.ndarray, shape (n_folds, n_strategies) or (n_combinations,)
        Out-of-sample performance metrics (same structure as is_performance).

    Returns
    -------
    PBOResult
        Result object with PBO and diagnostic metrics.
        Call .interpret() for human-readable assessment.

    Raises
    ------
    ValueError
        If arrays have different shapes or fewer than 2 strategies.

    Examples
    --------
    >>> import numpy as np
    >>> # 10 CV folds, 5 strategies
    >>> is_perf = np.random.randn(10, 5)
    >>> oos_perf = np.random.randn(10, 5)
    >>> result = compute_pbo(is_perf, oos_perf)
    >>> print(result.interpret())

    References
    ----------
    Bailey, D. H., & López de Prado, M. (2014). "The Probability of Backtest
    Overfitting." Journal of Computational Finance, 20(4), 39-69.
    """
    is_performance = np.asarray(is_performance)
    oos_performance = np.asarray(oos_performance)

    if is_performance.shape != oos_performance.shape:
        raise ValueError(
            f"is_performance and oos_performance must have same shape. "
            f"Got {is_performance.shape} vs {oos_performance.shape}"
        )

    # Handle 1D input (single combination with multiple strategies)
    if is_performance.ndim == 1:
        is_performance = is_performance.reshape(1, -1)
        oos_performance = oos_performance.reshape(1, -1)

    n_combinations, n_strategies = is_performance.shape

    if n_strategies < 2:
        raise ValueError(f"Need at least 2 strategies, got {n_strategies}")

    # For each combination, find the IS-best strategy and its OOS rank
    is_best_oos_ranks = []
    degradations = []

    for i in range(n_combinations):
        is_row = is_performance[i, :]
        oos_row = oos_performance[i, :]

        # Find strategy with best IS performance
        is_best_idx = np.argmax(is_row)
        is_best_is_perf = is_row[is_best_idx]
        is_best_oos_perf = oos_row[is_best_idx]

        # Compute OOS rank of IS-best strategy (1 = best, N = worst)
        oos_ranks = n_strategies - np.argsort(np.argsort(oos_row))
        is_best_oos_rank = oos_ranks[is_best_idx]
        is_best_oos_ranks.append(is_best_oos_rank)

        # Compute degradation (IS - OOS performance)
        degradations.append(is_best_is_perf - is_best_oos_perf)

    ranks_arr = np.array(is_best_oos_ranks)
    degrad_arr = np.array(degradations)

    # PBO = P(IS-best ranks in bottom half OOS)
    median_rank = (n_strategies + 1) / 2
    n_below_median = np.sum(ranks_arr > median_rank)
    pbo = n_below_median / n_combinations

    return PBOResult(
        pbo=float(pbo),
        pbo_pct=float(pbo * 100),
        n_combinations=int(n_combinations),
        n_strategies=int(n_strategies),
        is_best_rank_oos_median=float(np.median(ranks_arr)),
        is_best_rank_oos_mean=float(np.mean(ranks_arr)),
        degradation_mean=float(np.mean(degrad_arr)),
        degradation_std=float(np.std(degrad_arr)),
    )


__all__ = [
    "PBOResult",
    "compute_pbo",
]
