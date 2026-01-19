"""White's Reality Check for multiple strategy comparison.

This module implements White's Reality Check (2000), which tests whether
any strategy significantly outperforms a benchmark after adjusting for
multiple comparisons and data mining bias.

Reference:
White, H. (2000). "A Reality Check for Data Snooping."
Econometrica, 68(5), 1097-1126.
"""

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.backends.adapter import DataFrameAdapter

from .bootstrap import _stationary_bootstrap_indices

if TYPE_CHECKING:
    from numpy.typing import NDArray


def whites_reality_check(
    returns_benchmark: Union[pl.Series, pd.Series, "NDArray[Any]"],
    returns_strategies: Union[pd.DataFrame, pl.DataFrame, "NDArray[Any]"],
    bootstrap_samples: int = 1000,
    block_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Perform White's Reality Check for multiple strategy comparison.

    Tests whether any strategy significantly outperforms a benchmark after
    adjusting for multiple comparisons and data mining bias. Uses stationary
    bootstrap to preserve temporal dependencies.

    Parameters
    ----------
    returns_benchmark : Union[pl.Series, pd.Series, NDArray]
        Benchmark strategy returns
    returns_strategies : Union[pd.DataFrame, pl.DataFrame, NDArray]
        Returns for multiple strategies being tested
    bootstrap_samples : int, default 1000
        Number of bootstrap samples for null distribution
    block_size : Optional[int], default None
        Block size for stationary bootstrap. If None, uses optimal size
    random_state : Optional[int], default None
        Random seed for reproducible results

    Returns
    -------
    dict
        Dictionary with 'test_statistic', 'p_value', 'critical_values',
        'best_strategy_performance', 'null_distribution'

    Notes
    -----
    **Test Hypothesis**:
    - H0: No strategy beats the benchmark (max E[r_i - r_benchmark] <= 0)
    - H1: At least one strategy beats the benchmark

    **Interpretation**:
    - p_value < 0.05: Reject H0, at least one strategy beats benchmark
    - p_value >= 0.05: Cannot reject H0, no evidence of outperformance

    Examples
    --------
    >>> benchmark_returns = np.random.normal(0.001, 0.02, 252)
    >>> strategy_returns = np.random.normal(0.002, 0.02, (252, 10))
    >>> result = whites_reality_check(benchmark_returns, strategy_returns)
    >>> print(f"Reality Check p-value: {result['p_value']:.3f}")

    References
    ----------
    White, H. (2000). "A Reality Check for Data Snooping."
    Econometrica, 68(5), 1097-1126.
    """
    # Convert inputs
    benchmark = DataFrameAdapter.to_numpy(returns_benchmark).flatten()

    if isinstance(returns_strategies, pd.DataFrame | pl.DataFrame):
        strategies = DataFrameAdapter.to_numpy(returns_strategies)
        if strategies.ndim == 1:
            strategies = strategies.reshape(-1, 1)
    else:
        strategies = np.array(returns_strategies)
        if strategies.ndim == 1:
            strategies = strategies.reshape(-1, 1)

    n_periods, n_strategies = strategies.shape

    if len(benchmark) != n_periods:
        raise ValueError("Benchmark and strategies must have same number of periods")

    # Calculate relative performance (strategies vs benchmark)
    relative_returns = strategies - benchmark.reshape(-1, 1)

    # Test statistic: maximum mean relative performance
    mean_relative_returns = np.mean(relative_returns, axis=0)
    test_statistic = np.max(mean_relative_returns)
    best_strategy_idx = np.argmax(mean_relative_returns)

    # Bootstrap null distribution
    if random_state is not None:
        np.random.seed(random_state)

    # Optimal block size for stationary bootstrap (rule of thumb)
    if block_size is None:
        block_size = max(1, int(n_periods ** (1 / 3)))

    null_dist_list: list[float] = []

    for _ in range(bootstrap_samples):
        # Stationary bootstrap resampling
        bootstrap_indices = _stationary_bootstrap_indices(n_periods, float(block_size))

        # Resample relative returns
        bootstrap_relative = relative_returns[bootstrap_indices]

        # Center the bootstrap sample (impose null hypothesis)
        bootstrap_relative = bootstrap_relative - np.mean(bootstrap_relative, axis=0)

        # Calculate maximum mean for this bootstrap sample
        bootstrap_max = np.max(np.mean(bootstrap_relative, axis=0))
        null_dist_list.append(float(bootstrap_max))

    null_distribution = np.array(null_dist_list)

    # Calculate p-value
    p_value = np.mean(null_distribution >= test_statistic)

    # Calculate critical values
    critical_values = {
        "90%": np.percentile(null_distribution, 90),
        "95%": np.percentile(null_distribution, 95),
        "99%": np.percentile(null_distribution, 99),
    }

    return {
        "test_statistic": float(test_statistic),
        "p_value": float(p_value),
        "critical_values": critical_values,
        "best_strategy_idx": int(best_strategy_idx),
        "best_strategy_performance": float(mean_relative_returns[best_strategy_idx]),
        "null_distribution": null_distribution,
        "n_strategies": n_strategies,
        "n_periods": n_periods,
    }


__all__ = [
    "whites_reality_check",
]
