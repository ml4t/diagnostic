"""Numba-optimized utility functions for ML4T Diagnostic.

This module contains JIT-compiled functions for performance-critical operations.
Numba is used to optimize computationally intensive loops and array operations.

Note: Numba functions work best with NumPy arrays and simple Python types.
They cannot handle Pandas objects directly.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_drawdown_numba(
    cum_returns: np.ndarray,
) -> tuple[float, int, int, int]:
    """Numba-optimized maximum drawdown calculation.

    Parameters
    ----------
    cum_returns : np.ndarray
        Array of cumulative returns

    Returns
    -------
    Tuple[float, int, int, int]
        (max_drawdown, duration, peak_idx, trough_idx)
    """
    n = len(cum_returns)
    if n == 0:
        return np.nan, -1, -1, -1

    max_drawdown = 0.0
    max_duration = 0
    peak_idx = 0
    trough_idx = 0
    current_peak = cum_returns[0]
    current_peak_idx = 0

    for i in range(1, n):
        # Update peak if necessary
        if cum_returns[i] > current_peak:
            current_peak = cum_returns[i]
            current_peak_idx = i

        # Calculate current drawdown
        drawdown = cum_returns[i] - current_peak

        # Update max drawdown if necessary
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            peak_idx = current_peak_idx
            trough_idx = i
            max_duration = i - current_peak_idx

    return max_drawdown, max_duration, peak_idx, trough_idx


@jit(nopython=True, cache=True)
def purge_indices_numba(
    test_start: int,
    _test_end: int,
    label_horizon: int,
    n_samples: int,
) -> np.ndarray:
    """Numba-optimized calculation of purge indices.

    Parameters
    ----------
    test_start : int
        Start index of test period
    test_end : int
        End index of test period
    label_horizon : int
        Forward-looking period of labels
    n_samples : int
        Total number of samples

    Returns
    -------
    np.ndarray
        Array of indices to purge
    """
    purge_start = max(0, test_start - label_horizon)
    purge_end = min(test_start, n_samples)

    if purge_start >= purge_end:
        return np.empty(0, dtype=np.int64)

    return np.arange(purge_start, purge_end, dtype=np.int64)


@jit(nopython=True, cache=True)
def embargo_indices_numba(
    test_end: int,
    embargo_size: int,
    n_samples: int,
) -> np.ndarray:
    """Numba-optimized calculation of embargo indices.

    Parameters
    ----------
    test_end : int
        End index of test period
    embargo_size : int
        Number of samples to embargo after test set
    n_samples : int
        Total number of samples

    Returns
    -------
    np.ndarray
        Array of indices to embargo
    """
    embargo_start = test_end
    embargo_end = min(test_end + embargo_size, n_samples)

    if embargo_start >= embargo_end:
        return np.empty(0, dtype=np.int64)

    return np.arange(embargo_start, embargo_end, dtype=np.int64)


@jit(nopython=True, cache=True, parallel=True)
def block_bootstrap_numba(
    indices: np.ndarray,
    n_samples: int,
    sample_length: int,
    seed: int,
) -> np.ndarray:
    """Numba-optimized block bootstrap sampling.

    Parameters
    ----------
    indices : np.ndarray
        Array of indices to sample from
    n_samples : int
        Number of bootstrap samples to generate
    sample_length : int
        Length of each sequential sample
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Bootstrap sample indices
    """
    np.random.seed(seed)
    n_indices = len(indices)

    # Handle edge cases
    if sample_length >= n_indices:
        if n_samples <= n_indices:
            return indices[:n_samples].copy()
        # Repeat indices to meet n_samples requirement
        repeats = (n_samples // n_indices) + 1
        result = np.empty(repeats * n_indices, dtype=indices.dtype)
        for i in range(repeats):
            result[i * n_indices : (i + 1) * n_indices] = indices
        return result[:n_samples]

    # Pre-allocate result array
    result = np.empty(n_samples, dtype=indices.dtype)
    filled = 0

    while filled < n_samples:
        # Sample a random starting point
        start_idx = np.random.randint(0, n_indices - sample_length + 1)

        # Determine how many samples to take
        samples_to_take = min(sample_length, n_samples - filled)

        # Copy sequential samples
        for i in range(samples_to_take):
            result[filled + i] = indices[start_idx + i]

        filled += samples_to_take

    return result


@jit(nopython=True, cache=True)
def rolling_sharpe_numba(
    returns: np.ndarray,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Numba-optimized rolling Sharpe ratio calculation.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    window : int
        Rolling window size
    risk_free_rate : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods per year for annualization

    Returns
    -------
    np.ndarray
        Array of rolling Sharpe ratios
    """
    n = len(returns)
    if n < window:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)
    daily_rf = risk_free_rate / periods_per_year
    sqrt_periods = np.sqrt(periods_per_year)

    for i in range(window - 1, n):
        window_returns = returns[i - window + 1 : i + 1]
        excess_returns = window_returns - daily_rf

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess > 0:
            result[i] = mean_excess / std_excess * sqrt_periods
        else:
            # If std is zero, check if mean is also zero
            if abs(mean_excess) < 1e-10:
                result[i] = 0.0
            else:
                result[i] = np.nan

    return result


@jit(nopython=True, cache=True, parallel=True)
def calculate_ic_vectorized(
    predictions: np.ndarray,
    returns: np.ndarray,
    method: int = 0,  # 0=pearson, 1=spearman
) -> float:
    """Numba-optimized Information Coefficient calculation.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions
    returns : np.ndarray
        Array of returns
    method : int
        0 for Pearson, 1 for Spearman

    Returns
    -------
    float
        Information coefficient
    """
    n = len(predictions)
    if n != len(returns) or n < 2:
        return np.nan

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[valid_mask]
    ret_clean = returns[valid_mask]

    if len(pred_clean) < 2:
        return np.nan

    if method == 1:  # Spearman
        # Rank the data
        pred_clean = _rank_data_numba(pred_clean)
        ret_clean = _rank_data_numba(ret_clean)

    # Calculate Pearson correlation
    pred_mean = np.mean(pred_clean)
    ret_mean = np.mean(ret_clean)

    numerator = np.sum((pred_clean - pred_mean) * (ret_clean - ret_mean))
    denominator = np.sqrt(
        np.sum((pred_clean - pred_mean) ** 2) * np.sum((ret_clean - ret_mean) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


@jit(nopython=True, cache=True)
def _rank_data_numba(data: np.ndarray) -> np.ndarray:
    """Helper function to rank data for Spearman correlation."""
    n = len(data)
    indices = np.argsort(data)
    ranks = np.empty(n)

    for i in range(n):
        ranks[indices[i]] = i + 1

    # Handle ties by averaging ranks
    sorted_data = data[indices]
    i = 0
    while i < n:
        j = i
        # Find all equal values
        while j < n - 1 and sorted_data[j] == sorted_data[j + 1]:
            j += 1
        # Average ranks for ties
        if i != j:
            avg_rank = (ranks[indices[i]] + ranks[indices[j]]) / 2
            for k in range(i, j + 1):
                ranks[indices[k]] = avg_rank
        i = j + 1

    return ranks
