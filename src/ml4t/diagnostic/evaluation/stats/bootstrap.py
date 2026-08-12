"""Bootstrap methods for statistical inference on time series data.

This module implements bootstrap methods that preserve temporal dependence
structure, which is critical for financial time series:
- Stationary bootstrap (Politis & Romano, 1994)
- Block bootstrap variants

These methods are essential for valid statistical inference when data
exhibits autocorrelation, which is common in financial returns.
"""

import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from ml4t.diagnostic.backends.adapter import DataFrameAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def stationary_bootstrap_ic(
    predictions: Union[pl.Series, pd.Series, "NDArray[Any]"],
    returns: Union[pl.Series, pd.Series, "NDArray[Any]"],
    n_samples: int = 1000,
    block_size: float | None = None,
    confidence_level: float = 0.95,
    return_details: bool = True,
    seed: int | np.random.Generator | None = 0,
) -> float | dict[str, Any]:
    """Calculate p-value and confidence intervals for IC using stationary bootstrap.

    This method is more rigorous than the HAC approximation as it:
    1. Preserves the temporal dependence structure of the data
    2. Does not rely on asymptotic approximations for rank correlation
    3. Provides accurate confidence intervals for finite samples

    The stationary bootstrap (Politis & Romano, 1994) generates bootstrap samples
    by resampling blocks of random length from the original data, preserving the
    weak dependence structure of the time series.

    Parameters
    ----------
    predictions : array-like
        Model predictions or signals
    returns : array-like
        Actual returns or target values
    n_samples : int, default=1000
        Number of bootstrap samples to generate
    block_size : float, optional
        Expected block size for the stationary bootstrap.
        If None, uses optimal block size based on data autocorrelation.
    confidence_level : float, default=0.95
        Confidence level for the confidence interval
    return_details : bool, default=True
        If True, returns detailed results including CI and p-value
    seed : int or np.random.Generator or None, default=0
        Source of randomness for the resampling. The default makes a repeated
        call on the same data return the same p-value and the same interval,
        which is what a reader re-running a notebook needs. Pass a Generator to
        supply your own stream, or None for a fresh unpredictable one.

    Returns
    -------
    float or dict
        If return_details=False: p-value for the null hypothesis (IC=0)
        If return_details=True: Dictionary containing IC, p_value, CI, etc.

    Notes
    -----
    Before the ``seed`` parameter existed this drew from numpy's legacy global
    generator, so the p-value moved on every call and any decision taken against
    a threshold moved with it. Measured on ``etfs/04_model_based_features``: two
    consecutive production runs of identical code on identical data retained 7
    and then 8 of 14 features under Benjamini-Hochberg at 5%, because ``ffd_lqd``
    went p = 0.029 to p = 0.023 across the threshold.

    References
    ----------
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
    Journal of the American Statistical Association, 89(428), 1303-1313.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Convert to numpy arrays
    pred_array = DataFrameAdapter.to_numpy(predictions).flatten()
    ret_array = DataFrameAdapter.to_numpy(returns).flatten()

    # Validate inputs
    if len(pred_array) != len(ret_array):
        raise ValueError("Predictions and returns must have the same length")

    # Remove NaN pairs
    valid_mask = ~(np.isnan(pred_array) | np.isnan(ret_array))
    pred_clean = pred_array[valid_mask]
    ret_clean = ret_array[valid_mask]

    n = len(pred_clean)
    if n < 30:
        warnings.warn(
            f"Sample size ({n}) may be too small for reliable bootstrap inference",
            stacklevel=2,
        )

    if n > 50_000:
        warnings.warn(
            f"Input has {n:,} observations. stationary_bootstrap_ic() computes "
            f"spearmanr() per bootstrap iteration — this will be very slow for "
            f"large arrays. For panel data, bootstrap the cross-sectional IC time "
            f"series instead (one IC per date, then bootstrap that series).",
            stacklevel=2,
        )

    # Calculate observed IC
    observed_ic, _ = spearmanr(pred_clean, ret_clean)

    if np.isnan(observed_ic):
        if return_details:
            return {
                "ic": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
            }
        return np.nan

    # Determine optimal block size if not provided
    if block_size is None:
        block_size = _optimal_block_size(ret_clean)

    # Generate bootstrap samples under null hypothesis
    bootstrap_ics_null = np.zeros(n_samples)

    for i in range(n_samples):
        # Generate stationary bootstrap sample
        boot_indices = _stationary_bootstrap_indices(n, block_size, rng)
        # Break relationship by independently bootstrapping predictions
        boot_pred_null = pred_clean[_stationary_bootstrap_indices(n, block_size, rng)]
        boot_ret = ret_clean[boot_indices]

        # Calculate IC on bootstrap sample
        ic_boot, _ = spearmanr(boot_pred_null, boot_ret)
        bootstrap_ics_null[i] = ic_boot if not np.isnan(ic_boot) else 0.0

    # Calculate p-value (two-tailed test)
    p_value = np.mean(np.abs(bootstrap_ics_null) >= np.abs(observed_ic))

    # Calculate confidence interval using percentile method
    bootstrap_ics_actual = np.zeros(n_samples)
    for i in range(n_samples):
        boot_indices = _stationary_bootstrap_indices(n, block_size, rng)
        boot_pred = pred_clean[boot_indices]
        boot_ret = ret_clean[boot_indices]
        ic_boot, _ = spearmanr(boot_pred, boot_ret)
        bootstrap_ics_actual[i] = ic_boot if not np.isnan(ic_boot) else observed_ic

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_ics_actual, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_ics_actual, 100 * (1 - alpha / 2))

    if not return_details:
        return float(p_value)

    return {
        "ic": float(observed_ic),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_mean": float(np.mean(bootstrap_ics_actual)),
        "bootstrap_std": float(np.std(bootstrap_ics_actual)),
    }


def _stationary_bootstrap_indices(
    n: int,
    block_size: float,
    rng: np.random.Generator,
) -> "NDArray[np.int_]":
    """Generate indices for one stationary bootstrap sample.

    Parameters
    ----------
    n : int
        Sample size
    block_size : float
        Expected block size (1/p where p is the probability of ending a block)
    rng : np.random.Generator
        Source of randomness. Passed in rather than drawn from the module-level
        legacy generator so a caller's result reproduces.

    Returns
    -------
    np.ndarray
        Bootstrap indices of length n
    """
    p = 1.0 / block_size  # Probability of ending a block
    indices: list[int] = []

    while len(indices) < n:
        # Start a new block at a random position
        start_idx = rng.integers(0, n)
        # Generate block length from geometric distribution
        block_length = rng.geometric(p)
        # Add indices from this block (with wrapping)
        for j in range(block_length):
            if len(indices) >= n:
                break
            indices.append((start_idx + j) % n)

    return np.array(indices[:n], dtype=np.int_)


def _optimal_block_size(data: "NDArray[Any]") -> float:
    """Estimate optimal block size for stationary bootstrap using autocorrelation.

    Uses a simple rule based on lag-1 autocorrelation to determine block size.
    Higher autocorrelation requires larger blocks to preserve dependence structure.

    Parameters
    ----------
    data : np.ndarray
        Time series data

    Returns
    -------
    float
        Optimal block size
    """
    n = len(data)

    if n < 10:
        return max(1, n // 3)

    # Standardize the data
    data_std = (data - np.mean(data)) / (np.std(data) + 1e-10)

    # Calculate lag-1 autocorrelation
    acf_1 = np.corrcoef(data_std[:-1], data_std[1:])[0, 1]

    # Simple rule: block size increases with autocorrelation
    if np.isnan(acf_1) or acf_1 < 0:
        block_size = max(1, int(n ** (1 / 3)))
    else:
        # Positive autocorrelation: larger blocks needed
        block_size = max(1, int(n ** (1 / 3) * (1 + 2 * acf_1)))

    # Cap at n/3 to ensure reasonable variation
    return min(block_size, n // 3)


__all__ = [
    "stationary_bootstrap_ic",
    "_stationary_bootstrap_indices",
    "_optimal_block_size",
]
