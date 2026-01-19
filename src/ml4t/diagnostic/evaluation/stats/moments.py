"""Return statistics computation for Sharpe ratio analysis.

This module provides functions for computing the statistical moments
needed for Sharpe ratio inference: mean, std, skewness, kurtosis,
and autocorrelation.

These are the building blocks for DSR/PSR calculations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_return_statistics(
    returns: ArrayLike,
) -> tuple[float, float, float, float, int]:
    """Compute Sharpe ratio and distribution statistics from returns.

    Parameters
    ----------
    returns : array-like
        Array of returns (not prices). NaN values are removed.

    Returns
    -------
    tuple of (sharpe, skewness, kurtosis, autocorrelation, n_samples)
        - sharpe: Sharpe ratio (mean/std) at native frequency
        - skewness: Fisher's skewness (γ₃)
        - kurtosis: Pearson kurtosis (γ₄), normal = 3
        - autocorrelation: Lag-1 autocorrelation (ρ)
        - n_samples: Number of valid observations

    Raises
    ------
    ValueError
        If fewer than 2 observations or zero variance.

    Notes
    -----
    Kurtosis is returned in Pearson convention (normal=3) for internal use.
    Convert to Fisher (normal=0) for public API: excess_kurtosis = kurtosis - 3.
    """
    returns = np.asarray(returns).flatten()
    returns = returns[~np.isnan(returns)]

    n = len(returns)
    if n < 2:
        raise ValueError("Need at least 2 return observations")

    mean = np.mean(returns)
    std = np.std(returns, ddof=1)

    if std == 0:
        raise ValueError("Return series has zero variance")

    sharpe = mean / std

    # Skewness (γ₃) - Fisher's definition
    skewness = float(((returns - mean) ** 3).mean() / std**3)

    # Kurtosis (γ₄) - Pearson (normal = 3)
    kurtosis = float(((returns - mean) ** 4).mean() / std**4)

    # First-order autocorrelation (ρ)
    if n > 2:
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    return float(sharpe), skewness, kurtosis, float(autocorr), n


def compute_sharpe(returns: ArrayLike) -> float:
    """Compute Sharpe ratio from returns.

    Parameters
    ----------
    returns : array-like
        Array of returns.

    Returns
    -------
    float
        Sharpe ratio (mean/std) at native frequency.
    """
    sharpe, _, _, _, _ = compute_return_statistics(returns)
    return sharpe


def compute_skewness(returns: ArrayLike) -> float:
    """Compute skewness from returns.

    Parameters
    ----------
    returns : array-like
        Array of returns.

    Returns
    -------
    float
        Fisher's skewness (γ₃).
    """
    _, skewness, _, _, _ = compute_return_statistics(returns)
    return skewness


def compute_kurtosis(returns: ArrayLike, excess: bool = True) -> float:
    """Compute kurtosis from returns.

    Parameters
    ----------
    returns : array-like
        Array of returns.
    excess : bool, default True
        If True, return Fisher/excess kurtosis (normal=0).
        If False, return Pearson kurtosis (normal=3).

    Returns
    -------
    float
        Kurtosis value.
    """
    _, _, kurtosis, _, _ = compute_return_statistics(returns)
    return kurtosis - 3.0 if excess else kurtosis


def compute_autocorrelation(returns: ArrayLike, lag: int = 1) -> float:
    """Compute autocorrelation from returns.

    Parameters
    ----------
    returns : array-like
        Array of returns.
    lag : int, default 1
        Lag for autocorrelation. Currently only lag=1 is supported.

    Returns
    -------
    float
        Autocorrelation at specified lag.

    Raises
    ------
    ValueError
        If lag != 1 (not yet implemented).
    """
    if lag != 1:
        raise ValueError("Only lag=1 autocorrelation is currently supported")

    _, _, _, autocorr, _ = compute_return_statistics(returns)
    return autocorr


__all__ = [
    "compute_return_statistics",
    "compute_sharpe",
    "compute_skewness",
    "compute_kurtosis",
    "compute_autocorrelation",
]
