"""Sharpe ratio inference: variance estimation and multiple testing adjustment.

This module implements the statistical inference framework for Sharpe ratios:

- Variance of Sharpe ratio estimator (2025 formula with autocorrelation)
- Expected maximum Sharpe under null (for multiple testing)
- Variance rescaling factors for selection bias

References
----------
López de Prado, M., Lipton, A., & Zoonekynd, V. (2025).
"How to Use the Sharpe Ratio." ADIA Lab Research Paper Series, No. 19.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Euler-Mascheroni constant for E[max{Z}] calculation
EULER_GAMMA = 0.5772156649015329

# Standard deviation rescaling factors for maximum of K standard normals
# Source: López de Prado et al. (2025), Exhibit 3, page 13
# These are √V[max{X_k}] values for DSR variance adjustment
VARIANCE_RESCALING_FACTORS: dict[int, float] = {
    1: 1.00000,
    2: 0.82565,
    3: 0.74798,
    4: 0.70122,
    5: 0.66898,
    6: 0.64492,
    7: 0.62603,
    8: 0.61065,
    9: 0.59779,
    10: 0.58681,
    20: 0.52131,
    30: 0.49364,
    40: 0.47599,
    50: 0.46334,
    60: 0.45361,
    70: 0.44579,
    80: 0.43929,
    90: 0.43376,
    100: 0.42942,
}


def get_variance_rescaling_factor(k: int) -> float:
    """Get variance rescaling factor √V[max{X_k}] for K trials.

    Source: López de Prado et al. (2025), Exhibit 3, page 13.

    Parameters
    ----------
    k : int
        Number of independent trials

    Returns
    -------
    float
        Rescaling factor (uses linear interpolation for unlisted values)
    """
    if k in VARIANCE_RESCALING_FACTORS:
        return VARIANCE_RESCALING_FACTORS[k]

    keys = sorted(VARIANCE_RESCALING_FACTORS.keys())
    if k < keys[0]:
        return VARIANCE_RESCALING_FACTORS[keys[0]]
    if k > keys[-1]:
        return VARIANCE_RESCALING_FACTORS[keys[-1]]

    # Linear interpolation
    for i in range(len(keys) - 1):
        if keys[i] <= k <= keys[i + 1]:
            k1, k2 = keys[i], keys[i + 1]
            v1, v2 = VARIANCE_RESCALING_FACTORS[k1], VARIANCE_RESCALING_FACTORS[k2]
            return v1 + (v2 - v1) * (k - k1) / (k2 - k1)

    return VARIANCE_RESCALING_FACTORS[keys[-1]]


def compute_sharpe_variance(
    sharpe: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
    autocorrelation: float,
    n_trials: int = 1,
) -> float:
    """Compute variance of Sharpe ratio estimator.

    Implements the full 2025 formula with autocorrelation correction:

    .. math::

        \\sigma^2[\\widehat{SR}] = \\frac{1}{T} \\left[
            \\frac{1+\\rho}{1-\\rho}
            - \\left(1 + \\frac{\\rho}{1-\\rho} + \\frac{\\rho^2}{1-\\rho^2}\\right) \\gamma_3 SR
            + \\frac{1+\\rho^2}{1-\\rho^2} \\frac{\\gamma_4 - 1}{4} SR^2
        \\right] \\times V[\\max_k\\{X_k\\}]

    When ρ=0 (i.i.d. assumption), this reduces to the 2014 formula:

    .. math::

        \\sigma^2[\\widehat{SR}] = \\frac{1}{T} \\left[
            1 - \\gamma_3 SR + \\frac{\\gamma_4 - 1}{4} SR^2
        \\right]

    Parameters
    ----------
    sharpe : float
        Sharpe ratio (at native frequency)
    n_samples : int
        Number of observations (T)
    skewness : float
        Return skewness (γ₃)
    kurtosis : float
        Return kurtosis (γ₄), Pearson convention (normal = 3)
    autocorrelation : float
        First-order autocorrelation (ρ), must be in (-1, 1)
    n_trials : int, default 1
        Number of strategies (K) for variance rescaling

    Returns
    -------
    float
        Variance of Sharpe ratio estimator

    Raises
    ------
    ValueError
        If autocorrelation is not in (-1, 1).

    References
    ----------
    López de Prado et al. (2025), Equations 2-5, pages 5-7.
    """
    rho = autocorrelation

    # Validate autocorrelation
    if abs(rho) >= 1.0:
        raise ValueError(f"Autocorrelation must be in (-1, 1), got {rho}")

    # Compute coefficients (from reference implementation)
    # When rho=0: coef_a=1, coef_b=0, coef_c=0, so a=1, b=1, c=1 (reduces to 2014 formula)
    coef_a = 1.0
    if rho != 0:
        coef_b = rho / (1 - rho)
        coef_c = rho**2 / (1 - rho**2)
    else:
        coef_b = 0.0
        coef_c = 0.0

    a = coef_a + 2 * coef_b  # = (1+ρ)/(1-ρ) - base term coefficient
    b = coef_a + coef_b + coef_c  # = 1 + ρ/(1-ρ) + ρ²/(1-ρ²) - skewness coefficient
    c = coef_a + 2 * coef_c  # = (1+ρ²)/(1-ρ²) - kurtosis coefficient

    # Variance formula (Equation 5)
    variance = (a - b * skewness * sharpe + c * (kurtosis - 1) / 4 * sharpe**2) / n_samples

    # Apply variance rescaling for multiple testing (Equation 29)
    if n_trials > 1:
        rescaling_factor = get_variance_rescaling_factor(n_trials)
        variance *= rescaling_factor**2

    return max(variance, 0.0)  # Ensure non-negative


def compute_expected_max_sharpe(n_trials: int, variance_trials: float) -> float:
    """Compute expected maximum Sharpe ratio under null hypothesis.

    .. math::

        E[\\max_k\\{\\widehat{SR}_k\\}] \\approx \\sqrt{V[\\{\\widehat{SR}_k\\}]}
        \\left((1-\\gamma) \\Phi^{-1}(1-1/K) + \\gamma \\Phi^{-1}(1-1/(Ke))\\right)

    where γ is the Euler-Mascheroni constant ≈ 0.5772.

    Parameters
    ----------
    n_trials : int
        Number of strategies tested (K)
    variance_trials : float
        Cross-sectional variance of Sharpe ratios: Var[{SR_1, ..., SR_K}]

    Returns
    -------
    float
        Expected maximum Sharpe ratio E[max{SR}]

    References
    ----------
    López de Prado et al. (2025), Equation 26, page 13.
    Bailey & López de Prado (2014), Appendix A.1, Equation 6.
    """
    if n_trials <= 1:
        return 0.0

    if variance_trials <= 0:
        return 0.0

    # E[max{Z}] for K i.i.d. standard normals (Equation 26)
    quantile_1 = norm.ppf(1 - 1 / n_trials)
    quantile_2 = norm.ppf(1 - 1 / (n_trials * np.e))
    e_max_z = (1 - EULER_GAMMA) * quantile_1 + EULER_GAMMA * quantile_2

    # Scale by standard deviation of Sharpe ratios across trials
    return float(np.sqrt(variance_trials) * e_max_z)


__all__ = [
    "EULER_GAMMA",
    "VARIANCE_RESCALING_FACTORS",
    "get_variance_rescaling_factor",
    "compute_sharpe_variance",
    "compute_expected_max_sharpe",
]
