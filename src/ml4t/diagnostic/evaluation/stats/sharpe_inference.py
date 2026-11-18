"""Sharpe ratio inference: variance estimation and multiple testing adjustment.

This module implements the statistical inference framework for Sharpe ratios:

- Variance of Sharpe ratio estimator (2025 formula with autocorrelation)
- Expected maximum Sharpe under null (for multiple testing)
- Standard deviation of the maximum Sharpe ratio across trials

Equation and exhibit numbers refer to the version of the paper dated
7 March 2026. Earlier versions number the multiple-testing equations and the
re-scaling exhibit one lower.

References
----------
López de Prado, M., Lipton, A., & Zoonekynd, V. (2025).
"How to Use the Sharpe Ratio." ADIA Lab Research Paper Series, No. 19.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, ndtri
from scipy.stats import norm

# Euler-Mascheroni constant for E[max{Z}] calculation
EULER_GAMMA = 0.5772156649015329

# Published values of √V[max{X_k}], the factor in Equation 30, for selected K.
# Source: López de Prado et al. (2025), Exhibit 4. The exhibit lists K = 1 to
# 100 in rows of ten, so K = 20 is the last entry of the second row.
# get_variance_rescaling_factor computes the factor for any K and reproduces
# these values to five decimals.
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
    20: 0.52507,
    30: 0.49582,
    40: 0.47748,
    50: 0.46445,
    60: 0.45448,
    70: 0.44650,
    80: 0.43989,
    90: 0.43428,
    100: 0.42942,
}

# Probability mass left outside the integration range of the maximum's density.
_MAX_DENSITY_TAIL = 1e-18


@lru_cache(maxsize=1)
def _gauss_legendre_rule() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes and weights on [-1, 1]; 200 nodes give 8 decimals for K up to 1e9."""
    nodes, weights = np.polynomial.legendre.leggauss(200)
    return nodes, weights


@overload
def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = ...,
    n_samples: int = ...,
    skewness: float = ...,
    kurtosis: float = ...,
    return_components: Literal[False] = ...,
) -> float: ...


@overload
def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = ...,
    n_samples: int = ...,
    skewness: float = ...,
    kurtosis: float = ...,
    return_components: Literal[True] = ...,
) -> dict[str, float]: ...


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    n_samples: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Calculate the Probabilistic Sharpe Ratio.

    PSR estimates the probability that a strategy's true Sharpe ratio exceeds a
    benchmark Sharpe ratio after accounting for sample size, skewness, and
    kurtosis.
    """
    if n_samples < 2:
        if return_components:
            return {"psr": 0.5, "z_score": 0.0, "std_sr": np.inf}
        return 0.5

    variance_component = 1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2
    if variance_component <= 0:
        variance_component = 0.01

    std_sr = np.sqrt(variance_component / (n_samples - 1))
    if std_sr > 0:
        z_score = (observed_sharpe - benchmark_sharpe) / std_sr
    else:
        z_score = np.inf if observed_sharpe > benchmark_sharpe else -np.inf

    psr = float(norm.cdf(z_score))
    if return_components:
        return {
            "psr": psr,
            "z_score": float(z_score) if np.isfinite(z_score) else 0.0,
            "std_sr": float(std_sr),
        }
    return psr


def get_variance_rescaling_factor(k: float) -> float:
    """Standard deviation of the maximum of K independent standard normals.

    This is √V[max_k{X_k}], the factor in Equation 30. The maximum has density
    K φ(m) Z(m)^(K-1) (López de Prado et al. (2025), Appendix 4), whose first
    two moments are integrated numerically. The result reproduces Exhibit 4 to
    five decimals for K = 1 to 100, and it is defined for any K >= 1, including
    a non-integer effective number of trials and K above 100.

    Parameters
    ----------
    k : float
        Number of independent trials

    Returns
    -------
    float
        Rescaling factor; 1.0 for K <= 1
    """
    if k <= 1:
        return 1.0

    # Integration range that leaves _MAX_DENSITY_TAIL of probability on each
    # side: P[max <= lower] = Z(lower)^K and P[max > upper] <= K (1 - Z(upper)).
    lower_cdf = _MAX_DENSITY_TAIL ** (1.0 / k)
    if lower_cdf < 0.5:
        lower = float(ndtri(lower_cdf))
    else:
        lower = float(norm.isf(-np.expm1(np.log(_MAX_DENSITY_TAIL) / k)))
    upper = float(norm.isf(_MAX_DENSITY_TAIL / k))

    nodes, weights = _gauss_legendre_rule()
    half_width = 0.5 * (upper - lower)
    m = half_width * nodes + 0.5 * (upper + lower)
    density = k * np.exp((k - 1) * log_ndtr(m) + norm.logpdf(m))
    mass = half_width * weights * density

    mean = float(np.sum(m * mass))
    second_moment = float(np.sum(m * m * mass))
    return float(np.sqrt(second_moment - mean**2))


def compute_sharpe_variance(
    sharpe: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
    autocorrelation: float,
    n_trials: float = 1,
) -> float:
    """Compute variance of Sharpe ratio estimator.

    Implements the full 2025 formula with autocorrelation correction:

    .. math::

        \\sigma^2[\\widehat{SR}] = \\frac{1}{T} \\left[
            \\frac{1+\\rho}{1-\\rho}
            - \\left(1 + \\frac{\\rho}{1-\\rho} + \\frac{\\rho^2}{1-\\rho^2}\\right) \\gamma_3 SR
            + \\frac{1+\\rho^2}{1-\\rho^2} \\frac{\\gamma_4 - 1}{4} SR^2
        \\right]

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
    n_trials : float, default 1
        When greater than 1, the estimator variance is multiplied by
        V[max_k{X_k}] for K = ``n_trials``. The paper has no equation for this
        product: its Equation 30 rescales the dispersion of Sharpe ratios
        across trials, which :func:`compute_deflated_sharpe_std` implements.
        The deflated Sharpe ratio does not use this argument. Leave it at 1.

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
    López de Prado et al. (2025), Equations 2-5.
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

    # Not Equation 30, which rescales the dispersion across trials. See the
    # n_trials entry in the docstring.
    if n_trials > 1:
        rescaling_factor = get_variance_rescaling_factor(n_trials)
        variance *= rescaling_factor**2

    return max(variance, 0.0)  # Ensure non-negative


def compute_expected_max_sharpe(n_trials: float, variance_trials: float) -> float:
    """Compute expected maximum Sharpe ratio under null hypothesis.

    .. math::

        E[\\max_k\\{\\widehat{SR}_k\\}] \\approx \\sqrt{V[\\{\\widehat{SR}_k\\}]}
        \\left((1-\\gamma) \\Phi^{-1}(1-1/K) + \\gamma \\Phi^{-1}(1-1/(Ke))\\right)

    where γ is the Euler-Mascheroni constant ≈ 0.5772.

    Parameters
    ----------
    n_trials : float
        Number of strategies tested (K or K_eff)
    variance_trials : float
        Cross-sectional variance of Sharpe ratios: Var[{SR_1, ..., SR_K}]

    Returns
    -------
    float
        Expected maximum Sharpe ratio E[max{SR}]

    References
    ----------
    López de Prado et al. (2025), Equation 28.
    Bailey & López de Prado (2014), Appendix A.1, Equation 6.
    """
    if n_trials <= 1:
        return 0.0

    if variance_trials <= 0:
        return 0.0

    # E[max{Z}] for K i.i.d. standard normals (Equation 28)
    quantile_1 = norm.ppf(1 - 1 / n_trials)
    quantile_2 = norm.ppf(1 - 1 / (n_trials * np.e))
    e_max_z = (1 - EULER_GAMMA) * quantile_1 + EULER_GAMMA * quantile_2

    # Scale by standard deviation of Sharpe ratios across trials
    return float(np.sqrt(variance_trials) * e_max_z)


# np.var of equal floats can return a value of order 1e-34 instead of zero,
# because their mean is not always representable. A standard deviation across
# trials below this fraction of the estimator's standard deviation is treated
# as no dispersion.
_NEGLIGIBLE_TRIAL_DISPERSION = 1e-8


def compute_deflated_sharpe_std(
    threshold_sharpe: float,
    n_samples: int,
    skewness: float,
    kurtosis: float,
    autocorrelation: float,
    n_trials: float,
    variance_trials: float,
) -> float:
    """Standard deviation in the denominator of the deflated Sharpe ratio.

    With more than one trial and a positive variance of Sharpe ratios across
    trials, this is the standard deviation of the maximum Sharpe ratio:

    .. math::

        \\sigma[SR_{0,K}] = \\sqrt{V[\\{\\widehat{SR}_k\\}]}
        \\sqrt{V[\\max_k\\{X_k\\}]}

    where the X_k are K independent standard normal variables. This value does
    not depend on the sample length or on the skewness, kurtosis and
    autocorrelation of the selected strategy's returns. Those properties enter
    only through the variance of Sharpe ratios across trials.

    With one trial, or when the Sharpe ratios of the trials do not differ, no
    maximum over distinct trials exists. The result is then the standard
    deviation of the Sharpe ratio estimator at ``threshold_sharpe``, as in the
    probabilistic Sharpe ratio. Trials count as not differing when the standard
    deviation of their Sharpe ratios is below 1e-8 of the estimator's standard
    deviation: the variance of equal floating-point values is not always
    exactly zero.

    Parameters
    ----------
    threshold_sharpe : float
        Sharpe ratio under the null hypothesis (benchmark plus expected maximum)
    n_samples : int
        Number of observations (T)
    skewness : float
        Return skewness (γ₃)
    kurtosis : float
        Return kurtosis (γ₄), Pearson convention (normal = 3)
    autocorrelation : float
        First-order autocorrelation (ρ), must be in (-1, 1)
    n_trials : float
        Number of strategies tested (K or K_eff)
    variance_trials : float
        Cross-sectional variance of Sharpe ratios: Var[{SR_1, ..., SR_K}]

    Returns
    -------
    float
        Standard deviation for the deflated Sharpe ratio test statistic

    References
    ----------
    López de Prado et al. (2025), Equation 30 (more than one trial) and
    Equation 5 (one trial).
    """
    estimator_std = float(
        np.sqrt(
            compute_sharpe_variance(
                sharpe=threshold_sharpe,
                n_samples=n_samples,
                skewness=skewness,
                kurtosis=kurtosis,
                autocorrelation=autocorrelation,
            )
        )
    )
    if n_trials > 1 and variance_trials > 0:
        trials_std = float(np.sqrt(variance_trials))
        if trials_std > _NEGLIGIBLE_TRIAL_DISPERSION * estimator_std:
            return trials_std * get_variance_rescaling_factor(n_trials)

    return estimator_std


__all__ = [
    "EULER_GAMMA",
    "VARIANCE_RESCALING_FACTORS",
    "get_variance_rescaling_factor",
    "compute_sharpe_variance",
    "compute_expected_max_sharpe",
    "compute_deflated_sharpe_std",
]
