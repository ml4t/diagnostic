"""Minimum Track Record Length (MinTRL) calculation.

MinTRL is the minimum number of observations required to reject the null
hypothesis (SR ≤ target) at the specified confidence level.

References
----------
López de Prado, M., Lipton, A., & Zoonekynd, V. (2025).
"How to Use the Sharpe Ratio." ADIA Lab Research Paper Series, No. 19.
Equation 11, page 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ml4t.diagnostic.evaluation.stats.moments import compute_return_statistics
from ml4t.diagnostic.evaluation.stats.sharpe_inference import compute_expected_max_sharpe

# Type alias
Frequency = Literal["daily", "weekly", "monthly"]

# Default trading periods per year
DEFAULT_PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


@dataclass
class MinTRLResult:
    """Result of Minimum Track Record Length calculation.

    Attributes
    ----------
    min_trl : float
        Minimum observations needed to reject null at specified confidence.
        Can be math.inf if observed SR <= target SR.
    min_trl_years : float
        Minimum track record in calendar years. Can be math.inf.
    current_samples : int
        Current number of observations.
    has_adequate_sample : bool
        Whether current_samples >= min_trl.
    deficit : float
        Additional observations needed (0 if adequate). Can be math.inf.
    deficit_years : float
        Additional years needed (0 if adequate). Can be math.inf.
    observed_sharpe : float
        The observed Sharpe ratio used in calculation.
    target_sharpe : float
        The target Sharpe ratio (null hypothesis).
    confidence_level : float
        Confidence level for the test (e.g., 0.95).
    skewness : float
        Skewness of returns (0 for normal).
    excess_kurtosis : float
        Excess kurtosis of returns (Fisher convention: 0 for normal).
    autocorrelation : float
        Lag-1 autocorrelation of returns.
    frequency : str
        Return frequency ('daily', 'weekly', etc.).
    periods_per_year : int
        Periods per year for annualization.
    """

    min_trl: float
    min_trl_years: float
    current_samples: int
    has_adequate_sample: bool
    deficit: float
    deficit_years: float

    # Parameters used
    observed_sharpe: float
    target_sharpe: float
    confidence_level: float
    skewness: float
    excess_kurtosis: float
    autocorrelation: float
    frequency: str
    periods_per_year: int

    def interpret(self) -> str:
        """Generate human-readable interpretation."""
        if math.isinf(self.min_trl):
            return (
                f"Minimum Track Record Length (MinTRL)\n"
                f"  Observed Sharpe: {self.observed_sharpe:.4f}\n"
                f"  Target Sharpe: {self.target_sharpe:.4f}\n"
                f"  Confidence: {self.confidence_level:.0%}\n"
                f"\n"
                f"  MinTRL: INFINITE (observed SR <= target SR)\n"
                f"  Status: Cannot reject null hypothesis at any sample size"
            )

        if self.has_adequate_sample:
            status = f"ADEQUATE: {self.current_samples} >= {int(self.min_trl)} observations"
        else:
            status = (
                f"INSUFFICIENT: Need {int(self.deficit)} more observations "
                f"({self.deficit_years:.1f} more years)"
            )

        return (
            f"Minimum Track Record Length (MinTRL)\n"
            f"  Observed Sharpe: {self.observed_sharpe:.4f}\n"
            f"  Target Sharpe: {self.target_sharpe:.4f}\n"
            f"  Confidence: {self.confidence_level:.0%}\n"
            f"\n"
            f"  MinTRL: {int(self.min_trl)} observations ({self.min_trl_years:.1f} years)\n"
            f"  Current: {self.current_samples} observations\n"
            f"  Status: {status}"
        )


def _compute_min_trl_core(
    observed_sharpe: float,
    target_sharpe: float,
    confidence_level: float,
    skewness: float,
    kurtosis: float,
    autocorrelation: float,
) -> float:
    """Core MinTRL formula (internal).

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio at native frequency
    target_sharpe : float
        Null hypothesis threshold (SR₀)
    confidence_level : float
        Required confidence level (e.g., 0.95)
    skewness : float
        Return skewness (γ₃)
    kurtosis : float
        Return kurtosis (γ₄), Pearson convention (normal = 3)
    autocorrelation : float
        First-order autocorrelation (ρ)

    Returns
    -------
    float
        Minimum number of observations. Returns math.inf if
        observed SR <= target SR.
    """
    rho = autocorrelation
    sr_diff = observed_sharpe - target_sharpe

    # If observed SR <= target SR, MinTRL is infinite
    if sr_diff <= 1e-10:
        return float("inf")

    # z-score for confidence level
    z_alpha = norm.ppf(confidence_level)

    # Coefficients (same as variance formula)
    coef_a = 1.0
    if rho != 0 and abs(rho) < 1:
        coef_b = rho / (1 - rho)
        coef_c = rho**2 / (1 - rho**2)
    else:
        coef_b = 0.0
        coef_c = 0.0

    a = coef_a + 2 * coef_b
    b = coef_a + coef_b + coef_c
    c = coef_a + 2 * coef_c

    # Variance term (without 1/T factor)
    var_term = a - b * skewness * target_sharpe + c * (kurtosis - 1) / 4 * target_sharpe**2

    # MinTRL formula (Equation 11)
    try:
        min_trl = var_term * (z_alpha / sr_diff) ** 2
        if np.isinf(min_trl):
            return float("inf")
        return float(np.ceil(max(min_trl, 1)))
    except (OverflowError, FloatingPointError):
        return float("inf")


def compute_min_trl(
    returns: ArrayLike | None = None,
    observed_sharpe: float | None = None,
    target_sharpe: float = 0.0,
    confidence_level: float = 0.95,
    frequency: Frequency = "daily",
    periods_per_year: int | None = None,
    *,
    skewness: float | None = None,
    excess_kurtosis: float | None = None,
    autocorrelation: float | None = None,
) -> MinTRLResult:
    """Compute Minimum Track Record Length (MinTRL).

    MinTRL is the minimum number of observations required to reject the null
    hypothesis (SR <= target) at the specified confidence level.

    Parameters
    ----------
    returns : array-like, optional
        Return series. If provided, statistics are computed from it.
    observed_sharpe : float, optional
        Observed Sharpe ratio. Required if returns not provided.
    target_sharpe : float, default 0.0
        Null hypothesis threshold (SR₀).
    confidence_level : float, default 0.95
        Required confidence level (1 - α).
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    periods_per_year : int, optional
        Periods per year (for converting to calendar time).
    skewness : float, optional
        Override computed skewness.
    excess_kurtosis : float, optional
        Override computed excess kurtosis (Fisher convention, normal=0).
    autocorrelation : float, optional
        Override computed autocorrelation.

    Returns
    -------
    MinTRLResult
        Results including min_trl, min_trl_years, and adequacy assessment.
        min_trl can be math.inf if observed SR <= target SR.

    Examples
    --------
    From returns:

    >>> result = compute_min_trl(daily_returns, frequency="daily")
    >>> print(f"Need {result.min_trl_years:.1f} years of data")

    From statistics:

    >>> result = compute_min_trl(
    ...     observed_sharpe=0.5,
    ...     target_sharpe=0.0,
    ...     confidence_level=0.95,
    ...     skewness=-1.0,
    ...     excess_kurtosis=2.0,
    ...     autocorrelation=0.1,
    ... )
    """
    # Resolve periods per year
    if periods_per_year is None:
        periods_per_year = DEFAULT_PERIODS_PER_YEAR[frequency]

    # Get statistics from returns or use provided values
    if returns is not None:
        ret_arr = np.asarray(returns).flatten()
        ret_arr = ret_arr[~np.isnan(ret_arr)]
        obs_sr, comp_skew, comp_kurt, comp_rho, n_samples = compute_return_statistics(ret_arr)

        if observed_sharpe is None:
            observed_sharpe = obs_sr
    else:
        if observed_sharpe is None:
            raise ValueError("Either returns or observed_sharpe must be provided")
        n_samples = 0  # Unknown
        comp_skew = 0.0
        comp_kurt = 3.0  # Pearson
        comp_rho = 0.0

    # Use provided or computed statistics
    skew = skewness if skewness is not None else comp_skew
    if excess_kurtosis is not None:
        kurt = excess_kurtosis + 3.0  # Fisher -> Pearson
    else:
        kurt = comp_kurt
    rho = autocorrelation if autocorrelation is not None else comp_rho

    # Compute MinTRL
    min_trl = _compute_min_trl_core(
        observed_sharpe=observed_sharpe,
        target_sharpe=target_sharpe,
        confidence_level=confidence_level,
        skewness=skew,
        kurtosis=kurt,
        autocorrelation=rho,
    )

    is_inf = math.isinf(min_trl)
    min_trl_years = float("inf") if is_inf else min_trl / periods_per_year
    has_adequate = False if is_inf or n_samples == 0 else n_samples >= min_trl
    deficit = (
        float("inf") if is_inf else max(0.0, min_trl - n_samples) if n_samples > 0 else min_trl
    )
    deficit_years = float("inf") if is_inf else deficit / periods_per_year

    return MinTRLResult(
        min_trl=min_trl,
        min_trl_years=float(min_trl_years),
        current_samples=n_samples,
        has_adequate_sample=has_adequate,
        deficit=deficit,
        deficit_years=float(deficit_years),
        observed_sharpe=float(observed_sharpe),
        target_sharpe=target_sharpe,
        confidence_level=confidence_level,
        skewness=float(skew),
        excess_kurtosis=float(kurt - 3.0),
        autocorrelation=float(rho),
        frequency=frequency,
        periods_per_year=periods_per_year,
    )


def min_trl_fwer(
    observed_sharpe: float,
    n_trials: int,
    variance_trials: float,
    target_sharpe: float = 0.0,
    confidence_level: float = 0.95,
    frequency: Frequency = "daily",
    periods_per_year: int | None = None,
    *,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    autocorrelation: float = 0.0,
) -> MinTRLResult:
    """Compute MinTRL under FWER multiple testing adjustment.

    When selecting the best strategy from K trials, the MinTRL must be adjusted
    to account for the selection bias.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio of the best strategy.
    n_trials : int
        Number of strategies tested (K).
    variance_trials : float
        Cross-sectional variance of Sharpe ratios.
    target_sharpe : float, default 0.0
        Original null hypothesis threshold.
    confidence_level : float, default 0.95
        Required confidence level.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    periods_per_year : int, optional
        Periods per year.
    skewness : float, default 0.0
        Return skewness.
    excess_kurtosis : float, default 0.0
        Return excess kurtosis (Fisher, normal=0).
    autocorrelation : float, default 0.0
        Return autocorrelation.

    Returns
    -------
    MinTRLResult
        Results with min_trl adjusted for multiple testing.
    """
    if periods_per_year is None:
        periods_per_year = DEFAULT_PERIODS_PER_YEAR[frequency]

    kurtosis = excess_kurtosis + 3.0

    # Compute expected max Sharpe (selection bias adjustment)
    expected_max = compute_expected_max_sharpe(n_trials, variance_trials)
    adjusted_target = target_sharpe + expected_max

    # Compute MinTRL with adjusted target
    min_trl = _compute_min_trl_core(
        observed_sharpe=observed_sharpe,
        target_sharpe=adjusted_target,
        confidence_level=confidence_level,
        skewness=skewness,
        kurtosis=kurtosis,
        autocorrelation=autocorrelation,
    )

    is_inf = math.isinf(min_trl)
    min_trl_years = float("inf") if is_inf else min_trl / periods_per_year

    return MinTRLResult(
        min_trl=min_trl,
        min_trl_years=float(min_trl_years),
        current_samples=0,
        has_adequate_sample=False,
        deficit=min_trl,
        deficit_years=float(min_trl_years),
        observed_sharpe=float(observed_sharpe),
        target_sharpe=float(adjusted_target),
        confidence_level=confidence_level,
        skewness=float(skewness),
        excess_kurtosis=float(excess_kurtosis),
        autocorrelation=float(autocorrelation),
        frequency=frequency,
        periods_per_year=periods_per_year,
    )


__all__ = [
    "MinTRLResult",
    "compute_min_trl",
    "min_trl_fwer",
    "DEFAULT_PERIODS_PER_YEAR",
]
