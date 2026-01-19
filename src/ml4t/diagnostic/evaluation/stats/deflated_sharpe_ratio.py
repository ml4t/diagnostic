"""Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).

This module provides the main entry points for Sharpe ratio inference:

- deflated_sharpe_ratio: Compute DSR/PSR from raw returns (recommended)
- deflated_sharpe_ratio_from_statistics: Compute DSR from pre-computed statistics

The underlying components are in separate modules:
- moments.py: Return statistics computation
- sharpe_inference.py: Variance estimation and expected max
- min_trl.py: Minimum Track Record Length
- pbo.py: Probability of Backtest Overfitting

References
----------
López de Prado, M., Lipton, A., & Zoonekynd, V. (2025).
"How to Use the Sharpe Ratio." ADIA Lab Research Paper Series, No. 19.

Bailey, D. H., & López de Prado, M. (2014).
"The Deflated Sharpe Ratio." Journal of Portfolio Management.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ml4t.diagnostic.evaluation.stats.backtest_overfitting import PBOResult, compute_pbo

# Import from decomposed modules
from ml4t.diagnostic.evaluation.stats.minimum_track_record import (
    DEFAULT_PERIODS_PER_YEAR,
    MinTRLResult,
    _compute_min_trl_core,
    compute_min_trl,
    min_trl_fwer,
)
from ml4t.diagnostic.evaluation.stats.moments import compute_return_statistics
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    VARIANCE_RESCALING_FACTORS,
    compute_expected_max_sharpe,
    compute_sharpe_variance,
    get_variance_rescaling_factor,
)

# Type alias
Frequency = Literal["daily", "weekly", "monthly"]


@dataclass
class DSRResult:
    """Result of Deflated/Probabilistic Sharpe Ratio analysis.

    Attributes
    ----------
    probability : float
        Probability that the true Sharpe ratio exceeds the benchmark,
        after correcting for multiple testing (if applicable).
        Range: [0, 1]. Higher is better.
    is_significant : bool
        Whether the result is significant at the specified confidence level.
    z_score : float
        Test statistic (z-score) for the hypothesis test.
    p_value : float
        P-value for the null hypothesis that true SR <= benchmark.

    sharpe_ratio : float
        Observed Sharpe ratio at native frequency.
    sharpe_ratio_annualized : float
        Annualized Sharpe ratio (for interpretation).
    benchmark_sharpe : float
        Null hypothesis threshold (default 0).

    n_samples : int
        Number of return observations (T).
    n_trials : int
        Number of strategies tested (K). K=1 means PSR, K>1 means DSR.
    frequency : str
        Return frequency ("daily", "weekly", "monthly").

    skewness : float
        Return distribution skewness (gamma_3).
    excess_kurtosis : float
        Return distribution excess kurtosis (gamma_4 - 3). Normal = 0.
        This is what scipy.stats.kurtosis() returns by default.
    autocorrelation : float
        First-order return autocorrelation (rho).

    expected_max_sharpe : float
        Expected maximum Sharpe from noise under multiple testing.
        E[max{SR}] from Equation 26. Zero for single strategy (PSR).
    deflated_sharpe : float
        Observed Sharpe minus expected max: SR - E[max{SR}].
    variance_trials : float
        Cross-sectional variance of Sharpe ratios across trials.

    min_trl : float
        Minimum Track Record Length in observations.
        Can be math.inf if observed SR <= target SR.
    min_trl_years : float
        Minimum Track Record Length in calendar years.
        Can be math.inf if observed SR <= target SR.
    has_adequate_sample : bool
        Whether n_samples >= min_trl.

    confidence_level : float
        Confidence level used for significance testing.
    """

    # Core inference results
    probability: float
    is_significant: bool
    z_score: float
    p_value: float

    # Sharpe ratios
    sharpe_ratio: float
    sharpe_ratio_annualized: float
    benchmark_sharpe: float

    # Sample information
    n_samples: int
    n_trials: int
    frequency: str
    periods_per_year: int

    # Computed statistics
    skewness: float
    excess_kurtosis: float  # Fisher convention: normal = 0
    autocorrelation: float

    # Multiple testing adjustment
    expected_max_sharpe: float
    deflated_sharpe: float
    variance_trials: float

    # Minimum track record
    min_trl: float  # Can be inf
    min_trl_years: float  # Can be inf
    has_adequate_sample: bool

    # Configuration
    confidence_level: float

    def interpret(self) -> str:
        """Generate human-readable interpretation of results."""
        if self.n_trials == 1:
            test_type = "Probabilistic Sharpe Ratio (PSR)"
            selection_note = ""
        else:
            test_type = f"Deflated Sharpe Ratio (DSR) - best of {self.n_trials} strategies"
            selection_note = (
                f"\n  Expected max from noise: {self.expected_max_sharpe:.4f}"
                f"\n  Deflated Sharpe: {self.deflated_sharpe:.4f}"
            )

        significance = "Yes" if self.is_significant else "No"
        confidence_pct = self.confidence_level * 100

        lines = [
            f"{test_type}",
            f"  Frequency: {self.frequency} ({self.periods_per_year} periods/year)",
            f"  Sample size: {self.n_samples} observations",
            "",
            f"  Sharpe ratio: {self.sharpe_ratio:.4f} "
            f"({self.sharpe_ratio_annualized:.2f} annualized)",
            f"  Benchmark: {self.benchmark_sharpe:.4f}",
            selection_note,
            "",
            f"  Probability of skill: {self.probability:.1%}",
            f"  Significant at {confidence_pct:.0f}%: {significance}",
            f"  P-value: {self.p_value:.4f}",
            "",
            "  Statistics used:",
            f"    Skewness (gamma_3): {self.skewness:.3f}",
            f"    Excess kurtosis (gamma_4-3): {self.excess_kurtosis:.3f}",
            f"    Autocorrelation (rho): {self.autocorrelation:.3f}",
        ]

        if math.isinf(self.min_trl):
            lines.extend(
                [
                    "",
                    "  WARNING: MinTRL is infinite (observed SR <= target SR)",
                    "    Cannot reject null hypothesis at any sample size",
                ]
            )
        elif not self.has_adequate_sample:
            deficit = self.min_trl - self.n_samples
            lines.extend(
                [
                    "",
                    "  WARNING: Insufficient sample size",
                    f"    Need {deficit:.0f} more observations ({self.min_trl_years:.1f} years total)",
                ]
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "probability": self.probability,
            "is_significant": self.is_significant,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "sharpe_ratio": self.sharpe_ratio,
            "sharpe_ratio_annualized": self.sharpe_ratio_annualized,
            "benchmark_sharpe": self.benchmark_sharpe,
            "n_samples": self.n_samples,
            "n_trials": self.n_trials,
            "frequency": self.frequency,
            "periods_per_year": self.periods_per_year,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "autocorrelation": self.autocorrelation,
            "expected_max_sharpe": self.expected_max_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "variance_trials": self.variance_trials,
            "min_trl": self.min_trl,
            "min_trl_years": self.min_trl_years,
            "has_adequate_sample": self.has_adequate_sample,
            "confidence_level": self.confidence_level,
        }


def deflated_sharpe_ratio(
    returns: ArrayLike | Sequence[ArrayLike],
    frequency: Frequency = "daily",
    benchmark_sharpe: float = 0.0,
    confidence_level: float = 0.95,
    periods_per_year: int | None = None,
    *,
    skewness: float | None = None,
    excess_kurtosis: float | None = None,
    autocorrelation: float | None = None,
) -> DSRResult:
    """Compute Deflated Sharpe Ratio (DSR) or Probabilistic Sharpe Ratio (PSR).

    This function computes the probability that the true Sharpe ratio exceeds
    a benchmark threshold, correcting for:

    - **Non-normality**: Skewness and excess kurtosis of returns
    - **Serial correlation**: First-order autocorrelation of returns
    - **Multiple testing**: Selection bias when choosing the best of K strategies

    **Single strategy (PSR)**: Pass a single returns array.
    **Multiple strategies (DSR)**: Pass a list of returns arrays.

    Parameters
    ----------
    returns : array-like or Sequence[array-like]
        Strategy returns at the specified frequency.
        - Single array: Computes PSR (no multiple testing adjustment)
        - Sequence of K arrays: Computes DSR for the best strategy
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency. Affects annualization for display.
    benchmark_sharpe : float, default 0.0
        Null hypothesis threshold (SR_0) at native frequency.
    confidence_level : float, default 0.95
        Confidence level for significance testing.
    periods_per_year : int, optional
        Trading periods per year. Defaults: daily=252, weekly=52, monthly=12.
    skewness : float, optional
        Override computed skewness.
    excess_kurtosis : float, optional
        Override computed excess kurtosis (Fisher convention, normal=0).
    autocorrelation : float, optional
        Override computed autocorrelation.

    Returns
    -------
    DSRResult
        Comprehensive results. Use `.interpret()` for human-readable summary.

    Examples
    --------
    Single strategy (PSR):

    >>> result = deflated_sharpe_ratio(daily_returns, frequency="daily")
    >>> print(f"Probability of skill: {result.probability:.1%}")

    Multiple strategies (DSR):

    >>> strategies = [strat1_returns, strat2_returns, strat3_returns]
    >>> result = deflated_sharpe_ratio(strategies, frequency="daily")
    >>> print(f"Probability after deflation: {result.probability:.1%}")

    References
    ----------
    Lopez de Prado et al. (2025). "How to Use the Sharpe Ratio."
    """
    # Resolve periods per year
    if periods_per_year is None:
        periods_per_year = DEFAULT_PERIODS_PER_YEAR[frequency]

    annualization_factor = np.sqrt(periods_per_year)

    # Detect multiple strategies
    is_multiple = (
        isinstance(returns, list | tuple)
        and len(returns) > 1
        and not isinstance(returns[0], int | float)
    )

    if is_multiple:
        # Multiple strategies - DSR
        returns_seq = list(returns)  # type: ignore[arg-type]
        n_trials = len(returns_seq)

        # Compute Sharpe ratio for each strategy
        sharpe_ratios = []
        for ret in returns_seq:
            ret_arr = np.asarray(ret).flatten()
            ret_arr = ret_arr[~np.isnan(ret_arr)]
            sr, _, _, _, _ = compute_return_statistics(ret_arr)
            sharpe_ratios.append(sr)

        # Best strategy
        best_idx = int(np.argmax(sharpe_ratios))
        best_returns = np.asarray(returns_seq[best_idx]).flatten()
        best_returns = best_returns[~np.isnan(best_returns)]

        observed_sharpe, comp_skew, comp_kurt, comp_rho, n_samples = compute_return_statistics(
            best_returns
        )

        # Cross-sectional variance
        variance_trials = float(np.var(sharpe_ratios, ddof=1)) if n_trials > 1 else 0.0

    else:
        # Single strategy - PSR
        n_trials = 1
        variance_trials = 0.0

        if isinstance(returns, list | tuple) and len(returns) == 1:
            ret_arr = np.asarray(returns[0]).flatten()
        else:
            ret_arr = np.asarray(returns).flatten()

        observed_sharpe, comp_skew, comp_kurt, comp_rho, n_samples = compute_return_statistics(
            ret_arr
        )

    # Use provided statistics or computed ones
    skew = skewness if skewness is not None else comp_skew
    if excess_kurtosis is not None:
        kurt = excess_kurtosis + 3.0  # Fisher -> Pearson
    else:
        kurt = comp_kurt
    rho = autocorrelation if autocorrelation is not None else comp_rho

    # Expected max Sharpe (multiple testing adjustment)
    expected_max = compute_expected_max_sharpe(n_trials, variance_trials)
    adjusted_threshold = benchmark_sharpe + expected_max

    # Variance of Sharpe estimator
    variance_sr = compute_sharpe_variance(
        sharpe=adjusted_threshold,
        n_samples=n_samples,
        skewness=skew,
        kurtosis=kurt,
        autocorrelation=rho,
        n_trials=n_trials,
    )
    std_sr = np.sqrt(variance_sr)

    # Z-score
    if std_sr > 0:
        z_score = (observed_sharpe - adjusted_threshold) / std_sr
    else:
        z_score = np.inf if observed_sharpe > adjusted_threshold else -np.inf

    # Probability and p-value
    probability = float(norm.cdf(z_score))
    p_value = float(1 - probability)
    is_significant = probability >= confidence_level

    # Annualized Sharpe
    sharpe_annualized = observed_sharpe * annualization_factor
    deflated = observed_sharpe - expected_max

    # MinTRL
    min_trl = _compute_min_trl_core(
        observed_sharpe=observed_sharpe,
        target_sharpe=benchmark_sharpe,
        confidence_level=confidence_level,
        skewness=skew,
        kurtosis=kurt,
        autocorrelation=rho,
    )
    min_trl_years = min_trl / periods_per_year
    has_adequate = n_samples >= min_trl

    return DSRResult(
        probability=probability,
        is_significant=is_significant,
        z_score=float(z_score),
        p_value=p_value,
        sharpe_ratio=float(observed_sharpe),
        sharpe_ratio_annualized=float(sharpe_annualized),
        benchmark_sharpe=benchmark_sharpe,
        n_samples=n_samples,
        n_trials=n_trials,
        frequency=frequency,
        periods_per_year=periods_per_year,
        skewness=float(skew),
        excess_kurtosis=float(kurt - 3.0),
        autocorrelation=float(rho),
        expected_max_sharpe=float(expected_max),
        deflated_sharpe=float(deflated),
        variance_trials=float(variance_trials),
        min_trl=min_trl,
        min_trl_years=float(min_trl_years),
        has_adequate_sample=has_adequate,
        confidence_level=confidence_level,
    )


def deflated_sharpe_ratio_from_statistics(
    observed_sharpe: float,
    n_samples: int,
    n_trials: int = 1,
    variance_trials: float = 0.0,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    autocorrelation: float = 0.0,
    confidence_level: float = 0.95,
    frequency: Frequency = "daily",
    periods_per_year: int | None = None,
) -> DSRResult:
    """Compute DSR/PSR from pre-computed statistics.

    Use this when you have already computed the required statistics.
    For most users, `deflated_sharpe_ratio()` with raw returns is recommended.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio at native frequency.
    n_samples : int
        Number of return observations (T).
    n_trials : int, default 1
        Number of strategies tested (K).
    variance_trials : float, default 0.0
        Cross-sectional variance of Sharpe ratios.
    benchmark_sharpe : float, default 0.0
        Null hypothesis threshold.
    skewness : float, default 0.0
        Return skewness.
    excess_kurtosis : float, default 0.0
        Return excess kurtosis (Fisher, normal=0).
    autocorrelation : float, default 0.0
        First-order autocorrelation.
    confidence_level : float, default 0.95
        Confidence level for testing.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    periods_per_year : int, optional
        Periods per year.

    Returns
    -------
    DSRResult
        Same as `deflated_sharpe_ratio()`.
    """
    # Validate inputs
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if n_trials > 1 and variance_trials <= 0:
        raise ValueError("variance_trials must be positive when n_trials > 1")
    if abs(autocorrelation) >= 1:
        raise ValueError("autocorrelation must be in (-1, 1)")

    kurtosis = excess_kurtosis + 3.0

    if periods_per_year is None:
        periods_per_year = DEFAULT_PERIODS_PER_YEAR[frequency]

    annualization_factor = np.sqrt(periods_per_year)

    # Expected max Sharpe
    expected_max = compute_expected_max_sharpe(n_trials, variance_trials)
    adjusted_threshold = benchmark_sharpe + expected_max

    # Variance
    variance_sr = compute_sharpe_variance(
        sharpe=adjusted_threshold,
        n_samples=n_samples,
        skewness=skewness,
        kurtosis=kurtosis,
        autocorrelation=autocorrelation,
        n_trials=n_trials,
    )
    std_sr = np.sqrt(variance_sr)

    # Z-score
    if std_sr > 0:
        z_score = (observed_sharpe - adjusted_threshold) / std_sr
    else:
        z_score = np.inf if observed_sharpe > adjusted_threshold else -np.inf

    probability = float(norm.cdf(z_score))
    p_value = float(1 - probability)
    is_significant = probability >= confidence_level

    sharpe_annualized = observed_sharpe * annualization_factor
    deflated = observed_sharpe - expected_max

    # MinTRL
    min_trl = _compute_min_trl_core(
        observed_sharpe=observed_sharpe,
        target_sharpe=benchmark_sharpe,
        confidence_level=confidence_level,
        skewness=skewness,
        kurtosis=kurtosis,
        autocorrelation=autocorrelation,
    )
    min_trl_years = min_trl / periods_per_year
    has_adequate = n_samples >= min_trl

    return DSRResult(
        probability=probability,
        is_significant=is_significant,
        z_score=float(z_score),
        p_value=p_value,
        sharpe_ratio=float(observed_sharpe),
        sharpe_ratio_annualized=float(sharpe_annualized),
        benchmark_sharpe=benchmark_sharpe,
        n_samples=n_samples,
        n_trials=n_trials,
        frequency=frequency,
        periods_per_year=periods_per_year,
        skewness=float(skewness),
        excess_kurtosis=float(excess_kurtosis),
        autocorrelation=float(autocorrelation),
        expected_max_sharpe=float(expected_max),
        deflated_sharpe=float(deflated),
        variance_trials=float(variance_trials),
        min_trl=min_trl,
        min_trl_years=float(min_trl_years),
        has_adequate_sample=has_adequate,
        confidence_level=confidence_level,
    )


# =============================================================================
# BACKWARD COMPATIBILITY RE-EXPORTS
# =============================================================================
# These were previously defined in dsr.py but are now in separate modules.
# Re-export for backward compatibility.

_VARIANCE_RESCALING_FACTORS = VARIANCE_RESCALING_FACTORS
_get_variance_rescaling_factor = get_variance_rescaling_factor
_compute_return_statistics = compute_return_statistics
_compute_sharpe_variance = compute_sharpe_variance
_compute_expected_max_sharpe = compute_expected_max_sharpe
_compute_min_trl = _compute_min_trl_core

__all__ = [
    # Result classes
    "DSRResult",
    # Main functions
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_statistics",
    # Re-exports from other modules (for backward compat)
    "MinTRLResult",
    "PBOResult",
    "compute_min_trl",
    "min_trl_fwer",
    "compute_pbo",
    "DEFAULT_PERIODS_PER_YEAR",
    # Type aliases
    "Frequency",
    # Private backward compat
    "_VARIANCE_RESCALING_FACTORS",
    "_get_variance_rescaling_factor",
    "_compute_return_statistics",
    "_compute_sharpe_variance",
    "_compute_expected_max_sharpe",
    "_compute_min_trl",
]
