"""Distribution moments analysis with significance tests.

This module provides moment computation (skewness and kurtosis) with statistical
significance testing for financial returns analysis.

Moment Interpretation:
    - Skewness = 0: Symmetric distribution (normal)
    - Skewness > 0: Right-skewed (long right tail)
    - Skewness < 0: Left-skewed (long left tail, common for equity returns)
    - Excess Kurtosis = 0: Normal tail thickness
    - Excess Kurtosis > 0: Fat tails (leptokurtic, more extreme events)
    - Excess Kurtosis < 0: Thin tails (platykurtic, fewer extreme events)

References:
    - D'Agostino, R. B., & Pearson, E. S. (1973). Tests for departure from
      normality. Biometrika, 60(3), 613-622. (Standard errors for moments)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MomentsResult:
    """Distribution moments (skewness and excess kurtosis) with significance tests.

    Attributes:
        mean: Sample mean
        std: Sample standard deviation
        skewness: Sample skewness (third standardized moment)
        skewness_se: Standard error of skewness
        excess_kurtosis: Sample excess kurtosis (Fisher: normal=0)
        excess_kurtosis_se: Standard error of excess kurtosis
        skewness_significant: Whether skewness is significantly different from 0
        excess_kurtosis_significant: Whether excess kurtosis is significantly different from 0
        n_obs: Number of observations
        alpha: Significance level used for tests
    """

    mean: float
    std: float
    skewness: float
    skewness_se: float
    excess_kurtosis: float
    excess_kurtosis_se: float
    skewness_significant: bool
    excess_kurtosis_significant: bool
    n_obs: int
    alpha: float = 0.05

    def __repr__(self) -> str:
        """String representation."""
        return f"MomentsResult(skewness={self.skewness:.4f}, excess_kurtosis={self.excess_kurtosis:.4f}, n={self.n_obs})"

    def summary(self) -> str:
        """Human-readable summary of moment analysis.

        Returns:
            Formatted summary string
        """
        lines = [
            "Distribution Moments Analysis",
            "=" * 50,
            f"Observations:      {self.n_obs}",
            f"Mean:              {self.mean:.6f}",
            f"Std Dev:           {self.std:.6f}",
        ]

        lines.append("")
        lines.append("Skewness:")
        lines.append(f"  Value:           {self.skewness:.4f}")
        lines.append(f"  Std Error:       {self.skewness_se:.4f}")
        lines.append(f"  Z-score:         {self.skewness / self.skewness_se:.4f}")
        lines.append(
            f"  Significant:     {'Yes' if self.skewness_significant else 'No'} (α={self.alpha})"
        )

        if abs(self.skewness) < 0.1:
            interpretation = "approximately symmetric"
        elif self.skewness > 0:
            interpretation = "right-skewed (positive, long right tail)"
        else:
            interpretation = "left-skewed (negative, long left tail)"
        lines.append(f"  Interpretation:  {interpretation}")

        lines.append("")
        lines.append("Excess Kurtosis:")
        lines.append(f"  Value:           {self.excess_kurtosis:.4f}")
        lines.append(f"  Std Error:       {self.excess_kurtosis_se:.4f}")
        lines.append(f"  Z-score:         {self.excess_kurtosis / self.excess_kurtosis_se:.4f}")
        lines.append(
            f"  Significant:     {'Yes' if self.excess_kurtosis_significant else 'No'} (α={self.alpha})"
        )

        if abs(self.excess_kurtosis) < 0.1:
            interpretation = "approximately normal (mesokurtic)"
        elif self.excess_kurtosis > 0:
            interpretation = "fat tails (leptokurtic, more extreme events)"
        else:
            interpretation = "thin tails (platykurtic, fewer extreme events)"
        lines.append(f"  Interpretation:  {interpretation}")

        lines.append("")
        lines.append("Implications:")
        if self.skewness_significant or self.excess_kurtosis_significant:
            lines.append("  - Distribution deviates significantly from normality")
            if self.skewness_significant:
                lines.append("  - Skewness affects mean-variance optimization")
                lines.append("  - Asymmetric risk profiles require adjusted risk measures")
            if self.excess_kurtosis_significant:
                lines.append("  - Fat tails increase probability of extreme events")
                lines.append("  - VaR/CVaR estimates may underestimate tail risk")
        else:
            lines.append("  - Moments consistent with normal distribution")
            lines.append("  - Classical statistical methods appropriate")

        return "\n".join(lines)


def compute_moments(
    data: pd.Series | np.ndarray,
    test_significance: bool = True,
    alpha: float = 0.05,
) -> MomentsResult:
    """Compute distribution moments with significance tests.

    Calculates sample skewness and excess kurtosis with standard errors.
    Optionally tests whether moments are significantly different from zero
    (normal distribution values) using z-tests.

    Standard Errors:
        - SE(skewness) = sqrt(6/n)
        - SE(kurtosis) = sqrt(24/n)

    Significance Test:
        - H0: moment = 0 (consistent with normal distribution)
        - Reject if |moment| > 2 * SE (approximately α=0.05, two-tailed)

    Args:
        data: Time series data (1D array or Series)
        test_significance: Whether to test significance (default True)
        alpha: Significance level for tests (default 0.05)

    Returns:
        MomentsResult with moments and significance tests

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If computation fails

    Example:
        >>> import numpy as np
        >>> # Normal data (skewness ≈ 0, excess_kurtosis ≈ 0)
        >>> normal = np.random.normal(0, 1, 1000)
        >>> result = compute_moments(normal)
        >>> print(result.summary())
        >>>
        >>> # Lognormal data (skewed, fat-tailed)
        >>> lognormal = np.random.lognormal(0, 0.5, 1000)
        >>> result = compute_moments(lognormal)
        >>> print(f"Skewness significant: {result.skewness_significant}")
        >>> print(f"Excess kurtosis significant: {result.excess_kurtosis_significant}")

    Notes:
        - Skewness = 0 for symmetric distributions (normal)
        - Excess kurtosis = 0 for normal distribution (Fisher convention)
        - Financial returns typically show negative skew, positive excess kurtosis
        - Large samples (n > 1000) recommended for reliable inference
    """
    # Input validation
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "compute_moments"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
        logger.debug("Converted pandas Series to numpy array", shape=arr.shape)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "compute_moments", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "compute_moments", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "compute_moments", "length": 0}
        )

    # Check for missing values
    if np.any(np.isnan(arr)):
        n_missing = np.sum(np.isnan(arr))
        raise ValidationError(
            f"Data contains {n_missing} missing values (NaN)",
            context={"function": "compute_moments", "n_missing": n_missing, "length": len(arr)},
        )

    # Check for infinite values
    if np.any(np.isinf(arr)):
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(
            f"Data contains {n_inf} infinite values",
            context={"function": "compute_moments", "n_inf": n_inf, "length": len(arr)},
        )

    # Check minimum length
    min_length = 20  # Need reasonable sample size for moments
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for moment computation (need at least {min_length} observations)",
            context={"function": "compute_moments", "length": len(arr), "min_length": min_length},
        )

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "compute_moments",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    logger.info("Computing distribution moments", n_obs=len(arr))

    try:
        # Compute basic statistics
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))  # Sample std (n-1 denominator)

        # Compute skewness and excess kurtosis using scipy
        skewness = float(stats.skew(arr, bias=False))  # Sample skewness (bias=False)
        excess_kurtosis = float(
            stats.kurtosis(arr, bias=False)
        )  # Excess kurtosis (Fisher: normal=0)

        # Compute standard errors
        n = len(arr)
        skewness_se = float(np.sqrt(6 / n))
        excess_kurtosis_se = float(np.sqrt(24 / n))

        # Test significance if requested
        if test_significance:
            # Significance test: |moment| > critical_value * SE
            # For α=0.05 (two-tailed), critical value ≈ 1.96 ≈ 2
            critical_value = stats.norm.ppf(1 - alpha / 2)
            skewness_significant = abs(skewness) > critical_value * skewness_se
            excess_kurtosis_significant = abs(excess_kurtosis) > critical_value * excess_kurtosis_se
        else:
            skewness_significant = False
            excess_kurtosis_significant = False

        logger.info(
            "Moments computed",
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            skewness_sig=skewness_significant,
            excess_kurtosis_sig=excess_kurtosis_significant,
        )

        return MomentsResult(
            mean=mean,
            std=std,
            skewness=skewness,
            skewness_se=skewness_se,
            excess_kurtosis=excess_kurtosis,
            excess_kurtosis_se=excess_kurtosis_se,
            skewness_significant=skewness_significant,
            excess_kurtosis_significant=excess_kurtosis_significant,
            n_obs=n,
            alpha=alpha,
        )

    except Exception as e:
        logger.error("Moment computation failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"Moment computation failed: {e}",
            context={"function": "compute_moments", "n_obs": len(arr)},
            cause=e,
        )
