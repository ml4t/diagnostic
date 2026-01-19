"""Normality tests for distribution analysis.

This module provides statistical tests for normality:
- Jarque-Bera test: Based on sample skewness and kurtosis, asymptotically valid
- Shapiro-Wilk test: More powerful for small samples (n < 2000), recommended

Test Comparison:
    - Jarque-Bera: Based on sample skewness and kurtosis, asymptotically valid
    - Shapiro-Wilk: More powerful for small samples (n < 2000), recommended

References:
    - Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality,
      homoscedasticity and serial independence of regression residuals.
      Economics Letters, 6(3), 255-259. DOI: 10.1016/0165-1765(80)90024-5
    - Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
      for normality (complete samples). Biometrika, 52(3-4), 591-611.
      DOI: 10.1093/biomet/52.3-4.591
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
class JarqueBeraResult:
    """Jarque-Bera normality test result.

    Tests for normality based on sample skewness and kurtosis. The test
    statistic is: JB = (n/6) * (S^2 + K^2/4), where S is skewness and K
    is excess kurtosis. Under H0 (normality), JB ~ χ²(2).

    Attributes:
        statistic: Jarque-Bera test statistic
        p_value: P-value for null hypothesis (data is normally distributed)
        skewness: Sample skewness used in test
        excess_kurtosis: Sample excess kurtosis used in test (Fisher: normal=0)
        is_normal: Whether data is consistent with normality (p >= alpha)
        n_obs: Number of observations
        alpha: Significance level used
    """

    statistic: float
    p_value: float
    skewness: float
    excess_kurtosis: float
    is_normal: bool
    n_obs: int
    alpha: float = 0.05

    def __repr__(self) -> str:
        """String representation."""
        return f"JarqueBeraResult(statistic={self.statistic:.4f}, p_value={self.p_value:.4f}, is_normal={self.is_normal})"

    def summary(self) -> str:
        """Human-readable summary of Jarque-Bera test.

        Returns:
            Formatted summary string
        """
        lines = [
            "Jarque-Bera Normality Test",
            "=" * 50,
            f"Test Statistic:    {self.statistic:.4f}",
            f"P-value:           {self.p_value:.4f}",
            f"Observations:      {self.n_obs}",
            f"Significance:      α={self.alpha}",
        ]

        lines.append("")
        lines.append("Moments:")
        lines.append(f"  Skewness:        {self.skewness:.4f}")
        lines.append(f"  Excess Kurtosis: {self.excess_kurtosis:.4f}")

        lines.append("")
        conclusion = (
            "Data is consistent with normality"
            if self.is_normal
            else "Data deviates from normality"
        )
        lines.append(f"Conclusion: {conclusion}")
        lines.append(
            f"  (Fail to reject H0 at {self.alpha * 100:.0f}% level)"
            if self.is_normal
            else f"  (Reject H0 at {self.alpha * 100:.0f}% level)"
        )

        lines.append("")
        lines.append("Test Methodology:")
        lines.append("  - JB = (n/6) * (S² + K²/4)")
        lines.append("  - H0: Data is normally distributed")
        lines.append("  - Under H0: JB ~ χ²(2)")
        lines.append("  - Asymptotically valid (requires large n)")

        if not self.is_normal:
            lines.append("")
            lines.append("Implications:")
            lines.append("  - Normal distribution assumption violated")
            lines.append("  - Consider robust statistical methods")
            lines.append("  - Account for non-normality in risk models")

        return "\n".join(lines)


@dataclass
class ShapiroWilkResult:
    """Shapiro-Wilk normality test result.

    Tests for normality using order statistics. More powerful than Jarque-Bera
    for small samples (n < 2000). The test statistic W ranges from 0 to 1,
    with values close to 1 indicating normality.

    Attributes:
        statistic: Shapiro-Wilk test statistic (W)
        p_value: P-value for null hypothesis (data is normally distributed)
        is_normal: Whether data is consistent with normality (p >= alpha)
        n_obs: Number of observations
        alpha: Significance level used
    """

    statistic: float
    p_value: float
    is_normal: bool
    n_obs: int
    alpha: float = 0.05

    def __repr__(self) -> str:
        """String representation."""
        return f"ShapiroWilkResult(statistic={self.statistic:.4f}, p_value={self.p_value:.4f}, is_normal={self.is_normal})"

    def summary(self) -> str:
        """Human-readable summary of Shapiro-Wilk test.

        Returns:
            Formatted summary string
        """
        lines = [
            "Shapiro-Wilk Normality Test",
            "=" * 50,
            f"Test Statistic (W): {self.statistic:.4f}",
            f"P-value:            {self.p_value:.4f}",
            f"Observations:       {self.n_obs}",
            f"Significance:       α={self.alpha}",
        ]

        lines.append("")
        conclusion = (
            "Data is consistent with normality"
            if self.is_normal
            else "Data deviates from normality"
        )
        lines.append(f"Conclusion: {conclusion}")
        lines.append(
            f"  (Fail to reject H0 at {self.alpha * 100:.0f}% level)"
            if self.is_normal
            else f"  (Reject H0 at {self.alpha * 100:.0f}% level)"
        )

        lines.append("")
        lines.append("Test Methodology:")
        lines.append("  - Based on correlation between data and normal scores")
        lines.append("  - W statistic ranges from 0 (non-normal) to 1 (normal)")
        lines.append("  - H0: Data is normally distributed")
        lines.append("  - More powerful than Jarque-Bera for small samples")
        lines.append("  - Recommended for n < 2000")

        if not self.is_normal:
            lines.append("")
            lines.append("Implications:")
            lines.append("  - Normal distribution assumption violated")
            lines.append("  - Consider non-parametric methods")
            lines.append("  - Use robust estimators for inference")

        return "\n".join(lines)


def jarque_bera_test(
    data: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> JarqueBeraResult:
    """Jarque-Bera test for normality.

    Tests whether sample skewness and kurtosis match a normal distribution.
    The test statistic is:

        JB = (n/6) * (S^2 + K^2/4)

    where n is sample size, S is skewness, K is excess kurtosis.
    Under H0 (normality), JB ~ χ²(2).

    The null hypothesis is that the data is normally distributed. Low p-values
    (< alpha) indicate rejection of normality.

    Args:
        data: Time series data (1D array or Series)
        alpha: Significance level (default 0.05)

    Returns:
        JarqueBeraResult with test statistics and conclusion

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Example:
        >>> import numpy as np
        >>> # Normal data (should pass)
        >>> normal = np.random.normal(0, 1, 1000)
        >>> result = jarque_bera_test(normal)
        >>> print(f"p-value: {result.p_value:.4f}, normal: {result.is_normal}")
        >>>
        >>> # Lognormal data (should fail)
        >>> lognormal = np.random.lognormal(0, 0.5, 1000)
        >>> result = jarque_bera_test(lognormal)
        >>> print(f"p-value: {result.p_value:.4f}, normal: {result.is_normal}")

    Notes:
        - Test is asymptotically valid (requires large n)
        - More powerful for large samples (n > 2000)
        - For small samples, use Shapiro-Wilk test instead
        - Uses scipy.stats.jarque_bera
    """
    # Input validation (same as compute_moments)
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "jarque_bera_test"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "jarque_bera_test", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "jarque_bera_test", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "jarque_bera_test", "length": 0}
        )

    # Check for missing/infinite values
    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Data contains {n_invalid} NaN or infinite values",
            context={"function": "jarque_bera_test", "n_invalid": n_invalid, "length": len(arr)},
        )

    # Check minimum length
    min_length = 20
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for Jarque-Bera test (need at least {min_length} observations)",
            context={
                "function": "jarque_bera_test",
                "length": len(arr),
                "min_length": min_length,
            },
        )

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "jarque_bera_test",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    logger.info("Running Jarque-Bera test", n_obs=len(arr), alpha=alpha)

    try:
        # Run Jarque-Bera test using scipy
        # Returns (statistic, p_value)
        jb_stat, p_value = stats.jarque_bera(arr)

        # Compute moments for reporting
        skewness = float(stats.skew(arr, bias=False))
        excess_kurtosis = float(stats.kurtosis(arr, bias=False))

        # Determine normality
        is_normal = p_value >= alpha

        logger.info(
            "Jarque-Bera test completed",
            statistic=jb_stat,
            p_value=p_value,
            is_normal=is_normal,
        )

        return JarqueBeraResult(
            statistic=float(jb_stat),
            p_value=float(p_value),
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            is_normal=is_normal,
            n_obs=len(arr),
            alpha=alpha,
        )

    except Exception as e:
        logger.error("Jarque-Bera test failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"Jarque-Bera test computation failed: {e}",
            context={"function": "jarque_bera_test", "n_obs": len(arr), "alpha": alpha},
            cause=e,
        )


def shapiro_wilk_test(
    data: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> ShapiroWilkResult:
    """Shapiro-Wilk test for normality.

    Tests for normality using order statistics. More powerful than Jarque-Bera
    for small samples (n < 2000). The test statistic W ranges from 0 to 1,
    with values close to 1 indicating normality.

    The null hypothesis is that the data is normally distributed. Low p-values
    (< alpha) indicate rejection of normality.

    Args:
        data: Time series data (1D array or Series)
        alpha: Significance level (default 0.05)

    Returns:
        ShapiroWilkResult with test statistics and conclusion

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Example:
        >>> import numpy as np
        >>> # Normal data (should pass)
        >>> normal = np.random.normal(0, 1, 500)
        >>> result = shapiro_wilk_test(normal)
        >>> print(f"W: {result.statistic:.4f}, p-value: {result.p_value:.4f}")
        >>>
        >>> # Lognormal data (should fail)
        >>> lognormal = np.random.lognormal(0, 0.5, 500)
        >>> result = shapiro_wilk_test(lognormal)
        >>> print(f"Normal: {result.is_normal}")

    Notes:
        - More powerful than Jarque-Bera for small samples (n < 2000)
        - Recommended over Jarque-Bera when n < 2000
        - W statistic close to 1 indicates normality
        - Uses scipy.stats.shapiro
        - Maximum sample size: 5000 (scipy limitation)
    """
    # Input validation (same as jarque_bera_test)
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "shapiro_wilk_test"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "shapiro_wilk_test", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "shapiro_wilk_test", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "shapiro_wilk_test", "length": 0}
        )

    # Check for missing/infinite values
    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Data contains {n_invalid} NaN or infinite values",
            context={"function": "shapiro_wilk_test", "n_invalid": n_invalid, "length": len(arr)},
        )

    # Check minimum length (Shapiro-Wilk needs at least 3 observations)
    min_length = 3
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for Shapiro-Wilk test (need at least {min_length} observations)",
            context={
                "function": "shapiro_wilk_test",
                "length": len(arr),
                "min_length": min_length,
            },
        )

    # Check maximum length (scipy limitation)
    max_length = 5000
    if len(arr) > max_length:
        logger.warning(
            f"Data has {len(arr)} observations, using first {max_length} (scipy.stats.shapiro limitation)"
        )
        arr = arr[:max_length]

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "shapiro_wilk_test",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    logger.info("Running Shapiro-Wilk test", n_obs=len(arr), alpha=alpha)

    try:
        # Run Shapiro-Wilk test using scipy
        # Returns (statistic, p_value)
        w_stat, p_value = stats.shapiro(arr)

        # Determine normality
        is_normal = p_value >= alpha

        logger.info(
            "Shapiro-Wilk test completed",
            statistic=w_stat,
            p_value=p_value,
            is_normal=is_normal,
        )

        return ShapiroWilkResult(
            statistic=float(w_stat),
            p_value=float(p_value),
            is_normal=is_normal,
            n_obs=len(arr),
            alpha=alpha,
        )

    except Exception as e:
        logger.error("Shapiro-Wilk test failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"Shapiro-Wilk test computation failed: {e}",
            context={"function": "shapiro_wilk_test", "n_obs": len(arr), "alpha": alpha},
            cause=e,
        )
