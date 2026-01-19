"""ARCH Lagrange Multiplier test for conditional heteroscedasticity.

The ARCH-LM test (Engle, 1982) detects autoregressive conditional
heteroscedasticity (volatility clustering) in time series data.

References:
    Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with
    Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4),
    987-1007. DOI: 10.2307/1912773
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# het_arch is in statsmodels (required dependency)
from statsmodels.stats.diagnostic import het_arch

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


class ARCHLMResult:
    """Results from ARCH Lagrange Multiplier test.

    The ARCH-LM test detects autoregressive conditional heteroscedasticity
    (volatility clustering) in time series data. The null hypothesis is
    that there are no ARCH effects (constant variance).

    Attributes:
        test_statistic: LM test statistic (n * R² from auxiliary regression)
        p_value: P-value for null hypothesis (no ARCH effects)
        lags: Number of lags tested in auxiliary regression
        n_obs: Number of observations used in test
        alpha: Significance level used for the test
        has_arch_effects: Whether ARCH effects detected (p < alpha)
    """

    def __init__(
        self,
        test_statistic: float,
        p_value: float,
        lags: int,
        n_obs: int,
        alpha: float = 0.05,
    ):
        """Initialize ARCH-LM result.

        Args:
            test_statistic: LM test statistic
            p_value: P-value for no ARCH effects hypothesis
            lags: Number of lags used in test
            n_obs: Number of observations
            alpha: Significance level for the test (default 0.05)
        """
        self.test_statistic = test_statistic
        self.p_value = p_value
        self.lags = lags
        self.n_obs = n_obs
        self.alpha = alpha

        # Determine ARCH effects at specified significance level
        # Low p-value (< alpha) means reject H0 => ARCH effects present
        self.has_arch_effects = p_value < alpha

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ARCHLMResult(statistic={self.test_statistic:.4f}, "
            f"p_value={self.p_value:.4f}, "
            f"has_arch_effects={self.has_arch_effects})"
        )

    def summary(self) -> str:
        """Human-readable summary of ARCH-LM test results.

        Returns:
            Formatted summary string
        """
        lines = [
            "ARCH Lagrange Multiplier Test Results",
            "=" * 50,
            f"Test Statistic:    {self.test_statistic:.4f}",
            f"P-value:           {self.p_value:.4f}",
            f"Lags Used:         {self.lags}",
            f"Observations:      {self.n_obs}",
        ]

        lines.append("")
        lines.append(
            f"Conclusion: {'ARCH effects detected' if self.has_arch_effects else 'No ARCH effects'}"
        )
        alpha_pct = self.alpha * 100
        lines.append(
            f"  (Reject H0 at {alpha_pct:.0f}% level)"
            if self.has_arch_effects
            else f"  (Fail to reject H0 at {alpha_pct:.0f}% level)"
        )

        lines.append("")
        lines.append("Interpretation:")
        if self.has_arch_effects:
            lines.append("  - Volatility clustering present (time-varying variance)")
            lines.append("  - Consider GARCH/EGARCH models for volatility forecasting")
            lines.append("  - Standard errors may be unreliable without correction")
            lines.append("  - Risk models should account for conditional heteroscedasticity")
        else:
            lines.append("  - No evidence of volatility clustering")
            lines.append("  - Constant variance assumption is reasonable")
            lines.append("  - Classical methods with homoscedasticity are appropriate")

        lines.append("")
        lines.append("Test Methodology:")
        lines.append("  - Auxiliary regression: ε²_t = α₀ + Σ(α_i * ε²_{t-i})")
        lines.append(f"  - LM statistic = n * R² ~ χ²({self.lags})")
        lines.append("  - H0: No ARCH effects (α₁ = α₂ = ... = α_lags = 0)")

        return "\n".join(lines)


def arch_lm_test(
    data: pd.Series | np.ndarray,
    lags: int = 12,
    demean: bool = True,
    alpha: float = 0.05,
) -> ARCHLMResult:
    """Perform ARCH Lagrange Multiplier test for conditional heteroscedasticity.

    The ARCH-LM test (Engle, 1982) tests for autoregressive conditional
    heteroscedasticity (volatility clustering) in time series data. The test
    is based on the principle that if ARCH effects are present, squared
    residuals will be autocorrelated.

    Test Methodology:
        1. Compute residuals: ε_t (de-meaned if demean=True)
        2. Square residuals: ε²_t
        3. Regress ε²_t on ε²_{t-1}, ..., ε²_{t-lags}
        4. LM statistic = n * R² from auxiliary regression
        5. Under H0 (no ARCH): LM ~ χ²(lags)

    Args:
        data: Time series data to test (1D array or Series)
        lags: Number of lags to test (default 12, ~1 year of monthly data)
        demean: Whether to subtract mean before computing squared residuals.
                True is common for returns which are approximately zero-mean.
        alpha: Significance level for the test (default 0.05)

    Returns:
        ARCHLMResult with test statistics and conclusion

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Notes:
        - De-meaning (demean=True) is standard for return series
        - Lag selection: 12 for monthly, ~250 for daily returns
        - Test is asymptotically valid (needs large sample)
        - Presence of ARCH effects suggests GARCH models may be appropriate
        - Uses statsmodels.stats.diagnostic.het_arch (core dependency)

    References:
        Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with
        Estimates of the Variance of United Kingdom Inflation. Econometrica,
        50(4), 987-1007. DOI: 10.2307/1912773
    """
    # Input validation
    logger.debug(f"Running ARCH-LM test with lags={lags}, demean={demean}")

    # Convert to numpy array if needed
    arr = data.to_numpy() if isinstance(data, pd.Series) else np.asarray(data)

    # Validate input
    if arr.size == 0:
        raise ValidationError(
            "Cannot perform ARCH-LM test on empty data",
            context={"data_size": 0},
        )

    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got shape {arr.shape}",
            context={"data_shape": arr.shape},
        )

    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Data contains {n_invalid} NaN or infinite values",
            context={"n_invalid": n_invalid, "data_size": arr.size},
        )

    # Validate lags parameter FIRST (before computing min_obs)
    if lags < 1:
        raise ValidationError(
            f"Number of lags must be positive, got {lags}",
            context={"lags": lags},
        )

    # Check minimum sample size (now safe since lags >= 1)
    min_obs = lags + 10  # Need at least lags + some buffer
    if arr.size < min_obs:
        raise ValidationError(
            f"Insufficient data for ARCH-LM test with {lags} lags. "
            f"Need at least {min_obs} observations, got {arr.size}",
            context={"n_obs": arr.size, "lags": lags, "min_required": min_obs},
        )

    if lags >= arr.size:
        raise ValidationError(
            f"Number of lags ({lags}) must be less than data size ({arr.size})",
            context={"lags": lags, "data_size": arr.size},
        )

    try:
        # De-mean the data if requested (standard for returns)
        if demean:
            residuals = arr - np.mean(arr)
            logger.debug(f"De-meaned data: mean={np.mean(arr):.6f}")
        else:
            residuals = arr.copy()

        # Run ARCH-LM test using statsmodels
        # het_arch returns (statistic, p-value, f-stat, f-pvalue)
        # We use the LM test statistic (first two values)
        result_tuple = het_arch(residuals, nlags=lags)
        lm_stat = result_tuple[0]
        p_value = result_tuple[1]

        logger.info(
            f"ARCH-LM test complete: statistic={lm_stat:.4f}, p-value={p_value:.4f}",
            lags=lags,
            n_obs=arr.size,
        )

        return ARCHLMResult(
            test_statistic=float(lm_stat),
            p_value=float(p_value),
            lags=lags,
            n_obs=arr.size,
            alpha=alpha,
        )

    except Exception as e:
        # Handle computation errors
        logger.error(f"ARCH-LM test failed: {e}", lags=lags, n_obs=arr.size)
        raise ComputationError(  # noqa: B904
            f"ARCH-LM test computation failed: {e}",
            context={
                "n_obs": arr.size,
                "lags": lags,
                "demean": demean,
            },
            cause=e,
        )
