"""Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

IMPORTANT: KPSS tests the null hypothesis of stationarity, which is the
OPPOSITE of the ADF test. Rejecting H0 means the series is NON-stationary.

KPSS is typically used in conjunction with ADF to provide more robust
stationarity assessment:
- Stationary: ADF rejects + KPSS fails to reject
- Non-stationary: ADF fails to reject + KPSS rejects
- Quasi-stationary: Both reject or both fail (inconclusive)

References:
    - Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
      Testing the null hypothesis of stationarity against the alternative
      of a unit root. Journal of Econometrics, 54(1-3), 159-178.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


class KPSSResult:
    """Results from Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    IMPORTANT: KPSS tests the null hypothesis of stationarity, which is the
    OPPOSITE of the ADF test. Rejecting H0 means the series is NON-stationary.

    Attributes:
        test_statistic: KPSS test statistic
        p_value: Interpolated p-value for null hypothesis (stationarity)
        critical_values: Critical values at 10%, 5%, 2.5%, 1% significance levels
        lags_used: Number of lags used in Newey-West standard errors
        n_obs: Number of observations used
        is_stationary: Whether series is stationary (fails to reject H0 at 5%)
        regression: Type of regression ('c' for level, 'ct' for trend)
    """

    def __init__(
        self,
        test_statistic: float,
        p_value: float,
        critical_values: dict[str, float],
        lags_used: int,
        n_obs: int,
        regression: str,
    ):
        """Initialize KPSS result.

        Args:
            test_statistic: KPSS test statistic
            p_value: P-value for stationarity hypothesis
            critical_values: Critical values dict with keys '10%', '5%', '2.5%', '1%'
            lags_used: Number of lags used for Newey-West
            n_obs: Number of observations
            regression: Regression type ('c' or 'ct')
        """
        self.test_statistic = test_statistic
        self.p_value = p_value
        self.critical_values = critical_values
        self.lags_used = lags_used
        self.n_obs = n_obs
        self.regression = regression

        # CRITICAL: KPSS has opposite interpretation from ADF
        # H0 = stationary, so we're stationary if we FAIL to reject (p >= 0.05)
        self.is_stationary = p_value >= 0.05

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KPSSResult(statistic={self.test_statistic:.4f}, "
            f"p_value={self.p_value:.4f}, "
            f"stationary={self.is_stationary})"
        )

    def summary(self) -> str:
        """Human-readable summary of KPSS test results.

        Returns:
            Formatted summary string
        """
        lines = [
            "Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results",
            "=" * 50,
            f"Test Statistic:    {self.test_statistic:.4f}",
            f"P-value:           {self.p_value:.4f}",
            f"Lags Used:         {self.lags_used}",
            f"Observations:      {self.n_obs}",
            f"Regression Type:   {'Level' if self.regression == 'c' else 'Trend'}",
        ]

        lines.append("")
        lines.append("Critical Values:")
        for level, value in sorted(self.critical_values.items()):
            lines.append(f"  {level:>4s}: {value:>8.4f}")

        lines.append("")
        lines.append(f"Conclusion: {'Stationary' if self.is_stationary else 'Non-stationary'}")
        lines.append(
            "  (Fail to reject H0 at 5% level)"
            if self.is_stationary
            else f"  (Reject H0 at 5% level: {not self.is_stationary})"
        )
        lines.append("")
        lines.append("IMPORTANT: KPSS tests H0 = stationary (opposite of ADF)")
        lines.append("  - High p-value (>0.05) => stationary")
        lines.append("  - Low p-value (<0.05) => non-stationary")

        return "\n".join(lines)


def kpss_test(
    data: pd.Series | np.ndarray,
    regression: Literal["c", "ct"] = "c",
    nlags: int | Literal["auto", "legacy"] | None = "auto",
) -> KPSSResult:
    """Perform Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    The KPSS test tests the null hypothesis that the time series is stationary.
    This is the OPPOSITE of the ADF test. If the null is rejected (p < alpha),
    the series is considered NON-stationary.

    KPSS is typically used in conjunction with ADF to provide more robust
    stationarity assessment:
    - Stationary: ADF rejects + KPSS fails to reject
    - Non-stationary: ADF fails to reject + KPSS rejects
    - Quasi-stationary: Both reject or both fail (inconclusive)

    Regression types:
        - 'c': Level stationarity (constant mean, default)
        - 'ct': Trend stationarity (stationary around a trend)

    Lag selection for Newey-West standard errors:
        - 'auto': Uses int(12 * (nobs/100)^{1/4}) (default, recommended)
        - 'legacy': Uses int(4 * (nobs/100)^{1/4})
        - int: Manual specification of number of lags

    Args:
        data: Time series data to test (1D array or Series)
        regression: Type of stationarity to test ('c' for level, 'ct' for trend)
        nlags: Number of lags for Newey-West standard errors

    Returns:
        KPSSResult with test statistics and conclusion

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Example:
        >>> import numpy as np
        >>> # Test white noise (stationary)
        >>> wn = np.random.randn(1000)
        >>> result = kpss_test(wn)
        >>> print(result.summary())
        >>>
        >>> # Test random walk (non-stationary)
        >>> rw = np.cumsum(np.random.randn(1000))
        >>> result = kpss_test(rw)
        >>> print(f"Stationary: {result.is_stationary}")
        >>>
        >>> # Test with trend stationarity
        >>> result = kpss_test(rw, regression='ct')
        >>> print(f"Trend stationary: {result.is_stationary}")
        >>>
        >>> # Use with ADF for complementary testing
        >>> from ml4t.diagnostic.evaluation.stationarity import adf_test
        >>> adf_result = adf_test(wn)
        >>> kpss_result = kpss_test(wn)
        >>> if adf_result.is_stationary and kpss_result.is_stationary:
        ...     print("Strong evidence for stationarity")

    Notes:
        - For financial returns, 'c' (level) is typically appropriate
        - For price series with trend, 'ct' may be better
        - KPSS is more powerful against I(1) alternatives than ADF
        - Use both ADF and KPSS for robust stationarity assessment
        - White noise should pass both tests (ADF rejects, KPSS fails to reject)
    """
    # Input validation (same as ADF)
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "kpss_test"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
        logger.debug("Converted pandas Series to numpy array", shape=arr.shape)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "kpss_test", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "kpss_test", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "kpss_test", "length": 0}
        )

    # Check for missing values
    if np.any(np.isnan(arr)):
        n_missing = np.sum(np.isnan(arr))
        raise ValidationError(
            f"Data contains {n_missing} missing values (NaN)",
            context={"function": "kpss_test", "n_missing": n_missing, "length": len(arr)},
        )

    # Check for infinite values
    if np.any(np.isinf(arr)):
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(
            f"Data contains {n_inf} infinite values",
            context={"function": "kpss_test", "n_inf": n_inf, "length": len(arr)},
        )

    # Check minimum length
    min_length = 10
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for KPSS test (need at least {min_length} observations)",
            context={
                "function": "kpss_test",
                "length": len(arr),
                "min_length": min_length,
            },
        )

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "kpss_test",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    # Log test parameters
    logger.info(
        "Running KPSS test",
        n_obs=len(arr),
        regression=regression,
        nlags=nlags,
    )

    # Run KPSS test
    try:
        # Use "auto" if nlags is None (statsmodels doesn't accept None)
        nlags_param: int | Literal["auto", "legacy"] = nlags if nlags is not None else "auto"
        result = kpss(arr, regression=regression, nlags=nlags_param)

        # Unpack result
        # kpss returns: (kpss_stat, pvalue, lags, critical_values)
        kpss_stat = result[0]
        pvalue = result[1]
        usedlag = result[2]
        critical_vals = result[3]

        logger.info(
            "KPSS test completed",
            statistic=kpss_stat,
            p_value=pvalue,
            lags_used=usedlag,
            n_obs=len(arr),
            stationary=pvalue >= 0.05,  # Note: opposite of ADF
        )

        # Create result object
        return KPSSResult(
            test_statistic=float(kpss_stat),
            p_value=float(pvalue),
            critical_values=dict(critical_vals),
            lags_used=int(usedlag),
            n_obs=len(arr),
            regression=regression,
        )

    except Exception as e:
        logger.error("KPSS test failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"KPSS test computation failed: {e}",
            context={
                "function": "kpss_test",
                "n_obs": len(arr),
                "regression": regression,
                "nlags": nlags,
            },
            cause=e,
        )
