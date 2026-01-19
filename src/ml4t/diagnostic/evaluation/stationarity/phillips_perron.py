"""Phillips-Perron (PP) unit root test for stationarity.

The PP test is a non-parametric alternative to the ADF test that corrects
for serial correlation and heteroscedasticity using Newey-West estimator.

Like ADF, PP tests the null hypothesis that a unit root exists (non-stationary).
Rejecting H0 means the series is stationary.

Key Differences from ADF:
    - PP uses non-parametric Newey-West correction for serial correlation
    - PP estimates regression with only 1 lag vs ADF's multiple lags
    - PP more robust to general forms of heteroscedasticity
    - Both tests have same null hypothesis: unit root exists

References:
    - Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time
      series regression. Biometrika, 75(2), 335-346.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)

# Lazy check for optional arch package (PP test)
# Import is deferred to pp_test() to avoid slow module-level import (~200ms)
HAS_ARCH: bool | None = None  # Will be set on first pp_test() call


def _check_arch_available() -> bool:
    """Check if arch package is available (lazy check)."""
    global HAS_ARCH
    if HAS_ARCH is None:
        try:
            from arch.unitroot import PhillipsPerron  # noqa: F401

            HAS_ARCH = True
        except ImportError:
            HAS_ARCH = False
            logger.debug(
                "arch package not available - pp_test() will not work. "
                "Install with: pip install arch or pip install ml4t-diagnostic[advanced]"
            )
    return HAS_ARCH


class PPResult:
    """Results from Phillips-Perron (PP) unit root test.

    The PP test is a non-parametric alternative to the ADF test that corrects
    for serial correlation and heteroscedasticity using Newey-West estimator.

    Like ADF, PP tests the null hypothesis that a unit root exists (non-stationary).
    Rejecting H0 means the series is stationary.

    Attributes:
        test_statistic: PP test statistic
        p_value: MacKinnon p-value for null hypothesis (unit root exists)
        critical_values: Critical values at 1%, 5%, 10% significance levels
        lags_used: Number of lags used in Newey-West estimator
        n_obs: Number of observations used in test
        is_stationary: Whether series is stationary (rejects unit root at 5%)
        regression: Type of regression ('c', 'ct', 'n')
        test_type: Type of test ('tau' or 'rho')
    """

    def __init__(
        self,
        test_statistic: float,
        p_value: float,
        critical_values: dict[str, float],
        lags_used: int,
        n_obs: int,
        regression: str,
        test_type: str,
    ):
        """Initialize PP result.

        Args:
            test_statistic: PP test statistic
            p_value: P-value for unit root hypothesis
            critical_values: Critical values dict with keys '1%', '5%', '10%'
            lags_used: Number of lags used in Newey-West estimator
            n_obs: Number of observations
            regression: Regression type
            test_type: Test type ('tau' or 'rho')
        """
        self.test_statistic = test_statistic
        self.p_value = p_value
        self.critical_values = critical_values
        self.lags_used = lags_used
        self.n_obs = n_obs
        self.regression = regression
        self.test_type = test_type

        # Same interpretation as ADF: reject H0 => stationary
        self.is_stationary = p_value < 0.05

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PPResult(statistic={self.test_statistic:.4f}, "
            f"p_value={self.p_value:.4f}, "
            f"stationary={self.is_stationary})"
        )

    def summary(self) -> str:
        """Human-readable summary of PP test results.

        Returns:
            Formatted summary string
        """
        lines = [
            "Phillips-Perron Unit Root Test Results",
            "=" * 50,
            f"Test Statistic:    {self.test_statistic:.4f}",
            f"P-value:           {self.p_value:.4f}",
            f"Lags Used:         {self.lags_used}",
            f"Observations:      {self.n_obs}",
            f"Regression Type:   {self.regression}",
            f"Test Type:         {self.test_type}",
        ]

        lines.append("")
        lines.append("Critical Values:")
        for level, value in sorted(self.critical_values.items()):
            lines.append(f"  {level:>4s}: {value:>8.4f}")

        lines.append("")
        lines.append(f"Conclusion: {'Stationary' if self.is_stationary else 'Non-stationary'}")
        lines.append(
            f"  (Reject H0 at 5% level: {self.is_stationary})"
            if self.is_stationary
            else "  (Fail to reject H0 at 5% level)"
        )
        lines.append("")
        lines.append("IMPORTANT: PP tests H0 = unit root (same as ADF)")
        lines.append("  - Low p-value (<0.05) => stationary")
        lines.append("  - High p-value (>0.05) => non-stationary")
        lines.append("  - PP more robust to heteroscedasticity than ADF")

        return "\n".join(lines)


def pp_test(
    data: pd.Series | np.ndarray,
    lags: int | None = None,
    regression: Literal["c", "ct", "n"] = "c",
    test_type: Literal["tau", "rho"] = "tau",
) -> PPResult:
    """Perform Phillips-Perron test for unit root.

    The Phillips-Perron (PP) test is a non-parametric alternative to the
    Augmented Dickey-Fuller test. Like ADF, it tests the null hypothesis
    that a unit root is present in the time series. If the null is rejected
    (p < alpha), the series is considered stationary.

    Key Differences from ADF:
        - PP uses non-parametric Newey-West correction for serial correlation
        - PP estimates regression with only 1 lag (vs ADF's multiple lags)
        - PP more robust to general forms of heteroscedasticity
        - Both tests have same null hypothesis: unit root exists

    Regression types:
        - 'c': Constant only (default) - appropriate for returns
        - 'ct': Constant and trend - appropriate for prices
        - 'n': No constant, no trend - rarely used

    Test types:
        - 'tau': Based on t-statistic (default, recommended)
        - 'rho': Based on bias of regression coefficient

    Args:
        data: Time series data to test (1D array or Series)
        lags: Number of lags for Newey-West estimator. If None, uses
              automatic selection: 12*(nobs/100)^{1/4}
        regression: Type of regression to include in test
        test_type: Type of PP test statistic to compute

    Returns:
        PPResult with test statistics and conclusion

    Raises:
        ImportError: If arch package is not installed
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Example:
        >>> import numpy as np
        >>> # Test random walk (non-stationary)
        >>> rw = np.cumsum(np.random.randn(1000))
        >>> result = pp_test(rw)
        >>> print(result.summary())
        >>>
        >>> # Test with trend regression
        >>> result = pp_test(rw, regression='ct')
        >>> print(f"Stationary: {result.is_stationary}")
        >>>
        >>> # Compare PP with ADF on heteroscedastic data
        >>> # PP should be more reliable
        >>> from ml4t.diagnostic.evaluation.stationarity import adf_test
        >>> het_data = np.random.randn(1000) * (1 + 0.5 * np.random.randn(1000)**2)
        >>> adf_result = adf_test(het_data)
        >>> pp_result = pp_test(het_data)
        >>> print(f"ADF stationary: {adf_result.is_stationary}")
        >>> print(f"PP stationary: {pp_result.is_stationary}")

    Notes:
        - Requires arch package: pip install arch or pip install ml4t-diagnostic[advanced]
        - For financial returns, 'c' (constant only) is typically appropriate
        - For price series, 'ct' (constant + trend) may be better
        - PP is more robust than ADF for heteroscedastic time series
        - Use both PP and ADF for robust stationarity assessment
    """
    # Check if arch package is available (lazy check)
    if not _check_arch_available():
        raise ImportError(
            "Phillips-Perron test requires the arch package. "
            "Install with: pip install arch or pip install ml4t-diagnostic[advanced]"
        )

    # Input validation
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "pp_test"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
        logger.debug("Converted pandas Series to numpy array", shape=arr.shape)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "pp_test", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "pp_test", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError("Data cannot be empty", context={"function": "pp_test", "length": 0})

    # Check for missing values
    if np.any(np.isnan(arr)):
        n_missing = np.sum(np.isnan(arr))
        raise ValidationError(
            f"Data contains {n_missing} missing values (NaN)",
            context={"function": "pp_test", "n_missing": n_missing, "length": len(arr)},
        )

    # Check for infinite values
    if np.any(np.isinf(arr)):
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(
            f"Data contains {n_inf} infinite values",
            context={"function": "pp_test", "n_inf": n_inf, "length": len(arr)},
        )

    # Check minimum length
    min_length = 10
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for PP test (need at least {min_length} observations)",
            context={
                "function": "pp_test",
                "length": len(arr),
                "min_length": min_length,
            },
        )

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "pp_test",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    # Validate regression type
    valid_regressions = {"c", "ct", "n"}
    if regression not in valid_regressions:
        raise ValidationError(
            f"Invalid regression type: {regression}. Must be one of {valid_regressions}",
            context={"function": "pp_test", "regression": regression},
        )

    # Log test parameters
    logger.info(
        "Running PP test",
        n_obs=len(arr),
        lags=lags,
        regression=regression,
        test_type=test_type,
    )

    # Run PP test using arch package
    try:
        # Import here to avoid slow module-level import
        from arch.unitroot import PhillipsPerron

        # Create PP test object
        pp = PhillipsPerron(arr, lags=lags, trend=regression, test_type=test_type)

        # Extract results
        pp_stat = pp.stat
        pvalue = pp.pvalue
        usedlag = pp.lags
        nobs = pp.nobs
        critical_vals = pp.critical_values

        logger.info(
            "PP test completed",
            statistic=pp_stat,
            p_value=pvalue,
            lags_used=usedlag,
            n_obs=nobs,
            stationary=pvalue < 0.05,
        )

        # Create result object
        return PPResult(
            test_statistic=float(pp_stat),
            p_value=float(pvalue),
            critical_values=dict(critical_vals),
            lags_used=int(usedlag),
            n_obs=int(nobs),
            regression=regression,
            test_type=test_type,
        )

    except ImportError as e:
        # Re-raise ImportError with helpful message
        logger.error("PP test failed - arch package not available")
        raise ImportError(
            "Phillips-Perron test requires the arch package. "
            "Install with: pip install arch or pip install ml4t-diagnostic[advanced]"
        ) from e
    except Exception as e:
        logger.error("PP test failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"PP test computation failed: {e}",
            context={
                "function": "pp_test",
                "n_obs": len(arr),
                "lags": lags,
                "regression": regression,
                "test_type": test_type,
            },
            cause=e,
        )
