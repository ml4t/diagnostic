"""Augmented Dickey-Fuller test for unit root detection.

The ADF test tests the null hypothesis that a unit root is present
in the time series. If the null is rejected (p < alpha), the series
is considered stationary.

References:
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
      for autoregressive time series with a unit root.
    - MacKinnon, J. G. (1994). Approximate asymptotic distribution functions
      for unit-root and cointegration tests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


class ADFResult:
    """Results from Augmented Dickey-Fuller test.

    Attributes:
        test_statistic: ADF test statistic
        p_value: MacKinnon p-value for null hypothesis (unit root exists)
        critical_values: Critical values at 1%, 5%, 10% significance levels
        lags_used: Number of lags included in the test
        n_obs: Number of observations used in regression
        is_stationary: Whether series is stationary (rejects unit root at 5%)
        regression: Type of regression ('c', 'ct', 'ctt', 'n')
        autolag_method: Method used for lag selection if applicable
    """

    def __init__(
        self,
        test_statistic: float,
        p_value: float,
        critical_values: dict[str, float],
        lags_used: int,
        n_obs: int,
        regression: str,
        autolag_method: str | None = None,
    ):
        """Initialize ADF result.

        Args:
            test_statistic: ADF test statistic
            p_value: P-value for unit root hypothesis
            critical_values: Critical values dict with keys '1%', '5%', '10%'
            lags_used: Number of lags used in test
            n_obs: Number of observations
            regression: Regression type
            autolag_method: Automatic lag selection method if used
        """
        self.test_statistic = test_statistic
        self.p_value = p_value
        self.critical_values = critical_values
        self.lags_used = lags_used
        self.n_obs = n_obs
        self.regression = regression
        self.autolag_method = autolag_method

        # Determine stationarity at 5% significance level
        self.is_stationary = p_value < 0.05

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ADFResult(statistic={self.test_statistic:.4f}, "
            f"p_value={self.p_value:.4f}, "
            f"stationary={self.is_stationary})"
        )

    def summary(self) -> str:
        """Human-readable summary of ADF test results.

        Returns:
            Formatted summary string
        """
        lines = [
            "Augmented Dickey-Fuller Test Results",
            "=" * 50,
            f"Test Statistic:    {self.test_statistic:.4f}",
            f"P-value:           {self.p_value:.4f}",
            f"Lags Used:         {self.lags_used}",
            f"Observations:      {self.n_obs}",
            f"Regression Type:   {self.regression}",
        ]

        if self.autolag_method:
            lines.append(f"Autolag Method:    {self.autolag_method}")

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

        return "\n".join(lines)


def adf_test(
    data: pd.Series | np.ndarray,
    maxlag: int | None = None,
    regression: Literal["c", "ct", "ctt", "n"] = "c",
    autolag: Literal["AIC", "BIC", "t-stat"] | None = "AIC",
) -> ADFResult:
    """Perform Augmented Dickey-Fuller test for unit root.

    The ADF test tests the null hypothesis that a unit root is present
    in the time series. If the null is rejected (p < alpha), the series
    is considered stationary.

    Regression types:
        - 'c': Constant only (default)
        - 'ct': Constant and trend
        - 'ctt': Constant, linear and quadratic trend
        - 'n': No constant, no trend

    Lag selection methods:
        - 'AIC': Akaike Information Criterion (default)
        - 'BIC': Bayesian Information Criterion
        - 't-stat': Based on t-statistic of last lag
        - None: Use maxlag directly

    Args:
        data: Time series data to test (1D array or Series)
        maxlag: Maximum number of lags to use. If None, uses 12*(nobs/100)^{1/4}
        regression: Type of regression to include in test
        autolag: Method for automatic lag selection. If None, uses maxlag directly

    Returns:
        ADFResult with test statistics and conclusion

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If test computation fails

    Example:
        >>> import numpy as np
        >>> # Test random walk (non-stationary)
        >>> rw = np.cumsum(np.random.randn(1000))
        >>> result = adf_test(rw)
        >>> print(result.summary())
        >>>
        >>> # Test with manual lag specification
        >>> result = adf_test(rw, maxlag=10, autolag=None)
        >>> print(f"Used {result.lags_used} lags")
        >>>
        >>> # Test with trend
        >>> result = adf_test(rw, regression='ct')
        >>> print(f"Stationary: {result.is_stationary}")

    Notes:
        - For financial returns, 'c' (constant only) is typically appropriate
        - For price series, 'ct' (constant + trend) may be better
        - Larger maxlag increases power but reduces sample size
        - AIC tends to select more lags than BIC
    """
    # Input validation
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "adf_test"})

    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
        logger.debug("Converted pandas Series to numpy array", shape=arr.shape)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "adf_test", "data_type": type(data).__name__},
        )

    # Check array properties
    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "adf_test", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError("Data cannot be empty", context={"function": "adf_test", "length": 0})

    # Check for missing values
    if np.any(np.isnan(arr)):
        n_missing = np.sum(np.isnan(arr))
        raise ValidationError(
            f"Data contains {n_missing} missing values (NaN)",
            context={"function": "adf_test", "n_missing": n_missing, "length": len(arr)},
        )

    # Check for infinite values
    if np.any(np.isinf(arr)):
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(
            f"Data contains {n_inf} infinite values",
            context={"function": "adf_test", "n_inf": n_inf, "length": len(arr)},
        )

    # Check minimum length
    min_length = 10 if maxlag is None else max(10, maxlag + 3)
    if len(arr) < min_length:
        raise ValidationError(
            f"Insufficient data for ADF test (need at least {min_length} observations)",
            context={
                "function": "adf_test",
                "length": len(arr),
                "min_length": min_length,
                "maxlag": maxlag,
            },
        )

    # Check for constant series
    if np.std(arr) == 0:
        raise ValidationError(
            "Data is constant (zero variance)",
            context={
                "function": "adf_test",
                "length": len(arr),
                "mean": float(np.mean(arr)),
            },
        )

    # Log test parameters
    logger.info(
        "Running ADF test",
        n_obs=len(arr),
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,
    )

    # Run ADF test
    try:
        result = adfuller(
            arr, maxlag=maxlag, regression=regression, autolag=autolag, regresults=False
        )

        # Unpack result
        # adfuller returns: (adf, pvalue, usedlag, nobs, critical_values, icbest)
        adf_stat = result[0]
        pvalue = result[1]
        usedlag = result[2]
        nobs = result[3]
        critical_vals = result[4]

        logger.info(
            "ADF test completed",
            statistic=adf_stat,
            p_value=pvalue,
            lags_used=usedlag,
            n_obs=nobs,
            stationary=pvalue < 0.05,
        )

        # Create result object
        return ADFResult(
            test_statistic=float(adf_stat),
            p_value=float(pvalue),
            critical_values=dict(critical_vals),
            lags_used=int(usedlag),
            n_obs=int(nobs),
            regression=regression,
            autolag_method=autolag,
        )

    except Exception as e:
        logger.error("ADF test failed", error=str(e), n_obs=len(arr))
        raise ComputationError(  # noqa: B904
            f"ADF test computation failed: {e}",
            context={
                "function": "adf_test",
                "n_obs": len(arr),
                "maxlag": maxlag,
                "regression": regression,
                "autolag": autolag,
            },
            cause=e,
        )
