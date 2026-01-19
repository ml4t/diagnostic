"""Stationarity testing for time series features.

This module provides statistical tests for detecting unit roots and assessing
stationarity of financial time series:

- Augmented Dickey-Fuller (ADF) test - tests for unit root (H0: non-stationary)
- KPSS test - tests for stationarity (H0: stationary)
- Phillips-Perron (PP) test - robust alternative to ADF (H0: non-stationary)

Stationarity is a critical assumption for many time series models and
feature engineering techniques. Non-stationary series require transformation
(differencing, detrending) before use in predictive models.

Key Differences Between Tests:
    - ADF: Parametric test with lagged differences, H0 = unit root (non-stationary)
    - PP: Non-parametric correction for serial correlation, H0 = unit root (non-stationary)
    - KPSS: H0 = stationarity (opposite interpretation!)
    - Use multiple tests together for robust stationarity assessment
    - Stationary: ADF/PP rejects + KPSS fails to reject
    - Non-stationary: ADF/PP fails to reject + KPSS rejects
    - Quasi-stationary: Both reject or both fail to reject (inconclusive)

Phillips-Perron vs ADF:
    - PP uses non-parametric Newey-West correction for heteroscedasticity
    - PP estimates regression with only 1 lag vs ADF's multiple lags
    - PP more robust to general forms of serial correlation
    - Both have same null hypothesis: unit root exists (non-stationary)

References:
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
      for autoregressive time series with a unit root.
    - Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time
      series regression. Biometrika, 75(2), 335-346.
    - MacKinnon, J. G. (1994). Approximate asymptotic distribution functions
      for unit-root and cointegration tests.
    - Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992).
      Testing the null hypothesis of stationarity against the alternative
      of a unit root. Journal of Econometrics, 54(1-3), 159-178.

Example:
    >>> import numpy as np
    >>> from ml4t.diagnostic.evaluation.stationarity import adf_test, kpss_test
    >>>
    >>> # White noise (stationary)
    >>> white_noise = np.random.randn(1000)
    >>> adf = adf_test(white_noise)
    >>> kpss = kpss_test(white_noise)
    >>> print(f"ADF stationary: {adf.is_stationary}")   # Should be True
    >>> print(f"KPSS stationary: {kpss.is_stationary}") # Should be True
    >>>
    >>> # Random walk (non-stationary)
    >>> random_walk = np.cumsum(np.random.randn(1000))
    >>> adf = adf_test(random_walk)
    >>> kpss = kpss_test(random_walk)
    >>> print(f"ADF stationary: {adf.is_stationary}")   # Should be False
    >>> print(f"KPSS stationary: {kpss.is_stationary}") # Should be False
    >>>
    >>> # Comprehensive analysis with all tests
    >>> from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity
    >>> result = analyze_stationarity(random_walk)
    >>> print(result.summary())
"""

# Import from submodules and re-export
from ml4t.diagnostic.evaluation.stationarity.analysis import (
    StationarityAnalysisResult,
    analyze_stationarity,
)
from ml4t.diagnostic.evaluation.stationarity.augmented_dickey_fuller import (
    ADFResult,
    adf_test,
)
from ml4t.diagnostic.evaluation.stationarity.kpss_test import (
    KPSSResult,
    kpss_test,
)
from ml4t.diagnostic.evaluation.stationarity.phillips_perron import (
    HAS_ARCH,
    PPResult,
    pp_test,
)

__all__ = [
    # ADF test
    "adf_test",
    "ADFResult",
    # KPSS test
    "kpss_test",
    "KPSSResult",
    # PP test
    "pp_test",
    "PPResult",
    "HAS_ARCH",
    # Comprehensive analysis
    "analyze_stationarity",
    "StationarityAnalysisResult",
]
