"""Volatility clustering detection and modeling for time series.

This module provides statistical tests and models for analyzing conditional
heteroscedasticity (volatility clustering) in financial time series:

- ARCH-LM test - Tests for autoregressive conditional heteroscedasticity (ARCH effects)
- GARCH(p,q) fitting - Models time-varying volatility dynamics
- Comprehensive volatility analysis - Combines ARCH-LM and GARCH

Volatility clustering is a key stylized fact of financial returns where large
changes tend to be followed by large changes, and small changes by small changes.

Example:
    >>> import numpy as np
    >>> from ml4t.diagnostic.evaluation.volatility import arch_lm_test, analyze_volatility
    >>>
    >>> # White noise (no ARCH effects)
    >>> white_noise = np.random.randn(1000)
    >>> result = arch_lm_test(white_noise)
    >>> print(f"Has ARCH effects: {result.has_arch_effects}")  # Should be False
    >>>
    >>> # Comprehensive analysis
    >>> analysis = analyze_volatility(returns_data)
    >>> print(analysis.summary())

References:
    - Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity.
    - Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
"""

from .analysis import VolatilityAnalysisResult, analyze_volatility
from .arch import ARCHLMResult, arch_lm_test
from .garch import GARCHResult, fit_garch

__all__ = [
    # ARCH-LM test
    "ARCHLMResult",
    "arch_lm_test",
    # GARCH model
    "GARCHResult",
    "fit_garch",
    # Combined analysis
    "VolatilityAnalysisResult",
    "analyze_volatility",
]
