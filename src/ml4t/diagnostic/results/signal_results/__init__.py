"""Signal analysis result classes.

This package provides Pydantic result classes for storing and serializing
signal analysis outputs including IC metrics, quantile analysis, turnover,
and tear sheet data.

The package is decomposed into focused submodules:
- validation: Helper functions for key validation and period normalization
- ic: IC-related classes (ICStats, SignalICResult, RASICResult)
- quantile: Quantile analysis (QuantileAnalysisResult)
- turnover: Turnover analysis (TurnoverAnalysisResult)
- irtc: Transaction-cost adjusted IR (IRtcResult)
- tearsheet: Complete tear sheet (SignalTearSheet)

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
Paleologo, G. (2024). "Elements of Quantitative Investing"
"""

from __future__ import annotations

# IC-related classes
from ml4t.diagnostic.results.signal_results.ic import (
    ICStats,
    RASICResult,
    SignalICResult,
)

# IR_tc analysis
from ml4t.diagnostic.results.signal_results.irtc import (
    IRtcResult,
)

# Quantile analysis
from ml4t.diagnostic.results.signal_results.quantile import (
    QuantileAnalysisResult,
)

# Complete tear sheet
from ml4t.diagnostic.results.signal_results.tearsheet import (
    SignalTearSheet,
)

# Turnover analysis
from ml4t.diagnostic.results.signal_results.turnover import (
    TurnoverAnalysisResult,
)

# Validation helpers (for internal use, but exported for testing)
from ml4t.diagnostic.results.signal_results.validation import (
    _figure_from_data,
    _normalize_period,
    _validate_dict_keys_match,
)

__all__ = [
    # Validation helpers
    "_figure_from_data",
    "_normalize_period",
    "_validate_dict_keys_match",
    # IC classes
    "ICStats",
    "SignalICResult",
    "RASICResult",
    # Quantile
    "QuantileAnalysisResult",
    # Turnover
    "TurnoverAnalysisResult",
    # IR_tc
    "IRtcResult",
    # Tear sheet
    "SignalTearSheet",
]
