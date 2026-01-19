"""Data validation utilities for ML4T Diagnostic inputs.

Provides comprehensive validation for DataFrames, time series, returns,
and other common financial data inputs.

Examples:
    >>> from ml4t.diagnostic.validation import validate_dataframe, validate_returns
    >>>
    >>> # Validate DataFrame structure
    >>> validate_dataframe(
    ...     df,
    ...     required_columns=["close", "volume"],
    ...     numeric_columns=["close", "volume"]
    ... )
    >>>
    >>> # Validate returns series
    >>> validate_returns(returns, allow_nulls=False, bounds=(-0.5, 0.5))
"""

from ml4t.diagnostic.validation.dataframe import (
    DataFrameValidator,
    ValidationError,
    validate_dataframe,
    validate_schema,
)
from ml4t.diagnostic.validation.returns import (
    ReturnsValidator,
    validate_bounds,
    validate_returns,
)
from ml4t.diagnostic.validation.timeseries import (
    TimeSeriesValidator,
    validate_frequency,
    validate_index,
    validate_timeseries,
)

__all__ = [
    # Core
    "ValidationError",
    # DataFrame validation
    "DataFrameValidator",
    "validate_dataframe",
    "validate_schema",
    # Time series validation
    "TimeSeriesValidator",
    "validate_timeseries",
    "validate_index",
    "validate_frequency",
    # Returns validation
    "ReturnsValidator",
    "validate_returns",
    "validate_bounds",
]
