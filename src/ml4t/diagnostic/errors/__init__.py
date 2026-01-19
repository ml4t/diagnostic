"""
ML4T Diagnostic Error Handling Framework

Provides a comprehensive exception hierarchy for systematic error handling
across the ML4T Diagnostic library. All exceptions preserve context information and
provide actionable error messages.

Exception Hierarchy:
    QEvalError (base)
    ├── ConfigurationError      # Configuration and setup errors
    ├── ValidationError         # Data validation failures
    ├── ComputationError        # Calculation and numerical errors
    ├── DataError              # Data access and format errors
    └── IntegrationError       # External library integration errors

Example:
    >>> from ml4t.diagnostic.errors import ValidationError
    >>> try:
    ...     validate_returns(returns)
    ... except ValidationError as e:
    ...     print(f"Validation failed: {e}")
    ...     print(f"Context: {e.context}")
"""

from typing import Any


class QEvalError(Exception):
    """
    Base exception for all ML4T Diagnostic errors.

    All ML4T Diagnostic exceptions inherit from this base class, providing
    consistent error handling and context preservation.

    Attributes:
        message: Human-readable error description
        context: Additional error context (dict)
        cause: Original exception if error was wrapped

    Example:
        >>> raise QEvalError(
        ...     "Operation failed",
        ...     context={"operation": "compute_sharpe", "reason": "insufficient_data"}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize ML4T Diagnostic error.

        Args:
            message: Error description
            context: Additional error context
            cause: Original exception (for error chaining)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """Format error message with context."""
        parts = [self.message]

        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.cause:
            parts.append(f"\nCaused by: {type(self.cause).__name__}: {self.cause}")

        return "".join(parts)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r}, cause={self.cause!r})"


class ConfigurationError(QEvalError):
    """
    Configuration and setup errors.

    Raised when:
    - Invalid configuration values
    - Missing required configuration
    - Incompatible settings
    - Setup/initialization failures

    Example:
        >>> from ml4t.diagnostic.config import QEvalConfig
        >>> try:
        ...     config = QEvalConfig(n_splits=-1)  # Invalid
        ... except ConfigurationError as e:
        ...     print(f"Configuration error: {e}")
    """

    pass


class ValidationError(QEvalError):
    """
    Data validation failures.

    Raised when:
    - Required columns missing
    - Data type mismatches
    - Value constraints violated
    - Schema validation failures

    Note:
        This is distinct from the ValidationError in ml4t-diagnostic.validation.
        The validation module uses this exception type for all validation failures.

    Example:
        >>> from ml4t.diagnostic.validation import validate_returns
        >>> try:
        ...     validate_returns(invalid_returns)
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
        ...     print(f"Details: {e.context}")
    """

    pass


class ComputationError(QEvalError):
    """
    Calculation and numerical errors.

    Raised when:
    - Numerical instability (division by zero, overflow)
    - Insufficient data for calculation
    - Algorithm convergence failures
    - Invalid mathematical operations

    Example:
        >>> from ml4t.diagnostic.metrics import sharpe_ratio
        >>> try:
        ...     sr = sharpe_ratio([])  # Empty data
        ... except ComputationError as e:
        ...     print(f"Computation failed: {e}")
    """

    pass


class DataError(QEvalError):
    """
    Data access and format errors.

    Raised when:
    - Data cannot be loaded
    - Unexpected data format
    - Missing expected data
    - Data corruption

    Example:
        >>> from ml4t.diagnostic.integration.qfeatures import load_features
        >>> try:
        ...     features = load_features("missing_file.parquet")
        ... except DataError as e:
        ...     print(f"Data error: {e}")
    """

    pass


class IntegrationError(QEvalError):
    """
    External library integration errors.

    Raised when:
    - QFeatures integration fails
    - QEngine integration fails
    - External API errors
    - Version compatibility issues

    Example:
        >>> from ml4t.diagnostic.integration.qfeatures import FeaturesAdapter
        >>> try:
        ...     adapter = FeaturesAdapter()
        ...     features = adapter.load("data.parquet")
        ... except IntegrationError as e:
        ...     print(f"Integration error: {e}")
        ...     print(f"Library: {e.context.get('library')}")
    """

    pass


# Public API
__all__ = [
    "QEvalError",
    "ConfigurationError",
    "ValidationError",
    "ComputationError",
    "DataError",
    "IntegrationError",
]
