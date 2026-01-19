"""
Tests for ML4T Diagnostic Error Handling Framework

Tests all custom exceptions, context preservation, error chaining,
and message formatting.
"""

import pytest

from ml4t.diagnostic.errors import (
    ComputationError,
    ConfigurationError,
    DataError,
    IntegrationError,
    QEvalError,
    ValidationError,
)


class TestQEvalError:
    """Test base QEvalError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = QEvalError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}
        assert error.cause is None

    def test_error_with_context(self):
        """Test error with context information."""
        context = {"operation": "compute_sharpe", "n_samples": 100}
        error = QEvalError("Computation failed", context=context)

        assert "Computation failed" in str(error)
        assert "operation: compute_sharpe" in str(error)
        assert "n_samples: 100" in str(error)
        assert error.context == context

    def test_error_with_cause(self):
        """Test error chaining with cause."""
        cause = ValueError("Invalid value")
        error = QEvalError("Operation failed", cause=cause)

        assert "Operation failed" in str(error)
        assert "Caused by: ValueError: Invalid value" in str(error)
        assert error.cause is cause

    def test_error_with_context_and_cause(self):
        """Test error with both context and cause."""
        cause = ZeroDivisionError("division by zero")
        context = {"numerator": 10, "denominator": 0}
        error = QEvalError("Division failed", context=context, cause=cause)

        error_str = str(error)
        assert "Division failed" in error_str
        assert "numerator: 10" in error_str
        assert "denominator: 0" in error_str
        assert "ZeroDivisionError" in error_str

    def test_error_repr(self):
        """Test error __repr__ method."""
        error = QEvalError("Test", context={"key": "value"})
        repr_str = repr(error)

        assert "QEvalError" in repr_str
        assert "Test" in repr_str
        assert "key" in repr_str
        assert "value" in repr_str

    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        error = QEvalError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """Test error can be raised and caught."""
        with pytest.raises(QEvalError) as exc_info:
            raise QEvalError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_error_can_be_caught_as_exception(self):
        """Test error can be caught as general Exception."""
        with pytest.raises(Exception) as exc_info:
            raise QEvalError("Test error")

        assert isinstance(exc_info.value, QEvalError)


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid n_splits value")
        assert "Invalid n_splits value" in str(error)
        assert isinstance(error, QEvalError)

    def test_configuration_error_with_context(self):
        """Test configuration error with context."""
        context = {
            "parameter": "n_splits",
            "value": -1,
            "valid_range": ">= 2",
        }
        error = ConfigurationError("Invalid configuration", context=context)

        error_str = str(error)
        assert "Invalid configuration" in error_str
        assert "parameter: n_splits" in error_str
        assert "value: -1" in error_str

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits correctly."""
        error = ConfigurationError("Test")
        assert isinstance(error, QEvalError)
        assert isinstance(error, Exception)

    def test_configuration_error_can_be_caught(self):
        """Test configuration error can be caught specifically."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Invalid config")

    def test_configuration_error_can_be_caught_as_base(self):
        """Test configuration error can be caught as QEvalError."""
        with pytest.raises(QEvalError):
            raise ConfigurationError("Invalid config")


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Missing required column")
        assert "Missing required column" in str(error)
        assert isinstance(error, QEvalError)

    def test_validation_error_with_context(self):
        """Test validation error with context."""
        context = {
            "required": ["returns", "dates"],
            "found": ["returns"],
            "missing": ["dates"],
        }
        error = ValidationError("Validation failed", context=context)

        error_str = str(error)
        assert "Validation failed" in error_str
        assert "required:" in error_str
        assert "missing:" in error_str

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits correctly."""
        error = ValidationError("Test")
        assert isinstance(error, QEvalError)
        assert isinstance(error, Exception)


class TestComputationError:
    """Test ComputationError exception."""

    def test_computation_error(self):
        """Test basic computation error."""
        error = ComputationError("Division by zero")
        assert "Division by zero" in str(error)
        assert isinstance(error, QEvalError)

    def test_computation_error_with_context(self):
        """Test computation error with context."""
        context = {
            "operation": "sharpe_ratio",
            "n_samples": 5,
            "min_required": 30,
        }
        error = ComputationError("Insufficient data", context=context)

        error_str = str(error)
        assert "Insufficient data" in error_str
        assert "operation: sharpe_ratio" in error_str
        assert "min_required: 30" in error_str

    def test_computation_error_with_cause(self):
        """Test computation error with underlying cause."""
        try:
            pass
        except ZeroDivisionError as e:
            error = ComputationError(
                "Sharpe ratio calculation failed", context={"metric": "sharpe_ratio"}, cause=e
            )

            error_str = str(error)
            assert "Sharpe ratio calculation failed" in error_str
            assert "ZeroDivisionError" in error_str

    def test_computation_error_inheritance(self):
        """Test ComputationError inherits correctly."""
        error = ComputationError("Test")
        assert isinstance(error, QEvalError)
        assert isinstance(error, Exception)


class TestDataError:
    """Test DataError exception."""

    def test_data_error(self):
        """Test basic data error."""
        error = DataError("File not found")
        assert "File not found" in str(error)
        assert isinstance(error, QEvalError)

    def test_data_error_with_context(self):
        """Test data error with context."""
        context = {
            "file_path": "/data/returns.parquet",
            "operation": "load",
            "expected_format": "parquet",
        }
        error = DataError("Failed to load data", context=context)

        error_str = str(error)
        assert "Failed to load data" in error_str
        assert "file_path:" in error_str
        assert "expected_format: parquet" in error_str

    def test_data_error_inheritance(self):
        """Test DataError inherits correctly."""
        error = DataError("Test")
        assert isinstance(error, QEvalError)
        assert isinstance(error, Exception)


class TestIntegrationError:
    """Test IntegrationError exception."""

    def test_integration_error(self):
        """Test basic integration error."""
        error = IntegrationError("QFeatures import failed")
        assert "QFeatures import failed" in str(error)
        assert isinstance(error, QEvalError)

    def test_integration_error_with_context(self):
        """Test integration error with context."""
        context = {
            "library": "qfeatures",
            "version": "1.0.0",
            "required_version": ">= 2.0.0",
        }
        error = IntegrationError("Version mismatch", context=context)

        error_str = str(error)
        assert "Version mismatch" in error_str
        assert "library: qfeatures" in error_str
        assert "required_version:" in error_str

    def test_integration_error_with_cause(self):
        """Test integration error with underlying import error."""
        try:
            import nonexistent_module  # noqa: F401 - intentionally importing nonexistent module
        except ImportError as e:
            error = IntegrationError(
                "Failed to import QFeatures", context={"module": "qfeatures"}, cause=e
            )

            error_str = str(error)
            assert "Failed to import QFeatures" in error_str
            assert "ImportError" in error_str or "ModuleNotFoundError" in error_str

    def test_integration_error_inheritance(self):
        """Test IntegrationError inherits correctly."""
        error = IntegrationError("Test")
        assert isinstance(error, QEvalError)
        assert isinstance(error, Exception)


class TestErrorHierarchy:
    """Test exception hierarchy and polymorphism."""

    def test_all_errors_are_qeval_errors(self):
        """Test all custom errors inherit from QEvalError."""
        errors = [
            ConfigurationError("test"),
            ValidationError("test"),
            ComputationError("test"),
            DataError("test"),
            IntegrationError("test"),
        ]

        for error in errors:
            assert isinstance(error, QEvalError)

    def test_catch_any_qeval_error(self):
        """Test catching any ML4T Diagnostic error with base class."""
        errors = [
            ConfigurationError("config"),
            ValidationError("validation"),
            ComputationError("computation"),
            DataError("data"),
            IntegrationError("integration"),
        ]

        for error in errors:
            with pytest.raises(QEvalError):
                raise error

    def test_catch_specific_error_type(self):
        """Test catching specific error types."""
        with pytest.raises(ValidationError):
            raise ValidationError("test")

        with pytest.raises(ComputationError):
            raise ComputationError("test")

    def test_error_type_checking(self):
        """Test error type checking."""
        validation_error = ValidationError("test")
        computation_error = ComputationError("test")

        assert isinstance(validation_error, ValidationError)
        assert not isinstance(validation_error, ComputationError)

        assert isinstance(computation_error, ComputationError)
        assert not isinstance(computation_error, ValidationError)


class TestErrorPracticalUsage:
    """Test practical error handling scenarios."""

    def test_try_except_with_context(self):
        """Test practical try-except pattern with context."""

        def risky_operation(value):
            if value < 0:
                raise ValidationError(
                    "Value must be non-negative", context={"value": value, "constraint": ">= 0"}
                )
            return value * 2

        with pytest.raises(ValidationError) as exc_info:
            risky_operation(-5)

        error = exc_info.value
        assert error.context["value"] == -5
        assert error.context["constraint"] == ">= 0"

    def test_error_wrapping_pattern(self):
        """Test wrapping lower-level errors."""

        def low_level_operation():
            raise ValueError("Invalid value")

        def high_level_operation():
            try:
                low_level_operation()
            except ValueError as e:
                # B904: Using cause=e parameter instead of from e (custom error handling)
                raise ComputationError(  # noqa: B904
                    "High-level operation failed", context={"operation": "high_level"}, cause=e
                )

        with pytest.raises(ComputationError) as exc_info:
            high_level_operation()

        error = exc_info.value
        assert isinstance(error.cause, ValueError)
        assert "Invalid value" in str(error.cause)

    def test_multiple_error_types(self):
        """Test handling multiple error types."""

        def operation(mode):
            if mode == "config":
                raise ConfigurationError("Bad config")
            elif mode == "validation":
                raise ValidationError("Bad data")
            elif mode == "computation":
                raise ComputationError("Bad math")

        with pytest.raises(ConfigurationError):
            operation("config")

        with pytest.raises(ValidationError):
            operation("validation")

        with pytest.raises(ComputationError):
            operation("computation")

    def test_error_information_extraction(self):
        """Test extracting information from caught errors."""
        context = {
            "metric": "sharpe_ratio",
            "n_samples": 10,
            "min_samples": 30,
        }

        try:
            raise ComputationError("Insufficient data", context=context)
        except ComputationError as e:
            assert e.message == "Insufficient data"
            assert e.context["metric"] == "sharpe_ratio"
            assert e.context["n_samples"] == 10
            assert e.context["min_samples"] == 30


class TestErrorExports:
    """Test that all errors are properly exported."""

    def test_all_errors_importable(self):
        """Test all error classes can be imported from package."""
        from ml4t.diagnostic.errors import (
            ComputationError,
            ConfigurationError,
            DataError,
            IntegrationError,
            QEvalError,
            ValidationError,
        )

        # Verify they are classes
        assert isinstance(QEvalError, type)
        assert isinstance(ConfigurationError, type)
        assert isinstance(ValidationError, type)
        assert isinstance(ComputationError, type)
        assert isinstance(DataError, type)
        assert isinstance(IntegrationError, type)
