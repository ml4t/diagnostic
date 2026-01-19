"""
ML4T Evaluation Error Handling Examples

Demonstrates all error types, context preservation, error chaining,
and practical error handling patterns.
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import error types (in production, use: from ml4t.diagnostic.errors import ...)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ml4t.diagnostic.errors",
    Path(__file__).parent.parent / "src" / "ml4t-diagnostic" / "errors" / "__init__.py",
)
errors = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(errors)  # type: ignore[union-attr,union-attr]

ML4TEvaluationError = errors.ML4TEvaluationError
ConfigurationError = errors.ConfigurationError
ValidationError = errors.ValidationError
ComputationError = errors.ComputationError
DataError = errors.DataError
IntegrationError = errors.IntegrationError


def example_1_basic_error():
    """Example 1: Basic error with context."""
    print("=" * 70)
    print("Example 1: Basic Error with Context")
    print("=" * 70)

    try:
        raise ValidationError(
            "Missing required column",
            context={"required": ["returns", "dates"], "found": ["returns"], "missing": ["dates"]},
        )
    except ValidationError as e:
        print(f"Error caught: {e}\n")
        print(f"Error type: {type(e).__name__}")
        print(f"Message: {e.message}")
        print(f"Context: {e.context}")
        print()


def example_2_error_chaining():
    """Example 2: Error chaining with cause."""
    print("=" * 70)
    print("Example 2: Error Chaining")
    print("=" * 70)

    try:
        try:
            # Simulate low-level error
            pass
        except ZeroDivisionError as e:
            # Wrap with high-level error
            raise ComputationError(
                "Sharpe ratio calculation failed",
                context={"metric": "sharpe_ratio", "n_samples": 100, "std_dev": 0.0},
                cause=e,
            )
    except ComputationError as e:
        print(f"High-level error:\n{e}\n")
        print(f"Original cause: {type(e.cause).__name__}")
        print()


def example_3_configuration_error():
    """Example 3: Configuration error."""
    print("=" * 70)
    print("Example 3: Configuration Error")
    print("=" * 70)

    try:
        n_splits = -1
        if n_splits < 2:
            raise ConfigurationError(
                "Invalid n_splits value",
                context={
                    "parameter": "n_splits",
                    "value": n_splits,
                    "constraint": ">= 2",
                    "suggestion": "Use n_splits >= 2 for cross-validation",
                },
            )
    except ConfigurationError as e:
        print(f"Configuration error:\n{e}\n")
        print(f"Suggestion: {e.context['suggestion']}")
        print()


def example_4_validation_error():
    """Example 4: Validation error with detailed context."""
    print("=" * 70)
    print("Example 4: Validation Error")
    print("=" * 70)

    try:
        # Simulate validation failure
        null_count = 1

        raise ValidationError(
            "Returns contain null values",
            context={
                "column": "returns",
                "null_count": null_count,
                "total_rows": 3,
                "null_percentage": 33.3,
                "suggestion": "Use .dropna() or .fillna() to clean data",
            },
        )
    except ValidationError as e:
        print(f"Validation error:\n{e}\n")
        print(f"Action needed: {e.context['suggestion']}")
        print()


def example_5_computation_error():
    """Example 5: Computation error."""
    print("=" * 70)
    print("Example 5: Computation Error")
    print("=" * 70)

    try:
        n_samples = 10
        min_required = 30

        if n_samples < min_required:
            raise ComputationError(
                "Insufficient data for Sharpe ratio calculation",
                context={
                    "metric": "sharpe_ratio",
                    "n_samples": n_samples,
                    "min_required": min_required,
                    "shortfall": min_required - n_samples,
                    "suggestion": f"Collect {min_required - n_samples} more samples",
                },
            )
    except ComputationError as e:
        print(f"Computation error:\n{e}\n")
        print(f"Data needed: {e.context['shortfall']} more samples")
        print()


def example_6_data_error():
    """Example 6: Data error."""
    print("=" * 70)
    print("Example 6: Data Error")
    print("=" * 70)

    try:
        file_path = "/data/missing_file.parquet"

        raise DataError(
            f"Failed to load data file: {file_path}",
            context={
                "file_path": file_path,
                "operation": "load",
                "expected_format": "parquet",
                "suggestion": "Check file path and permissions",
            },
        )
    except DataError as e:
        print(f"Data error:\n{e}\n")
        print(f"File: {e.context['file_path']}")
        print(f"Suggestion: {e.context['suggestion']}")
        print()


def example_7_integration_error():
    """Example 7: Integration error."""
    print("=" * 70)
    print("Example 7: Integration Error")
    print("=" * 70)

    try:
        # Simulate version mismatch
        raise IntegrationError(
            "ML4T Engineer version mismatch",
            context={
                "library": "qfeatures",
                "found_version": "1.0.0",
                "required_version": ">= 2.0.0",
                "suggestion": "Upgrade qfeatures: pip install --upgrade ml4t-features",
            },
        )
    except IntegrationError as e:
        print(f"Integration error:\n{e}\n")
        print(f"Library: {e.context['library']}")
        print(f"Fix: {e.context['suggestion']}")
        print()


def example_8_error_hierarchy():
    """Example 8: Error hierarchy and polymorphic catching."""
    print("=" * 70)
    print("Example 8: Error Hierarchy")
    print("=" * 70)

    errors_to_test = [
        ConfigurationError("Config error"),
        ValidationError("Validation error"),
        ComputationError("Computation error"),
        DataError("Data error"),
        IntegrationError("Integration error"),
    ]

    print("Testing polymorphic catch with ML4TEvaluationError:\n")
    for error in errors_to_test:
        try:
            raise error
        except ML4TEvaluationError as e:
            print(f"  ✅ Caught {type(e).__name__}: {e.message}")

    print("\nAll errors are ML4TEvaluationError instances:")
    for error in errors_to_test:
        is_qeval = isinstance(error, ML4TEvaluationError)
        print(f"  {type(error).__name__}: {is_qeval}")
    print()


def example_9_practical_validation():
    """Example 9: Practical validation pattern."""
    print("=" * 70)
    print("Example 9: Practical Validation Pattern")
    print("=" * 70)

    def validate_returns(data: dict[str, Any]) -> bool:
        """Validate returns data."""
        # Check required fields
        if "returns" not in data:
            raise ValidationError(
                "Missing required field 'returns'",
                context={
                    "required": ["returns", "dates"],
                    "found": list(data.keys()),
                    "missing": ["returns"],
                },
            )

        # Check data length
        if len(data["returns"]) < 30:
            raise ValidationError(
                "Insufficient data", context={"n_samples": len(data["returns"]), "min_required": 30}
            )

        return True

    # Test with invalid data
    try:
        invalid_data = {"prices": [100, 101, 102]}
        validate_returns(invalid_data)
    except ValidationError as e:
        print(f"Validation failed:\n{e}\n")
        print(f"Missing fields: {e.context['missing']}")
    print()


def example_10_error_recovery():
    """Example 10: Error recovery with fallback."""
    print("=" * 70)
    print("Example 10: Error Recovery with Fallback")
    print("=" * 70)

    def compute_metrics_safe(data: dict[str, Any]) -> dict[str, float | None]:
        """Compute metrics with graceful degradation."""
        metrics = {}

        # Required metric - propagate error
        try:
            metrics["mean"] = sum(data["returns"]) / len(data["returns"])
        except (KeyError, ZeroDivisionError) as e:
            raise ComputationError(
                "Failed to compute mean return", context={"data_keys": list(data.keys())}, cause=e
            )

        # Optional metric - use fallback
        try:
            if len(data["returns"]) < 30:
                raise ComputationError("Insufficient data for Sharpe ratio")
            metrics["sharpe"] = 1.5  # Placeholder calculation
        except ComputationError as e:
            print(f"  ⚠️  Optional metric failed: {e.message}")
            metrics["sharpe"] = None  # Fallback value

        return metrics

    # Test with minimal data
    data = {"returns": [0.01, 0.02, 0.015]}
    result = compute_metrics_safe(data)
    print("\nMetrics computed:")
    print(f"  Mean: {result['mean']:.4f}")
    print(f"  Sharpe: {result['sharpe']} (fallback due to insufficient data)")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "ML4T Evaluation Error Handling Examples" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    examples = [
        example_1_basic_error,
        example_2_error_chaining,
        example_3_configuration_error,
        example_4_validation_error,
        example_5_computation_error,
        example_6_data_error,
        example_7_integration_error,
        example_8_error_hierarchy,
        example_9_practical_validation,
        example_10_error_recovery,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example failed: {e}\n")

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
