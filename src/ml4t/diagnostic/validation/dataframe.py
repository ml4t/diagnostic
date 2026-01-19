"""DataFrame validation utilities."""

from __future__ import annotations

from typing import Any

import polars as pl


class ValidationError(ValueError):
    """Validation error with helpful context."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        """Initialize validation error.

        Args:
            message: Error message
            context: Additional context (columns, types, etc.)
        """
        self.context = context or {}
        super().__init__(self._format_message(message))

    def _format_message(self, message: str) -> str:
        """Format error message with context."""
        if not self.context:
            return message

        context_str = "\n".join(f"  {k}: {v}" for k, v in self.context.items())
        return f"{message}\nContext:\n{context_str}"


class DataFrameValidator:
    """Validator for Polars DataFrames.

    Examples:
        >>> validator = DataFrameValidator(df)
        >>> validator.require_columns(["close", "volume"])
        >>> validator.require_numeric(["close", "volume"])
        >>> validator.check_nulls(allow_nulls=False)
    """

    def __init__(self, df: pl.DataFrame):
        """Initialize validator.

        Args:
            df: DataFrame to validate
        """
        self.df = df

    def require_columns(self, columns: list[str]) -> DataFrameValidator:
        """Require specific columns to exist.

        Args:
            columns: Required column names

        Returns:
            Self for chaining

        Raises:
            ValidationError: If required columns missing
        """
        missing = [col for col in columns if col not in self.df.columns]

        if missing:
            raise ValidationError(
                f"Missing required columns: {missing}",
                context={
                    "required": columns,
                    "available": self.df.columns,
                    "missing": missing,
                },
            )

        return self

    def require_numeric(self, columns: list[str]) -> DataFrameValidator:
        """Require columns to be numeric types.

        Args:
            columns: Column names that must be numeric

        Returns:
            Self for chaining

        Raises:
            ValidationError: If columns not numeric
        """
        non_numeric = []

        for col in columns:
            if col not in self.df.columns:
                continue

            dtype = self.df[col].dtype
            if not dtype.is_numeric():
                non_numeric.append((col, str(dtype)))

        if non_numeric:
            raise ValidationError(
                f"Non-numeric columns: {[col for col, _ in non_numeric]}",
                context={
                    "expected": "numeric types (Int*, Float*, Decimal)",
                    "actual": dict(non_numeric),
                },
            )

        return self

    def check_nulls(
        self, columns: list[str] | None = None, allow_nulls: bool = False
    ) -> DataFrameValidator:
        """Check for null values in columns.

        Args:
            columns: Columns to check (None = all columns)
            allow_nulls: Whether nulls are allowed

        Returns:
            Self for chaining

        Raises:
            ValidationError: If nulls found when not allowed
        """
        check_columns = columns or self.df.columns

        if not allow_nulls:
            null_counts = {}

            for col in check_columns:
                if col not in self.df.columns:
                    continue

                null_count = self.df[col].null_count()
                if null_count > 0:
                    null_counts[col] = null_count

            if null_counts:
                total_nulls = sum(null_counts.values())
                raise ValidationError(
                    f"Found {total_nulls} null values",
                    context={
                        "null_columns": list(null_counts.keys()),
                        "null_counts": null_counts,
                    },
                )

        return self

    def check_empty(self) -> DataFrameValidator:
        """Check if DataFrame is empty.

        Returns:
            Self for chaining

        Raises:
            ValidationError: If DataFrame is empty
        """
        if len(self.df) == 0:
            raise ValidationError(
                "DataFrame is empty",
                context={"shape": self.df.shape, "columns": self.df.columns},
            )

        return self

    def check_min_rows(self, min_rows: int) -> DataFrameValidator:
        """Check minimum number of rows.

        Args:
            min_rows: Minimum required rows

        Returns:
            Self for chaining

        Raises:
            ValidationError: If too few rows
        """
        if len(self.df) < min_rows:
            raise ValidationError(
                f"Insufficient rows: {len(self.df)} < {min_rows}",
                context={"required": min_rows, "actual": len(self.df)},
            )

        return self


def validate_dataframe(
    df: pl.DataFrame,
    required_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    allow_nulls: bool = True,
    min_rows: int = 1,
) -> None:
    """Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: Columns that must exist
        numeric_columns: Columns that must be numeric
        allow_nulls: Whether null values are allowed
        min_rows: Minimum number of rows required

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_dataframe(
        ...     df,
        ...     required_columns=["close", "volume"],
        ...     numeric_columns=["close", "volume"],
        ...     allow_nulls=False,
        ...     min_rows=100
        ... )
    """
    validator = DataFrameValidator(df)

    validator.check_empty().check_min_rows(min_rows)

    if required_columns:
        validator.require_columns(required_columns)

    if numeric_columns:
        validator.require_numeric(numeric_columns)

    if not allow_nulls:
        validator.check_nulls(allow_nulls=False)


def validate_schema(df: pl.DataFrame, expected_schema: dict[str, str | type]) -> None:
    """Validate DataFrame schema matches expected types.

    Args:
        df: DataFrame to validate
        expected_schema: Map of column name to expected type

    Raises:
        ValidationError: If schema doesn't match

    Examples:
        >>> validate_schema(df, {
        ...     "close": "Float64",
        ...     "volume": "Int64",
        ...     "date": pl.Date
        ... })
    """
    mismatches = {}

    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            mismatches[col] = ("missing", expected_type)
            continue

        actual_type = df[col].dtype

        # Handle string type names
        if isinstance(expected_type, str):
            expected_type_str = expected_type
            actual_type_str = str(actual_type)
        else:
            expected_type_str = str(expected_type)
            actual_type_str = str(actual_type)

        if expected_type_str not in actual_type_str:
            mismatches[col] = (actual_type_str, expected_type_str)

    if mismatches:
        raise ValidationError(
            "Schema mismatch",
            context={
                "mismatches": {
                    col: f"expected {exp}, got {act}" for col, (act, exp) in mismatches.items()
                }
            },
        )
