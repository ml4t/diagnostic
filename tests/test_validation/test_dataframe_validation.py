"""Tests for DataFrame validation utilities."""

from __future__ import annotations

import polars as pl
import pytest

from ml4t.diagnostic.validation.dataframe import (
    DataFrameValidator,
    ValidationError,
    validate_dataframe,
    validate_schema,
)


class TestValidationError:
    """Tests for ValidationError class."""

    def test_basic_error_message(self):
        """Test error with no context."""
        error = ValidationError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.context == {}

    def test_error_with_context(self):
        """Test error with context information."""
        context = {"columns": ["a", "b"], "missing": ["c"]}
        error = ValidationError("Missing columns", context=context)

        message = str(error)
        assert "Missing columns" in message
        assert "columns:" in message
        assert "missing:" in message

    def test_error_context_formatting(self):
        """Test that context is properly formatted."""
        error = ValidationError("Error", context={"key1": "value1", "key2": [1, 2, 3]})

        message = str(error)
        assert "Context:" in message
        assert "key1: value1" in message
        assert "key2:" in message


class TestDataFrameValidator:
    """Tests for DataFrameValidator class."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create a sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "close": [100.0, 101.5, 99.8, 102.3],
                "volume": [1000, 1500, 1200, 1800],
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            }
        )

    def test_require_columns_success(self, sample_df: pl.DataFrame):
        """Test requiring columns that exist."""
        validator = DataFrameValidator(sample_df)
        result = validator.require_columns(["close", "volume"])

        assert result is validator  # Check chaining works

    def test_require_columns_missing(self, sample_df: pl.DataFrame):
        """Test requiring columns that don't exist."""
        validator = DataFrameValidator(sample_df)

        with pytest.raises(ValidationError) as exc_info:
            validator.require_columns(["close", "missing_col", "another_missing"])

        error = exc_info.value
        assert "Missing required columns" in str(error)
        assert "missing_col" in str(error)
        assert "another_missing" in str(error)
        assert "missing" in error.context

    def test_require_numeric_success(self, sample_df: pl.DataFrame):
        """Test requiring numeric columns that are numeric."""
        validator = DataFrameValidator(sample_df)
        result = validator.require_numeric(["close", "volume"])

        assert result is validator

    def test_require_numeric_failure(self, sample_df: pl.DataFrame):
        """Test requiring numeric on non-numeric column."""
        validator = DataFrameValidator(sample_df)

        with pytest.raises(ValidationError) as exc_info:
            validator.require_numeric(["symbol"])  # symbol is string

        error = exc_info.value
        assert "Non-numeric" in str(error)
        assert "symbol" in str(error)

    def test_require_numeric_skips_missing_columns(self, sample_df: pl.DataFrame):
        """Test that require_numeric skips columns that don't exist."""
        validator = DataFrameValidator(sample_df)
        # Should not raise - missing column is skipped
        validator.require_numeric(["close", "nonexistent"])

    def test_check_nulls_no_nulls(self, sample_df: pl.DataFrame):
        """Test check_nulls when no nulls exist."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_nulls(allow_nulls=False)

        assert result is validator

    def test_check_nulls_with_nulls(self):
        """Test check_nulls when nulls exist."""
        df = pl.DataFrame(
            {
                "a": [1, 2, None, 4],
                "b": [None, None, 3, 4],
            }
        )
        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_nulls(allow_nulls=False)

        error = exc_info.value
        assert "null values" in str(error)
        assert "null_counts" in error.context

    def test_check_nulls_allowed(self):
        """Test check_nulls when nulls are allowed."""
        df = pl.DataFrame(
            {
                "a": [1, 2, None, 4],
            }
        )
        validator = DataFrameValidator(df)
        result = validator.check_nulls(allow_nulls=True)

        assert result is validator

    def test_check_nulls_specific_columns(self):
        """Test check_nulls on specific columns."""
        df = pl.DataFrame(
            {
                "a": [1, 2, None, 4],  # Has nulls
                "b": [1, 2, 3, 4],  # No nulls
            }
        )
        validator = DataFrameValidator(df)

        # Checking only column 'b' should pass
        validator.check_nulls(columns=["b"], allow_nulls=False)

        # Checking column 'a' should fail
        with pytest.raises(ValidationError):
            validator.check_nulls(columns=["a"], allow_nulls=False)

    def test_check_nulls_nonexistent_column(self, sample_df: pl.DataFrame):
        """Test check_nulls skips columns that don't exist."""
        validator = DataFrameValidator(sample_df)
        # Should not raise - nonexistent column is skipped
        validator.check_nulls(columns=["nonexistent"], allow_nulls=False)

    def test_check_empty_not_empty(self, sample_df: pl.DataFrame):
        """Test check_empty on non-empty DataFrame."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_empty()

        assert result is validator

    def test_check_empty_is_empty(self):
        """Test check_empty on empty DataFrame."""
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Float64, "b": pl.Float64})
        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_empty()

        assert "empty" in str(exc_info.value).lower()

    def test_check_min_rows_success(self, sample_df: pl.DataFrame):
        """Test check_min_rows with sufficient rows."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_min_rows(4)

        assert result is validator

    def test_check_min_rows_failure(self, sample_df: pl.DataFrame):
        """Test check_min_rows with insufficient rows."""
        validator = DataFrameValidator(sample_df)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_min_rows(10)

        error = exc_info.value
        assert "Insufficient rows" in str(error)
        assert error.context["required"] == 10
        assert error.context["actual"] == 4

    def test_chaining(self, sample_df: pl.DataFrame):
        """Test method chaining."""
        validator = DataFrameValidator(sample_df)

        # All these should chain successfully
        result = (
            validator.require_columns(["close", "volume"])
            .require_numeric(["close", "volume"])
            .check_nulls(allow_nulls=False)
            .check_empty()
            .check_min_rows(1)
        )

        assert result is validator


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_all_valid(self):
        """Test with fully valid DataFrame."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.5, 99.8, 102.3],
                "volume": [1000, 1500, 1200, 1800],
            }
        )

        # Should not raise
        validate_dataframe(
            df,
            required_columns=["close", "volume"],
            numeric_columns=["close", "volume"],
            allow_nulls=False,
            min_rows=4,
        )

    def test_missing_required_columns(self):
        """Test with missing required columns."""
        df = pl.DataFrame({"close": [100.0]})

        with pytest.raises(ValidationError):
            validate_dataframe(df, required_columns=["close", "volume"])

    def test_non_numeric_columns(self):
        """Test with non-numeric columns where numeric expected."""
        df = pl.DataFrame({"value": ["a", "b", "c"]})

        with pytest.raises(ValidationError):
            validate_dataframe(df, numeric_columns=["value"])

    def test_nulls_not_allowed(self):
        """Test with nulls when not allowed."""
        df = pl.DataFrame({"value": [1.0, None, 3.0]})

        with pytest.raises(ValidationError):
            validate_dataframe(df, allow_nulls=False)

    def test_nulls_allowed(self):
        """Test with nulls when allowed."""
        df = pl.DataFrame({"value": [1.0, None, 3.0]})

        # Should not raise
        validate_dataframe(df, allow_nulls=True)

    def test_insufficient_rows(self):
        """Test with insufficient rows."""
        df = pl.DataFrame({"value": [1.0, 2.0]})

        with pytest.raises(ValidationError):
            validate_dataframe(df, min_rows=10)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"value": []}).cast({"value": pl.Float64})

        with pytest.raises(ValidationError):
            validate_dataframe(df)

    def test_default_parameters(self):
        """Test with default parameters."""
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        # Should not raise with defaults
        validate_dataframe(df)


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_valid_schema_string_types(self):
        """Test schema validation with string type names."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.5],
                "volume": [1000, 1500],
            }
        )

        # Should not raise
        validate_schema(df, {"close": "Float64", "volume": "Int64"})

    def test_valid_schema_type_objects(self):
        """Test schema validation with Polars type objects."""
        df = pl.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
            }
        ).with_columns(pl.col("date").str.to_date())

        # Should not raise
        validate_schema(df, {"date": pl.Date})

    def test_missing_column(self):
        """Test schema validation with missing column."""
        df = pl.DataFrame({"a": [1, 2]})

        with pytest.raises(ValidationError) as exc_info:
            validate_schema(df, {"a": "Int64", "b": "Float64"})

        assert "Schema mismatch" in str(exc_info.value)
        assert "b" in str(exc_info.value)

    def test_type_mismatch(self):
        """Test schema validation with type mismatch."""
        df = pl.DataFrame({"value": ["a", "b"]})

        with pytest.raises(ValidationError) as exc_info:
            validate_schema(df, {"value": "Float64"})

        assert "Schema mismatch" in str(exc_info.value)

    def test_partial_match(self):
        """Test that type names with partial matches work."""
        df = pl.DataFrame({"value": [1.0, 2.0]})

        # Float64 contains "Float"
        validate_schema(df, {"value": "Float"})

    def test_empty_schema(self):
        """Test with empty expected schema."""
        df = pl.DataFrame({"a": [1, 2]})

        # Should not raise with empty schema
        validate_schema(df, {})


class TestValidatorEdgeCases:
    """Edge case tests for validators."""

    def test_single_row_dataframe(self):
        """Test validation with single row."""
        df = pl.DataFrame({"value": [1.0]})

        validate_dataframe(df, min_rows=1)

        with pytest.raises(ValidationError):
            validate_dataframe(df, min_rows=2)

    def test_single_column_dataframe(self):
        """Test validation with single column."""
        df = pl.DataFrame({"only_column": [1.0, 2.0, 3.0]})

        validate_dataframe(
            df,
            required_columns=["only_column"],
            numeric_columns=["only_column"],
        )

    def test_many_null_columns(self):
        """Test with many columns containing nulls."""
        df = pl.DataFrame({f"col{i}": [1, None, 3] for i in range(10)})

        with pytest.raises(ValidationError) as exc_info:
            validate_dataframe(df, allow_nulls=False)

        error = exc_info.value
        # Should report multiple null columns
        assert len(error.context["null_columns"]) == 10

    def test_mixed_numeric_non_numeric(self):
        """Test DataFrame with mixed column types."""
        df = pl.DataFrame(
            {
                "numeric1": [1.0, 2.0],
                "string1": ["a", "b"],
                "numeric2": [10, 20],
                "string2": ["x", "y"],
            }
        )

        # Numeric columns should validate
        validator = DataFrameValidator(df)
        validator.require_numeric(["numeric1", "numeric2"])

        # Non-numeric columns should fail
        with pytest.raises(ValidationError):
            validator.require_numeric(["string1", "string2"])

    def test_boolean_column_as_numeric(self):
        """Test that boolean columns are not considered numeric."""
        df = pl.DataFrame({"flag": [True, False, True]})

        # Boolean is not numeric in Polars
        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError):
            validator.require_numeric(["flag"])

    def test_datetime_column_as_numeric(self):
        """Test that datetime columns are not considered numeric."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 3), eager=True),
            }
        )

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError):
            validator.require_numeric(["date"])
