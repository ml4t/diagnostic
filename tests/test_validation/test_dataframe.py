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
    """Tests for ValidationError exception."""

    def test_basic_message(self):
        """Test error with just a message."""
        error = ValidationError("Test error")
        assert "Test error" in str(error)
        assert error.context == {}

    def test_with_context(self):
        """Test error with context dictionary."""
        error = ValidationError(
            "Missing columns",
            context={"required": ["a", "b"], "available": ["a"]},
        )
        assert "Missing columns" in str(error)
        assert "required" in str(error)
        assert error.context["required"] == ["a", "b"]

    def test_empty_context(self):
        """Test error with explicitly empty context."""
        error = ValidationError("Error", context={})
        assert "Error" in str(error)
        assert "Context" not in str(error)


class TestDataFrameValidator:
    """Tests for DataFrameValidator class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "close": [100.0, 101.5, 99.8, 102.1],
                "volume": [1000, 1200, 800, 1500],
                "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            }
        )

    @pytest.fixture
    def df_with_nulls(self):
        """Create DataFrame with null values."""
        return pl.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [4.0, 5.0, None],
            }
        )

    def test_require_columns_success(self, sample_df):
        """Test require_columns with all columns present."""
        validator = DataFrameValidator(sample_df)
        result = validator.require_columns(["close", "volume"])
        assert result is validator  # Chaining works

    def test_require_columns_missing(self, sample_df):
        """Test require_columns with missing columns."""
        validator = DataFrameValidator(sample_df)
        with pytest.raises(ValidationError) as exc_info:
            validator.require_columns(["close", "missing_col"])

        assert "missing_col" in str(exc_info.value)
        assert exc_info.value.context["missing"] == ["missing_col"]

    def test_require_numeric_success(self, sample_df):
        """Test require_numeric with numeric columns."""
        validator = DataFrameValidator(sample_df)
        result = validator.require_numeric(["close", "volume"])
        assert result is validator

    def test_require_numeric_failure(self, sample_df):
        """Test require_numeric with non-numeric column."""
        validator = DataFrameValidator(sample_df)
        with pytest.raises(ValidationError) as exc_info:
            validator.require_numeric(["symbol"])

        assert "symbol" in str(exc_info.value)

    def test_require_numeric_missing_column_ignored(self, sample_df):
        """Test require_numeric ignores missing columns."""
        validator = DataFrameValidator(sample_df)
        # Should not raise - missing column is ignored
        validator.require_numeric(["close", "nonexistent"])

    def test_check_nulls_no_nulls(self, sample_df):
        """Test check_nulls with no null values."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_nulls(allow_nulls=False)
        assert result is validator

    def test_check_nulls_with_nulls_not_allowed(self, df_with_nulls):
        """Test check_nulls when nulls exist but not allowed."""
        validator = DataFrameValidator(df_with_nulls)
        with pytest.raises(ValidationError) as exc_info:
            validator.check_nulls(allow_nulls=False)

        assert "null" in str(exc_info.value).lower()
        assert "a" in exc_info.value.context["null_columns"]

    def test_check_nulls_allowed(self, df_with_nulls):
        """Test check_nulls when nulls are allowed."""
        validator = DataFrameValidator(df_with_nulls)
        result = validator.check_nulls(allow_nulls=True)
        assert result is validator

    def test_check_nulls_specific_columns(self, df_with_nulls):
        """Test check_nulls on specific columns only."""
        # Create df where only 'a' has nulls
        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        validator = DataFrameValidator(df)

        # Check only 'b' column - should pass
        validator.check_nulls(columns=["b"], allow_nulls=False)

        # Check 'a' column - should fail
        with pytest.raises(ValidationError):
            validator.check_nulls(columns=["a"], allow_nulls=False)

    def test_check_nulls_missing_column_ignored(self, sample_df):
        """Test check_nulls ignores missing columns."""
        validator = DataFrameValidator(sample_df)
        # Should not raise - missing column is ignored
        validator.check_nulls(columns=["close", "nonexistent"], allow_nulls=False)

    def test_check_empty_with_data(self, sample_df):
        """Test check_empty with non-empty DataFrame."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_empty()
        assert result is validator

    def test_check_empty_with_empty_df(self):
        """Test check_empty with empty DataFrame."""
        empty_df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Float64, "b": pl.Float64})
        validator = DataFrameValidator(empty_df)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_empty()

        assert "empty" in str(exc_info.value).lower()

    def test_check_min_rows_success(self, sample_df):
        """Test check_min_rows with sufficient rows."""
        validator = DataFrameValidator(sample_df)
        result = validator.check_min_rows(4)
        assert result is validator

    def test_check_min_rows_failure(self, sample_df):
        """Test check_min_rows with insufficient rows."""
        validator = DataFrameValidator(sample_df)
        with pytest.raises(ValidationError) as exc_info:
            validator.check_min_rows(10)

        assert "rows" in str(exc_info.value).lower()
        assert exc_info.value.context["required"] == 10
        assert exc_info.value.context["actual"] == 4

    def test_chaining(self, sample_df):
        """Test method chaining."""
        validator = DataFrameValidator(sample_df)
        result = (
            validator.require_columns(["close"])
            .require_numeric(["close"])
            .check_nulls(allow_nulls=False)
            .check_empty()
            .check_min_rows(1)
        )
        assert result is validator


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "close": [100.0, 101.5, 99.8, 102.1, 103.0],
                "volume": [1000, 1200, 800, 1500, 1100],
            }
        )

    def test_basic_validation(self, sample_df):
        """Test basic validation passes."""
        validate_dataframe(sample_df)  # Should not raise

    def test_required_columns_present(self, sample_df):
        """Test with all required columns present."""
        validate_dataframe(sample_df, required_columns=["close", "volume"])

    def test_required_columns_missing(self, sample_df):
        """Test with missing required columns."""
        with pytest.raises(ValidationError):
            validate_dataframe(sample_df, required_columns=["close", "missing"])

    def test_numeric_columns(self, sample_df):
        """Test numeric column validation."""
        validate_dataframe(sample_df, numeric_columns=["close", "volume"])

    def test_allow_nulls_true(self):
        """Test with nulls allowed."""
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        validate_dataframe(df, allow_nulls=True)

    def test_allow_nulls_false(self):
        """Test with nulls not allowed."""
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        with pytest.raises(ValidationError):
            validate_dataframe(df, allow_nulls=False)

    def test_min_rows(self, sample_df):
        """Test minimum rows validation."""
        validate_dataframe(sample_df, min_rows=5)

        with pytest.raises(ValidationError):
            validate_dataframe(sample_df, min_rows=10)

    def test_empty_dataframe(self):
        """Test empty DataFrame fails."""
        empty_df = pl.DataFrame({"a": []}).cast({"a": pl.Float64})
        with pytest.raises(ValidationError):
            validate_dataframe(empty_df)


class TestValidateSchema:
    """Tests for validate_schema function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "close": [100.0, 101.5],
                "volume": [1000, 1200],
                "date": pl.Series(["2024-01-01", "2024-01-02"]).str.to_date(),
            }
        )

    def test_matching_schema_string_types(self, sample_df):
        """Test with matching schema using string type names."""
        validate_schema(
            sample_df,
            {"close": "Float64", "volume": "Int64"},
        )

    def test_matching_schema_polars_types(self, sample_df):
        """Test with matching schema using Polars types."""
        validate_schema(
            sample_df,
            {"close": pl.Float64, "date": pl.Date},
        )

    def test_missing_column(self, sample_df):
        """Test with missing column in schema."""
        with pytest.raises(ValidationError) as exc_info:
            validate_schema(sample_df, {"missing_col": "Float64"})

        assert "missing" in str(exc_info.value).lower()

    def test_type_mismatch(self, sample_df):
        """Test with type mismatch."""
        with pytest.raises(ValidationError) as exc_info:
            validate_schema(sample_df, {"close": "Int64"})  # close is Float64

        assert "mismatch" in str(exc_info.value).lower()

    def test_partial_schema(self, sample_df):
        """Test validating only some columns."""
        # Should pass - only checking close column
        validate_schema(sample_df, {"close": "Float64"})
