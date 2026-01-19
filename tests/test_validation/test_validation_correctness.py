"""High-quality correctness tests for validation utilities.

These tests verify that validation functions correctly detect:
1. Missing columns
2. Invalid data types
3. Null values
4. Out-of-bounds values
5. Time series issues (unsorted, duplicates, gaps)

Each test verifies that validators catch real data quality issues
that would cause downstream analysis failures.
"""

from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

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


class TestDataFrameValidatorCorrectness:
    """Tests for DataFrame validation correctness."""

    def test_require_columns_detects_missing(self):
        """Missing columns should raise ValidationError with context."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError) as exc_info:
            validator.require_columns(["a", "b", "c"])

        assert "c" in str(exc_info.value)
        assert exc_info.value.context["missing"] == ["c"]

    def test_require_columns_passes_when_present(self):
        """Present columns should not raise."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        validator = DataFrameValidator(df)
        result = validator.require_columns(["a", "b", "c"])

        # Should return self for chaining
        assert result is validator

    def test_require_numeric_detects_non_numeric(self):
        """Non-numeric columns should raise ValidationError."""
        df = pl.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "string": ["a", "b", "c"],
            }
        )

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError) as exc_info:
            validator.require_numeric(["numeric", "string"])

        assert "string" in str(exc_info.value)
        assert "non-numeric" in str(exc_info.value).lower()

    def test_require_numeric_accepts_all_numeric_types(self):
        """All numeric types (int, float, decimal) should pass."""
        df = pl.DataFrame(
            {
                "int32": pl.Series([1, 2, 3], dtype=pl.Int32),
                "int64": pl.Series([1, 2, 3], dtype=pl.Int64),
                "float32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
                "float64": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            }
        )

        validator = DataFrameValidator(df)
        result = validator.require_numeric(["int32", "int64", "float32", "float64"])

        assert result is validator

    def test_check_nulls_detects_null_values(self):
        """Null values should be detected when allow_nulls=False."""
        df = pl.DataFrame(
            {
                "with_nulls": [1.0, None, 3.0],
                "no_nulls": [1.0, 2.0, 3.0],
            }
        )

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError) as exc_info:
            validator.check_nulls(allow_nulls=False)

        assert exc_info.value.context["null_counts"]["with_nulls"] == 1

    def test_check_nulls_allows_nulls_when_specified(self):
        """Null values should pass when allow_nulls=True."""
        df = pl.DataFrame({"with_nulls": [1.0, None, 3.0]})

        validator = DataFrameValidator(df)
        result = validator.check_nulls(allow_nulls=True)

        assert result is validator

    def test_check_empty_detects_empty_dataframe(self):
        """Empty DataFrame should raise ValidationError."""
        df = pl.DataFrame({"a": []})

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError, match="empty"):
            validator.check_empty()

    def test_check_min_rows_detects_insufficient_rows(self):
        """Insufficient rows should raise ValidationError."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        validator = DataFrameValidator(df)

        with pytest.raises(ValidationError, match="Insufficient rows"):
            validator.check_min_rows(10)

    def test_chaining_works_correctly(self):
        """Validators should support method chaining."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000, 2000, 3000],
            }
        )

        # Should not raise - all validations pass
        validator = (
            DataFrameValidator(df)
            .check_empty()
            .check_min_rows(3)
            .require_columns(["close", "volume"])
            .require_numeric(["close", "volume"])
            .check_nulls(allow_nulls=False)
        )

        assert validator is not None


class TestValidateDataframeFunction:
    """Tests for the validate_dataframe convenience function."""

    def test_validates_all_conditions(self):
        """Should check all specified conditions."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )

        # Should not raise
        validate_dataframe(
            df,
            required_columns=["close", "volume"],
            numeric_columns=["close", "volume"],
            allow_nulls=False,
            min_rows=5,
        )

    def test_detects_missing_columns(self):
        """Should detect missing required columns."""
        df = pl.DataFrame({"close": [100.0]})

        with pytest.raises(ValidationError, match="Missing"):
            validate_dataframe(df, required_columns=["close", "volume"])


class TestValidateSchemaFunction:
    """Tests for schema validation."""

    def test_validates_correct_schema(self):
        """Correct schema should pass."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0],
                "volume": [1000, 2000],
            }
        )

        # Should not raise
        validate_schema(
            df,
            {
                "close": "Float64",
                "volume": "Int64",
            },
        )

    def test_detects_schema_mismatch(self):
        """Wrong types should raise ValidationError."""
        df = pl.DataFrame(
            {
                "close": ["100", "101"],  # String instead of float
            }
        )

        with pytest.raises(ValidationError, match="Schema mismatch"):
            validate_schema(df, {"close": "Float64"})

    def test_detects_missing_columns_in_schema(self):
        """Missing columns should be reported."""
        df = pl.DataFrame({"a": [1]})

        with pytest.raises(ValidationError, match="Schema mismatch"):
            validate_schema(df, {"b": "Int64"})


class TestTimeSeriesValidatorCorrectness:
    """Tests for time series validation correctness."""

    def test_check_index_exists(self):
        """Missing index column should raise."""
        df = pl.DataFrame({"value": [1, 2, 3]})

        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError, match="not found"):
            validator.check_index_exists()

    def test_check_index_type_rejects_non_temporal(self):
        """Non-temporal index should raise."""
        df = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],  # String, not Date
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError, match="temporal"):
            validator.check_index_type()

    def test_check_index_type_accepts_date(self):
        """Date type should pass."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_index_type()

        assert result is validator

    def test_check_index_type_accepts_datetime(self):
        """Datetime type should pass."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_index_type()

        assert result is validator

    def test_check_sorted_detects_unsorted(self):
        """Unsorted time series should raise."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
                "value": [3, 1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError, match="not sorted"):
            validator.check_sorted()

    def test_check_sorted_passes_for_sorted(self):
        """Sorted time series should pass."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "value": [1, 2, 3],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_sorted()

        assert result is validator

    def test_check_duplicates_detects_duplicates(self):
        """Duplicate timestamps should raise."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
                "value": [1, 2, 3],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError, match="duplicate"):
            validator.check_duplicates()

    def test_check_duplicates_passes_for_unique(self):
        """Unique timestamps should pass."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "value": [1, 2, 3],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_duplicates()

        assert result is validator

    def test_check_gaps_detects_large_gaps(self):
        """Large gaps in time series should raise."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 15)],  # 14 day gap
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError, match="gap"):
            validator.check_gaps(max_gap_days=7)

    def test_check_gaps_passes_for_small_gaps(self):
        """Small gaps within threshold should pass."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 5)],  # 4 day gap
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_gaps(max_gap_days=7)

        assert result is validator


class TestValidateTimeseriesFunction:
    """Tests for the validate_timeseries convenience function."""

    def test_validates_all_conditions(self):
        """Should check all specified conditions."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "value": [1, 2, 3],
            }
        )

        # Should not raise
        validate_timeseries(
            df,
            index_col="date",
            require_sorted=True,
            check_duplicates=True,
            max_gap_days=7,
        )

    def test_detects_issues(self):
        """Should detect various time series issues."""
        # Unsorted
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 1)],
                "value": [2, 1],
            }
        )

        with pytest.raises(ValidationError, match="sorted"):
            validate_timeseries(df, index_col="date", require_sorted=True)


class TestReturnsValidatorCorrectness:
    """Tests for returns validation correctness."""

    def test_check_numeric_detects_non_numeric(self):
        """Non-numeric returns should raise."""
        returns = pl.Series(["1%", "2%", "3%"])

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="numeric"):
            validator.check_numeric()

    def test_check_bounds_detects_out_of_bounds(self):
        """Out-of-bounds returns should raise."""
        returns = pl.Series([0.01, 0.02, 1.5, -0.01])  # 1.5 = 150% return

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="upper bound"):
            validator.check_bounds(lower=-0.5, upper=0.5)

    def test_check_bounds_detects_lower_violation(self):
        """Returns below lower bound should raise."""
        returns = pl.Series([0.01, -0.8, 0.02])  # -80% return

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="lower bound"):
            validator.check_bounds(lower=-0.5, upper=0.5)

    def test_check_bounds_passes_valid_returns(self):
        """Valid returns within bounds should pass."""
        returns = pl.Series([0.01, -0.02, 0.03, -0.01])

        validator = ReturnsValidator(returns)
        result = validator.check_bounds(lower=-0.1, upper=0.1)

        assert result is validator

    def test_check_finite_detects_infinite(self):
        """Infinite values should raise."""
        returns = pl.Series([0.01, float("inf"), -0.02])

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="infinite"):
            validator.check_finite()

    def test_check_finite_detects_negative_infinite(self):
        """Negative infinite values should raise."""
        returns = pl.Series([0.01, float("-inf"), -0.02])

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="infinite"):
            validator.check_finite()

    def test_check_nulls_detects_nulls(self):
        """Null values should raise when not allowed."""
        returns = pl.Series([0.01, None, -0.02])

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="null"):
            validator.check_nulls(allow_nulls=False)

    def test_check_distribution_detects_extreme_skew(self):
        """Extreme skewness should raise."""
        # Create highly skewed returns
        np.random.seed(42)
        returns = pl.Series(
            np.concatenate(
                [
                    np.random.randn(100) * 0.01,  # Normal returns
                    [0.5, 0.6, 0.7, 0.8, 0.9],  # Large positive outliers
                ]
            )
        )

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="skewness"):
            validator.check_distribution(max_abs_skew=1.0)

    def test_check_distribution_detects_extreme_kurtosis(self):
        """Extreme kurtosis should raise."""
        # Create high-kurtosis returns
        np.random.seed(42)
        returns = pl.Series(
            np.concatenate(
                [
                    np.random.randn(100) * 0.01,
                    [0.5, -0.5, 0.6, -0.6],  # Fat tails
                ]
            )
        )

        validator = ReturnsValidator(returns)

        with pytest.raises(ValidationError, match="kurtosis"):
            validator.check_distribution(max_abs_kurtosis=5.0)

    def test_dataframe_input_with_column(self):
        """DataFrame input should work with column specification."""
        df = pl.DataFrame(
            {
                "returns": [0.01, -0.02, 0.03],
                "other": ["a", "b", "c"],
            }
        )

        validator = ReturnsValidator(df, column="returns")
        result = validator.check_numeric()

        assert result is validator

    def test_dataframe_input_requires_column(self):
        """DataFrame input without column should raise."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.03]})

        with pytest.raises(ValueError, match="column required"):
            ReturnsValidator(df)


class TestValidateReturnsFunction:
    """Tests for the validate_returns convenience function."""

    def test_validates_all_conditions(self):
        """Should check all specified conditions."""
        returns = pl.Series([0.01, -0.02, 0.03, -0.01])

        # Should not raise
        validate_returns(
            returns,
            bounds=(-0.5, 0.5),
            allow_nulls=False,
            check_finite=True,
        )

    def test_detects_issues(self):
        """Should detect validation issues."""
        returns = pl.Series([0.01, 1.5, -0.02])  # 150% return out of bounds

        with pytest.raises(ValidationError):
            validate_returns(returns, bounds=(-0.5, 0.5))


class TestValidateBoundsFunction:
    """Tests for the validate_bounds convenience function."""

    def test_validates_bounds_correctly(self):
        """Should validate bounds correctly."""
        returns = pl.Series([0.01, -0.02, 0.03])

        # Should not raise
        validate_bounds(returns, lower=-0.1, upper=0.1)

    def test_detects_bound_violations(self):
        """Should detect bound violations."""
        returns = pl.Series([0.01, 0.5, -0.02])  # 50% exceeds 10%

        with pytest.raises(ValidationError, match="upper bound"):
            validate_bounds(returns, lower=-0.1, upper=0.1)


class TestValidationErrorContext:
    """Tests verifying ValidationError provides useful context."""

    def test_error_includes_context(self):
        """Errors should include helpful context dict."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        try:
            DataFrameValidator(df).require_columns(["a", "b", "c"])
        except ValidationError as e:
            assert "missing" in e.context
            assert e.context["missing"] == ["b", "c"]

    def test_error_message_formatted_with_context(self):
        """Error message should include formatted context."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        try:
            DataFrameValidator(df).require_columns(["a", "b"])
        except ValidationError as e:
            message = str(e)
            assert "Context:" in message
            assert "missing" in message


class TestEdgeCases:
    """Edge cases for validation functions."""

    def test_empty_column_list(self):
        """Empty column list should pass validation."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        validator = DataFrameValidator(df)
        result = validator.require_columns([])

        assert result is validator

    def test_single_row_dataframe(self):
        """Single-row DataFrames should validate correctly."""
        df = pl.DataFrame({"a": [1], "b": [2.0]})

        validate_dataframe(
            df,
            required_columns=["a", "b"],
            numeric_columns=["a", "b"],
            min_rows=1,
        )

    def test_all_null_returns_validation(self):
        """All-null returns should be detected as non-numeric (Null dtype)."""
        returns = pl.Series([None, None, None])

        # All-null series has dtype "Null" which is not numeric
        # This is detected as a numeric validation failure
        with pytest.raises(ValidationError, match="numeric"):
            validate_returns(returns, allow_nulls=False)


class TestValidateIndexFunction:
    """Tests for the validate_index convenience function."""

    def test_validates_index_exists_and_type(self):
        """Should validate index exists and is temporal."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "value": [1, 2],
            }
        )

        # Should not raise
        validate_index(df, index_col="date")

    def test_detects_missing_index(self):
        """Should detect missing index column."""
        df = pl.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(ValidationError, match="not found"):
            validate_index(df, index_col="date")

    def test_detects_non_temporal_index(self):
        """Should detect non-temporal index."""
        df = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],  # String, not Date
                "value": [1, 2],
            }
        )

        with pytest.raises(ValidationError, match="temporal"):
            validate_index(df, index_col="date")


class TestValidateFrequencyFunction:
    """Tests for the validate_frequency function."""

    def test_validates_consistent_daily_frequency(self):
        """Consistent daily frequency should pass."""
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 4),
                    datetime(2024, 1, 5),
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        # Should not raise
        validate_frequency(df, index_col="date", expected_freq="daily")

    def test_detects_inconsistent_frequency(self):
        """Inconsistent spacing should raise ValidationError."""
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),  # 1 day gap
                    datetime(2024, 1, 10),  # 8 day gap - inconsistent!
                    datetime(2024, 1, 11),  # 1 day gap
                    datetime(2024, 1, 12),  # 1 day gap
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        with pytest.raises(ValidationError, match="Inconsistent"):
            validate_frequency(df, index_col="date", expected_freq="daily")

    def test_validates_consistent_hourly_frequency(self):
        """Consistent hourly frequency should pass."""
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1, 0, 0),
                    datetime(2024, 1, 1, 1, 0),
                    datetime(2024, 1, 1, 2, 0),
                    datetime(2024, 1, 1, 3, 0),
                ],
                "value": [1, 2, 3, 4],
            }
        )

        # Should not raise
        validate_frequency(df, index_col="date", expected_freq="hourly")

    def test_no_expected_freq_skips_validation(self):
        """When expected_freq is None, should not check consistency."""
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 5),  # 4 day gap
                    datetime(2024, 1, 6),  # 1 day gap - inconsistent
                ],
                "value": [1, 2, 3],
            }
        )

        # Should not raise when expected_freq=None
        validate_frequency(df, index_col="date", expected_freq=None)

    def test_single_row_passes(self):
        """Single row DataFrame should pass frequency validation."""
        df = pl.DataFrame(
            {
                "date": [datetime(2024, 1, 1)],
                "value": [1],
            }
        )

        # Should not raise (no gaps to check)
        validate_frequency(df, index_col="date", expected_freq="daily")


class TestTimeSeriesValidatorGaps:
    """Additional tests for gap checking."""

    def test_check_gaps_none_threshold_skips_check(self):
        """None threshold should skip gap checking."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 3, 1)],  # 60 day gap
                "value": [1, 2],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_gaps(max_gap_days=None)

        assert result is validator

    def test_check_gaps_empty_dataframe(self):
        """Empty dataframe should pass gap check."""
        df = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "value": pl.Series([], dtype=pl.Float64),
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_gaps(max_gap_days=7)

        assert result is validator

    def test_check_gaps_single_row(self):
        """Single row should pass gap check (no gaps)."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "value": [1],
            }
        )

        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_gaps(max_gap_days=1)

        assert result is validator


class TestReturnsValidatorDistributionEdgeCases:
    """Edge cases for distribution checking."""

    def test_check_distribution_insufficient_data(self):
        """Insufficient data (<30) should skip distribution checks."""
        returns = pl.Series([0.01, -0.02, 0.03] * 5)  # Only 15 values

        validator = ReturnsValidator(returns)
        # Should not raise even with strict thresholds
        result = validator.check_distribution(max_abs_skew=0.1, max_abs_kurtosis=0.1)

        assert result is validator

    def test_check_distribution_zero_std(self):
        """Zero standard deviation should skip distribution checks."""
        returns = pl.Series([0.01] * 50)  # Constant returns, std=0

        validator = ReturnsValidator(returns)
        # Should not raise (std=0 guard)
        result = validator.check_distribution(max_abs_skew=0.1, max_abs_kurtosis=0.1)

        assert result is validator


class TestReturnsValidatorBoundsEdgeCases:
    """Edge cases for bounds checking."""

    def test_check_bounds_all_nulls(self):
        """All null values should pass bounds check (no clean values)."""
        returns = pl.Series([None, None, None], dtype=pl.Float64)

        validator = ReturnsValidator(returns)
        result = validator.check_bounds(lower=-0.5, upper=0.5)

        assert result is validator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
