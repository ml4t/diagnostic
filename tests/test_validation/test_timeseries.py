"""Tests for Time Series validation utilities."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from ml4t.diagnostic.validation.dataframe import ValidationError
from ml4t.diagnostic.validation.timeseries import (
    TimeSeriesValidator,
    validate_frequency,
    validate_index,
    validate_timeseries,
)


class TestTimeSeriesValidator:
    """Tests for TimeSeriesValidator class."""

    @pytest.fixture
    def sample_ts_df(self):
        """Create sample time series DataFrame."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        return pl.DataFrame(
            {
                "date": dates,
                "value": [100.0, 101.0, 99.0, 102.0, 103.0],
            }
        )

    @pytest.fixture
    def ts_df_datetime(self):
        """Create time series with datetime index."""
        datetimes = [datetime(2024, 1, 1, i) for i in range(5)]
        return pl.DataFrame(
            {
                "timestamp": datetimes,
                "value": [100.0, 101.0, 99.0, 102.0, 103.0],
            }
        )

    @pytest.fixture
    def unsorted_ts_df(self):
        """Create unsorted time series DataFrame."""
        dates = [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 5)]
        return pl.DataFrame(
            {
                "date": dates,
                "value": [99.0, 100.0, 103.0],
            }
        )

    @pytest.fixture
    def ts_with_duplicates(self):
        """Create time series with duplicate timestamps."""
        dates = [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)]
        return pl.DataFrame(
            {
                "date": dates,
                "value": [100.0, 100.5, 101.0],
            }
        )

    @pytest.fixture
    def ts_with_gaps(self):
        """Create time series with gaps."""
        dates = [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 16)]
        return pl.DataFrame(
            {
                "date": dates,
                "value": [100.0, 101.0, 102.0],
            }
        )

    def test_check_index_exists_success(self, sample_ts_df):
        """Test check_index_exists with existing column."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_index_exists()
        assert result is validator

    def test_check_index_exists_missing(self, sample_ts_df):
        """Test check_index_exists with missing column."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="missing")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_index_exists()

        assert "missing" in str(exc_info.value).lower()
        assert "available_columns" in exc_info.value.context

    def test_check_index_type_date(self, sample_ts_df):
        """Test check_index_type with Date type."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_index_type()
        assert result is validator

    def test_check_index_type_datetime(self, ts_df_datetime):
        """Test check_index_type with Datetime type."""
        validator = TimeSeriesValidator(ts_df_datetime, index_col="timestamp")
        result = validator.check_index_type()
        assert result is validator

    def test_check_index_type_non_temporal(self):
        """Test check_index_type with non-temporal column."""
        df = pl.DataFrame({"date": [1, 2, 3], "value": [100.0, 101.0, 102.0]})
        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_index_type()

        assert "temporal" in str(exc_info.value).lower()

    def test_check_sorted_ascending_success(self, sample_ts_df):
        """Test check_sorted with sorted data."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_sorted(ascending=True)
        assert result is validator

    def test_check_sorted_ascending_failure(self, unsorted_ts_df):
        """Test check_sorted with unsorted data."""
        validator = TimeSeriesValidator(unsorted_ts_df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_sorted(ascending=True)

        assert "sorted" in str(exc_info.value).lower()
        assert "ascending" in str(exc_info.value).lower()

    def test_check_sorted_descending(self):
        """Test check_sorted with descending order."""
        dates = [date(2024, 1, 5), date(2024, 1, 3), date(2024, 1, 1)]
        df = pl.DataFrame({"date": dates, "value": [103.0, 102.0, 100.0]})
        validator = TimeSeriesValidator(df, index_col="date")

        result = validator.check_sorted(ascending=False)
        assert result is validator

    def test_check_duplicates_success(self, sample_ts_df):
        """Test check_duplicates with no duplicates."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_duplicates()
        assert result is validator

    def test_check_duplicates_failure(self, ts_with_duplicates):
        """Test check_duplicates with duplicates."""
        validator = TimeSeriesValidator(ts_with_duplicates, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_duplicates()

        assert "duplicate" in str(exc_info.value).lower()
        assert exc_info.value.context["duplicate_count"] == 2

    def test_check_gaps_success(self, sample_ts_df):
        """Test check_gaps with no large gaps."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_gaps(max_gap_days=2)
        assert result is validator

    def test_check_gaps_failure(self, ts_with_gaps):
        """Test check_gaps with large gaps."""
        validator = TimeSeriesValidator(ts_with_gaps, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_gaps(max_gap_days=7)

        assert "gap" in str(exc_info.value).lower()
        assert exc_info.value.context["max_allowed"] == 7

    def test_check_gaps_none_skips(self, ts_with_gaps):
        """Test check_gaps with None skips check."""
        validator = TimeSeriesValidator(ts_with_gaps, index_col="date")
        result = validator.check_gaps(max_gap_days=None)
        assert result is validator

    def test_check_gaps_empty_series(self):
        """Test check_gaps with single row."""
        df = pl.DataFrame({"date": [date(2024, 1, 1)], "value": [100.0]})
        validator = TimeSeriesValidator(df, index_col="date")

        # Should not raise - no gaps to check
        result = validator.check_gaps(max_gap_days=1)
        assert result is validator

    def test_chaining(self, sample_ts_df):
        """Test method chaining."""
        validator = TimeSeriesValidator(sample_ts_df, index_col="date")
        result = validator.check_index_exists().check_index_type().check_sorted().check_duplicates()
        assert result is validator


class TestValidateTimeseries:
    """Tests for validate_timeseries function."""

    @pytest.fixture
    def sample_ts_df(self):
        """Create sample time series DataFrame."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        return pl.DataFrame(
            {
                "date": dates,
                "value": [100.0, 101.0, 99.0, 102.0, 103.0],
            }
        )

    def test_basic_validation(self, sample_ts_df):
        """Test basic validation passes."""
        validate_timeseries(sample_ts_df, index_col="date")

    def test_custom_index_column(self):
        """Test with custom index column name."""
        dates = [date(2024, 1, i) for i in range(1, 4)]
        df = pl.DataFrame({"timestamp": dates, "value": [100.0, 101.0, 102.0]})
        validate_timeseries(df, index_col="timestamp")

    def test_require_sorted_true(self):
        """Test with require_sorted=True on unsorted data."""
        dates = [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)]
        df = pl.DataFrame({"date": dates, "value": [102.0, 100.0, 101.0]})

        with pytest.raises(ValidationError):
            validate_timeseries(df, index_col="date", require_sorted=True)

    def test_require_sorted_false(self):
        """Test with require_sorted=False on unsorted data."""
        dates = [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)]
        df = pl.DataFrame({"date": dates, "value": [102.0, 100.0, 101.0]})

        # Should not raise
        validate_timeseries(df, index_col="date", require_sorted=False)

    def test_check_duplicates_true(self):
        """Test with check_duplicates=True on duplicate data."""
        dates = [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)]
        df = pl.DataFrame({"date": dates, "value": [100.0, 100.5, 101.0]})

        with pytest.raises(ValidationError):
            validate_timeseries(df, index_col="date", check_duplicates=True)

    def test_check_duplicates_false(self):
        """Test with check_duplicates=False on duplicate data."""
        dates = [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)]
        df = pl.DataFrame({"date": dates, "value": [100.0, 100.5, 101.0]})

        # Should not raise
        validate_timeseries(df, index_col="date", check_duplicates=False)

    def test_max_gap_days(self):
        """Test max_gap_days parameter."""
        dates = [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 16)]
        df = pl.DataFrame({"date": dates, "value": [100.0, 101.0, 102.0]})

        with pytest.raises(ValidationError):
            validate_timeseries(df, index_col="date", max_gap_days=7)


class TestValidateIndex:
    """Tests for validate_index function."""

    def test_valid_index(self):
        """Test valid index."""
        dates = [date(2024, 1, i) for i in range(1, 4)]
        df = pl.DataFrame({"date": dates, "value": [100.0, 101.0, 102.0]})
        validate_index(df, index_col="date")

    def test_missing_index(self):
        """Test missing index column."""
        df = pl.DataFrame({"value": [100.0, 101.0, 102.0]})

        with pytest.raises(ValidationError):
            validate_index(df, index_col="date")

    def test_non_temporal_index(self):
        """Test non-temporal index column."""
        df = pl.DataFrame({"date": [1, 2, 3], "value": [100.0, 101.0, 102.0]})

        with pytest.raises(ValidationError):
            validate_index(df, index_col="date")


class TestValidateFrequency:
    """Tests for validate_frequency function."""

    def test_consistent_frequency(self):
        """Test consistent daily frequency using datetime."""
        # Use datetime for consistent gap calculations
        datetimes = [datetime(2024, 1, 1, i) for i in range(10)]
        df = pl.DataFrame({"date": datetimes, "value": list(range(10))})

        # Should not raise - consistent spacing (1 hour gaps)
        validate_frequency(df, index_col="date", expected_freq="hourly")

    def test_inconsistent_frequency(self):
        """Test inconsistent frequency using datetime."""
        # Mix of 1-hour and 10-hour gaps
        datetimes = [
            datetime(2024, 1, 1, 0),
            datetime(2024, 1, 1, 1),
            datetime(2024, 1, 1, 11),
            datetime(2024, 1, 1, 12),
        ]
        df = pl.DataFrame({"date": datetimes, "value": [100.0, 101.0, 102.0, 103.0]})

        with pytest.raises(ValidationError) as exc_info:
            validate_frequency(df, index_col="date", expected_freq="hourly")

        assert "inconsistent" in str(exc_info.value).lower()

    def test_no_expected_freq_skips_check(self):
        """Test that None expected_freq skips check."""
        datetimes = [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 10)]
        df = pl.DataFrame({"date": datetimes, "value": [100.0, 101.0, 102.0]})

        # Should not raise
        validate_frequency(df, index_col="date", expected_freq=None)

    def test_single_row_skips_check(self):
        """Test that single row skips frequency check."""
        df = pl.DataFrame({"date": [datetime(2024, 1, 1, 0)], "value": [100.0]})

        # Should not raise - no gaps to check
        validate_frequency(df, index_col="date", expected_freq="hourly")
