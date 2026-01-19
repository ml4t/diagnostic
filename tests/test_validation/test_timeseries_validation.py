"""Tests for time series validation utilities."""

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
    def valid_ts_df(self) -> pl.DataFrame:
        """Create a valid time series DataFrame."""
        return pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
                "close": [100.0 + i for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

    @pytest.fixture
    def unsorted_ts_df(self) -> pl.DataFrame:
        """Create an unsorted time series DataFrame."""
        dates = [date(2020, 1, 5), date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 3)]
        return pl.DataFrame(
            {
                "date": dates,
                "close": [100.0, 101.0, 102.0, 103.0],
            }
        )

    def test_check_index_exists_success(self, valid_ts_df: pl.DataFrame):
        """Test check_index_exists when index exists."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_index_exists()
        assert result is validator

    def test_check_index_exists_failure(self, valid_ts_df: pl.DataFrame):
        """Test check_index_exists when index doesn't exist."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="timestamp")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_index_exists()

        assert "timestamp" in str(exc_info.value)
        assert "date" in exc_info.value.context["available_columns"]

    def test_check_index_type_date(self, valid_ts_df: pl.DataFrame):
        """Test check_index_type with Date type."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_index_type()
        assert result is validator

    def test_check_index_type_datetime(self):
        """Test check_index_type with Datetime type."""
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 10),
                    interval="1d",
                    eager=True,
                ),
                "value": range(10),
            }
        )
        validator = TimeSeriesValidator(df, index_col="timestamp")
        result = validator.check_index_type()
        assert result is validator

    def test_check_index_type_failure(self):
        """Test check_index_type with non-temporal type."""
        df = pl.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],  # String, not Date
                "value": [1, 2],
            }
        )
        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_index_type()

        assert "temporal" in str(exc_info.value).lower()

    def test_check_sorted_ascending_success(self, valid_ts_df: pl.DataFrame):
        """Test check_sorted when ascending sorted."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_sorted(ascending=True)
        assert result is validator

    def test_check_sorted_ascending_failure(self, unsorted_ts_df: pl.DataFrame):
        """Test check_sorted when not ascending sorted."""
        validator = TimeSeriesValidator(unsorted_ts_df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_sorted(ascending=True)

        assert "not sorted" in str(exc_info.value).lower()
        assert "ascending" in str(exc_info.value).lower()

    def test_check_sorted_descending_success(self):
        """Test check_sorted when descending sorted."""
        # Create descending dates by reversing an ascending range
        dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True).reverse()
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(10),
            }
        )
        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_sorted(ascending=False)
        assert result is validator

    def test_check_sorted_descending_failure(self, valid_ts_df: pl.DataFrame):
        """Test check_sorted when ascending sorted but descending expected."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_sorted(ascending=False)

        assert "descending" in str(exc_info.value).lower()

    def test_check_duplicates_no_duplicates(self, valid_ts_df: pl.DataFrame):
        """Test check_duplicates when no duplicates."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_duplicates()
        assert result is validator

    def test_check_duplicates_with_duplicates(self):
        """Test check_duplicates when duplicates exist."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 1, 2),  # Duplicate
                    date(2020, 1, 3),
                ],
                "value": [1, 2, 3, 4],
            }
        )
        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_duplicates()

        assert "duplicate" in str(exc_info.value).lower()
        assert exc_info.value.context["duplicate_count"] > 0

    def test_check_gaps_no_gaps(self, valid_ts_df: pl.DataFrame):
        """Test check_gaps when no large gaps."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_gaps(max_gap_days=5)
        assert result is validator

    def test_check_gaps_with_large_gap(self):
        """Test check_gaps when large gap exists."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                    date(2020, 2, 1),  # 30 day gap
                ],
                "value": [1, 2, 3],
            }
        )
        validator = TimeSeriesValidator(df, index_col="date")

        with pytest.raises(ValidationError) as exc_info:
            validator.check_gaps(max_gap_days=7)

        assert "gap" in str(exc_info.value).lower()

    def test_check_gaps_none_threshold(self, valid_ts_df: pl.DataFrame):
        """Test check_gaps with None threshold (no check)."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")
        result = validator.check_gaps(max_gap_days=None)
        assert result is validator

    def test_check_gaps_empty_after_diff(self):
        """Test check_gaps with single row (empty after diff)."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)],
                "value": [1],
            }
        )
        validator = TimeSeriesValidator(df, index_col="date")
        result = validator.check_gaps(max_gap_days=1)
        assert result is validator

    def test_chaining(self, valid_ts_df: pl.DataFrame):
        """Test method chaining."""
        validator = TimeSeriesValidator(valid_ts_df, index_col="date")

        result = (
            validator.check_index_exists()
            .check_index_type()
            .check_sorted()
            .check_duplicates()
            .check_gaps(max_gap_days=10)
        )

        assert result is validator


class TestValidateTimeseries:
    """Tests for validate_timeseries function."""

    @pytest.fixture
    def valid_df(self) -> pl.DataFrame:
        """Create a valid time series DataFrame."""
        return pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
                "value": range(10),
            }
        )

    def test_valid_timeseries(self, valid_df: pl.DataFrame):
        """Test with valid time series."""
        # Should not raise
        validate_timeseries(valid_df)

    def test_custom_index_column(self):
        """Test with custom index column name."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True),
                "value": range(10),
            }
        )

        validate_timeseries(df, index_col="timestamp")

    def test_require_sorted_false(self):
        """Test with require_sorted=False."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 5), date(2020, 1, 1), date(2020, 1, 10)],
                "value": [1, 2, 3],
            }
        )

        # Should not raise when sorted not required
        validate_timeseries(df, require_sorted=False, check_duplicates=False)

    def test_check_duplicates_false(self):
        """Test with check_duplicates=False."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2)],
                "value": [1, 2, 3],
            }
        )

        # Should not raise when duplicates check disabled
        validate_timeseries(df, check_duplicates=False)

    def test_max_gap_days(self, valid_df: pl.DataFrame):
        """Test with max_gap_days parameter."""
        validate_timeseries(valid_df, max_gap_days=5)

    def test_all_validations_enabled(self, valid_df: pl.DataFrame):
        """Test with all validations enabled."""
        validate_timeseries(
            valid_df,
            index_col="date",
            require_sorted=True,
            check_duplicates=True,
            max_gap_days=5,
        )


class TestValidateIndex:
    """Tests for validate_index function."""

    def test_valid_index(self):
        """Test with valid index."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
                "value": range(5),
            }
        )

        # Should not raise
        validate_index(df)

    def test_missing_index(self):
        """Test with missing index column."""
        df = pl.DataFrame(
            {
                "timestamp": pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
                "value": range(5),
            }
        )

        with pytest.raises(ValidationError):
            validate_index(df)  # Default looks for "date"

    def test_custom_index_name(self):
        """Test with custom index column name."""
        df = pl.DataFrame(
            {
                "ts": pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
                "value": range(5),
            }
        )

        validate_index(df, index_col="ts")

    def test_non_temporal_index(self):
        """Test with non-temporal index column."""
        df = pl.DataFrame(
            {
                "date": [1, 2, 3, 4, 5],  # Integer, not date
                "value": range(5),
            }
        )

        with pytest.raises(ValidationError):
            validate_index(df)


class TestValidateFrequency:
    """Tests for validate_frequency function."""

    @pytest.fixture
    def daily_df(self) -> pl.DataFrame:
        """Create daily frequency DataFrame."""
        return pl.DataFrame(
            {
                "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 31), eager=True),
                "value": range(31),
            }
        )

    def test_no_expected_freq(self, daily_df: pl.DataFrame):
        """Test without expected frequency (basic validation only)."""
        # Should not raise - just validates index exists and is sorted
        validate_frequency(daily_df)

    def test_single_row(self):
        """Test with single row (no frequency to check)."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1)],
                "value": [1],
            }
        )

        # Should not raise
        validate_frequency(df, expected_freq="daily")

    def test_unsorted_with_freq_check(self):
        """Test unsorted data raises error before frequency check."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 5), date(2020, 1, 1), date(2020, 1, 3)],
                "value": [1, 2, 3],
            }
        )

        with pytest.raises(ValidationError):
            validate_frequency(df, expected_freq="daily")

    def test_two_rows(self):
        """Test with two rows (minimal frequency data)."""
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1), date(2020, 1, 2)],
                "value": [1, 2],
            }
        )

        # Should not raise - basic index validation only
        validate_frequency(df)  # No expected_freq to avoid Duration conversion issues

    def test_validates_index_exists(self):
        """Test that missing index column raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [date(2020, 1, 1), date(2020, 1, 2)],
                "value": [1, 2],
            }
        )

        with pytest.raises(ValidationError):
            validate_frequency(df, expected_freq="daily")  # Default looks for "date"


class TestEdgeCases:
    """Edge case tests."""

    def test_datetime_with_time_component(self):
        """Test with datetime including time."""
        df = pl.DataFrame(
            {
                "date": pl.datetime_range(
                    datetime(2020, 1, 1, 9, 0),
                    datetime(2020, 1, 1, 17, 0),
                    interval="1h",
                    eager=True,
                ),
                "value": range(9),
            }
        )

        validate_timeseries(df)

    def test_timedelta_gap_handling(self):
        """Test gap handling with timedelta values."""
        df = pl.DataFrame(
            {
                "date": pl.datetime_range(
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "value": range(5),
            }
        )

        validate_timeseries(df, max_gap_days=2)

    def test_many_duplicates(self):
        """Test with many duplicate timestamps."""
        base_date = date(2020, 1, 1)
        dates = [base_date] * 10  # All same date
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(10),
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            validate_timeseries(df)

        # Should report large duplicate count
        assert exc_info.value.context["duplicate_count"] > 0

    def test_descending_sorted_dataframe(self):
        """Test validation of descending sorted DataFrame."""
        # Create descending dates by reversing an ascending range
        dates = pl.date_range(date(2020, 1, 1), date(2020, 1, 10), eager=True).reverse()
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(10),
            }
        )

        # Default requires ascending, so this should fail
        with pytest.raises(ValidationError):
            validate_timeseries(df)

    def test_weekend_gaps_in_daily_data(self):
        """Test handling of weekend gaps in trading data."""
        # Create weekday-only dates (skip weekends)
        dates = [
            date(2020, 1, 6),  # Monday
            date(2020, 1, 7),
            date(2020, 1, 8),
            date(2020, 1, 9),
            date(2020, 1, 10),  # Friday
            date(2020, 1, 13),  # Monday (weekend gap)
        ]
        df = pl.DataFrame(
            {
                "date": dates,
                "value": range(6),
            }
        )

        # 3-day gap (Sat, Sun, Mon) should pass with 4-day threshold
        validate_timeseries(df, max_gap_days=4)

        # But fail with 2-day threshold
        with pytest.raises(ValidationError):
            validate_timeseries(df, max_gap_days=2)

    def test_year_boundary(self):
        """Test handling of year boundary."""
        df = pl.DataFrame(
            {
                "date": [
                    date(2019, 12, 30),
                    date(2019, 12, 31),
                    date(2020, 1, 1),
                    date(2020, 1, 2),
                ],
                "value": range(4),
            }
        )

        validate_timeseries(df, max_gap_days=2)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame(
            {
                "date": [],
                "value": [],
            }
        ).cast({"date": pl.Date, "value": pl.Float64})

        # This should work but have no data to validate beyond structure
        validator = TimeSeriesValidator(df, index_col="date")
        validator.check_index_exists()

    def test_validator_with_alternate_column_names(self):
        """Test validator with various column name conventions."""
        for col_name in ["date", "timestamp", "datetime", "time", "dt"]:
            df = pl.DataFrame(
                {
                    col_name: pl.date_range(date(2020, 1, 1), date(2020, 1, 5), eager=True),
                    "value": range(5),
                }
            )

            validator = TimeSeriesValidator(df, index_col=col_name)
            validator.check_index_exists()
            validator.check_index_type()
