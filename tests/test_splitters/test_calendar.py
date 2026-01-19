"""Tests for calendar-aware time parsing for financial data cross-validation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ml4t.diagnostic.splitters.calendar import (
    TradingCalendar,
    _parse_time_size_naive,
    parse_time_size_calendar_aware,
)
from ml4t.diagnostic.splitters.calendar_config import CalendarConfig


class TestTradingCalendar:
    """Tests for TradingCalendar class."""

    def test_init_with_string(self):
        """Test initialization with exchange name string."""
        cal = TradingCalendar("NYSE")

        assert cal.config.exchange == "NYSE"
        assert cal.calendar is not None

    def test_init_with_config(self):
        """Test initialization with CalendarConfig."""
        config = CalendarConfig(
            exchange="CME_Equity",
            timezone="America/Chicago",
            localize_naive=True,
        )
        cal = TradingCalendar(config)

        assert cal.config.exchange == "CME_Equity"
        assert cal.config.timezone == "America/Chicago"

    def test_init_default_exchange(self):
        """Test initialization with default CME_Equity exchange."""
        cal = TradingCalendar()

        assert cal.config.exchange == "CME_Equity"

    def test_ensure_timezone_aware_naive(self):
        """Test timezone handling for naive timestamps."""
        cal = TradingCalendar("NYSE")

        # Create naive timestamps
        timestamps = pd.DatetimeIndex(
            [
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 11, 0),
            ]
        )

        result = cal._ensure_timezone_aware(timestamps)

        assert result.tz is not None

    def test_ensure_timezone_aware_already_aware(self):
        """Test timezone handling for already aware timestamps."""
        cal = TradingCalendar("NYSE")

        # Create tz-aware timestamps
        timestamps = pd.date_range("2024-01-02 10:00", periods=3, freq="1h", tz="America/New_York")

        result = cal._ensure_timezone_aware(timestamps)

        assert result.tz is not None

    def test_ensure_timezone_aware_reject_naive(self):
        """Test rejection of naive timestamps when localize_naive=False."""
        config = CalendarConfig(
            exchange="NYSE",
            timezone="UTC",
            localize_naive=False,
        )
        cal = TradingCalendar(config)

        timestamps = pd.DatetimeIndex([datetime(2024, 1, 2, 10, 0)])

        with pytest.raises(ValueError, match="timezone-naive"):
            cal._ensure_timezone_aware(timestamps)

    def test_get_sessions(self):
        """Test session assignment for timestamps."""
        cal = TradingCalendar("NYSE")

        # Create timestamps during a trading session
        timestamps = pd.date_range(
            "2024-01-02 10:00",
            periods=10,
            freq="30min",
            tz="America/New_York",
        )

        sessions = cal.get_sessions(timestamps)

        # All timestamps from same day should have same session date
        assert len(sessions.unique()) == 1

    def test_get_sessions_multiple_days(self):
        """Test session assignment across multiple days."""
        cal = TradingCalendar("NYSE")

        # Create timestamps across multiple days
        timestamps = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-02 10:00", tz="America/New_York"),
                pd.Timestamp("2024-01-02 14:00", tz="America/New_York"),
                pd.Timestamp("2024-01-03 10:00", tz="America/New_York"),
                pd.Timestamp("2024-01-03 14:00", tz="America/New_York"),
            ]
        )

        sessions = cal.get_sessions(timestamps)

        # Should have 2 unique sessions
        assert len(sessions.unique()) == 2

    def test_count_samples_in_period_daily(self):
        """Test sample counting by daily periods."""
        cal = TradingCalendar("NYSE")

        # Create intraday timestamps
        timestamps = pd.date_range(
            "2024-01-02",
            periods=100,
            freq="1h",
            tz="America/New_York",
        )

        counts = cal.count_samples_in_period(timestamps, "1D")

        assert len(counts) > 0
        assert all(c > 0 for c in counts)

    def test_count_samples_in_period_invalid_spec(self):
        """Test error for invalid period specification."""
        cal = TradingCalendar("NYSE")

        timestamps = pd.date_range(
            "2024-01-02",
            periods=100,
            freq="1h",
            tz="America/New_York",
        )

        with pytest.raises(ValueError, match="Invalid period specification"):
            cal.count_samples_in_period(timestamps, "invalid")


class TestParseTimeSizeCalendarAware:
    """Tests for parse_time_size_calendar_aware function."""

    def test_with_calendar(self):
        """Test time size parsing with calendar."""
        cal = TradingCalendar("NYSE")

        timestamps = pd.date_range(
            "2024-01-02",
            periods=1000,
            freq="1h",
            tz="America/New_York",
        )

        n_samples = parse_time_size_calendar_aware("1D", timestamps, cal)

        assert n_samples > 0
        assert isinstance(n_samples, int)

    def test_without_calendar_fallback(self):
        """Test fallback to naive calculation without calendar."""
        timestamps = pd.date_range("2024-01-01", periods=1000, freq="1h")

        n_samples = parse_time_size_calendar_aware("1D", timestamps, calendar=None)

        assert n_samples > 0
        assert isinstance(n_samples, int)


class TestParseTimeSizeNaive:
    """Tests for _parse_time_size_naive function."""

    def test_days_spec(self):
        """Test parsing with days specification."""
        timestamps = pd.date_range("2024-01-01", periods=1000, freq="1h")

        n_samples = _parse_time_size_naive("1D", timestamps)

        # 1 day = 24 hours with hourly data
        assert n_samples == pytest.approx(24, rel=0.1)

    def test_weeks_spec(self):
        """Test parsing with weeks specification."""
        timestamps = pd.date_range("2024-01-01", periods=1000, freq="1h")

        n_samples = _parse_time_size_naive("1W", timestamps)

        # 1 week = 7 days = 168 hours
        assert n_samples == pytest.approx(168, rel=0.1)

    def test_single_timestamp_error(self):
        """Test error for single timestamp."""
        timestamps = pd.DatetimeIndex([datetime(2024, 1, 1)])

        with pytest.raises(ValueError, match="single-timestamp"):
            _parse_time_size_naive("1D", timestamps)

    def test_invalid_spec_error(self):
        """Test error for invalid specification."""
        timestamps = pd.date_range("2024-01-01", periods=100, freq="1h")

        with pytest.raises(ValueError, match="Invalid time specification"):
            _parse_time_size_naive("invalid", timestamps)

    def test_months_spec(self):
        """Test parsing with months specification."""
        timestamps = pd.date_range("2024-01-01", periods=365, freq="1D")

        n_samples = _parse_time_size_naive("1M", timestamps)

        # ~30 days per month
        assert 25 <= n_samples <= 35


class TestCalendarWeeklyPeriods:
    """Tests for weekly period handling in calendar module."""

    def test_count_samples_weekly_intraday(self):
        """Test weekly sample counting for intraday data."""
        cal = TradingCalendar("NYSE")

        # Create 6 weeks of intraday data (more than enough for complete 4W periods)
        timestamps = pd.date_range(
            "2024-01-02",
            periods=2000,
            freq="1h",
            tz="America/New_York",
        )

        counts = cal.count_samples_in_period(timestamps, "1W")

        # Should have at least some complete weeks
        assert len(counts) > 0
        # Each week should have significant samples
        for count in counts:
            assert count > 0

    def test_count_samples_4w_intraday(self):
        """Test 4-week sample counting for intraday data."""
        cal = TradingCalendar("NYSE")

        # Create 12 weeks of intraday data
        timestamps = pd.date_range(
            "2024-01-02",
            periods=3000,
            freq="1h",
            tz="America/New_York",
        )

        counts = cal.count_samples_in_period(timestamps, "4W")

        # Should have complete 4-week blocks
        assert len(counts) >= 1


class TestCountSamplesByCalendar:
    """Tests for _count_samples_by_calendar method."""

    def test_daily_data_daily_period(self):
        """Test daily period counting with daily data."""
        cal = TradingCalendar("NYSE")

        # Create daily data (NOT intraday)
        timestamps = pd.date_range(
            "2024-01-02",
            periods=30,
            freq="1D",
            tz="America/New_York",
        )

        # For daily data, it should use calendar periods
        counts = cal.count_samples_in_period(timestamps, "1D")

        # Each day should have 1 sample
        assert len(counts) == 30
        assert all(c == 1 for c in counts)

    def test_daily_data_weekly_period(self):
        """Test weekly period counting with daily data."""
        cal = TradingCalendar("NYSE")

        # Create 4+ weeks of daily data
        timestamps = pd.date_range(
            "2024-01-02",
            periods=30,
            freq="1D",
            tz="America/New_York",
        )

        counts = cal.count_samples_in_period(timestamps, "1W")

        # Should have complete weeks with 7 or fewer samples
        assert len(counts) >= 3  # At least 3 weeks

    def test_daily_data_monthly_period(self):
        """Test monthly period counting with daily data."""
        cal = TradingCalendar("NYSE")

        # Create 3 months of daily data
        timestamps = pd.date_range(
            "2024-01-01",
            periods=90,
            freq="1D",
            tz="America/New_York",
        )

        counts = cal.count_samples_in_period(timestamps, "1M")

        # Should have complete months
        assert len(counts) >= 2
        # Each month should have ~30 samples
        for count in counts:
            assert 28 <= count <= 31


class TestParseTimeSizeCalendarAwareEdgeCases:
    """Edge case tests for parse_time_size_calendar_aware."""

    def test_empty_sample_counts_raises_error(self):
        """Test error when no complete periods are found."""
        cal = TradingCalendar("NYSE")

        # Create very short data that won't have any complete periods
        timestamps = pd.date_range(
            "2024-01-02 10:00",
            periods=5,
            freq="1h",
            tz="America/New_York",
        )

        # Asking for 4 weeks of data from 5 hours should fail
        with pytest.raises(ValueError, match="Could not find any complete periods"):
            parse_time_size_calendar_aware("4W", timestamps, cal)


class TestCalendarSessionsEdgeCases:
    """Edge case tests for session handling."""

    def test_get_sessions_outside_market_hours(self):
        """Test session assignment for timestamps outside market hours."""
        cal = TradingCalendar("NYSE")

        # Create timestamps at 3 AM (well before market open)
        timestamps = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-02 03:00", tz="America/New_York"),
                pd.Timestamp("2024-01-03 03:00", tz="America/New_York"),
            ]
        )

        sessions = cal.get_sessions(timestamps)

        # Should assign to the next trading session
        assert len(sessions) == 2
        assert len(sessions.unique()) == 2  # Two different sessions

    def test_get_sessions_preserves_original_index(self):
        """Test that get_sessions preserves the original timestamp index."""
        cal = TradingCalendar("NYSE")

        timestamps = pd.date_range(
            "2024-01-02 10:00",
            periods=5,
            freq="1h",
            tz="America/New_York",
        )

        sessions = cal.get_sessions(timestamps)

        # Index should match the input timestamps
        assert list(sessions.index) == list(timestamps)
