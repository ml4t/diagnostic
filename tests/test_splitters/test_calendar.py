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

    def test_months_spec_no_futurewarning(self):
        """Test deprecated month alias is normalized without FutureWarning."""
        import warnings

        timestamps = pd.date_range("2024-01-01", periods=365, freq="1D")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _parse_time_size_naive("1M", timestamps)
        month_alias_warnings = [
            w
            for w in captured
            if issubclass(w.category, FutureWarning)
            and "deprecated" in str(w.message).lower()
            and "'m'" in str(w.message).lower()
        ]
        assert not month_alias_warnings


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


class TestGetSessionsAndMask:
    """Tests for TradingCalendar.get_sessions_and_mask()."""

    def test_all_trading_days(self):
        """All weekday timestamps should be trading rows."""
        cal = TradingCalendar("NYSE")
        # Tue-Thu in Jan 2024 (all trading days)
        timestamps = pd.date_range("2024-01-02 10:00", periods=3, freq="1D", tz="America/New_York")
        sessions, mask = cal.get_sessions_and_mask(timestamps)

        assert mask.all()
        assert sessions.notna().all()

    def test_weekend_excluded(self):
        """Weekend timestamps should be non-trading."""
        cal = TradingCalendar("NYSE")
        timestamps = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-05 12:00", tz="America/New_York"),  # Friday
                pd.Timestamp("2024-01-06 12:00", tz="America/New_York"),  # Saturday
                pd.Timestamp("2024-01-07 12:00", tz="America/New_York"),  # Sunday
                pd.Timestamp("2024-01-08 12:00", tz="America/New_York"),  # Monday
            ]
        )
        sessions, mask = cal.get_sessions_and_mask(timestamps)

        assert mask[0]  # Friday = trading
        assert not mask[1]  # Saturday = non-trading
        assert not mask[2]  # Sunday = non-trading
        assert mask[3]  # Monday = trading
        assert pd.isna(sessions.iloc[1])
        assert pd.isna(sessions.iloc[2])

    def test_holiday_excluded(self):
        """Holiday timestamps should be non-trading."""
        cal = TradingCalendar("NYSE")
        # 2024-01-01 is New Year's Day (Monday holiday)
        timestamps = pd.DatetimeIndex(
            [
                pd.Timestamp("2023-12-29 12:00", tz="America/New_York"),  # Friday
                pd.Timestamp("2024-01-01 12:00", tz="America/New_York"),  # Holiday
                pd.Timestamp("2024-01-02 12:00", tz="America/New_York"),  # Tuesday
            ]
        )
        sessions, mask = cal.get_sessions_and_mask(timestamps)

        assert mask[0]  # Friday = trading
        assert not mask[1]  # Holiday = non-trading
        assert mask[2]  # Tuesday = trading

    def test_mask_length_matches_input(self):
        """Mask and sessions should have same length as input."""
        cal = TradingCalendar("NYSE")
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
        sessions, mask = cal.get_sessions_and_mask(timestamps)

        assert len(mask) == len(timestamps)
        assert len(sessions) == len(timestamps)

    def test_all_non_trading_days(self):
        """All non-trading timestamps should return empty mask and NaT sessions."""
        cal = TradingCalendar("NYSE")
        timestamps = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-06 12:00", tz="America/New_York"),  # Saturday
                pd.Timestamp("2024-01-07 12:00", tz="America/New_York"),  # Sunday
            ]
        )
        sessions, mask = cal.get_sessions_and_mask(timestamps)

        assert not mask.any()
        assert sessions.isna().all()


class TestTimeSpecToSessions:
    """Tests for TradingCalendar.time_spec_to_sessions()."""

    def test_1d_returns_1(self):
        """1D should return 1 session."""
        cal = TradingCalendar("NYSE")
        assert cal.time_spec_to_sessions("1D") == 1

    def test_5d_returns_5(self):
        """5D should return 5 sessions."""
        cal = TradingCalendar("NYSE")
        assert cal.time_spec_to_sessions("5D") == 5

    def test_4w_returns_20(self):
        """4W should return ~20 sessions (4 weeks x 5 days/week)."""
        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("4W")
        assert result == 20

    def test_1w_returns_5(self):
        """1W should return ~5 sessions."""
        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("1W")
        assert result == 5

    def test_1m_returns_about_21(self):
        """1M should return ~21 sessions."""
        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("1M")
        assert 20 <= result <= 23

    def test_1y_returns_about_252(self):
        """1Y should return ~252 sessions."""
        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("1Y")
        assert 250 <= result <= 253

    def test_invalid_spec_raises(self):
        """Invalid spec should raise ValueError."""
        cal = TradingCalendar("NYSE")
        with pytest.raises(ValueError, match="Invalid time specification"):
            cal.time_spec_to_sessions("invalid")

    def test_case_insensitive(self):
        """Spec should be case-insensitive."""
        cal = TradingCalendar("NYSE")
        assert cal.time_spec_to_sessions("4w") == cal.time_spec_to_sessions("4W")
