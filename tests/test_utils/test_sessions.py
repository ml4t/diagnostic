"""Tests for session assignment utilities."""

from __future__ import annotations

import pandas as pd
import pytest


class TestAssignSessionDates:
    """Tests for assign_session_dates function."""

    def test_basic_session_assignment(self):
        """Test basic session date assignment."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        # Create intraday data with DatetimeIndex
        dates = pd.date_range("2024-01-02 09:00", periods=100, freq="1min", tz="America/New_York")
        df = pd.DataFrame({"value": range(100)}, index=dates)

        result = assign_session_dates(df, calendar="NYSE")

        assert "session_date" in result.columns
        # All timestamps from same day should have same session
        assert result["session_date"].nunique() == 1

    def test_multi_day_session_assignment(self):
        """Test session assignment across multiple days."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        # Create multi-day data
        dates = pd.date_range("2024-01-02 10:00", periods=100, freq="1h", tz="America/New_York")
        df = pd.DataFrame({"value": range(100)}, index=dates)

        result = assign_session_dates(df, calendar="NYSE")

        assert "session_date" in result.columns
        # Should have multiple sessions
        assert result["session_date"].nunique() > 1

    def test_custom_column_name(self):
        """Test custom session column name."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        dates = pd.date_range("2024-01-02 10:00", periods=10, freq="1h", tz="America/New_York")
        df = pd.DataFrame({"value": range(10)}, index=dates)

        result = assign_session_dates(df, calendar="NYSE", session_column="trading_day")

        assert "trading_day" in result.columns
        assert "session_date" not in result.columns

    def test_invalid_index_raises_error(self):
        """Test that non-DatetimeIndex raises ValueError."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        df = pd.DataFrame({"value": range(10)})  # Default RangeIndex

        with pytest.raises(ValueError, match="DatetimeIndex"):
            assign_session_dates(df)

    def test_original_dataframe_unchanged(self):
        """Test that original DataFrame is not modified."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        dates = pd.date_range("2024-01-02 10:00", periods=10, freq="1h", tz="America/New_York")
        df = pd.DataFrame({"value": range(10)}, index=dates)
        original_columns = list(df.columns)

        _ = assign_session_dates(df, calendar="NYSE")

        assert list(df.columns) == original_columns
        assert "session_date" not in df.columns

    def test_cme_calendar(self):
        """Test with CME_Equity calendar."""
        from ml4t.diagnostic.utils.sessions import assign_session_dates

        dates = pd.date_range("2024-01-02 10:00", periods=10, freq="1h", tz="America/Chicago")
        df = pd.DataFrame({"value": range(10)}, index=dates)

        result = assign_session_dates(df, calendar="CME_Equity", timezone="America/Chicago")

        assert "session_date" in result.columns

    def test_missing_market_calendars(self):
        """Test error when pandas_market_calendars not installed."""
        from ml4t.diagnostic.utils import sessions

        # Temporarily set HAS_MARKET_CALENDARS to False
        original = sessions.HAS_MARKET_CALENDARS
        sessions.HAS_MARKET_CALENDARS = False

        try:
            dates = pd.date_range("2024-01-02", periods=10, freq="1h", tz="UTC")
            df = pd.DataFrame({"value": range(10)}, index=dates)

            with pytest.raises(ImportError, match="pandas_market_calendars"):
                sessions.assign_session_dates(df)
        finally:
            sessions.HAS_MARKET_CALENDARS = original


class TestGetCompleteSessions:
    """Tests for get_complete_sessions function."""

    def test_basic_complete_sessions(self):
        """Test getting complete sessions."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        # Create DataFrame with session_date column
        df = pd.DataFrame(
            {
                "session_date": ["2024-01-02"] * 150 + ["2024-01-03"] * 50,
                "value": range(200),
            }
        )

        result = get_complete_sessions(df, min_samples=100)

        # Only first session has >= 100 samples
        assert len(result) == 1
        assert "2024-01-02" in result.values

    def test_all_complete(self):
        """Test when all sessions are complete."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        df = pd.DataFrame(
            {
                "session_date": ["2024-01-02"] * 200 + ["2024-01-03"] * 200,
                "value": range(400),
            }
        )

        result = get_complete_sessions(df, min_samples=100)

        assert len(result) == 2

    def test_none_complete(self):
        """Test when no sessions are complete."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        df = pd.DataFrame(
            {
                "session_date": ["2024-01-02"] * 10 + ["2024-01-03"] * 10,
                "value": range(20),
            }
        )

        result = get_complete_sessions(df, min_samples=100)

        assert len(result) == 0

    def test_custom_column_name(self):
        """Test with custom session column name."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        df = pd.DataFrame(
            {
                "trading_day": ["2024-01-02"] * 150,
                "value": range(150),
            }
        )

        result = get_complete_sessions(df, session_column="trading_day", min_samples=100)

        assert len(result) == 1

    def test_missing_column_raises_error(self):
        """Test error when session column is missing."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        df = pd.DataFrame({"value": range(100)})

        with pytest.raises(ValueError, match="does not have"):
            get_complete_sessions(df)

    def test_custom_min_samples(self):
        """Test with different min_samples thresholds."""
        from ml4t.diagnostic.utils.sessions import get_complete_sessions

        df = pd.DataFrame(
            {
                "session_date": ["2024-01-02"] * 50 + ["2024-01-03"] * 100,
                "value": range(150),
            }
        )

        # With min_samples=25, both are complete
        result_low = get_complete_sessions(df, min_samples=25)
        assert len(result_low) == 2

        # With min_samples=75, only second is complete
        result_high = get_complete_sessions(df, min_samples=75)
        assert len(result_high) == 1
        assert "2024-01-03" in result_high.values
