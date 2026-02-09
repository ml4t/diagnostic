"""Tests for WalkForwardCV held-out test and backward validation features.

These tests verify the new architecture for:
1. Held-out test period - Reserve most recent data for final evaluation
2. Backward-stepping validation folds - Iterate backward from test boundary
3. Trading-day-aware gaps - label_horizon respects trading calendar
"""

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.config import WalkForwardConfig
from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV


def _has_market_calendars() -> bool:
    """Check if pandas_market_calendars is installed."""
    try:
        import pandas_market_calendars  # noqa: F401

        return True
    except ImportError:
        return False


class TestHeldOutTestPeriod:
    """Test held-out test period specification."""

    def test_test_period_as_string(self):
        """Test test_period as time-based string (e.g., '52D')."""
        # Create data with 200 days
        dates = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(200)}, index=dates)

        cv = WalkForwardCV(
            n_splits=3,
            test_period="30D",  # Reserve last 30 days for held-out test
            test_size=20,
            train_size=50,
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        assert len(splits) == 3

        # Check held-out test indices are available
        test_indices = cv.test_indices_
        assert len(test_indices) > 0

        # Test indices should be at the end of the data
        assert test_indices[-1] == 199

        # No validation fold should overlap with held-out test
        for train_idx, val_idx in splits:
            assert val_idx.max() < test_indices.min()

    def test_test_period_as_int_without_calendar(self):
        """Test test_period as integer (samples) without calendar."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_period=30,  # Last 30 samples as held-out test
            test_size=20,
            fold_direction="forward",
        )

        splits = list(cv.split(X))
        assert len(splits) == 3

        # Check held-out test indices
        test_indices = cv.test_indices_
        assert len(test_indices) == 30
        assert test_indices[0] == 170  # n_samples - test_period

    def test_test_start_explicit_date(self):
        """Test explicit test_start date for held-out period."""
        from datetime import date

        dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = WalkForwardCV(
            n_splits=2,
            test_start="2024-03-15",  # Explicit start date
            test_size=10,
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        test_indices = cv.test_indices_

        # Test should start at 2024-03-15
        test_start_date = X.index[test_indices[0]].date()
        assert test_start_date == date(2024, 3, 15)

    def test_test_start_and_test_end(self):
        """Test explicit test_start and test_end dates."""
        from datetime import date

        dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
        X = pd.DataFrame({"feature": np.arange(100)}, index=dates)

        cv = WalkForwardCV(
            n_splits=2,
            test_start="2024-03-01",
            test_end="2024-03-15",  # Explicit end date
            test_size=10,
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        test_indices = cv.test_indices_

        # Test should be in the specified range
        test_start_date = X.index[test_indices[0]].date()
        test_end_date = X.index[test_indices[-1]].date()
        assert test_start_date == date(2024, 3, 1)
        # test_end specifies the boundary - last included date is at or before test_end
        assert test_end_date <= date(2024, 3, 15)

    def test_test_period_and_test_start_mutually_exclusive(self):
        """Test that test_period and test_start cannot both be specified."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            WalkForwardConfig(
                n_splits=3,
                test_period="30D",
                test_start="2024-03-01",
            )

    def test_test_end_requires_test_start_or_test_period(self):
        """Test that test_end requires test_start or test_period."""
        with pytest.raises(ValueError, match="requires either"):
            WalkForwardConfig(
                n_splits=3,
                test_end="2024-03-15",
            )

    def test_no_held_out_test_raises_on_test_indices(self):
        """Test that accessing test_indices_ without held-out test raises."""
        cv = WalkForwardCV(n_splits=3)
        X = np.arange(100).reshape(100, 1)
        list(cv.split(X))  # Must call split first

        with pytest.raises(ValueError, match="No held-out test period configured"):
            _ = cv.test_indices_


class TestBackwardValidationFolds:
    """Test backward-stepping validation fold generation."""

    def test_backward_folds_step_backward_from_test(self):
        """Test that backward folds step backward from held-out test boundary."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_period=30,
            test_size=20,
            train_size=50,
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        test_start_idx = 170  # n_samples - test_period

        # Validation folds should step backward from test_start_idx
        val_ends = [val_idx.max() + 1 for _, val_idx in splits]

        # First fold's validation should end at test boundary
        assert val_ends[0] == test_start_idx

        # Each subsequent fold should be further back
        for i in range(1, len(val_ends)):
            assert val_ends[i] < val_ends[i - 1]

    def test_backward_folds_no_overlap_with_test(self):
        """Test that backward validation folds don't overlap with held-out test."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=5,
            test_period=30,
            test_size=15,
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        test_indices = cv.test_indices_

        for train_idx, val_idx in splits:
            # No validation index should be >= test start
            assert val_idx.max() < test_indices.min()

            # No training index should be >= test start
            assert train_idx.max() < test_indices.min()

    def test_backward_expanding_window(self):
        """Test backward folds with expanding training window."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_period=30,
            test_size=20,
            expanding=True,
            fold_direction="backward",
        )

        splits = list(cv.split(X))

        # With expanding=True and backward direction:
        # Each fold's training should start at 0
        for train_idx, _ in splits:
            assert train_idx.min() == 0

    def test_backward_rolling_window(self):
        """Test backward folds with rolling (fixed-size) training window."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_period=30,
            test_size=20,
            train_size=50,
            expanding=False,
            fold_direction="backward",
        )

        splits = list(cv.split(X))

        # Training sizes should be approximately equal
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert max(train_sizes) - min(train_sizes) < 10  # Allow some variation


class TestForwardWithHeldOutTest:
    """Test forward-stepping validation with held-out test."""

    def test_forward_folds_stay_before_test(self):
        """Test that forward folds stay before held-out test boundary."""
        n_samples = 200
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_period=30,
            test_size=20,
            fold_direction="forward",  # Forward direction with held-out test
        )

        splits = list(cv.split(X))
        test_indices = cv.test_indices_

        for train_idx, val_idx in splits:
            # All validation indices should be before held-out test
            assert val_idx.max() < test_indices.min()


class TestTradingDayAwareGaps:
    """Test trading-day-aware gap calculations."""

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_label_horizon_with_calendar(self):
        """Test that label_horizon respects trading calendar."""
        # Create daily data spanning 6 months (enough for calendar-aware purging)
        dates = pd.date_range("2024-01-01", periods=180, freq="D", tz="America/New_York")
        X = pd.DataFrame({"feature": np.arange(180)}, index=dates)

        cv = WalkForwardCV(
            n_splits=2,
            test_size=20,
            train_size=60,  # Explicit train size
            label_horizon=5,  # 5 TRADING days
            calendar="NYSE",
        )

        splits = list(cv.split(X))

        # Verify splits were created
        assert len(splits) == 2

        for train_idx, val_idx in splits:
            # Verify we have training data (not all purged)
            assert len(train_idx) > 0
            # Training should precede validation
            assert train_idx.max() < val_idx.min()

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_test_period_with_calendar_trading_days(self):
        """Test test_period as trading days with calendar."""
        # Create data spanning several weeks
        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
        X = pd.DataFrame({"feature": np.arange(30)}, index=dates)

        cv = WalkForwardCV(
            n_splits=2,
            test_period=5,  # 5 trading days
            test_size=3,
            calendar="NYSE",
            fold_direction="backward",
        )

        splits = list(cv.split(X))
        test_indices = cv.test_indices_

        # Test period should span more than 5 calendar days due to weekends
        test_dates = X.index[test_indices]
        calendar_days = (test_dates[-1] - test_dates[0]).days
        assert calendar_days >= 5  # At least 5 calendar days (likely 7+ with weekend)


class TestBackwardCompatibility:
    """Test that legacy behavior is unchanged without held-out test config."""

    def test_legacy_forward_walk_forward(self):
        """Test that traditional walk-forward still works without held-out test."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_size=20,
            label_horizon=5,
        )

        splits = list(cv.split(X))
        assert len(splits) == 3

        # Check sequential test sets
        test_starts = [val_idx[0] for _, val_idx in splits]
        assert all(test_starts[i] < test_starts[i + 1] for i in range(len(test_starts) - 1))

        # No held-out test configured
        with pytest.raises(ValueError, match="No held-out test period configured"):
            _ = cv.test_indices_

    def test_legacy_consecutive_walk_forward(self):
        """Test consecutive walk-forward without held-out test."""
        n_samples = 100
        X = np.arange(n_samples).reshape(n_samples, 1)

        cv = WalkForwardCV(
            n_splits=3,
            test_size=20,
            consecutive=True,
        )

        splits = list(cv.split(X))
        assert len(splits) == 3

        # Check back-to-back test sets
        prev_test_end = None
        for _, val_idx in splits:
            if prev_test_end is not None:
                assert val_idx[0] == prev_test_end
            prev_test_end = val_idx[-1] + 1


class TestConfigSerialization:
    """Test that new config fields serialize/deserialize correctly."""

    def test_config_roundtrip(self):
        """Test config can be serialized and restored."""
        from datetime import date
        import json

        config = WalkForwardConfig(
            n_splits=5,
            test_period="30D",
            test_size=20,
            train_size=100,
            fold_direction="backward",
            calendar_id="NYSE",
        )

        # Serialize to JSON
        json_str = config.model_dump_json()
        data = json.loads(json_str)

        # Verify fields
        assert data["test_period"] == "30D"
        assert data["fold_direction"] == "backward"
        assert data["calendar_id"] == "NYSE"

        # Deserialize back
        restored = WalkForwardConfig.model_validate_json(json_str)
        assert restored.test_period == "30D"
        assert restored.fold_direction == "backward"
        assert restored.calendar_id == "NYSE"

    def test_config_with_explicit_dates(self):
        """Test config with explicit date fields."""
        from datetime import date

        config = WalkForwardConfig(
            n_splits=3,
            test_start="2024-03-01",
            test_end="2024-03-31",
            fold_direction="backward",
        )

        assert config.test_start == date(2024, 3, 1)
        assert config.test_end == date(2024, 3, 31)


class TestCalendarUtilities:
    """Test trading calendar utility functions."""

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_previous_trading_day(self):
        """Test previous_trading_day navigation."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar

        calendar = TradingCalendar("NYSE")

        # Monday 2024-02-05 (no holidays nearby)
        monday = pd.Timestamp("2024-02-05", tz="America/New_York")

        # Previous trading day from Monday should be Friday
        prev_day = calendar.previous_trading_day(monday, n=1)
        assert prev_day.day_of_week == 4  # Friday (2024-02-02)

        # 5 trading days back from Monday should be Monday of previous week
        five_back = calendar.previous_trading_day(monday, n=5)
        # 5 trading days: Fri 2, Thu 1, Wed 31, Tue 30, Mon 29 -> 2024-01-29
        assert five_back.day_of_week == 0  # Monday 2024-01-29

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_next_trading_day(self):
        """Test next_trading_day navigation."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar

        calendar = TradingCalendar("NYSE")

        # Friday 2024-01-05
        friday = pd.Timestamp("2024-01-05", tz="America/New_York")

        # Next trading day from Friday should be Monday
        next_day = calendar.next_trading_day(friday, n=1)
        assert next_day.day_of_week == 0  # Monday

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_trading_days_between(self):
        """Test trading_days_between count."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar

        calendar = TradingCalendar("NYSE")

        # Mon to Fri (same week) should be 4 trading days [Mon, Tue, Wed, Thu)
        start = pd.Timestamp("2024-01-08", tz="America/New_York")  # Monday
        end = pd.Timestamp("2024-01-12", tz="America/New_York")  # Friday

        count = calendar.trading_days_between(start, end)
        assert count == 4

    @pytest.mark.skipif(
        not _has_market_calendars(),
        reason="pandas_market_calendars not installed",
    )
    def test_trading_days_to_timedelta(self):
        """Test trading_days_to_timedelta conversion."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar, trading_days_to_timedelta

        calendar = TradingCalendar("NYSE")

        # 5 trading days backward should span more than 5 calendar days (weekend)
        ref = pd.Timestamp("2024-01-15", tz="UTC")  # Monday
        delta = trading_days_to_timedelta(5, calendar, ref, "backward")

        # 5 trading days back from Monday spans over a weekend = 7 calendar days
        assert delta.days >= 5


def _has_market_calendars() -> bool:
    """Check if pandas_market_calendars is installed."""
    try:
        import pandas_market_calendars

        return True
    except ImportError:
        return False
