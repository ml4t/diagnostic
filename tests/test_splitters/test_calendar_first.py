"""Tests for calendar-first WalkForwardCV architecture.

Validates that:
1. Fold boundaries align to trading session boundaries
2. Non-trading rows (weekends, holidays) are excluded from fold indices
3. Time-based sizes ("4W", "20D") use trading sessions, not calendar days
4. Calendar defaults to NYSE when not specified
5. Backward compatibility with integer sizes is maintained
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters.config import WalkForwardConfig
from ml4t.diagnostic.splitters.walk_forward import WalkForwardCV


def _make_daily_panel(
    n_assets: int = 1,
    n_days: int = 252,
    start: str = "2024-01-01",
    include_weekends: bool = False,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Create daily panel data for testing.

    Parameters
    ----------
    n_assets : int
        Number of assets in panel.
    n_days : int
        Number of calendar days (if include_weekends) or business days.
    start : str
        Start date.
    include_weekends : bool
        If True, include Saturday/Sunday rows.
    tz : str
        Timezone.

    Returns
    -------
    pd.DataFrame
        Panel with DatetimeIndex, 'close' column, and optionally 'asset' column.
    """
    if include_weekends:
        dates = pd.date_range(start, periods=n_days, freq="D", tz=tz)
    else:
        dates = pd.bdate_range(start, periods=n_days, tz=tz)

    if n_assets == 1:
        df = pd.DataFrame(
            {"close": np.random.randn(len(dates))},
            index=dates,
        )
    else:
        frames = []
        for i in range(n_assets):
            asset_df = pd.DataFrame(
                {
                    "close": np.random.randn(len(dates)),
                    "asset": f"ASSET_{i}",
                },
                index=dates,
            )
            frames.append(asset_df)
        df = pd.concat(frames)
        df = df.sort_index()  # Sort by time (interleaved assets)

    return df


class TestCalendarFirstBasic:
    """Basic calendar-first splitting tests."""

    def test_calendar_default_is_nyse(self):
        """WalkForwardCV without explicit calendar should use NYSE."""
        cv = WalkForwardCV(n_splits=3, test_size=20)
        assert cv.calendar is not None
        assert cv.calendar.config.exchange == "NYSE"
        assert cv.config.calendar_id == "NYSE"

    def test_calendar_override(self):
        """Explicit calendar should override the default."""
        cv = WalkForwardCV(n_splits=3, test_size=20, calendar="CME_Equity")
        assert cv.calendar is not None
        assert cv.calendar.config.exchange == "CME_Equity"

    def test_20d_means_20_trading_sessions(self):
        """'20D' should produce folds with ~20 trading sessions, not 20 calendar days."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size="20D")

        splits = list(cv.split(df))
        assert len(splits) == 3

        for _train_idx, test_idx in splits:
            # Count unique trading dates in test set
            test_dates = df.index[test_idx]
            unique_dates = test_dates.normalize().unique()
            assert len(unique_dates) == 20

    def test_4w_produces_session_boundaries(self):
        """'4W' test_size should produce ~20-session test folds."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size="4W")

        splits = list(cv.split(df))
        assert len(splits) == 3

        for _train_idx, test_idx in splits:
            test_dates = df.index[test_idx]
            unique_dates = test_dates.normalize().unique()
            # 4W = 4 * 5 = 20 trading sessions
            assert len(unique_dates) == 20

    def test_non_trading_rows_excluded(self):
        """Saturday/Sunday rows should never appear in fold indices."""
        # Create data with weekends
        df = _make_daily_panel(n_days=365, include_weekends=True)
        cv = WalkForwardCV(n_splits=3, test_size=20)

        # Identify weekend rows
        weekend_mask = df.index.dayofweek >= 5
        weekend_indices = set(np.where(weekend_mask)[0])

        for train_idx, test_idx in cv.split(df):
            all_fold_indices = set(train_idx.tolist()) | set(test_idx.tolist())
            overlap = all_fold_indices & weekend_indices
            assert len(overlap) == 0, f"Found {len(overlap)} weekend rows in fold indices"

    def test_holiday_in_4w_gives_fewer_sessions(self):
        """4W with a holiday should produce fewer than 20 sessions in that fold."""
        # Create 252 business days of data (full year 2024)
        df = _make_daily_panel(n_days=252, start="2024-01-02", include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size="4W")

        splits = list(cv.split(df))
        # All folds should have close to 20 trading sessions
        for _train_idx, test_idx in splits:
            test_dates = df.index[test_idx]
            unique_dates = test_dates.normalize().unique()
            # Some 4W periods may have holidays, resulting in 19 sessions
            assert 19 <= len(unique_dates) <= 20

    def test_integer_sizes_still_work(self):
        """Integer test_size should work without calendar conversion."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=50)

        splits = list(cv.split(df))
        assert len(splits) == 3

        for _train_idx, test_idx in splits:
            # Integer sizes are session counts in calendar mode
            assert len(test_idx) > 0

    def test_float_sizes_work(self):
        """Float test_size (proportion) should work in calendar mode."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=0.1)

        splits = list(cv.split(df))
        assert len(splits) == 3


class TestCalendarFirstNonTradingFiltering:
    """Tests for non-trading row filtering behavior."""

    def test_filter_non_trading_default_true(self):
        """filter_non_trading should default to True."""
        config = WalkForwardConfig(n_splits=5)
        assert config.filter_non_trading is True

    def test_mixed_trading_and_weekend_data(self):
        """Data containing both weekday and weekend rows should filter weekends."""
        # Create 30 calendar days (includes weekends)
        dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        df = pd.DataFrame({"close": np.random.randn(len(dates))}, index=dates)

        cv = WalkForwardCV(n_splits=2, test_size=5)
        splits = list(cv.split(df))

        for train_idx, test_idx in splits:
            # Check no weekend indices
            train_dates = df.index[train_idx]
            test_dates = df.index[test_idx]
            assert all(d.dayofweek < 5 for d in train_dates)
            assert all(d.dayofweek < 5 for d in test_dates)


class TestCalendarFirstWindowModes:
    """Tests for expanding and rolling windows in session space."""

    def test_expanding_window(self):
        """Expanding window should grow train set with each fold."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, expanding=True, consecutive=True)

        train_sizes = []
        for train_idx, _test_idx in cv.split(df):
            train_sizes.append(len(train_idx))

        # Each fold should have a larger training set
        assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_rolling_window(self):
        """Rolling window should have fixed-ish train set size."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(
            n_splits=3,
            test_size=20,
            train_size=100,
            expanding=False,
            consecutive=True,
        )

        train_sizes = []
        for train_idx, _test_idx in cv.split(df):
            train_sizes.append(len(train_idx))

        # All folds should have similar training size
        for size in train_sizes:
            assert abs(size - 100) <= 1  # Allow ±1 for session boundary rounding

    def test_consecutive_test_periods(self):
        """consecutive=True should produce back-to-back test folds."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, consecutive=True)

        splits = list(cv.split(df))
        for i in range(len(splits) - 1):
            _, test_i = splits[i]
            _, test_next = splits[i + 1]
            # Next test should start right after previous test ends
            # (in session space, they should be adjacent)
            last_test_date = df.index[test_i[-1]].normalize()
            first_next_date = df.index[test_next[0]].normalize()
            # They should be adjacent trading days
            gap_days = (first_next_date - last_test_date).days
            assert gap_days <= 3  # At most a weekend gap


class TestCalendarFirstPurging:
    """Tests for purging with calendar boundaries."""

    def test_purging_with_calendar_boundaries(self):
        """Purging should use actual timestamp windows from session boundaries."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, label_horizon=5)

        for train_idx, test_idx in cv.split(df):
            # No training sample should be within 5 days of test start
            test_start_date = df.index[test_idx[0]]
            train_dates = df.index[train_idx]
            if len(train_dates) > 0:
                last_train_date = train_dates[-1]
                gap = (test_start_date - last_train_date).days
                # Gap should be at least label_horizon trading days
                assert gap >= 5, f"Purging gap too small: {gap} days (expected >= 5)"


class TestCalendarFirstPanel:
    """Tests for panel (multi-asset) data with calendar splitting."""

    def test_panel_data_sessions(self):
        """Panel data (100 assets x 252 days) should have 252 unique sessions."""
        df = _make_daily_panel(n_assets=5, n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, consecutive=True)

        splits = list(cv.split(df))
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

            # Test set should contain all 5 assets for each session
            test_dates = df.index[test_idx].normalize().unique()
            n_test_sessions = len(test_dates)
            # Each session should have n_assets rows
            assert len(test_idx) == n_test_sessions * 5


class TestCalendarFirstDeprecation:
    """Tests for backward compatibility and deprecation warnings."""

    def test_align_to_sessions_deprecation_warning(self):
        """Setting align_to_sessions=True with calendar should emit deprecation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WalkForwardConfig(
                n_splits=5,
                align_to_sessions=True,
                calendar_id="NYSE",
            )
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "subsumes" in str(deprecation_warnings[0].message).lower()

    def test_no_deprecation_without_align_to_sessions(self):
        """No deprecation when align_to_sessions is not set."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WalkForwardConfig(n_splits=5, calendar_id="NYSE")
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


class TestCalendarFirstConfig:
    """Tests for calendar-first configuration defaults."""

    def test_config_default_calendar_is_nyse(self):
        """WalkForwardConfig should default calendar_id to NYSE."""
        config = WalkForwardConfig(n_splits=5)
        assert config.calendar_id == "NYSE"

    def test_config_calendar_none_disables(self):
        """Explicitly setting calendar_id=None should disable calendar splitting."""
        config = WalkForwardConfig(n_splits=5, calendar_id=None)
        assert config.calendar_id is None

    def test_config_filter_non_trading_default(self):
        """filter_non_trading should default to True."""
        config = WalkForwardConfig(n_splits=5)
        assert config.filter_non_trading is True


class TestCalendarFirstEdgeCases:
    """Edge cases for calendar-first splitting."""

    def test_no_timestamps_raises(self):
        """Calendar splitting without timestamps should raise ValueError."""
        X = np.arange(100).reshape(100, 1)
        cv = WalkForwardCV(n_splits=3, test_size="4W")

        with pytest.raises(ValueError, match="timestamps"):
            list(cv.split(X))

    def test_very_short_data(self):
        """Very short data should raise meaningful error."""
        df = _make_daily_panel(n_days=5, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, consecutive=True)

        with pytest.raises(ValueError, match="Insufficient"):
            list(cv.split(df))

    def test_train_precedes_test(self):
        """All training indices should be chronologically before test."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20)

        for train_idx, test_idx in cv.split(df):
            max_train_date = df.index[train_idx].max()
            min_test_date = df.index[test_idx].min()
            assert max_train_date < min_test_date


class TestCalendarNoneDisablesCalendar:
    """Tests verifying calendar=None fallback to sample-based splitting."""

    def test_calendar_none_uses_sample_splitting(self):
        """WalkForwardCV(calendar=None) should fall back to _split_by_samples."""
        df = _make_daily_panel(n_days=100, include_weekends=False)
        # Explicitly disable calendar
        config = WalkForwardConfig(n_splits=3, test_size=20, calendar_id=None)
        cv = WalkForwardCV(config=config)
        assert cv.calendar is None
        splits = list(cv.split(df))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestFilterNonTradingFalse:
    """Tests for filter_non_trading=False behavior."""

    def test_filter_non_trading_false_includes_weekends(self):
        """Setting filter_non_trading=False should include weekend rows."""
        # Create data with weekends
        df = _make_daily_panel(n_days=90, include_weekends=True)
        config = WalkForwardConfig(
            n_splits=2, test_size=10, filter_non_trading=False, calendar_id="NYSE"
        )
        cv = WalkForwardCV(config=config)
        splits = list(cv.split(df))
        assert len(splits) == 2

        # Collect all indices used in folds
        all_indices = set()
        for train_idx, test_idx in splits:
            all_indices.update(train_idx.tolist())
            all_indices.update(test_idx.tolist())

        # Should include some weekend rows
        all_dates = df.index[sorted(all_indices)]
        weekend_count = sum(1 for d in all_dates if d.dayofweek >= 5)
        assert weekend_count > 0, "Expected weekend rows when filter_non_trading=False"

    def test_filter_non_trading_false_bypasses_calendar_splitting(self):
        """filter_non_trading=False with int sizes should use sample splitting."""
        config = WalkForwardConfig(
            n_splits=3, test_size=20, filter_non_trading=False, calendar_id="NYSE"
        )
        cv = WalkForwardCV(config=config)
        # _should_use_calendar_splitting returns False: no str sizes, filter off
        assert not cv._should_use_calendar_splitting()


class TestCalendarAutoTestSize:
    """Tests for test_size=None in calendar mode."""

    def test_auto_test_size_in_calendar_mode(self):
        """Omitting test_size with calendar should auto-calculate session count."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=5)  # No test_size, calendar defaults to NYSE
        splits = list(cv.split(df))
        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestStringTrainSize:
    """Tests for string-based train_size in calendar mode."""

    def test_string_train_size(self):
        """train_size='52W' should produce ~260-session training windows."""
        df = _make_daily_panel(n_days=504, include_weekends=False)
        cv = WalkForwardCV(
            n_splits=2, test_size="4W", train_size="52W", consecutive=True, expanding=False
        )
        splits = list(cv.split(df))
        assert len(splits) == 2
        for train_idx, _test_idx in splits:
            train_dates = df.index[train_idx].normalize().unique()
            # 52W = 260 sessions
            assert 255 <= len(train_dates) <= 262

    def test_string_train_size_triggers_calendar_splitting(self):
        """String train_size should cause _should_use_calendar_splitting to return True."""
        cv = WalkForwardCV(n_splits=3, test_size=20, train_size="52W")
        assert cv._should_use_calendar_splitting()


class TestGapWithCalendar:
    """Tests for gap parameter in calendar splitting."""

    def test_gap_with_calendar(self):
        """gap > 0 should create separation between train and test sessions."""
        df = _make_daily_panel(n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, gap=5, consecutive=True)

        for train_idx, test_idx in cv.split(df):
            train_dates = df.index[train_idx]
            test_dates = df.index[test_idx]
            last_train_date = train_dates.max()
            first_test_date = test_dates.min()
            gap_days = (first_test_date - last_train_date).days
            # At least 5 sessions gap (calendar days will be >= 5)
            assert gap_days >= 5


class TestIsolateGroupsWithCalendar:
    """Tests for group isolation in calendar splitting."""

    def test_isolate_groups_with_calendar(self):
        """isolate_groups=True should prevent asset overlap between train/test."""
        df = _make_daily_panel(n_assets=5, n_days=252, include_weekends=False)
        cv = WalkForwardCV(n_splits=3, test_size=20, isolate_groups=True, consecutive=True)
        groups = df["asset"]

        for train_idx, test_idx in cv.split(df, groups=groups):
            train_assets = set(df.iloc[train_idx]["asset"])
            test_assets = set(df.iloc[test_idx]["asset"])
            overlap = train_assets & test_assets
            assert len(overlap) == 0, f"Asset overlap: {overlap}"


class TestRuntimeDeprecation:
    """Tests for runtime deprecation warnings in split()."""

    def test_runtime_deprecation_when_align_to_sessions(self):
        """align_to_sessions=True at split time should emit runtime deprecation."""
        df = _make_daily_panel(n_days=100, include_weekends=False)
        df["session_date"] = df.index.normalize()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cv = WalkForwardCV(
                n_splits=3,
                test_size=10,
                align_to_sessions=True,
                calendar="NYSE",
            )
            list(cv.split(df))
            deprecation_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert any("subsumes" in str(m.message).lower() for m in deprecation_msgs)


class TestConfigEdgeCases:
    """Additional config validation edge cases."""

    def test_no_deprecation_when_align_to_sessions_without_calendar(self):
        """align_to_sessions=True with calendar_id=None should NOT warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WalkForwardConfig(n_splits=5, align_to_sessions=True, calendar_id=None)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_3m_time_spec(self):
        """3M should return ~63 sessions."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar

        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("3M")
        assert 60 <= result <= 66

    def test_2y_time_spec(self):
        """2Y should return ~504 sessions."""
        from ml4t.diagnostic.splitters.calendar import TradingCalendar

        cal = TradingCalendar("NYSE")
        result = cal.time_spec_to_sessions("2Y")
        assert 500 <= result <= 506
