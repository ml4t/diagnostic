"""Calendar-aware time parsing for financial data cross-validation.

This module provides calendar-aware time period calculations for time-series CV,
ensuring that train/test splits respect trading calendar boundaries (sessions, weeks).

Key Features:
-----------
- Uses pandas_market_calendars for accurate trading session detection
- For intraday data: Sessions are atomic units (don't split trading sessions)
- For 'D' selections: Select complete trading sessions
- For 'W' selections: Select complete trading weeks (groups of sessions)
- Handles varying data density (dollar bars, trade bars) correctly

Background:
----------
Traditional time-based CV approaches use fixed sample counts computed from
time periods, which fails for activity-based data (dollar bars, trade bars) where
sample density varies with market activity. This module ensures proper time-based
selection by using calendar boundaries as atomic units.

Example Issue (Dollar Bars):
- High volatility week: 100K samples in 7 calendar days
- Low volatility week: 65K samples in 7 calendar days
- Fixed sample approach: 82K samples = 3.14 to 5.0 weeks (WRONG!)
- Calendar approach: Exactly 7 calendar days with varying samples (CORRECT!)
"""

import re
from typing import Any, cast

import numpy as np
import pandas as pd
import pytz

try:
    import pandas_market_calendars as mcal

    HAS_MARKET_CALENDARS = True
except ImportError:
    HAS_MARKET_CALENDARS = False

from ml4t.diagnostic.splitters.calendar_config import CalendarConfig


class TradingCalendar:
    """Trading calendar for session-aware time period calculations.

    This class handles proper timezone conversion and trading session detection
    for financial time-series cross-validation.

    Parameters
    ----------
    config : CalendarConfig or str
        Calendar configuration or exchange name (will use default config)

    Attributes
    ----------
    config : CalendarConfig
        Configuration for calendar and timezone handling
    calendar : mcal.MarketCalendar
        The underlying market calendar instance
    tz : pytz.timezone
        Timezone object for conversions
    """

    def __init__(self, config: CalendarConfig | str = "CME_Equity"):
        """Initialize trading calendar with configuration."""
        if not HAS_MARKET_CALENDARS:
            raise ImportError(
                "pandas_market_calendars is required for calendar-aware CV. "
                "Install with: pip install pandas_market_calendars"
            )

        # Handle string input (exchange name) by creating default config
        if isinstance(config, str):
            from ml4t.diagnostic.splitters.calendar_config import CalendarConfig

            config = CalendarConfig(exchange=config, timezone="UTC", localize_naive=True)

        self.config = config
        self.calendar = mcal.get_calendar(config.exchange)
        self.tz = pytz.timezone(config.timezone)

    def _ensure_timezone_aware(self, timestamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Ensure timestamps are timezone-aware.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Input timestamps (may be tz-naive or tz-aware)

        Returns
        -------
        pd.DatetimeIndex
            Timezone-aware timestamps in calendar's timezone
        """
        if timestamps.tz is None:
            # Tz-naive data
            if self.config.localize_naive:
                # Localize to calendar timezone
                return timestamps.tz_localize(self.tz)
            else:
                raise ValueError(
                    f"Data is timezone-naive but localize_naive=False in config. "
                    f"Either localize data to {self.config.timezone} or set "
                    f"localize_naive=True in CalendarConfig."
                )
        else:
            # Tz-aware data - convert to calendar timezone
            return timestamps.tz_convert(self.tz)

    def get_sessions(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> pd.Series:
        """Assign each timestamp to its trading session date (vectorized).

        A trading session for futures typically runs from Sunday 5pm CT to Friday 4pm CT.
        For stocks, it's the standard trading day.

        Uses vectorized pandas operations for efficiency - handles 1M+ timestamps quickly.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps to assign to sessions (may be tz-naive or tz-aware)

        Returns
        -------
        pd.Series
            Session dates for each timestamp (tz-naive dates, index matches timestamps)
        """
        # Ensure all timestamps are in calendar timezone
        timestamps_tz = self._ensure_timezone_aware(timestamps)

        # Get schedule for the data period (with buffer for edge cases)
        start_date = timestamps_tz[0].normalize() - pd.Timedelta(days=7)
        end_date = timestamps_tz[-1].normalize() + pd.Timedelta(days=7)

        # Get schedule (~250 sessions/year, very small)
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        # Ensure schedule is in calendar timezone
        if schedule["market_open"].dt.tz is None:
            # Schedule is tz-naive - localize to calendar timezone
            schedule["market_open"] = schedule["market_open"].dt.tz_localize(self.tz)
            schedule["market_close"] = schedule["market_close"].dt.tz_localize(self.tz)
        else:
            # Schedule is tz-aware - convert to calendar timezone
            schedule["market_open"] = schedule["market_open"].dt.tz_convert(self.tz)
            schedule["market_close"] = schedule["market_close"].dt.tz_convert(self.tz)

        # Convert to ns-resolution int64 for fast searchsorted joins.
        ts_ns = timestamps_tz.as_unit("ns").asi8
        market_open_ns = schedule["market_open"].dt.as_unit("ns").to_numpy(dtype=np.int64)
        market_close_ns = schedule["market_close"].dt.as_unit("ns").to_numpy(dtype=np.int64)
        session_dates = schedule.index.to_numpy(dtype="datetime64[ns]")

        order = np.argsort(ts_ns, kind="mergesort")
        ts_sorted = ts_ns[order]

        # Candidate session from most recent market open.
        prev_idx = np.searchsorted(market_open_ns, ts_sorted, side="right") - 1
        assigned_sorted = np.full(len(ts_sorted), np.datetime64("NaT"), dtype="datetime64[ns]")

        valid_prev = prev_idx >= 0
        within_session = np.zeros(len(ts_sorted), dtype=bool)
        if np.any(valid_prev):
            within_session[valid_prev] = (
                ts_sorted[valid_prev] < market_close_ns[prev_idx[valid_prev]]
            )
            valid_assign = valid_prev & within_session
            assigned_sorted[valid_assign] = session_dates[prev_idx[valid_assign]]

        # For timestamps outside session windows, assign to next session.
        outside_mask = ~within_session
        if np.any(outside_mask):
            next_idx = np.searchsorted(market_open_ns, ts_sorted[outside_mask], side="left")
            valid_next = next_idx < len(session_dates)
            outside_positions = np.where(outside_mask)[0]
            assigned_sorted[outside_positions[valid_next]] = session_dates[next_idx[valid_next]]

        # Restore original row order.
        assigned = np.full(len(ts_ns), np.datetime64("NaT"), dtype="datetime64[ns]")
        assigned[order] = assigned_sorted
        return pd.Series(assigned, index=timestamps)

    def get_sessions_and_mask(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> tuple[pd.Series, np.ndarray]:
        """Assign each timestamp to its trading session date and identify non-trading rows.

        Unlike get_sessions(), this method does NOT forward-fill non-trading timestamps.
        Non-trading rows (weekends, holidays) get NaT session and False mask.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Timestamps to assign to sessions (may be tz-naive or tz-aware)

        Returns
        -------
        sessions : pd.Series
            Session dates for each timestamp. Non-trading rows have NaT.
        trading_mask : np.ndarray of bool
            True for rows that fall on trading days, False otherwise.
        """
        # Ensure all timestamps are in calendar timezone
        timestamps_tz = self._ensure_timezone_aware(timestamps)

        # Get schedule for the data period (with buffer for edge cases)
        start_date = timestamps_tz[0].normalize() - pd.Timedelta(days=7)
        end_date = timestamps_tz[-1].normalize() + pd.Timedelta(days=7)

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        # Get the set of trading dates from the schedule
        trading_dates = set(schedule.index.normalize())

        # Normalize each timestamp to its date and check if it's a trading day
        ts_dates = timestamps_tz.normalize()  # type: ignore[unresolved-attribute]
        # For tz-aware DatetimeIndex, we need to compare tz-naive dates
        ts_dates_naive = ts_dates.tz_localize(None) if ts_dates.tz else ts_dates

        trading_mask = np.array([d in trading_dates for d in ts_dates_naive], dtype=bool)

        # Assign sessions using get_sessions() for trading rows, NaT for non-trading
        # Use positional indexing to handle duplicate timestamps (panel data)
        session_values = np.full(len(timestamps), np.datetime64("NaT"), dtype="datetime64[ns]")
        if trading_mask.any():
            trading_timestamps = timestamps[trading_mask]
            trading_sessions = self.get_sessions(trading_timestamps)
            session_values[trading_mask] = trading_sessions.values

        sessions = pd.Series(session_values, index=timestamps)
        return sessions, trading_mask

    def time_spec_to_sessions(self, spec: str) -> int:
        """Convert a time-period string to a trading session count.

        Uses the calendar's schedule for a representative year to produce
        accurate counts for month/year specifications.

        Parameters
        ----------
        spec : str
            Time period specification, e.g. "1D", "4W", "3M", "1Y".

        Returns
        -------
        int
            Number of trading sessions in the specified period.

        Examples
        --------
        >>> cal = TradingCalendar("NYSE")
        >>> cal.time_spec_to_sessions("1D")
        1
        >>> cal.time_spec_to_sessions("4W")
        20
        >>> cal.time_spec_to_sessions("1Y")
        252
        """
        import re

        match = re.match(r"(\d+)([DWMY])", spec.upper())
        if not match:
            raise ValueError(
                f"Invalid time specification '{spec}'. Use format like '1D', '4W', '3M', '1Y'"
            )

        n = int(match.group(1))
        freq = match.group(2)

        if freq == "D":
            return n

        # Query a representative year for session counts
        ref_year = 2024
        year_start = pd.Timestamp(f"{ref_year}-01-01")
        year_end = pd.Timestamp(f"{ref_year}-12-31")
        schedule = self.calendar.schedule(start_date=year_start, end_date=year_end)
        sessions_per_year = len(schedule)

        if freq == "W":
            # Standard trading week = 5 sessions (Mon-Fri)
            # Use 5 rather than yearly average (252/52 ≈ 4.85) because
            # "4 weeks" conventionally means 20 trading sessions.
            return 5 * n
        elif freq == "M":
            sessions_per_month = sessions_per_year / 12
            return int(round(sessions_per_month * n))
        elif freq == "Y":
            return int(round(sessions_per_year * n))

        raise ValueError(f"Unsupported frequency: {freq}")

    def count_samples_in_period(
        self,
        timestamps: pd.DatetimeIndex,
        period_spec: str,
    ) -> list[int]:
        """Count samples in complete calendar periods across the dataset.

        This method identifies complete periods (sessions, weeks, months) and counts
        samples in each, providing the basis for calendar-aware fold creation.

        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            Full dataset timestamps (may be tz-naive or tz-aware)
        period_spec : str
            Period specification (e.g., '1D', '4W', '3M')

        Returns
        -------
        list[int]
            Sample counts for each complete period found

        Notes
        -----
        For intraday data with 'D' spec: Returns samples per session
        For intraday data with 'W' spec: Returns samples per trading week
        For daily data: Returns samples per calendar period
        """
        import re

        # Ensure timezone-aware
        timestamps_tz = self._ensure_timezone_aware(timestamps)

        # Parse period specification (D/W/M/Y supported)
        match = re.match(r"(\d+)([DWMY])", period_spec.upper())
        if not match:
            raise ValueError(
                f"Invalid period specification '{period_spec}'. Use format like '1D', '4W', '3M', '1Y'"
            )

        n_periods = int(match.group(1))
        freq = match.group(2)

        # Normalize Y → M (1Y = 12M, 10Y = 120M)
        if freq == "Y":
            n_periods = n_periods * 12
            freq = "M"

        # Determine if data is intraday (multiple samples per day)
        # Cast to Any for DatetimeIndex.normalize() which is valid but type stubs don't recognize
        normalized_days = np.asarray(cast(Any, timestamps_tz).normalize())
        _, daily_counts = np.unique(normalized_days, return_counts=True)
        is_intraday = bool(np.any(daily_counts > 1))

        if is_intraday and freq in ["D", "W"]:
            # Use trading calendar sessions
            return self._count_samples_by_sessions(timestamps_tz, freq, n_periods)
        else:
            # Use calendar periods for daily data or monthly specs
            return self._count_samples_by_calendar(timestamps_tz, freq, n_periods)

    def _count_samples_by_sessions(
        self,
        timestamps: pd.DatetimeIndex,
        freq: str,
        n_periods: int,
    ) -> list[int]:
        """Count samples by trading sessions.

        For 'D': Each session is one period
        For 'W': Each n_periods sessions form one period (e.g., 5 sessions = 1 week)
        """
        # Assign each timestamp to its session
        sessions = self.get_sessions(timestamps)

        # Get unique sessions in order
        unique_sessions = np.sort(cast(Any, sessions.unique()))

        if freq == "D":
            # Each session is one period
            sample_counts = []
            for session in unique_sessions:
                count = (sessions == session).sum()
                sample_counts.append(count)
            return sample_counts

        elif freq == "W":
            # Group sessions into weeks, then count samples in n_periods weeks
            # For '4W': 4 weeks × 5 sessions/week = 20 sessions per period
            # Standard trading week = 5 sessions (Mon-Fri)
            sessions_per_week = 5
            sessions_per_period = sessions_per_week * n_periods  # e.g., 5 × 4 = 20

            sample_counts = []
            for i in range(0, len(unique_sessions), sessions_per_period):
                period_sessions = unique_sessions[i : i + sessions_per_period]
                if len(period_sessions) == sessions_per_period:
                    # Only count complete periods (complete 4-week blocks)
                    count = sessions.isin(period_sessions).sum()
                    sample_counts.append(count)
            return sample_counts

        return []

    def _count_samples_by_calendar(
        self,
        timestamps: pd.DatetimeIndex,
        freq: str,
        n_periods: int,
    ) -> list[int]:
        """Count samples by calendar periods (for daily data or monthly specs).

        Groups timestamps into blocks of ``n_periods`` calendar units and
        counts samples in each complete block.
        """
        ts_naive = timestamps.tz_localize(None) if timestamps.tz is not None else timestamps
        day_ns = 24 * 60 * 60 * 1_000_000_000
        day_keys = ts_naive.normalize().asi8 // day_ns

        # Group by calendar period (atomic unit) with integer keys to avoid
        # expensive Period conversions and timezone warnings.
        period_keys: np.ndarray
        if freq == "D":
            period_keys = day_keys
        elif freq == "W":
            weekday = ts_naive.weekday.to_numpy(dtype=np.int64, copy=False)
            period_keys = day_keys - weekday
        elif freq == "M":
            year = ts_naive.year.to_numpy(dtype=np.int64, copy=False)
            month = ts_naive.month.to_numpy(dtype=np.int64, copy=False)
            period_keys = year * 12 + (month - 1)
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        # Count samples per atomic period (sorted by key)
        _, counts_per_unit = np.unique(period_keys, return_counts=True)
        counts_per_unit_list = counts_per_unit.tolist()

        if n_periods <= 1:
            return counts_per_unit_list

        # Aggregate atomic units into blocks of n_periods
        n_units = len(counts_per_unit_list)
        block_counts = []
        for i in range(0, n_units, n_periods):
            block = counts_per_unit_list[i : i + n_periods]
            if len(block) == n_periods:  # Only complete blocks
                block_counts.append(int(sum(block)))

        return block_counts

    def previous_trading_day(
        self,
        from_date: pd.Timestamp | str,
        n: int = 1,
    ) -> pd.Timestamp:
        """Get the nth previous trading day from a given date.

        Parameters
        ----------
        from_date : pd.Timestamp or str
            Reference date (will be converted to date for calendar lookup)
        n : int, default=1
            Number of trading days to go back (must be >= 1)

        Returns
        -------
        pd.Timestamp
            The nth previous trading day (tz-aware in calendar's timezone)

        Examples
        --------
        >>> calendar = TradingCalendar('NYSE')
        >>> # If from_date is Monday 2024-01-08
        >>> calendar.previous_trading_day('2024-01-08', n=1)
        Timestamp('2024-01-05 00:00:00-0500', tz='America/New_York')  # Friday
        >>> calendar.previous_trading_day('2024-01-08', n=5)
        Timestamp('2024-01-02 00:00:00-0500', tz='America/New_York')  # Tue (skips MLK day)
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        # Convert to timestamp if string
        if isinstance(from_date, str):
            from_date = pd.Timestamp(from_date)

        # Normalize to tz-naive date for schedule comparison
        from_date_naive = from_date.tz_localize(None) if from_date.tz else from_date
        from_date_normalized = from_date_naive.normalize()

        # Get schedule for a reasonable lookback period
        # Use 2x the number of days to account for weekends/holidays
        lookback_days = max(n * 2 + 10, 30)
        start_date = from_date_normalized - pd.Timedelta(days=lookback_days)
        end_date = from_date_normalized

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        # Filter to dates strictly before from_date (schedule.index is tz-naive dates)
        valid_dates = schedule.index[schedule.index < from_date_normalized]

        if len(valid_dates) < n:
            raise ValueError(
                f"Not enough trading days in calendar before {from_date}. "
                f"Requested n={n}, found {len(valid_dates)}"
            )

        # Get the nth previous trading day (1-indexed)
        result_date = valid_dates[-n]
        return pd.Timestamp(result_date).tz_localize(self.tz)

    def next_trading_day(
        self,
        from_date: pd.Timestamp | str,
        n: int = 1,
    ) -> pd.Timestamp:
        """Get the nth next trading day from a given date.

        Parameters
        ----------
        from_date : pd.Timestamp or str
            Reference date (will be converted to date for calendar lookup)
        n : int, default=1
            Number of trading days to go forward (must be >= 1)

        Returns
        -------
        pd.Timestamp
            The nth next trading day (tz-aware in calendar's timezone)

        Examples
        --------
        >>> calendar = TradingCalendar('NYSE')
        >>> # If from_date is Friday 2024-01-05
        >>> calendar.next_trading_day('2024-01-05', n=1)
        Timestamp('2024-01-08 00:00:00-0500', tz='America/New_York')  # Monday
        >>> calendar.next_trading_day('2024-01-05', n=5)
        Timestamp('2024-01-12 00:00:00-0500', tz='America/New_York')  # Friday
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        # Convert to timestamp if string
        if isinstance(from_date, str):
            from_date = pd.Timestamp(from_date)

        # Normalize to tz-naive date for schedule comparison
        from_date_naive = from_date.tz_localize(None) if from_date.tz else from_date
        from_date_normalized = from_date_naive.normalize()

        # Get schedule for a reasonable lookahead period
        lookahead_days = max(n * 2 + 10, 30)
        start_date = from_date_normalized
        end_date = from_date_normalized + pd.Timedelta(days=lookahead_days)

        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        # Filter to dates strictly after from_date (schedule.index is tz-naive dates)
        valid_dates = schedule.index[schedule.index > from_date_normalized]

        if len(valid_dates) < n:
            raise ValueError(
                f"Not enough trading days in calendar after {from_date}. "
                f"Requested n={n}, found {len(valid_dates)}"
            )

        # Get the nth next trading day (1-indexed)
        result_date = valid_dates[n - 1]
        return pd.Timestamp(result_date).tz_localize(self.tz)

    def trading_days_between(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> int:
        """Count the number of trading days between two dates (exclusive of end).

        Parameters
        ----------
        start : pd.Timestamp or str
            Start date (inclusive)
        end : pd.Timestamp or str
            End date (exclusive)

        Returns
        -------
        int
            Number of trading days in [start, end)

        Examples
        --------
        >>> calendar = TradingCalendar('NYSE')
        >>> # Monday to Friday (same week)
        >>> calendar.trading_days_between('2024-01-08', '2024-01-12')
        4  # Mon, Tue, Wed, Thu (Fri excluded)
        """
        # Convert to timestamps if strings
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        # Normalize to tz-naive dates for schedule comparison
        start_naive = start.tz_localize(None) if start.tz else start
        end_naive = end.tz_localize(None) if end.tz else end
        start_normalized = start_naive.normalize()
        end_normalized = end_naive.normalize()

        if start_normalized >= end_normalized:
            return 0

        schedule = self.calendar.schedule(
            start_date=start_normalized - pd.Timedelta(days=1),
            end_date=end_normalized + pd.Timedelta(days=1),
        )

        # Count trading days in [start, end)
        mask = (schedule.index >= start_normalized) & (schedule.index < end_normalized)
        return int(mask.sum())


def trading_days_to_timedelta(
    n_trading_days: int,
    calendar: TradingCalendar,
    reference_date: pd.Timestamp,
    direction: str = "backward",
) -> pd.Timedelta:
    """Convert trading days to a calendar timedelta.

    This function calculates the actual calendar time span that covers
    a specified number of trading days, accounting for weekends and holidays.

    Parameters
    ----------
    n_trading_days : int
        Number of trading days
    calendar : TradingCalendar
        Trading calendar instance
    reference_date : pd.Timestamp
        Reference date from which to calculate
    direction : {"backward", "forward"}, default="backward"
        Direction to calculate from reference_date:
        - "backward": Calculate timedelta to reach n trading days before reference_date
        - "forward": Calculate timedelta to reach n trading days after reference_date

    Returns
    -------
    pd.Timedelta
        Calendar timedelta spanning n_trading_days

    Examples
    --------
    >>> calendar = TradingCalendar('NYSE')
    >>> ref = pd.Timestamp('2024-01-15', tz='UTC')
    >>> # 5 trading days backward might span 7 calendar days (weekend)
    >>> delta = trading_days_to_timedelta(5, calendar, ref, 'backward')
    >>> delta
    Timedelta('7 days 00:00:00')

    Notes
    -----
    This is useful for converting label_horizon from "5 trading days" to an
    actual timedelta that can be used for purging calculations. The timedelta
    will vary based on the reference date due to weekends and holidays.
    """
    if n_trading_days <= 0:
        return pd.Timedelta(0)

    if direction == "backward":
        target_date = calendar.previous_trading_day(reference_date, n=n_trading_days)
        return reference_date - target_date
    elif direction == "forward":
        target_date = calendar.next_trading_day(reference_date, n=n_trading_days)
        return target_date - reference_date
    else:
        raise ValueError(f"direction must be 'backward' or 'forward', got '{direction}'")


def parse_time_size_calendar_aware(
    size_spec: str,
    timestamps: pd.DatetimeIndex,
    calendar: TradingCalendar | None = None,
) -> int:
    """Parse time-based size specification using calendar-aware logic.

    This function replaces the naive sample-counting approach with proper
    calendar-based selection that respects trading session boundaries.

    Parameters
    ----------
    size_spec : str
        Time period specification (e.g., '4W', '1D', '3M')
    timestamps : pd.DatetimeIndex
        Timestamps from the dataset
    calendar : TradingCalendar, optional
        Trading calendar to use. If None, uses naive time-based calculation.

    Returns
    -------
    int
        Number of samples corresponding to the time period

    Notes
    -----
    Key difference from naive approach:
    - Naive: Computes median samples/period, returns fixed count
    - Calendar-aware: Returns sample count for actual calendar period

    For activity-based data (dollar bars, trade bars), the calendar-aware
    approach correctly allows sample counts to vary by market activity.

    Examples
    --------
    >>> timestamps = pd.date_range('2024-01-01', periods=10000, freq='1min')
    >>> calendar = TradingCalendar('CME_Equity')
    >>> # Returns samples in exactly 4 trading weeks
    >>> n_samples = parse_time_size_calendar_aware('4W', timestamps, calendar)
    """
    if calendar is None:
        # Fallback to naive time-based calculation
        return _parse_time_size_naive(size_spec, timestamps)

    # Use calendar-aware counting
    sample_counts = calendar.count_samples_in_period(timestamps, size_spec)

    if not sample_counts:
        raise ValueError(
            f"Could not find any complete periods matching '{size_spec}' in the provided timestamps"
        )

    # Use median sample count as representative value
    # This handles variability in activity-based data (dollar/trade bars)
    median_count = int(np.median(sample_counts))

    return median_count


def _parse_time_size_naive(
    size_spec: str,
    timestamps: pd.DatetimeIndex,
) -> int:
    """Naive time-based size calculation (fallback when no calendar provided).

    This is the original ml4t-diagnostic logic - kept for backward compatibility.
    """

    # Parse the time period
    try:
        time_delta = pd.Timedelta(size_spec)
    except ValueError:
        try:
            # pandas deprecated plain "M" month-end alias in favor of "ME".
            # Keep backward compatibility for inputs like "1M", "3M".
            normalized_spec = size_spec
            month_match = re.fullmatch(r"(\d+)M", size_spec.strip().upper())
            if month_match:
                normalized_spec = f"{month_match.group(1)}ME"
            offset = pd.tseries.frequencies.to_offset(normalized_spec)
            ref_date = timestamps[0]
            time_delta = (ref_date + offset) - ref_date
        except Exception as e:
            raise ValueError(
                f"Invalid time specification '{size_spec}'. "
                f"Use pandas offset aliases like '4W', '30D', '3M', '1Y'. "
                f"Error: {e}"
            ) from e

    # Simple proportion-based calculation
    total_duration = timestamps[-1] - timestamps[0]
    if total_duration.total_seconds() == 0:
        raise ValueError("Cannot calculate time-based size for single-timestamp data")

    n_samples = len(timestamps)
    samples_per_second = n_samples / total_duration.total_seconds()
    size_in_samples = int(samples_per_second * time_delta.total_seconds())

    return size_in_samples
