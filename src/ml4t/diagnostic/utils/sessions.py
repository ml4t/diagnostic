"""Session assignment utilities for financial time-series data.

This module provides utilities to assign session dates to intraday data,
enabling session-aware cross-validation where sessions are the atomic unit.
"""

import pandas as pd

try:
    import pandas_market_calendars as mcal  # noqa: F401 (availability check)

    HAS_MARKET_CALENDARS = True
except ImportError:
    HAS_MARKET_CALENDARS = False


def assign_session_dates(
    df: pd.DataFrame,
    calendar: str = "CME_Equity",
    timezone: str = "UTC",
    session_column: str = "session_date",
) -> pd.DataFrame:
    """Assign trading session dates to intraday data.

    This function adds a session_date column to the DataFrame, where each
    timestamp is assigned to its trading session. Sessions are atomic units
    for cross-validation - we don't split within a session.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (may be tz-naive or tz-aware)
    calendar : str, default='CME_Equity'
        Name of pandas_market_calendars calendar
        Examples: 'CME_Equity', 'NYSE', 'LSE', 'TSX'
    timezone : str, default='UTC'
        Timezone for calendar operations
    session_column : str, default='session_date'
        Name of the column to add with session dates

    Returns
    -------
    pd.DataFrame
        DataFrame with added session_date column

    Notes
    -----
    - For CME futures: Sunday 5pm CT - Friday 4pm CT is one session
    - For US equities: Standard trading day 9:30am - 4pm ET
    - If df already has the session_column, it will be overwritten

    Examples
    --------
    >>> df = pd.read_parquet('nq_data.parquet')  # Has DatetimeIndex
    >>> df = assign_session_dates(df, calendar='CME_Equity', timezone='America/Chicago')
    >>> df.groupby('session_date').size()  # Samples per session

    For data that already has session_date:
    >>> if 'session_date' not in df.columns:
    ...     df = assign_session_dates(df)
    """
    if not HAS_MARKET_CALENDARS:
        raise ImportError(
            "pandas_market_calendars is required for session assignment. "
            "Install with: pip install pandas_market_calendars"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame must have a DatetimeIndex. Got index type: {type(df.index)}")

    # Import here to avoid circular dependency
    from ml4t.diagnostic.splitters.calendar import TradingCalendar
    from ml4t.diagnostic.splitters.calendar_config import CalendarConfig

    # Create calendar configuration
    config = CalendarConfig(exchange=calendar, timezone=timezone, localize_naive=True)

    # Get trading calendar
    trading_calendar = TradingCalendar(config)

    # Assign sessions (vectorized, fast)
    sessions = trading_calendar.get_sessions(df.index)

    # Add as column (copy to avoid modifying original)
    result = df.copy()
    result[session_column] = sessions

    return result


def get_complete_sessions(
    df: pd.DataFrame, session_column: str = "session_date", min_samples: int = 100
) -> pd.Series:
    """Get list of complete sessions with sufficient data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with session_date column
    session_column : str, default='session_date'
        Name of the column containing session dates
    min_samples : int, default=100
        Minimum samples per session to consider complete

    Returns
    -------
    pd.Series
        Session dates that are complete (have >= min_samples)

    Examples
    --------
    >>> df = assign_session_dates(df)
    >>> complete = get_complete_sessions(df, min_samples=500)
    >>> df_clean = df[df['session_date'].isin(complete)]
    """
    if session_column not in df.columns:
        raise ValueError(
            f"DataFrame does not have '{session_column}' column. Run assign_session_dates() first."
        )

    # Count samples per session
    session_counts = df.groupby(session_column).size()

    # Filter to complete sessions
    complete_sessions = session_counts[session_counts >= min_samples].index

    return complete_sessions.to_series(name=session_column)
