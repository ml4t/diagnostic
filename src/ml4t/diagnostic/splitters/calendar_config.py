"""Configuration for calendar-aware cross-validation.

This module defines configuration schemas for trading calendar integration,
ensuring proper timezone handling and session awareness.
"""

from pydantic import BaseModel, Field


class CalendarConfig(BaseModel):
    """Configuration for trading calendar in cross-validation.

    This configuration ensures proper handling of:
    - Trading sessions (don't split session boundaries)
    - Timezones (consistent tz-aware comparisons)
    - Market-specific calendars (CME, NYSE, LSE, etc.)

    Attributes
    ----------
    exchange : str
        Name of the exchange calendar from pandas_market_calendars.
        Examples: 'CME_Equity', 'NYSE', 'LSE', 'TSX', 'HKEX'
        See: https://pandas-market-calendars.readthedocs.io/

    timezone : str, default='UTC'
        Timezone for calendar operations. All timestamps will be converted
        to this timezone for calendar comparisons.
        - 'UTC': Universal Coordinated Time (default, safest)
        - 'America/New_York': US Eastern (NYSE, NASDAQ)
        - 'America/Chicago': US Central (CME futures)
        - 'Europe/London': UK (LSE)
        - See pytz documentation for full list

    localize_naive : bool, default=True
        If True, tz-naive data will be localized to the specified timezone.
        If False, tz-naive data will raise an error.
        Recommended: True for safety (assumes data is in calendar timezone)

    Examples
    --------
    For CME futures (NQ, ES, etc.):
    >>> config = CalendarConfig(
    ...     exchange='CME_Equity',
    ...     timezone='America/Chicago'
    ... )

    For US equities:
    >>> config = CalendarConfig(
    ...     exchange='NYSE',
    ...     timezone='America/New_York'
    ... )

    For international markets:
    >>> config = CalendarConfig(
    ...     exchange='LSE',
    ...     timezone='Europe/London'
    ... )
    """

    exchange: str = Field(..., description="Exchange calendar name from pandas_market_calendars")

    timezone: str = Field(
        default="UTC", description="Timezone for calendar operations (pytz timezone name)"
    )

    localize_naive: bool = Field(
        default=True, description="Whether to localize tz-naive data to the specified timezone"
    )

    class Config:
        """Pydantic configuration."""

        frozen = True  # Immutable after creation

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CalendarConfig(exchange='{self.exchange}', "
            f"timezone='{self.timezone}', "
            f"localize_naive={self.localize_naive})"
        )


# Preset configurations for common markets
CME_CONFIG = CalendarConfig(exchange="CME_Equity", timezone="America/Chicago", localize_naive=True)

NYSE_CONFIG = CalendarConfig(exchange="NYSE", timezone="America/New_York", localize_naive=True)

NASDAQ_CONFIG = CalendarConfig(exchange="NASDAQ", timezone="America/New_York", localize_naive=True)

LSE_CONFIG = CalendarConfig(exchange="LSE", timezone="Europe/London", localize_naive=True)
