"""Explicit legacy observation-clock conversions, not exchange-grid validation.

These conventions support analytical migration and supplied expiry keys. An
inferred weekday mask is not an authoritative schedule, and its duration label
is not necessarily physical elapsed time, particularly across DST transitions.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from datetime import date
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset, BusinessDay, CustomBusinessDay, Day


def make_naive_ts(t: pd.Timestamp) -> pd.Timestamp:
    """Remove an aware timestamp's timezone after conversion to UTC, not local time."""
    return t.tz_convert(None) if t.tzinfo is not None else t.tz_localize(None)


def infer_trading_calendar(
    factor_idx: pd.DatetimeIndex, prices_idx: pd.DatetimeIndex
) -> CustomBusinessDay:
    """Infer the named legacy weekday/holiday convention from observed labels.

    Missing observed weekdays are not proof of exchange closures. This function
    does not resolve a FeedSpec calendar or validate trading sessions or bars.
    """
    dates = factor_idx.union(prices_idx)
    if dates.empty:
        raise ValueError("observed-calendar inference requires at least one date")
    weekdays, holidays = [], []
    for day, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        observed = dates[dates.dayofweek == day].normalize()
        if not len(observed):
            continue
        weekdays.append(name)
        expected = pd.date_range(
            dates.min(), dates.max(), freq=CustomBusinessDay(weekmask=name)
        ).normalize()
        holidays.extend(expected.difference(observed).date)
    return CustomBusinessDay(weekmask=" ".join(weekdays), holidays=holidays)


def add_custom_calendar_timedelta(
    input: pd.Timestamp | pd.DatetimeIndex,
    timedelta: pd.Timedelta,
    freq: Day | BusinessDay | CustomBusinessDay,
) -> pd.Timestamp | pd.DatetimeIndex:
    """Apply declared daily offsets and then the subday remainder.

    Business-day offsets preserve local wall-clock time. The declared pandas
    offset's installed-version semantics apply: Day became a calendar-day in
    pandas3, whereas older Day offsets are 24-hour ticks. The remainder is elapsed
    time. No exchange schedule is inferred.
    """
    if not isinstance(freq, Day | BusinessDay | CustomBusinessDay):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")
    days = timedelta.components.days
    remainder = timedelta - pd.Timedelta(days, unit="D")
    return input + days * freq + remainder


def diff_custom_calendar_timedeltas(
    start: pd.Timestamp, end: pd.Timestamp, freq: BaseOffset
) -> pd.Timedelta:
    """Replace elapsed whole days with the legacy observed-calendar count.

    Daily offsets count UTC-naive date labels. Other explicit offsets count
    generated anchors. The physical subday remainder is retained. Across DST
    this is deliberately not the inverse of local-wall-clock offset addition.
    """
    if not isinstance(freq, BaseOffset):
        raise ValueError("freq must be a pandas DateOffset")
    mask, holidays = getattr(freq, "weekmask", None), getattr(freq, "holidays", None)
    if mask is None and isinstance(freq, Day | BusinessDay):
        mask = "1111111" if isinstance(freq, Day) else "1111100"
        holidays = []
    if mask is not None and holidays is not None:
        count = int(
            np.busday_count(
                make_naive_ts(start).date(),
                make_naive_ts(end).date(),
                weekmask=mask,
                holidays=np.asarray(holidays, dtype="datetime64[D]"),
            )
        )
    else:
        count = len(pd.date_range(start, end, freq=freq)) - 1 - int(not freq.is_on_offset(start))
    elapsed = end - start
    return elapsed - pd.Timedelta(elapsed.components.days - count, unit="D")


def timedelta_to_string(timedelta: pd.Timedelta) -> str:
    """Format exact duration components without collapsing nanoseconds to days."""
    if pd.isna(timedelta):
        raise ValueError("duration must be finite")
    if timedelta < pd.Timedelta(0):
        return "-" + timedelta_to_string(-timedelta)
    components = timedelta.components
    return "".join(
        f"{getattr(components, name)}{unit}"
        for name, unit in (
            ("days", "D"),
            ("hours", "h"),
            ("minutes", "m"),
            ("seconds", "s"),
            ("milliseconds", "ms"),
            ("microseconds", "us"),
            ("nanoseconds", "ns"),
        )
        if getattr(components, name)
    )


def timedelta_strings_to_integers(sequence: Sequence[str]) -> list[int]:
    """Return legacy floored day components, not safe automatic turnover lags."""
    return [pd.Timedelta(value).days for value in sequence]


def get_forward_returns_columns(
    columns: pd.Index | Sequence[str], require_exact_day_multiple: bool = False
) -> pd.Index:
    """Select legacy duration-shaped names, preserving labels and order.

    Native supplied-selector analysis can also accept arbitrary non-duration
    names. This helper only supplies the legacy duration-name convenience.
    """
    index = columns.copy(deep=True) if isinstance(columns, pd.Index) else pd.Index(columns)
    pattern = r"(?:\d+D)+" if require_exact_day_multiple else r"(?:\d+(?:[Dhms]|ms|us|ns))+"
    selected = [
        isinstance(value, str) and re.fullmatch(pattern, value, re.IGNORECASE) is not None
        for value in index
    ]
    if require_exact_day_multiple and not all(selected):
        warnings.warn(
            "Skipping return periods that are not exact day multiples.", UserWarning, stacklevel=2
        )
    return index[selected]


def turnover_periods_from_columns(columns: pd.Index | Sequence[str]) -> list[int]:
    """Select only positive exact-day legacy labels; subday labels are not zero lags."""
    return [
        days
        for days in timedelta_strings_to_integers(get_forward_returns_columns(columns, True))
        if days > 0
    ]


def backshift_returns_series(series: pd.Series, N: int) -> pd.Series:
    """Rekey backward observations on their common declared date-level clock.

    Values and missing asset rows are preserved; no per-asset compression,
    price inference or return calculation occurs. Zero shift is an identity;
    an overlong clock produces a typed empty Series instead of an index error.
    """
    if isinstance(N, bool) or not isinstance(N, Integral) or N < 0:
        raise ValueError("N must be a nonnegative observation count")
    if (
        not isinstance(series, pd.Series)
        or not isinstance(series.index, pd.MultiIndex)
        or series.index.nlevels != 2
    ):
        raise TypeError("returns require a two-level date/asset Series")
    if not isinstance(series.index.levels[0], pd.DatetimeIndex) and not all(
        type(value) is date for value in series.index.levels[0]
    ):
        raise TypeError("the first return index level must contain dates")
    if not series.index.is_unique or (np.asarray(series.index.codes) < 0).any():
        raise ValueError("return date/asset keys must be unique and nonmissing")
    dates, assets = series.index.levels
    date_codes = np.asarray(series.index.codes[0], dtype=np.int64)
    asset_codes = np.asarray(series.index.codes[1])
    shift = min(int(N), len(dates))
    retained = date_codes >= shift
    result = series.iloc[np.flatnonzero(retained)].copy(deep=True)
    result.index = pd.MultiIndex(
        levels=[dates[: max(0, len(dates) - shift)], assets],
        codes=[date_codes[retained] - shift, asset_codes[retained]],
        names=series.index.names,
    )
    result.name = series.name if N == 0 else None
    return result


__all__ = [
    "make_naive_ts",
    "infer_trading_calendar",
    "add_custom_calendar_timedelta",
    "diff_custom_calendar_timedeltas",
    "timedelta_to_string",
    "timedelta_strings_to_integers",
    "get_forward_returns_columns",
    "turnover_periods_from_columns",
    "backshift_returns_series",
]
