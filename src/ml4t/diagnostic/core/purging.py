"""Core purging and embargo functionality for time-series cross-validation.

This module implements the fundamental algorithms for preventing data leakage
in financial time-series validation through purging (removing training samples
whose labels overlap with test data) and embargo (adding gaps to account for
serial correlation).

Based on López de Prado (2018) "Advances in Financial Machine Learning".
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ml4t.diagnostic.splitters.calendar import TradingCalendar


def _normalize_timezone_aware_inputs(
    timestamps: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    """Validate timezone awareness and normalize all inputs to UTC."""
    if timestamps.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    if test_start.tz is None:
        raise ValueError("test_start must be timezone-aware")
    if test_end.tz is None:
        raise ValueError("test_end must be timezone-aware")

    return (
        timestamps.tz_convert("UTC"),
        test_start.tz_convert("UTC"),
        test_end.tz_convert("UTC"),
    )


def calculate_purge_indices(
    n_samples: int | None = None,
    test_start: int | pd.Timestamp | None = None,
    test_end: int | pd.Timestamp | None = None,
    label_horizon: int | pd.Timedelta = 0,
    timestamps: pd.DatetimeIndex | None = None,
    calendar: TradingCalendar | None = None,
) -> list[int]:
    """Calculate indices to purge from training set to prevent label leakage.

    Purging removes training samples whose labels could contain information
    from the test period. If a feature at time t is used to predict a label
    that depends on information up to time t+h, we must remove training
    samples from [test_start - h, test_start).

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples when using integer indices.

    test_start : int or pandas.Timestamp
        Start index/time of test period.

    test_end : int or pandas.Timestamp
        End index/time of test period (exclusive).

    label_horizon : int or pandas.Timedelta, default=0
        Forward-looking period of labels. For example, if predicting
        20-day returns, label_horizon=20 (days).

        When timestamps and calendar are provided and label_horizon is int,
        it is interpreted as trading days (not calendar days).

    timestamps : pandas.DatetimeIndex, optional
        Timestamps for each sample when using time-based indices.

    calendar : TradingCalendar, optional
        Trading calendar for trading-day-aware conversion.
        When provided with integer label_horizon and timestamps,
        label_horizon is interpreted as trading days.

    Returns:
    -------
    purged_indices : list of int
        Integer positions of samples to remove from training set.

    Examples:
    --------
    >>> # Integer indices
    >>> purged = calculate_purge_indices(
    ...     n_samples=100, test_start=50, test_end=60, label_horizon=5
    ... )
    >>> purged
    [45, 46, 47, 48, 49]

    >>> # Timestamp indices
    >>> times = pd.date_range("2020-01-01", periods=100, freq="D")
    >>> purged = calculate_purge_indices(
    ...     timestamps=times,
    ...     test_start=times[50],
    ...     test_end=times[60],
    ...     label_horizon=pd.Timedelta("5D")
    ... )

    >>> # Trading-day-aware purging
    >>> from ml4t.diagnostic.splitters.calendar import TradingCalendar
    >>> calendar = TradingCalendar('NYSE')
    >>> purged = calculate_purge_indices(
    ...     timestamps=times,
    ...     test_start=times[50],
    ...     test_end=times[60],
    ...     label_horizon=5,  # 5 TRADING days
    ...     calendar=calendar
    ... )
    """
    if timestamps is not None:
        # Time-based purging
        if not isinstance(test_start, pd.Timestamp) or not isinstance(
            test_end,
            pd.Timestamp,
        ):
            raise TypeError(
                "test_start and test_end must be Timestamps when using timestamps",
            )

        timestamps, test_start, test_end = _normalize_timezone_aware_inputs(
            timestamps, test_start, test_end
        )

        if not isinstance(label_horizon, pd.Timedelta):
            # label_horizon is int - need to convert
            if calendar is not None:
                # Trading-day-aware conversion
                from ml4t.diagnostic.splitters.calendar import trading_days_to_timedelta

                label_horizon = trading_days_to_timedelta(
                    n_trading_days=label_horizon,
                    calendar=calendar,
                    reference_date=test_start,
                    direction="backward",
                )
            else:
                # Fallback to calendar days with warning
                if label_horizon > 0:
                    warnings.warn(
                        f"label_horizon={label_horizon} (int) with timestamps but no calendar. "
                        "Interpreting as calendar days. For trading days, provide a calendar.",
                        UserWarning,
                        stacklevel=2,
                    )
                label_horizon = pd.Timedelta(days=label_horizon)

        # Calculate purge start time
        purge_start_time = test_start - label_horizon

        # Find indices to purge
        purge_mask = (timestamps >= purge_start_time) & (timestamps < test_start)
        purged_indices = np.where(purge_mask)[0].tolist()

    else:
        # Integer-based purging
        if n_samples is None:
            raise ValueError("n_samples required for integer-based purging")

        # In this branch, test_start and label_horizon are integers
        test_start_int = cast(int, test_start)
        label_horizon_int = cast(int, label_horizon)

        # Calculate purge start
        purge_start = max(0, test_start_int - label_horizon_int)

        # Indices to purge are [purge_start, test_start)
        purged_indices = list(range(purge_start, test_start_int))

    return purged_indices


def calculate_embargo_indices(
    n_samples: int | None = None,
    test_start: int | pd.Timestamp | None = None,
    test_end: int | pd.Timestamp | None = None,
    embargo_size: int | pd.Timedelta | None = None,
    embargo_pct: float | None = None,
    timestamps: pd.DatetimeIndex | None = None,
) -> list[int]:
    """Calculate indices to embargo after test set to prevent serial correlation.

    Embargo removes training samples immediately after the test set to account
    for serial correlation in predictions. This prevents the model from learning
    patterns that persist across the test/train boundary.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples when using integer indices.

    test_start : int or pandas.Timestamp
        Start index/time of test period.

    test_end : int or pandas.Timestamp
        End index/time of test period (exclusive).

    embargo_size : int or pandas.Timedelta, optional
        Size of embargo period after test set.

    embargo_pct : float, optional
        Embargo size as percentage of total samples.
        Either embargo_size or embargo_pct should be specified.

    timestamps : pandas.DatetimeIndex, optional
        Timestamps for each sample when using time-based indices.

    Returns:
    -------
    embargo_indices : list of int
        Integer positions of samples to embargo.

    Examples:
    --------
    >>> # Fixed embargo size
    >>> embargoed = calculate_embargo_indices(
    ...     n_samples=100, test_start=50, test_end=60, embargo_size=5
    ... )
    >>> embargoed
    [60, 61, 62, 63, 64]

    >>> # Percentage embargo
    >>> embargoed = calculate_embargo_indices(
    ...     n_samples=100, test_start=50, test_end=60, embargo_pct=0.05
    ... )
    """
    if embargo_size is None and embargo_pct is None:
        return []

    if embargo_size is not None and embargo_pct is not None:
        raise ValueError("Specify either embargo_size or embargo_pct, not both")

    if timestamps is not None:
        # Time-based embargo
        if not isinstance(test_start, pd.Timestamp) or not isinstance(
            test_end,
            pd.Timestamp,
        ):
            raise TypeError(
                "test_start and test_end must be Timestamps when using timestamps",
            )

        timestamps, test_start, test_end = _normalize_timezone_aware_inputs(
            timestamps, test_start, test_end
        )

        # Calculate embargo size if percentage given
        if embargo_pct is not None:
            total_duration = timestamps[-1] - timestamps[0]
            embargo_size = total_duration * embargo_pct

        if not isinstance(embargo_size, pd.Timedelta):
            # Convert integer days to Timedelta
            embargo_size = pd.Timedelta(days=cast(int, embargo_size))

        # Calculate embargo end time
        embargo_end_time = test_end + embargo_size

        # Find indices to embargo
        embargo_mask = (timestamps >= test_end) & (timestamps < embargo_end_time)
        embargo_indices = np.where(embargo_mask)[0].tolist()

    else:
        # Integer-based embargo
        if n_samples is None:
            raise ValueError("n_samples required for integer-based embargo")

        # Calculate embargo size if percentage given
        if embargo_pct is not None:
            embargo_size = int(n_samples * embargo_pct)

        # Calculate embargo end
        # Either embargo_size was provided or calculated from embargo_pct
        assert embargo_size is not None
        # In this branch, test_end and embargo_size are integers
        test_end_int = cast(int, test_end)
        embargo_size_int = cast(int, embargo_size)
        embargo_end = min(n_samples, test_end_int + embargo_size_int)

        # Indices to embargo are [test_end, embargo_end)
        embargo_indices = list(range(test_end_int, embargo_end))

    return embargo_indices


def apply_purging_and_embargo(
    train_indices: NDArray[np.intp],
    test_start: int | pd.Timestamp,
    test_end: int | pd.Timestamp,
    label_horizon: int | pd.Timedelta = 0,
    embargo_size: int | pd.Timedelta | None = None,
    embargo_pct: float | None = None,
    n_samples: int | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    calendar: TradingCalendar | None = None,
) -> NDArray[np.intp]:
    """Apply both purging and embargo to training indices.

    This is a convenience function that combines purging and embargo
    to clean a set of training indices, removing any that could lead
    to data leakage or serial correlation issues.

    Parameters
    ----------
    train_indices : numpy.ndarray
        Initial training indices before purging/embargo.

    test_start : int or pandas.Timestamp
        Start index/time of test period.

    test_end : int or pandas.Timestamp
        End index/time of test period (exclusive).

    label_horizon : int or pandas.Timedelta, default=0
        Forward-looking period of labels.
        When timestamps and calendar are provided and label_horizon is int,
        it is interpreted as trading days (not calendar days).

    embargo_size : int or pandas.Timedelta, optional
        Size of embargo period after test set.

    embargo_pct : float, optional
        Embargo size as percentage of total samples.

    n_samples : int, optional
        Total number of samples (required for integer indices).

    timestamps : pandas.DatetimeIndex, optional
        Timestamps for each sample when using time-based indices.

    calendar : TradingCalendar, optional
        Trading calendar for trading-day-aware conversion.
        When provided with integer label_horizon and timestamps,
        label_horizon is interpreted as trading days.

    Returns:
    -------
    clean_indices : numpy.ndarray
        Training indices after removing purged and embargoed samples.

    Examples:
    --------
    >>> train = np.arange(100)
    >>> clean = apply_purging_and_embargo(
    ...     train_indices=train,
    ...     test_start=50,
    ...     test_end=60,
    ...     label_horizon=5,
    ...     embargo_size=5,
    ...     n_samples=100
    ... )
    >>> # Removes [45,50) for purging and [60,65) for embargo
    >>> len(clean)
    85
    """
    if timestamps is not None:
        if not isinstance(test_start, pd.Timestamp) or not isinstance(test_end, pd.Timestamp):
            raise TypeError("test_start and test_end must be Timestamps when using timestamps")
        timestamps, test_start, test_end = _normalize_timezone_aware_inputs(
            timestamps, test_start, test_end
        )

    # Calculate indices to remove - convert to numpy arrays immediately
    purged_list = calculate_purge_indices(
        n_samples=n_samples,
        test_start=test_start,
        test_end=test_end,
        label_horizon=label_horizon,
        timestamps=timestamps,
        calendar=calendar,
    )
    purged_arr = np.asarray(purged_list, dtype=np.intp)

    embargoed_list = calculate_embargo_indices(
        n_samples=n_samples,
        test_start=test_start,
        test_end=test_end,
        embargo_size=embargo_size,
        embargo_pct=embargo_pct,
        timestamps=timestamps,
    )
    embargoed_arr = np.asarray(embargoed_list, dtype=np.intp)

    # Also remove test indices themselves
    if timestamps is not None:
        # Use searchsorted for more robust boundary handling
        test_start_idx = timestamps.searchsorted(test_start, side="left")
        test_end_idx = timestamps.searchsorted(test_end, side="left")
        test_arr = np.arange(test_start_idx, test_end_idx, dtype=np.intp)
    else:
        # When timestamps is None, test_start/test_end are integer indices
        # Accept both Python int and numpy integer types
        assert isinstance(test_start, int | np.integer), f"Expected int, got {type(test_start)}"
        assert isinstance(test_end, int | np.integer), f"Expected int, got {type(test_end)}"
        test_arr = np.arange(int(test_start), int(test_end), dtype=np.intp)

    # Combine all indices to remove using numpy (faster than Python sets)
    # Filter out empty arrays before concatenating
    arrays_to_concat = [arr for arr in (purged_arr, embargoed_arr, test_arr) if len(arr) > 0]
    if arrays_to_concat:
        remove_indices = np.unique(np.concatenate(arrays_to_concat))
    else:
        remove_indices = np.array([], dtype=np.intp)

    # Keep only indices not in remove set
    clean_mask = ~np.isin(train_indices, remove_indices)
    clean_indices = train_indices[clean_mask]

    return clean_indices
