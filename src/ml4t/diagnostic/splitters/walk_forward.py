"""Walk-forward cross-validation for time-series data.

This module implements walk-forward cross-validation with optional safeguards
against data leakage from overlapping labels and autocorrelation.

Supports two modes:
1. Traditional walk-forward: Folds step forward through time
2. Held-out test with backward validation: Reserve most recent data for
   final evaluation, validation folds step backward from test boundary
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Generator
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.core.purging import apply_purging_and_embargo
from ml4t.diagnostic.splitters.base import BaseSplitter
from ml4t.diagnostic.splitters.calendar import (
    TradingCalendar,
    parse_time_size_calendar_aware,
)
from ml4t.diagnostic.splitters.calendar_config import CalendarConfig
from ml4t.diagnostic.splitters.config import WalkForwardConfig
from ml4t.diagnostic.splitters.group_isolation import isolate_groups_from_train
from ml4t.diagnostic.splitters.utils import convert_indices_to_timestamps

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WalkForwardCV(BaseSplitter):
    """Walk-forward cross-validator for time-series data.

    Walk-forward CV creates sequential train/test splits where training data
    always precedes test data. Includes optional safeguards against data leakage
    from overlapping labels and autocorrelation.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits to generate.

    test_size : int, float, str, or None, optional
        Size of each test set:
        - If int: number of samples (e.g., 1000)
        - If float: proportion of dataset (e.g., 0.1)
        - If str: time period using pandas offset aliases (e.g., "4W", "30D", "3M")
        - If None: uses 1 / (n_splits + 1)
        Time-based specifications require X to have a DatetimeIndex.

    train_size : int, float, str, or None, optional
        Size of each training set:
        - If int: number of samples (e.g., 10000)
        - If float: proportion of dataset (e.g., 0.5)
        - If str: time period using pandas offset aliases (e.g., "78W", "6M", "2Y")
        - If None: uses all available data before test set
        Time-based specifications require X to have a DatetimeIndex.

    gap : int, default=0
        Gap between training and test set (in addition to label_horizon).

    label_horizon : int or pd.Timedelta, default=0
        How far ahead labels look into the future. Removes training samples whose
        prediction targets overlap with validation/test data.

        Example: If predicting 5-day forward returns, a training sample at day 95
        has a label computed from prices on days 95-100. If validation starts at
        day 98, this training sample's label "sees" validation data, creating leakage.
        Setting label_horizon=5 removes training samples from days 93-97.

    embargo_size : int or pd.Timedelta, optional
        Buffer zone after test periods. For standard walk-forward CV where training
        always precedes test, this has no effect. It is primarily used by CombinatorialCV.

    embargo_pct : float, optional
        Embargo size as percentage of total samples.

    expanding : bool, default=True
        If True, training window expands with each split.
        If False, uses fixed-size rolling window.

    consecutive : bool, default=False
        If True, uses consecutive (back-to-back) test periods with no gaps.
        This is appropriate for walk-forward validation where you want to
        simulate realistic trading with sequential validation periods.
        If False, spreads test periods across the dataset to sample different
        time periods (useful for testing robustness across market regimes).

    calendar : str, CalendarConfig, or TradingCalendar, optional
        Trading calendar for calendar-aware time period calculations.
        - If str: Name of pandas_market_calendars calendar (e.g., 'CME_Equity', 'NYSE')
          Creates default CalendarConfig with UTC timezone
        - If CalendarConfig: Full configuration with exchange, timezone, and options
        - If TradingCalendar: Pre-configured calendar instance
        - If None: Uses naive time-based calculation (backward compatible)

        For intraday data with time-based test_size/train_size (e.g., '4W'),
        using a calendar ensures proper session-aware splitting:
        - Trading sessions are atomic units (won't split Sunday 5pm - Friday 4pm)
        - Handles varying data density in activity-based data (dollar bars, trade bars)
        - Proper timezone handling for tz-naive and tz-aware data
        - '1D' selections: Complete trading sessions
        - '4W' selections: Complete trading weeks (e.g., 4 weeks of 5 sessions each)

        Examples:
        >>> from ml4t.diagnostic.splitters.calendar_config import CME_CONFIG
        >>> cv = WalkForwardCV(test_size='4W', calendar=CME_CONFIG)  # CME futures
        >>> cv = WalkForwardCV(test_size='1W', calendar='NYSE')  # US equities (simple)

    align_to_sessions : bool, default=False
        If True, align fold boundaries to trading session boundaries.
        Requires X to have a session column (specified by session_col parameter).

        Trading sessions should be assigned using the qdata library before cross-validation:
        - Use DataManager with exchange/calendar parameters, or
        - Use SessionAssigner.from_exchange('CME') directly

        When enabled, fold boundaries will never split a trading session, preventing
        subtle lookahead bias in intraday strategies.

    session_col : str, default='session_date'
        Name of the column containing session identifiers.
        Only used if align_to_sessions=True.
        This column should be added by qdata.sessions.SessionAssigner

    isolate_groups : bool, default=False
        If True, prevent the same group (asset/symbol) from appearing in both
        train and test sets. This is critical for multi-asset validation to
        avoid data leakage.

        Requires passing `groups` parameter to split() method with asset IDs.

        Example:
        >>> cv = WalkForwardCV(n_splits=5, isolate_groups=True)
        >>> for train, test in cv.split(df, groups=df['symbol']):
        ...     # train and test will have completely different symbols
        ...     pass

    Attributes:
    ----------
    n_splits_ : int
        The number of splits.

    Examples:
    --------
    >>> import numpy as np
    >>> from ml4t.diagnostic.splitters import WalkForwardCV
    >>> X = np.arange(100).reshape(100, 1)
    >>> cv = WalkForwardCV(n_splits=3, label_horizon=5, embargo_size=2)
    >>> for train, test in cv.split(X):
    ...     print(f"Train: {len(train)}, Test: {len(test)}")
    Train: 17, Test: 25
    Train: 40, Test: 25
    Train: 63, Test: 25
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        *,
        n_splits: int = 5,
        test_size: float | None = None,
        train_size: float | None = None,
        gap: int = 0,
        label_horizon: int | pd.Timedelta = 0,
        embargo_size: int | pd.Timedelta | None = None,
        embargo_pct: float | None = None,
        expanding: bool = True,
        consecutive: bool = False,
        calendar: str | CalendarConfig | TradingCalendar | None = None,
        align_to_sessions: bool = False,
        session_col: str = "session_date",
        timestamp_col: str | None = None,
        isolate_groups: bool = False,
        # New parameters for held-out test
        test_period: int | str | None = None,
        test_start: date | str | None = None,
        test_end: date | str | None = None,
        fold_direction: Literal["forward", "backward"] = "forward",
    ) -> None:
        """Initialize WalkForwardCV.

        This splitter uses a config-first architecture. You can either:
        1. Pass a config object: WalkForwardCV(config=my_config)
        2. Pass individual parameters: WalkForwardCV(n_splits=5, test_size=100)

        Parameters are automatically converted to a config object internally,
        ensuring a single source of truth for all validation and logic.

        Examples
        --------
        >>> # Approach 1: Direct parameters (convenient)
        >>> cv = WalkForwardCV(n_splits=5, test_size=100)
        >>>
        >>> # Approach 2: Config object (for serialization/reproducibility)
        >>> from ml4t.diagnostic.splitters.config import WalkForwardConfig
        >>> config = WalkForwardConfig(n_splits=5, test_size=100)
        >>> cv = WalkForwardCV(config=config)
        >>>
        >>> # Approach 3: With held-out test period
        >>> cv = WalkForwardCV(
        ...     n_splits=5,
        ...     test_period="52D",      # Reserve most recent 52 days for final evaluation
        ...     test_size=20,           # 20-day validation folds
        ...     train_size=252,         # 1-year training windows
        ...     label_horizon=5,        # 5 trading days gap
        ...     calendar="NYSE",        # NYSE trading calendar
        ...     fold_direction="backward",  # Folds step backward from test
        ... )
        >>>
        >>> # Validation folds (step backward from held-out test)
        >>> for train_idx, val_idx in cv.split(X):
        ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
        >>> # Final evaluation on held-out test
        >>> test_score = model.score(X.iloc[cv.test_indices_], y.iloc[cv.test_indices_])
        """
        # Config-first: either use provided config or create from params
        if config is not None:
            # Explicit config provided
            # Verify no conflicting parameters were passed
            non_default_params = []
            if n_splits != 5:
                non_default_params.append("n_splits")
            if test_size is not None:
                non_default_params.append("test_size")
            if train_size is not None:
                non_default_params.append("train_size")
            if gap != 0:
                non_default_params.append("gap")
            if label_horizon != 0:
                non_default_params.append("label_horizon")
            if embargo_size is not None:
                non_default_params.append("embargo_size")
            if embargo_pct is not None:
                non_default_params.append("embargo_pct")
            if not expanding:
                non_default_params.append("expanding")
            if consecutive:
                non_default_params.append("consecutive")
            if calendar is not None:
                non_default_params.append("calendar")
            if align_to_sessions:
                non_default_params.append("align_to_sessions")
            if session_col != "session_date":
                non_default_params.append("session_col")
            if timestamp_col is not None:
                non_default_params.append("timestamp_col")
            if isolate_groups:
                non_default_params.append("isolate_groups")
            if test_period is not None:
                non_default_params.append("test_period")
            if test_start is not None:
                non_default_params.append("test_start")
            if test_end is not None:
                non_default_params.append("test_end")
            if fold_direction != "forward":
                non_default_params.append("fold_direction")

            if non_default_params:
                raise ValueError(
                    f"Cannot specify both 'config' and individual parameters. "
                    f"Got config plus: {', '.join(non_default_params)}"
                )

            self.config = config
        else:
            # Create config from individual parameters
            # Note: embargo_size maps to embargo_td in config
            # Determine calendar_id: explicit str overrides default,
            # CalendarConfig/TradingCalendar handled separately,
            # None means use config default ("NYSE")
            config_kwargs: dict[str, Any] = {
                "n_splits": n_splits,
                "test_size": test_size,
                "train_size": train_size,
                "label_horizon": label_horizon,
                "embargo_td": embargo_size,
                "align_to_sessions": align_to_sessions,
                "session_col": session_col,
                "timestamp_col": timestamp_col,
                "isolate_groups": isolate_groups,
                "test_period": test_period,
                "test_start": test_start,
                "test_end": test_end,
                "fold_direction": fold_direction,
            }
            if isinstance(calendar, str):
                config_kwargs["calendar_id"] = calendar
            elif calendar is not None:
                # CalendarConfig or TradingCalendar — extract exchange name
                if isinstance(calendar, CalendarConfig):
                    config_kwargs["calendar_id"] = calendar.exchange
                elif isinstance(calendar, TradingCalendar):
                    config_kwargs["calendar_id"] = calendar.config.exchange
            # When calendar is None, omit calendar_id to use config default ("NYSE")

            self.config = WalkForwardConfig(**config_kwargs)

        # Handle calendar initialization
        # Use calendar_id from config if no calendar parameter provided
        effective_calendar = calendar
        if effective_calendar is None and self.config.calendar_id is not None:
            effective_calendar = self.config.calendar_id

        if effective_calendar is None:
            self.calendar = None
        elif isinstance(effective_calendar, str | CalendarConfig):
            self.calendar = TradingCalendar(effective_calendar)
        elif isinstance(effective_calendar, TradingCalendar):
            self.calendar = effective_calendar
        else:
            raise TypeError(
                f"calendar must be str, CalendarConfig, TradingCalendar, or None, got {type(effective_calendar)}"
            )

        # Legacy attributes for compatibility with existing split() implementation
        # These reference the config values
        self.gap = gap
        self.embargo_pct = embargo_pct
        self.expanding = expanding
        self.consecutive = consecutive

        # Private state for held-out test (populated after split() is called)
        self._test_indices: np.ndarray | None = None
        self._test_start_idx: int | None = None
        self._test_end_idx: int | None = None

    # Property accessors for config values (clean API)
    @property
    def n_splits(self) -> int:
        """Number of cross-validation folds."""
        return self.config.n_splits

    @property
    def test_size(self) -> int | float | str | None:
        """Test set size specification."""
        return self.config.test_size

    @property
    def train_size(self) -> int | float | str | None:
        """Training set size specification."""
        return self.config.train_size

    @property
    def label_horizon(self) -> int:
        """Forward-looking period of labels."""
        return self.config.label_horizon

    @property
    def embargo_size(self) -> int | None:
        """Embargo buffer size."""
        return self.config.embargo_td

    @property
    def align_to_sessions(self) -> bool:
        """Whether to align fold boundaries to sessions."""
        return self.config.align_to_sessions

    @property
    def session_col(self) -> str:
        """Column name containing session identifiers."""
        return self.config.session_col

    @property
    def timestamp_col(self) -> str | None:
        """Column name containing timestamps for time-based sizes."""
        return self.config.timestamp_col

    @property
    def isolate_groups(self) -> bool:
        """Whether to prevent group overlap between train/test."""
        return self.config.isolate_groups

    @property
    def test_period(self) -> int | str | None:
        """Held-out test period specification."""
        return self.config.test_period

    @property
    def test_start_date(self) -> date | None:
        """Explicit start date for held-out test period."""
        # Validator converts str to date, so this is always date | None at runtime
        return cast(date | None, self.config.test_start)

    @property
    def test_end_date(self) -> date | None:
        """Explicit end date for held-out test period."""
        # Validator converts str to date, so this is always date | None at runtime
        return cast(date | None, self.config.test_end)

    @property
    def fold_direction(self) -> Literal["forward", "backward"]:
        """Direction of validation folds."""
        return self.config.fold_direction

    @property
    def calendar_id(self) -> str | None:
        """Trading calendar identifier."""
        return self.config.calendar_id

    @property
    def test_indices_(self) -> NDArray[np.intp]:
        """Held-out test indices (populated after split() is called).

        Returns
        -------
        ndarray
            Indices reserved for the held-out test period.

        Raises
        ------
        ValueError
            If no held-out test is configured or split() hasn't been called.

        Examples
        --------
        >>> cv = WalkForwardCV(n_splits=5, test_period="52D")
        >>> for train_idx, val_idx in cv.split(X):
        ...     pass  # Training loop
        >>> # Now test_indices_ is available
        >>> final_score = model.score(X.iloc[cv.test_indices_], y.iloc[cv.test_indices_])
        """
        if self._test_indices is None:
            if not self._has_held_out_test():
                raise ValueError(
                    "No held-out test period configured. "
                    "Set test_period or test_start to enable held-out test."
                )
            raise ValueError("test_indices_ not available. Call split() first to compute indices.")
        return self._test_indices

    def _has_held_out_test(self) -> bool:
        """Check if a held-out test period is configured."""
        return self.config.test_period is not None or self.config.test_start is not None

    def _compute_test_period(
        self,
        n_samples: int,
        timestamps: pd.DatetimeIndex | None,
    ) -> tuple[int, int]:
        """Compute held-out test period boundaries.

        Returns
        -------
        tuple[int, int]
            (test_start_idx, test_end_idx) as sample indices
        """
        if self.config.test_start is not None:
            # Explicit start date provided
            if timestamps is None:
                raise ValueError(
                    "test_start requires timestamps. Use a DatetimeIndex or set timestamp_col."
                )

            test_start_date = self.config.test_start
            test_start_ts = pd.Timestamp(test_start_date, tz=timestamps.tz)
            test_start_idx = int(timestamps.searchsorted(test_start_ts, side="left"))

            if self.config.test_end is not None:
                test_end_date = self.config.test_end
                test_end_ts = pd.Timestamp(test_end_date, tz=timestamps.tz)
                # Find the index of the first timestamp >= test_end
                test_end_idx = int(timestamps.searchsorted(test_end_ts, side="right"))
            else:
                # Default: end of data
                test_end_idx = n_samples

        elif self.config.test_period is not None:
            # test_period specifies duration to reserve from end
            if isinstance(self.config.test_period, int):
                # Integer: trading days (requires calendar) or samples
                if self.calendar is not None and timestamps is not None:
                    # Trading-day interpretation
                    end_timestamp = timestamps[-1]
                    test_start_ts = self.calendar.previous_trading_day(
                        end_timestamp, n=self.config.test_period
                    )
                    test_start_idx = int(timestamps.searchsorted(test_start_ts, side="left"))
                else:
                    # Sample count interpretation
                    test_start_idx = max(0, n_samples - self.config.test_period)
            else:
                # String: time-based (e.g., "52D")
                if timestamps is None:
                    raise ValueError(
                        "Time-based test_period requires timestamps. "
                        "Use a DatetimeIndex or set timestamp_col."
                    )
                time_delta = pd.Timedelta(self.config.test_period)
                end_timestamp = timestamps[-1]
                test_start_ts = end_timestamp - time_delta
                test_start_idx = int(timestamps.searchsorted(test_start_ts, side="left"))

            # test_end is always end of data for test_period mode
            if self.config.test_end is not None and timestamps is not None:
                test_end_date = self.config.test_end
                test_end_ts = pd.Timestamp(test_end_date, tz=timestamps.tz)
                test_end_idx = int(timestamps.searchsorted(test_end_ts, side="right"))
            else:
                test_end_idx = n_samples
        else:
            # No held-out test configured
            return n_samples, n_samples

        return test_start_idx, test_end_idx

    def _parse_time_size(
        self,
        size_spec: int | float | str,
        timestamps: pd.DatetimeIndex | None,
        n_samples: int,
    ) -> int:
        """Parse size specification and convert to sample count.

        Uses calendar-aware logic if calendar is configured, otherwise falls back
        to naive time-based calculation.

        Parameters
        ----------
        size_spec : int, float, or str
            Size specification to parse.
        timestamps : pd.DatetimeIndex
            Datetime index of the data.
        n_samples : int
            Total number of samples in dataset.

        Returns
        -------
        int
            Number of samples corresponding to the size specification.
        """
        if isinstance(size_spec, str):
            # Time-based specification (e.g., "4W", "30D", "3M")
            if timestamps is None:
                raise ValueError(
                    "Time-based size specifications require timestamps. "
                    "For pandas DataFrames: use a DatetimeIndex. "
                    "For Polars DataFrames: set timestamp_col='your_datetime_column'. "
                    "Example: WalkForwardCV(test_size='4W', timestamp_col='date')"
                )

            # Use calendar-aware parsing if calendar is configured
            return parse_time_size_calendar_aware(
                size_spec=size_spec,
                timestamps=timestamps,
                calendar=self.calendar,
            )

        elif isinstance(size_spec, float):
            # Proportion of dataset
            return int(n_samples * size_spec)
        else:
            # Integer sample count
            return size_spec

    def get_n_splits(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None = None,
        y: pl.Series | pd.Series | NDArray[Any] | None = None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    ) -> int:
        """Get number of splits.

        Parameters
        ----------
        X : array-like, optional
            Always ignored, exists for compatibility.

        y : array-like, optional
            Always ignored, exists for compatibility.

        groups : array-like, optional
            Always ignored, exists for compatibility.

        Returns:
        -------
        n_splits : int
            Number of splits.
        """
        del X, y, groups  # Unused, for sklearn compatibility
        return self.n_splits

    def split(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        y: pl.Series | pd.Series | NDArray[Any] | None = None,
        groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate train/validation indices for walk-forward splits.

        When a held-out test period is configured (test_period or test_start),
        this method yields train/validation splits for cross-validation, and
        the held-out test indices are accessible via test_indices_ property.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), optional
            Target variable.

        groups : array-like of shape (n_samples,), optional
            Group labels for samples.

        Yields:
        ------
        train : ndarray
            Training set indices for this split.

        val : ndarray
            Validation set indices for this split (or test if no held-out test).

        Notes
        -----
        When using held-out test mode with fold_direction="backward":

        ```
        [train1][val1][train2][val2][train3][val3] | [HELD-OUT TEST]
                ←     ←     ←     ←     ←     ←     test_start
        ```

        Validation folds step backward from the test boundary, ensuring that
        all validation is done on data chronologically before the held-out test.
        """
        # Validate inputs and get sample count
        n_samples = self._validate_data(X, y, groups)

        # Validate session alignment if enabled
        self._validate_session_alignment(X, self.align_to_sessions, self.session_col)

        # Extract timestamps for held-out test computation
        timestamps = self._extract_timestamps(X, self.timestamp_col)

        # Compute held-out test period if configured
        if self._has_held_out_test():
            test_start_idx, test_end_idx = self._compute_test_period(n_samples, timestamps)
            self._test_start_idx = test_start_idx
            self._test_end_idx = test_end_idx
            self._test_indices = np.arange(test_start_idx, test_end_idx, dtype=np.intp)

            # Use backward or forward splitting for validation folds
            if self.fold_direction == "backward":
                yield from self._split_backward(X, y, groups, n_samples, test_start_idx, timestamps)
            else:
                # Forward direction with held-out test: traditional walk-forward up to test boundary
                yield from self._split_forward_with_test(
                    X, y, groups, n_samples, test_start_idx, timestamps
                )
        else:
            # Legacy behavior: no held-out test, standard walk-forward
            self._test_indices = None
            self._test_start_idx = None
            self._test_end_idx = None

            # Dispatch: calendar-first > session-based > sample-based
            # Calendar splitting requires timestamps; fall back for numpy arrays
            has_timestamps = self._extract_timestamps(X, self.timestamp_col) is not None
            if self._should_use_calendar_splitting() and has_timestamps:
                if self.align_to_sessions:
                    warnings.warn(
                        "align_to_sessions=True is deprecated when calendar is active. "
                        "Calendar-first splitting subsumes session alignment.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                yield from self._split_by_calendar(X, y, groups, n_samples)
            elif self.align_to_sessions:
                yield from self._split_by_sessions(
                    cast(pl.DataFrame | pd.DataFrame, X), y, groups, n_samples
                )
            else:
                yield from self._split_by_samples(X, y, groups, n_samples)

    def _split_by_samples(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        _y: pl.Series | pd.Series | NDArray[Any] | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None,
        n_samples: int,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate splits using sample indices (original implementation)."""
        # Extract timestamps if available (supports both Polars and pandas)
        timestamps = self._extract_timestamps(X, self.timestamp_col)

        # Calculate test size
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self._parse_time_size(self.test_size, timestamps, n_samples)

        # Calculate train size if specified
        if self.train_size is not None:
            train_size = self._parse_time_size(self.train_size, timestamps, n_samples)
        else:
            train_size = None

        # Calculate split points
        if self.consecutive:
            # Consecutive walk-forward: back-to-back test periods with no gaps
            # Useful for realistic trading simulation where test periods are sequential
            step_size = test_size

            # Determine where first test period starts
            if train_size is not None and not self.expanding:
                # Rolling window: first test comes after initial training window
                first_test_start = train_size
            elif self.expanding:
                # Expanding window: ensure we have enough data for minimum train_size
                # or default to test_size if train_size not specified
                first_test_start = train_size if train_size is not None else test_size
            else:
                # No train_size specified and not expanding: start after first test-sized chunk
                first_test_start = test_size

            # Validate we have enough data for all consecutive periods
            total_required = first_test_start + self.n_splits * test_size
            if total_required > n_samples:
                raise ValueError(
                    f"Insufficient data for consecutive={self.consecutive}: "
                    f"need {total_required:,} samples (first_test at {first_test_start:,} "
                    f"+ {self.n_splits} × {test_size:,}), but only have {n_samples:,}"
                )
        else:
            # Spread folds across available data to sample different time periods
            # Useful for testing robustness across different market regimes
            available_for_splits = n_samples - test_size
            step_size = available_for_splits // self.n_splits
            first_test_start = test_size

        for i in range(self.n_splits):
            # Calculate test indices
            test_start = first_test_start + i * step_size
            test_end = min(test_start + test_size, n_samples)

            # For the last split, optionally use all remaining data
            # (only if test_size was not explicitly specified)
            if i == self.n_splits - 1 and self.test_size is None:
                test_end = n_samples

            # Calculate train indices
            if self.expanding:
                # Expanding window: use all data from start
                train_start = 0
            else:
                # Rolling window
                if train_size is not None:
                    train_start = max(0, test_start - self.gap - train_size)
                else:
                    # If no train_size specified, use all available data
                    train_start = 0

            # Apply gap
            train_end = test_start - self.gap

            # Initial train indices (before purging/embargo)
            train_indices = np.arange(train_start, train_end)

            # Convert test boundaries to timestamps if needed
            test_start_time, test_end_time = convert_indices_to_timestamps(
                test_start,
                test_end,
                timestamps,
            )

            # Apply purging and embargo
            clean_train_indices = apply_purging_and_embargo(
                train_indices=train_indices,
                test_start=test_start_time,
                test_end=test_end_time,
                label_horizon=self.label_horizon,
                embargo_size=self.embargo_size,
                embargo_pct=self.embargo_pct,
                n_samples=n_samples,
                timestamps=timestamps,
                calendar=self.calendar,
            )

            # Test indices
            test_indices = np.arange(test_start, test_end, dtype=np.intp)

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, test_indices, groups
                )

            yield clean_train_indices.astype(np.intp), test_indices

    def _split_by_sessions(
        self,
        X: pl.DataFrame | pd.DataFrame,
        _y: pl.Series | pd.Series | NDArray[Any] | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None,
        n_samples: int,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate splits using session boundaries (session-aware)."""
        # Get unique sessions in chronological order
        unique_sessions = self._get_unique_sessions(X, self.session_col)
        n_sessions = len(unique_sessions)

        # Extract timestamps if available (for purging/embargo)
        timestamps = self._extract_timestamps(X, self.timestamp_col)

        # Calculate test size in sessions
        if self.test_size is None:
            test_size_sessions = n_sessions // (self.n_splits + 1)
        elif isinstance(self.test_size, int):
            # Integer test_size: interpret as number of sessions
            test_size_sessions = self.test_size
        elif isinstance(self.test_size, float):
            # Float test_size: proportion of sessions
            test_size_sessions = int(n_sessions * self.test_size)
        else:
            # Time-based test_size not supported with sessions
            raise ValueError(
                f"align_to_sessions=True does not support time-based test_size. "
                f"Use integer (number of sessions) or float (proportion). Got: {self.test_size}"
            )

        # Calculate train size in sessions if specified
        if self.train_size is not None:
            if isinstance(self.train_size, int):
                train_size_sessions = self.train_size
            elif isinstance(self.train_size, float):
                train_size_sessions = int(n_sessions * self.train_size)
            else:
                raise ValueError(
                    f"align_to_sessions=True does not support time-based train_size. "
                    f"Use integer (number of sessions) or float (proportion). Got: {self.train_size}"
                )
        else:
            train_size_sessions = None

        # Calculate split points in session space
        if self.consecutive:
            step_size_sessions = test_size_sessions

            if train_size_sessions is not None and not self.expanding:
                first_test_start_session = train_size_sessions
            elif self.expanding:
                first_test_start_session = (
                    train_size_sessions if train_size_sessions is not None else test_size_sessions
                )
            else:
                first_test_start_session = test_size_sessions

            total_required_sessions = first_test_start_session + self.n_splits * test_size_sessions
            if total_required_sessions > n_sessions:
                raise ValueError(
                    f"Insufficient sessions for consecutive={self.consecutive}: "
                    f"need {total_required_sessions:,} sessions (first_test at {first_test_start_session:,} "
                    f"+ {self.n_splits} × {test_size_sessions:,}), but only have {n_sessions:,}"
                )
        else:
            available_for_splits_sessions = n_sessions - test_size_sessions
            step_size_sessions = available_for_splits_sessions // self.n_splits
            first_test_start_session = test_size_sessions

        # Generate splits by mapping session ranges to row indices
        for i in range(self.n_splits):
            # Calculate test session range
            test_start_session = first_test_start_session + i * step_size_sessions
            test_end_session = min(test_start_session + test_size_sessions, n_sessions)

            if i == self.n_splits - 1 and self.test_size is None:
                test_end_session = n_sessions

            # Calculate train session range
            if self.expanding:
                train_start_session = 0
            else:
                if train_size_sessions is not None:
                    train_start_session = max(
                        0, test_start_session - self.gap - train_size_sessions
                    )
                else:
                    train_start_session = 0

            train_end_session = test_start_session - self.gap

            # Get session IDs for train and test
            if isinstance(unique_sessions, pl.Series):
                train_sessions = unique_sessions[train_start_session:train_end_session].to_list()
                test_sessions = unique_sessions[test_start_session:test_end_session].to_list()
                session_col_values = X[self.session_col]
            else:  # pandas Series
                train_sessions = unique_sessions.iloc[
                    train_start_session:train_end_session
                ].tolist()
                test_sessions = unique_sessions.iloc[test_start_session:test_end_session].tolist()
                session_col_values = X[self.session_col]

            # Map sessions to row indices
            if isinstance(X, pl.DataFrame):
                train_mask = session_col_values.is_in(train_sessions)
                test_mask = session_col_values.is_in(test_sessions)
                train_indices = np.where(train_mask.to_numpy())[0]
                test_indices = np.where(test_mask.to_numpy())[0]
            else:  # pandas DataFrame
                # Cast to pd.Series since X is pd.DataFrame here
                session_col_pd = cast(pd.Series, session_col_values)
                train_mask = session_col_pd.isin(train_sessions)
                test_mask = session_col_pd.isin(test_sessions)
                train_indices = np.where(train_mask.to_numpy())[0]
                test_indices = np.where(test_mask.to_numpy())[0]

            # Apply purging and embargo if configured
            if self._has_purging_or_embargo():
                # Compute actual timestamp bounds from test indices
                # This is critical for multi-asset data where rows may be sorted by
                # asset rather than time - using positional indices [0] and [-1] would
                # give incorrect timestamp bounds
                test_start_time, test_end_time = self._timestamp_window_from_indices(
                    test_indices, timestamps
                )

                clean_train_indices = apply_purging_and_embargo(
                    train_indices=train_indices,
                    test_start=test_start_time,
                    test_end=test_end_time,
                    label_horizon=self.label_horizon,
                    embargo_size=self.embargo_size,
                    embargo_pct=self.embargo_pct,
                    n_samples=n_samples,
                    timestamps=timestamps,
                    calendar=self.calendar,
                )
            else:
                clean_train_indices = train_indices

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, test_indices, groups
                )

            yield clean_train_indices.astype(np.intp), test_indices.astype(np.intp)

    def _should_use_calendar_splitting(self) -> bool:
        """Determine if calendar-first splitting should be used.

        Calendar splitting is used when:
        1. A calendar is configured (always true now with NYSE default), AND
        2. Time-based sizes are used (str test_size/train_size), OR
        3. The config has filter_non_trading=True (default)
        """
        if self.calendar is None:
            return False

        # Time-based sizes always need calendar
        if isinstance(self.test_size, str) or isinstance(self.train_size, str):
            return True

        # Calendar filtering of non-trading rows
        if self.config.filter_non_trading:
            return True

        return False

    def _size_to_sessions(
        self,
        size_spec: int | float | str,
        n_sessions: int,
    ) -> int:
        """Convert a size specification to a session count.

        Parameters
        ----------
        size_spec : int, float, or str
            - int: pass through (already a session count)
            - float: proportion of total sessions
            - str: time period, converted via calendar.time_spec_to_sessions()
        n_sessions : int
            Total number of unique sessions (for proportion calculation).

        Returns
        -------
        int
            Number of sessions.
        """
        if isinstance(size_spec, str):
            if self.calendar is None:
                raise ValueError("Calendar required for time-based size specifications.")
            return self.calendar.time_spec_to_sessions(size_spec)
        elif isinstance(size_spec, float):
            return max(1, int(n_sessions * size_spec))
        else:
            return int(size_spec)

    def _split_by_calendar(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        _y: pl.Series | pd.Series | NDArray[Any] | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None,
        n_samples: int,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate splits using calendar-aware session boundaries.

        This is the calendar-first code path. All fold arithmetic happens in
        session space, and non-trading rows are filtered out.
        """
        # Extract timestamps
        timestamps = self._extract_timestamps(X, self.timestamp_col)
        if timestamps is None:
            raise ValueError(
                "Calendar-first splitting requires timestamps. "
                "For pandas DataFrames: use a DatetimeIndex. "
                "For Polars DataFrames: set timestamp_col='your_datetime_column'."
            )

        assert self.calendar is not None  # guaranteed by _should_use_calendar_splitting

        # Get sessions and trading mask
        sessions, trading_mask = self.calendar.get_sessions_and_mask(timestamps)

        # Log non-trading row exclusion
        n_excluded = int((~trading_mask).sum())
        if n_excluded > 0 and self.config.filter_non_trading:
            logger.info(
                "Calendar %s: %d non-trading rows excluded from %d total",
                self.calendar.config.exchange,
                n_excluded,
                n_samples,
            )

        # Build unique sessions and session-to-indices mapping
        # Only consider trading rows
        if self.config.filter_non_trading:
            trading_indices = np.where(trading_mask)[0]
            trading_sessions = sessions.iloc[trading_indices].dropna()
        else:
            trading_indices = np.arange(n_samples)
            trading_sessions = sessions

        # Get unique sessions in order
        unique_sessions_raw = trading_sessions.unique()
        unique_sessions_raw = unique_sessions_raw[~pd.isna(unique_sessions_raw)]
        unique_sessions = np.sort(unique_sessions_raw)
        n_sessions = len(unique_sessions)

        if n_sessions == 0:
            raise ValueError(
                "No trading sessions found in the data for calendar "
                f"'{self.calendar.config.exchange}'."
            )

        # Build session -> row indices mapping (only trading rows)
        session_to_indices: dict[Any, NDArray[np.intp]] = {}
        for session_date in unique_sessions:
            mask = (sessions.values == session_date) & trading_mask
            session_to_indices[session_date] = np.where(mask)[0].astype(np.intp)

        # Convert size specs to session counts
        if self.test_size is None:
            test_size_sessions = n_sessions // (self.n_splits + 1)
        else:
            test_size_sessions = self._size_to_sessions(self.test_size, n_sessions)

        if self.train_size is not None:
            train_size_sessions = self._size_to_sessions(self.train_size, n_sessions)
        else:
            train_size_sessions = None

        # Calculate split points in session space
        if self.consecutive:
            step_size_sessions = test_size_sessions
            if train_size_sessions is not None and not self.expanding:
                first_test_start_session = train_size_sessions
            elif self.expanding:
                first_test_start_session = (
                    train_size_sessions if train_size_sessions is not None else test_size_sessions
                )
            else:
                first_test_start_session = test_size_sessions

            total_required = first_test_start_session + self.n_splits * test_size_sessions
            if total_required > n_sessions:
                raise ValueError(
                    f"Insufficient trading sessions for consecutive splits: "
                    f"need {total_required} sessions "
                    f"({first_test_start_session} + {self.n_splits} x {test_size_sessions}), "
                    f"but only have {n_sessions} sessions in calendar "
                    f"'{self.calendar.config.exchange}'."
                )
        else:
            available = n_sessions - test_size_sessions
            step_size_sessions = available // self.n_splits
            first_test_start_session = test_size_sessions

        # Generate splits
        for i in range(self.n_splits):
            test_start_s = first_test_start_session + i * step_size_sessions
            test_end_s = min(test_start_s + test_size_sessions, n_sessions)

            if i == self.n_splits - 1 and self.test_size is None:
                test_end_s = n_sessions

            # Train session range
            if self.expanding:
                train_start_s = 0
            else:
                if train_size_sessions is not None:
                    train_start_s = max(0, test_start_s - self.gap - train_size_sessions)
                else:
                    train_start_s = 0

            train_end_s = max(0, test_start_s - self.gap)

            # Map session ranges to row indices
            train_sessions = unique_sessions[train_start_s:train_end_s]
            test_sessions = unique_sessions[test_start_s:test_end_s]

            train_indices = (
                np.concatenate([session_to_indices[s] for s in train_sessions])
                if len(train_sessions) > 0
                else np.array([], dtype=np.intp)
            )
            test_indices = (
                np.concatenate([session_to_indices[s] for s in test_sessions])
                if len(test_sessions) > 0
                else np.array([], dtype=np.intp)
            )

            # Sort indices (sessions may have interleaved assets)
            train_indices = np.sort(train_indices)
            test_indices = np.sort(test_indices)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            # Apply purging and embargo using actual timestamp boundaries
            if self._has_purging_or_embargo():
                test_start_time, test_end_time = self._timestamp_window_from_indices(
                    test_indices, timestamps
                )
                clean_train_indices = apply_purging_and_embargo(
                    train_indices=train_indices,
                    test_start=test_start_time,
                    test_end=test_end_time,
                    label_horizon=self.label_horizon,
                    embargo_size=self.embargo_size,
                    embargo_pct=self.embargo_pct,
                    n_samples=n_samples,
                    timestamps=timestamps,
                    calendar=self.calendar,
                )
            else:
                clean_train_indices = train_indices

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, test_indices, groups
                )

            yield clean_train_indices.astype(np.intp), test_indices.astype(np.intp)

    def _has_purging_or_embargo(self) -> bool:
        """Check if purging or embargo is needed.

        Handles both int and pd.Timedelta values for label_horizon and embargo_size.

        Returns
        -------
        bool
            True if purging or embargo should be applied.
        """
        # Check label_horizon (can be int or Timedelta)
        has_label_horizon = False
        if isinstance(self.label_horizon, int | float):
            has_label_horizon = self.label_horizon > 0
        elif hasattr(self.label_horizon, "total_seconds"):  # pd.Timedelta
            has_label_horizon = self.label_horizon.total_seconds() > 0

        # Check embargo (embargo_size can be int or Timedelta, embargo_pct is always float or None)
        has_embargo = self.embargo_size is not None or self.embargo_pct is not None

        return has_label_horizon or has_embargo

    @staticmethod
    def _timestamp_window_from_indices(
        indices: NDArray[np.intp],
        timestamps: pd.DatetimeIndex | None,
    ) -> tuple[int | pd.Timestamp, int | pd.Timestamp]:
        """Compute timestamp window from actual indices (for session-aligned purging).

        This is critical for correct purging in session-aligned mode. Instead of
        using positional indices [0] and [-1] which assume chronological ordering,
        we compute the actual timestamp bounds from all test indices.

        For multi-asset data where rows may be sorted by asset rather than time,
        test_indices[0] may not have the minimum timestamp.

        Parameters
        ----------
        indices : ndarray
            Row indices of test samples.
        timestamps : pd.DatetimeIndex or None
            Timestamps for all samples. If None, returns index bounds.

        Returns
        -------
        start_time : int or pd.Timestamp
            Minimum timestamp of test indices (or min index if no timestamps).
        end_time_exclusive : int or pd.Timestamp
            Maximum timestamp + 1 nanosecond (or max index + 1 if no timestamps).
        """
        if len(indices) == 0:
            # Empty indices - return minimal bounds
            if timestamps is None:
                return 0, 0
            return timestamps[0], timestamps[0]

        if timestamps is None:
            # No timestamps - return index bounds
            return int(indices.min()), int(indices.max()) + 1

        test_timestamps = timestamps.take(indices)
        start_time = test_timestamps.min()
        # Add 1 nanosecond to make end exclusive (handles duplicate timestamps)
        end_time_exclusive = test_timestamps.max() + pd.Timedelta(1, "ns")
        return start_time, end_time_exclusive

    def _split_backward(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        _y: pl.Series | pd.Series | NDArray[Any] | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None,
        n_samples: int,
        test_start_idx: int,
        timestamps: pd.DatetimeIndex | None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate splits stepping backward from held-out test boundary.

        Validation folds step backward in time from the held-out test start,
        ensuring all validation data is chronologically before the final test.

        ```
        [train1][val1][train2][val2][train3][val3] | [HELD-OUT TEST]
                ←     ←     ←     ←     ←     ←     test_start_idx
        ```

        Parameters
        ----------
        X : array-like
            Input data.
        _y : array-like, optional
            Target (unused).
        groups : array-like, optional
            Group labels.
        n_samples : int
            Total samples in X.
        test_start_idx : int
            Index where held-out test period starts.
        timestamps : pd.DatetimeIndex, optional
            Timestamps for each sample.
        """
        # Available data for validation (before held-out test)
        available_samples = test_start_idx

        # Calculate validation fold size
        if self.test_size is None:
            val_size = available_samples // (self.n_splits + 1)
        else:
            val_size = self._parse_time_size(self.test_size, timestamps, available_samples)

        # Calculate train size if specified
        if self.train_size is not None:
            train_size = self._parse_time_size(self.train_size, timestamps, available_samples)
        else:
            train_size = None

        # Calculate step size
        if self.config.step_size is not None:
            step_size = self.config.step_size
        else:
            step_size = val_size  # Non-overlapping validation folds

        # Generate splits stepping backward from test_start_idx
        # Fold 0 has validation ending at test_start_idx
        # Fold 1 has validation ending at test_start_idx - step_size
        # etc.
        for i in range(self.n_splits):
            # Validation fold boundaries (stepping backward)
            val_end = test_start_idx - i * step_size
            val_start = max(0, val_end - val_size)

            if val_start >= val_end:
                # No more room for validation folds
                break

            # Training boundaries
            if self.expanding:
                # Expanding window: use all data from start up to val_start
                train_start = 0
            else:
                # Rolling window
                if train_size is not None:
                    train_start = max(0, val_start - self.gap - train_size)
                else:
                    # Use all available data before validation
                    train_start = 0

            train_end = max(0, val_start - self.gap)

            if train_end <= train_start:
                # No room for training data
                continue

            # Create initial indices
            train_indices = np.arange(train_start, train_end, dtype=np.intp)
            val_indices = np.arange(val_start, val_end, dtype=np.intp)

            # Convert boundaries to timestamps for purging
            val_start_time, val_end_time = convert_indices_to_timestamps(
                val_start, val_end, timestamps
            )

            # Apply purging and embargo
            clean_train_indices = apply_purging_and_embargo(
                train_indices=train_indices,
                test_start=val_start_time,
                test_end=val_end_time,
                label_horizon=self.label_horizon,
                embargo_size=self.embargo_size,
                embargo_pct=self.embargo_pct,
                n_samples=n_samples,
                timestamps=timestamps,
                calendar=self.calendar,
            )

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, val_indices, groups
                )

            yield clean_train_indices, val_indices

    def _split_forward_with_test(
        self,
        X: pl.DataFrame | pd.DataFrame | NDArray[Any],
        _y: pl.Series | pd.Series | NDArray[Any] | None,
        groups: pl.Series | pd.Series | NDArray[Any] | None,
        n_samples: int,
        test_start_idx: int,
        timestamps: pd.DatetimeIndex | None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate splits stepping forward, stopping before held-out test.

        Traditional walk-forward validation but constrained to stay before
        the held-out test boundary.

        Parameters
        ----------
        X : array-like
            Input data.
        _y : array-like, optional
            Target (unused).
        groups : array-like, optional
            Group labels.
        n_samples : int
            Total samples in X.
        test_start_idx : int
            Index where held-out test period starts.
        timestamps : pd.DatetimeIndex, optional
            Timestamps for each sample.
        """
        # Available data for validation (before held-out test)
        available_samples = test_start_idx

        # Calculate validation fold size
        if self.test_size is None:
            val_size = available_samples // (self.n_splits + 1)
        else:
            val_size = self._parse_time_size(self.test_size, timestamps, available_samples)

        # Calculate train size if specified
        if self.train_size is not None:
            train_size = self._parse_time_size(self.train_size, timestamps, available_samples)
        else:
            train_size = None

        # Calculate step size
        if self.config.step_size is not None:
            step_size = self.config.step_size
        elif self.consecutive:
            step_size = val_size  # Consecutive: back-to-back folds
        else:
            # Spread folds across available data
            available_for_splits = available_samples - val_size
            step_size = available_for_splits // self.n_splits

        # Determine where first validation fold starts
        if self.consecutive:
            if train_size is not None and not self.expanding:
                first_val_start = train_size
            elif self.expanding:
                first_val_start = train_size if train_size is not None else val_size
            else:
                first_val_start = val_size
        else:
            first_val_start = val_size

        # Generate forward-stepping splits
        for i in range(self.n_splits):
            val_start = first_val_start + i * step_size
            val_end = min(val_start + val_size, test_start_idx)

            if val_start >= test_start_idx:
                # Would overlap with held-out test
                break

            # Training boundaries
            if self.expanding:
                train_start = 0
            else:
                if train_size is not None:
                    train_start = max(0, val_start - self.gap - train_size)
                else:
                    train_start = 0

            train_end = max(0, val_start - self.gap)

            if train_end <= train_start:
                continue

            # Create initial indices
            train_indices = np.arange(train_start, train_end, dtype=np.intp)
            val_indices = np.arange(val_start, val_end, dtype=np.intp)

            # Convert boundaries to timestamps for purging
            val_start_time, val_end_time = convert_indices_to_timestamps(
                val_start, val_end, timestamps
            )

            # Apply purging and embargo
            clean_train_indices = apply_purging_and_embargo(
                train_indices=train_indices,
                test_start=val_start_time,
                test_end=val_end_time,
                label_horizon=self.label_horizon,
                embargo_size=self.embargo_size,
                embargo_pct=self.embargo_pct,
                n_samples=n_samples,
                timestamps=timestamps,
                calendar=self.calendar,
            )

            # Apply group isolation if requested
            if self.isolate_groups and groups is not None:
                clean_train_indices = isolate_groups_from_train(
                    clean_train_indices, val_indices, groups
                )

            yield clean_train_indices, val_indices
