"""Configuration classes for cross-validation splitters.

This module provides Pydantic-based configuration for all CV splitters,
enabling reproducible, serializable, and validated split strategies.

Integration with qdata
----------------------
Session-aware splitting consumes `session_date` column from qdata library:

    from ml4t.data.sessions import SessionAssigner
    assigner = SessionAssigner.from_exchange('CME')
    df_with_sessions = assigner.assign_sessions(df)

    config = WalkForwardConfig(
        n_splits=5,
        test_size=4,  # 4 sessions
        align_to_sessions=True
    )
    cv = WalkForwardCV.from_config(config)
    for train, test in cv.split(df_with_sessions):
        # Fold boundaries aligned to session boundaries
        pass

Examples
--------
>>> # Parameter-based initialization (backward compatible)
>>> cv = WalkForwardCV(n_splits=5, test_size=100)
>>>
>>> # Config-based initialization
>>> config = WalkForwardConfig(n_splits=5, test_size=100)
>>> cv = WalkForwardCV.from_config(config)
>>>
>>> # Serialize config for reproducibility
>>> config.to_json("cv_config.json")
>>> loaded_config = WalkForwardConfig.from_json("cv_config.json")
>>> cv = WalkForwardCV.from_config(loaded_config)
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig


class SplitterConfig(BaseConfig):
    """Base configuration for all cross-validation splitters.

    All splitter configs inherit from this class to ensure consistent
    serialization, validation, and reproducibility.

    Attributes
    ----------
    n_splits : int
        Number of cross-validation folds.

    label_horizon : int or pd.Timedelta
        Gap between train_end and val_start sized to the label horizon.
        Removes training samples whose prediction targets overlap with
        validation/test data ("label buffer").

        Example: If predicting 5-day forward returns, a training sample at day 95
        has a label computed from days 95-100. If the test set starts at day 98,
        this training sample's label "sees" test data, creating leakage.
        Setting label_horizon=5 removes training samples from days 93-97.

        Aliases: ``label_buffer`` is accepted as an equivalent input name.

    embargo_td : int or pd.Timedelta or None
        Buffer zone after test periods where training samples are also excluded
        ("feature buffer"). Prevents autocorrelation leakage in combinatorial CV
        where training data can follow test data chronologically.

        For standard walk-forward CV (training always before test), this has no effect.

        Aliases: ``feature_buffer`` is accepted as an equivalent input name.

    align_to_sessions : bool
        If True, fold boundaries are aligned to trading session boundaries.
        Requires 'session_date' column in data (from ml4t.data.sessions.SessionAssigner).

    session_col : str
        Column name containing session identifiers.
        Default: 'session_date' (standard qdata column name).

    isolate_groups : bool
        If True, ensures no overlap between train/test group identifiers.
        Useful for multi-asset validation to prevent data leakage.
    """

    n_splits: int = Field(5, gt=0, description="Number of cross-validation folds")
    label_horizon: Any = Field(
        0,
        description=(
            "Gap between train_end and val_start sized to the label horizon "
            "(int bars or pd.Timedelta). Alias: label_buffer."
        ),
    )
    embargo_td: Any = Field(
        None,
        description=(
            "Buffer zone after test periods (int bars, pd.Timedelta, or None). "
            "Alias: feature_buffer."
        ),
    )
    align_to_sessions: bool = Field(
        False,
        description=(
            "Align fold boundaries to session boundaries. "
            "Requires 'session_date' column from ml4t.data.sessions.SessionAssigner."
        ),
    )
    session_col: str = Field(
        "session_date",
        description="Column name containing session identifiers (default: qdata standard)",
    )
    timestamp_col: str | None = Field(
        None,
        description=(
            "Column name containing timestamps for time-based sizes. "
            "Required for Polars DataFrames with time-based test_size/train_size. "
            "If None, falls back to pandas DatetimeIndex (backward compatible)."
        ),
    )
    isolate_groups: bool = Field(
        False,
        description=(
            "Prevent same group (symbol/contract) from appearing in both train and test sets"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        """Accept label_buffer -> label_horizon and feature_buffer -> embargo_td."""
        if isinstance(data, dict):
            if "label_buffer" in data and "label_horizon" not in data:
                data["label_horizon"] = data.pop("label_buffer")
            elif "label_buffer" in data:
                data.pop("label_buffer")  # label_horizon takes precedence
            if "feature_buffer" in data and "embargo_td" not in data:
                data["embargo_td"] = data.pop("feature_buffer")
            elif "feature_buffer" in data:
                data.pop("feature_buffer")  # embargo_td takes precedence
        return data

    @property
    def label_buffer(self) -> Any:
        """Alias for label_horizon (preferred name)."""
        return self.label_horizon

    @property
    def feature_buffer(self) -> Any:
        """Alias for embargo_td (preferred name)."""
        return self.embargo_td

    @field_validator("label_horizon")
    @classmethod
    def validate_label_horizon(cls, v: Any) -> Any:
        """Validate label_horizon is either int >= 0 or a timedelta-like object."""
        if isinstance(v, int):
            if v < 0:
                raise ValueError("label_horizon must be greater than or equal to 0")
            return v
        # Allow timedelta-like objects (pd.Timedelta, datetime.timedelta)
        if hasattr(v, "total_seconds"):
            return v
        # Handle ISO 8601 duration strings from JSON serialization
        if isinstance(v, str):
            import pandas as pd

            try:
                return pd.Timedelta(v)
            except Exception as e:
                raise ValueError(  # noqa: B904
                    f"Could not parse label_horizon/label_buffer string '{v}' as Timedelta. "
                    f"Expected formats: '5D', '21D', '1W', '8h'. Error: {e}"
                )
        raise ValueError(f"label_horizon must be int >= 0 or timedelta-like object, got {type(v)}")

    @field_validator("embargo_td")
    @classmethod
    def validate_embargo_td(cls, v: Any) -> Any:
        """Validate embargo_td is either None, int >= 0, or a timedelta-like object."""
        if v is None:
            return v
        if isinstance(v, int):
            if v < 0:
                raise ValueError("embargo_td must be greater than or equal to 0")
            return v
        # Allow timedelta-like objects (pd.Timedelta, datetime.timedelta)
        if hasattr(v, "total_seconds"):
            return v
        # Handle ISO 8601 duration strings from JSON serialization
        if isinstance(v, str):
            import pandas as pd

            try:
                return pd.Timedelta(v)
            except Exception as e:
                raise ValueError(  # noqa: B904
                    f"Could not parse embargo_td/feature_buffer string '{v}' as Timedelta. "
                    f"Expected formats: '5D', '1W', '0D'. Error: {e}"
                )
        raise ValueError(
            f"embargo_td must be None, int >= 0, or timedelta-like object, got {type(v)}"
        )


class WalkForwardConfig(SplitterConfig):
    """Configuration for Walk-Forward Cross-Validation.

    Walk-forward validation is the standard approach for time-series backtesting,
    where the model is trained on historical data and tested on future periods.

    Attributes
    ----------
    test_size : int | float | str | None
        Size of validation folds. Alias: ``val_size``.
        - int: Number of samples (or sessions if align_to_sessions=True)
        - float: Proportion of dataset (0.0 to 1.0)
        - str: Time-based ('4W', '3M') - NOT supported with align_to_sessions=True
        - None: Auto-calculated to maintain equal test set sizes
    train_size : int | float | str | None
        Training set size specification (same format as test_size).
        If None, uses expanding window (all data before test set).
    step_size : int | None
        Step size between consecutive splits:
        - int: Number of samples (or sessions if align_to_sessions=True)
        - None: Defaults to test_size (non-overlapping test sets)
    test_period : int | str | None
        Held-out test period specification (reserves most recent data for final evaluation):
        - int: Number of trading days (requires calendar_id)
        - str: Time-based ('52D', '4W')
        - None: No held-out test period (default, legacy behavior)
    test_start : date | str | None
        Explicit start date for held-out test period. Mutually exclusive with test_period.
        Accepts date object or ISO format string ('2024-01-01').
        Alias: ``holdout_start``.
    test_end : date | str | None
        Explicit end date for held-out test period. Default: end of data.
        Accepts date object or ISO format string ('2024-12-31').
        Alias: ``holdout_end``.
    fold_direction : Literal["forward", "backward"]
        Direction of validation folds:
        - "forward": Traditional walk-forward (folds step forward in time)
        - "backward": Folds step backward from held-out test boundary
    calendar_id : str | None
        Trading calendar for trading-day-aware gap calculations.
        Examples: "NYSE", "CME_Equity", "LSE"
        Required when label_horizon is int and you want trading-day interpretation.
    """

    test_size: int | float | str | None = Field(
        None,
        description=(
            "Validation fold size: int (samples/sessions), float (proportion), "
            "str (time-based, e.g., '4W'). Alias: val_size."
        ),
    )
    train_size: int | float | str | None = Field(
        None,
        description=(
            "Train set size: int (samples/sessions), float (proportion), "
            "str (time-based, e.g., '12W'). "
            "None uses expanding window (all data before test)."
        ),
    )
    step_size: int | None = Field(
        None,
        ge=1,
        description=(
            "Step size between splits (int: samples/sessions). "
            "None defaults to test_size (non-overlapping)."
        ),
    )

    # Held-out test period specification
    test_period: int | str | None = Field(
        None,
        description=(
            "Held-out test period: int (trading days, requires calendar_id), "
            "str (time-based, e.g., '52D'). Reserves most recent data for final evaluation."
        ),
    )
    test_start: date | str | None = Field(
        None,
        description=(
            "Explicit start date for held-out test period. "
            "Mutually exclusive with test_period. Alias: holdout_start."
        ),
    )
    test_end: date | str | None = Field(
        None,
        description=(
            "Explicit end date for held-out test period. "
            "Default: end of data. Alias: holdout_end."
        ),
    )

    # Fold direction
    fold_direction: Literal["forward", "backward"] = Field(
        "forward",
        description=(
            "Direction of validation folds: 'forward' (traditional) or "
            "'backward' (step backward from held-out test boundary)."
        ),
    )

    # Calendar for trading-day-aware calculations
    calendar_id: str | None = Field(
        None,
        description=(
            "Trading calendar for trading-day-aware gap calculations. "
            "Examples: 'NYSE', 'CME_Equity', 'LSE'. "
            "Required when label_horizon is int and you want trading-day interpretation."
        ),
    )

    isolate_groups: bool = Field(
        False,
        description=(
            "Default False for walk-forward (opt-in). "
            "Set True for multi-asset validation to prevent group leakage."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_wf_aliases(cls, data: Any) -> Any:
        """Accept val_size -> test_size and holdout_start/end -> test_start/end."""
        if isinstance(data, dict):
            if "val_size" in data and "test_size" not in data:
                data["test_size"] = data.pop("val_size")
            elif "val_size" in data:
                data.pop("val_size")
            if "holdout_start" in data and "test_start" not in data:
                data["test_start"] = data.pop("holdout_start")
            elif "holdout_start" in data:
                data.pop("holdout_start")
            if "holdout_end" in data and "test_end" not in data:
                data["test_end"] = data.pop("holdout_end")
            elif "holdout_end" in data:
                data.pop("holdout_end")
            if "calendar" in data and "calendar_id" not in data:
                data["calendar_id"] = data.pop("calendar")
            elif "calendar" in data:
                data.pop("calendar")
        return data

    @property
    def val_size(self) -> int | float | str | None:
        """Alias for test_size (preferred name in book context)."""
        return self.test_size

    @property
    def holdout_start(self) -> date | None:
        """Alias for test_start."""
        return self.test_start

    @property
    def holdout_end(self) -> date | None:
        """Alias for test_end."""
        return self.test_end

    @field_validator("test_size", "train_size")
    @classmethod
    def validate_size_with_sessions(
        cls, v: int | float | str | None, info
    ) -> int | float | str | None:
        """Validate that time-based sizes are not used with session alignment."""
        if v is None:
            return v

        align_to_sessions = info.data.get("align_to_sessions", False)
        if align_to_sessions and isinstance(v, str):
            raise ValueError(
                f"align_to_sessions=True does not support time-based size specifications. "
                f"Use integer (number of sessions) or float (proportion). Got: {v!r}"
            )
        return v

    @field_validator("test_start", "test_end")
    @classmethod
    def validate_test_dates(cls, v: date | str | None) -> date | None:
        """Convert string dates to date objects."""
        if v is None:
            return v
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse date string '{v}'. Use ISO format: 'YYYY-MM-DD'"
                ) from e
        raise ValueError(f"test_start/test_end must be date or ISO string, got {type(v)}")

    @field_validator("test_period")
    @classmethod
    def validate_test_period(cls, v: int | str | None, info) -> int | str | None:
        """Validate test_period specification."""
        if v is None:
            return v

        if isinstance(v, int):
            if v <= 0:
                raise ValueError("test_period must be a positive integer (trading days)")
            return v

        if isinstance(v, str):
            # Validate time-based format (e.g., "52D", "4W")
            import pandas as pd

            try:
                pd.Timedelta(v)
            except Exception as e:
                raise ValueError(
                    f"Could not parse test_period string '{v}' as Timedelta. "
                    f"Use formats like '52D', '4W', '3M'. Error: {e}"
                ) from e
            return v

        raise ValueError(f"test_period must be int or str, got {type(v)}")

    @model_validator(mode="after")
    def validate_held_out_test_config(self) -> WalkForwardConfig:
        """Validate held-out test configuration consistency."""
        # test_period and test_start are mutually exclusive
        if self.test_period is not None and self.test_start is not None:
            raise ValueError(
                "Cannot specify both 'test_period' and 'test_start'. "
                "'test_period' reserves most recent data, "
                "'test_start' specifies an explicit date range."
            )

        # test_end without test_start or test_period is invalid
        if self.test_end is not None and self.test_start is None and self.test_period is None:
            raise ValueError(
                "'test_end' requires either 'test_period' or 'test_start' to define the held-out test."
            )

        # test_period as int requires calendar_id for trading-day interpretation
        if isinstance(self.test_period, int) and self.calendar_id is None:
            warnings.warn(
                f"test_period={self.test_period} (int) without calendar_id will be interpreted "
                "as calendar days, not trading days. Set calendar_id for trading-day interpretation.",
                UserWarning,
                stacklevel=2,
            )

        # label_horizon as int with calendar_id should use trading days
        if (
            isinstance(self.label_horizon, int)
            and self.label_horizon > 0
            and self.calendar_id is not None
        ):
            # Valid configuration - label_horizon will be converted to trading days
            pass

        return self


class CombinatorialConfig(SplitterConfig):
    """Configuration for Combinatorial Cross-Validation (CPCV).

    Combinatorial CV is designed for multi-asset strategies and combating overfitting
    by creating multiple test sets from combinatorial group selections.

    Reference: Bailey & Lopez de Prado (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"

    Attributes
    ----------
    n_groups : int
        Number of groups to partition the timeline into (typically 8-12).
    n_test_groups : int
        Number of groups used for each test set (typically 2-3).
        Total folds = C(n_groups, n_test_groups).
    max_combinations : int | None
        Maximum number of folds to generate.
        If C(n_groups, n_test_groups) > max_combinations, randomly sample.
    contiguous_test_blocks : bool
        If True, only use contiguous test groups (reduces overfitting).
        If False, allow any combination (more folds).
    """

    n_groups: int = Field(
        8, gt=1, description="Number of groups to partition timeline into (typically 8-12)"
    )
    n_test_groups: int = Field(2, gt=0, description="Number of groups per test set (typically 2-3)")
    max_combinations: int | None = Field(
        None,
        gt=0,
        description=(
            "Maximum folds to generate. If C(n_groups, n_test_groups) exceeds this, randomly sample."
        ),
    )
    contiguous_test_blocks: bool = Field(
        False,
        description=(
            "If True, only use contiguous test groups (less overfitting). "
            "If False, allow any combination."
        ),
    )
    embargo_pct: float | None = Field(
        None,
        ge=0.0,
        lt=1.0,
        description=(
            "Embargo size as percentage of total samples. "
            "Alternative to embargo_td. Cannot specify both."
        ),
    )
    isolate_groups: bool = Field(
        True,
        description=(
            "Default True for CPCV (opt-out). "
            "CPCV is designed for multi-asset validation, so group isolation is aggressive by default."
        ),
    )
    random_state: int | None = Field(
        None,
        description=(
            "Random seed for sampling when max_combinations limits the number of folds. "
            "Use for reproducible subset selection from C(n_groups, n_test_groups) combinations."
        ),
    )

    @field_validator("n_test_groups")
    @classmethod
    def validate_n_test_groups(cls, v: int, info) -> int:
        """Validate that n_test_groups < n_groups (must leave groups for training)."""
        n_groups = info.data.get("n_groups")
        if n_groups is not None and v >= n_groups:
            raise ValueError(
                f"n_test_groups ({v}) cannot exceed n_groups ({n_groups}). "
                f"Must leave at least one group for training. "
                f"Typically n_test_groups is 2-3 for CPCV."
            )
        return v

    @model_validator(mode="after")
    def validate_embargo_mutual_exclusivity(self) -> CombinatorialConfig:
        """Validate that embargo_td and embargo_pct are mutually exclusive."""
        if self.embargo_td is not None and self.embargo_pct is not None:
            raise ValueError(
                "Cannot specify both 'embargo_td' and 'embargo_pct'. "
                "Choose one method for setting the embargo period."
            )
        return self


# Export all config classes
__all__ = [
    "SplitterConfig",
    "WalkForwardConfig",
    "CombinatorialConfig",
]
