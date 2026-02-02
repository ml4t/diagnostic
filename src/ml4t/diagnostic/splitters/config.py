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

from typing import Any

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
        How far ahead labels look into the future. Used to remove training samples
        whose prediction targets overlap with validation/test data.

        Example: If predicting 5-day forward returns, a training sample at day 95
        has a label computed from days 95-100. If the test set starts at day 98,
        this training sample's label "sees" test data, creating leakage.
        Setting label_horizon=5 removes training samples from days 93-97.

    embargo_td : int or pd.Timedelta or None
        Buffer zone after test periods where training samples are also excluded.
        Prevents autocorrelation leakage in combinatorial CV where training data
        can follow test data chronologically.

        Example: When using CombinatorialCV with test groups in the middle of
        the timeline, training may include data from after the test period.
        If you're predicting volatility and the test period has a volatility spike,
        samples immediately after likely share similar volatility due to clustering.
        An embargo of 5 bars excludes those autocorrelated samples from training.

        For standard walk-forward CV (training always before test), this has no effect.

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
            "How far ahead labels look (int bars or pd.Timedelta). "
            "Removes training samples whose targets overlap with validation/test data."
        ),
    )
    embargo_td: Any = Field(
        None,
        description=(
            "Buffer zone after test periods (int bars, pd.Timedelta, or None). "
            "Excludes autocorrelated training samples that follow test data. "
            "Only affects CombinatorialCV where training can follow test chronologically."
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
                raise ValueError(f"Could not parse label_horizon string '{v}' as Timedelta: {e}")  # noqa: B904
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
                raise ValueError(f"Could not parse embargo_td string '{v}' as Timedelta: {e}")  # noqa: B904
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
        Test set size specification:
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
    """

    test_size: int | float | str | None = Field(
        None,
        description=(
            "Test set size: int (samples/sessions), float (proportion), "
            "str (time-based, e.g., '4W'). "
            "Time-based NOT supported with align_to_sessions=True."
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
            "Step size between splits (int: samples/sessions). None defaults to test_size (non-overlapping)."
        ),
    )
    isolate_groups: bool = Field(
        False,
        description=(
            "Default False for walk-forward (opt-in). Set True for multi-asset validation to prevent group leakage."
        ),
    )

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


class CombinatorialConfig(SplitterConfig):
    """Configuration for Combinatorial Cross-Validation (CPCV).

    Combinatorial CV is designed for multi-asset strategies and combating overfitting
    by creating multiple test sets from combinatorial group selections.

    Reference: Bailey & López de Prado (2014)
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
            "If True, only use contiguous test groups (less overfitting). If False, allow any combination."
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
