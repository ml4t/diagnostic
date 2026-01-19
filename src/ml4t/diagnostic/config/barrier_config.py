"""Configuration for Barrier Analysis module.

This module provides Pydantic configuration classes for controlling
barrier analysis behavior including hit rate calculation, profit factor
analysis, and visualization options.

Triple barrier outcomes from ml4t.features:
- label: int (-1=SL hit, 0=timeout, 1=TP hit)
- label_return: float (actual return at exit)
- label_bars: int (bars from entry to exit)

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
    Chapter 3: Labeling (Triple Barrier Method)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig


class BarrierLabel(int, Enum):
    """Triple barrier outcome labels."""

    STOP_LOSS = -1  # Lower barrier hit
    TIMEOUT = 0  # Time barrier hit
    TAKE_PROFIT = 1  # Upper barrier hit


class DecileMethod(str, Enum):
    """Method for assigning decile labels."""

    QUANTILE = "quantile"  # Equal frequency bins (pd.qcut style)
    UNIFORM = "uniform"  # Equal width bins (pd.cut style)


# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class AnalysisSettings(BaseConfig):
    """Settings for barrier analysis calculations."""

    n_quantiles: int = Field(
        default=10,
        ge=2,
        le=20,
        description="Number of quantile bins (deciles=10)",
    )
    decile_method: DecileMethod = Field(
        default=DecileMethod.QUANTILE,
        description="Method for quantile binning",
    )
    min_observations_per_quantile: int = Field(
        default=30,
        ge=10,
        le=1000,
        description="Minimum observations per quantile",
    )
    filter_zscore: float | None = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Z-score threshold for outlier filtering (None to disable)",
    )
    drop_timeout: bool = Field(
        default=False,
        description="Exclude timeout outcomes from calculations",
    )
    significance_level: float = Field(
        default=0.05,
        ge=0.001,
        le=0.20,
        description="Significance level for statistical tests",
    )
    bootstrap_n_resamples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of bootstrap resamples",
    )
    hit_rate_min_observations: int = Field(
        default=20,
        ge=5,
        le=500,
        description="Minimum observations for hit rate reporting",
    )
    profit_factor_epsilon: float = Field(
        default=1e-10,
        ge=1e-15,
        le=1e-6,
        description="Epsilon for division safety",
    )


class ColumnSettings(BaseConfig):
    """Settings for column name mappings."""

    signal_col: str = Field(default="signal", description="Signal column name")
    date_col: str = Field(default="date", description="Date column name")
    asset_col: str = Field(default="asset", description="Asset column name")
    label_col: str = Field(default="label", description="Barrier label column")
    label_return_col: str = Field(default="label_return", description="Return at exit column")
    label_bars_col: str = Field(default="label_bars", description="Bars to exit column")

    @field_validator(
        "signal_col", "date_col", "asset_col", "label_col", "label_return_col", "label_bars_col"
    )
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Ensure column names are non-empty."""
        if not v or not v.strip():
            raise ValueError("Column name must be non-empty")
        return v.strip()


class VisualizationSettings(BaseConfig):
    """Settings for barrier tear sheet visualization."""

    theme: Literal["default", "dark", "print", "presentation"] = Field(
        default="default",
        description="Plotly theme",
    )
    width: int = Field(
        default=1000,
        ge=400,
        le=2000,
        description="Plot width in pixels",
    )
    height_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Height scaling factor",
    )
    include_hit_rate_heatmap: bool = Field(default=True)
    include_profit_factor: bool = Field(default=True)
    include_time_to_target: bool = Field(default=True)
    include_precision_recall: bool = Field(default=True)
    include_summary_table: bool = Field(default=True)
    html_self_contained: bool = Field(default=True)
    html_include_plotlyjs: Literal["cdn", "directory", True, False] = Field(default="cdn")
    export_data: bool = Field(default=False)


# =============================================================================
# Consolidated Config
# =============================================================================


class BarrierConfig(BaseConfig):
    """Consolidated configuration for barrier analysis.

    Combines analysis settings, column mappings, and visualization options
    into a single configuration class.

    Examples
    --------
    >>> config = BarrierConfig(
    ...     analysis=AnalysisSettings(n_quantiles=5),
    ...     visualization=VisualizationSettings(theme="dark"),
    ... )
    >>> config.to_yaml("barrier_config.yaml")
    """

    analysis: AnalysisSettings = Field(
        default_factory=AnalysisSettings,
        description="Analysis calculation settings",
    )
    columns: ColumnSettings = Field(
        default_factory=ColumnSettings,
        description="Column name mappings",
    )
    visualization: VisualizationSettings = Field(
        default_factory=VisualizationSettings,
        description="Visualization settings",
    )
    signal_name: str = Field(
        default="signal",
        description="Signal name for reports",
    )
    return_pandas: bool = Field(
        default=False,
        description="Return pandas instead of Polars",
    )

    @model_validator(mode="after")
    def validate_column_uniqueness(self) -> BarrierConfig:
        """Ensure column names don't conflict."""
        cols = [self.columns.signal_col, self.columns.date_col, self.columns.asset_col]
        if len(cols) != len(set(cols)):
            raise ValueError(f"Column names must be unique: {cols}")
        return self

    # Convenience properties for flat access (backward compatibility)
    @property
    def n_quantiles(self) -> int:
        """Number of quantiles (shortcut)."""
        return self.analysis.n_quantiles

    @property
    def significance_level(self) -> float:
        """Significance level (shortcut)."""
        return self.analysis.significance_level

    @property
    def decile_method(self) -> DecileMethod:
        """Decile method (shortcut)."""
        return self.analysis.decile_method

    @property
    def min_observations_per_quantile(self) -> int:
        """Minimum observations per quantile (shortcut)."""
        return self.analysis.min_observations_per_quantile

    @property
    def filter_zscore(self) -> float | None:
        """Z-score filter threshold (shortcut)."""
        return self.analysis.filter_zscore

    @property
    def drop_timeout(self) -> bool:
        """Drop timeout outcomes (shortcut)."""
        return self.analysis.drop_timeout

    @property
    def bootstrap_n_resamples(self) -> int:
        """Bootstrap resamples (shortcut)."""
        return self.analysis.bootstrap_n_resamples

    @property
    def hit_rate_min_observations(self) -> int:
        """Hit rate minimum observations (shortcut)."""
        return self.analysis.hit_rate_min_observations

    @property
    def profit_factor_epsilon(self) -> float:
        """Profit factor epsilon (shortcut)."""
        return self.analysis.profit_factor_epsilon

    # Column name properties
    @property
    def signal_col(self) -> str:
        """Signal column name (shortcut)."""
        return self.columns.signal_col

    @property
    def date_col(self) -> str:
        """Date column name (shortcut)."""
        return self.columns.date_col

    @property
    def asset_col(self) -> str:
        """Asset column name (shortcut)."""
        return self.columns.asset_col

    @property
    def label_col(self) -> str:
        """Label column name (shortcut)."""
        return self.columns.label_col

    @property
    def label_return_col(self) -> str:
        """Label return column name (shortcut)."""
        return self.columns.label_return_col

    @property
    def label_bars_col(self) -> str:
        """Label bars column name (shortcut)."""
        return self.columns.label_bars_col
