"""Signal Analysis Configuration.

This module provides configuration for signal analysis including IC calculation,
quantile analysis, turnover metrics, and multi-signal batch analysis.

Consolidated Config:
- SignalConfig: Single config with analysis, visualization, and multi-signal settings

References
----------
López de Prado, M. (2018). "Advances in Financial Machine Learning"
Paleologo, G. (2024). "Elements of Quantitative Investing"
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig


class ICMethod(str, Enum):
    """Information Coefficient calculation method."""

    SPEARMAN = "spearman"
    PEARSON = "pearson"


class QuantileMethod(str, Enum):
    """Method for assigning quantile labels."""

    QUANTILE = "quantile"  # pd.qcut - equal frequency
    UNIFORM = "uniform"  # pd.cut - equal width bins


# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class AnalysisSettings(BaseConfig):
    """Settings for signal analysis calculations."""

    quantiles: int = Field(default=5, ge=2, le=20, description="Number of quantile bins")
    periods: tuple[int, ...] = Field(default=(1, 5, 10), description="Forward return periods")
    max_loss: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Max fraction of data to drop"
    )
    filter_zscore: float | None = Field(
        default=3.0, ge=1.0, le=10.0, description="Outlier z-score threshold"
    )
    zero_aware: bool = Field(default=False, description="Separate quantiles around zero")
    ic_method: ICMethod = Field(default=ICMethod.SPEARMAN, description="IC correlation method")
    ic_by_group: bool = Field(default=False, description="Compute IC by group")
    hac_lags: int | None = Field(default=None, ge=0, description="Newey-West lags")
    quantile_method: QuantileMethod = Field(
        default=QuantileMethod.QUANTILE, description="Quantile binning method"
    )
    quantile_labels: list[str] | None = Field(default=None, description="Custom quantile labels")
    cumulative_returns: bool = Field(default=True, description="Compute cumulative returns")
    spread_confidence: float = Field(
        default=0.95, ge=0.80, le=0.99, description="Spread confidence level"
    )
    compute_turnover: bool = Field(default=True, description="Compute turnover metrics")
    autocorrelation_lags: int = Field(default=5, ge=1, le=20, description="Autocorrelation lags")
    cost_per_trade: float = Field(default=0.001, ge=0.0, le=0.05, description="Transaction cost")
    group_column: str | None = Field(default=None, description="Group column for analysis")

    @field_validator("periods", mode="before")
    @classmethod
    def validate_periods(cls, v: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Ensure periods is a tuple of positive integers."""
        if isinstance(v, list):
            v = tuple(v)
        if not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError("All periods must be positive integers")
        if len(v) == 0:
            raise ValueError("At least one period is required")
        return tuple(sorted(set(v)))


class RASSettings(BaseConfig):
    """Settings for Rademacher Anti-Serum adjustment."""

    enabled: bool = Field(default=True, description="Apply RAS adjustment")
    delta: float = Field(default=0.05, ge=0.001, le=0.20, description="Significance level")
    kappa: float = Field(default=0.02, ge=0.001, le=1.0, description="IC bound")
    n_simulations: int = Field(
        default=10000, ge=1000, le=100000, description="Monte Carlo simulations"
    )


class VisualizationSettings(BaseConfig):
    """Settings for signal tear sheet visualization."""

    theme: Literal["default", "dark", "print", "presentation"] = Field(default="default")
    width: int = Field(default=1000, ge=400, le=2000, description="Plot width")
    height_multiplier: float = Field(default=1.0, ge=0.5, le=2.0, description="Height scaling")
    include_ic_plots: bool = Field(default=True)
    include_quantile_plots: bool = Field(default=True)
    include_turnover_plots: bool = Field(default=True)
    include_summary_table: bool = Field(default=True)
    ic_rolling_window: int = Field(default=21, ge=5, le=252, description="IC rolling window")
    ic_significance_bands: bool = Field(default=True)
    ic_heatmap_freq: Literal["M", "Q", "Y"] = Field(default="M")
    html_self_contained: bool = Field(default=True)
    html_include_plotlyjs: Literal["cdn", "directory", True, False] = Field(default="cdn")
    export_data: bool = Field(default=False)


class MultiSignalSettings(BaseConfig):
    """Settings for multi-signal batch analysis."""

    fdr_alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="FDR alpha")
    fwer_alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="FWER alpha")
    min_ic_threshold: float = Field(default=0.0, ge=-1.0, le=1.0, description="Min IC threshold")
    min_observations: int = Field(default=100, ge=10, description="Min observations")
    n_jobs: int = Field(default=-1, ge=-1, description="Parallel jobs")
    backend: Literal["loky", "threading", "multiprocessing"] = Field(default="loky")
    cache_enabled: bool = Field(default=True)
    cache_max_items: int = Field(default=200, ge=10, le=10000)
    cache_ttl: int | None = Field(default=3600, ge=60)
    max_signals_summary: int = Field(default=200, ge=10, le=1000)
    max_signals_comparison: int = Field(default=20, ge=2, le=50)
    max_signals_heatmap: int = Field(default=100, ge=10, le=500)
    default_selection_metric: str = Field(default="ic_ir")
    default_correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("default_selection_metric")
    @classmethod
    def validate_selection_metric(cls, v: str) -> str:
        """Validate selection metric."""
        valid = {"ic_mean", "ic_ir", "ic_t_stat", "turnover_adj_ic", "quantile_spread"}
        if v not in valid:
            raise ValueError(f"Invalid selection metric '{v}'. Valid: {valid}")
        return v


# =============================================================================
# Consolidated Config
# =============================================================================


class SignalConfig(BaseConfig):
    """Consolidated configuration for signal analysis.

    Combines analysis settings, RAS adjustment, visualization, and
    multi-signal batch analysis into a single configuration class.

    Examples
    --------
    >>> config = SignalConfig(
    ...     analysis=AnalysisSettings(quantiles=10, periods=(1, 5)),
    ...     visualization=VisualizationSettings(theme="dark"),
    ... )
    >>> config.to_yaml("signal_config.yaml")
    """

    analysis: AnalysisSettings = Field(
        default_factory=AnalysisSettings, description="Analysis settings"
    )
    ras: RASSettings = Field(default_factory=RASSettings, description="RAS adjustment settings")
    visualization: VisualizationSettings = Field(
        default_factory=VisualizationSettings, description="Visualization settings"
    )
    multi: MultiSignalSettings = Field(
        default_factory=MultiSignalSettings, description="Multi-signal settings"
    )

    signal_name: str = Field(default="signal", description="Signal name for reports")
    return_pandas: bool = Field(default=False, description="Return pandas instead of Polars")

    @model_validator(mode="after")
    def validate_quantile_labels_count(self) -> SignalConfig:
        """Ensure quantile_labels matches quantiles count if provided."""
        if self.analysis.quantile_labels is not None:
            if len(self.analysis.quantile_labels) != self.analysis.quantiles:
                raise ValueError(
                    f"quantile_labels length ({len(self.analysis.quantile_labels)}) "
                    f"must match quantiles ({self.analysis.quantiles})"
                )
        return self

    # Convenience properties
    @property
    def quantiles(self) -> int:
        """Number of quantiles (shortcut)."""
        return self.analysis.quantiles

    @property
    def periods(self) -> tuple[int, ...]:
        """Forward return periods (shortcut)."""
        return self.analysis.periods

    @property
    def filter_zscore(self) -> float | None:
        """Outlier z-score threshold (shortcut)."""
        return self.analysis.filter_zscore

    @property
    def compute_turnover(self) -> bool:
        """Compute turnover metrics (shortcut)."""
        return self.analysis.compute_turnover
