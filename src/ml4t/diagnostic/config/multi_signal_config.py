"""Configuration for multi-signal analysis.

Provides configuration for analyzing and comparing multiple trading signals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from ml4t.diagnostic.config.base import BaseConfig
from ml4t.diagnostic.config.signal_config import SignalConfig


class MultiSignalAnalysisConfig(BaseConfig):
    """Configuration for multi-signal analysis.

    Controls behavior for analyzing and comparing multiple trading signals,
    including FDR/FWER corrections and parallelization settings.
    """

    signal_config: SignalConfig = Field(
        default_factory=SignalConfig,
        description="Configuration applied to all individual signal analyses",
    )
    fdr_alpha: float = Field(default=0.05, ge=0.001, le=0.5)
    fwer_alpha: float = Field(default=0.05, ge=0.001, le=0.5)
    min_ic_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    min_observations: int = Field(default=100, ge=10)
    n_jobs: int = Field(default=-1, ge=-1)
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
        """Validate selection metric is supported."""
        valid_metrics = {
            "ic_mean",
            "ic_ir",
            "ic_t_stat",
            "turnover_adj_ic",
            "quantile_spread",
        }
        if v not in valid_metrics:
            raise ValueError(f"Invalid selection metric '{v}'. Valid options: {valid_metrics}")
        return v
