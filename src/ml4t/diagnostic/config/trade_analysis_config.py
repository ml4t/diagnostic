"""Trade Analysis Configuration.

This module provides consolidated configuration for trade-level analysis:
- Trade extraction (worst/best trades by PnL)
- Trade filtering (duration, regime, symbol)
- SHAP alignment (map SHAP values to trades)
- Error pattern clustering
- Automated hypothesis generation

Consolidated Config:
- TradeConfig: Single config with all trade analysis settings

References
----------
López de Prado, M. (2018). "Advances in Financial Machine Learning"
Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig
from ml4t.diagnostic.config.validation import (
    ClusteringMethod,
    DistanceMetric,
    LinkageMethod,
    NonNegativeInt,
    PositiveInt,
    Probability,
)

# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class FilterSettings(BaseConfig):
    """Settings for filtering trades before analysis."""

    min_duration: timedelta | None = Field(None, description="Minimum trade duration")
    max_duration: timedelta | None = Field(None, description="Maximum trade duration")
    min_pnl: float | None = Field(None, description="Minimum PnL threshold")
    exclude_symbols: list[str] | None = Field(None, description="Symbols to exclude")
    regime_filter: str | None = Field(None, description="Only analyze specific regime")

    @field_validator("min_duration", "max_duration")
    @classmethod
    def validate_duration_positive(cls, v: timedelta | None) -> timedelta | None:
        """Ensure durations are positive if provided."""
        if v is not None and v.total_seconds() <= 0:
            raise ValueError("Duration must be positive")
        return v

    @field_validator("max_duration")
    @classmethod
    def validate_max_greater_than_min(cls, v: timedelta | None, info) -> timedelta | None:
        """Ensure max_duration > min_duration if both provided."""
        min_duration = info.data.get("min_duration")
        if v is not None and min_duration is not None and v <= min_duration:
            raise ValueError("max_duration must be greater than min_duration")
        return v


class ExtractionSettings(BaseConfig):
    """Settings for extracting worst/best trades."""

    n_worst: PositiveInt = Field(20, description="Number of worst trades to extract")
    n_best: NonNegativeInt = Field(10, description="Number of best trades for comparison")
    percentile_mode: bool = Field(False, description="Interpret n_worst/n_best as percentiles")
    compute_statistics: bool = Field(True, description="Compute aggregate statistics")
    group_by_symbol: bool = Field(False, description="Group analysis by symbol")
    group_by_regime: bool = Field(False, description="Group analysis by regime")

    @field_validator("n_worst")
    @classmethod
    def check_n_worst_reasonable(cls, v: int, info) -> int:
        """Warn if n_worst is very small or very large."""
        import warnings

        percentile_mode = info.data.get("percentile_mode", False)
        if percentile_mode:
            if v > 50:
                warnings.warn(
                    f"n_worst={v}% includes majority of trades. Consider 5-20%.",
                    stacklevel=2,
                )
        else:
            if v < 10:
                warnings.warn(
                    f"n_worst={v} may be too few. Consider 20-50 for better signal.",
                    stacklevel=2,
                )
            elif v > 100:
                warnings.warn(
                    f"n_worst={v} may dilute signal. Consider 20-50 for clearer patterns.",
                    stacklevel=2,
                )
        return v

    @field_validator("n_best", "n_worst")
    @classmethod
    def validate_percentile_range(cls, v: int, info) -> int:
        """Ensure percentile values are in valid range."""
        percentile_mode = info.data.get("percentile_mode", False)
        if percentile_mode and (v < 1 or v > 100):
            raise ValueError(f"Percentile must be 1-100, got {v}")
        return v


class AlignmentSettings(BaseConfig):
    """Settings for aligning SHAP values to trade timestamps."""

    mode: Literal["entry", "nearest", "average"] = Field("entry", description="Alignment mode")
    tolerance: PositiveInt = Field(
        300, description="Max time difference for 'nearest' mode (seconds)"
    )
    missing_strategy: Literal["error", "skip", "zero"] = Field(
        "skip", description="Missing value handling"
    )
    top_n_features: PositiveInt | None = Field(
        None, description="Top N features per trade (None=all)"
    )

    @field_validator("tolerance")
    @classmethod
    def warn_large_tolerance(cls, v: int) -> int:
        """Warn if tolerance is very large."""
        if v > 3600:
            import warnings

            warnings.warn(
                f"tolerance={v}s (>{v // 3600}h) may misalign SHAP values.",
                stacklevel=2,
            )
        return v


class ClusteringSettings(BaseConfig):
    """Settings for clustering error patterns in trades."""

    method: ClusteringMethod = Field(
        ClusteringMethod.HIERARCHICAL, description="Clustering algorithm"
    )
    linkage: LinkageMethod = Field(LinkageMethod.WARD, description="Linkage for hierarchical")
    distance_metric: DistanceMetric = Field(DistanceMetric.EUCLIDEAN, description="Distance metric")
    min_cluster_size: PositiveInt = Field(5, description="Minimum trades per cluster")
    max_clusters: PositiveInt | None = Field(None, description="Max clusters (None=auto)")
    normalization: Literal["l2", "l1", "standardize", None] = Field(
        "l2", description="SHAP normalization"
    )

    @model_validator(mode="after")
    def validate_ward_requires_euclidean(self) -> ClusteringSettings:
        """Ensure Ward linkage uses Euclidean distance."""
        if self.linkage == LinkageMethod.WARD and self.distance_metric != DistanceMetric.EUCLIDEAN:
            raise ValueError(
                f"Ward linkage requires Euclidean distance, got {self.distance_metric}. "
                "Use linkage='average' or 'complete' for other metrics."
            )
        return self

    @field_validator("min_cluster_size")
    @classmethod
    def warn_small_cluster_size(cls, v: int) -> int:
        """Warn if min_cluster_size is very small."""
        if v < 3:
            import warnings

            warnings.warn(f"min_cluster_size={v} may not be reliable. Use >= 5.", stacklevel=2)
        return v


class HypothesisSettings(BaseConfig):
    """Settings for automated hypothesis generation."""

    enabled: bool = Field(True, description="Generate hypotheses automatically")
    min_confidence: Probability = Field(0.6, description="Minimum confidence threshold")
    max_per_cluster: PositiveInt = Field(5, description="Max hypotheses per cluster")
    include_interactions: bool = Field(True, description="Look for feature × regime interactions")
    template_library: Literal["comprehensive", "minimal", "custom"] = Field(
        "comprehensive", description="Template set"
    )


# =============================================================================
# Consolidated Config
# =============================================================================


class TradeConfig(BaseConfig):
    """Consolidated configuration for trade analysis.

    Combines trade extraction, filtering, SHAP alignment, error pattern
    clustering, and hypothesis generation into a single configuration.

    Examples
    --------
    >>> config = TradeConfig(
    ...     extraction=ExtractionSettings(n_worst=50),
    ...     clustering=ClusteringSettings(min_cluster_size=10),
    ... )
    >>> config.to_yaml("trade_config.yaml")
    """

    extraction: ExtractionSettings = Field(
        default_factory=ExtractionSettings, description="Trade extraction settings"
    )
    filter: FilterSettings = Field(
        default_factory=FilterSettings, description="Trade filtering settings"
    )
    alignment: AlignmentSettings = Field(
        default_factory=AlignmentSettings, description="SHAP alignment settings"
    )
    clustering: ClusteringSettings = Field(
        default_factory=ClusteringSettings, description="Clustering settings"
    )
    hypothesis: HypothesisSettings = Field(
        default_factory=HypothesisSettings, description="Hypothesis generation"
    )

    min_trades_for_clustering: PositiveInt = Field(
        20, description="Minimum trades required for clustering"
    )
    generate_visualizations: bool = Field(True, description="Generate SHAP waterfall plots")
    cache_shap_vectors: bool = Field(True, description="Cache SHAP vectors for performance")

    # Convenience properties
    @property
    def n_worst(self) -> int:
        """Number of worst trades (shortcut)."""
        return self.extraction.n_worst

    @property
    def n_best(self) -> int:
        """Number of best trades (shortcut)."""
        return self.extraction.n_best

    @field_validator("min_trades_for_clustering")
    @classmethod
    def warn_low_min_trades(cls, v: int) -> int:
        """Warn if min_trades is very low."""
        if v < 10:
            import warnings

            warnings.warn(
                f"min_trades_for_clustering={v} may not identify reliable patterns. Use >= 20.",
                stacklevel=2,
            )
        return v

    @classmethod
    def for_quick_diagnostics(cls) -> TradeConfig:
        """Preset for quick diagnostics (minimal clustering)."""
        return cls(
            extraction=ExtractionSettings(n_worst=20, n_best=10),
            alignment=AlignmentSettings(top_n_features=10),
            clustering=ClusteringSettings(min_cluster_size=3, max_clusters=5),
            hypothesis=HypothesisSettings(template_library="minimal", max_per_cluster=3),
            min_trades_for_clustering=10,
            generate_visualizations=False,
        )

    @classmethod
    def for_deep_analysis(cls) -> TradeConfig:
        """Preset for comprehensive analysis."""
        return cls(
            extraction=ExtractionSettings(n_worst=50, n_best=20, compute_statistics=True),
            alignment=AlignmentSettings(top_n_features=None, mode="average"),
            clustering=ClusteringSettings(
                method=ClusteringMethod.HIERARCHICAL,
                linkage=LinkageMethod.WARD,
                min_cluster_size=10,
                max_clusters=None,
                normalization="l2",
            ),
            hypothesis=HypothesisSettings(
                min_confidence=0.6,
                max_per_cluster=10,
                include_interactions=True,
                template_library="comprehensive",
            ),
            min_trades_for_clustering=30,
            generate_visualizations=True,
        )

    @classmethod
    def for_production(cls) -> TradeConfig:
        """Preset for production monitoring (efficient, focused)."""
        return cls(
            extraction=ExtractionSettings(n_worst=20, n_best=5, group_by_symbol=True),
            alignment=AlignmentSettings(top_n_features=15),
            clustering=ClusteringSettings(min_cluster_size=5, max_clusters=8),
            hypothesis=HypothesisSettings(min_confidence=0.7, max_per_cluster=3),
            min_trades_for_clustering=15,
            generate_visualizations=False,
            cache_shap_vectors=True,
        )


# Rebuild models
FilterSettings.model_rebuild()
ExtractionSettings.model_rebuild()
AlignmentSettings.model_rebuild()
ClusteringSettings.model_rebuild()
HypothesisSettings.model_rebuild()
TradeConfig.model_rebuild()
