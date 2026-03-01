"""Feature Evaluation Configuration.

This module provides configuration for comprehensive feature analysis:
- Stationarity testing (ADF, KPSS, Phillips-Perron)
- Autocorrelation (ACF/PACF)
- Volatility analysis (GARCH effects)
- Distribution analysis (normality, outliers)
- Correlation analysis
- PCA and dimensionality reduction
- Redundancy detection
- Information Coefficient (IC)
- ML diagnostics (SHAP, drift)

Consolidated Config:
- DiagnosticConfig: Single config with all feature analysis settings (single-level nesting)

References
----------
López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.diagnostic.config.base import BaseConfig, StatisticalTestConfig
from ml4t.diagnostic.config.validation import (
    ClusteringMethod,
    CorrelationMethod,
    DistanceMetric,
    DriftDetectionMethod,
    LinkageMethod,
    NonNegativeInt,
    NormalityTest,
    OutlierMethod,
    PositiveFloat,
    PositiveInt,
    Probability,
    RegressionType,
    ThresholdOptimizationTarget,
    VolatilityClusterMethod,
    validate_min_max_range,
)

# =============================================================================
# Settings Classes (Single-Level Nesting)
# =============================================================================


class StationaritySettings(StatisticalTestConfig):
    """Settings for stationarity testing (ADF, KPSS, PP)."""

    adf_enabled: bool = Field(True, description="Run ADF test")
    kpss_enabled: bool = Field(True, description="Run KPSS test")
    pp_enabled: bool = Field(False, description="Run Phillips-Perron test")
    adf_regression: RegressionType = Field(
        RegressionType.CONSTANT, description="ADF regression type"
    )
    kpss_regression: Literal["c", "ct"] = Field("c", description="KPSS regression type")
    pp_regression: RegressionType = Field(RegressionType.CONSTANT, description="PP regression type")
    max_lag: Literal["auto"] | PositiveInt = Field("auto", description="Max lag for tests")

    @model_validator(mode="after")
    def check_at_least_one_test(self) -> StationaritySettings:
        """Ensure at least one test is enabled."""
        if not (self.adf_enabled or self.kpss_enabled or self.pp_enabled):
            raise ValueError("At least one stationarity test must be enabled")
        return self


class ACFSettings(BaseConfig):
    """Settings for autocorrelation (ACF/PACF) analysis."""

    enabled: bool = Field(True, description="Run ACF/PACF analysis")
    n_lags: Literal["auto"] | PositiveInt = Field(40, description="Number of lags")
    alpha: Probability = Field(0.05, description="Significance level for bands")
    compute_pacf: bool = Field(True, description="Also compute PACF")
    pacf_method: Literal["yw", "ols", "mle"] = Field("yw", description="PACF method")
    use_fft: bool = Field(True, description="Use FFT (faster)")


class VolatilitySettings(BaseConfig):
    """Settings for volatility analysis."""

    enabled: bool = Field(True, description="Run volatility analysis")
    window_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [21], description="Rolling windows"
    )
    detect_clustering: bool = Field(True, description="Test for GARCH effects")
    arch_lags: PositiveInt = Field(12, description="Lags for ARCH-LM test")
    fit_garch_model: bool = Field(True, description="Fit GARCH model when clustering detected")
    cluster_method: VolatilityClusterMethod = Field(
        VolatilityClusterMethod.LJUNG_BOX, description="Detection method"
    )
    significance_level: Probability = Field(0.05, description="Significance level")
    compute_rolling_vol: bool = Field(True, description="Compute rolling volatility")

    @field_validator("window_sizes")
    @classmethod
    def check_window_sizes(cls, v: list[int]) -> list[int]:
        """Ensure window sizes are valid."""
        if not v:
            raise ValueError("Must specify at least one window size")
        if any(w < 2 for w in v):
            raise ValueError("Window sizes must be >= 2")
        return sorted(v)


class DistributionSettings(BaseConfig):
    """Settings for distribution analysis."""

    enabled: bool = Field(True, description="Run distribution analysis")
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level")
    test_normality: bool = Field(True, description="Test for normality")
    normality_tests: list[NormalityTest] = Field(
        default_factory=lambda: [NormalityTest.JARQUE_BERA], description="Normality tests"
    )
    compute_moments: bool = Field(True, description="Compute skew/kurtosis")
    compute_tails: bool = Field(True, description="Analyze tail behavior")
    detect_outliers: bool = Field(False, description="Detect outliers")
    outlier_method: OutlierMethod = Field(OutlierMethod.ZSCORE, description="Outlier method")
    outlier_threshold: PositiveFloat = Field(3.0, description="Z-score threshold")


class CorrelationSettings(BaseConfig):
    """Settings for correlation analysis."""

    enabled: bool = Field(True, description="Run correlation analysis")
    methods: list[CorrelationMethod] = Field(
        default_factory=lambda: [CorrelationMethod.PEARSON], description="Correlation methods"
    )
    compute_pairwise: bool = Field(True, description="Compute pairwise correlations")
    min_periods: PositiveInt = Field(30, description="Minimum observations")
    lag_correlations: bool = Field(False, description="Compute lagged correlations")
    max_lag: PositiveInt = Field(10, description="Max lag")

    @field_validator("methods")
    @classmethod
    def check_methods(cls, v: list[CorrelationMethod]) -> list[CorrelationMethod]:
        """Ensure at least one method specified."""
        if not v:
            raise ValueError("Must specify at least one correlation method")
        return v


class PCASettings(BaseConfig):
    """Settings for PCA analysis."""

    enabled: bool = Field(False, description="Run PCA (opt-in)")
    n_components: PositiveInt | Probability | Literal["auto"] = Field(
        "auto", description="Components"
    )
    variance_threshold: Probability = Field(0.95, description="Variance to explain")
    standardize: bool = Field(True, description="Standardize features")
    rotation: Literal["varimax", "quartimax"] | None = Field(None, description="Rotation")

    @model_validator(mode="after")
    def check_n_components_config(self) -> PCASettings:
        """Validate n_components configuration."""
        if not self.enabled:
            return self
        if self.n_components == "auto" and not (0 < self.variance_threshold < 1):
            raise ValueError("variance_threshold must be in (0, 1) when n_components='auto'")
        return self


class ClusteringSettings(BaseConfig):
    """Settings for feature clustering."""

    enabled: bool = Field(False, description="Run clustering (opt-in)")
    method: ClusteringMethod = Field(ClusteringMethod.HIERARCHICAL, description="Algorithm")
    n_clusters: PositiveInt | Literal["auto"] = Field("auto", description="Number of clusters")
    linkage: LinkageMethod = Field(LinkageMethod.WARD, description="Linkage method")
    distance_metric: DistanceMetric = Field(DistanceMetric.EUCLIDEAN, description="Distance metric")
    min_cluster_size: PositiveInt = Field(5, description="Min cluster size")
    eps: PositiveFloat = Field(0.5, description="DBSCAN epsilon")


class RedundancySettings(BaseConfig):
    """Settings for redundancy detection."""

    enabled: bool = Field(True, description="Run redundancy detection")
    correlation_threshold: Probability = Field(0.95, description="Correlation threshold")
    compute_vif: bool = Field(False, description="Compute VIF")
    vif_threshold: PositiveFloat = Field(10.0, description="VIF threshold")
    keep_strategy: Literal["first", "last", "highest_ic"] = Field(
        "highest_ic", description="Keep strategy"
    )


class ICSettings(BaseConfig):
    """Settings for Information Coefficient analysis."""

    enabled: bool = Field(True, description="Run IC analysis")
    method: CorrelationMethod = Field(CorrelationMethod.PEARSON, description="Correlation method")
    lag_structure: list[NonNegativeInt] = Field(
        default_factory=lambda: [0, 1, 5], description="Lags to analyze"
    )
    hac_adjustment: bool = Field(False, description="Newey-West HAC")
    max_lag_hac: PositiveInt | Literal["auto"] = Field("auto", description="Max HAC lag")
    compute_t_stats: bool = Field(True, description="Compute t-stats")
    compute_decay: bool = Field(False, description="Analyze IC decay")

    @field_validator("lag_structure")
    @classmethod
    def check_lag_structure(cls, v: list[int]) -> list[int]:
        """Ensure lag structure is valid."""
        if not v:
            raise ValueError("Must specify at least one lag")
        if any(lag < 0 for lag in v):
            raise ValueError("Lags must be non-negative")
        return sorted(v)


class BinaryClassificationSettings(BaseConfig):
    """Settings for binary classification metrics."""

    enabled: bool = Field(False, description="Run binary classification (opt-in)")
    thresholds: list[float] = Field(default_factory=lambda: [0.0], description="Thresholds")
    metrics: list[Literal["precision", "recall", "f1", "lift", "coverage"]] = Field(
        default_factory=lambda: ["precision", "recall", "f1"],  # type: ignore[arg-type]
        description="Metrics",
    )
    positive_class: int | str = Field(1, description="Positive class label")
    compute_confusion_matrix: bool = Field(True, description="Compute confusion matrix")
    compute_roc_curve: bool = Field(False, description="Compute ROC curve")


class ThresholdAnalysisSettings(BaseConfig):
    """Settings for threshold optimization."""

    enabled: bool = Field(False, description="Run threshold analysis (opt-in)")
    sweep_range: tuple[float, float] = Field((-2.0, 2.0), description="Threshold range")
    n_points: PositiveInt = Field(50, description="Sweep points")
    optimization_target: ThresholdOptimizationTarget = Field(
        ThresholdOptimizationTarget.SHARPE, description="Optimization target"
    )
    constraint_metric: str | None = Field(None, description="Constraint metric")
    constraint_value: float | None = Field(None, description="Constraint value")
    constraint_type: Literal[">=", "<=", "=="] = Field(">=", description="Constraint type")

    @model_validator(mode="after")
    def validate_sweep_range(self) -> ThresholdAnalysisSettings:
        """Validate sweep range."""
        if self.enabled:
            validate_min_max_range(self.sweep_range[0], self.sweep_range[1], "sweep_range")
        return self

    @model_validator(mode="after")
    def validate_constraint(self) -> ThresholdAnalysisSettings:
        """Validate constraint configuration."""
        has_metric = self.constraint_metric is not None
        has_value = self.constraint_value is not None
        if has_metric != has_value:
            raise ValueError(
                "Both constraint_metric and constraint_value must be set (or both None)"
            )
        return self


class MLDiagnosticsSettings(BaseConfig):
    """Settings for ML diagnostics (importance, SHAP, drift)."""

    enabled: bool = Field(True, description="Run ML diagnostics")
    feature_importance: bool = Field(True, description="Compute importance")
    importance_method: Literal["tree", "permutation"] = Field(
        "tree", description="Importance method"
    )
    shap_analysis: bool = Field(False, description="Compute SHAP (expensive)")
    shap_sample_size: PositiveInt | None = Field(None, description="SHAP subsample size")
    drift_detection: bool = Field(False, description="Detect drift")
    drift_method: DriftDetectionMethod = Field(
        DriftDetectionMethod.KOLMOGOROV_SMIRNOV, description="Drift method"
    )
    drift_window: PositiveInt = Field(63, description="Drift window")


# =============================================================================
# Consolidated Config
# =============================================================================


class DiagnosticConfig(BaseConfig):
    """Consolidated configuration for feature analysis (single-level nesting).

    Provides comprehensive feature diagnostics with direct access to all settings:
    - config.stationarity.enabled (not config.module_a.stationarity.enabled)

    Examples
    --------
    >>> config = DiagnosticConfig(
    ...     stationarity=StationaritySettings(significance_level=0.01),
    ...     ic=ICSettings(lag_structure=[0, 1, 5, 10, 21]),
    ... )
    >>> config.to_yaml("diagnostic_config.yaml")
    """

    # Feature Diagnostics (Module A)
    stationarity: StationaritySettings = Field(
        default_factory=StationaritySettings, description="Stationarity testing"
    )
    acf: ACFSettings = Field(default_factory=ACFSettings, description="ACF/PACF analysis")
    volatility: VolatilitySettings = Field(
        default_factory=VolatilitySettings, description="Volatility analysis"
    )
    distribution: DistributionSettings = Field(
        default_factory=DistributionSettings, description="Distribution analysis"
    )

    # Cross-Feature Analysis (Module B)
    correlation: CorrelationSettings = Field(
        default_factory=CorrelationSettings, description="Correlation analysis"
    )
    pca: PCASettings = Field(default_factory=PCASettings, description="PCA analysis")
    clustering: ClusteringSettings = Field(
        default_factory=ClusteringSettings, description="Feature clustering"
    )
    redundancy: RedundancySettings = Field(
        default_factory=RedundancySettings, description="Redundancy detection"
    )

    # Feature-Outcome (Module C)
    ic: ICSettings = Field(default_factory=ICSettings, description="IC analysis")
    binary_classification: BinaryClassificationSettings = Field(
        default_factory=BinaryClassificationSettings, description="Binary classification"
    )
    threshold_analysis: ThresholdAnalysisSettings = Field(
        default_factory=ThresholdAnalysisSettings, description="Threshold optimization"
    )
    ml_diagnostics: MLDiagnosticsSettings = Field(
        default_factory=MLDiagnosticsSettings, description="ML diagnostics"
    )

    # Execution settings
    export_recommendations: bool = Field(True, description="Export recommendations")
    export_to_qfeatures: bool = Field(False, description="Export in qfeatures format")
    return_dataframes: bool = Field(True, description="Return as DataFrames")
    n_jobs: int = Field(-1, ge=-1, description="Parallel jobs")
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ml4t-diagnostic" / "features",
        description="Cache directory",
    )
    verbose: bool = Field(False, description="Verbose output")

    @classmethod
    def for_quick_analysis(cls) -> DiagnosticConfig:
        """Preset for quick exploratory analysis."""
        return cls(
            stationarity=StationaritySettings(pp_enabled=False),
            volatility=VolatilitySettings(detect_clustering=False),
            distribution=DistributionSettings(detect_outliers=False),
            correlation=CorrelationSettings(lag_correlations=False),
            pca=PCASettings(enabled=False),
            clustering=ClusteringSettings(enabled=False),
            ic=ICSettings(hac_adjustment=False, compute_decay=False),
            ml_diagnostics=MLDiagnosticsSettings(shap_analysis=False, drift_detection=False),
        )

    @classmethod
    def for_research(cls) -> DiagnosticConfig:
        """Preset for academic research (comprehensive)."""
        return cls(
            stationarity=StationaritySettings(pp_enabled=True),
            volatility=VolatilitySettings(window_sizes=[10, 21, 63]),
            distribution=DistributionSettings(
                detect_outliers=True,
                normality_tests=[
                    NormalityTest.JARQUE_BERA,
                    NormalityTest.SHAPIRO,
                    NormalityTest.ANDERSON,
                ],
            ),
            correlation=CorrelationSettings(
                methods=[
                    CorrelationMethod.PEARSON,
                    CorrelationMethod.SPEARMAN,
                    CorrelationMethod.KENDALL,
                ],
                lag_correlations=True,
            ),
            pca=PCASettings(enabled=True),
            clustering=ClusteringSettings(enabled=True),
            ic=ICSettings(lag_structure=[0, 1, 5, 10, 21], hac_adjustment=True, compute_decay=True),
            binary_classification=BinaryClassificationSettings(enabled=True),
            threshold_analysis=ThresholdAnalysisSettings(enabled=True),
            ml_diagnostics=MLDiagnosticsSettings(shap_analysis=True, drift_detection=True),
        )

    @classmethod
    def for_production(cls) -> DiagnosticConfig:
        """Preset for production monitoring (fast, focused on drift)."""
        return cls(
            stationarity=StationaritySettings(pp_enabled=False),
            acf=ACFSettings(enabled=False),
            volatility=VolatilitySettings(enabled=False),
            distribution=DistributionSettings(test_normality=False, compute_moments=True),
            correlation=CorrelationSettings(lag_correlations=False),
            pca=PCASettings(enabled=False),
            clustering=ClusteringSettings(enabled=False),
            ic=ICSettings(compute_decay=False),
            ml_diagnostics=MLDiagnosticsSettings(
                feature_importance=True, drift_detection=True, drift_window=21
            ),
        )
