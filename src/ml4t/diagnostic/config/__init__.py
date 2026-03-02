"""ML4T Diagnostic Configuration System.

This module provides comprehensive Pydantic v2 configuration schemas for the
ML4T Diagnostic framework, covering:

- **Feature Evaluation**: Diagnostics, cross-feature analysis, feature-outcome relationships
- **Portfolio Evaluation**: Risk/return metrics, Bayesian comparison
- **Statistical Framework**: PSR, MinTRL, DSR, FDR for multiple testing correction
- **Reporting**: HTML, JSON, visualization settings

Examples:
    Quick start with defaults:

    >>> from ml4t.diagnostic.config import DiagnosticConfig
    >>> config = DiagnosticConfig()

    Custom configuration:

    >>> config = DiagnosticConfig(
    ...     stationarity=StationaritySettings(significance_level=0.01),
    ...     ic=ICSettings(lag_structure=[0, 1, 5, 10]),
    ... )

    Load from YAML:

    >>> config = DiagnosticConfig.from_yaml("config.yaml")

    Use presets:

    >>> config = DiagnosticConfig.for_quick_analysis()
"""

# Base configuration
# Barrier analysis configuration
from ml4t.diagnostic.config.barrier_config import (
    AnalysisSettings as BarrierAnalysisSettings,
)
from ml4t.diagnostic.config.barrier_config import (
    BarrierConfig,
    BarrierLabel,
    DecileMethod,
)
from ml4t.diagnostic.config.barrier_config import (
    ColumnSettings as BarrierColumnSettings,
)
from ml4t.diagnostic.config.barrier_config import (
    VisualizationSettings as BarrierVisualizationSettings,
)
from ml4t.diagnostic.config.base import (
    BaseConfig,
    RuntimeConfig,
    StatisticalTestConfig,
)

# Event study configuration
from ml4t.diagnostic.config.event_config import (
    EventConfig,
)
from ml4t.diagnostic.config.event_config import (
    WindowSettings as EventWindowSettings,
)

# Feature evaluation configuration (consolidated)
from ml4t.diagnostic.config.feature_config import (
    ACFSettings,
    BinaryClassificationSettings,
    ClusteringSettings,
    CorrelationSettings,
    DiagnosticConfig,
    DistributionSettings,
    ICSettings,
    MLDiagnosticsSettings,
    PCASettings,
    RedundancySettings,
    StationaritySettings,
    ThresholdAnalysisSettings,
    VolatilitySettings,
)

# Multi-signal analysis configuration
from ml4t.diagnostic.config.multi_signal_config import (
    MultiSignalAnalysisConfig,
)

# Portfolio evaluation configuration (consolidated)
from ml4t.diagnostic.config.portfolio_config import (
    AggregationSettings as PortfolioAggregationSettings,
)
from ml4t.diagnostic.config.portfolio_config import (
    BayesianSettings as PortfolioBayesianSettings,
)
from ml4t.diagnostic.config.portfolio_config import (
    DrawdownSettings as PortfolioDrawdownSettings,
)
from ml4t.diagnostic.config.portfolio_config import (
    MetricsSettings as PortfolioMetricsSettings,
)
from ml4t.diagnostic.config.portfolio_config import (
    PortfolioConfig,
)

# Report configuration
from ml4t.diagnostic.config.report_config import (
    HTMLConfig,
    JSONConfig,
    OutputFormatConfig,
    ReportConfig,
    VisualizationConfig,
)

# Statistical testing configuration (consolidated)
from ml4t.diagnostic.config.sharpe_config import (
    DSRSettings,
    FDRSettings,
    MinTRLSettings,
    PSRSettings,
    StatisticalConfig,
)

# Signal analysis configuration (consolidated)
from ml4t.diagnostic.config.signal_config import (
    AnalysisSettings as SignalAnalysisSettings,
)
from ml4t.diagnostic.config.signal_config import (
    ICMethod,
    MultiSignalSettings,
    QuantileMethod,
    RASSettings,
    SignalConfig,
)
from ml4t.diagnostic.config.signal_config import (
    VisualizationSettings as SignalVisualizationSettings,
)

# Trade analysis configuration (consolidated)
from ml4t.diagnostic.config.trade_analysis_config import (
    AlignmentSettings as TradeAlignmentSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    CharacterizationSettings as TradeCharacterizationSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    ClusteringSettings as TradeClusteringSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    ExtractionSettings as TradeExtractionSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    FilterSettings as TradeFilterSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    HypothesisSettings as TradeHypothesisSettings,
)
from ml4t.diagnostic.config.trade_analysis_config import (
    TradeConfig,
)
from ml4t.diagnostic.config.validated_cv_config import (
    ValidatedCrossValidationConfig,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "StatisticalTestConfig",
    "RuntimeConfig",
    # Feature evaluation (consolidated)
    "DiagnosticConfig",
    "StationaritySettings",
    "ACFSettings",
    "VolatilitySettings",
    "DistributionSettings",
    "CorrelationSettings",
    "PCASettings",
    "ClusteringSettings",
    "RedundancySettings",
    "ICSettings",
    "BinaryClassificationSettings",
    "ThresholdAnalysisSettings",
    "MLDiagnosticsSettings",
    # Portfolio evaluation (consolidated)
    "PortfolioConfig",
    "PortfolioMetricsSettings",
    "PortfolioBayesianSettings",
    "PortfolioAggregationSettings",
    "PortfolioDrawdownSettings",
    # Statistical testing (consolidated)
    "StatisticalConfig",
    "PSRSettings",
    "MinTRLSettings",
    "DSRSettings",
    "FDRSettings",
    # Reporting
    "ReportConfig",
    "OutputFormatConfig",
    "HTMLConfig",
    "VisualizationConfig",
    "JSONConfig",
    # Trade analysis (consolidated)
    "TradeConfig",
    "TradeExtractionSettings",
    "TradeFilterSettings",
    "TradeAlignmentSettings",
    "TradeClusteringSettings",
    "TradeCharacterizationSettings",
    "TradeHypothesisSettings",
    # Validated cross-validation
    "ValidatedCrossValidationConfig",
    # Signal analysis (consolidated)
    "SignalConfig",
    "SignalAnalysisSettings",
    "RASSettings",
    "SignalVisualizationSettings",
    "MultiSignalSettings",
    "ICMethod",
    "QuantileMethod",
    # Event study (consolidated)
    "EventConfig",
    "EventWindowSettings",
    # Multi-signal analysis
    "MultiSignalAnalysisConfig",
    # Barrier analysis (consolidated)
    "BarrierConfig",
    "BarrierAnalysisSettings",
    "BarrierColumnSettings",
    "BarrierVisualizationSettings",
    "BarrierLabel",
    "DecileMethod",
]
