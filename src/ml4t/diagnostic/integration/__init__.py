"""Integration contracts for external libraries."""

from ml4t.diagnostic.integration.backtest_contract import (
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    EnvironmentType,
    EvaluationExport,
    PromotionWorkflow,
    StrategyMetadata,
    TradeRecord,
)
from ml4t.diagnostic.integration.data_contract import (
    AnomalyType,
    DataAnomaly,
    DataQualityMetrics,
    DataQualityReport,
    DataValidationRequest,
    Severity,
)
from ml4t.diagnostic.integration.engineer_contract import (
    EngineerConfig,
    PreprocessingRecommendation,
    TransformType,
)

__all__ = [
    # ml4t.data integration
    "AnomalyType",
    "DataAnomaly",
    "DataQualityMetrics",
    "DataQualityReport",
    "DataValidationRequest",
    "Severity",
    # ml4t.engineer integration
    "PreprocessingRecommendation",
    "EngineerConfig",
    "TransformType",
    # ml4t.backtest integration
    "ComparisonRequest",
    "ComparisonResult",
    "ComparisonType",
    "EnvironmentType",
    "EvaluationExport",
    "PromotionWorkflow",
    "StrategyMetadata",
    "TradeRecord",
]
