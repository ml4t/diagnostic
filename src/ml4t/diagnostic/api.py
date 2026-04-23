"""Canonical stable API for ml4t-diagnostic."""

from ml4t.diagnostic.config import DiagnosticConfig, ValidatedCrossValidationConfig
from ml4t.diagnostic.evaluation.barrier_analysis import BarrierAnalysis
from ml4t.diagnostic.evaluation.feature_diagnostics import (
    FeatureDiagnostics,
    FeatureDiagnosticsResult,
)
from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis
from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis
from ml4t.diagnostic.evaluation.validated_cv import (
    ValidatedCrossValidation,
    ValidationFoldResult,
    ValidationResult,
    validated_cross_val_score,
)
from ml4t.diagnostic.metrics import (
    analyze_interactions,
    analyze_ml_importance,
    compute_h_statistic,
    compute_ic_hac_stats,
    compute_mdi_importance,
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_interactions,
    cross_sectional_ic_series,
)
from ml4t.diagnostic.signal import SignalResult, analyze_signal
from ml4t.diagnostic.splitters import CombinatorialCV, WalkForwardCV

__all__ = [
    "ValidatedCrossValidation",
    "ValidatedCrossValidationConfig",
    "ValidationFoldResult",
    "ValidationResult",
    "validated_cross_val_score",
    "BarrierAnalysis",
    "FeatureDiagnostics",
    "DiagnosticConfig",
    "FeatureDiagnosticsResult",
    "TradeAnalysis",
    "PortfolioAnalysis",
    "analyze_signal",
    "SignalResult",
    "CombinatorialCV",
    "WalkForwardCV",
    "cross_sectional_ic_series",
    "compute_ic_hac_stats",
    "compute_mdi_importance",
    "compute_permutation_importance",
    "compute_shap_importance",
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_ml_importance",
    "analyze_interactions",
]
