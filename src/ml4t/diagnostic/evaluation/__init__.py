"""Evaluation framework implementing the Three-Tier Validation Framework.

This module re-exports the public API for evaluation workflows.
For low-level functions (binary metrics, stationarity tests, distribution
analysis, portfolio metric functions, etc.), import from the submodule directly::

    from ml4t.diagnostic.evaluation.binary_metrics import precision, recall
    from ml4t.diagnostic.evaluation.stationarity import adf_test
    from ml4t.diagnostic.evaluation.portfolio_analysis import sharpe_ratio

For the stable integration surface, use ``ml4t.diagnostic.api``.
"""

from ml4t.diagnostic.caching.smart_cache import SmartCache  # noqa: F401
from ml4t.diagnostic.results.multi_signal_results import (  # noqa: F401
    ComparisonResult,
    MultiSignalSummary,
)

from . import drift, metrics, stats  # noqa: F401 (module re-export)
from .barrier_analysis import BarrierAnalysis  # noqa: F401
from .event_analysis import EventStudyAnalysis  # noqa: F401

# Factor exposure and attribution
from .factor import (  # noqa: F401
    AttributionResult,
    FactorAnalysis,
    FactorData,
    FactorModelResult,
    RiskAttributionResult,
    RollingExposureResult,
    compute_factor_model,
    compute_return_attribution,
    compute_risk_attribution,
    compute_rolling_exposures,
)
from .feature_diagnostics import (  # noqa: F401
    FeatureDiagnostics,
    FeatureDiagnosticsAnalysisResult,
    FeatureDiagnosticsResult,
)
from .framework import EvaluationResult, Evaluator, get_metric_directionality  # noqa: F401
from .metric_registry import MetricRegistry  # noqa: F401
from .metrics import (  # noqa: F401
    analyze_feature_outcome,
    analyze_interactions,
    analyze_ml_importance,
    compute_h_statistic,
    compute_ic_hac_stats,
    compute_ic_series,
    compute_mdi_importance,
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_interactions,
)
from .multi_signal import MultiSignalAnalysis  # noqa: F401
from .portfolio_analysis import (  # noqa: F401
    PortfolioAnalysis,
    PortfolioMetrics,
)
from .signal_selector import SignalSelector  # noqa: F401
from .stat_registry import StatTestRegistry  # noqa: F401
from .trade_analysis import (  # noqa: F401
    TradeAnalysis,
    TradeAnalysisResult,
    TradeMetrics,
    TradeStatistics,
)
from .trade_shap_diagnostics import (  # noqa: F401
    ClusteringResult,
    ErrorPattern,
    TradeExplainFailure,
    TradeShapAnalyzer,
    TradeShapExplanation,
    TradeShapResult,
)
from .validated_cv import (  # noqa: F401
    ValidatedCrossValidation,
    ValidationFoldResult,
    ValidationResult,
    validated_cross_val_score,
)

# Optional visualization/report stack (plotly/matplotlib/seaborn).
# Keep core evaluation importable without viz extras.
try:
    from . import visualization  # noqa: F401
    from .dashboard import create_evaluation_dashboard  # noqa: F401
except ImportError:
    visualization = None  # type: ignore[assignment]
    create_evaluation_dashboard = None  # type: ignore[assignment]

# Lazy import for dashboard functions to avoid slow Streamlit import at module load
# This saves ~1.3 seconds on every import of ml4t.diagnostic
_dashboard_module = None
_HAS_STREAMLIT: bool | None = None  # Will be set on first access


def __getattr__(name: str):
    """Lazy load dashboard functions to avoid importing Streamlit at module load."""
    global _dashboard_module, _HAS_STREAMLIT

    if name == "run_diagnostics_dashboard":
        if _dashboard_module is None:
            try:
                from . import trade_shap_dashboard as _mod

                _dashboard_module = _mod
                _HAS_STREAMLIT = True
            except ImportError:
                _HAS_STREAMLIT = False
                return None

        return _dashboard_module.run_diagnostics_dashboard

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    # Core framework
    "EvaluationResult",
    "Evaluator",
    "get_metric_directionality",
    "MetricRegistry",
    "StatTestRegistry",
    # Validation workflows
    "ValidatedCrossValidation",
    "validated_cross_val_score",
    "ValidationFoldResult",
    "ValidationResult",
    # Analysis workflows
    "FeatureDiagnostics",
    "FeatureDiagnosticsAnalysisResult",
    "FeatureDiagnosticsResult",
    "TradeAnalysis",
    "TradeAnalysisResult",
    "TradeMetrics",
    "TradeStatistics",
    "TradeShapAnalyzer",
    "TradeShapResult",
    "TradeShapExplanation",
    "TradeExplainFailure",
    "ErrorPattern",
    "ClusteringResult",
    "BarrierAnalysis",
    "PortfolioAnalysis",
    "PortfolioMetrics",
    "EventStudyAnalysis",
    "SignalSelector",
    "MultiSignalAnalysis",
    "MultiSignalSummary",
    "ComparisonResult",
    # Stable metrics facade
    "analyze_feature_outcome",
    "compute_ic_series",
    "compute_ic_hac_stats",
    "compute_permutation_importance",
    "compute_mdi_importance",
    "compute_shap_importance",
    "analyze_ml_importance",
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_interactions",
    # Namespaces and dashboard entry points
    "create_evaluation_dashboard",
    "metrics",
    "stats",
    "visualization",
    "run_diagnostics_dashboard",
    # Utilities
    "SmartCache",
    # Factor exposure and attribution
    "FactorAnalysis",
    "FactorData",
    "FactorModelResult",
    "RollingExposureResult",
    "AttributionResult",
    "RiskAttributionResult",
    "compute_factor_model",
    "compute_rolling_exposures",
    "compute_return_attribution",
    "compute_risk_attribution",
]
