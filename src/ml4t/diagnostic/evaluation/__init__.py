"""Evaluation framework implementing the Three-Tier Validation Framework.

This module provides the Evaluator class, metrics, statistical tests, and
visualization tools for comprehensive model validation.
"""

from ml4t.diagnostic.caching.smart_cache import SmartCache
from ml4t.diagnostic.results.barrier_results import (
    BarrierTearSheet,
    HitRateResult,
    PrecisionRecallResult,
    ProfitFactorResult,
    TimeToTargetResult,
)
from ml4t.diagnostic.results.multi_signal_results import (
    ComparisonResult,
    MultiSignalSummary,
)

from . import (  # noqa: F401 (module re-export)
    diagnostic_plots,
    drift,
    metrics,
    report_generation,
    stats,
    visualization,
)
from .autocorrelation import (
    ACFResult,
    AutocorrelationAnalysisResult,
    PACFResult,
    analyze_autocorrelation,
    compute_acf,
    compute_pacf,
)
from .barrier_analysis import BarrierAnalysis
from .binary_metrics import (
    BinaryClassificationReport,
    ConfusionMatrix,
    balanced_accuracy,
    binary_classification_report,
    binomial_test_precision,
    compare_precisions_z_test,
    compute_all_metrics,
    compute_confusion_matrix,
    coverage,
    f1_score,
    format_classification_report,
    lift,
    precision,
    proportions_z_test,
    recall,
    specificity,
    wilson_score_interval,
)
from .dashboard import create_evaluation_dashboard
from .diagnostic_plots import (
    plot_acf_pacf,
    plot_distribution,
    plot_qq,
    plot_volatility_clustering,
)
from .distribution import (
    DistributionAnalysisResult,
    HillEstimatorResult,
    JarqueBeraResult,
    MomentsResult,
    QQPlotData,
    ShapiroWilkResult,
    TailAnalysisResult,
    analyze_distribution,
    analyze_tails,
    compute_moments,
    generate_qq_data,
    hill_estimator,
    jarque_bera_test,
    shapiro_wilk_test,
)
from .drift import (
    PSIResult,
    compute_psi,
)
from .event_analysis import EventStudyAnalysis
from .excursion import (
    ExcursionAnalysisResult,
    ExcursionStats,
    analyze_excursions,
    compute_excursions,
)
from .feature_diagnostics import (
    FeatureDiagnostics,
    FeatureDiagnosticsConfig,
    FeatureDiagnosticsResult,
)
from .feature_outcome import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcome,
    FeatureOutcomeResult,
)
from .framework import EvaluationResult, Evaluator, get_metric_directionality
from .metric_registry import MetricRegistry
from .metrics import (
    analyze_feature_outcome,
    analyze_interactions,
    analyze_ml_importance,
    compute_conditional_ic,
    compute_forward_returns,
    compute_h_statistic,
    compute_ic_by_horizon,
    compute_ic_decay,
    compute_ic_hac_stats,
    compute_ic_ir,
    compute_ic_series,
    compute_mda_importance,
    compute_mdi_importance,
    compute_monotonicity,
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_interactions,
    information_coefficient,
)
from .multi_signal import MultiSignalAnalysis
from .portfolio_analysis import (
    DistributionResult,
    DrawdownPeriod,
    DrawdownResult,
    # Portfolio Analysis (pyfolio replacement)
    PortfolioAnalysis,
    PortfolioMetrics,
    RollingMetricsResult,
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    compute_portfolio_turnover,
    conditional_var,
    information_ratio,
    max_drawdown,
    omega_ratio,
    # Core metric functions
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    up_down_capture,
    value_at_risk,
)
from .report_generation import (
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
    generate_multi_feature_html_report,
    save_report,
)
from .signal_selector import SignalSelector
from .stat_registry import StatTestRegistry
from .stationarity import (
    ADFResult,
    KPSSResult,
    PPResult,
    StationarityAnalysisResult,
    adf_test,
    analyze_stationarity,
    kpss_test,
    pp_test,
)
from .stats import (
    benjamini_hochberg_fdr,
    compute_pbo,
    holm_bonferroni,
    ras_ic_adjustment,
)
from .threshold_analysis import (
    MonotonicityResult,
    OptimalThresholdResult,
    SensitivityResult,
    ThresholdAnalysisSummary,
    analyze_all_metrics_monotonicity,
    analyze_threshold_sensitivity,
    check_monotonicity,
    create_threshold_analysis_summary,
    evaluate_percentile_thresholds,
    evaluate_threshold_sweep,
    find_optimal_threshold,
    find_threshold_for_target_coverage,
    format_threshold_analysis,
)
from .trade_analysis import (
    TradeAnalysis,
    TradeAnalysisResult,
    TradeMetrics,
    TradeStatistics,
)
from .trade_shap_diagnostics import (
    ClusteringResult,
    ErrorPattern,
    TradeExplainFailure,
    TradeShapAnalyzer,
    TradeShapExplanation,
    TradeShapResult,
)
from .validated_cv import (
    ValidatedCrossValidation,
    ValidatedCrossValidationConfig,
    ValidationFoldResult,
    ValidationResult,
    validated_cross_val_score,
)
from .volatility import (
    ARCHLMResult,
    GARCHResult,
    VolatilityAnalysisResult,
    analyze_volatility,
    arch_lm_test,
    fit_garch,
)

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
    "EvaluationResult",
    "Evaluator",
    "get_metric_directionality",
    "MetricRegistry",
    "StatTestRegistry",
    "create_evaluation_dashboard",
    "metrics",
    "stats",
    "visualization",
    "diagnostic_plots",
    # IC Time Series (Alphalens-style)
    "information_coefficient",
    "compute_forward_returns",
    "compute_ic_series",
    "compute_ic_by_horizon",
    "compute_ic_ir",
    "compute_ic_hac_stats",
    "compute_ic_decay",
    "compute_conditional_ic",
    "compute_monotonicity",
    "analyze_feature_outcome",
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_interactions",
    "compute_permutation_importance",
    "compute_mdi_importance",
    "compute_mda_importance",
    "compute_shap_importance",
    "analyze_ml_importance",
    # Diagnostic Plotting Functions
    "plot_acf_pacf",
    "plot_qq",
    "plot_volatility_clustering",
    "plot_distribution",
    # Feature Diagnostics (Main API)
    "FeatureDiagnostics",
    "FeatureDiagnosticsConfig",
    "FeatureDiagnosticsResult",
    # Stationarity tests
    "adf_test",
    "ADFResult",
    "kpss_test",
    "KPSSResult",
    "pp_test",
    "PPResult",
    "analyze_stationarity",
    "StationarityAnalysisResult",
    # Autocorrelation
    "compute_acf",
    "ACFResult",
    "compute_pacf",
    "PACFResult",
    "analyze_autocorrelation",
    "AutocorrelationAnalysisResult",
    # Volatility
    "arch_lm_test",
    "ARCHLMResult",
    "fit_garch",
    "GARCHResult",
    "analyze_volatility",
    "VolatilityAnalysisResult",
    # Distribution
    "compute_moments",
    "MomentsResult",
    "jarque_bera_test",
    "JarqueBeraResult",
    "shapiro_wilk_test",
    "ShapiroWilkResult",
    "hill_estimator",
    "HillEstimatorResult",
    "generate_qq_data",
    "QQPlotData",
    "analyze_tails",
    "TailAnalysisResult",
    "analyze_distribution",
    "DistributionAnalysisResult",
    # Report generation
    "generate_html_report",
    "generate_json_report",
    "generate_markdown_report",
    "generate_multi_feature_html_report",
    "save_report",
    # Drift detection
    "compute_psi",
    "PSIResult",
    # Price Excursion Analysis (TP/SL parameter selection)
    "analyze_excursions",
    "compute_excursions",
    "ExcursionAnalysisResult",
    "ExcursionStats",
    # Feature-Outcome Analysis (Module C Orchestration)
    "FeatureOutcome",
    "FeatureOutcomeResult",
    "FeatureICResults",
    "FeatureImportanceResults",
    # Trade Analysis (ml4t-diagnostics v1.0)
    "TradeAnalysis",
    "TradeAnalysisResult",
    "TradeMetrics",
    "TradeStatistics",
    # Trade-SHAP Diagnostics (ml4t-diagnostics v1.0 - KILLER FEATURE)
    "ClusteringResult",
    "ErrorPattern",
    "TradeExplainFailure",
    "TradeShapAnalyzer",
    "TradeShapExplanation",
    "TradeShapResult",
    # Dashboard (optional - requires streamlit)
    "run_diagnostics_dashboard",
    # Event Study Analysis (Phase 2 - MacKinlay 1997)
    "EventStudyAnalysis",
    # Multiple Testing Corrections (Phase 3 - Multi-Signal)
    "benjamini_hochberg_fdr",
    "holm_bonferroni",
    # Backtest Overfitting Detection
    "compute_pbo",
    "ras_ic_adjustment",
    # Signal Selection Algorithms (Phase 3 - Multi-Signal)
    "SignalSelector",
    # Multi-Signal Analysis (Phase 3)
    "MultiSignalAnalysis",
    "MultiSignalSummary",
    "ComparisonResult",
    # Caching (Phase 3)
    "SmartCache",
    # Barrier Analysis (Phase 4)
    "BarrierAnalysis",
    "HitRateResult",
    "ProfitFactorResult",
    "PrecisionRecallResult",
    "TimeToTargetResult",
    "BarrierTearSheet",
    # Portfolio Analysis (pyfolio replacement)
    "PortfolioAnalysis",
    "PortfolioMetrics",
    "RollingMetricsResult",
    "DrawdownResult",
    "DrawdownPeriod",
    "DistributionResult",
    # Portfolio metric functions
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "max_drawdown",
    "annual_return",
    "annual_volatility",
    "value_at_risk",
    "conditional_var",
    "stability_of_timeseries",
    "alpha_beta",
    "information_ratio",
    "up_down_capture",
    "compute_portfolio_turnover",
    # Binary Classification Metrics (Phase 2 - Book Alignment)
    "BinaryClassificationReport",
    "ConfusionMatrix",
    "balanced_accuracy",
    "binary_classification_report",
    "binomial_test_precision",
    "compare_precisions_z_test",
    "compute_all_metrics",
    "compute_confusion_matrix",
    "coverage",
    "f1_score",
    "format_classification_report",
    "lift",
    "precision",
    "proportions_z_test",
    "recall",
    "specificity",
    "wilson_score_interval",
    # Threshold Analysis (Phase 2 - Book Alignment)
    "MonotonicityResult",
    "OptimalThresholdResult",
    "SensitivityResult",
    "ThresholdAnalysisSummary",
    "analyze_all_metrics_monotonicity",
    "analyze_threshold_sensitivity",
    "check_monotonicity",
    "create_threshold_analysis_summary",
    "evaluate_percentile_thresholds",
    "evaluate_threshold_sweep",
    "find_optimal_threshold",
    "find_threshold_for_target_coverage",
    "format_threshold_analysis",
    # Validated Cross-Validation (CPCV + DSR)
    "ValidatedCrossValidation",
    "ValidatedCrossValidationConfig",
    "validated_cross_val_score",
    "ValidationFoldResult",
    "ValidationResult",
]
