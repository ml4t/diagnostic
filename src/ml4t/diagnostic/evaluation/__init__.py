"""Evaluation framework implementing the Three-Tier Validation Framework.

This module provides the Evaluator class, metrics, statistical tests, and
visualization tools for comprehensive model validation.
"""

# ruff: noqa: F401

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
    FeatureDiagnosticsAnalysisResult,
    FeatureDiagnosticsResult,
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
    # Core framework
    "EvaluationResult",
    "Evaluator",
    "get_metric_directionality",
    "MetricRegistry",
    "StatTestRegistry",
    # Main workflows
    "ValidatedCrossValidation",
    "ValidatedCrossValidationConfig",
    "validated_cross_val_score",
    "ValidationFoldResult",
    "ValidationResult",
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
    "HitRateResult",
    "ProfitFactorResult",
    "PrecisionRecallResult",
    "TimeToTargetResult",
    "BarrierTearSheet",
    "PortfolioAnalysis",
    "PortfolioMetrics",
    "RollingMetricsResult",
    "DrawdownResult",
    "DrawdownPeriod",
    "DistributionResult",
    "EventStudyAnalysis",
    "SignalSelector",
    "MultiSignalAnalysis",
    "MultiSignalSummary",
    "ComparisonResult",
    # Metrics and statistical analysis
    "analyze_feature_outcome",
    "compute_forward_returns",
    "information_coefficient",
    "compute_ic_series",
    "compute_ic_by_horizon",
    "compute_ic_ir",
    "compute_ic_hac_stats",
    "compute_ic_decay",
    "compute_conditional_ic",
    "compute_monotonicity",
    "compute_permutation_importance",
    "compute_mdi_importance",
    "compute_mda_importance",
    "compute_shap_importance",
    "analyze_ml_importance",
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_interactions",
    "benjamini_hochberg_fdr",
    "holm_bonferroni",
    "compute_pbo",
    "ras_ic_adjustment",
    # API namespaces and dashboards
    "create_evaluation_dashboard",
    "metrics",
    "stats",
    "visualization",
    "run_diagnostics_dashboard",
    # Utilities
    "SmartCache",
]
