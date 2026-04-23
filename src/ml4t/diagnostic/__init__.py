"""ml4t-diagnostic - A hierarchical framework for financial time-series validation.

ml4t-diagnostic provides rigorous validation tools for financial machine learning models,
implementing a Four-Tier Validation Framework to combat data leakage, backtest
overfitting, and statistical fallacies.

Main Features
-------------
- **Cross-Validation**: Combinatorial CV, Walk-Forward CV with label overlap prevention
- **Statistical Validity**: DSR, RAS, FDR corrections for multiple testing
- **Feature Analysis**: IC, importance (MDI/PFI/MDA/SHAP), interactions
- **Trade Diagnostics**: SHAP-based error pattern analysis
- **Data Quality**: Integration contracts with ml4t-data

Quick Start
-----------
>>> from ml4t.diagnostic import ValidatedCrossValidation
>>> from ml4t.diagnostic.config import ValidatedCrossValidationConfig
>>> from ml4t.diagnostic.splitters import CombinatorialCV
>>>
>>> # One-step validated cross-validation
>>> config = ValidatedCrossValidationConfig(n_groups=10, n_test_groups=2)
>>> vcv = ValidatedCrossValidation(config=config)
>>> result = vcv.fit_evaluate(X, y, model, times=times)
>>> if result.is_significant:
...     print(f"Mean Sharpe: {result.mean_sharpe:.2f}, DSR: {result.dsr:.4f}")

API Stability
-------------
This library follows semantic versioning. The public API consists of all symbols
exported in __all__. Breaking changes will only occur in major version bumps.
"""

__version__ = "0.1.0b14"

# Sub-modules for advanced usage
from . import (
    api,
    backends,
    caching,
    config,
    core,
    evaluation,
    integration,
    logging,
    selection,
    signal,
    splitters,
)

# Configuration classes
from .config import (
    BarrierConfig,
    DiagnosticConfig,
    EventConfig,
    PortfolioConfig,
    ReportConfig,
    RuntimeConfig,
    SignalConfig,
    StatisticalConfig,
    TradeConfig,
)

# Main evaluation framework
from .evaluation import BarrierAnalysis, EvaluationResult, Evaluator

# ValidatedCrossValidation - combines CPCV + DSR in one step
from .evaluation.validated_cv import ValidatedCrossValidation

# Data quality integration
from .integration.data_contract import (
    AnomalyType,
    DataAnomaly,
    DataQualityMetrics,
    DataQualityReport,
    DataValidationRequest,
    Severity,
)

# Feature selection
from .selection import FeatureSelector, SelectionReport

# Signal analysis
from .signal import SignalResult, analyze_signal

# Visualization (optional - may fail if plotly not installed)
try:
    from .visualization import (
        plot_hit_rate_heatmap,
        plot_precision_recall_curve,
        plot_profit_factor_bar,
        plot_time_to_target_box,
    )

    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False
    plot_hit_rate_heatmap = None
    plot_precision_recall_curve = None
    plot_profit_factor_bar = None
    plot_time_to_target_box = None


__all__ = [
    # Version
    "__version__",
    # Core Framework
    "Evaluator",
    "EvaluationResult",
    "ValidatedCrossValidation",
    # Feature Selection
    "FeatureSelector",
    "SelectionReport",
    # Signal Analysis
    "analyze_signal",
    "SignalResult",
    # Barrier Analysis
    "BarrierAnalysis",
    # Configuration (10 primary configs)
    "DiagnosticConfig",
    "StatisticalConfig",
    "PortfolioConfig",
    "TradeConfig",
    "SignalConfig",
    "EventConfig",
    "BarrierConfig",
    "ReportConfig",
    "RuntimeConfig",
    # Data Quality Integration
    "DataQualityReport",
    "DataQualityMetrics",
    "DataAnomaly",
    "DataValidationRequest",
    "AnomalyType",
    "Severity",
    # Visualization (optional)
    "plot_hit_rate_heatmap",
    "plot_profit_factor_bar",
    "plot_precision_recall_curve",
    "plot_time_to_target_box",
    # Sub-modules
    "backends",
    "caching",
    "api",
    "config",
    "core",
    "evaluation",
    "integration",
    "logging",
    "selection",
    "signal",
    "splitters",
]
