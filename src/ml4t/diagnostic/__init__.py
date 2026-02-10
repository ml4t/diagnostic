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
>>> from ml4t.diagnostic.splitters import CombinatorialCV
>>>
>>> # One-step validated cross-validation
>>> vcv = ValidatedCrossValidation(n_splits=10)
>>> result = vcv.fit_validate(model, X, y, times)
>>> if result.is_significant:
...     print(f"Sharpe: {result.sharpe:.2f}, DSR p-value: {result.dsr_pvalue:.4f}")

API Stability
-------------
This library follows semantic versioning. The public API consists of all symbols
exported in __all__. Breaking changes will only occur in major version bumps.
"""

__version__ = "0.1.0a5"

# Sub-modules for advanced usage
from . import backends, caching, config, core, evaluation, integration, logging, signal, splitters

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

# Signal analysis (new clean API)
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


def get_agent_docs() -> dict[str, str]:
    """Get AGENT.md documentation for AI agent navigation.

    Returns a dictionary mapping relative paths to AGENT.md content.
    Useful for AI agents to understand the library structure.

    Returns
    -------
    dict[str, str]
        Mapping of relative path to AGENT.md content.

    Example
    -------
    >>> docs = get_agent_docs()
    >>> print(docs.keys())
    dict_keys(['AGENT.md', 'signal/AGENT.md', 'splitters/AGENT.md', ...])
    """
    from pathlib import Path

    package_dir = Path(__file__).parent
    agent_docs = {}

    # Find all AGENT.md files
    for agent_file in package_dir.rglob("AGENT.md"):
        rel_path = agent_file.relative_to(package_dir)
        try:
            agent_docs[str(rel_path)] = agent_file.read_text()
        except OSError:
            continue

    return agent_docs


__all__ = [
    # Version
    "__version__",
    # Agent Navigation
    "get_agent_docs",
    # Core Framework
    "Evaluator",
    "EvaluationResult",
    "ValidatedCrossValidation",
    # Signal Analysis (new clean API)
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
    "config",
    "core",
    "evaluation",
    "integration",
    "logging",
    "signal",
    "splitters",
]
