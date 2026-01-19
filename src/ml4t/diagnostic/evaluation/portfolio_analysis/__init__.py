"""Portfolio analysis module.

This package provides comprehensive portfolio performance analysis:
- PortfolioAnalysis: Main analyzer class for portfolio diagnostics
- PortfolioMetrics: Complete portfolio performance metrics
- RollingMetricsResult: Rolling metrics over multiple windows
- DrawdownResult: Detailed drawdown analysis
- DistributionResult: Returns distribution analysis

Decomposed from portfolio_analysis.py (1,620 lines) into:
- results.py: Result dataclasses (~335 lines)
- metrics.py: Core metric functions (~588 lines)
- analysis.py: PortfolioAnalysis class (~672 lines)
"""

from __future__ import annotations

# Main analysis class
from ml4t.diagnostic.evaluation.portfolio_analysis.analysis import PortfolioAnalysis

# Core metric functions
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import (
    _annualization_factor,
    _safe_cumprod,
    _safe_prod,
    _to_numpy,
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    compute_portfolio_turnover,
    conditional_var,
    information_ratio,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    stability_of_timeseries,
    tail_ratio,
    up_down_capture,
    value_at_risk,
)

# Result classes
from ml4t.diagnostic.evaluation.portfolio_analysis.results import (
    DistributionResult,
    DrawdownPeriod,
    DrawdownResult,
    PortfolioMetrics,
    RollingMetricsResult,
)

__all__ = [
    # Main class
    "PortfolioAnalysis",
    # Result classes
    "PortfolioMetrics",
    "RollingMetricsResult",
    "DrawdownPeriod",
    "DrawdownResult",
    "DistributionResult",
    # Core metric functions
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "tail_ratio",
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
    # Internal helpers (exported for testing)
    "_to_numpy",
    "_safe_prod",
    "_safe_cumprod",
    "_annualization_factor",
]
