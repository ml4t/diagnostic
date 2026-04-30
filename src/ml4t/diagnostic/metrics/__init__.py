"""
Metrics module for ML4T Diagnostic.

Provides statistical metrics and percentile computation utilities for model evaluation.
"""

from statsmodels.stats.sandwich_covariance import cov_hac

from ml4t.diagnostic.metrics.basic import compute_forward_returns, hit_rate
from ml4t.diagnostic.metrics.conditional import compute_conditional_ic
from ml4t.diagnostic.metrics.feature_outcome import analyze_feature_outcome
from ml4t.diagnostic.metrics.ic import (
    compute_ic_by_horizon,
    compute_ic_ir,
    cross_sectional_ic,
    cross_sectional_ic_series,
    information_coefficient,
    pooled_ic,
)
from ml4t.diagnostic.metrics.ic_inference import (
    compute_ic_decay,
    compute_ic_hac_stats,
    compute_ic_summary_stats,
)
from ml4t.diagnostic.metrics.importance import (
    analyze_ml_importance,
    compute_mda_importance,
    compute_mdi_importance,
    compute_permutation_importance,
    compute_shap_importance,
)
from ml4t.diagnostic.metrics.interactions import (
    analyze_interactions,
    compute_h_statistic,
    compute_shap_interactions,
)
from ml4t.diagnostic.metrics.monotonicity import compute_monotonicity
from ml4t.diagnostic.metrics.percentiles import compute_fold_percentiles
from ml4t.diagnostic.metrics.risk_adjusted import (
    maximum_drawdown,
    periodic_sharpe_ratio,
    periodic_sortino_ratio,
    sharpe_ratio,
    sharpe_ratio_with_ci,
    sortino_ratio,
)
from ml4t.diagnostic.metrics.uncertainty import (
    compute_auc_uncertainty,
    compute_ic_uncertainty,
    cross_sectional_auc_series,
)

__all__ = [
    "hit_rate",
    "compute_forward_returns",
    "pooled_ic",
    "information_coefficient",
    "cross_sectional_ic",
    "cross_sectional_ic_series",
    "compute_ic_by_horizon",
    "compute_ic_ir",
    "compute_ic_summary_stats",
    "compute_ic_hac_stats",
    "compute_ic_decay",
    "cov_hac",
    "compute_conditional_ic",
    "compute_monotonicity",
    "sharpe_ratio",
    "periodic_sharpe_ratio",
    "sharpe_ratio_with_ci",
    "maximum_drawdown",
    "sortino_ratio",
    "periodic_sortino_ratio",
    "analyze_feature_outcome",
    "compute_permutation_importance",
    "compute_mdi_importance",
    "compute_mda_importance",
    "compute_shap_importance",
    "analyze_ml_importance",
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_interactions",
    "compute_fold_percentiles",
    "cross_sectional_auc_series",
    "compute_ic_uncertainty",
    "compute_auc_uncertainty",
]
