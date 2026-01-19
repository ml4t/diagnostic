"""Core performance metrics for financial ML evaluation.

This package implements the core metrics used across ml4t-diagnostic's
Four-Tier Validation Framework:

- **Tier 3**: Fast screening metrics (IC, hit rate)
- **Tier 2**: Statistical significance metrics (HAC-adjusted IC, Sharpe with CI)
- **Tier 1**: Comprehensive metrics (deflated Sharpe, maximum drawdown)

All metrics are implemented with:
- Mathematical correctness validated by property-based tests
- Numerical stability for edge cases
- Polars-native implementation for performance
- Support for confidence intervals and statistical inference

Submodules
----------
ic : Core Information Coefficient calculations
ic_statistics : HAC-adjusted IC and decay analysis
conditional_ic : IC conditional on feature quantiles
monotonicity : Monotonic relationship tests
risk_adjusted : Sharpe, Sortino, Maximum Drawdown
basic : Hit rate, forward returns
feature_outcome : Comprehensive feature-outcome analysis
importance_classical : Permutation and MDI importance
importance_shap : SHAP-based importance
importance_mda : Mean Decrease in Accuracy importance
importance_analysis : Multi-method importance comparison
interactions : Feature interaction detection (H-statistic, SHAP)
"""

# IC metrics
# Re-export cov_hac from statsmodels for backward compatibility
from statsmodels.stats.sandwich_covariance import cov_hac

# Basic metrics
from ml4t.diagnostic.evaluation.metrics.basic import (
    compute_forward_returns,
    hit_rate,
)

# Conditional IC
from ml4t.diagnostic.evaluation.metrics.conditional_ic import (
    compute_conditional_ic,
)

# Feature outcome analysis
from ml4t.diagnostic.evaluation.metrics.feature_outcome import (
    analyze_feature_outcome,
)

# IC statistics
from ml4t.diagnostic.evaluation.metrics.ic_statistics import (
    compute_ic_decay,
    compute_ic_hac_stats,
)
from ml4t.diagnostic.evaluation.metrics.importance_analysis import (
    analyze_ml_importance,
)

# Importance methods
from ml4t.diagnostic.evaluation.metrics.importance_classical import (
    compute_mdi_importance,
    compute_permutation_importance,
)
from ml4t.diagnostic.evaluation.metrics.importance_mda import (
    compute_mda_importance,
)
from ml4t.diagnostic.evaluation.metrics.importance_shap import (
    compute_shap_importance,
)
from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_by_horizon,
    compute_ic_ir,
    compute_ic_series,
    information_coefficient,
)

# Interaction detection
from ml4t.diagnostic.evaluation.metrics.interactions import (
    analyze_interactions,
    compute_h_statistic,
    compute_shap_interactions,
)

# Monotonicity
from ml4t.diagnostic.evaluation.metrics.monotonicity import (
    compute_monotonicity,
)

# Risk-adjusted metrics
from ml4t.diagnostic.evaluation.metrics.risk_adjusted import (
    maximum_drawdown,
    sharpe_ratio,
    sharpe_ratio_with_ci,
    sortino_ratio,
)

__all__ = [
    # IC metrics
    "information_coefficient",
    "compute_ic_series",
    "compute_ic_by_horizon",
    "compute_ic_ir",
    # IC statistics
    "compute_ic_hac_stats",
    "compute_ic_decay",
    "cov_hac",  # Re-exported from statsmodels for backward compatibility
    # Conditional IC
    "compute_conditional_ic",
    # Monotonicity
    "compute_monotonicity",
    # Risk-adjusted
    "sharpe_ratio",
    "sharpe_ratio_with_ci",
    "maximum_drawdown",
    "sortino_ratio",
    # Basic
    "hit_rate",
    "compute_forward_returns",
    # Feature outcome
    "analyze_feature_outcome",
    # Importance
    "compute_permutation_importance",
    "compute_mdi_importance",
    "compute_shap_importance",
    "compute_mda_importance",
    "analyze_ml_importance",
    # Interactions
    "compute_h_statistic",
    "compute_shap_interactions",
    "analyze_interactions",
]
