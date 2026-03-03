"""Result types for feature selection.

These dataclasses define the interface between feature evaluation
(IC analysis, importance scoring, drift detection) and feature selection.

The FeatureSelector consumes these types to apply filtering criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ml4t.diagnostic.evaluation.drift import DriftSummaryResult


@dataclass
class FeatureICResults:
    """IC analysis results for a single feature.

    Attributes:
        feature: Feature name
        ic_mean: Mean information coefficient across periods
        ic_std: Standard deviation of IC
        ic_ir: Information Ratio (ic_mean / ic_std)
        t_stat: T-statistic for IC significance
        p_value: P-value for IC significance
        ic_by_lag: IC values at specific forward lags
        n_observations: Number of observations used
    """

    feature: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    t_stat: float
    p_value: float
    ic_by_lag: dict[int, float]
    n_observations: int


@dataclass
class FeatureImportanceResults:
    """Feature importance results from ML models.

    Attributes:
        feature: Feature name
        mdi_importance: Mean Decrease in Impurity importance
        permutation_importance: Permutation importance
        permutation_std: Standard deviation of permutation importance
        shap_mean: Mean absolute SHAP value (None if not computed)
        shap_std: Standard deviation of SHAP values (None if not computed)
        rank_mdi: Rank by MDI importance (1 = most important)
        rank_permutation: Rank by permutation importance
    """

    feature: str
    mdi_importance: float
    permutation_importance: float
    permutation_std: float
    shap_mean: float | None = None
    shap_std: float | None = None
    rank_mdi: int = 0
    rank_permutation: int = 0


@dataclass
class FeatureOutcomeResult:
    """Aggregated feature-outcome analysis results.

    Combines IC analysis, importance scoring, and drift detection
    for a set of features. This is the primary input to FeatureSelector.

    Attributes:
        features: List of feature names analyzed
        ic_results: IC analysis per feature (keyed by feature name)
        importance_results: Importance results per feature (keyed by feature name)
        drift_results: Optional drift detection results
    """

    features: list[str]
    ic_results: dict[str, FeatureICResults] = field(default_factory=dict)
    importance_results: dict[str, FeatureImportanceResults] = field(default_factory=dict)
    drift_results: DriftSummaryResult | None = None
