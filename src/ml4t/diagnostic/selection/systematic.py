"""Systematic feature selection for ML pipelines.

Provides a comprehensive feature selection workflow combining multiple
filtering criteria: IC analysis, importance scoring, correlation filtering,
and drift detection.

Example - Basic Usage:
    >>> from ml4t.diagnostic.selection import FeatureSelector
    >>>
    >>> selector = FeatureSelector(
    ...     outcome_results=results,
    ...     correlation_matrix=corr_matrix,
    ... )
    >>> selector.filter_by_ic(threshold=0.02)
    >>> selector.filter_by_correlation(threshold=0.8)
    >>> selected = selector.get_selected_features()

Example - Pipeline:
    >>> selector = FeatureSelector(results, corr_matrix)
    >>> selector.run_pipeline([
    ...     ("ic", {"threshold": 0.02, "min_periods": 20}),
    ...     ("correlation", {"threshold": 0.8}),
    ...     ("importance", {"threshold": 0.01, "method": "mdi"}),
    ... ])
    >>> report = selector.get_selection_report()
    >>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from ml4t.diagnostic.selection.types import FeatureOutcomeResult


@dataclass
class SelectionStep:
    """Record of a single selection step.

    Attributes:
        step_name: Name of the filter applied
        parameters: Parameters used for the filter
        features_before: Number of features before filter
        features_after: Number of features after filter
        features_removed: List of features removed in this step
        features_kept: List of features kept after this step
        reasoning: Explanation of why features were removed
    """

    step_name: str
    parameters: dict[str, Any]
    features_before: int
    features_after: int
    features_removed: list[str]
    features_kept: list[str]
    reasoning: str

    def summary(self) -> str:
        """Generate summary of this selection step."""
        pct_removed = 100 * len(self.features_removed) / max(1, self.features_before)
        return (
            f"{self.step_name}: {self.features_before} → {self.features_after} "
            f"({len(self.features_removed)} removed, {pct_removed:.1f}%)\n"
            f"  Parameters: {self.parameters}\n"
            f"  Reasoning: {self.reasoning}"
        )


@dataclass
class SelectionReport:
    """Complete feature selection report.

    Attributes:
        initial_features: Features at start of selection
        final_features: Features after all filters
        steps: List of selection steps applied
        total_removed: Total number of features removed
        removal_rate: Percentage of features removed
    """

    initial_features: list[str]
    final_features: list[str]
    steps: list[SelectionStep] = field(default_factory=list)
    total_removed: int = field(init=False)
    removal_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.total_removed = len(self.initial_features) - len(self.final_features)
        self.removal_rate = 100 * self.total_removed / max(1, len(self.initial_features))

    def summary(self) -> str:
        """Generate comprehensive selection report."""
        lines = [
            "=" * 70,
            "Feature Selection Report",
            "=" * 70,
            f"Initial Features: {len(self.initial_features)}",
            f"Final Features: {len(self.final_features)}",
            f"Removed: {self.total_removed} ({self.removal_rate:.1f}%)",
            "",
            "Selection Pipeline:",
            "-" * 70,
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(f"\nStep {i}: {step.summary()}")

        lines.extend(
            [
                "",
                "-" * 70,
                "Final Selected Features:",
                "-" * 70,
            ]
        )
        for feature in sorted(self.final_features):
            lines.append(f"  - {feature}")

        lines.append("=" * 70)

        return "\n".join(lines)


class FeatureSelector:
    """Systematic feature selection with multiple filtering criteria.

    Combines IC analysis, importance scoring, correlation filtering, and
    drift detection to select the most promising features for ML models.

    Parameters
    ----------
    outcome_results : FeatureOutcomeResult
        Results from feature-outcome analysis (IC, importance, drift).
    correlation_matrix : pl.DataFrame, optional
        Feature correlation matrix.
    initial_features : list[str], optional
        Initial set of features to select from.
        If None, uses all features from outcome_results.

    Attributes
    ----------
    selected_features : set[str]
        Current set of selected features (updated by filters)
    removed_features : set[str]
        Features removed by filters
    selection_steps : list[SelectionStep]
        History of selection steps applied
    """

    def __init__(
        self,
        outcome_results: FeatureOutcomeResult,
        correlation_matrix: pl.DataFrame | None = None,
        initial_features: list[str] | None = None,
    ):
        self.outcome_results = outcome_results
        self.correlation_matrix = correlation_matrix

        if initial_features is not None:
            self.initial_features = set(initial_features)
        else:
            self.initial_features = set(outcome_results.features)

        self.selected_features = self.initial_features.copy()
        self.removed_features: set[str] = set()
        self.selection_steps: list[SelectionStep] = []

    def filter_by_ic(
        self,
        threshold: float,
        min_periods: int = 1,
        lag: int | None = None,
    ) -> FeatureSelector:
        """Filter features by Information Coefficient.

        Keeps features with |IC| > threshold.

        Parameters
        ----------
        threshold : float
            Minimum absolute IC value to keep a feature.
        min_periods : int, default 1
            Minimum number of observations required.
        lag : int | None, default None
            Specific forward lag to use. If None, uses mean IC.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining.
        """
        features_before = len(self.selected_features)
        features_to_remove = []

        for feature in self.selected_features:
            if feature not in self.outcome_results.ic_results:
                continue

            ic_result = self.outcome_results.ic_results[feature]

            if ic_result.n_observations < min_periods:
                features_to_remove.append(feature)
                continue

            if lag is not None:
                if lag not in ic_result.ic_by_lag:
                    features_to_remove.append(feature)
                    continue
                ic_value = abs(ic_result.ic_by_lag[lag])
            else:
                ic_value = abs(ic_result.ic_mean)

            if ic_value < threshold:
                features_to_remove.append(feature)

        self.selected_features -= set(features_to_remove)
        self.removed_features |= set(features_to_remove)

        step = SelectionStep(
            step_name="IC Filtering",
            parameters={"threshold": threshold, "min_periods": min_periods, "lag": lag},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=f"Removed features with |IC| < {threshold}",
        )
        self.selection_steps.append(step)

        return self

    def filter_by_importance(
        self,
        threshold: float,
        method: Literal["mdi", "permutation", "shap"] = "mdi",
        top_k: int | None = None,
    ) -> FeatureSelector:
        """Filter features by ML importance scores.

        Parameters
        ----------
        threshold : float
            Minimum importance value to keep a feature.
        method : {"mdi", "permutation", "shap"}, default "mdi"
            Importance method to use.
        top_k : int | None, default None
            If provided, keeps only the top K most important features.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining.
        """
        features_before = len(self.selected_features)

        feature_importance = []
        for feature in self.selected_features:
            if feature not in self.outcome_results.importance_results:
                continue

            imp_result = self.outcome_results.importance_results[feature]

            if method == "mdi":
                importance = imp_result.mdi_importance
            elif method == "permutation":
                importance = imp_result.permutation_importance
            elif method == "shap":
                if imp_result.shap_mean is None:
                    continue
                importance = imp_result.shap_mean
            else:
                raise ValueError(
                    f"Unknown importance method: {method}. Choose from 'mdi', 'permutation', 'shap'"
                )

            feature_importance.append((feature, importance))

        feature_importance.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            features_to_keep = [f for f, _ in feature_importance[:top_k]]
            reasoning = f"Kept top {top_k} features by {method} importance"
        else:
            features_to_keep = [f for f, imp in feature_importance if imp >= threshold]
            reasoning = f"Removed features with {method} importance < {threshold}"

        features_to_remove = [f for f in self.selected_features if f not in features_to_keep]
        self.selected_features = set(features_to_keep)
        self.removed_features |= set(features_to_remove)

        step = SelectionStep(
            step_name=f"Importance Filtering ({method.upper()})",
            parameters={"threshold": threshold, "method": method, "top_k": top_k},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=reasoning,
        )
        self.selection_steps.append(step)

        return self

    def filter_by_correlation(
        self,
        threshold: float,
        keep_strategy: Literal["higher_ic", "higher_importance", "first"] = "higher_ic",
    ) -> FeatureSelector:
        """Remove highly correlated features to reduce redundancy.

        When two features have correlation > threshold, keeps one based on
        the keep_strategy.

        Parameters
        ----------
        threshold : float
            Maximum absolute correlation allowed between features.
        keep_strategy : {"higher_ic", "higher_importance", "first"}, default "higher_ic"
            Strategy for choosing which feature to keep.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If correlation_matrix was not provided.
        """
        if self.correlation_matrix is None:
            raise ValueError(
                "Correlation matrix required for correlation filtering. "
                "Provide correlation_matrix during FeatureSelector initialization."
            )

        features_before = len(self.selected_features)
        features_to_remove: set[str] = set()

        # Build correlation lookup dict from Polars DataFrame
        corr_matrix = self.correlation_matrix
        if "feature" in corr_matrix.columns:
            feature_names = corr_matrix["feature"].to_list()
            value_columns = [c for c in corr_matrix.columns if c != "feature"]
        else:
            feature_names = value_columns = list(corr_matrix.columns)

        # Build {(feat1, feat2): correlation} lookup for O(1) access
        corr_lookup: dict[tuple[str, str], float] = {}
        feature_set = set(feature_names)
        for row_idx, row_name in enumerate(feature_names):
            row_data = corr_matrix.row(row_idx)
            col_offset = 1 if "feature" in corr_matrix.columns else 0
            for col_idx, col_name in enumerate(value_columns):
                corr_lookup[(row_name, col_name)] = row_data[col_idx + col_offset]

        selected_list = sorted(self.selected_features)
        selected_list = [f for f in selected_list if f in feature_set]

        if len(selected_list) < 2:
            step = SelectionStep(
                step_name="Correlation Filtering",
                parameters={"threshold": threshold, "keep_strategy": keep_strategy},
                features_before=features_before,
                features_after=features_before,
                features_removed=[],
                features_kept=list(self.selected_features),
                reasoning="Insufficient features for correlation filtering",
            )
            self.selection_steps.append(step)
            return self

        for i, feat1 in enumerate(selected_list):
            if feat1 in features_to_remove:
                continue

            for feat2 in selected_list[i + 1 :]:
                if feat2 in features_to_remove:
                    continue

                corr_value = abs(corr_lookup[(feat1, feat2)])

                if corr_value > threshold:
                    if keep_strategy == "higher_ic":
                        ic_results = self.outcome_results.ic_results
                        ic1 = abs(ic_results[feat1].ic_mean) if feat1 in ic_results else 0.0
                        ic2 = abs(ic_results[feat2].ic_mean) if feat2 in ic_results else 0.0
                        to_remove = feat2 if ic1 > ic2 else feat1

                    elif keep_strategy == "higher_importance":
                        imp_results = self.outcome_results.importance_results
                        imp1 = imp_results[feat1].mdi_importance if feat1 in imp_results else 0.0
                        imp2 = imp_results[feat2].mdi_importance if feat2 in imp_results else 0.0
                        to_remove = feat2 if imp1 > imp2 else feat1

                    else:  # "first"
                        to_remove = feat2

                    features_to_remove.add(to_remove)

        self.selected_features -= features_to_remove
        self.removed_features |= features_to_remove

        step = SelectionStep(
            step_name="Correlation Filtering",
            parameters={"threshold": threshold, "keep_strategy": keep_strategy},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=list(features_to_remove),
            features_kept=list(self.selected_features),
            reasoning=(
                f"Removed features with correlation > {threshold} using {keep_strategy} strategy"
            ),
        )
        self.selection_steps.append(step)

        return self

    def filter_by_drift(
        self,
        threshold: float = 0.2,
        method: Literal["psi", "consensus"] = "psi",
    ) -> FeatureSelector:
        """Remove features with unstable distributions (drift).

        Parameters
        ----------
        threshold : float, default 0.2
            Drift threshold. For PSI: >= 0.2 indicates significant drift.
            For consensus: drift_probability >= threshold.
        method : {"psi", "consensus"}, default "psi"
            Drift detection method.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If drift_results not available in outcome_results.
        """
        if self.outcome_results.drift_results is None:
            raise ValueError(
                "Drift results not available. Run outcome analysis with drift_detection=True."
            )

        features_before = len(self.selected_features)
        features_to_remove = []

        drift_results = self.outcome_results.drift_results

        for feature_result in drift_results.feature_results:
            feature = feature_result.feature

            if feature not in self.selected_features:
                continue

            if method == "psi":
                if (
                    feature_result.psi_result is not None
                    and feature_result.psi_result.alert_level == "red"
                ):
                    features_to_remove.append(feature)

            elif method == "consensus":
                if feature_result.drift_probability >= threshold:
                    features_to_remove.append(feature)

            else:
                raise ValueError(f"Unknown drift method: {method}. Choose from 'psi', 'consensus'")

        self.selected_features -= set(features_to_remove)
        self.removed_features |= set(features_to_remove)

        step = SelectionStep(
            step_name="Drift Filtering",
            parameters={"threshold": threshold, "method": method},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=f"Removed features with {method} drift >= {threshold}",
        )
        self.selection_steps.append(step)

        return self

    def run_pipeline(
        self,
        steps: list[tuple[str, dict[str, Any]]],
    ) -> FeatureSelector:
        """Execute multiple selection filters in sequence.

        Parameters
        ----------
        steps : list[tuple[str, dict]]
            List of (filter_name, parameters) tuples.
            Valid filter names: "ic", "importance", "correlation", "drift".

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining.
        """
        for filter_name, params in steps:
            if filter_name == "ic":
                self.filter_by_ic(**params)
            elif filter_name == "importance":
                self.filter_by_importance(**params)
            elif filter_name == "correlation":
                self.filter_by_correlation(**params)
            elif filter_name == "drift":
                self.filter_by_drift(**params)
            else:
                raise ValueError(
                    f"Unknown filter: {filter_name}. "
                    "Valid filters: ic, importance, correlation, drift"
                )

        return self

    def get_selected_features(self) -> list[str]:
        """Get current list of selected features (sorted)."""
        return sorted(self.selected_features)

    def get_removed_features(self) -> list[str]:
        """Get list of features that were removed (sorted)."""
        return sorted(self.removed_features)

    def get_selection_report(self) -> SelectionReport:
        """Generate comprehensive selection report."""
        return SelectionReport(
            initial_features=sorted(self.initial_features),
            final_features=self.get_selected_features(),
            steps=self.selection_steps,
        )

    def reset(self) -> FeatureSelector:
        """Reset selector to initial feature set."""
        self.selected_features = self.initial_features.copy()
        self.removed_features = set()
        self.selection_steps = []
        return self
