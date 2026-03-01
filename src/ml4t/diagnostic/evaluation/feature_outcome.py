"""Feature-outcome relationship analysis (Module C).

This module provides comprehensive analysis of how features relate to outcomes:
- **IC Analysis**: Information Coefficient for predictive power
- **Binary Classification**: Precision, recall, lift for signal quality
- **Threshold Optimization**: Find optimal thresholds for signals
- **ML Diagnostics**: Feature importance, SHAP, interactions
- **Drift Detection**: Monitor feature distribution stability

The FeatureOutcome class orchestrates all analyses into a unified workflow.

Example:
    >>> from ml4t.diagnostic.evaluation.feature_outcome import FeatureOutcome
    >>> from ml4t.diagnostic.config.feature_config import DiagnosticConfig
    >>>
    >>> # Basic usage
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, returns_df)
    >>> print(results.summary)
    >>>
    >>> # Custom configuration
    >>> config = DiagnosticConfig(
    ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21]),
    ...     ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True)
    ... )
    >>> analyzer = FeatureOutcome(config=config)
    >>> results = analyzer.run_analysis(features_df, returns_df, verbose=True)
    >>>
    >>> # Get recommendations
    >>> for rec in results.get_recommendations():
    ...     print(f"• {rec}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.config.feature_config import DiagnosticConfig
from ml4t.diagnostic.evaluation.drift import DriftSummaryResult, analyze_drift
from ml4t.diagnostic.evaluation.metrics.feature_outcome import analyze_feature_outcome_series
from ml4t.diagnostic.utils.dependencies import DEPS, warn_if_missing


@dataclass
class FeatureICResults:
    """IC analysis results for a single feature.

    Attributes:
        feature: Feature name
        ic_mean: Mean IC across time
        ic_std: Standard deviation of IC
        ic_ir: IC Information Ratio (mean/std)
        t_stat: T-statistic for IC
        p_value: P-value for IC significance
        ic_by_lag: IC values by forward horizon
        hac_adjusted: Whether HAC adjustment was applied
        n_observations: Number of observations used
    """

    feature: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    ic_by_lag: dict[int, float] = field(default_factory=dict)
    hac_adjusted: bool = False
    n_observations: int = 0


@dataclass
class FeatureImportanceResults:
    """ML feature importance results.

    Attributes:
        feature: Feature name
        mdi_importance: Mean Decrease in Impurity (tree-based)
        permutation_importance: Permutation-based importance
        permutation_std: Standard deviation of permutation importance
        shap_mean: Mean absolute SHAP value (if computed)
        shap_std: Standard deviation of SHAP values (if computed)
        rank_mdi: Rank by MDI importance
        rank_permutation: Rank by permutation importance
    """

    feature: str
    mdi_importance: float = 0.0
    permutation_importance: float = 0.0
    permutation_std: float = 0.0
    shap_mean: float | None = None
    shap_std: float | None = None
    rank_mdi: int = 0
    rank_permutation: int = 0


@dataclass
class FeatureOutcomeAnalysisResult:
    """Comprehensive feature-outcome analysis results.

    This aggregates all Module C analyses into a single result object.

    Attributes:
        features: List of features analyzed
        ic_results: IC analysis per feature
        importance_results: ML importance per feature
        drift_results: Drift detection results (if enabled)
        interaction_matrix: H-statistic interaction matrix (if computed)
        summary: High-level summary DataFrame
        recommendations: Actionable recommendations
        config: Configuration used
        metadata: Analysis metadata (runtime, samples, etc.)
        errors: Dict of features that failed analysis
    """

    features: list[str]
    ic_results: dict[str, FeatureICResults] = field(default_factory=dict)
    importance_results: dict[str, FeatureImportanceResults] = field(default_factory=dict)
    drift_results: DriftSummaryResult | None = None
    interaction_matrix: pd.DataFrame | None = None
    summary: pd.DataFrame | None = None
    recommendations: list[str] = field(default_factory=list)
    config: DiagnosticConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary as DataFrame.

        Returns:
            DataFrame with one row per feature, columns for all metrics
        """
        if self.summary is not None:
            return self.summary

        # Build summary from individual results
        rows = []
        for feature in self.features:
            row: dict[str, str | float | bool | int] = {"feature": feature}

            # IC metrics (with defaults for missing)
            if feature in self.ic_results:
                ic = self.ic_results[feature]
                row.update(
                    {
                        "ic_mean": ic.ic_mean,
                        "ic_std": ic.ic_std,
                        "ic_ir": ic.ic_ir,
                        "ic_pvalue": ic.p_value,
                        "ic_significant": ic.p_value < 0.05,
                    }
                )
            else:
                # Add NaN placeholders if IC not computed
                row.update(
                    {
                        "ic_mean": np.nan,
                        "ic_std": np.nan,
                        "ic_ir": np.nan,
                        "ic_pvalue": np.nan,
                        "ic_significant": False,
                    }
                )

            # Importance metrics (with defaults for missing)
            if feature in self.importance_results:
                imp = self.importance_results[feature]
                row.update(
                    {
                        "mdi_importance": imp.mdi_importance,
                        "permutation_importance": imp.permutation_importance,
                        "rank_mdi": imp.rank_mdi,
                        "rank_permutation": imp.rank_permutation,
                    }
                )
            else:
                row.update(
                    {
                        "mdi_importance": np.nan,
                        "permutation_importance": np.nan,
                        "rank_mdi": np.nan,
                        "rank_permutation": np.nan,
                    }
                )

            # Drift metrics
            if self.drift_results is not None:
                drift_df = self.drift_results.to_dataframe()
                # Convert to pandas if polars
                if isinstance(drift_df, pl.DataFrame):
                    drift_df = drift_df.to_pandas()

                feature_drift = drift_df[drift_df["feature"] == feature]
                if len(feature_drift) > 0:
                    row["drifted"] = feature_drift["drifted"].iloc[0]
                    if "psi" in feature_drift.columns:
                        row["psi"] = feature_drift["psi"].iloc[0]
                else:
                    row["drifted"] = False
            else:
                row["drifted"] = False

            # Error status
            row["error"] = feature in self.errors

            rows.append(row)

        return pd.DataFrame(rows)

    def get_top_features(
        self, n: int = 10, by: str = "ic_ir", ascending: bool = False
    ) -> list[str]:
        """Get top N features by specified metric.

        Args:
            n: Number of features to return
            by: Metric to sort by ('ic_ir', 'ic_mean', 'mdi_importance', etc.)
            ascending: Sort in ascending order (default: descending)

        Returns:
            List of top feature names

        Example:
            >>> # Get 5 features with highest IC IR
            >>> top_features = results.get_top_features(n=5, by='ic_ir')
        """
        df = self.to_dataframe()

        if by not in df.columns:
            available = [c for c in df.columns if c != "feature"]
            raise ValueError(f"Metric '{by}' not found. Available: {available}")

        # Remove features with errors or NaN values
        df = df[~df["error"]]
        df = df.dropna(subset=[by])

        # Sort and return top N
        df = df.sort_values(by=by, ascending=ascending)
        return df.head(n)["feature"].tolist()

    def get_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation strings

        Example:
            >>> for rec in results.get_recommendations():
            ...     print(f"• {rec}")
        """
        if self.recommendations:
            return self.recommendations

        # Generate recommendations from results
        recommendations = []
        df = self.to_dataframe()

        # Strong signals (high IC IR, no drift)
        strong = df[(df["ic_ir"] > 2.0) & (~df.get("drifted", False))]
        if len(strong) > 0:
            for _, row in strong.iterrows():
                recommendations.append(
                    f"{row['feature']}: Strong predictive power (IC IR={row['ic_ir']:.2f}), stable distribution"
                )

        # Weak signals (low IC)
        weak = df[df["ic_ir"].abs() < 0.5]
        if len(weak) > 0:
            features = ", ".join(weak["feature"].tolist()[:5])
            more = f" (+{len(weak) - 5} more)" if len(weak) > 5 else ""
            recommendations.append(f"Consider removing weak signals: {features}{more}")

        # Drifted features
        if "drifted" in df.columns:
            drifted = df[df["drifted"] == True]  # noqa: E712
            if len(drifted) > 0:
                for _, row in drifted.iterrows():
                    recommendations.append(
                        f"{row['feature']}: Distribution drift detected - consider retraining or investigation"
                    )

        # Features with errors
        if len(self.errors) > 0:
            error_features = ", ".join(list(self.errors.keys())[:3])
            more = f" (+{len(self.errors) - 3} more)" if len(self.errors) > 3 else ""
            recommendations.append(f"Analysis failed for: {error_features}{more}")

        return recommendations


# Backward-compatible alias: evaluation-layer domain result.
FeatureOutcomeResult = FeatureOutcomeAnalysisResult


class FeatureOutcome:
    """Main orchestration class for feature-outcome analysis (Module C).

    Coordinates comprehensive analysis of feature-outcome relationships:
    - IC analysis (Information Coefficient for predictive power)
    - Binary classification metrics (precision, recall, lift)
    - Threshold optimization
    - ML feature importance (MDI, permutation, SHAP)
    - Feature interactions (H-statistic)
    - Drift detection

    This class provides a unified interface for all Module C analyses,
    handling configuration, execution, and result aggregation.

    Examples:
        >>> # Basic usage with defaults
        >>> analyzer = FeatureOutcome()
        >>> results = analyzer.run_analysis(features_df, returns_df)
        >>> print(results.summary)
        >>>
        >>> # Custom configuration
        >>> config = DiagnosticConfig(
        ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21]),
        ...     ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True)
        ... )
        >>> analyzer = FeatureOutcome(config=config)
        >>> results = analyzer.run_analysis(features_df, returns_df, verbose=True)
        >>>
        >>> # Select specific features
        >>> results = analyzer.run_analysis(
        ...     features_df,
        ...     returns_df,
        ...     feature_names=['momentum', 'volume', 'volatility']
        ... )
        >>>
        >>> # Get actionable insights
        >>> top_features = results.get_top_features(n=10, by='ic_ir')
        >>> recommendations = results.get_recommendations()
    """

    def __init__(self, config: DiagnosticConfig | None = None):
        """Initialize FeatureOutcome analyzer.

        Args:
            config: Module C configuration. Uses defaults if None.

        Example:
            >>> # Default configuration
            >>> analyzer = FeatureOutcome()
            >>>
            >>> # Custom configuration
            >>> config = DiagnosticConfig(
            ...     ic=ICConfig(hac_adjustment=True),
            ...     ml_diagnostics=MLDiagnosticsConfig(drift_detection=True)
            ... )
            >>> analyzer = FeatureOutcome(config=config)
        """
        self.config = config or DiagnosticConfig()

    def run_analysis(
        self,
        features: pd.DataFrame | pl.DataFrame,
        outcomes: pd.DataFrame | pl.DataFrame | pd.Series | np.ndarray,
        feature_names: list[str] | None = None,
        _date_col: str | None = None,
        verbose: bool = False,
    ) -> FeatureOutcomeResult:
        """Run comprehensive feature-outcome analysis.

        Executes all enabled analyses in Module C configuration:
        1. IC analysis (if ic.enabled)
        2. ML feature importance (if ml_diagnostics.enabled)
        3. Feature interactions (if ml_diagnostics.enabled)
        4. Drift detection (if ml_diagnostics.drift_detection)

        Args:
            features: Feature DataFrame (T x N) with date index or date column
            outcomes: Outcome/returns DataFrame, Series, or array (T x 1 or T)
            feature_names: Specific features to analyze (None = all numeric)
            date_col: Date column name if not in index
            verbose: Print progress messages

        Returns:
            FeatureOutcomeResult with all analyses

        Raises:
            ValueError: If inputs are invalid or incompatible

        Example:
            >>> # Basic usage
            >>> results = analyzer.run_analysis(features_df, returns_df)
            >>>
            >>> # With progress tracking
            >>> results = analyzer.run_analysis(
            ...     features_df, returns_df, verbose=True
            ... )
            >>> # Output:
            >>> # Analyzing 10 features...
            >>> # [1/10] feature1: IC=0.15, importance=0.25
            >>> # [2/10] feature2: IC=0.08, importance=0.12
            >>> # ...
            >>> # Analysis complete in 12.3s
        """
        start_time = time.time()

        # ===================================================================
        # 0. Configuration and Dependency Validation
        # ===================================================================
        if verbose:
            print("Validating configuration and dependencies...")

        # Check dependencies based on configuration
        missing_deps = []
        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.feature_importance:
            if not DEPS.check("lightgbm"):
                missing_deps.append("lightgbm")
                if verbose:
                    print("  ⚠️  LightGBM not available - feature importance will be skipped")
                    print(f"      Install with: {DEPS.lightgbm.install_cmd}")

        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.shap_analysis:
            if not DEPS.check("shap"):
                missing_deps.append("shap")
                if verbose:
                    print("  ⚠️  SHAP not available - SHAP analysis will be skipped")
                    print(f"      Install with: {DEPS.shap.install_cmd}")

        # Log what features are available
        if verbose and not missing_deps:
            print("  ✓ All required dependencies available")

        # ===================================================================
        # 1. Input Validation and Preprocessing
        # ===================================================================
        if verbose:
            print("Validating inputs...")

        # Convert to pandas for consistency
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
        if isinstance(outcomes, pl.DataFrame):
            outcomes = outcomes.to_pandas()
        if isinstance(outcomes, pl.Series):
            outcomes = outcomes.to_pandas()

        # Handle outcomes format
        if isinstance(outcomes, pd.Series):
            outcomes_series = outcomes
        elif isinstance(outcomes, np.ndarray):
            if outcomes.ndim == 1:
                outcomes_series = pd.Series(outcomes, index=features.index)
            else:
                # Take first column
                outcomes_series = pd.Series(outcomes[:, 0], index=features.index)
        elif isinstance(outcomes, pd.DataFrame):
            # Take first column
            outcomes_series = outcomes.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported outcomes type: {type(outcomes)}")

        # Validate alignment
        if len(features) != len(outcomes_series):
            raise ValueError(
                f"Features ({len(features)}) and outcomes ({len(outcomes_series)}) must have same length"
            )

        # ===================================================================
        # 2. Determine Features to Analyze
        # ===================================================================
        if feature_names is None:
            # Use all numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_names = numeric_cols
        else:
            # Validate specified features exist
            missing = set(feature_names) - set(features.columns)
            if missing:
                raise ValueError(f"Features not found in DataFrame: {missing}")

        if not feature_names:
            raise ValueError("No features to analyze")

        if verbose:
            print(f"Analyzing {len(feature_names)} features...")

        # ===================================================================
        # 3. Initialize Results Storage
        # ===================================================================
        ic_results = {}
        importance_results = {}
        errors = {}

        # ===================================================================
        # 4. Run IC Analysis (if enabled)
        # ===================================================================
        if self.config.ic.enabled:
            if verbose:
                print("Running IC analysis...")

            ic_method = (
                self.config.ic.method.value
                if hasattr(self.config.ic.method, "value")
                else str(self.config.ic.method)
            )
            for i, feature in enumerate(feature_names, 1):
                try:
                    feature_data = features[feature].to_numpy()
                    outcome_data = outcomes_series.to_numpy()

                    if np.sum(~(np.isnan(feature_data) | np.isnan(outcome_data))) < 10:
                        errors[feature] = "Insufficient non-NaN samples"
                        continue

                    # Canonical metrics-layer series analysis (compatibility adapter path).
                    series_result = analyze_feature_outcome_series(
                        predictions=feature_data,
                        outcomes=outcome_data,
                        method=ic_method,
                        include_monotonicity=False,
                    )

                    ic_results[feature] = FeatureICResults(
                        feature=feature,
                        ic_mean=series_result["ic_mean"],
                        ic_std=series_result["ic_std"],
                        ic_ir=series_result["ic_ir"],
                        p_value=series_result["p_value"],
                        n_observations=series_result["n_observations"],
                    )

                    if verbose and i % max(1, len(feature_names) // 10) == 0:
                        print(
                            f"  [{i}/{len(feature_names)}] {feature}: IC={series_result['ic_mean']:.3f}"
                        )

                except Exception as e:
                    errors[feature] = str(e)
                    if verbose:
                        print(f"  [{i}/{len(feature_names)}] {feature}: ERROR - {e}")

        # ===================================================================
        # 5. Run ML Diagnostics (if enabled)
        # ===================================================================
        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.feature_importance:
            if verbose:
                print("Running feature importance analysis...")

            try:
                # Check if LightGBM is available
                if warn_if_missing("lightgbm", "feature importance", "skipping analysis"):
                    import lightgbm as lgb

                    # Prepare data
                    X = features[feature_names].values
                    y = outcomes_series.to_numpy()

                    # Remove NaN rows
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    X_clean = X[mask]
                    y_clean = y[mask]

                    if len(X_clean) >= 100:
                        # Train simple model for importance
                        model = lgb.LGBMRegressor(
                            n_estimators=100, max_depth=3, random_state=42, verbose=-1
                        )
                        model.fit(X_clean, y_clean)

                        # Get MDI importance
                        mdi_importances = model.feature_importances_

                        # Rank features
                        ranks = np.argsort(mdi_importances)[::-1]

                        for idx, feature in enumerate(feature_names):
                            if feature not in errors:
                                rank = int(np.where(ranks == idx)[0][0]) + 1
                                importance_results[feature] = FeatureImportanceResults(
                                    feature=feature,
                                    mdi_importance=float(mdi_importances[idx]),
                                    rank_mdi=rank,
                                )

                        if verbose:
                            top_feature = feature_names[ranks[0]]
                            print(
                                f"  Top feature by MDI: {top_feature} (importance={mdi_importances[ranks[0]]:.3f})"
                            )
                    else:
                        if verbose:
                            print(
                                f"  Insufficient clean samples for importance analysis ({len(X_clean)}/100)"
                            )
                else:
                    if verbose:
                        print("  Feature importance skipped (LightGBM not available)")

            except Exception as e:
                if verbose:
                    print(f"  Feature importance failed: {e}")

        # ===================================================================
        # 6. Run Drift Detection (if enabled)
        # ===================================================================
        drift_results = None
        if self.config.ml_diagnostics.drift_detection:
            if verbose:
                print("Running drift detection...")

            try:
                # Split data into reference (first half) and test (second half)
                split_idx = len(features) // 2
                reference = features[feature_names].iloc[:split_idx]
                test = features[feature_names].iloc[split_idx:]

                drift_results = analyze_drift(
                    reference,
                    test,
                    features=feature_names,
                    methods=["psi", "wasserstein"],  # Fast methods only
                )

                if verbose:
                    n_drifted = drift_results.n_features_drifted
                    print(f"  Drift detected in {n_drifted}/{len(feature_names)} features")

            except Exception as e:
                if verbose:
                    print(f"  Drift detection failed: {e}")

        # ===================================================================
        # 7. Build Summary and Generate Recommendations
        # ===================================================================
        result = FeatureOutcomeResult(
            features=feature_names,
            ic_results=ic_results,
            importance_results=importance_results,
            drift_results=drift_results,
            config=self.config,
            errors=errors,
            metadata={
                "n_features": len(feature_names),
                "n_observations": len(features),
                "n_errors": len(errors),
                "computation_time": time.time() - start_time,
                "ic_enabled": self.config.ic.enabled,
                "ml_diagnostics_enabled": self.config.ml_diagnostics.enabled,
                "drift_detection_enabled": self.config.ml_diagnostics.drift_detection,
            },
        )

        # Build summary DataFrame
        result.summary = result.to_dataframe()

        # Generate recommendations
        result.recommendations = result.get_recommendations()

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nAnalysis complete in {elapsed:.1f}s")
            print(f"  Features analyzed: {len(feature_names)}")
            print(f"  Errors: {len(errors)}")
            if result.recommendations:
                print(f"  Recommendations: {len(result.recommendations)}")

        return result


# Re-export for convenience
__all__ = [
    "FeatureICResults",
    "FeatureImportanceResults",
    "FeatureOutcomeResult",
    "FeatureOutcome",
]
