"""Unified drift analysis using multiple detection methods.

This module provides the main analyze_drift() function that combines
PSI, Wasserstein, and Domain Classifier methods for comprehensive
drift detection.

Consensus Logic:
    A feature is flagged as drifted if the fraction of methods detecting drift
    exceeds the consensus_threshold. For example, with threshold=0.5:
    - If 2/3 methods detect drift → flagged as drifted
    - If 1/3 methods detect drift → not flagged as drifted
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.drift.domain_classifier import (
    DomainClassifierResult,
    compute_domain_classifier_drift,
)
from ml4t.diagnostic.evaluation.drift.population_stability_index import (
    PSIResult,
    compute_psi,
)
from ml4t.diagnostic.evaluation.drift.wasserstein import (
    WassersteinResult,
    compute_wasserstein_distance,
)


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature across multiple methods.

    Attributes:
        feature: Feature name
        psi_result: PSI drift detection result (if method was run)
        wasserstein_result: Wasserstein drift detection result (if method was run)
        drifted: Consensus drift flag (based on multiple methods)
        n_methods_run: Number of methods that were run on this feature
        n_methods_detected: Number of methods that detected drift
        drift_probability: Fraction of methods that detected drift
        interpretation: Human-readable interpretation
    """

    feature: str
    psi_result: PSIResult | None = None
    wasserstein_result: WassersteinResult | None = None
    drifted: bool = False
    n_methods_run: int = 0
    n_methods_detected: int = 0
    drift_probability: float = 0.0
    interpretation: str = ""

    def summary(self) -> str:
        """Generate summary string for this feature's drift analysis."""
        lines = [f"Feature: {self.feature}"]
        lines.append(
            f"  Drifted: {self.drifted} ({self.n_methods_detected}/{self.n_methods_run} methods)"
        )
        lines.append(f"  Drift Probability: {self.drift_probability:.2%}")

        if self.psi_result is not None:
            lines.append(f"  PSI: {self.psi_result.psi:.4f} ({self.psi_result.alert_level})")

        if self.wasserstein_result is not None:
            drifted_str = "drifted" if self.wasserstein_result.drifted else "no drift"
            lines.append(f"  Wasserstein: {self.wasserstein_result.distance:.4f} ({drifted_str})")

        return "\n".join(lines)


@dataclass
class DriftSummaryResult:
    """Summary of multi-method drift analysis across features.

    This result aggregates drift detection across multiple methods (PSI,
    Wasserstein, Domain Classifier) to provide a comprehensive drift assessment.

    Attributes:
        feature_results: Per-feature drift results (PSI + Wasserstein)
        domain_classifier_result: Multivariate drift result (if domain classifier was run)
        n_features: Total number of features analyzed
        n_features_drifted: Number of features flagged as drifted
        drifted_features: List of feature names that drifted
        overall_drifted: Overall drift flag (True if any feature drifted or domain classifier detected drift)
        consensus_threshold: Minimum fraction of methods that must agree to flag drift
        methods_used: List of drift detection methods used
        univariate_methods: Methods run on individual features
        multivariate_methods: Methods run on all features jointly
        interpretation: Human-readable interpretation
        computation_time: Total time taken for all methods (seconds)
    """

    feature_results: list[FeatureDriftResult]
    domain_classifier_result: DomainClassifierResult | None = None
    n_features: int = 0
    n_features_drifted: int = 0
    drifted_features: list[str] = field(default_factory=list)
    overall_drifted: bool = False
    consensus_threshold: float = 0.5
    methods_used: list[str] = field(default_factory=list)
    univariate_methods: list[str] = field(default_factory=list)
    multivariate_methods: list[str] = field(default_factory=list)
    interpretation: str = ""
    computation_time: float = 0.0

    def summary(self) -> str:
        """Generate comprehensive summary of drift analysis."""
        lines = ["=" * 60]
        lines.append("Drift Analysis Summary")
        lines.append("=" * 60)
        lines.append(f"Methods Used: {', '.join(self.methods_used)}")
        lines.append(f"Consensus Threshold: {self.consensus_threshold:.0%}")
        lines.append(f"Total Features: {self.n_features}")
        lines.append(
            f"Drifted Features: {self.n_features_drifted} ({self.n_features_drifted / max(1, self.n_features):.0%})"
        )
        lines.append(f"Overall Drift Detected: {self.overall_drifted}")
        lines.append("")

        if self.drifted_features:
            lines.append("Drifted Features:")
            for feature in self.drifted_features:
                lines.append(f"  - {feature}")
            lines.append("")

        if self.domain_classifier_result is not None:
            lines.append("Multivariate Drift (Domain Classifier):")
            lines.append(f"  AUC: {self.domain_classifier_result.auc:.4f}")
            lines.append(f"  Drifted: {self.domain_classifier_result.drifted}")
            lines.append("")

        lines.append(f"Computation Time: {self.computation_time:.2f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert feature-level results to a DataFrame.

        Returns:
            Polars DataFrame with per-feature drift analysis results
        """
        data = []
        for result in self.feature_results:
            row = {
                "feature": result.feature,
                "drifted": result.drifted,
                "drift_probability": result.drift_probability,
                "n_methods_detected": result.n_methods_detected,
                "n_methods_run": result.n_methods_run,
            }

            if result.psi_result is not None:
                row["psi"] = result.psi_result.psi
                row["psi_alert"] = result.psi_result.alert_level

            if result.wasserstein_result is not None:
                row["wasserstein_distance"] = result.wasserstein_result.distance
                row["wasserstein_drifted"] = result.wasserstein_result.drifted
                if result.wasserstein_result.p_value is not None:
                    row["wasserstein_pvalue"] = result.wasserstein_result.p_value

            data.append(row)

        return pl.DataFrame(data)


def analyze_drift(
    reference: pd.DataFrame | pl.DataFrame,
    test: pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
    *,
    methods: list[str] | None = None,
    consensus_threshold: float = 0.5,
    # PSI parameters
    psi_config: dict[str, Any] | None = None,
    # Wasserstein parameters
    wasserstein_config: dict[str, Any] | None = None,
    # Domain classifier parameters
    domain_classifier_config: dict[str, Any] | None = None,
) -> DriftSummaryResult:
    """Comprehensive drift analysis using multiple detection methods.

    This function provides a unified interface for drift detection across multiple
    methods (PSI, Wasserstein, Domain Classifier). It runs univariate methods on
    each feature and optionally multivariate methods on all features jointly.

    **Univariate Methods** (run per feature):
        - PSI: Population Stability Index (binning-based)
        - Wasserstein: Earth Mover's Distance (metric-based)

    **Multivariate Methods** (run on all features):
        - Domain Classifier: ML-based drift detection with feature importance

    **Consensus Logic**:
        A feature is flagged as drifted if the fraction of methods detecting drift
        exceeds the consensus_threshold. For example, with threshold=0.5:
        - If 2/3 methods detect drift → flagged as drifted
        - If 1/3 methods detect drift → not flagged as drifted

    Args:
        reference: Reference distribution (e.g., training data)
            Can be pandas or polars DataFrame
        test: Test distribution (e.g., production data)
            Can be pandas or polars DataFrame
        features: List of feature names to analyze. If None, uses all numeric columns
        methods: List of methods to use. Options: ["psi", "wasserstein", "domain_classifier"]
            Default: ["psi", "wasserstein", "domain_classifier"]
        consensus_threshold: Minimum fraction of methods that must detect drift
            to flag a feature as drifted (default: 0.5)
        psi_config: Configuration dict for PSI. Keys:
            - n_bins: int (default: 10)
            - is_categorical: bool (default: False)
            - psi_threshold_yellow: float (default: 0.1)
            - psi_threshold_red: float (default: 0.2)
        wasserstein_config: Configuration dict for Wasserstein. Keys:
            - p: int (default: 1)
            - threshold_calibration: bool (default: True)
            - n_permutations: int (default: 1000)
            - alpha: float (default: 0.05)
        domain_classifier_config: Configuration dict for domain classifier. Keys:
            - model_type: str (default: "lightgbm")
            - n_estimators: int (default: 100)
            - max_depth: int (default: 5)
            - threshold: float (default: 0.6)
            - cv_folds: int (default: 5)

    Returns:
        DriftSummaryResult with per-feature results, multivariate results,
        and overall drift assessment

    Raises:
        ValueError: If inputs are invalid or methods list is empty

    Example:
        >>> import pandas as pd
        >>> from ml4t.diagnostic.evaluation.drift import analyze_drift
        >>>
        >>> # Create reference and test data
        >>> reference = pd.DataFrame({
        ...     'feature1': np.random.normal(0, 1, 1000),
        ...     'feature2': np.random.normal(0, 1, 1000)
        ... })
        >>> test = pd.DataFrame({
        ...     'feature1': np.random.normal(0.5, 1, 1000),  # Mean shifted
        ...     'feature2': np.random.normal(0, 1, 1000)      # No shift
        ... })
        >>>
        >>> # Run drift analysis
        >>> result = analyze_drift(reference, test)
        >>> print(result.summary())
        >>>
        >>> # Check which features drifted
        >>> print(f"Drifted features: {result.drifted_features}")
        >>>
        >>> # Get per-feature details
        >>> df = result.to_dataframe()
        >>> print(df)
    """
    start_time = time.time()

    # Input validation
    if reference is None or test is None:
        raise ValueError("reference and test must not be None")

    # Convert to pandas for easier processing
    reference_pd: pd.DataFrame
    test_pd: pd.DataFrame
    if isinstance(reference, pl.DataFrame):
        reference_pd = reference.to_pandas()
    else:
        reference_pd = reference
    if isinstance(test, pl.DataFrame):
        test_pd = test.to_pandas()
    else:
        test_pd = test

    # Determine features to analyze
    if features is None:
        # Use all numeric columns
        numeric_cols = reference_pd.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_cols
    else:
        # Validate features exist
        missing_in_ref = set(features) - set(reference_pd.columns)
        missing_in_test = set(features) - set(test_pd.columns)
        if missing_in_ref or missing_in_test:
            raise ValueError(
                f"Features not found - reference: {missing_in_ref}, test: {missing_in_test}"
            )

    if not features:
        raise ValueError("No features to analyze")

    # Determine methods to use
    if methods is None:
        methods = ["psi", "wasserstein", "domain_classifier"]

    valid_methods = ["psi", "wasserstein", "domain_classifier"]
    invalid_methods = set(methods) - set(valid_methods)
    if invalid_methods:
        raise ValueError(f"Invalid methods: {invalid_methods}. Valid: {valid_methods}")

    # Separate univariate and multivariate methods
    univariate_methods = [m for m in methods if m in ["psi", "wasserstein"]]
    multivariate_methods = [m for m in methods if m == "domain_classifier"]

    # Set default configs
    if psi_config is None:
        psi_config = {}
    if wasserstein_config is None:
        wasserstein_config = {}
    if domain_classifier_config is None:
        domain_classifier_config = {}

    # Run univariate methods on each feature
    feature_results = []
    for feature in features:
        # Explicitly convert to ndarray to handle ExtensionArray types
        ref_values = np.asarray(reference_pd[feature].values, dtype=np.float64)
        test_values = np.asarray(test_pd[feature].values, dtype=np.float64)

        psi_result = None
        wasserstein_result = None
        n_methods_run = 0
        n_methods_detected = 0

        # PSI
        if "psi" in methods:
            try:
                psi_result = compute_psi(ref_values, test_values, **psi_config)
                n_methods_run += 1
                if psi_result.alert_level in ["yellow", "red"]:
                    n_methods_detected += 1
            except Exception as e:
                # Log warning but continue
                print(f"Warning: PSI failed for feature {feature}: {e}")

        # Wasserstein
        if "wasserstein" in methods:
            try:
                wasserstein_result = compute_wasserstein_distance(
                    ref_values, test_values, **wasserstein_config
                )
                n_methods_run += 1
                if wasserstein_result.drifted:
                    n_methods_detected += 1
            except Exception as e:
                # Log warning but continue
                print(f"Warning: Wasserstein failed for feature {feature}: {e}")

        # Consensus drift flag
        drift_probability = n_methods_detected / max(1, n_methods_run)
        drifted = drift_probability >= consensus_threshold

        # Interpretation
        if drifted:
            interpretation = f"{n_methods_detected}/{n_methods_run} methods detected drift (probability: {drift_probability:.0%})"
        else:
            interpretation = (
                f"No consensus drift ({n_methods_detected}/{n_methods_run} methods, "
                f"threshold: {consensus_threshold:.0%})"
            )

        feature_results.append(
            FeatureDriftResult(
                feature=feature,
                psi_result=psi_result,
                wasserstein_result=wasserstein_result,
                drifted=drifted,
                n_methods_run=n_methods_run,
                n_methods_detected=n_methods_detected,
                drift_probability=drift_probability,
                interpretation=interpretation,
            )
        )

    # Run multivariate domain classifier if requested
    domain_classifier_result = None
    if "domain_classifier" in methods:
        try:
            domain_classifier_result = compute_domain_classifier_drift(
                reference[features], test[features], **domain_classifier_config
            )
        except Exception as e:
            # Log warning but continue
            print(f"Warning: Domain classifier failed: {e}")

    # Aggregate results
    n_features = len(features)
    n_features_drifted = sum(r.drifted for r in feature_results)
    drifted_features = [r.feature for r in feature_results if r.drifted]

    # Overall drift flag
    overall_drifted = n_features_drifted > 0
    if domain_classifier_result is not None and domain_classifier_result.drifted:
        overall_drifted = True

    # Interpretation
    if overall_drifted:
        interpretation = (
            f"Drift detected in {n_features_drifted}/{n_features} features "
            f"({n_features_drifted / max(1, n_features):.0%})"
        )
    else:
        interpretation = f"No drift detected across {n_features} features"

    computation_time = time.time() - start_time

    return DriftSummaryResult(
        feature_results=feature_results,
        domain_classifier_result=domain_classifier_result,
        n_features=n_features,
        n_features_drifted=n_features_drifted,
        drifted_features=drifted_features,
        overall_drifted=overall_drifted,
        consensus_threshold=consensus_threshold,
        methods_used=methods,
        univariate_methods=univariate_methods,
        multivariate_methods=multivariate_methods,
        interpretation=interpretation,
        computation_time=computation_time,
    )
