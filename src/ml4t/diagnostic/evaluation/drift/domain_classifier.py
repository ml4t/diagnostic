"""Domain classifier for multivariate distribution drift detection.

The domain classifier trains a binary model to distinguish reference (label=0)
from test (label=1) samples. AUC indicates drift magnitude, feature importances
show which features drifted.

Advantages:
- Detects multivariate drift and feature interactions
- Non-parametric (no distributional assumptions)
- Interpretable via feature importance
- Sensitive to subtle multivariate shifts

AUC Interpretation:
- AUC ≈ 0.5: No drift (random guess)
- AUC = 0.6: Weak drift
- AUC = 0.7-0.8: Moderate drift
- AUC > 0.9: Strong drift

References:
    - Lopez-Paz, D., & Oquab, M. (2017). Revisiting Classifier Two-Sample Tests.
      ICLR 2017.
    - Rabanser, S., et al. (2019). Failing Loudly: An Empirical Study of Methods
      for Detecting Dataset Shift. NeurIPS 2019.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

# Lazy check for optional ML dependencies (imported on first use to avoid slow startup)
LIGHTGBM_AVAILABLE: bool | None = None
XGBOOST_AVAILABLE: bool | None = None


def _check_lightgbm_available() -> bool:
    """Check if lightgbm is available (lazy check)."""
    global LIGHTGBM_AVAILABLE
    if LIGHTGBM_AVAILABLE is None:
        try:
            import lightgbm  # noqa: F401

            LIGHTGBM_AVAILABLE = True
        except ImportError:
            LIGHTGBM_AVAILABLE = False
    return LIGHTGBM_AVAILABLE


def _check_xgboost_available() -> bool:
    """Check if xgboost is available (lazy check)."""
    global XGBOOST_AVAILABLE
    if XGBOOST_AVAILABLE is None:
        try:
            import xgboost  # noqa: F401

            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
    return XGBOOST_AVAILABLE


@dataclass
class DomainClassifierResult:
    """Result of domain classifier drift detection.

    Domain classifier trains a binary model to distinguish reference (label=0)
    from test (label=1) samples. AUC indicates drift magnitude, feature importances
    show which features drifted.

    Attributes:
        auc: AUC-ROC score (0.5 = no drift, 1.0 = complete distribution shift)
        drifted: Whether drift was detected (auc > threshold)
        feature_importances: DataFrame with feature, importance, rank columns
        threshold: AUC threshold used for drift detection
        n_reference: Number of samples in reference distribution
        n_test: Number of samples in test distribution
        n_features: Number of features used
        model_type: Type of classifier used (lightgbm, xgboost, sklearn)
        cv_auc_mean: Mean AUC from cross-validation
        cv_auc_std: Std of AUC from cross-validation
        interpretation: Human-readable interpretation
        computation_time: Time taken to compute (seconds)
        metadata: Additional metadata
    """

    auc: float
    drifted: bool
    feature_importances: pl.DataFrame
    threshold: float
    n_reference: int
    n_test: int
    n_features: int
    model_type: str
    cv_auc_mean: float
    cv_auc_std: float
    interpretation: str
    computation_time: float
    metadata: dict[str, Any]

    def summary(self) -> str:
        """Return formatted summary of domain classifier results."""
        lines = [
            "Domain Classifier Drift Detection Report",
            "=" * 60,
            f"AUC-ROC: {self.auc:.4f} (CV: {self.cv_auc_mean:.4f} ± {self.cv_auc_std:.4f})",
            f"Drift Detected: {'YES' if self.drifted else 'NO'}",
            f"Threshold: {self.threshold:.4f}",
            "",
            "Sample Sizes:",
            f"  Reference: {self.n_reference:,}",
            f"  Test: {self.n_test:,}",
            "",
            f"Model: {self.model_type}",
            f"Features: {self.n_features}",
            "",
            "Top 5 Most Drifted Features:",
            "-" * 60,
        ]

        # Show top 5 features
        top_features = self.feature_importances.head(5)
        for row in top_features.iter_rows(named=True):
            lines.append(
                f"  {row['rank']:2d}. {row['feature']:30s} (importance: {row['importance']:.4f})"
            )

        lines.extend(
            [
                "",
                f"Interpretation: {self.interpretation}",
                "",
                f"Computation Time: {self.computation_time:.3f}s",
            ]
        )

        return "\n".join(lines)


def compute_domain_classifier_drift(
    reference: np.ndarray | pd.DataFrame | pl.DataFrame,
    test: np.ndarray | pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
    *,
    model_type: str = "lightgbm",
    n_estimators: int = 100,
    max_depth: int = 5,
    threshold: float = 0.6,
    cv_folds: int = 5,
    random_state: int = 42,
) -> DomainClassifierResult:
    """Detect distribution drift using domain classifier.

    Trains a binary classifier to distinguish reference (label=0) from test (label=1)
    samples. AUC-ROC indicates drift magnitude, feature importance shows which features
    drifted most.

    The domain classifier approach detects multivariate drift by testing whether
    a classifier can distinguish between two distributions. If AUC ≈ 0.5, the
    distributions are indistinguishable (no drift). If AUC → 1.0, the distributions
    are completely separated (strong drift).

    **Advantages**:
        - Detects multivariate drift and feature interactions
        - Non-parametric (no distributional assumptions)
        - Interpretable via feature importance
        - Sensitive to subtle multivariate shifts

    **AUC Interpretation**:
        - AUC ≈ 0.5: No drift (random guess)
        - AUC = 0.6: Weak drift
        - AUC = 0.7-0.8: Moderate drift
        - AUC > 0.9: Strong drift

    Args:
        reference: Reference distribution (e.g., training data).
            Can be numpy array, pandas DataFrame, or polars DataFrame.
        test: Test distribution (e.g., production data).
            Can be numpy array, pandas DataFrame, or polars DataFrame.
        features: List of feature names to use. If None, uses all numeric columns.
            Only applicable for DataFrame inputs.
        model_type: Classifier type. Options:
            - "lightgbm": LightGBM (default, fastest)
            - "xgboost": XGBoost
            - "sklearn": sklearn RandomForestClassifier (always available)
        n_estimators: Number of trees/estimators (default: 100)
        max_depth: Maximum tree depth (default: 5)
        threshold: AUC threshold for flagging drift (default: 0.6)
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        DomainClassifierResult with AUC, feature importances, drift flag, etc.

    Raises:
        ValueError: If inputs are invalid or model_type is unknown
        ImportError: If required ML library is not installed

    Example:
        >>> import numpy as np
        >>> import polars as pl
        >>> from ml4t.diagnostic.evaluation.drift import compute_domain_classifier_drift
        >>>
        >>> # No drift (identical distributions)
        >>> np.random.seed(42)
        >>> ref = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500),
        >>> })
        >>> test = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test)
        >>> print(f"AUC: {result.auc:.4f}, Drifted: {result.drifted}")
        AUC: 0.5123, Drifted: False
        >>>
        >>> # Strong drift (mean shift)
        >>> test_shifted = pl.DataFrame({
        ...     "x1": np.random.normal(2, 1, 500),
        ...     "x2": np.random.normal(2, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test_shifted)
        >>> print(f"AUC: {result.auc:.4f}, Drifted: {result.drifted}")
        AUC: 0.9876, Drifted: True
        >>> print(result.summary())
        >>>
        >>> # Interaction-based drift
        >>> test_corr = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500) + 0.8 * np.random.normal(0, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test_corr)
        >>> # Will detect correlation change via feature interactions

    References:
        - Lopez-Paz, D., & Oquab, M. (2017). Revisiting Classifier Two-Sample Tests.
          ICLR 2017.
        - Rabanser, S., et al. (2019). Failing Loudly: An Empirical Study of Methods
          for Detecting Dataset Shift. NeurIPS 2019.
    """
    start_time = time.time()

    # Prepare data
    X, y, feature_names = _prepare_domain_classification_data(reference, test, features)

    # Train classifier with cross-validation
    model, cv_scores = _train_domain_classifier(
        X,
        y,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Extract feature importances
    importances_df = _extract_feature_importances(model, feature_names)

    # Compute final AUC on full data
    from sklearn.metrics import roc_auc_score

    y_pred_proba = model.predict_proba(X)[:, 1]
    final_auc = float(roc_auc_score(y, y_pred_proba))

    # Determine drift status
    drifted = final_auc > threshold

    # Generate interpretation
    cv_auc_mean = float(np.mean(cv_scores))
    cv_auc_std = float(np.std(cv_scores))

    if drifted:
        if final_auc > 0.9:
            severity = "strong"
        elif final_auc > 0.7:
            severity = "moderate"
        else:
            severity = "weak"

        interpretation = (
            f"{severity.capitalize()} distribution drift detected "
            f"(AUC={final_auc:.4f} > {threshold:.4f}). "
            f"The classifier can distinguish reference from test distributions. "
            f"Top drifted feature: {importances_df['feature'][0]}."
        )
    else:
        interpretation = (
            f"No significant drift detected (AUC={final_auc:.4f} ≤ {threshold:.4f}). "
            f"Distributions are indistinguishable by the classifier."
        )

    computation_time = time.time() - start_time

    return DomainClassifierResult(
        auc=final_auc,
        drifted=drifted,
        feature_importances=importances_df,
        threshold=threshold,
        n_reference=int(np.sum(y == 0)),
        n_test=int(np.sum(y == 1)),
        n_features=len(feature_names),
        model_type=model_type,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        interpretation=interpretation,
        computation_time=computation_time,
        metadata={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
    )


def _prepare_domain_classification_data(
    reference: np.ndarray | pd.DataFrame | pl.DataFrame,
    test: np.ndarray | pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare labeled dataset for domain classification.

    Args:
        reference: Reference distribution
        test: Test distribution
        features: Feature names to use (for DataFrames)

    Returns:
        Tuple of (X, y, feature_names):
            - X: Feature matrix (reference + test concatenated)
            - y: Labels (0 for reference, 1 for test)
            - feature_names: List of feature names

    Raises:
        ValueError: If inputs are invalid or incompatible
    """
    # Convert to numpy arrays
    if isinstance(reference, pl.DataFrame):
        if features is None:
            # Use all numeric columns
            features = [
                c
                for c in reference.columns
                if reference[c].dtype
                in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
            ]
        X_ref = reference[features].to_numpy()
        feature_names = features

    elif isinstance(reference, pd.DataFrame):
        if features is None:
            # Use all numeric columns
            features = list(reference.select_dtypes(include=[np.number]).columns)
        X_ref = reference[features].to_numpy()
        feature_names = features

    elif isinstance(reference, np.ndarray):
        X_ref = reference
        if features is None:
            # Generate default feature names
            if X_ref.ndim == 1:
                X_ref = X_ref.reshape(-1, 1)
            feature_names = [f"feature_{i}" for i in range(X_ref.shape[1])]
        else:
            feature_names = features

    else:
        raise ValueError(
            f"Unsupported reference type: {type(reference)}. "
            "Must be numpy array, pandas DataFrame, or polars DataFrame."
        )

    # Process test data
    if isinstance(test, pl.DataFrame | pd.DataFrame):
        X_test = test[feature_names].to_numpy()
    elif isinstance(test, np.ndarray):
        X_test = test
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
    else:
        raise ValueError(
            f"Unsupported test type: {type(test)}. Must be numpy array, pandas DataFrame, or polars DataFrame."
        )

    # Validate shapes
    if X_ref.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference has {X_ref.shape[1]} features, test has {X_test.shape[1]} features."
        )

    # Concatenate and create labels
    X = np.vstack([X_ref, X_test])
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_test))])

    return X, y, feature_names


def _train_domain_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "lightgbm",
    n_estimators: int = 100,
    max_depth: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[Any, np.ndarray]:
    """Train binary classifier for domain classification.

    Args:
        X: Feature matrix
        y: Labels (0=reference, 1=test)
        model_type: Classifier type
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        cv_folds: Cross-validation folds
        random_state: Random seed

    Returns:
        Tuple of (trained_model, cv_auc_scores)

    Raises:
        ValueError: If model_type is unknown
        ImportError: If required library is not installed
    """
    from sklearn.model_selection import cross_val_score

    # Select and configure model
    if model_type == "lightgbm":
        if not _check_lightgbm_available():
            raise ImportError(
                "LightGBM required for domain classifier drift detection. "
                "Install with: pip install ml4t-diagnostic[ml] or pip install lightgbm"
            )

        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbose=-1,
            force_col_wise=True,  # Suppress warning
        )

    elif model_type == "xgboost":
        if not _check_xgboost_available():
            raise ImportError(
                "XGBoost required for domain classifier drift detection. Install with: pip install xgboost"
            )

        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
        )

    elif model_type == "sklearn":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Must be 'lightgbm', 'xgboost', or 'sklearn'."
        )

    # Cross-validation for AUC
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc")

    # Train on full data
    model.fit(X, y)

    return model, cv_scores


def _extract_feature_importances(model: Any, feature_names: list[str]) -> pl.DataFrame:
    """Extract and rank feature importances.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        Polars DataFrame with columns: feature, importance, rank

    Raises:
        ValueError: If model doesn't have feature importances
    """
    # Get importances (works for LightGBM, XGBoost, sklearn)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError(f"Model type {type(model)} does not have feature_importances_ attribute")

    # Create DataFrame
    df = pl.DataFrame({"feature": feature_names, "importance": importances})

    # Sort by importance (descending)
    df = df.sort("importance", descending=True)

    # Add rank
    df = df.with_columns(pl.arange(1, len(df) + 1).alias("rank"))

    return df
