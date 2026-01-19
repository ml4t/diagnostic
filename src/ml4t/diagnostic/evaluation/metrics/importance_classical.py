"""Classical feature importance: Permutation (PFI) and Mean Decrease Impurity (MDI).

This module provides model-agnostic permutation importance and tree-based MDI
importance calculations.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_permutation_importance(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    y: Union[pl.Series, pd.Series, "NDArray[Any]"],
    feature_names: list[str] | None = None,
    scoring: str | Callable | None = None,
    n_repeats: int = 10,
    random_state: int | None = 42,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Compute Permutation Feature Importance (PFI) for model-agnostic feature ranking.

    Permutation Feature Importance measures the increase in model error when a
    feature's values are randomly shuffled. Features with high importance cause
    large performance drops when permuted, indicating they are critical for
    the model's predictions.

    This is a model-agnostic method that works with any fitted estimator,
    making it superior to model-specific importance measures (e.g., tree-based
    feature importances) which can be biased toward high-cardinality features.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator (must have `predict` or `predict_proba`)
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    y : Union[pl.Series, pd.Series, np.ndarray]
        Target values (n_samples,)
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses column names from DataFrame
        or generates numeric names for arrays
    scoring : str | Callable | None, default None
        Scoring function to evaluate model performance. If None, uses model's
        default score method. Common options:
        - Classification: 'accuracy', 'roc_auc', 'f1'
        - Regression: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    n_repeats : int, default 10
        Number of times to permute each feature (more repeats = more stable estimates)
    random_state : int | None, default 42
        Random seed for reproducibility
    n_jobs : int | None, default None
        Number of parallel jobs (-1 for all CPUs)

    Returns
    -------
    dict[str, Any]
        Dictionary with permutation importance results:
        - importances_mean: Mean importance per feature
        - importances_std: Standard deviation of importance per feature
        - importances_raw: All permutation results (n_features, n_repeats)
        - feature_names: Feature labels
        - baseline_score: Model score before permutation
        - n_repeats: Number of permutation rounds
        - scoring: Scoring function used

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Train a simple model
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> model.fit(X, y)
    >>>
    >>> # Compute permutation importance
    >>> pfi = compute_permutation_importance(
    ...     model=model,
    ...     X=X,
    ...     y=y,
    ...     n_repeats=10,
    ...     scoring='accuracy'
    ... )
    >>>
    >>> # Examine results
    >>> print(f"Baseline score: {pfi['baseline_score']:.3f}")
    >>> print(f"Most important feature: {pfi['feature_names'][np.argmax(pfi['importances_mean'])]}")
    >>> print(f"Importance: {np.max(pfi['importances_mean']):.3f} ± {pfi['importances_std'][np.argmax(pfi['importances_mean'])]:.3f}")
    Baseline score: 0.920
    Most important feature: feature_0
    Importance: 0.124 ± 0.015

    Notes
    -----
    **Interpretation**:
    - Importance = 0: Feature not useful
    - Importance > 0: Feature contributes to predictions
    - Importance < 0: Feature hurts performance (may indicate overfitting)
    - Higher importance = More critical feature

    **Advantages over MDI** (Mean Decrease in Impurity):
    - Model-agnostic: Works with any estimator
    - Unbiased: Not inflated by high-cardinality features
    - Realistic: Measures actual predictive power, not just tree splits

    **Computational Cost**:
    - Time complexity: O(n_features * n_repeats * prediction_time)
    - Can be slow for large datasets or complex models
    - Use n_jobs=-1 for parallel computation

    **Best Practices**:
    - Use hold-out validation set (not training data) for unbiased estimates
    - Increase n_repeats (20-30) for more stable results
    - Check for negative importances (may indicate model instability)
    - Compare with other importance methods (SHAP, MDI) for robustness

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    """
    from sklearn.inspection import permutation_importance as sklearn_pfi

    # Convert inputs to numpy arrays
    X_array: NDArray[Any]
    if isinstance(X, pl.DataFrame):
        if feature_names is None:
            feature_names = X.columns
        X_array = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_array = X.to_numpy()
    else:
        X_array = np.asarray(X)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    # Type assertion: feature_names is guaranteed to be set at this point
    assert feature_names is not None, "feature_names should be set by this point"

    y_array: NDArray[Any]
    if isinstance(y, pl.Series):
        y_array = y.to_numpy()
    elif isinstance(y, pd.Series):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)

    # Compute baseline score
    if scoring is None:
        baseline_score = model.score(X_array, y_array)
    else:
        from sklearn.metrics import get_scorer

        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        baseline_score = scorer(model, X_array, y_array)

    # Compute permutation importance using sklearn
    result = sklearn_pfi(
        estimator=model,
        X=X_array,
        y=y_array,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Extract and format results
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    importances_raw = result.importances  # Shape: (n_features, n_repeats)

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances_mean)[::-1]

    return {
        "importances_mean": importances_mean[sorted_idx],
        "importances_std": importances_std[sorted_idx],
        "importances_raw": importances_raw[sorted_idx],
        "feature_names": [feature_names[i] for i in sorted_idx],
        "baseline_score": float(baseline_score),
        "n_repeats": n_repeats,
        "scoring": scoring if scoring is not None else "default",
        "n_features": len(feature_names),
    }


def compute_mdi_importance(
    model: Any,
    feature_names: list[str] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Compute Mean Decrease in Impurity (MDI) feature importance from tree-based models.

    MDI measures how much each feature contributes to decreasing the weighted
    impurity (Gini for classification, MSE/MAE for regression) across all trees.
    This is computed during model training and is available via the model's
    `feature_importances_` attribute.

    **Supported Models**:
    - LightGBM: `lightgbm.LGBMClassifier`, `lightgbm.LGBMRegressor` (recommended)
    - XGBoost: `xgboost.XGBClassifier`, `xgboost.XGBRegressor` (recommended)
    - sklearn: `RandomForestClassifier`, `RandomForestRegressor` (not recommended - slow)
    - sklearn: `GradientBoostingClassifier`, `GradientBoostingRegressor` (not recommended - slow)

    **Not supported**:
    - sklearn's HistGradientBoosting* (doesn't expose feature_importances_)

    Parameters
    ----------
    model : Any
        Fitted tree-based model with `feature_importances_` attribute.
        Must be one of: LightGBM, XGBoost, or sklearn tree ensembles.
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses feature names from model
        or generates numeric names.
    normalize : bool, default True
        If True, ensures importances sum to 1.0 (some models already normalize).

    Returns
    -------
    dict[str, Any]
        Dictionary with MDI importance results:
        - importances: Feature importance values (sorted descending)
        - feature_names: Feature labels (sorted by importance)
        - n_features: Number of features
        - normalized: Whether values sum to 1.0
        - model_type: Type of model used

    Raises
    ------
    AttributeError
        If model doesn't have `feature_importances_` attribute
    ImportError
        If LightGBM/XGBoost not installed and trying to use those models

    Examples
    --------
    >>> import lightgbm as lgb
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Train LightGBM model
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    >>> model.fit(X, y)
    >>>
    >>> # Extract MDI importance
    >>> mdi = compute_mdi_importance(
    ...     model=model,
    ...     feature_names=[f'feature_{i}' for i in range(10)]
    ... )
    >>>
    >>> # Examine results
    >>> print(f"Most important feature: {mdi['feature_names'][0]}")
    >>> print(f"Importance: {mdi['importances'][0]:.3f}")
    >>> print(f"Model type: {mdi['model_type']}")
    Most important feature: feature_3
    Importance: 0.245
    Model type: lightgbm.LGBMClassifier

    Notes
    -----
    **MDI vs PFI** (Permutation Feature Importance):

    **MDI Advantages**:
    - Very fast: Computed during training (no additional overhead)
    - No additional data required
    - Deterministic: Same result every time

    **MDI Disadvantages**:
    - **Biased toward high-cardinality features**: Features with many unique values
      get inflated importance even if not truly predictive
    - **Only for tree-based models**: Not model-agnostic
    - **Train set importance**: May not reflect test set predictive power
    - **Correlated features**: Can split importance between correlated predictors

    **When to use MDI**:
    - Quick exploratory analysis
    - When computational budget is limited
    - When working with tree-based models exclusively

    **When to use PFI instead**:
    - Need unbiased importance estimates
    - Have high-cardinality categorical features
    - Want model-agnostic importance
    - Need to validate importance on test set

    **Comparison workflow**:
    >>> # Compare MDI and PFI
    >>> mdi = compute_mdi_importance(model, feature_names=features)
    >>> pfi = compute_permutation_importance(model, X_test, y_test, feature_names=features)
    >>>
    >>> # Large discrepancies may indicate:
    >>> # - High-cardinality bias in MDI
    >>> # - Correlated features splitting importance
    >>> # - Overfitting (high MDI, low PFI)

    **Performance notes**:
    - LightGBM and XGBoost: Production-ready speed and accuracy (RECOMMENDED)
    - sklearn RandomForest/GradientBoosting: 10-100x slower, avoid for large datasets
    - sklearn HistGradientBoosting: Fast but doesn't expose feature_importances_ (use PFI instead)

    References
    ----------
    - Breiman, L. (2001). "Random Forests". Machine Learning.
    - Louppe, G. et al. (2013). "Understanding variable importances in forests of
      randomized trees". NeurIPS.
    - Strobl, C. et al. (2007). "Bias in random forest variable importance measures".
      BMC Bioinformatics.
    """
    # Check if model has feature_importances_
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model of type {type(model).__name__} does not have 'feature_importances_' attribute. "
            "MDI is only available for tree-based models (LightGBM, XGBoost, sklearn tree ensembles)."
        )

    # Extract raw importances
    importances = model.feature_importances_

    # Get feature names
    if feature_names is None:
        # Try to get from model
        if hasattr(model, "feature_name_"):
            # LightGBM
            feature_names = model.feature_name_
        elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
            # XGBoost
            feature_names = model.get_booster().feature_names
        elif hasattr(model, "feature_names_in_"):
            # sklearn
            feature_names = list(model.feature_names_in_)
        else:
            # Fallback to numeric names
            feature_names = [f"feature_{i}" for i in range(len(importances))]
    else:
        feature_names = list(feature_names)

    # Validate length match
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not match number of importances ({len(importances)})"
        )

    # Normalize if requested
    if normalize:
        importance_sum = importances.sum()
        if importance_sum > 0:
            importances = importances / importance_sum
        else:
            # All zeros - already normalized
            pass

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]

    # Determine model type
    model_type = f"{type(model).__module__}.{type(model).__name__}"

    return {
        "importances": importances[sorted_idx],
        "feature_names": [feature_names[i] for i in sorted_idx],
        "n_features": len(feature_names),
        "normalized": normalize,
        "model_type": model_type,
    }
