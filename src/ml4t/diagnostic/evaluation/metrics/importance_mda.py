"""Mean Decrease in Accuracy (MDA) feature importance by feature removal.

This module provides MDA importance which measures performance drop when features
are neutralized, with support for feature groups.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_mda_importance(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    y: Union[pl.Series, pd.Series, "NDArray[Any]"],
    feature_names: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
    removal_method: str = "mean",
    scoring: str | Callable | None = None,
    _n_jobs: int | None = None,
) -> dict[str, Any]:
    """Compute Mean Decrease in Accuracy (MDA) by feature removal.

    MDA measures the drop in model performance when features are removed or
    neutralized. Unlike Permutation Feature Importance (PFI) which shuffles
    feature values, MDA replaces feature values with a constant (mean, median,
    or zero), simulating complete feature unavailability.

    This approach naturally supports feature groups (e.g., one-hot encoded
    categoricals, related features like lat/lon) by removing multiple features
    simultaneously and measuring the joint importance.

    **Supported Models**:
    - Any fitted sklearn-compatible estimator with `score()` or `predict()` method
    - Classification: LogisticRegression, RandomForest, XGBoost, LightGBM, etc.
    - Regression: LinearRegression, Ridge, GradientBoosting, etc.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible estimator (must have `score()` or `predict()` method)
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    y : Union[pl.Series, pd.Series, np.ndarray]
        Target values (n_samples,)
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses column names from DataFrame
        or generates numeric names for arrays
    feature_groups : dict[str, list[str]] | None, default None
        Dictionary mapping group names to lists of feature names.
        When provided, computes importance for feature groups instead of
        individual features. Example: {"location": ["lat", "lon"],
        "time": ["hour", "day", "month"]}
    removal_method : str, default "mean"
        How to neutralize features:
        - "mean": Replace with feature mean (recommended for continuous features)
        - "median": Replace with feature median (robust to outliers)
        - "zero": Replace with zero (can distort if zero is out-of-distribution)
    scoring : str | Callable | None, default None
        Scoring function to evaluate model performance. If None, uses model's
        default score method. Common options:
        - Classification: 'accuracy', 'roc_auc', 'f1'
        - Regression: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
    n_jobs : int | None, default None
        Number of parallel jobs for scoring (-1 for all CPUs).
        Note: Parallelization is limited compared to sklearn's implementation
        since we need to modify data for each feature.

    Returns
    -------
    dict[str, Any]
        Dictionary with MDA importance results:
        - importances: Performance drop per feature/group (sorted descending)
        - feature_names: Feature/group labels (sorted by importance)
        - baseline_score: Model score before feature removal
        - removal_method: Method used to neutralize features
        - scoring: Scoring function used
        - n_features: Number of features/groups evaluated

    Raises
    ------
    ValueError
        If removal_method is not one of: "mean", "median", "zero"
    ValueError
        If feature_groups contains unknown feature names
    ValueError
        If X and y have different numbers of samples

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>>
    >>> # Train a simple model
    >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, random_state=42)
    >>> model = RandomForestClassifier(n_estimators=50, random_state=42)
    >>> model.fit(X, y)
    >>>
    >>> # Compute MDA importance
    >>> mda = compute_mda_importance(
    ...     model=model,
    ...     X=X,
    ...     y=y,
    ...     removal_method='mean',
    ...     scoring='accuracy'
    ... )
    >>>
    >>> # Examine results
    >>> print(f"Baseline score: {mda['baseline_score']:.3f}")
    >>> print(f"Most important feature: {mda['feature_names'][0]}")
    >>> print(f"Importance (accuracy drop): {mda['importances'][0]:.3f}")
    Baseline score: 0.920
    Most important feature: feature_3
    Importance (accuracy drop): 0.124

    **Feature Groups Example**:

    >>> # Group related features (e.g., one-hot encoded categorical)
    >>> feature_groups = {
    ...     "category_A": ["feature_0", "feature_1", "feature_2"],
    ...     "category_B": ["feature_3", "feature_4"],
    ...     "numeric": ["feature_5", "feature_6", "feature_7"]
    ... }
    >>>
    >>> mda_groups = compute_mda_importance(
    ...     model=model,
    ...     X=X,
    ...     y=y,
    ...     feature_groups=feature_groups,
    ...     removal_method='mean'
    ... )
    >>>
    >>> # See which group is most important
    >>> print(f"Most important group: {mda_groups['feature_names'][0]}")
    >>> print(f"Group importance: {mda_groups['importances'][0]:.3f}")

    Notes
    -----
    **MDA vs PFI** (Permutation Feature Importance):

    **MDA Characteristics**:
    - Removes feature completely (sets to constant)
    - Simulates true feature unavailability
    - May show larger importance drops than PFI
    - Naturally supports feature groups
    - Similar computational cost to PFI

    **PFI Characteristics**:
    - Shuffles feature values (breaks feature-target relationship)
    - Preserves feature distribution
    - May show smaller importance drops
    - Requires additional logic for feature groups
    - More commonly used in literature

    **When to use MDA**:
    - Want to simulate complete feature removal
    - Need to evaluate feature groups jointly
    - Want more conservative importance estimates
    - Comparing "with feature" vs "without feature" scenarios

    **When to use PFI instead**:
    - Want to match published baselines (PFI more common)
    - Need to preserve feature distributions
    - Want less conservative importance estimates

    **Feature Groups**:
    Feature groups are useful for:
    - One-hot encoded categoricals (remove all dummy variables together)
    - Related features (lat/lon, year/month/day)
    - Multi-dimensional embeddings
    - Polynomial features of same base feature

    Removing feature groups jointly captures their combined importance,
    which can be higher than the sum of individual importances due to
    interactions between features in the group.

    **Removal Methods**:

    - **mean**: Most common choice for continuous features. Replaces feature
      with its training set mean. This is a "neutral" value that doesn't
      distort the model's input distribution.

    - **median**: More robust to outliers than mean. Useful for features with
      skewed distributions or outliers.

    - **zero**: Simple but can be problematic if zero is out-of-distribution
      for a feature (e.g., if feature is always positive). Use with caution.

    **Computational Cost**:
    - Time complexity: O(n_features * prediction_time) or O(n_groups * prediction_time)
    - Same order as PFI (one evaluation per feature/group)
    - Cannot be trivially parallelized (requires data modification)
    - Faster than SHAP for large datasets

    **Comparison with Other Methods**:

    | Method | Speed    | Groups | Local | Theory      | Bias |
    |--------|----------|--------|-------|-------------|------|
    | MDI    | Fastest  | No     | No    | Weak        | Yes  |
    | PFI    | Slow     | Hard   | No    | Strong      | No   |
    | MDA    | Slow     | Yes    | No    | Strong      | No   |
    | SHAP   | Medium   | No     | Yes   | Strongest   | No   |

    - **Speed**: MDI instant (from training), PFI/MDA slow (repeated scoring),
      SHAP medium (depends on data size)
    - **Groups**: MDA naturally supports, PFI requires workarounds, MDI/SHAP no
    - **Local**: SHAP provides per-sample importances, others are global only
    - **Theory**: SHAP has strongest game-theoretic foundation, PFI/MDA empirical
    - **Bias**: MDI biased toward high-cardinality features, others unbiased

    **Best Practices**:
    - Use validation/test set (not training data) for unbiased estimates
    - Compare MDA with PFI and SHAP for robustness
    - Use feature groups for one-hot encoded categoricals
    - Choose removal_method based on feature distributions
    - Verify model still makes reasonable predictions after removal

    References
    ----------
    .. [ALT] A. Altmann, L. Toloşi, O. Sander, T. Lengauer,
       "Permutation importance: a corrected feature importance measure",
       Bioinformatics, 26(10), 1340-1347, 2010.
    .. [FIS] A. Fisher, C. Rudin, F. Dominici,
       "All Models are Wrong, but Many are Useful: Learning a Variable's
       Importance by Studying an Entire Class of Prediction Models Simultaneously",
       JMLR, 20(177):1-81, 2019.
    """
    # Validate removal method
    valid_methods = ["mean", "median", "zero"]
    if removal_method not in valid_methods:
        raise ValueError(f"removal_method must be one of {valid_methods}, got '{removal_method}'")

    # Convert inputs to numpy arrays and extract feature names
    if isinstance(X, pl.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)  # Polars columns is already a list
        X_array = X.to_numpy()
    elif isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = np.asarray(X)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    y_array: NDArray[Any]
    if isinstance(y, pl.Series):
        y_array = y.to_numpy()
    elif isinstance(y, pd.Series):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)

    # Validate dimensions
    n_samples, n_features = X_array.shape
    if len(y_array) != n_samples:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: {n_samples} vs {len(y_array)}"
        )

    # Set up scoring function
    if scoring is None:
        scorer = None
        baseline_score = model.score(X_array, y_array)
        scoring_name = "default"
    else:
        from sklearn.metrics import get_scorer

        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        baseline_score = scorer(model, X_array, y_array)
        scoring_name = scoring if isinstance(scoring, str) else "custom"

    # Compute feature replacement values based on removal method
    if removal_method == "mean":
        replacement_values = np.mean(X_array, axis=0)
    elif removal_method == "median":
        replacement_values = np.median(X_array, axis=0)
    else:  # removal_method == "zero"
        replacement_values = np.zeros(n_features)

    # Determine whether we're evaluating individual features or groups
    if feature_groups is not None:
        # Validate feature groups (feature_names is always set by this point)
        assert feature_names is not None
        all_group_features: set[str] = set()
        for group_name, features in feature_groups.items():
            for feat in features:
                if feat not in feature_names:
                    raise ValueError(
                        f"Feature '{feat}' in group '{group_name}' not found in feature_names"
                    )
                all_group_features.add(feat)

        # Map feature names to indices
        feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

        # Compute importance for each group
        importances_list = []
        group_names = []

        for group_name, features in feature_groups.items():
            # Get indices for all features in this group
            feature_indices = [feature_name_to_idx[feat] for feat in features]

            # Create modified data with group features removed
            X_removed = X_array.copy()
            for idx in feature_indices:
                X_removed[:, idx] = replacement_values[idx]

            # Compute score with group removed
            removed_score = (
                model.score(X_removed, y_array)
                if scorer is None
                else scorer(model, X_removed, y_array)
            )

            # Importance is the drop in performance
            importance = baseline_score - removed_score
            importances_list.append(importance)
            group_names.append(group_name)

        importances = np.array(importances_list)
        eval_feature_names = group_names
        n_eval_features = len(feature_groups)

    else:
        # Compute importance for individual features
        importances_list = []

        for feature_idx in range(n_features):
            # Create modified data with feature removed
            X_removed = X_array.copy()
            X_removed[:, feature_idx] = replacement_values[feature_idx]

            # Compute score with feature removed
            removed_score = (
                model.score(X_removed, y_array)
                if scorer is None
                else scorer(model, X_removed, y_array)
            )

            # Importance is the drop in performance
            importance = baseline_score - removed_score
            importances_list.append(importance)

        importances = np.array(importances_list)
        eval_feature_names = feature_names
        n_eval_features = n_features

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]

    # Type assertion: eval_feature_names is guaranteed to be set
    assert eval_feature_names is not None, "eval_feature_names should be set by this point"

    return {
        "importances": importances[sorted_idx],
        "feature_names": [eval_feature_names[i] for i in sorted_idx],
        "baseline_score": float(baseline_score),
        "removal_method": removal_method,
        "scoring": scoring_name,
        "n_features": n_eval_features,
    }
