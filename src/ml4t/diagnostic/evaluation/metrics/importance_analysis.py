"""Comprehensive ML feature importance analysis comparing multiple methods.

This module provides a tear sheet function that runs MDI, PFI, MDA, and SHAP
importance methods and generates a comparison report with consensus ranking.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from ml4t.diagnostic.evaluation.metrics.importance_classical import (
    compute_mdi_importance,
    compute_permutation_importance,
)
from ml4t.diagnostic.evaluation.metrics.importance_mda import compute_mda_importance
from ml4t.diagnostic.evaluation.metrics.importance_shap import compute_shap_importance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _generate_ml_importance_interpretation(
    top_features: list[str],
    method_agreement: dict[str, float],
    warnings: list[str],
    n_consensus: int,
) -> str:
    """Generate human-readable interpretation of ML importance analysis.

    Parameters
    ----------
    top_features : list[str]
        Top features from consensus ranking
    method_agreement : dict[str, float]
        Pairwise correlations between methods
    warnings : list[str]
        List of potential issues detected
    n_consensus : int
        Number of features in top 10 across all methods

    Returns
    -------
    str
        Human-readable interpretation summary
    """
    lines = []

    # Consensus features
    if n_consensus > 0:
        lines.append(f"Strong consensus: {n_consensus} features rank in top 10 across all methods")
        lines.append(f"  Top consensus features: {', '.join(top_features[:5])}")
    else:
        lines.append("Weak consensus: Different methods identify different important features")

    # Method agreement
    if method_agreement:
        avg_agreement = float(np.mean(list(method_agreement.values())))
        if avg_agreement > 0.7:
            lines.append(f"High agreement between methods (avg correlation: {avg_agreement:.2f})")
        elif avg_agreement > 0.5:
            lines.append(
                f"Moderate agreement between methods (avg correlation: {avg_agreement:.2f})"
            )
        else:
            lines.append(
                f"Low agreement between methods (avg correlation: {avg_agreement:.2f}) - investigate further"
            )

    # Warnings
    if warnings:
        lines.append("\nPotential Issues:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def analyze_ml_importance(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    y: Union[pl.Series, pd.Series, "NDArray[Any]"],
    feature_names: list[str] | None = None,
    methods: list[str] | None = None,
    scoring: str | Callable | None = None,
    n_repeats: int = 10,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Comprehensive ML feature importance analysis comparing multiple methods.

    **This is a TEAR SHEET function** - it runs multiple importance methods and
    generates a comparison report with consensus ranking and interpretation.

    **Use Case**: "Which features does my model rely on? Do different methods agree?"

    This function replaces 100+ lines of manual comparison code by providing
    integrated analysis showing:
    - Individual method results (MDI, PFI, MDA, SHAP)
    - Consensus ranking (features important across methods)
    - Method agreement/disagreement analysis
    - Auto-generated insights and warnings

    **Why Compare Methods?**

    Different importance methods measure different aspects:
    - **MDI** (Mean Decrease Impurity): Fast, but biased toward high-cardinality features
    - **PFI** (Permutation): Unbiased, measures predictive importance
    - **MDA** (Mean Decrease Accuracy): Similar to PFI but removes features completely
    - **SHAP**: Theoretically sound, based on game theory

    Strong consensus across methods indicates robust feature importance.
    Disagreement suggests model-specific artifacts or feature interactions.

    Parameters
    ----------
    model : Any
        Fitted model. Requirements vary by method:
        - MDI: Must have `feature_importances_` (tree-based models)
        - PFI, MDA: Must have `predict()` or `score()`
        - SHAP: Must be compatible with TreeExplainer
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    y : Union[pl.Series, pd.Series, np.ndarray]
        Target values (n_samples,)
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses column names from DataFrame
        or generates numeric names
    methods : list[str] | None, default ["mdi", "pfi", "shap"]
        Which methods to run. Options: "mdi", "pfi", "mda", "shap"
    scoring : str | Callable | None, default None
        Scoring metric for PFI and MDA
    n_repeats : int, default 10
        Number of permutations for PFI
    random_state : int | None, default 42
        Random seed for reproducibility

    Returns
    -------
    dict[str, Any]
        Comprehensive analysis results:
        - method_results: Dict of individual method outputs
        - consensus_ranking: Features ranked by average rank across methods
        - method_agreement: Spearman correlations between method rankings
        - top_features_consensus: Features in top 10 for ALL methods
        - warnings: Detected issues
        - interpretation: Auto-generated summary
        - methods_run: Methods successfully executed
        - methods_failed: Failed methods with error messages

    Raises
    ------
    ValueError
        If no methods specified or all methods fail

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Create synthetic dataset
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(n_estimators=50, random_state=42)
    >>> model.fit(X, y)
    >>>
    >>> # Comprehensive importance analysis
    >>> result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"])
    >>>
    >>> # Quick summary
    >>> print(result["interpretation"])
    """
    if methods is None:
        methods = ["mdi", "pfi", "shap"]

    if not methods:
        raise ValueError("At least one method must be specified")

    # Extract feature names if not provided
    if feature_names is None:
        if isinstance(X, pl.DataFrame | pd.DataFrame):
            feature_names = list(X.columns)
        else:
            # Generate numeric feature names
            n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
            feature_names = [f"f{i}" for i in range(n_features)]

    # Run each method with try/except for optional dependencies
    results = {}
    method_failures = []

    if "mdi" in methods:
        try:
            results["mdi"] = compute_mdi_importance(model, feature_names=feature_names)
        except Exception as e:
            method_failures.append(("mdi", str(e)))

    if "pfi" in methods:
        try:
            results["pfi"] = compute_permutation_importance(
                model,
                X,
                y,
                feature_names=feature_names,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        except Exception as e:
            method_failures.append(("pfi", str(e)))

    if "mda" in methods:
        try:
            results["mda"] = compute_mda_importance(
                model, X, y, feature_names=feature_names, scoring=scoring
            )
        except Exception as e:
            method_failures.append(("mda", str(e)))

    if "shap" in methods:
        try:
            results["shap"] = compute_shap_importance(model, X, feature_names=feature_names)
        except ImportError:
            method_failures.append(
                (
                    "shap",
                    "shap library not installed. Install with: pip install ml4t-diagnostic[ml]",
                )
            )
        except Exception as e:
            method_failures.append(("shap", str(e)))

    # Check if at least one method succeeded
    if not results:
        error_msg = "All methods failed:\n" + "\n".join(
            f"  - {method}: {error}" for method, error in method_failures
        )
        raise ValueError(error_msg)

    # 2. Compute consensus ranking
    # Convert each method's importance to rankings (1 = most important)
    rankings = {}
    for method_name, result in results.items():
        # Get feature names and importances for this method
        method_feature_names = result["feature_names"]

        if method_name == "pfi":
            importances = result["importances_mean"]
        elif method_name in ["shap", "mdi", "mda"]:
            importances = result["importances"]
        else:
            # Shouldn't happen, but handle gracefully
            continue

        # Create a mapping from feature name to importance
        feature_to_importance = dict(zip(method_feature_names, importances, strict=False))

        # Map to our canonical feature_names list (handle missing features)
        importance_values = np.array(
            [feature_to_importance.get(fname, 0.0) for fname in feature_names]
        )

        # Rank (higher importance = lower rank number, i.e., rank 0 is most important)
        ranks = np.argsort(np.argsort(importance_values)[::-1])
        rankings[method_name] = ranks

    # Average ranks across methods
    avg_ranks = np.mean(list(rankings.values()), axis=0)
    consensus_order = np.argsort(avg_ranks)

    # Get feature names in consensus order
    consensus_ranking = [feature_names[i] for i in consensus_order]

    # 3. Compute method agreement (Spearman correlation between rankings)
    method_agreement = {}
    method_names = list(rankings.keys())
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1 :]:
            corr, _ = spearmanr(rankings[m1], rankings[m2])
            method_agreement[f"{m1}_vs_{m2}"] = float(corr)

    # 4. Identify consensus top features (top 10 in all methods)
    top_n = 10
    top_features_by_method = {}
    for method_name, result in results.items():
        # Get top N feature names from this method
        method_top_features = result["feature_names"][:top_n]
        top_features_by_method[method_name] = set(method_top_features)

    consensus_top = (
        set.intersection(*top_features_by_method.values()) if top_features_by_method else set()
    )

    # 5. Generate warnings
    warnings = []

    # Warning: High MDI but low PFI (possible overfitting)
    if "mdi" in results and "pfi" in results:
        mdi_top = set(results["mdi"]["feature_names"][:5])
        pfi_top = set(results["pfi"]["feature_names"][:5])
        disagreement = mdi_top - pfi_top
        if disagreement:
            warnings.append(
                f"Features {disagreement} rank high in MDI but not PFI - possible overfitting to tree structure"
            )

    # Warning: Low agreement between methods
    if method_agreement:
        min_agreement = min(method_agreement.values())
        if min_agreement < 0.5:
            warnings.append(
                f"Low agreement between methods (min correlation: {min_agreement:.2f}) - results may be unreliable"
            )

    # Add method failures to warnings
    if method_failures:
        for method, error in method_failures:
            warnings.append(f"Method '{method}' failed: {error}")

    # 6. Generate interpretation
    interpretation = _generate_ml_importance_interpretation(
        consensus_ranking[:10],
        method_agreement,
        warnings,
        len(consensus_top),
    )

    return {
        "method_results": results,
        "consensus_ranking": consensus_ranking,
        "method_agreement": method_agreement,
        "top_features_consensus": list(consensus_top),
        "warnings": warnings,
        "interpretation": interpretation,
        "methods_run": list(results.keys()),
        "methods_failed": method_failures,
    }
