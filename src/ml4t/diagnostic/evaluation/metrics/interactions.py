"""Feature interaction detection: H-statistic, SHAP interactions, and comprehensive analysis.

This module provides methods for detecting and analyzing feature interactions
including Friedman's H-statistic and SHAP interaction values.
"""

import time
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from ml4t.diagnostic.evaluation.metrics.conditional_ic import compute_conditional_ic

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _to_numpy_with_feature_names(
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    feature_names: list[str] | None = None,
) -> tuple["NDArray[Any]", list[str]]:
    """Normalize input to numpy array plus feature names."""
    if isinstance(X, pl.DataFrame):
        names = X.columns if feature_names is None else feature_names
        return X.to_numpy(), list(names)
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if feature_names is None else feature_names
        return pl.from_pandas(X).to_numpy(), list(names)
    names = feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    return X, list(names)


def compute_h_statistic(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    feature_pairs: list[tuple[int, int]] | list[tuple[str, str]] | None = None,
    feature_names: list[str] | None = None,
    n_samples: int = 100,
    grid_resolution: int = 20,
) -> dict[str, Any]:
    """Compute Friedman's H-statistic for feature interaction strength.

    The H-statistic (Friedman & Popescu 2008) measures how much of the variation
    in predictions can be attributed to interactions between feature pairs, beyond
    their individual main effects.

    **Algorithm**:
    1. For each feature pair (j, k):
       - Compute 2D partial dependence PD_{jk}(x_j, x_k)
       - Compute 1D partial dependences PD_j(x_j) and PD_k(x_k)
       - Compute H^2 = sum[PD_{jk} - PD_j - PD_k]^2 / sum[PD_{jk}^2]
       - H ranges from 0 (no interaction) to 1 (pure interaction)

    Parameters
    ----------
    model : Any
        Trained model with .predict() method
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    feature_pairs : list[tuple[int, int]] | list[tuple[str, str]] | None, default None
        List of (i, j) pairs to test. If None, tests all pairs.
    feature_names : list[str] | None, default None
        Feature names. If None, uses column names or f0, f1, ...
    n_samples : int, default 100
        Number of samples to use for PD computation (subsample if needed)
    grid_resolution : int, default 20
        Grid size for PD evaluation

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - h_statistics: List of (feature_i, feature_j, H_value) sorted by H descending
        - feature_names: List of feature names used
        - n_features: Number of features
        - n_pairs_tested: Number of pairs tested
        - computation_time: Time in seconds

    References
    ----------
    - Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles.
      The Annals of Applied Statistics, 2(3), 916-954.

    Examples
    --------
    >>> import lightgbm as lgb
    >>> model = lgb.LGBMRegressor()
    >>> model.fit(X_train, y_train)
    >>> results = compute_h_statistic(model, X_test)
    >>> for feat_i, feat_j, h_val in results["h_statistics"][:5]:
    ...     print(f"  {feat_i} x {feat_j}: H = {h_val:.4f}")
    """
    start_time = time.time()

    # Convert input to numpy
    X_array, feature_names_list = _to_numpy_with_feature_names(X, feature_names)
    feature_names = feature_names_list

    n_total_samples, n_features = X_array.shape

    # Subsample if needed
    if n_total_samples > n_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total_samples, size=n_samples, replace=False)
        X_sample = X_array[indices]
    else:
        X_sample = X_array
        n_samples = n_total_samples

    # Generate feature pairs if not provided - always convert to int pairs
    pairs_int: list[tuple[int, int]]
    if feature_pairs is None:
        # Test all pairs
        pairs_int = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    elif feature_names and len(feature_pairs) > 0 and isinstance(feature_pairs[0][0], str):
        # Convert string pairs to indices
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        pairs_int = [(name_to_idx[str(i)], name_to_idx[str(j)]) for i, j in feature_pairs]
    else:
        # Already integer pairs
        pairs_int = [(int(i), int(j)) for i, j in feature_pairs]

    # Ensure feature_names is a list for indexing
    feature_names_list = list(feature_names)

    h_results: list[tuple[str, str, float]] = []

    for feat_i, feat_j in pairs_int:
        # Create grids for features i and j
        x_i_grid = np.linspace(
            float(X_sample[:, feat_i].min()), float(X_sample[:, feat_i].max()), grid_resolution
        )
        x_j_grid = np.linspace(
            float(X_sample[:, feat_j].min()), float(X_sample[:, feat_j].max()), grid_resolution
        )

        # Compute 2D partial dependence PD_{ij}
        pd_2d = np.zeros((grid_resolution, grid_resolution))
        for gi, x_i_val in enumerate(x_i_grid):
            for gj, x_j_val in enumerate(x_j_grid):
                # Replace features i and j with grid values
                X_temp = X_sample.copy()
                X_temp[:, feat_i] = x_i_val
                X_temp[:, feat_j] = x_j_val
                # Average prediction over all samples
                pd_2d[gi, gj] = model.predict(X_temp).mean()

        # Compute 1D partial dependences PD_i and PD_j
        pd_i = np.zeros(grid_resolution)
        for gi, x_i_val in enumerate(x_i_grid):
            X_temp = X_sample.copy()
            X_temp[:, feat_i] = x_i_val
            pd_i[gi] = model.predict(X_temp).mean()

        pd_j = np.zeros(grid_resolution)
        for gj, x_j_val in enumerate(x_j_grid):
            X_temp = X_sample.copy()
            X_temp[:, feat_j] = x_j_val
            pd_j[gj] = model.predict(X_temp).mean()

        # Compute H-statistic
        # H^2 = sum[PD_{ij} - PD_i - PD_j + PD_const]^2 / sum[PD_{ij}^2]

        # For numerical stability, center everything
        pd_const = pd_2d.mean()
        pd_i_centered = pd_i - pd_const
        pd_j_centered = pd_j - pd_const
        pd_2d_centered = pd_2d - pd_const

        # Interaction component: PD_{ij} - PD_i - PD_j
        # Need to broadcast pd_i and pd_j to 2D
        pd_i_broadcast = pd_i_centered[:, np.newaxis]  # Shape: (grid_resolution, 1)
        pd_j_broadcast = pd_j_centered[np.newaxis, :]  # Shape: (1, grid_resolution)

        interaction = pd_2d_centered - pd_i_broadcast - pd_j_broadcast

        # H-statistic
        numerator = np.sum(interaction**2)
        denominator = np.sum(pd_2d_centered**2)

        if denominator > 1e-10:  # Avoid division by zero
            h_squared = numerator / denominator
            h_stat = np.sqrt(max(0, h_squared))  # Ensure non-negative
        else:
            h_stat = 0.0

        h_results.append((feature_names_list[feat_i], feature_names_list[feat_j], float(h_stat)))

    # Sort by H-statistic descending
    h_results.sort(key=lambda x: x[2], reverse=True)

    computation_time = time.time() - start_time

    return {
        "h_statistics": h_results,
        "feature_names": feature_names,
        "n_features": n_features,
        "n_pairs_tested": len(h_results),
        "n_samples_used": n_samples,
        "grid_resolution": grid_resolution,
        "computation_time": computation_time,
    }


def compute_shap_interactions(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    feature_names: list[str] | None = None,
    _check_additivity: bool = False,
    max_samples: int | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Compute SHAP interaction values for feature pairs.

    SHAP interaction values decompose the SHAP value of each feature into:
    - Main effect (the feature's individual contribution)
    - Interaction effects (how the feature's impact changes with other features)

    Parameters
    ----------
    model : Any
        Trained tree-based model
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    feature_names : list[str] | None, default None
        Feature names. If None, uses column names or f0, f1, ...
    _check_additivity : bool, default False
        Internal parameter (not used for interaction values)
    max_samples : int | None, default None
        Maximum samples to use (subsample if larger)
    top_k : int | None, default None
        Return only top K interactions by absolute magnitude

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - interaction_matrix: (n_features, n_features) mean absolute interactions
        - feature_names: List of feature names
        - top_interactions: List of (feature_i, feature_j, mean_interaction) sorted by magnitude
        - n_features: Number of features
        - n_samples_used: Number of samples used
        - computation_time: Time in seconds

    Notes
    -----
    - Requires shap package (install with: pip install ml4t-diagnostic[ml])
    - Only works with tree-based models (uses TreeExplainer)
    - Interaction matrix is symmetric: interaction(i,j) = interaction(j,i)
    """
    start_time = time.time()

    # Check shap availability
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "SHAP is required for interaction values. "
            "Install with: pip install ml4t-diagnostic[ml] "
            "or: pip install shap>=0.43.0"
        ) from e

    # Convert input to numpy and extract feature names
    X_array, feature_names = _to_numpy_with_feature_names(X, feature_names)

    n_total_samples, n_features = X_array.shape

    # Subsample if needed
    if max_samples is not None and n_total_samples > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total_samples, size=max_samples, replace=False)
        X_sample = X_array[indices]
        n_samples_used = max_samples
    else:
        X_sample = X_array
        n_samples_used = n_total_samples

    # Compute SHAP interaction values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_sample)

    # Handle multi-output models (classification)
    if isinstance(shap_interaction_values, list):
        # List format: use positive class for binary, average for multiclass
        if len(shap_interaction_values) == 2:
            shap_interaction_values = shap_interaction_values[1]
        else:
            shap_interaction_values = np.mean(shap_interaction_values, axis=0)

    # Check if we have a 4D array (n_samples, n_features, n_features, n_classes)
    if shap_interaction_values.ndim == 4:
        if shap_interaction_values.shape[-1] == 2:
            # Binary classification: use positive class (index 1)
            shap_interaction_values = shap_interaction_values[:, :, :, 1]
        else:
            # Multiclass: average absolute values across classes
            shap_interaction_values = np.mean(np.abs(shap_interaction_values), axis=-1)

    # Shape should now be: (n_samples, n_features, n_features)

    # Compute mean absolute interaction matrix
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)

    # Ensure 2D matrix (n_features, n_features)
    if interaction_matrix.ndim != 2:
        raise ValueError(
            f"Interaction matrix should be 2D but got shape {interaction_matrix.shape}. "
            f"Raw SHAP values shape: {shap_interaction_values.shape}"
        )

    # Extract top interactions (off-diagonal, upper triangle to avoid duplicates)
    interactions_list = []
    for i in range(n_features):
        for j in range(i + 1, n_features):  # Upper triangle only
            mean_interaction = float(interaction_matrix[i, j])
            interactions_list.append((feature_names[i], feature_names[j], mean_interaction))

    # Sort by absolute interaction strength descending
    interactions_list.sort(key=lambda x: abs(x[2]), reverse=True)

    # Limit to top K if requested
    if top_k is not None:
        interactions_list = interactions_list[:top_k]

    computation_time = time.time() - start_time

    return {
        "interaction_matrix": interaction_matrix,
        "feature_names": feature_names,
        "top_interactions": interactions_list,
        "n_features": n_features,
        "n_samples_used": n_samples_used,
        "computation_time": computation_time,
    }


def _generate_interaction_interpretation(
    top_interactions: list[tuple[str, str]],
    method_agreement: dict[tuple[str, str], float],
    warnings: list[str],
    n_consensus: int,
) -> str:
    """Generate human-readable interpretation of interaction analysis.

    Parameters
    ----------
    top_interactions : list[tuple[str, str]]
        Top feature pairs from consensus ranking
    method_agreement : dict[tuple[str, str], float]
        Pairwise correlations between method rankings
    warnings : list[str]
        List of potential issues detected
    n_consensus : int
        Number of interactions in top 10 across all methods

    Returns
    -------
    str
        Human-readable interpretation summary
    """
    lines = []

    # Consensus interactions
    if n_consensus > 0:
        lines.append(
            f"Strong consensus: {n_consensus} interactions rank in top 10 across all methods"
        )
        pairs_str = ", ".join([f"({a}, {b})" for a, b in top_interactions[:3]])
        lines.append(f"  Top consensus interactions: {pairs_str}")
    else:
        lines.append("Weak consensus: Different methods identify different important interactions")

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


def analyze_interactions(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, "NDArray[Any]"],
    y: Union[pl.Series, pd.Series, "NDArray[Any]"],
    feature_pairs: list[tuple[str, str]] | None = None,
    methods: list[str] | None = None,
    n_quantiles: int = 5,
    grid_resolution: int = 20,
    max_samples: int = 200,
) -> dict[str, Any]:
    """Comprehensive feature interaction analysis comparing multiple methods.

    **This is a TEAR SHEET function** - it runs multiple interaction detection methods
    and generates a comparison report with consensus ranking and interpretation.

    **Use Case**: "Which feature pairs interact in my model? Do different methods agree?"

    This function replaces 100+ lines of manual comparison code by providing
    integrated analysis showing:
    - Individual method results (Conditional IC, H-statistic, SHAP interactions)
    - Consensus ranking (interactions important across methods)
    - Method agreement/disagreement analysis
    - Auto-generated insights and warnings

    Parameters
    ----------
    model : Any
        Fitted model. Requirements vary by method:
        - Conditional IC: Not used (analyzes feature correlations)
        - H-statistic: Must have `predict()` method
        - SHAP: Must be compatible with TreeExplainer
    X : Union[pl.DataFrame, pd.DataFrame, np.ndarray]
        Feature matrix (n_samples, n_features)
    y : Union[pl.Series, pd.Series, np.ndarray]
        Target values (n_samples,)
    feature_pairs : list[tuple[str, str]] | None, default None
        Specific feature pairs to analyze. If None, tests all pairs.
    methods : list[str] | None, default ["conditional_ic", "h_statistic", "shap"]
        Which methods to run.
    n_quantiles : int, default 5
        Number of quantile bins for Conditional IC
    grid_resolution : int, default 20
        Grid size for partial dependence in H-statistic
    max_samples : int, default 200
        Maximum samples for SHAP and H-statistic

    Returns
    -------
    dict[str, Any]
        Comprehensive analysis results:
        - method_results: Dict of individual method outputs
        - consensus_ranking: Feature pairs ranked by average rank across methods
        - method_agreement: Spearman correlations between method rankings
        - top_interactions_consensus: Pairs in top 10 for ALL methods
        - warnings: Detected issues
        - interpretation: Auto-generated summary
        - methods_run: Methods successfully executed
        - methods_failed: Failed methods with error messages

    Raises
    ------
    ValueError
        If all methods fail or no methods specified
    """
    if methods is None:
        methods = ["conditional_ic", "h_statistic", "shap"]

    if not methods:
        raise ValueError("At least one method must be specified")

    # Extract feature names if not provided
    if isinstance(X, pl.DataFrame | pd.DataFrame):
        feature_names = list(X.columns)
    else:
        # Generate numeric feature names
        n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
        feature_names = [f"f{i}" for i in range(n_features)]

    # Determine feature pairs to analyze
    if feature_pairs is None:
        # Test all pairs
        n_features = len(feature_names)
        all_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                all_pairs.append((feature_names[i], feature_names[j]))
        feature_pairs = all_pairs
    else:
        # Validate provided pairs
        feature_set = set(feature_names)
        for pair in feature_pairs:
            if len(pair) != 2:
                raise ValueError(f"Feature pair must have exactly 2 elements: {pair}")
            if pair[0] not in feature_set or pair[1] not in feature_set:
                raise ValueError(
                    f"Feature pair contains unknown features: {pair}. Available features: {feature_names}"
                )

    # Run each method with try/except for optional dependencies and errors
    results = {}
    method_failures = []

    if "conditional_ic" in methods:
        try:
            # For Conditional IC, we need to run it for each pair
            ic_results: list[tuple[str, str, float | None]] = []
            for feat_a, feat_b in feature_pairs:
                # Extract columns
                x_a: pl.Series | pd.Series | NDArray[Any]
                x_b: pl.Series | pd.Series | NDArray[Any]
                if isinstance(X, pl.DataFrame):
                    x_a = X[feat_a]
                    x_b = X[feat_b]
                elif isinstance(X, pd.DataFrame):
                    x_a = X[feat_a]
                    x_b = X[feat_b]
                else:
                    # numpy array - need to find indices
                    idx_a = feature_names.index(feat_a)
                    idx_b = feature_names.index(feat_b)
                    X_arr = cast("NDArray[Any]", X)
                    x_a = X_arr[:, idx_a]
                    x_b = X_arr[:, idx_b]

                result = compute_conditional_ic(
                    feature_a=x_a,
                    feature_b=x_b,
                    forward_returns=y,
                    n_quantiles=n_quantiles,
                )

                # Extract interaction strength metric
                ic_range = result.get("ic_range", 0.0)
                ic_results.append((feat_a, feat_b, ic_range))

            # Sort by IC range descending
            ic_results.sort(key=lambda x: abs(x[2]) if x[2] is not None else 0.0, reverse=True)

            results["conditional_ic"] = {
                "top_interactions": ic_results,
                "n_pairs_tested": len(ic_results),
            }
        except Exception as e:
            method_failures.append(("conditional_ic", str(e)))

    if "h_statistic" in methods:
        try:
            # Convert feature pairs to indices for h_statistic
            pair_indices = []
            for feat_a, feat_b in feature_pairs:
                idx_a = feature_names.index(feat_a)
                idx_b = feature_names.index(feat_b)
                pair_indices.append((idx_a, idx_b))

            results["h_statistic"] = compute_h_statistic(
                model,
                X,
                feature_pairs=pair_indices,
                feature_names=feature_names,
                n_samples=max_samples,
                grid_resolution=grid_resolution,
            )
        except Exception as e:
            method_failures.append(("h_statistic", str(e)))

    if "shap" in methods:
        try:
            shap_result = compute_shap_interactions(
                model,
                X,
                feature_names=feature_names,
                max_samples=max_samples,
            )

            # Filter to requested pairs if feature_pairs was specified
            if feature_pairs is not None:
                pair_set = set(feature_pairs) | {(b, a) for a, b in feature_pairs}
                filtered_interactions = [
                    (a, b, score)
                    for a, b, score in shap_result["top_interactions"]
                    if (a, b) in pair_set or (b, a) in pair_set
                ]
                shap_result["top_interactions"] = filtered_interactions

            results["shap"] = shap_result
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
    rankings: dict[str, NDArray[Any]] = {}
    for method_name, result in results.items():
        # Get interaction scores for this method
        method_interactions: list[tuple[str, str, float]]
        if "top_interactions" in result:
            method_interactions = cast(list[tuple[str, str, float]], result["top_interactions"])
        elif "h_statistics" in result:
            method_interactions = cast(list[tuple[str, str, float]], result["h_statistics"])
        else:
            continue

        # Create a mapping from pair to rank
        pair_to_rank: dict[tuple[str, str], int] = {}
        for rank_idx, interaction_tuple in enumerate(method_interactions):
            feat_a_int, feat_b_int = str(interaction_tuple[0]), str(interaction_tuple[1])
            pair_key = (min(feat_a_int, feat_b_int), max(feat_a_int, feat_b_int))
            pair_to_rank[pair_key] = rank_idx

        # Map all requested pairs to ranks (handle missing pairs)
        ranks_array: list[int] = []
        for feat_a, feat_b in feature_pairs:
            pair_key = (min(feat_a, feat_b), max(feat_a, feat_b))
            rank_val = pair_to_rank.get(pair_key, len(method_interactions))
            ranks_array.append(rank_val)

        rankings[method_name] = np.array(ranks_array)

    # Average ranks across methods
    avg_ranks = np.mean(list(rankings.values()), axis=0)

    # Create consensus ranking with scores from each method
    consensus_ranking: list[tuple[str, str, float, dict[str, float]]] = []
    for idx, avg_rank in enumerate(avg_ranks):
        feat_a, feat_b = feature_pairs[idx]
        pair_tuple: tuple[str, str] = (min(feat_a, feat_b), max(feat_a, feat_b))

        # Collect scores from each method
        scores_dict: dict[str, float] = {}
        for method_name, result in results.items():
            method_ints: list[tuple[str, str, float]]
            if "top_interactions" in result:
                method_ints = cast(list[tuple[str, str, float]], result["top_interactions"])
            elif "h_statistics" in result:
                method_ints = cast(list[tuple[str, str, float]], result["h_statistics"])
            else:
                continue

            for int_tuple in method_ints:
                check_pair = (
                    min(str(int_tuple[0]), str(int_tuple[1])),
                    max(str(int_tuple[0]), str(int_tuple[1])),
                )
                if check_pair == pair_tuple:
                    scores_dict[method_name] = float(int_tuple[2])
                    break

        consensus_ranking.append((feat_a, feat_b, float(avg_rank), scores_dict))

    # Sort by average rank
    consensus_ranking.sort(key=lambda x: x[2])

    # 3. Compute method agreement (Spearman correlation between rankings)
    method_agreement = {}
    method_names = list(rankings.keys())
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1 :]:
            corr, _ = spearmanr(rankings[m1], rankings[m2])
            method_agreement[(m1, m2)] = float(corr)

    # 4. Identify consensus top interactions (top 10 in all methods)
    top_n = 10
    top_interactions_by_method: dict[str, set[tuple[str, str]]] = {}
    for method_name, result in results.items():
        method_ints_list: list[tuple[str, str, float]]
        if "top_interactions" in result:
            method_ints_list = cast(list[tuple[str, str, float]], result["top_interactions"])
        elif "h_statistics" in result:
            method_ints_list = cast(list[tuple[str, str, float]], result["h_statistics"])
        else:
            continue

        method_top_pairs: list[tuple[str, str]] = []
        for int_entry in method_ints_list[:top_n]:
            pair_sorted: tuple[str, str] = (
                min(str(int_entry[0]), str(int_entry[1])),
                max(str(int_entry[0]), str(int_entry[1])),
            )
            method_top_pairs.append(pair_sorted)
        top_interactions_by_method[method_name] = set(method_top_pairs)

    if top_interactions_by_method:
        consensus_top_pairs = set.intersection(*top_interactions_by_method.values())
    else:
        consensus_top_pairs = set()

    consensus_top_list = list(consensus_top_pairs)

    # 5. Generate warnings
    warnings = []

    # Warning: Disagreement between specific methods
    if "conditional_ic" in results and "h_statistic" in results:
        ic_interactions: list[tuple[str, str, float]]
        if "top_interactions" in results["conditional_ic"]:
            ic_interactions = cast(
                list[tuple[str, str, float]], results["conditional_ic"]["top_interactions"]
            )
        else:
            ic_interactions = []

        h_interactions: list[tuple[str, str, float]] = cast(
            list[tuple[str, str, float]], results["h_statistic"].get("h_statistics", [])
        )

        ic_top: set[tuple[str, str]] = {
            (min(str(x[0]), str(x[1])), max(str(x[0]), str(x[1]))) for x in ic_interactions[:5]
        }
        h_top: set[tuple[str, str]] = {
            (min(str(x[0]), str(x[1])), max(str(x[0]), str(x[1]))) for x in h_interactions[:5]
        }

        disagreement = ic_top - h_top
        if disagreement:
            pairs_str = ", ".join([f"({a}, {b})" for a, b in disagreement])
            warnings.append(
                f"Pairs {pairs_str} rank high in Conditional IC but not H-statistic - "
                "possible regime-specific interaction (time-varying)"
            )

    # Warning: Low agreement between methods
    if method_agreement:
        min_agreement = min(method_agreement.values())
        if min_agreement < 0.5:
            warnings.append(
                f"Low agreement between methods (min correlation: {min_agreement:.2f}) - "
                "results may be unreliable or methods capture different interaction types"
            )

    # Add method failures to warnings
    if method_failures:
        for method, error in method_failures:
            warnings.append(f"Method '{method}' failed: {error}")

    # 6. Generate interpretation
    top_pairs = [(a, b) for a, b, _, _ in consensus_ranking[:10]]
    interpretation = _generate_interaction_interpretation(
        top_pairs,
        method_agreement,
        warnings,
        len(consensus_top_list),
    )

    return {
        "method_results": results,
        "consensus_ranking": consensus_ranking,
        "method_agreement": method_agreement,
        "top_interactions_consensus": consensus_top_list,
        "warnings": warnings,
        "interpretation": interpretation,
        "methods_run": list(results.keys()),
        "methods_failed": method_failures,
    }
