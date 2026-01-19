"""Interaction data extraction for visualization layer.

Extracts comprehensive visualization data from feature interaction analysis results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np

from .types import (
    FeatureInteractionData,
    InteractionMatrixData,
    InteractionVizData,
    LLMContextData,
    NetworkGraphData,
)
from .validation import _validate_matrix_feature_alignment


def extract_interaction_viz_data(
    interaction_results: dict[str, Any],
    importance_results: dict[str, Any] | None = None,
    n_top_partners: int = 5,
    cluster_threshold: float = 0.3,
    include_llm_context: bool = True,
) -> InteractionVizData:
    """Extract comprehensive visualization data from interaction analysis results.

    This function transforms raw SHAP interaction results into structured data
    optimized for rich interactive visualization, including per-feature summaries,
    network graph data, interaction matrices, and auto-generated insights.

    Parameters
    ----------
    interaction_results : dict
        Results from compute_shap_interactions() containing:
        - 'interaction_matrix': DataFrame with pairwise interactions
        - 'feature_names': list of feature names
        - 'shap_values': raw SHAP values (optional)
        - 'shap_interaction_values': raw interaction values (optional)
    importance_results : dict, optional
        Optional importance results to cross-reference for node sizing.
        If provided, will use consensus ranking to size network nodes.
    n_top_partners : int, default=5
        Number of top interaction partners to include per feature.
    cluster_threshold : float, default=0.3
        Minimum interaction strength to consider for clustering.
        Features with interactions above this threshold are clustered.
    include_llm_context : bool, default=True
        Whether to generate auto-narratives for LLM consumption.

    Returns
    -------
    InteractionVizData
        Complete structured data package with:
        - Per-feature interaction summaries
        - Network graph data (nodes, edges, clusters)
        - Interaction matrix data
        - Strength distribution statistics
        - Auto-generated LLM narratives

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization.data_extraction import extract_interaction_viz_data
    >>>
    >>> # Compute interactions
    >>> interaction_results = compute_shap_interactions(model, X, y)
    >>>
    >>> # Extract visualization data
    >>> viz_data = extract_interaction_viz_data(interaction_results)
    >>>
    >>> # Access different views
    >>> print(viz_data['summary']['strongest_interaction'])
    >>> print(viz_data['per_feature']['momentum']['top_partners'][:3])
    >>> print(viz_data['network_graph']['nodes'])
    >>> print(viz_data['llm_context']['key_insights'])

    Notes
    -----
    - Network graph data is pre-computed for custom rendering
    - Clustering identifies groups of strongly interacting features
    - Per-feature summaries enable drill-down dashboards
    - Cross-referencing with importance results enables better node sizing
    """
    # Extract basic info
    interaction_matrix_df = interaction_results.get("interaction_matrix")
    feature_names = interaction_results.get("feature_names", [])

    if interaction_matrix_df is None:
        raise ValueError("interaction_results must contain 'interaction_matrix'")

    # Convert to numpy for easier manipulation
    if hasattr(interaction_matrix_df, "to_numpy"):
        interaction_matrix = interaction_matrix_df.to_numpy()
    else:
        interaction_matrix = np.array(interaction_matrix_df)

    # Validate matrix dimensions match feature names
    _validate_matrix_feature_alignment(interaction_matrix, feature_names)

    n_features = len(feature_names)

    # Build summary statistics
    summary = _build_interaction_summary(interaction_matrix, feature_names)

    # Build per-feature interaction data
    per_feature = _build_per_feature_interactions(interaction_matrix, feature_names, n_top_partners)

    # Build network graph data
    network_graph = _build_network_graph(
        interaction_matrix, feature_names, importance_results, cluster_threshold
    )

    # Build matrix data
    matrix_data = _build_interaction_matrix_data(interaction_matrix, feature_names)

    # Build strength distribution
    strength_distribution = _build_strength_distribution(interaction_matrix)

    # Build metadata
    metadata = {
        "n_features": n_features,
        "n_interactions": int(n_features * (n_features - 1) / 2),
        "analysis_timestamp": datetime.now().isoformat(),
        "cluster_threshold": cluster_threshold,
        "n_top_partners": n_top_partners,
    }

    # Generate LLM context
    llm_context: LLMContextData = {
        "summary_narrative": "",
        "key_insights": [],
        "recommendations": [],
        "caveats": [],
        "analysis_quality": "medium",
    }
    if include_llm_context:
        llm_context = _generate_interaction_llm_context(
            summary, per_feature, network_graph, strength_distribution
        )

    return InteractionVizData(
        summary=summary,
        per_feature=per_feature,
        network_graph=network_graph,
        interaction_matrix=matrix_data,
        strength_distribution=strength_distribution,
        metadata=metadata,
        llm_context=llm_context,
    )


# =============================================================================
# Interaction Analysis Helpers
# =============================================================================


def _build_interaction_summary(
    interaction_matrix: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    """Build high-level summary statistics for interactions."""
    n_features = len(feature_names)

    # Get upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(n_features, k=1)
    interaction_values = interaction_matrix[triu_indices]

    # Find strongest interaction
    abs_values = np.abs(interaction_values)
    max_idx = np.argmax(abs_values)
    max_interaction = float(interaction_values[max_idx])

    # Get feature pair for strongest interaction
    i, j = triu_indices[0][max_idx], triu_indices[1][max_idx]
    strongest_pair = (feature_names[i], feature_names[j])

    # Compute distribution statistics
    mean_interaction = float(np.mean(abs_values))
    median_interaction = float(np.median(abs_values))
    std_interaction = float(np.std(abs_values))

    # Identify features with strongest overall interactions
    total_interactions = np.sum(np.abs(interaction_matrix), axis=1)
    top_idx = np.argmax(total_interactions)
    most_interactive_feature = feature_names[top_idx]

    return {
        "n_features": n_features,
        "n_interactions": len(interaction_values),
        "strongest_interaction": max_interaction,
        "strongest_pair": strongest_pair,
        "mean_interaction": mean_interaction,
        "median_interaction": median_interaction,
        "std_interaction": std_interaction,
        "most_interactive_feature": most_interactive_feature,
        "max_total_interaction": float(total_interactions[top_idx]),
    }


def _build_per_feature_interactions(
    interaction_matrix: np.ndarray, feature_names: list[str], n_top_partners: int = 5
) -> dict[str, FeatureInteractionData]:
    """Build per-feature interaction summaries."""
    per_feature: dict[str, FeatureInteractionData] = {}
    n_features = len(feature_names)

    for i, feature_name in enumerate(feature_names):
        # Get all interactions for this feature
        interactions = interaction_matrix[i, :]

        # Exclude self-interaction
        partner_indices = [j for j in range(n_features) if j != i]
        partner_interactions = [(feature_names[j], float(interactions[j])) for j in partner_indices]

        # Sort by absolute interaction strength
        partner_interactions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Get top N partners
        top_partners = partner_interactions[:n_top_partners]

        # Total interaction strength
        total_strength = float(np.sum(np.abs(interactions)))

        # Generate interpretation
        interpretation = _generate_interaction_interpretation(feature_name, top_partners)

        per_feature[feature_name] = FeatureInteractionData(
            feature_name=feature_name,
            top_partners=top_partners,
            total_interaction_strength=total_strength,
            cluster_id=None,  # Will be filled by clustering
            interpretation=interpretation,
        )

    return per_feature


def _build_network_graph(
    interaction_matrix: np.ndarray,
    feature_names: list[str],
    importance_results: dict[str, Any] | None,
    cluster_threshold: float,
) -> NetworkGraphData:
    """Build network graph data (nodes, edges, clusters)."""
    n_features = len(feature_names)

    # Build nodes
    nodes = []
    for i, feature_name in enumerate(feature_names):
        # Node importance (for sizing) - use importance if available
        if importance_results and "consensus_ranking" in importance_results:
            consensus_ranking = importance_results["consensus_ranking"]
            if feature_name in consensus_ranking:
                rank = consensus_ranking.index(feature_name) + 1
                # Higher rank = smaller number = more important = larger node
                node_importance = 1.0 / rank
            else:
                node_importance = 0.1
        else:
            # Use total interaction strength as proxy
            node_importance = float(np.sum(np.abs(interaction_matrix[i, :])))

        nodes.append(
            {
                "id": feature_name,
                "label": feature_name,
                "importance": node_importance,
                "total_interaction": float(np.sum(np.abs(interaction_matrix[i, :]))),
            }
        )

    # Build edges (only upper triangle to avoid duplicates)
    edges = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction_value = float(interaction_matrix[i, j])
            if abs(interaction_value) > 0:  # Include all non-zero interactions
                edges.append(
                    {
                        "source": feature_names[i],
                        "target": feature_names[j],
                        "weight": interaction_value,
                        "abs_weight": abs(interaction_value),
                    }
                )

    # Sort edges by absolute weight
    edges.sort(key=lambda e: cast(float, e["abs_weight"]), reverse=True)

    # Perform simple clustering based on strong interactions
    clusters = _detect_interaction_clusters(interaction_matrix, feature_names, cluster_threshold)

    return NetworkGraphData(nodes=nodes, edges=edges, clusters=clusters)


def _build_interaction_matrix_data(
    interaction_matrix: np.ndarray, feature_names: list[str]
) -> InteractionMatrixData:
    """Build matrix data for heatmap visualization."""
    # Convert to list of lists for JSON serialization
    matrix_list = interaction_matrix.tolist()

    # Compute statistics
    triu_indices = np.triu_indices(len(feature_names), k=1)
    interaction_values = interaction_matrix[triu_indices]

    max_interaction = float(np.max(np.abs(interaction_values)))
    mean_interaction = float(np.mean(np.abs(interaction_values)))

    return InteractionMatrixData(
        features=feature_names,
        matrix=matrix_list,
        max_interaction=max_interaction,
        mean_interaction=mean_interaction,
    )


def _build_strength_distribution(interaction_matrix: np.ndarray) -> dict[str, Any]:
    """Build distribution statistics for interaction strengths."""
    n_features = interaction_matrix.shape[0]
    triu_indices = np.triu_indices(n_features, k=1)
    interaction_values = interaction_matrix[triu_indices]
    abs_values = np.abs(interaction_values)

    # Compute percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = {f"p{p}": float(np.percentile(abs_values, p)) for p in percentiles}

    # Binning for histogram
    hist, bin_edges = np.histogram(abs_values, bins=20)

    return {
        "mean": float(np.mean(abs_values)),
        "median": float(np.median(abs_values)),
        "std": float(np.std(abs_values)),
        "min": float(np.min(abs_values)),
        "max": float(np.max(abs_values)),
        "percentiles": percentile_values,
        "histogram": {"counts": hist.tolist(), "bin_edges": bin_edges.tolist()},
    }


def _detect_interaction_clusters(
    interaction_matrix: np.ndarray, feature_names: list[str], threshold: float
) -> list[list[str]]:
    """Detect clusters of strongly interacting features using simple thresholding.

    This is a basic clustering approach based on connected components in the
    interaction graph. More sophisticated methods could be added later.
    """
    n_features = len(feature_names)

    # Create adjacency matrix based on threshold
    adj_matrix = np.abs(interaction_matrix) > threshold
    np.fill_diagonal(adj_matrix, False)  # No self-loops

    # Find connected components (simple DFS)
    visited = [False] * n_features
    clusters = []

    def dfs(node: int, cluster: list[int]) -> None:
        visited[node] = True
        cluster.append(node)
        for neighbor in range(n_features):
            if adj_matrix[node, neighbor] and not visited[neighbor]:
                dfs(neighbor, cluster)

    for i in range(n_features):
        if not visited[i]:
            cluster_indices: list[int] = []
            dfs(i, cluster_indices)
            if len(cluster_indices) > 1:  # Only include clusters with >1 feature
                clusters.append([feature_names[idx] for idx in cluster_indices])

    return clusters


def _generate_interaction_interpretation(
    feature_name: str, top_partners: list[tuple[str, float]]
) -> str:
    """Generate auto-interpretation for a single feature's interactions."""
    if not top_partners:
        return f"'{feature_name}' has no significant interactions."

    # Get top 3 for narrative
    top_3 = top_partners[:3]
    partner_str = ", ".join([f"'{p[0]}' ({p[1]:.3f})" for p in top_3])

    return (
        f"'{feature_name}' shows strongest interactions with {partner_str}. "
        f"These interaction effects suggest the feature's predictive power "
        f"depends on the values of these partner features."
    )


def _generate_interaction_llm_context(
    summary: dict[str, Any],
    _per_feature: dict[str, FeatureInteractionData],
    network_graph: NetworkGraphData,
    strength_distribution: dict[str, Any],
) -> LLMContextData:
    """Generate auto-narratives for interaction analysis."""
    n_features = summary["n_features"]
    n_interactions = summary["n_interactions"]
    strongest_pair = summary["strongest_pair"]
    strongest_value = summary["strongest_interaction"]
    most_interactive = summary["most_interactive_feature"]

    # Build summary narrative
    summary_narrative = (
        f"This interaction analysis examined {n_features} features, identifying "
        f"{n_interactions} pairwise interactions. "
    )

    summary_narrative += (
        f"The strongest interaction ({strongest_value:.3f}) occurs between "
        f"'{strongest_pair[0]}' and '{strongest_pair[1]}'. "
    )

    if network_graph["clusters"]:
        n_clusters = len(network_graph["clusters"])
        summary_narrative += (
            f"Cluster analysis identified {n_clusters} group(s) of strongly interacting features. "
        )

    # Key insights
    key_insights = []

    # Insight 1: Strongest interaction
    key_insights.append(
        f"Strongest interaction: {strongest_pair[0]} <-> {strongest_pair[1]} (strength: {strongest_value:.3f})"
    )

    # Insight 2: Most interactive feature
    key_insights.append(
        f"Most interactive feature: '{most_interactive}' (total interaction: {summary['max_total_interaction']:.3f})"
    )

    # Insight 3: Distribution characteristics
    mean_strength = strength_distribution["mean"]
    median_strength = strength_distribution["median"]
    if mean_strength > median_strength * 1.5:
        key_insights.append(
            f"Interaction strength distribution is right-skewed "
            f"(mean: {mean_strength:.3f}, median: {median_strength:.3f}) - "
            "a few strong interactions dominate"
        )

    # Insight 4: Clustering
    if network_graph["clusters"]:
        largest_cluster = list(max(network_graph["clusters"], key=len))  # type: ignore[arg-type]
        key_insights.append(
            f"Largest interaction cluster has {len(largest_cluster)} features: "
            f"{', '.join(largest_cluster[:5])}" + ("..." if len(largest_cluster) > 5 else "")
        )

    # Recommendations
    recommendations = []

    # Rec 1: Focus on strong interactions
    recommendations.append(
        f"Investigate the {strongest_pair[0]}/{strongest_pair[1]} interaction further. "
        "Strong interactions suggest conditional effects or non-linear relationships."
    )

    # Rec 2: Feature engineering
    if network_graph["clusters"]:
        recommendations.append(
            "Consider creating interaction features (products, ratios) for clustered "
            "feature groups to capture non-linear effects explicitly."
        )

    # Rec 3: Model selection
    recommendations.append(
        "Tree-based models and neural networks can capture these interactions naturally. "
        "Linear models may benefit from explicit interaction terms."
    )

    # Caveats
    caveats = [
        "SHAP interactions measure feature contribution interactions, not statistical "
        "correlations. High interaction doesn't imply high correlation.",
        "Interaction values are model-specific and depend on the underlying model structure.",
    ]

    # Determine quality
    if n_features >= 5 and summary["max_total_interaction"] > 0.1:
        analysis_quality = "high"
    elif n_features >= 3:
        analysis_quality = "medium"
    else:
        analysis_quality = "low"

    return LLMContextData(
        summary_narrative=summary_narrative,
        key_insights=key_insights,
        recommendations=recommendations,
        caveats=caveats,
        analysis_quality=analysis_quality,
    )
