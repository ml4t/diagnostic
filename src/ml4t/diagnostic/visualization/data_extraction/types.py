"""Type definitions for data extraction.

TypedDict classes for structured visualization data packages.
"""

from __future__ import annotations

from typing import Any, TypedDict


class MethodImportanceData(TypedDict, total=False):
    """Importance data for a single method."""

    importances: dict[str, float]  # feature_name -> importance_score
    ranking: list[str]  # Features sorted by importance
    std: dict[str, float] | None  # Standard deviation if available (PFI)
    confidence_intervals: dict[str, tuple[float, float]] | None  # 95% CI if available
    raw_values: list[dict[str, float]] | None  # Per-repeat values (PFI)
    metadata: dict[str, Any]  # Method-specific metadata


class FeatureDetailData(TypedDict):
    """Complete data for a single feature across all analyses."""

    consensus_rank: int  # Overall ranking
    consensus_score: float  # Consensus importance score
    method_ranks: dict[str, int]  # Method name -> rank in that method
    method_scores: dict[str, float]  # Method name -> importance score
    method_stds: dict[str, float]  # Method name -> std dev (if available)
    agreement_level: str  # 'high', 'medium', 'low'
    stability_score: float  # 0-1, higher = more stable
    interpretation: str  # Auto-generated interpretation


class MethodComparisonData(TypedDict):
    """Method agreement and comparison metrics."""

    correlation_matrix: list[list[float]]  # Method x Method correlation matrix
    correlation_methods: list[str]  # Method names for matrix axes
    rank_differences: dict[
        tuple[str, str], dict[str, int]
    ]  # (method1, method2) -> {feature: rank_diff}
    agreement_summary: dict[str, float]  # Pairwise correlations as dict


class UncertaintyData(TypedDict):
    """Uncertainty and stability metrics."""

    method_stability: dict[str, float]  # Method -> stability score (0-1)
    rank_stability: dict[str, list[int]]  # Feature -> list of ranks across bootstraps
    confidence_intervals: dict[str, dict[str, tuple[float, float]]]  # Method -> {feature: (lo, hi)}
    coefficient_of_variation: dict[str, dict[str, float]]  # Method -> {feature: CV}


class LLMContextData(TypedDict):
    """Structured data for LLM interpretation."""

    summary_narrative: str  # High-level summary in natural language
    key_insights: list[str]  # Bullet points of findings
    recommendations: list[str]  # Actionable recommendations
    caveats: list[str]  # Limitations and warnings
    analysis_quality: str  # 'high', 'medium', 'low'


class ImportanceVizData(TypedDict):
    """Complete visualization data package for importance analysis."""

    summary: dict[str, Any]  # High-level metrics
    per_method: dict[str, MethodImportanceData]  # Method name -> detailed data
    per_feature: dict[str, FeatureDetailData]  # Feature name -> aggregated view
    uncertainty: UncertaintyData  # Stability and confidence metrics
    method_comparison: MethodComparisonData  # Cross-method analysis
    metadata: dict[str, Any]  # Context information
    llm_context: LLMContextData  # LLM-friendly narratives


class FeatureInteractionData(TypedDict):
    """Interaction data for a single feature."""

    feature_name: str
    top_partners: list[tuple[str, float]]  # (partner_feature, interaction_strength)
    total_interaction_strength: float  # Sum of absolute interactions
    cluster_id: int | None  # ID of interaction cluster (if clustering performed)
    interpretation: str  # Auto-generated interpretation


class NetworkGraphData(TypedDict):
    """Network graph representation of interactions."""

    nodes: list[dict[str, Any]]  # [{id: str, label: str, importance: float, ...}]
    edges: list[dict[str, Any]]  # [{source: str, target: str, weight: float, ...}]
    clusters: list[list[str]]  # List of feature clusters based on interactions


class InteractionMatrixData(TypedDict):
    """Matrix representation of pairwise interactions."""

    features: list[str]  # Ordered feature names
    matrix: list[list[float]]  # Symmetric interaction matrix
    max_interaction: float  # Maximum interaction value
    mean_interaction: float  # Mean interaction strength


class InteractionVizData(TypedDict):
    """Complete visualization data package for interaction analysis."""

    summary: dict[str, Any]  # High-level metrics
    per_feature: dict[str, FeatureInteractionData]  # Feature -> interaction details
    network_graph: NetworkGraphData  # Graph visualization data
    interaction_matrix: InteractionMatrixData  # Matrix visualization data
    strength_distribution: dict[str, Any]  # Distribution of interaction strengths
    metadata: dict[str, Any]  # Context information
    llm_context: LLMContextData  # LLM-friendly narratives
