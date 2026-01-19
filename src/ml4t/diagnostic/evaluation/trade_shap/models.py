"""Pydantic models for Trade SHAP diagnostics.

This module contains the data models used throughout the Trade SHAP analysis:
- TradeShapExplanation: SHAP explanation for a single trade
- ClusteringResult: Result of error pattern clustering
- ErrorPattern: Characterized error pattern from clustered trades
- TradeShapResult: Complete result of trade-level SHAP analysis
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class TradeExplainFailure(BaseModel):
    """Structured failure result for trade explanation.

    Used instead of exceptions for expected failure cases (alignment missing,
    feature mismatch, etc.) to enable batch processing without try/except.

    Attributes:
        trade_id: Unique trade identifier
        timestamp: Trade entry timestamp
        reason: Machine-readable failure reason code
        details: Additional context about the failure
    """

    trade_id: str = Field(..., description="Unique trade identifier")
    timestamp: datetime = Field(..., description="Trade entry timestamp")
    reason: str = Field(
        ...,
        description="Failure reason: 'alignment_missing', 'shap_error', 'feature_mismatch'",
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Additional failure context")


class TradeShapExplanation(BaseModel):
    """SHAP explanation for a single trade.

    Contains SHAP attribution details for one trade, including:
        - Top contributing features (sorted by absolute SHAP value)
        - Feature values at trade entry
        - Full SHAP vector for all features
        - Waterfall plot data (future enhancement)

    Attributes:
        trade_id: Unique trade identifier (symbol_timestamp)
        timestamp: Trade entry timestamp
        top_features: List of (feature_name, shap_value) sorted by |shap_value| descending
        feature_values: Dictionary of feature values at trade entry
        shap_vector: Full SHAP vector for all features (numpy array)

    Example:
        >>> explanation.top_features[:3]
        [('momentum_20d', 0.342), ('volatility_10d', -0.215), ('rsi_14d', 0.108)]

        >>> explanation.feature_values['momentum_20d']
        1.235

        >>> explanation.shap_vector.shape
        (50,)  # 50 features
    """

    trade_id: str = Field(..., description="Unique trade identifier")
    timestamp: datetime = Field(..., description="Trade entry timestamp")
    top_features: list[tuple[str, float]] = Field(
        ..., description="Top N features by absolute SHAP value (descending)"
    )
    feature_values: dict[str, float] = Field(
        ..., description="Feature values at trade entry timestamp"
    )
    shap_vector: NDArray[np.floating[Any]] = Field(
        ..., description="Full SHAP vector for all features"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ClusteringResult(BaseModel):
    """Result of error pattern clustering.

    Contains cluster assignments, centroids, quality metrics, and linkage matrix
    for dendrogram visualization.

    Attributes:
        n_clusters: Number of clusters identified
        cluster_assignments: Cluster ID for each trade (0-indexed list)
        linkage_matrix: Scipy linkage matrix for dendrogram plotting
        centroids: Mean SHAP vector for each cluster (shape: n_clusters x n_features)
        silhouette_score: Quality metric (range: -1 to 1, higher is better)
            - 1.0: Perfect separation
            - 0.5: Good separation
            - 0.0: Overlapping clusters
            - <0.0: Poor clustering (mis-assigned trades)
        davies_bouldin_score: Davies-Bouldin Index (lower = better, min: 0)
            - Measures ratio of within-cluster to between-cluster distances
            - < 1.0: Good clustering
            - 1.0-2.0: Acceptable clustering
            - > 2.0: Poor clustering
        calinski_harabasz_score: Calinski-Harabasz Score (higher = better, min: 0)
            - Also known as Variance Ratio Criterion
            - Measures ratio of between-cluster to within-cluster dispersion
            - Higher values indicate better-defined clusters
        cluster_sizes: Number of trades in each cluster
        distance_metric: Distance metric used ('euclidean', 'cosine', etc.)
        linkage_method: Linkage method used ('ward', 'average', 'complete', 'single')

    Example - Basic inspection:
        >>> result = analyzer.cluster_patterns(shap_vectors)
        >>> print(f"Found {result.n_clusters} clusters")
        >>> print(f"Cluster sizes: {result.cluster_sizes}")
        >>> print(f"Quality (silhouette): {result.silhouette_score:.3f}")

    Example - Visualize dendrogram:
        >>> from scipy.cluster.hierarchy import dendrogram
        >>> import matplotlib.pyplot as plt
        >>> dendrogram(result.linkage_matrix)
        >>> plt.title("Error Pattern Dendrogram")
        >>> plt.xlabel("Trade Index")
        >>> plt.ylabel("Distance")
        >>> plt.show()

    Example - Analyze specific cluster:
        >>> cluster_id = 0
        >>> trades_in_cluster = [i for i, c in enumerate(result.cluster_assignments) if c == cluster_id]
        >>> cluster_centroid = result.centroids[cluster_id]
        >>> print(f"Cluster {cluster_id}: {len(trades_in_cluster)} trades")
        >>> print(f"Centroid (mean SHAP): {cluster_centroid}")

    Note:
        - linkage_matrix can be used directly with scipy.cluster.hierarchy.dendrogram()
        - centroids represent "typical" SHAP pattern for each cluster
        - silhouette_score > 0.5 indicates well-separated clusters
    """

    n_clusters: int = Field(..., description="Number of clusters identified")
    cluster_assignments: list[int] = Field(..., description="Cluster ID for each trade (0-indexed)")
    linkage_matrix: NDArray[np.floating[Any]] = Field(
        ..., description="Scipy linkage matrix for dendrogram"
    )
    centroids: NDArray[np.floating[Any]] = Field(
        ..., description="Mean SHAP vector per cluster (n_clusters x n_features)"
    )
    silhouette_score: float = Field(
        ..., description="Cluster quality metric (range: -1 to 1, higher is better)"
    )
    davies_bouldin_score: float | None = Field(
        None,
        description="Davies-Bouldin Index (lower = better, min: 0, no upper bound). "
        "Measures ratio of within-cluster to between-cluster distances. "
        "Values < 1.0 indicate good clustering.",
    )
    calinski_harabasz_score: float | None = Field(
        None,
        description="Calinski-Harabasz Score (higher = better, min: 0, no upper bound). "
        "Also known as Variance Ratio Criterion. "
        "Measures ratio of between-cluster to within-cluster dispersion.",
    )
    cluster_sizes: list[int] = Field(..., description="Number of trades per cluster")
    distance_metric: str = Field(..., description="Distance metric used for clustering")
    linkage_method: str = Field(..., description="Linkage method used for clustering")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ErrorPattern(BaseModel):
    """Characterized error pattern from clustered trades.

    Represents a distinct pattern of trading errors identified through SHAP-based
    clustering and statistical characterization. Contains the defining features,
    quality metrics, and (optionally) generated hypotheses and action suggestions.

    Attributes:
        cluster_id: Unique identifier for this error pattern (0-indexed)
        n_trades: Number of trades exhibiting this pattern
        description: Human-readable pattern description
            Format: "High feature_X (up 0.45) + Low feature_Y (down -0.32) -> Losses"
        top_features: Top contributing SHAP features
            List of (feature_name, mean_shap, p_value_t, p_value_mw, is_significant)
        separation_score: Distance to nearest other cluster (higher = more distinct)
        distinctiveness: Ratio of max SHAP vs other clusters (higher = more unique)
        hypothesis: Optional generated hypothesis about why pattern causes losses
        actions: Optional list of suggested remediation actions
        confidence: Optional confidence score for hypothesis (0-1)

    Example - Basic pattern:
        >>> pattern = ErrorPattern(
        ...     cluster_id=0,
        ...     n_trades=15,
        ...     description="High momentum (up 0.45) + Low volatility (down -0.32) -> Losses",
        ...     top_features=[
        ...         ("momentum_20d", 0.45, 0.001, 0.002, True),
        ...         ("volatility_10d", -0.32, 0.003, 0.004, True)
        ...     ],
        ...     separation_score=1.2,
        ...     distinctiveness=1.8
        ... )
        >>> print(pattern.summary())
        "Pattern 0: 15 trades - High momentum (up 0.45) + Low volatility (down -0.32) -> Losses"

    Example - With hypothesis and actions:
        >>> pattern = ErrorPattern(
        ...     cluster_id=1,
        ...     n_trades=22,
        ...     description="High RSI (up 0.38) + High volume (up 0.29) -> Losses",
        ...     top_features=[("rsi_14", 0.38, 0.001, 0.001, True)],
        ...     separation_score=0.9,
        ...     distinctiveness=1.5,
        ...     hypothesis="Trades entering overbought conditions with high volume (potential reversals)",
        ...     actions=[
        ...         "Add overbought filter: skip trades when RSI > 70",
        ...         "Consider volume profile: avoid high volume in overbought zones",
        ...         "Add mean reversion features to capture reversal dynamics"
        ...     ],
        ...     confidence=0.85
        ... )
        >>> for action in pattern.actions:
        ...     print(f"  - {action}")

    Note:
        - hypothesis, actions, and confidence are populated by HypothesisGenerator
        - top_features are sorted by absolute SHAP value (descending)
        - separation_score and distinctiveness are quality metrics for pattern validation
    """

    cluster_id: int = Field(..., description="Cluster identifier (0-indexed)", ge=0)
    n_trades: int = Field(..., description="Number of trades in this pattern", gt=0)
    description: str = Field(..., description="Human-readable pattern description", min_length=1)
    top_features: list[tuple[str, float, float, float, bool]] = Field(
        ...,
        description="Top SHAP features: (name, mean_shap, p_value_t, p_value_mw, is_significant)",
    )
    separation_score: float = Field(
        ..., description="Distance to nearest other cluster (higher = better)", ge=0.0
    )
    distinctiveness: float = Field(
        ..., description="Ratio of max SHAP vs other clusters (higher = better)", gt=0.0
    )
    hypothesis: str | None = Field(
        None, description="Generated hypothesis about why this pattern causes losses"
    )
    actions: list[str] | None = Field(
        None, description="Suggested remediation actions for this pattern"
    )
    confidence: float | None = Field(
        None, description="Confidence score for hypothesis (0-1)", ge=0.0, le=1.0
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert ErrorPattern to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> pattern_dict = pattern.to_dict()
            >>> import json
            >>> json.dumps(pattern_dict, indent=2)
        """
        return {
            "cluster_id": self.cluster_id,
            "n_trades": self.n_trades,
            "description": self.description,
            "top_features": [
                {
                    "feature_name": feat[0],
                    "mean_shap": feat[1],
                    "p_value_t": feat[2],
                    "p_value_mw": feat[3],
                    "is_significant": feat[4],
                }
                for feat in self.top_features
            ],
            "separation_score": self.separation_score,
            "distinctiveness": self.distinctiveness,
            "hypothesis": self.hypothesis,
            "actions": self.actions if self.actions else [],
            "confidence": self.confidence,
        }

    def summary(self, include_actions: bool = False) -> str:
        """Generate human-readable summary of error pattern.

        Args:
            include_actions: Whether to include action suggestions in summary

        Returns:
            Formatted summary string

        Example:
            >>> print(pattern.summary())
            "Pattern 0: 15 trades - High momentum (up 0.45) + Low volatility (down -0.32) -> Losses"

            >>> print(pattern.summary(include_actions=True))
            '''
            Pattern 0: 15 trades
            Description: High momentum (up 0.45) + Low volatility (down -0.32) -> Losses
            Hypothesis: Trades entering overbought conditions
            Actions:
              - Add overbought filter: skip trades when RSI > 70
              - Consider volume profile
            Confidence: 85%
            '''
        """
        if not include_actions or not self.hypothesis:
            # Simple one-line summary
            return f"Pattern {self.cluster_id}: {self.n_trades} trades - {self.description}"

        # Detailed multi-line summary with hypothesis and actions
        lines = [
            f"Pattern {self.cluster_id}: {self.n_trades} trades",
            f"Description: {self.description}",
        ]

        if self.hypothesis:
            lines.append(f"Hypothesis: {self.hypothesis}")

        if self.actions:
            lines.append("Actions:")
            for action in self.actions:
                lines.append(f"  - {action}")

        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.0%}")

        return "\n".join(lines)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class TradeShapResult(BaseModel):
    """Complete result of trade-level SHAP analysis.

    Contains SHAP explanations for multiple trades, along with error patterns
    and actionable recommendations.

    Attributes:
        n_trades_analyzed: Total number of trades attempted to analyze
        n_trades_explained: Number of trades successfully explained
        n_trades_failed: Number of trades that failed explanation
        explanations: List of successful TradeShapExplanation objects
        failed_trades: List of (trade_id, error_message) tuples for failed trades
        error_patterns: Identified error patterns from clustering

    Example:
        >>> result = analyzer.explain_worst_trades(trades, n=20)
        >>> print(f"Success rate: {result.n_trades_explained}/{result.n_trades_analyzed}")
        >>> for explanation in result.explanations:
        ...     print(f"Trade {explanation.trade_id}: top feature = {explanation.top_features[0]}")
    """

    n_trades_analyzed: int = Field(..., description="Total trades analyzed")
    n_trades_explained: int = Field(..., description="Trades successfully explained")
    n_trades_failed: int = Field(..., description="Trades that failed explanation")
    explanations: list[TradeShapExplanation] = Field(
        default_factory=list, description="Successful SHAP explanations"
    )
    failed_trades: list[tuple[str, str]] = Field(
        default_factory=list, description="Failed trades: (trade_id, error_message)"
    )
    error_patterns: list[ErrorPattern] = Field(
        default_factory=list,
        description="Identified error patterns (populated by clustering and characterization)",
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
