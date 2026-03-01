"""Trade SHAP analysis pipeline.

This module provides the TradeShapPipeline class that orchestrates
all components of trade SHAP analysis:
- TradeShapExplainer for individual trade explanations
- HierarchicalClusterer for error pattern clustering
- PatternCharacterizer for statistical characterization
- HypothesisGenerator for actionable insights
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ml4t.diagnostic.config import TradeConfig
from ml4t.diagnostic.evaluation.trade_shap.characterize import (
    CharacterizationConfig,
    PatternCharacterizer,
)
from ml4t.diagnostic.evaluation.trade_shap.cluster import (
    ClusteringConfig,
    HierarchicalClusterer,
)
from ml4t.diagnostic.evaluation.trade_shap.explain import TradeShapExplainer
from ml4t.diagnostic.evaluation.trade_shap.hypotheses import (
    HypothesisConfig,
    HypothesisGenerator,
)
from ml4t.diagnostic.evaluation.trade_shap.models import (
    TradeExplainFailure,
    TradeShapExplanation,
    TradeShapResult,
)
from ml4t.diagnostic.evaluation.trade_shap.normalize import normalize

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

    from ml4t.diagnostic.evaluation.trade_analysis import TradeMetrics


class TradeShapPipeline:
    """Orchestrates trade SHAP analysis components.

    This is the main entry point for trade SHAP analysis, providing a clean
    interface that uses the refactored components internally.

    Attributes:
        features_df: Polars DataFrame with timestamp and feature columns
        shap_values: SHAP values array (n_samples x n_features)
        feature_names: List of feature column names
        config: Pipeline configuration

    Example:
        >>> pipeline = TradeShapPipeline(
        ...     features_df=features,
        ...     shap_values=shap_values,
        ...     feature_names=feature_names,
        ... )
        >>> result = pipeline.analyze_worst_trades(trades, n=20)
        >>> for pattern in result.error_patterns:
        ...     print(pattern.hypothesis)
        ...     print(pattern.actions)
    """

    def __init__(
        self,
        features_df: pl.DataFrame,
        shap_values: NDArray[np.floating[Any]],
        feature_names: list[str],
        config: TradeConfig | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            features_df: Polars DataFrame with 'timestamp' column and feature columns
            shap_values: SHAP values array (n_samples x n_features)
            feature_names: List of feature column names
            config: Pipeline configuration (uses defaults if None)
        """
        self.features_df = features_df
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.config = config or TradeConfig()

        # Initialize explainer
        alignment_cfg = self.config.alignment
        self.explainer = TradeShapExplainer(
            features_df=features_df,
            shap_values=shap_values,
            feature_names=feature_names,
            tolerance_seconds=float(alignment_cfg.tolerance),
            top_n_features=alignment_cfg.top_n_features,
            alignment_mode=alignment_cfg.mode,
            missing_value_strategy=alignment_cfg.missing_strategy,
        )

        # Initialize clusterer
        self.clusterer = HierarchicalClusterer(
            config=ClusteringConfig(
                distance_metric=self.config.clustering.distance_metric,
                linkage_method=self.config.clustering.linkage,
                min_cluster_size=self.config.clustering.min_cluster_size,
                min_trades_for_clustering=self.config.min_trades_for_clustering,
            )
        )

        # Initialize characterizer
        self.characterizer = PatternCharacterizer(
            feature_names=feature_names,
            config=CharacterizationConfig(),
        )

        # Initialize hypothesis generator
        self.hypothesis_generator = HypothesisGenerator(
            config=HypothesisConfig(
                template_library=self.config.hypothesis.template_library,
                min_confidence=self.config.hypothesis.min_confidence,
                max_actions=self.config.hypothesis.max_per_cluster,
            )
        )

    def explain_trade(
        self,
        trade: TradeMetrics,
    ) -> TradeShapExplanation | TradeExplainFailure:
        """Explain a single trade.

        Args:
            trade: Trade to explain

        Returns:
            TradeShapExplanation on success, TradeExplainFailure on failure
        """
        return self.explainer.explain(trade)

    def explain_trades(
        self,
        trades: list[TradeMetrics],
    ) -> tuple[list[TradeShapExplanation], list[TradeExplainFailure]]:
        """Explain multiple trades.

        Args:
            trades: List of trades to explain

        Returns:
            Tuple of (successful explanations, failures)
        """
        return self.explainer.explain_many(trades)

    def analyze_worst_trades(
        self,
        trades: list[TradeMetrics],
        n: int | None = None,
    ) -> TradeShapResult:
        """Analyze worst trades with full pipeline.

        This is the main entry point that:
        1. Explains each trade
        2. Clusters the SHAP vectors
        3. Characterizes each cluster as an error pattern
        4. Generates hypotheses for each pattern

        Args:
            trades: List of trades (should be sorted by loss, worst first)
            n: Number of trades to analyze (defaults to all)

        Returns:
            TradeShapResult with explanations, error patterns, and insights
        """
        # Limit trades
        trades_to_analyze = trades[:n] if n is not None else trades

        # Step 1: Explain trades
        explanations, failures = self.explain_trades(trades_to_analyze)

        if not explanations:
            # No successful explanations
            return TradeShapResult(
                n_trades_analyzed=len(trades_to_analyze),
                n_trades_explained=0,
                n_trades_failed=len(failures),
                explanations=[],
                failed_trades=[(f.trade_id, f.reason) for f in failures],
                error_patterns=[],
            )

        # Step 2: Extract and normalize SHAP vectors for clustering
        shap_vectors = np.array([exp.shap_vector for exp in explanations])

        # Normalize if configured
        if self.config.clustering.normalization:
            shap_vectors = normalize(shap_vectors, method=self.config.clustering.normalization)

        # Step 3: Cluster patterns (if enough trades)
        error_patterns = []
        min_trades = self.config.min_trades_for_clustering

        if len(explanations) >= min_trades:
            try:
                clustering_result = self.clusterer.cluster(shap_vectors)

                # Step 4: Characterize each cluster
                patterns = self.characterizer.characterize_all_clusters(
                    shap_vectors=shap_vectors,
                    cluster_labels=clustering_result.cluster_assignments,
                    n_clusters=clustering_result.n_clusters,
                    centroids=clustering_result.centroids,
                )

                # Step 5: Generate hypotheses for each pattern
                for pattern in patterns:
                    enriched = self.hypothesis_generator.generate_hypothesis(pattern)
                    error_patterns.append(enriched)

            except ValueError:
                # Clustering failed (e.g., insufficient samples)
                # Continue without error patterns
                pass

        return TradeShapResult(
            n_trades_analyzed=len(trades_to_analyze),
            n_trades_explained=len(explanations),
            n_trades_failed=len(failures),
            explanations=explanations,
            failed_trades=[(f.trade_id, f.reason) for f in failures],
            error_patterns=error_patterns,
        )

    def generate_actions(
        self,
        pattern_index: int = 0,
        max_actions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate prioritized actions for an error pattern.

        Args:
            pattern_index: Index of pattern in last result (default: 0 = first)
            max_actions: Maximum actions to return

        Returns:
            List of action dictionaries

        Note:
            Must call analyze_worst_trades first.
        """
        # This is a convenience method - in practice, use the hypothesis generator
        # directly with the error pattern from results
        raise NotImplementedError("Use hypothesis_generator.generate_actions(pattern) directly")
