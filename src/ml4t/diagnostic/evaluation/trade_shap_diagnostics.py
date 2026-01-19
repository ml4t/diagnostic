"""Trade-level SHAP diagnostics for ML trading feedback loop.

Connects SHAP values to trade outcomes for systematic debugging and improvement.

This module is a thin wrapper around the modular trade_shap package.
Implementation has been refactored into:
- ml4t.diagnostic.evaluation.trade_shap.models (data models)
- ml4t.diagnostic.evaluation.trade_shap.pipeline (TradeShapPipeline)
- ml4t.diagnostic.evaluation.trade_shap.explain (TradeShapExplainer)
- ml4t.diagnostic.evaluation.trade_shap.cluster (HierarchicalClusterer)
- ml4t.diagnostic.evaluation.trade_shap.characterize (PatternCharacterizer)
- ml4t.diagnostic.evaluation.trade_shap.hypotheses (HypothesisGenerator)

Example:
    >>> analyzer = TradeShapAnalyzer(model, features_df, shap_values)
    >>> result = analyzer.explain_worst_trades(worst_trades)
    >>> for pattern in result.error_patterns:
    ...     print(pattern.hypothesis, pattern.actions)

See: docs/trimmed/evaluation/trade_shap_diagnostics.md for full documentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

# Re-export all models and components from modular package
from ml4t.diagnostic.evaluation.trade_shap import (
    # Alignment
    AlignmentResult,
    # Characterization
    CharacterizationConfig,
    # Clustering
    ClusteringConfig,
    ClusteringResult,
    # Result models
    ErrorPattern,
    FeatureStatistics,
    HierarchicalClusterer,
    # Hypothesis generation
    HypothesisConfig,
    HypothesisGenerator,
    # Normalization
    NormalizationType,
    PatternCharacterizer,
    Template,
    TemplateMatcher,
    TimestampAligner,
    TradeExplainFailure,
    # Explainer
    TradeShapExplainer,
    TradeShapExplanation,
    # Pipeline
    TradeShapPipeline,
    TradeShapPipelineConfig,
    TradeShapResult,
    benjamini_hochberg,
    compute_centroids,
    compute_cluster_sizes,
    find_optimal_clusters,
    load_templates,
    normalize,
    normalize_l1,
    normalize_l2,
    standardize,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.config import TradeConfig
    from ml4t.diagnostic.evaluation.trade_analysis import TradeMetrics


class TradeShapAnalyzer:
    """Analyze trade failures using SHAP explanations.

    This class wraps TradeShapPipeline with additional features:
    - On-demand SHAP value computation from a model
    - Pandas/Polars DataFrame conversion
    - GPU acceleration support

    For simpler use cases with pre-computed SHAP values, use TradeShapPipeline
    directly.

    Example:
        >>> analyzer = TradeShapAnalyzer(model, features_df, shap_values)
        >>> result = analyzer.explain_worst_trades(worst_trades)
    """

    def __init__(
        self,
        model: Any,
        features_df: pl.DataFrame | Any,
        shap_values: NDArray[np.floating[Any]] | None = None,
        config: TradeConfig | None = None,
        explainer_type: str = "auto",
        use_gpu: bool | str = "auto",
        background_data: NDArray[Any] | None = None,
        explainer_kwargs: dict | None = None,
        show_progress: bool = False,
        performance_warning: bool = True,
    ):
        """Initialize with model, features DataFrame, and optional SHAP values.

        Args:
            model: Trained model for SHAP computation
            features_df: DataFrame with 'timestamp' column and feature columns
            shap_values: Pre-computed SHAP values (optional, computed if None)
            config: TradeConfig for analysis parameters
            explainer_type: SHAP explainer type ('auto', 'tree', 'kernel', etc.)
            use_gpu: Whether to use GPU acceleration
            background_data: Background data for SHAP computation
            explainer_kwargs: Additional kwargs for SHAP explainer
            show_progress: Show progress bars during computation
            performance_warning: Warn about performance issues
        """
        self.model = model
        self.features_df = self._validate_and_convert_features(features_df)
        self.shap_values = shap_values
        self.config = config or self._get_default_config()

        # Store API parameters for on-demand SHAP computation
        self._explainer_type = explainer_type
        self._use_gpu = use_gpu
        self._background_data = background_data
        self._explainer_kwargs = explainer_kwargs or {}
        self._show_progress = show_progress
        self._performance_warning = performance_warning

        # Extract feature names
        self.feature_names = self._extract_feature_names()

        # Validate SHAP values if provided
        if self.shap_values is not None:
            self._validate_shap_values()

        # Pipeline created lazily after SHAP values are available
        self._pipeline: TradeShapPipeline | None = None
        self._hypothesis_generator: HypothesisGenerator | None = None

    def _validate_and_convert_features(self, features_df: Any) -> pl.DataFrame:
        """Validate and convert features DataFrame to Polars."""
        if not isinstance(features_df, pl.DataFrame):
            import pandas as pd

            if isinstance(features_df, pd.DataFrame):
                features_df = pl.from_pandas(features_df)
            else:
                raise TypeError(
                    f"features_df must be pl.DataFrame or pd.DataFrame, got {type(features_df)}"
                )

        if "timestamp" not in features_df.columns:
            raise ValueError(
                "features_df must have 'timestamp' column for SHAP alignment to trades."
            )

        return features_df

    def _extract_feature_names(self) -> list[str]:
        """Extract feature names from DataFrame."""
        feature_names = [col for col in self.features_df.columns if col != "timestamp"]
        if not feature_names:
            raise ValueError("No feature columns found in features_df.")
        return feature_names

    def _validate_shap_values(self) -> None:
        """Validate SHAP values shape matches features."""
        if self.shap_values is None:
            return

        n_samples = len(self.features_df)
        n_features = len(self.feature_names)

        if self.shap_values.shape != (n_samples, n_features):
            raise ValueError(
                f"SHAP values shape {self.shap_values.shape} doesn't match "
                f"features_df shape ({n_samples}, {n_features})."
            )

    def _get_default_config(self) -> TradeConfig:
        """Get default configuration."""
        from ml4t.diagnostic.config import TradeConfig

        return TradeConfig()

    def _compute_shap_values(self) -> None:
        """Compute SHAP values on-demand if not provided."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        feature_cols = [col for col in self.features_df.columns if col != "timestamp"]
        features_df = self.features_df.select(feature_cols)

        result = compute_shap_importance(
            model=self.model,
            X=features_df,
            feature_names=feature_cols,
            explainer_type=self._explainer_type,
            use_gpu=self._use_gpu,
            background_data=self._background_data,
            show_progress=self._show_progress,
            explainer_kwargs=self._explainer_kwargs,
        )

        self.shap_values = result["shap_values"]

    def _ensure_pipeline(self) -> TradeShapPipeline:
        """Ensure pipeline is initialized with SHAP values."""
        if self._pipeline is None:
            # Compute SHAP values if not provided
            if self.shap_values is None:
                self._compute_shap_values()

            # Build pipeline config from TradeConfig
            # Check for nested alignment config
            alignment_cfg = getattr(self.config, "alignment", None)
            if alignment_cfg is not None:
                # AlignmentSettings has: tolerance, mode, missing_strategy, top_n_features
                tolerance = getattr(alignment_cfg, "tolerance", 0.0)
                mode = getattr(alignment_cfg, "mode", "entry")
                missing_strategy = getattr(alignment_cfg, "missing_strategy", "skip")
                top_n = getattr(alignment_cfg, "top_n_features", 10)
                normalization = getattr(alignment_cfg, "normalization", "l2")
            else:
                tolerance = getattr(self.config, "alignment_tolerance_seconds", 0.0)
                mode = getattr(self.config, "alignment_mode", "entry")
                missing_strategy = getattr(self.config, "missing_value_strategy", "skip")
                top_n = getattr(self.config, "top_n_features", 10)
                normalization = getattr(self.config, "normalization", "l2")

            pipeline_config = TradeShapPipelineConfig(
                alignment_tolerance_seconds=tolerance,
                alignment_mode=mode,
                missing_value_strategy=missing_strategy,
                top_n_features=top_n,
                normalization=normalization,
            )

            self._pipeline = TradeShapPipeline(
                features_df=self.features_df,
                shap_values=self.shap_values,
                feature_names=self.feature_names,
                config=pipeline_config,
            )

        return self._pipeline

    def explain_worst_trades(
        self,
        worst_trades: list[TradeMetrics],
        n: int | None = None,
    ) -> TradeShapResult:
        """Explain worst trades with full SHAP analysis pipeline.

        Args:
            worst_trades: List of trades sorted by loss (worst first)
            n: Number of trades to analyze (None = all)

        Returns:
            TradeShapResult with explanations, patterns, and hypotheses
        """
        pipeline = self._ensure_pipeline()
        return pipeline.analyze_worst_trades(worst_trades, n=n)

    def explain_trade(
        self,
        trade: TradeMetrics,
    ) -> TradeShapExplanation | TradeExplainFailure:
        """Explain a single trade."""
        pipeline = self._ensure_pipeline()
        return pipeline.explain_trade(trade)

    def explain_trades(
        self,
        trades: list[TradeMetrics],
    ) -> tuple[list[TradeShapExplanation], list[TradeExplainFailure]]:
        """Explain multiple trades."""
        pipeline = self._ensure_pipeline()
        return pipeline.explain_trades(trades)

    _UNSET: Any = object()  # Sentinel for "use config default"

    def extract_shap_vectors(
        self,
        explanations: list[TradeShapExplanation],
        normalization: str | None | Any = _UNSET,
        top_n_features: int | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Extract SHAP vectors from explanations.

        Args:
            explanations: List of TradeShapExplanation objects
            normalization: Normalization type ('l1', 'l2', 'standardize', None for none,
                          or omit to use config default)
            top_n_features: Reduce to top N features (by mean |SHAP|)

        Returns:
            2D array of shape (n_explanations, n_features)

        Raises:
            ValueError: If explanations is empty or normalization is invalid
        """
        if not explanations:
            raise ValueError("Cannot extract vectors from empty explanations list")

        # Stack SHAP vectors
        vectors = np.vstack([exp.shap_vector for exp in explanations])

        # Handle top_n reduction
        if top_n_features is not None:
            n_features = vectors.shape[1]
            if top_n_features > n_features:
                raise ValueError(
                    f"top_n_features ({top_n_features}) exceeds feature count ({n_features})"
                )
            if top_n_features < 1:
                raise ValueError("top_n_features must be positive")
            # Select top features by mean absolute SHAP
            importance = np.abs(vectors).mean(axis=0)
            top_idx = np.argsort(importance)[-top_n_features:]
            vectors = vectors[:, top_idx]

        # Apply normalization
        # If normalization is _UNSET, use config default; if None, no normalization
        if normalization is self._UNSET:
            # Use config default if available (check clustering then alignment)
            normalization = getattr(getattr(self.config, "clustering", None), "normalization", None)
            if normalization is None:
                normalization = getattr(
                    getattr(self.config, "alignment", None), "normalization", None
                )

        if normalization is not None:
            vectors = normalize(vectors, normalization)

        return vectors

    def cluster_patterns(
        self,
        shap_vectors: NDArray[np.floating[Any]],
        n_clusters: int | None = None,
    ) -> ClusteringResult:
        """Cluster SHAP vectors to identify error patterns.

        Args:
            shap_vectors: 2D array of SHAP vectors (n_trades, n_features)
            n_clusters: Number of clusters (auto-detected if None)

        Returns:
            ClusteringResult with cluster assignments and metrics

        Raises:
            ValueError: If insufficient trades for clustering
        """
        n_trades = len(shap_vectors)

        if n_trades < 3:
            raise ValueError("Need at least 3 trades for clustering")

        # Check against min_trades_for_clustering config
        min_trades = getattr(self.config, "min_trades_for_clustering", 10)
        if n_trades < min_trades:
            raise ValueError(
                f"Insufficient trades for clustering: {n_trades} < {min_trades} "
                "(set min_trades_for_clustering to lower this threshold)"
            )

        if n_clusters is not None:
            if n_clusters < 1:
                raise ValueError("n_clusters must be positive")
            if n_clusters > n_trades:
                raise ValueError(f"n_clusters ({n_clusters}) exceeds trade count ({n_trades})")

        # Get clustering config
        clustering_cfg = getattr(self.config, "clustering", None)
        if clustering_cfg is not None:
            config = ClusteringConfig(
                min_cluster_size=getattr(clustering_cfg, "min_cluster_size", 3),
                distance_metric=getattr(clustering_cfg, "distance_metric", "euclidean"),
                linkage_method=getattr(clustering_cfg, "linkage_method", "ward"),
            )
        else:
            config = ClusteringConfig()

        clusterer = HierarchicalClusterer(config=config)
        return clusterer.cluster(shap_vectors, n_clusters=n_clusters)

    def characterize_pattern(
        self,
        shap_vectors: NDArray[np.floating[Any]] | None = None,
        clustering_result: ClusteringResult | None = None,
        cluster_id: int | None = None,
        feature_names: list[str] | None = None,
        top_n: int = 5,
        *,
        # Backward-compat kwargs
        cluster_assignments: list[int] | None = None,
    ) -> dict[str, Any]:
        """Characterize a single error pattern.

        Supports both old dict-return API and new object-based API.

        Args:
            shap_vectors: 2D array of SHAP vectors
            clustering_result: Result from cluster_patterns() (new API)
            cluster_id: Which cluster to characterize
            feature_names: Feature names (uses self.feature_names if None)
            top_n: Number of top features to include
            cluster_assignments: Cluster labels (backward compat, use clustering_result instead)

        Returns:
            Dict with pattern info (cluster_id, n_trades, top_features, etc.)

        Raises:
            ValueError: If cluster_id is invalid
        """
        if shap_vectors is None:
            raise ValueError("shap_vectors is required")
        if cluster_id is None:
            raise ValueError("cluster_id is required")

        # Handle backward compat: cluster_assignments list vs ClusteringResult
        if cluster_assignments is not None:
            # Old API: create minimal ClusteringResult-like structure
            labels = cluster_assignments
            n_clusters = len(set(labels))
            centroids = None  # Will compute from shap_vectors
        elif clustering_result is not None:
            labels = clustering_result.cluster_assignments
            n_clusters = clustering_result.n_clusters
            centroids = clustering_result.centroids
        else:
            raise ValueError("Either clustering_result or cluster_assignments is required")

        if cluster_id < 0 or cluster_id >= n_clusters:
            raise ValueError(f"cluster_id {cluster_id} out of range [0, {n_clusters})")

        if feature_names is None:
            feature_names = self.feature_names

        # Validate feature count
        if shap_vectors.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: vectors have {shap_vectors.shape[1]} features, "
                f"but got {len(feature_names)} feature names"
            )

        # Get cluster mask
        cluster_mask = np.array([lbl == cluster_id for lbl in labels])
        other_mask = ~cluster_mask

        cluster_shap = shap_vectors[cluster_mask]
        other_shap = shap_vectors[other_mask]
        n_trades = int(cluster_mask.sum())

        # Compute centroids if not provided
        if centroids is None:
            centroids = np.zeros((n_clusters, shap_vectors.shape[1]))
            for c in range(n_clusters):
                c_mask = np.array([lbl == c for lbl in labels])
                if c_mask.sum() > 0:
                    centroids[c] = shap_vectors[c_mask].mean(axis=0)

        # Use characterizer
        char_cfg = getattr(self.config, "characterization", None)
        if char_cfg is not None:
            config = CharacterizationConfig(
                top_n_features=top_n,
                significance_level=getattr(char_cfg, "significance_level", 0.05),
            )
        else:
            config = CharacterizationConfig(top_n_features=top_n)

        characterizer = PatternCharacterizer(
            feature_names=feature_names,
            config=config,
        )
        pattern = characterizer.characterize_cluster(
            cluster_shap=cluster_shap,
            other_shap=other_shap,
            cluster_id=cluster_id,
            centroids=centroids,
        )

        # Return dict for backward compat
        # top_features is list[tuple[str, float, float, float, bool]]
        # (name, mean_shap, p_value_t, p_value_mw, is_significant)
        return {
            "cluster_id": cluster_id,
            "n_trades": n_trades,
            "top_features": [
                {
                    "feature": tf[0],
                    "mean_shap": tf[1],
                    "p_value_t": tf[2],
                    "p_value_mw": tf[3],
                    "significant": tf[4],
                }
                for tf in pattern.top_features
            ],
            "pattern_description": pattern.description,
            "separation_score": pattern.separation_score,
            "distinctiveness": pattern.distinctiveness,
            # Include ErrorPattern object for callers that want it
            "_pattern_object": pattern,
        }

    @property
    def hypothesis_generator(self) -> HypothesisGenerator:
        """Get hypothesis generator for custom hypothesis generation."""
        if self._hypothesis_generator is None:
            # Get hypothesis config from TradeConfig
            ext_config = getattr(self.config, "hypothesis", None)

            # Convert HypothesisGenerationConfig to HypothesisConfig if needed
            if ext_config is not None and hasattr(ext_config, "min_confidence"):
                # It's a HypothesisGenerationConfig - convert to HypothesisConfig
                config = HypothesisConfig(
                    template_library=getattr(ext_config, "template_library", "comprehensive"),
                    min_confidence=getattr(ext_config, "min_confidence", 0.5),
                    max_actions=getattr(ext_config, "max_hypotheses_per_cluster", 4),
                )
            elif isinstance(ext_config, HypothesisConfig):
                config = ext_config
            else:
                config = HypothesisConfig()

            self._hypothesis_generator = HypothesisGenerator(config=config)
        return self._hypothesis_generator

    def generate_hypothesis(
        self,
        error_pattern: ErrorPattern,
    ) -> ErrorPattern:
        """Generate hypothesis for an error pattern.

        Args:
            error_pattern: Error pattern to analyze

        Returns:
            ErrorPattern with hypothesis, actions, and confidence fields populated
        """
        return self.hypothesis_generator.generate_hypothesis(
            error_pattern,
            feature_names=self.feature_names,
        )


__all__ = [
    # Main analyzer class
    "TradeShapAnalyzer",
    # Pipeline (new recommended interface)
    "TradeShapPipeline",
    "TradeShapPipelineConfig",
    # Result models
    "TradeShapResult",
    "TradeShapExplanation",
    "TradeExplainFailure",
    "ErrorPattern",
    "ClusteringResult",
    # Components
    "TradeShapExplainer",
    "TimestampAligner",
    "AlignmentResult",
    "HierarchicalClusterer",
    "ClusteringConfig",
    "PatternCharacterizer",
    "CharacterizationConfig",
    "FeatureStatistics",
    "HypothesisGenerator",
    "HypothesisConfig",
    # Utilities
    "normalize",
    "normalize_l1",
    "normalize_l2",
    "standardize",
    "NormalizationType",
    "benjamini_hochberg",
    "find_optimal_clusters",
    "compute_cluster_sizes",
    "compute_centroids",
    "Template",
    "TemplateMatcher",
    "load_templates",
]
