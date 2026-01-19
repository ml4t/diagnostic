"""Trade-level SHAP diagnostics models.

This package contains the data models for Trade SHAP analysis.
The main TradeShapAnalyzer and HypothesisGenerator classes are imported
from the parent module for backward compatibility.

For analysis, import from the evaluation module:
    >>> from ml4t.diagnostic.evaluation import TradeShapAnalyzer

For models only:
    >>> from ml4t.diagnostic.evaluation.trade_shap import (
    ...     TradeShapResult,
    ...     ErrorPattern,
    ...     ClusteringResult,
    ...     TradeShapExplanation,
    ... )
"""

# Import models from the dedicated models module
from ml4t.diagnostic.evaluation.trade_shap.alignment import (
    AlignmentResult,
    TimestampAligner,
)
from ml4t.diagnostic.evaluation.trade_shap.characterize import (
    CharacterizationConfig,
    FeatureStatistics,
    PatternCharacterizer,
    benjamini_hochberg,
)
from ml4t.diagnostic.evaluation.trade_shap.cluster import (
    ClusteringConfig,
    HierarchicalClusterer,
    compute_centroids,
    compute_cluster_sizes,
    find_optimal_clusters,
)
from ml4t.diagnostic.evaluation.trade_shap.explain import TradeShapExplainer
from ml4t.diagnostic.evaluation.trade_shap.hypotheses import (
    HypothesisConfig,
    HypothesisGenerator,
    Template,
    TemplateMatcher,
    load_templates,
)
from ml4t.diagnostic.evaluation.trade_shap.models import (
    ClusteringResult,
    ErrorPattern,
    TradeExplainFailure,
    TradeShapExplanation,
    TradeShapResult,
)
from ml4t.diagnostic.evaluation.trade_shap.normalize import (
    NormalizationType,
    normalize,
    normalize_l1,
    normalize_l2,
    standardize,
)
from ml4t.diagnostic.evaluation.trade_shap.pipeline import (
    TradeShapPipeline,
    TradeShapPipelineConfig,
)

__all__ = [
    # Alignment
    "TimestampAligner",
    "AlignmentResult",
    # Explainer
    "TradeShapExplainer",
    # Normalization
    "normalize",
    "normalize_l1",
    "normalize_l2",
    "standardize",
    "NormalizationType",
    # Clustering
    "HierarchicalClusterer",
    "ClusteringConfig",
    "find_optimal_clusters",
    "compute_cluster_sizes",
    "compute_centroids",
    # Characterization
    "PatternCharacterizer",
    "CharacterizationConfig",
    "FeatureStatistics",
    "benjamini_hochberg",
    # Hypothesis generation
    "HypothesisGenerator",
    "HypothesisConfig",
    "TemplateMatcher",
    "Template",
    "load_templates",
    # Pipeline
    "TradeShapPipeline",
    "TradeShapPipelineConfig",
    # Result models
    "TradeShapResult",
    "TradeShapExplanation",
    "TradeExplainFailure",
    "ClusteringResult",
    "ErrorPattern",
]
