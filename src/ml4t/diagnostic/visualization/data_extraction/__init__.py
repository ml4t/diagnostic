"""Data extraction for visualization layer.

This module extracts structured data from analysis results for interactive
dashboards and reports, including:
- Per-method breakdowns with uncertainty
- Per-feature aggregations for drill-down views
- Method comparison metrics
- Auto-generated narrative summaries
"""

from .importance import extract_importance_viz_data
from .interaction import extract_interaction_viz_data
from .types import (
    FeatureDetailData,
    FeatureInteractionData,
    ImportanceVizData,
    InteractionMatrixData,
    InteractionVizData,
    LLMContextData,
    MethodComparisonData,
    MethodImportanceData,
    NetworkGraphData,
    UncertaintyData,
)
from .validation import _validate_lengths_match, _validate_matrix_feature_alignment

__all__ = [
    # Main extraction functions
    "extract_importance_viz_data",
    "extract_interaction_viz_data",
    # TypedDict types for importance
    "ImportanceVizData",
    "MethodImportanceData",
    "FeatureDetailData",
    "MethodComparisonData",
    "UncertaintyData",
    "LLMContextData",
    # TypedDict types for interaction
    "InteractionVizData",
    "FeatureInteractionData",
    "NetworkGraphData",
    "InteractionMatrixData",
    # Validation helpers (for internal use)
    "_validate_lengths_match",
    "_validate_matrix_feature_alignment",
]
