"""Systematic feature selection for ML pipelines.

Provides filtering and selection of features based on multiple criteria:
IC analysis, ML importance, correlation, and drift detection.

Example:
    >>> from ml4t.diagnostic.selection import FeatureSelector
    >>>
    >>> selector = FeatureSelector(outcome_results, corr_matrix)
    >>> selector.run_pipeline([
    ...     ("ic", {"threshold": 0.02}),
    ...     ("correlation", {"threshold": 0.8}),
    ...     ("importance", {"threshold": 0.01, "method": "mdi"}),
    ... ])
    >>> selected = selector.get_selected_features()
"""

from ml4t.diagnostic.selection.systematic import (
    FeatureSelector,
    SelectionReport,
    SelectionStep,
)
from ml4t.diagnostic.selection.types import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcomeResult,
)

__all__ = [
    "FeatureICResults",
    "FeatureImportanceResults",
    "FeatureOutcomeResult",
    "FeatureSelector",
    "SelectionReport",
    "SelectionStep",
]
