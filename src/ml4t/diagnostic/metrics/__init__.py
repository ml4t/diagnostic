"""
Metrics module for ML4T Diagnostic.

Provides statistical metrics and percentile computation utilities for model evaluation.
"""

from ml4t.diagnostic.metrics.percentiles import compute_fold_percentiles

__all__ = ["compute_fold_percentiles"]
