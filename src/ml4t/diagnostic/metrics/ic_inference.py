"""Inference helpers for Information Coefficient series."""

from statsmodels.stats.sandwich_covariance import cov_hac

from ml4t.diagnostic.evaluation.metrics.ic_statistics import (
    compute_ic_decay,
    compute_ic_hac_stats,
    compute_ic_summary_stats,
)

__all__ = ["compute_ic_summary_stats", "compute_ic_hac_stats", "compute_ic_decay", "cov_hac"]
