"""Core functionality for ml4t-diagnostic.

This module contains the fundamental logic for purging, embargo, and sampling
that underlies all cross-validation splitters.
"""

from ml4t.diagnostic.core.purging import (
    apply_purging_and_embargo,
    calculate_embargo_indices,
    calculate_purge_indices,
)
from ml4t.diagnostic.core.sampling import (
    balanced_subsample,
    block_bootstrap,
    event_based_sample,
    sample_weights_by_importance,
    stratified_sample_time_series,
)

__all__: list[str] = [
    "apply_purging_and_embargo",
    "balanced_subsample",
    "calculate_embargo_indices",
    "calculate_purge_indices",
    "event_based_sample",
    "sample_weights_by_importance",
    "block_bootstrap",
    "stratified_sample_time_series",
]
