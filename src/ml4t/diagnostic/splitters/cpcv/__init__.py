"""Combinatorial Purged Cross-Validation (CPCV) submodules.

This package provides modular components for CPCV:

- combinations: Combination generation and reservoir sampling
- partitioning: Group partitioning strategies
- windows: Time window computation for purging
- purge_engine: Core purging and embargo logic
"""

from ml4t.diagnostic.splitters.cpcv.combinations import (
    iter_combinations,
    reservoir_sample_combinations,
)
from ml4t.diagnostic.splitters.cpcv.partitioning import (
    boundaries_to_indices,
    create_contiguous_partitions,
    create_session_partitions,
    exact_indices_to_array,
    validate_contiguous_partitions,
)
from ml4t.diagnostic.splitters.cpcv.purge_engine import (
    apply_multi_asset_purging,
    apply_segment_purging,
    apply_single_asset_purging,
    prepare_test_groups_data,
    process_asset_purging,
)
from ml4t.diagnostic.splitters.cpcv.windows import (
    TimeWindow,
    find_contiguous_segments,
    merge_windows,
    timestamp_window_from_indices,
)

__all__ = [
    # combinations
    "iter_combinations",
    "reservoir_sample_combinations",
    # partitioning
    "create_contiguous_partitions",
    "validate_contiguous_partitions",
    "create_session_partitions",
    "boundaries_to_indices",
    "exact_indices_to_array",
    # windows
    "TimeWindow",
    "timestamp_window_from_indices",
    "find_contiguous_segments",
    "merge_windows",
    # purge_engine
    "apply_single_asset_purging",
    "apply_multi_asset_purging",
    "prepare_test_groups_data",
    "process_asset_purging",
    "apply_segment_purging",
]
