"""Unified artifact resolution for ML4T libraries."""

from .lineage import resolve_feature_lineage, resolve_label_lineage, validate_prediction_lineage
from .resolver import (
    dump_spec,
    load_feature_spec,
    load_label_spec,
    load_market_data_spec,
    load_prediction_spec,
    load_spec,
    spec_from_mapping,
)

__all__ = [
    "dump_spec",
    "load_feature_spec",
    "load_label_spec",
    "load_market_data_spec",
    "load_prediction_spec",
    "load_spec",
    "resolve_feature_lineage",
    "resolve_label_lineage",
    "spec_from_mapping",
    "validate_prediction_lineage",
]
