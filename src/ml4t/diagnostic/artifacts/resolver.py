"""Unified loading and dumping for ML4T artifact specifications."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml4t.engineer.artifacts import FeatureSpec, LabelSpec, PredictionSpec
from ml4t.specs import (
    ArtifactKind,
    ArtifactSpec,
    MarketDataSpec,
    read_spec_payload,
    write_spec_payload,
)


def spec_from_mapping(mapping: Mapping[str, Any]) -> ArtifactSpec:
    """Instantiate the correct spec class from a generic mapping."""
    kind = ArtifactKind(str(mapping["kind"]))
    concrete = dict(mapping)
    if kind == ArtifactKind.MARKET_DATA:
        return MarketDataSpec.from_mapping(concrete)
    if kind == ArtifactKind.LABELS:
        return LabelSpec.from_mapping(concrete)
    if kind == ArtifactKind.FEATURES:
        return FeatureSpec.from_mapping(concrete)
    if kind == ArtifactKind.PREDICTIONS:
        return PredictionSpec.from_mapping(concrete)
    raise ValueError(f"Unsupported artifact kind: {kind}")


def load_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> ArtifactSpec:
    """Load any artifact specification from a path or in-memory mapping."""
    return spec_from_mapping(read_spec_payload(path_or_mapping))


def load_market_data_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> MarketDataSpec:
    """Load a market-data artifact spec."""
    spec = load_spec(path_or_mapping)
    if not isinstance(spec, MarketDataSpec):
        raise TypeError(f"Expected market_data spec, got {spec.kind.value}")
    return spec


def load_label_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> LabelSpec:
    """Load a label artifact spec."""
    spec = load_spec(path_or_mapping)
    if not isinstance(spec, LabelSpec):
        raise TypeError(f"Expected labels spec, got {spec.kind.value}")
    return spec


def load_feature_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> FeatureSpec:
    """Load a feature artifact spec."""
    spec = load_spec(path_or_mapping)
    if not isinstance(spec, FeatureSpec):
        raise TypeError(f"Expected features spec, got {spec.kind.value}")
    return spec


def load_prediction_spec(path_or_mapping: str | Path | Mapping[str, Any]) -> PredictionSpec:
    """Load a prediction artifact spec."""
    spec = load_spec(path_or_mapping)
    if not isinstance(spec, PredictionSpec):
        raise TypeError(f"Expected predictions spec, got {spec.kind.value}")
    return spec


def dump_spec(spec: ArtifactSpec, path: str | Path) -> Path:
    """Serialize an artifact specification to YAML or JSON."""
    return write_spec_payload(spec.to_dict(), path)


__all__ = [
    "dump_spec",
    "load_feature_spec",
    "load_label_spec",
    "load_market_data_spec",
    "load_prediction_spec",
    "load_spec",
    "spec_from_mapping",
]
