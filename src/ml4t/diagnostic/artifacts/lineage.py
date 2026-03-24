"""Cross-artifact lineage helpers."""

from __future__ import annotations

from collections.abc import Sequence

from ml4t.engineer.artifacts import FeatureSpec, LabelSpec, PredictionSpec


def resolve_label_lineage(spec: LabelSpec) -> dict[str, str | None]:
    """Project label lineage metadata into a simple mapping."""
    return {
        "label_artifact": spec.artifact_id,
        "label_source_artifact": spec.definition.source_artifact,
        "label_family": spec.definition.family,
    }


def resolve_feature_lineage(specs: Sequence[FeatureSpec]) -> dict[str, tuple[str, ...]]:
    """Project feature lineage metadata into a simple mapping."""
    return {
        "feature_artifacts": tuple(spec.artifact_id for spec in specs),
        "feature_families": tuple(spec.definition.family for spec in specs),
    }


def validate_prediction_lineage(spec: PredictionSpec) -> None:
    """Validate that required prediction lineage fields are present."""
    if not spec.definition.label_artifact:
        raise ValueError("PredictionSpec.definition.label_artifact is required")
    if not spec.definition.training_hash:
        raise ValueError("PredictionSpec.definition.training_hash is required")


__all__ = [
    "resolve_feature_lineage",
    "resolve_label_lineage",
    "validate_prediction_lineage",
]
