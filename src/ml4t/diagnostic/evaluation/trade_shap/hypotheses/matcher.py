"""Template matching for hypothesis generation.

Loads templates from YAML and matches them against error pattern features.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Template:
    """A hypothesis generation template.

    Attributes:
        name: Unique template identifier
        description: Human-readable description
        feature_patterns: Glob patterns to match feature names
        conditions: Matching conditions (direction, significance)
        hypothesis_template: String template with {feature} placeholder
        actions: List of remediation suggestions
        confidence_base: Base confidence score (0-1)
    """

    name: str
    description: str
    feature_patterns: list[str]
    conditions: dict[str, str]
    hypothesis_template: str
    actions: list[str]
    confidence_base: float


def load_templates(library: str = "comprehensive") -> list[Template]:
    """Load templates from the YAML file.

    Args:
        library: Which template library to load ('comprehensive' or 'minimal')

    Returns:
        List of Template objects

    Raises:
        ValueError: If library name is invalid
        FileNotFoundError: If templates.yaml is not found
    """
    # Try to load from package resources first
    try:
        files = resources.files("ml4t.diagnostic.evaluation.trade_shap.hypotheses")
        yaml_content = files.joinpath("templates.yaml").read_text()
    except (TypeError, FileNotFoundError):
        # Fall back to direct file path
        template_path = Path(__file__).parent / "templates.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}") from None
        yaml_content = template_path.read_text()

    data = yaml.safe_load(yaml_content)

    if library not in data:
        raise ValueError(f"Unknown template library: '{library}'. Available: {list(data.keys())}")

    templates = []
    for item in data[library]:
        templates.append(
            Template(
                name=item["name"],
                description=item["description"],
                feature_patterns=item["feature_patterns"],
                conditions=item["conditions"],
                hypothesis_template=item["hypothesis_template"],
                actions=item["actions"],
                confidence_base=item["confidence_base"],
            )
        )

    return templates


@dataclass
class MatchResult:
    """Result of template matching.

    Attributes:
        template: The matched template
        confidence: Adjusted confidence score
        matched_features: Features that matched the template
        primary_feature: Primary feature for hypothesis generation
    """

    template: Template
    confidence: float
    matched_features: list[dict[str, Any]]
    primary_feature: dict[str, Any] | None


class TemplateMatcher:
    """Matches error patterns against hypothesis templates.

    Attributes:
        templates: List of templates to match against

    Example:
        >>> matcher = TemplateMatcher(load_templates('comprehensive'))
        >>> result = matcher.match(pattern_features)
        >>> if result:
        ...     print(result.template.hypothesis_template)
    """

    def __init__(self, templates: list[Template]) -> None:
        """Initialize matcher with templates.

        Args:
            templates: List of templates to match against
        """
        self.templates = templates

    def match(
        self,
        pattern_features: list[dict[str, Any]],
    ) -> MatchResult | None:
        """Find best matching template for pattern features.

        Args:
            pattern_features: List of feature dicts with:
                - name: Feature name
                - mean_shap: Mean SHAP value
                - p_value_t: T-test p-value
                - p_value_mw: Mann-Whitney p-value
                - is_significant: Whether feature is significant

        Returns:
            MatchResult if a match is found, None otherwise
        """
        best_match: MatchResult | None = None
        best_confidence = 0.0

        for template in self.templates:
            result = self._match_template(template, pattern_features)
            if result and result.confidence > best_confidence:
                best_match = result
                best_confidence = result.confidence

        return best_match

    def _match_template(
        self,
        template: Template,
        pattern_features: list[dict[str, Any]],
    ) -> MatchResult | None:
        """Check if pattern matches a specific template.

        Returns MatchResult if match successful, None otherwise.
        """
        # Find features matching template patterns
        matched_features = []
        for feat in pattern_features:
            feat_name = feat["name"]
            for pattern in template.feature_patterns:
                if fnmatch.fnmatch(feat_name.lower(), pattern.lower()):
                    matched_features.append(feat)
                    break

        # No matches
        if not matched_features:
            return None

        conditions = template.conditions

        # Check significance requirement
        if conditions.get("significance") == "required":
            if not any(f["is_significant"] for f in matched_features):
                return None

        # Get significant features
        sig_features = [f for f in matched_features if f["is_significant"]]

        # Get primary feature for direction check
        if sig_features:
            # Use most significant feature (lowest p-value)
            primary_feature = min(
                sig_features,
                key=lambda x: min(x["p_value_t"], x["p_value_mw"]),
            )
        else:
            primary_feature = matched_features[0]

        # Check direction condition
        direction = conditions.get("direction", "any")
        shap_val = primary_feature["mean_shap"]

        if direction == "high" and shap_val <= 0:
            return None
        elif direction == "low" and shap_val >= 0:
            return None
        elif direction == "positive" and shap_val <= 0:
            return None
        elif direction == "negative" and shap_val >= 0:
            return None
        elif direction == "extreme" and abs(shap_val) < 0.1:
            return None
        elif direction == "moderate" and (abs(shap_val) < 0.05 or abs(shap_val) > 0.3):
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            template, matched_features, sig_features, pattern_features
        )

        return MatchResult(
            template=template,
            confidence=confidence,
            matched_features=matched_features,
            primary_feature=primary_feature,
        )

    def _calculate_confidence(
        self,
        template: Template,
        matched_features: list[dict[str, Any]],
        sig_features: list[dict[str, Any]],
        all_features: list[dict[str, Any]],
    ) -> float:
        """Calculate adjusted confidence for a match."""
        base_confidence = template.confidence_base

        # Boost for matching multiple features
        match_ratio = len(matched_features) / max(len(all_features), 1)
        match_boost = min(0.1, match_ratio * 0.15)

        # Boost for significant features
        n_significant = sum(1 for f in matched_features if f["is_significant"])
        significance_boost = min(0.1, n_significant / max(len(all_features), 1) * 0.2)

        # Boost for strong statistical significance
        if sig_features:
            avg_p = sum(min(f["p_value_t"], f["p_value_mw"]) for f in sig_features) / len(
                sig_features
            )
            p_value_boost = max(0.0, (0.05 - avg_p) * 2.0)
        else:
            p_value_boost = 0.0

        confidence = base_confidence + match_boost + significance_boost + p_value_boost
        return min(1.0, confidence)
