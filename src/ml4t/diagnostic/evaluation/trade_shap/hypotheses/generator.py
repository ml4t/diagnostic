"""Hypothesis generator for trade SHAP error patterns.

Generates actionable hypotheses and improvement suggestions based on
template matching against error pattern features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import (
    TemplateMatcher,
    load_templates,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import ErrorPattern


@dataclass
class HypothesisConfig:
    """Configuration for hypothesis generation.

    Attributes:
        template_library: Which template library to use ('comprehensive' or 'minimal')
        min_confidence: Minimum confidence threshold for generating hypothesis
        max_actions: Maximum number of actions to include
    """

    template_library: str = "comprehensive"
    min_confidence: float = 0.5
    max_actions: int = 4


class HypothesisGenerator:
    """Generates hypotheses for error patterns using template matching.

    Matches error pattern features against a library of templates and
    generates actionable hypotheses about why the pattern causes losses.

    Attributes:
        config: Hypothesis generation configuration
        matcher: Template matcher

    Example:
        >>> generator = HypothesisGenerator()
        >>> enriched = generator.generate_hypothesis(error_pattern)
        >>> print(enriched.hypothesis)
        >>> print(enriched.actions)
    """

    def __init__(self, config: HypothesisConfig | Any | None = None) -> None:
        """Initialize generator.

        Args:
            config: Hypothesis configuration (uses defaults if None).
                   Accepts HypothesisConfig dataclass or HypothesisGenerationConfig Pydantic model.
        """
        # Normalize config to HypothesisConfig dataclass
        self.config = self._normalize_config(config)

        # Load templates and create matcher
        templates = load_templates(self.config.template_library)
        self.matcher = TemplateMatcher(templates)

    def _normalize_config(self, config: Any) -> HypothesisConfig:
        """Normalize config to HypothesisConfig dataclass.

        Supports both HypothesisConfig dataclass and HypothesisGenerationConfig Pydantic model.
        """
        if config is None:
            return HypothesisConfig()

        if isinstance(config, HypothesisConfig):
            return config

        # Handle Pydantic HypothesisGenerationConfig or similar
        return HypothesisConfig(
            template_library=getattr(config, "template_library", "comprehensive"),
            min_confidence=getattr(config, "min_confidence", 0.5),
            max_actions=getattr(config, "max_actions", 4),
        )

    def generate_hypothesis(
        self,
        error_pattern: ErrorPattern,
        feature_names: list[str] | None = None,
    ) -> ErrorPattern:
        """Generate hypothesis for an error pattern.

        Args:
            error_pattern: Error pattern to analyze
            feature_names: Optional list of all feature names for context

        Returns:
            ErrorPattern with hypothesis, actions, and confidence fields populated
        """
        from ml4t.diagnostic.evaluation.trade_shap.models import ErrorPattern

        # Parse top_features into dict format for matcher
        pattern_features = [
            {
                "name": feat[0],
                "mean_shap": feat[1],
                "p_value_t": feat[2],
                "p_value_mw": feat[3],
                "is_significant": feat[4],
            }
            for feat in error_pattern.top_features
        ]

        # Try to match a template
        match_result = self.matcher.match(pattern_features)

        if match_result is None or match_result.confidence < self.config.min_confidence:
            # No good match - return pattern unchanged
            return error_pattern

        # Format hypothesis from template
        hypothesis = self._format_hypothesis(
            match_result.template.hypothesis_template,
            match_result.matched_features,
        )

        # Get actions (limit to max)
        actions = match_result.template.actions[: self.config.max_actions]

        # Adjust confidence based on pattern characteristics
        adjusted_confidence = self._adjust_confidence(
            match_result.confidence,
            error_pattern.n_trades,
            error_pattern.separation_score,
        )

        # Return enriched pattern
        return ErrorPattern(
            cluster_id=error_pattern.cluster_id,
            n_trades=error_pattern.n_trades,
            description=error_pattern.description,
            top_features=error_pattern.top_features,
            separation_score=error_pattern.separation_score,
            distinctiveness=error_pattern.distinctiveness,
            hypothesis=hypothesis,
            actions=actions,
            confidence=adjusted_confidence,
        )

    def _format_hypothesis(
        self,
        template: str,
        matched_features: list[dict[str, Any]],
    ) -> str:
        """Format hypothesis string from template.

        Substitutes {feature} placeholder with actual feature name(s).
        """
        if not matched_features:
            return template.replace("{feature}", "the feature")

        # Use first matched feature name
        feature_name = matched_features[0]["name"]

        # If multiple significant features, mention them
        sig_features = [f for f in matched_features if f["is_significant"]]
        if len(sig_features) > 1:
            names = [f["name"] for f in sig_features[:2]]
            feature_name = " and ".join(names)

        return template.replace("{feature}", feature_name)

    def _adjust_confidence(
        self,
        base_confidence: float,
        n_trades: int,
        separation_score: float,
    ) -> float:
        """Adjust confidence based on pattern characteristics.

        - More trades = higher confidence (larger sample)
        - Higher separation = higher confidence (more distinct pattern)
        - Very small samples or poor separation get significant penalties
        """
        # Trade count adjustment - penalize small samples heavily
        if n_trades >= 20:
            trade_boost = 0.05
        elif n_trades >= 10:
            trade_boost = 0.02
        elif n_trades >= 5:
            trade_boost = -0.10
        elif n_trades >= 2:
            trade_boost = -0.25
        else:
            # Single trade - very unreliable
            trade_boost = -0.50

        # Separation score adjustment - penalize poor cluster separation
        if separation_score >= 1.5:
            sep_boost = 0.05
        elif separation_score >= 1.0:
            sep_boost = 0.02
        elif separation_score >= 0.5:
            sep_boost = -0.20  # Moderate separation needs noticeable penalty
        elif separation_score >= 0.3:
            sep_boost = -0.35
        else:
            # Very poor separation - cluster is not distinct
            sep_boost = -0.50

        adjusted = base_confidence + trade_boost + sep_boost
        return max(0.0, min(1.0, adjusted))

    def generate_actions(
        self,
        error_pattern: ErrorPattern,
        max_actions: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate prioritized action suggestions for an error pattern.

        Args:
            error_pattern: Error pattern with hypothesis
            max_actions: Maximum actions to return (defaults to config)

        Returns:
            List of action dictionaries with category, description, priority, etc.
        """
        if max_actions is None:
            max_actions = self.config.max_actions

        if not error_pattern.actions:
            return []

        # Categorize and prioritize actions
        categorized_actions = []

        for i, action in enumerate(error_pattern.actions[:max_actions]):
            # Determine category from action text
            category = self._categorize_action(action)

            # Priority based on position and confidence
            priority = self._determine_priority(i, error_pattern.confidence)

            categorized_actions.append(
                {
                    "category": category,
                    "description": action,
                    "priority": priority,
                    "implementation_difficulty": self._estimate_difficulty(action),
                    "rationale": f"Based on pattern: {error_pattern.description}",
                }
            )

        return categorized_actions

    def _categorize_action(self, action: str) -> str:
        """Categorize an action based on its text."""
        action_lower = action.lower()

        if any(word in action_lower for word in ["feature", "indicator", "add"]):
            return "feature_engineering"
        elif any(word in action_lower for word in ["filter", "regime", "threshold"]):
            return "filter_regime"
        elif any(word in action_lower for word in ["size", "position", "stop", "risk"]):
            return "risk_management"
        elif any(word in action_lower for word in ["tune", "parameter", "adjust"]):
            return "model_adjustment"
        else:
            return "general"

    def _determine_priority(self, position: int, confidence: float | None) -> str:
        """Determine action priority."""
        conf = confidence or 0.5

        if position == 0 and conf >= 0.7:
            return "high"
        elif position <= 1 and conf >= 0.5:
            return "medium"
        else:
            return "low"

    def _estimate_difficulty(self, action: str) -> str:
        """Estimate implementation difficulty from action text."""
        action_lower = action.lower()

        if any(word in action_lower for word in ["implement", "hmm", "model", "ensemble"]):
            return "hard"
        elif any(word in action_lower for word in ["add", "consider", "track"]):
            return "medium"
        else:
            return "easy"
