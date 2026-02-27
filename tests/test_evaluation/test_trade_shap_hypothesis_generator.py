"""Unit tests for Trade SHAP hypothesis generation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml4t.diagnostic.evaluation.trade_shap.hypotheses.generator import (
    HypothesisConfig,
    HypothesisGenerator,
)
from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import MatchResult, Template
from ml4t.diagnostic.evaluation.trade_shap.models import ErrorPattern


def _error_pattern(
    *,
    n_trades: int = 12,
    separation_score: float = 1.2,
    actions: list[str] | None = None,
    confidence: float | None = None,
) -> ErrorPattern:
    return ErrorPattern(
        cluster_id=1,
        n_trades=n_trades,
        description="High momentum -> losses",
        top_features=[
            ("momentum_20d", 0.42, 0.001, 0.002, True),
            ("volatility_10d", -0.28, 0.01, 0.02, True),
        ],
        separation_score=separation_score,
        distinctiveness=1.5,
        actions=actions,
        confidence=confidence,
    )


class TestHypothesisGeneratorConfig:
    """Tests for configuration normalization."""

    def test_normalize_config_defaults_when_none(self) -> None:
        generator = HypothesisGenerator(config=None)
        assert isinstance(generator.config, HypothesisConfig)
        assert generator.config.template_library == "comprehensive"
        assert generator.config.min_confidence == 0.5
        assert generator.config.max_actions == 4

    def test_normalize_config_from_object_attributes(self) -> None:
        config = SimpleNamespace(template_library="minimal", min_confidence=0.7, max_actions=2)
        generator = HypothesisGenerator(config=config)
        assert generator.config.template_library == "minimal"
        assert generator.config.min_confidence == 0.7
        assert generator.config.max_actions == 2


class TestHypothesisGeneratorCore:
    """Tests for generate_hypothesis behavior."""

    def test_generate_hypothesis_returns_unchanged_when_confidence_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generator = HypothesisGenerator(HypothesisConfig(min_confidence=0.8))
        pattern = _error_pattern()

        template = Template(
            name="momentum",
            description="Momentum pattern",
            feature_patterns=["momentum*"],
            conditions={"direction": "high"},
            hypothesis_template="Losses occur when {feature} spikes",
            actions=["Add momentum filter"],
            confidence_base=0.6,
        )
        low_confidence = MatchResult(
            template=template,
            confidence=0.7,
            matched_features=[{"name": "momentum_20d", "is_significant": True}],
            primary_feature={"name": "momentum_20d"},
        )
        monkeypatch.setattr(generator.matcher, "match", lambda _: low_confidence)

        result = generator.generate_hypothesis(pattern)
        assert result is pattern

    def test_generate_hypothesis_enriches_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        generator = HypothesisGenerator(HypothesisConfig(min_confidence=0.3, max_actions=1))
        pattern = _error_pattern(n_trades=20, separation_score=1.6)

        template = Template(
            name="multi_feature",
            description="Multi-feature pattern",
            feature_patterns=["*"],
            conditions={"direction": "any"},
            hypothesis_template="Losses increase when {feature} diverges",
            actions=["Add filter", "Tune threshold"],
            confidence_base=0.5,
        )
        matched = MatchResult(
            template=template,
            confidence=0.8,
            matched_features=[
                {"name": "momentum_20d", "is_significant": True},
                {"name": "volatility_10d", "is_significant": True},
            ],
            primary_feature={"name": "momentum_20d"},
        )
        monkeypatch.setattr(generator.matcher, "match", lambda _: matched)

        enriched = generator.generate_hypothesis(pattern)
        assert enriched is not pattern
        assert "momentum_20d and volatility_10d" in (enriched.hypothesis or "")
        assert enriched.actions == ["Add filter"]
        assert enriched.confidence is not None
        assert 0.0 <= enriched.confidence <= 1.0

    def test_format_hypothesis_without_matches_uses_generic_feature(self) -> None:
        generator = HypothesisGenerator()
        text = generator._format_hypothesis("Losses follow {feature}", [])
        assert text == "Losses follow the feature"

    def test_adjust_confidence_penalties_and_clamping(self) -> None:
        generator = HypothesisGenerator()

        assert generator._adjust_confidence(0.6, n_trades=20, separation_score=1.6) == pytest.approx(0.7)
        assert generator._adjust_confidence(0.6, n_trades=10, separation_score=1.0) == pytest.approx(0.64)
        assert generator._adjust_confidence(0.6, n_trades=1, separation_score=0.2) == 0.0
        assert generator._adjust_confidence(0.6, n_trades=4, separation_score=0.4) == 0.0


class TestHypothesisGeneratorActions:
    """Tests for action generation and categorization helpers."""

    def test_generate_actions_returns_empty_when_pattern_has_no_actions(self) -> None:
        generator = HypothesisGenerator()
        pattern = _error_pattern(actions=None)
        assert generator.generate_actions(pattern) == []

    def test_generate_actions_limits_and_assigns_priority(self) -> None:
        generator = HypothesisGenerator(HypothesisConfig(max_actions=3))
        pattern = _error_pattern(
            actions=[
                "Add momentum feature",
                "Tune threshold by regime",
                "Implement ensemble model",
                "Track residual drift",
            ],
            confidence=0.85,
        )
        actions = generator.generate_actions(pattern, max_actions=3)

        assert len(actions) == 3
        assert actions[0]["priority"] == "high"
        assert actions[1]["priority"] == "medium"
        assert actions[2]["priority"] == "low"
        assert actions[0]["category"] == "feature_engineering"
        assert actions[1]["category"] == "filter_regime"
        assert actions[2]["implementation_difficulty"] == "hard"

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ("Add feature for volatility", "feature_engineering"),
            ("Apply regime filter", "filter_regime"),
            ("Tighten stop and position size", "risk_management"),
            ("Tune model parameter", "model_adjustment"),
            ("Review logs", "general"),
        ],
    )
    def test_categorize_action(self, action: str, expected: str) -> None:
        generator = HypothesisGenerator()
        assert generator._categorize_action(action) == expected

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ("Implement ensemble model", "hard"),
            ("Add volume filter", "medium"),
            ("Recheck signal", "easy"),
        ],
    )
    def test_estimate_difficulty(self, action: str, expected: str) -> None:
        generator = HypothesisGenerator()
        assert generator._estimate_difficulty(action) == expected

    def test_determine_priority_uses_default_confidence_when_none(self) -> None:
        generator = HypothesisGenerator()
        assert generator._determine_priority(0, None) == "medium"
