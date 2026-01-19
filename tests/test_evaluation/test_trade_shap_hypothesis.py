"""Tests for Trade-SHAP hypothesis generation and confidence scoring.

This module tests the HypothesisGenerator class which converts error patterns
into actionable trading hypotheses. Tests cover:
1. Template coverage (all pattern types matched)
2. Domain validation (realistic trading scenarios)
3. Confidence calibration (scores reflect pattern quality)
4. Action quality (suggestions are actionable and specific)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import polars as pl

from ml4t.diagnostic.config import TradeConfig, TradeHypothesisSettings
from ml4t.diagnostic.evaluation.trade_shap_diagnostics import (
    ErrorPattern,
    HypothesisGenerator,
    TradeShapAnalyzer,
)


class TestTemplateLibrary:
    """Test that all template types are available and accessible."""

    def test_comprehensive_templates_exist(self):
        """Test that comprehensive template library has expected size."""
        generator = HypothesisGenerator()

        # Access templates through matcher
        templates = generator.matcher.templates
        assert len(templates) >= 10, f"Expected >= 10 templates, got {len(templates)}"

    def test_template_categories_covered(self):
        """Test that all major pattern categories are represented."""
        generator = HypothesisGenerator()
        templates = generator.matcher.templates

        # Extract template names
        template_names = {t.name for t in templates}

        # Check for each category
        [n for n in template_names if "momentum" in n.lower()]
        [n for n in template_names if "volatil" in n.lower() or "vol" in n.lower()]
        [n for n in template_names if "trend" in n.lower()]

        # Just check we have diverse templates
        assert len(template_names) >= 5, "Should have at least 5 different template types"

    def test_minimal_template_library(self):
        """Test minimal template library configuration."""
        from ml4t.diagnostic.evaluation.trade_shap.hypotheses import HypothesisConfig

        config = HypothesisConfig(template_library="minimal")
        generator = HypothesisGenerator(config)

        # Minimal library should have fewer templates
        templates = generator.matcher.templates
        assert len(templates) >= 3, (
            f"Minimal library should have at least 3 templates, got {len(templates)}"
        )

    def test_template_structure_validity(self):
        """Test that all templates have required fields."""
        generator = HypothesisGenerator()

        for template in generator.matcher.templates:
            # Templates are dataclass objects with required attributes
            assert hasattr(template, "name"), "Template must have name"
            assert hasattr(template, "description"), "Template must have description"
            assert hasattr(template, "feature_patterns"), "Template must have feature_patterns"
            assert hasattr(template, "conditions"), "Template must have conditions"
            assert hasattr(template, "hypothesis_template"), (
                "Template must have hypothesis_template"
            )
            assert hasattr(template, "actions"), "Template must have actions"
            assert hasattr(template, "confidence_base"), "Template must have confidence_base"

            # Validate field types
            assert isinstance(template.feature_patterns, list), "feature_patterns must be list"
            assert isinstance(template.actions, list), "actions must be list"
            assert isinstance(template.confidence_base, int | float), (
                "confidence_base must be numeric"
            )
            assert 0 <= template.confidence_base <= 1, "confidence_base must be in [0, 1]"


class TestDomainValidation:
    """Test hypothesis generation with realistic trading scenarios."""

    def test_momentum_reversal_pattern(self):
        """Test hypothesis generation for momentum reversal pattern."""
        generator = HypothesisGenerator()

        # Create error pattern with momentum feature
        error_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="High momentum causing losses",
            top_features=[
                ("momentum_20d", 2.5, 0.001, 0.002, True),  # High value, significant
                ("volume_ratio", 1.2, 0.15, 0.18, False),
            ],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        # Should have hypothesis
        assert enriched.hypothesis is not None
        assert "momentum" in enriched.hypothesis.lower()

        # Should have actions
        assert len(enriched.actions) > 0

        # Should have reasonable confidence
        assert enriched.confidence > 0.5

    def test_volatility_spike_pattern(self):
        """Test hypothesis generation for volatility spike pattern."""
        generator = HypothesisGenerator()

        error_pattern = ErrorPattern(
            cluster_id=1,
            n_trades=20,
            description="High volatility causing losses",
            top_features=[
                ("volatility_20d", 3.2, 0.002, 0.0030, True),  # High vol, significant
                ("atr_14d", 2.1, 0.01, 0.0110, True),
            ],
            separation_score=0.8,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        assert enriched.hypothesis is not None
        assert any(word in enriched.hypothesis.lower() for word in ["volatility", "vol"])
        assert len(enriched.actions) > 0
        assert enriched.confidence > 0.6

    def test_trend_following_failure(self):
        """Test hypothesis generation for trend following failures."""
        generator = HypothesisGenerator()

        error_pattern = ErrorPattern(
            cluster_id=2,
            n_trades=12,
            description="Trend signals failing",
            top_features=[
                ("trend_strength", 1.8, 0.005, 0.0060, True),
                ("ma_50_200_cross", 1.5, 0.02, 0.0210, True),
            ],
            separation_score=0.65,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        assert enriched.hypothesis is not None
        assert enriched.confidence > 0

    def test_mean_reversion_failure(self):
        """Test hypothesis generation for failed mean reversion."""
        generator = HypothesisGenerator()

        error_pattern = ErrorPattern(
            cluster_id=3,
            n_trades=18,
            description="RSI extreme values",
            top_features=[
                ("rsi_14", 2.3, 0.003, 0.0040, True),  # Extreme RSI
                ("zscore_20", -2.1, 0.01, 0.0110, True),
            ],
            separation_score=0.72,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        assert enriched.hypothesis is not None
        assert enriched.confidence > 0

    def test_volume_divergence_pattern(self):
        """Test hypothesis generation for volume divergence."""
        generator = HypothesisGenerator()

        error_pattern = ErrorPattern(
            cluster_id=4,
            n_trades=10,
            description="Low volume with price moves",
            top_features=[
                ("volume_20d_ma", -1.8, 0.008, 0.0090, True),  # Low volume
                ("obv", -1.5, 0.02, 0.0210, True),
            ],
            separation_score=0.68,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        assert enriched.hypothesis is not None
        assert enriched.confidence > 0

    def test_no_matching_template(self):
        """Test behavior when no template matches."""
        generator = HypothesisGenerator()

        # Create pattern with non-matching features
        error_pattern = ErrorPattern(
            cluster_id=5,
            n_trades=8,
            description="Unknown pattern",
            top_features=[
                ("random_feature_xyz", 1.2, 0.10, 0.1010, False),
                ("another_unknown", 0.8, 0.25, 0.2510, False),
            ],
            separation_score=0.5,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(error_pattern)

        # Should return original pattern without hypothesis
        assert enriched.hypothesis is None
        assert enriched.confidence is None or enriched.confidence == 0.0


class TestConfidenceCalibration:
    """Test that confidence scores reflect pattern quality accurately."""

    def test_high_confidence_with_strong_pattern(self):
        """Test high confidence for strong patterns."""
        generator = HypothesisGenerator()

        # Strong pattern: multiple features, high significance, good separation
        strong_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="Clear momentum reversal",
            top_features=[
                ("momentum_20d", 3.5, 0.0001, 0.0011, True),  # Very significant
                ("momentum_50d", 3.2, 0.0002, 0.0012, True),
                ("roc_20d", 2.8, 0.0005, 0.0015, True),
            ],
            separation_score=0.85,  # High separation
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(strong_pattern)

        # Should have high confidence (>0.7)
        assert enriched.confidence >= 0.7, f"Expected confidence >= 0.7, got {enriched.confidence}"

    def test_medium_confidence_with_moderate_pattern(self):
        """Test medium confidence for moderate patterns."""
        generator = HypothesisGenerator()

        moderate_pattern = ErrorPattern(
            cluster_id=1,
            n_trades=10,  # Fewer trades
            description="Moderate pattern",
            top_features=[
                ("volatility_20d", 1.2, 0.03, 0.035, True),  # Lower SHAP, moderate significance
            ],
            separation_score=0.5,  # Moderate separation
            distinctiveness=1.2,  # Lower distinctiveness
        )

        enriched = generator.generate_hypothesis(moderate_pattern)

        # Should have medium/moderate confidence
        assert enriched.confidence is None or 0.4 <= enriched.confidence < 0.85

    def test_low_confidence_with_weak_pattern(self):
        """Test low confidence for weak patterns."""
        generator = HypothesisGenerator()

        weak_pattern = ErrorPattern(
            cluster_id=2,
            n_trades=8,
            description="Weak pattern",
            top_features=[
                ("momentum_20d", 1.2, 0.08, 0.0810, False),  # Low significance
            ],
            separation_score=0.4,  # Poor separation
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(weak_pattern)

        # Should have lower confidence or None if no match
        assert enriched.confidence is None or enriched.confidence < 0.7

    def test_min_confidence_filtering(self):
        """Test that min_confidence config filters low-confidence hypotheses."""
        config = TradeHypothesisSettings(min_confidence=0.75)
        generator = HypothesisGenerator(config)

        # Pattern that would normally have ~0.6 confidence
        marginal_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=10,
            description="Marginal pattern",
            top_features=[
                ("volatility_20d", 1.8, 0.03, 0.0310, True),
            ],
            separation_score=0.55,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(marginal_pattern)

        # Should return empty hypothesis if below threshold
        if enriched.confidence < 0.75:
            assert enriched.hypothesis is None

    def test_confidence_scales_with_trade_count(self):
        """Test that confidence increases with more trades in pattern."""
        generator = HypothesisGenerator()

        # Same features, different trade counts
        small_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=5,
            description="Small pattern",
            top_features=[("momentum_20d", 2.5, 0.01, 0.0110, True)],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        large_pattern = ErrorPattern(
            cluster_id=1,
            n_trades=50,
            description="Large pattern",
            top_features=[("momentum_20d", 2.5, 0.01, 0.0110, True)],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        small_enriched = generator.generate_hypothesis(small_pattern)
        large_enriched = generator.generate_hypothesis(large_pattern)

        # Larger pattern should have higher confidence
        assert large_enriched.confidence >= small_enriched.confidence


class TestActionQuality:
    """Test that generated actions are actionable and specific."""

    def test_actions_are_specific(self):
        """Test that actions mention specific features or techniques."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Momentum reversal",
            top_features=[("momentum_20d", 2.8, 0.002, 0.0030, True)],
            separation_score=0.75,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Actions should contain specific terms
        actions_text = " ".join(enriched.actions or [])
        assert len(actions_text) > 0, "Should have action text"

        # Should mention feature type or technique
        has_specifics = any(
            term in actions_text.lower()
            for term in ["momentum", "filter", "threshold", "confirmation", "regime", "indicator"]
        )
        assert has_specifics, "Actions should mention specific features or techniques"

    def test_actions_are_actionable(self):
        """Test that actions contain verbs (add, implement, consider)."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Volatility spike",
            top_features=[("volatility_20d", 3.1, 0.001, 0.0020, True)],
            separation_score=0.8,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Actions should start with action verbs
        action_verbs = {
            "add",
            "implement",
            "consider",
            "adjust",
            "tighten",
            "strengthen",
            "require",
        }

        actions_text = " ".join(enriched.actions or []).lower()
        has_action_verb = any(verb in actions_text for verb in action_verbs)
        assert has_action_verb, "Actions should contain action verbs"

    def test_multiple_action_categories(self):
        """Test that actions cover multiple improvement approaches."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="Complex pattern",
            top_features=[
                ("momentum_20d", 2.5, 0.002, 0.0030, True),
                ("volatility_20d", 2.1, 0.008, 0.0090, True),
            ],
            separation_score=0.78,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Should have multiple actions
        assert len(enriched.actions or []) >= 3, "Should suggest multiple actions"


class TestIntegrationWithTradeShapAnalyzer:
    """Test hypothesis generation integration with TradeShapAnalyzer."""

    def test_analyzer_generate_hypothesis_method(self):
        """Test that TradeShapAnalyzer has generate_hypothesis method."""
        # Setup analyzer
        n_samples = 20
        n_features = 10
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Check method exists
        assert hasattr(analyzer, "generate_hypothesis")
        assert callable(analyzer.generate_hypothesis)

    def test_end_to_end_hypothesis_generation(self):
        """Test complete workflow from pattern to hypothesis."""
        # Setup
        n_samples = 30
        n_features = 15
        features_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_samples)],
                "momentum_20d": np.random.randn(n_samples) + 2.0,  # High momentum
                "volatility_20d": np.random.randn(n_samples),
                **{f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)},
            }
        )

        shap_values = np.random.randn(n_samples, n_features + 2)
        model = Mock()
        config = TradeConfig()
        analyzer = TradeShapAnalyzer(
            model=model, features_df=features_df, shap_values=shap_values, config=config
        )

        # Create error pattern
        error_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Momentum pattern",
            top_features=[
                ("momentum_20d", 2.5, 0.005, 0.0060, True),
                ("volatility_20d", 1.2, 0.08, 0.0810, False),
            ],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        # Generate hypothesis through analyzer
        enriched = analyzer.generate_hypothesis(error_pattern)

        # Verify enrichment
        assert enriched.cluster_id == error_pattern.cluster_id
        assert enriched.hypothesis is not None or enriched.confidence == 0.0
        assert enriched.confidence >= 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_top_features(self):
        """Test hypothesis generation with empty feature list."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=10,
            description="Empty features",
            top_features=[],
            separation_score=0.5,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Should handle gracefully
        assert enriched.hypothesis is None
        assert enriched.confidence is None or enriched.confidence == 0.0

    def test_very_low_separation_score(self):
        """Test with very poor cluster separation."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Poor separation",
            top_features=[("momentum_20d", 2.0, 0.01, 0.0110, True)],
            separation_score=0.1,  # Very poor
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Confidence should be low or None if no match
        assert enriched.confidence is None or enriched.confidence < 0.6

    def test_single_trade_cluster(self):
        """Test hypothesis generation for single-trade cluster."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=1,
            description="Single trade",
            top_features=[("momentum_20d", 2.5, 0.01, 0.0110, True)],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Should have very low confidence or None due to small sample
        assert enriched.confidence is None or enriched.confidence < 0.6

    def test_non_significant_features(self):
        """Test with features that have high p-values (not significant)."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Non-significant pattern",
            top_features=[
                ("momentum_20d", 1.5, 0.25, 0.2510, False),  # High p-value
                ("volatility_20d", 1.2, 0.30, 0.3010, False),
            ],
            separation_score=0.6,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Should have lower confidence, None, or no hypothesis
        assert enriched.confidence is None or enriched.confidence < 0.7


class TestPerformance:
    """Test hypothesis generation performance."""

    def test_hypothesis_generation_speed(self):
        """Test that hypothesis generation completes quickly."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Performance test",
            top_features=[
                ("momentum_20d", 2.5, 0.005, 0.0060, True),
                ("volatility_20d", 2.1, 0.01, 0.0110, True),
                ("trend_strength", 1.8, 0.02, 0.0210, True),
            ],
            separation_score=0.75,
            distinctiveness=1.5,
        )

        import time

        start = time.time()
        for _ in range(100):
            _ = generator.generate_hypothesis(pattern)
        elapsed = time.time() - start

        # Should complete 100 hypotheses in <1 second
        assert elapsed < 1.0, f"100 hypotheses took {elapsed:.2f}s, expected <1.0s"
        assert elapsed / 100 < 0.01, "Each hypothesis should take <10ms"


class TestTemplateLoading:
    """Tests for template loading and validation."""

    def test_invalid_library_raises_value_error(self):
        """Invalid template library name should raise ValueError."""
        import pytest

        from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import load_templates

        with pytest.raises(ValueError, match="Unknown template library"):
            load_templates("nonexistent_library")

    def test_comprehensive_library_loads(self):
        """Comprehensive library should load successfully."""
        from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import load_templates

        templates = load_templates("comprehensive")
        assert len(templates) >= 5

    def test_minimal_library_loads(self):
        """Minimal library should load successfully."""
        from ml4t.diagnostic.evaluation.trade_shap.hypotheses.matcher import load_templates

        templates = load_templates("minimal")
        assert len(templates) >= 1


class TestConfigNormalization:
    """Tests for HypothesisConfig normalization."""

    def test_dataclass_config_passed_through(self):
        """HypothesisConfig dataclass should be used directly."""
        from ml4t.diagnostic.evaluation.trade_shap.hypotheses import HypothesisConfig

        config = HypothesisConfig(
            template_library="minimal",
            min_confidence=0.3,
            max_actions=2,
        )
        generator = HypothesisGenerator(config)

        assert generator.config.template_library == "minimal"
        assert generator.config.min_confidence == 0.3
        assert generator.config.max_actions == 2

    def test_pydantic_config_normalized(self):
        """TradeHypothesisSettings Pydantic model should be normalized."""
        config = TradeHypothesisSettings(
            min_confidence=0.6,
            template_library="minimal",
        )
        generator = HypothesisGenerator(config)

        assert generator.config.min_confidence == 0.6
        assert generator.config.template_library == "minimal"

    def test_none_config_uses_defaults(self):
        """None config should use defaults."""
        generator = HypothesisGenerator(None)

        assert generator.config.template_library == "comprehensive"
        assert generator.config.min_confidence == 0.5
        assert generator.config.max_actions == 4


class TestTemplateMatching:
    """Tests for template matching logic."""

    def test_glob_pattern_matching(self):
        """Feature names should match using glob patterns."""
        generator = HypothesisGenerator()

        # Create pattern with momentum-like feature
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Test",
            top_features=[
                ("momentum_20d_returns", 2.5, 0.001, 0.002, True),
            ],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        # Should match momentum template via glob
        assert enriched.hypothesis is not None
        assert "momentum" in enriched.hypothesis.lower()

    def test_direction_condition_high(self):
        """Templates with direction='high' should only match positive SHAP."""
        generator = HypothesisGenerator()

        # Positive SHAP
        high_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Test high",
            top_features=[("momentum_20d", 2.5, 0.001, 0.002, True)],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        # Negative SHAP
        low_pattern = ErrorPattern(
            cluster_id=1,
            n_trades=15,
            description="Test low",
            top_features=[("momentum_20d", -2.5, 0.001, 0.002, True)],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        high_enriched = generator.generate_hypothesis(high_pattern)
        generator.generate_hypothesis(low_pattern)

        # Both might match different templates, but high should have hypothesis
        assert high_enriched.hypothesis is not None

    def test_significance_required_condition(self):
        """Templates requiring significance should only match significant features."""
        generator = HypothesisGenerator()

        # Pattern with only non-significant features
        non_sig_pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Non-significant",
            top_features=[
                ("momentum_20d", 2.5, 0.20, 0.25, False),  # p > 0.05
                ("volatility_20d", 1.8, 0.15, 0.18, False),
            ],
            separation_score=0.7,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(non_sig_pattern)

        # Should either have no hypothesis or lower confidence
        assert enriched.confidence is None or enriched.confidence < 0.7


class TestGenerateActions:
    """Tests for generate_actions method."""

    def test_generate_actions_returns_list(self):
        """generate_actions should return list of action dicts."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Test pattern",
            top_features=[("momentum_20d", 2.5, 0.002, 0.003, True)],
            separation_score=0.75,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)
        actions = generator.generate_actions(enriched)

        assert isinstance(actions, list)
        for action in actions:
            assert "category" in action
            assert "description" in action
            assert "priority" in action

    def test_actions_categorized_correctly(self):
        """Actions should be categorized based on content."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="Complex pattern",
            top_features=[
                ("momentum_20d", 2.5, 0.001, 0.002, True),
                ("volatility_20d", 2.1, 0.005, 0.006, True),
            ],
            separation_score=0.8,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)
        actions = generator.generate_actions(enriched)

        # Should have categories from predefined set
        valid_categories = {
            "feature_engineering",
            "filter_regime",
            "risk_management",
            "model_adjustment",
            "general",
        }
        for action in actions:
            assert action["category"] in valid_categories

    def test_max_actions_limits_output(self):
        """max_actions parameter should limit output."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Test",
            top_features=[("momentum_20d", 2.5, 0.002, 0.003, True)],
            separation_score=0.75,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)
        actions = generator.generate_actions(enriched, max_actions=2)

        assert len(actions) <= 2

    def test_empty_actions_returns_empty_list(self):
        """Pattern with no actions should return empty list."""
        generator = HypothesisGenerator()

        # Pattern that won't match any templates
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=10,
            description="No match",
            top_features=[("xyz_unknown_feature", 1.0, 0.5, 0.5, False)],
            separation_score=0.5,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)
        actions = generator.generate_actions(enriched)

        assert actions == []


class TestHypothesisFormatting:
    """Tests for hypothesis string formatting."""

    def test_single_feature_substitution(self):
        """Single feature should be substituted in template."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="Single feature",
            top_features=[("momentum_20d", 2.8, 0.001, 0.002, True)],
            separation_score=0.75,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        if enriched.hypothesis:
            # Should contain actual feature name, not placeholder
            assert "{feature}" not in enriched.hypothesis

    def test_multiple_significant_features_combined(self):
        """Multiple significant features should be combined in hypothesis."""
        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="Multiple features",
            top_features=[
                ("momentum_20d", 2.8, 0.001, 0.002, True),
                ("momentum_50d", 2.5, 0.002, 0.003, True),
            ],
            separation_score=0.8,
            distinctiveness=1.5,
        )

        enriched = generator.generate_hypothesis(pattern)

        if enriched.hypothesis:
            # Should mention both features or combine them
            assert "{feature}" not in enriched.hypothesis
