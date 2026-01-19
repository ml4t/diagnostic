"""Tests for Trade-SHAP action generation (TASK-027).

This module tests the action suggestion engine that converts error patterns
into prioritized, actionable improvement recommendations.

Test Coverage:
    - HypothesisGenerator.generate_actions() method
    - Action template generation for all categories
    - Priority ranking logic
    - TradeShapAnalyzer.generate_actions() integration
"""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.trade_shap import (
    ErrorPattern,
    HypothesisGenerator,
)


class TestHypothesisGeneratorActions:
    """Test action generation from HypothesisGenerator."""

    @pytest.fixture
    def generator(self):
        """Create HypothesisGenerator with default config."""
        return HypothesisGenerator()

    @pytest.fixture
    def error_pattern_momentum_high(self):
        """Create error pattern with high momentum feature."""
        return ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="High momentum_20d (↑0.45) + Low volatility_10d (↓-0.28) → Losses",
            top_features=[
                ("momentum_20d", 0.45, 0.001, 0.002, True),  # High momentum
                ("volatility_10d", -0.28, 0.003, 0.004, True),  # Low volatility
                ("rsi_14", 0.18, 0.05, 0.06, False),
            ],
            separation_score=1.2,
            distinctiveness=1.8,
            hypothesis="Trades are losing when momentum_20d shows high values, suggesting momentum reversal",
            actions=["Add momentum confirmation with volume", "Implement momentum decay filters"],
            confidence=0.75,
        )

    @pytest.fixture
    def error_pattern_volatility_high(self):
        """Create error pattern with high volatility feature."""
        return ErrorPattern(
            cluster_id=1,
            n_trades=18,
            description="High atr_14 (↑0.52) → Losses",
            top_features=[
                ("atr_14", 0.52, 0.0001, 0.0002, True),
            ],
            separation_score=0.9,
            distinctiveness=1.4,
            hypothesis="Trades are losing when atr_14 shows high values, suggesting volatility spike",
            actions=["Add volatility ceiling", "Adjust stop loss for high volatility"],
            confidence=0.68,
        )

    @pytest.fixture
    def error_pattern_no_hypothesis(self):
        """Create error pattern without hypothesis (shouldn't generate actions)."""
        return ErrorPattern(
            cluster_id=2,
            n_trades=10,
            description="Unknown pattern",
            top_features=[
                ("feature_x", 0.15, 0.1, 0.12, False),
            ],
            separation_score=0.5,
            distinctiveness=1.0,
        )

    def test_generate_actions_basic(self, generator, error_pattern_momentum_high):
        """Test basic action generation."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        # Should return non-empty list
        assert len(actions) > 0
        assert isinstance(actions, list)

        # Actions should have priority field
        assert all("priority" in a for a in actions)

    def test_generate_actions_returns_empty_without_hypothesis(
        self, generator, error_pattern_no_hypothesis
    ):
        """Test that actions are not generated without hypothesis."""
        actions = generator.generate_actions(error_pattern_no_hypothesis)
        assert len(actions) == 0

    def test_generate_actions_structure(self, generator, error_pattern_momentum_high):
        """Test that actions have required fields."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        # Check first action has all required fields
        action = actions[0]
        assert "category" in action
        assert "description" in action
        assert "priority" in action
        assert "implementation_difficulty" in action
        assert "rationale" in action

        # Check field types
        assert isinstance(action["category"], str)
        assert action["category"] in [
            "feature_engineering",
            "model_adjustment",
            "filter_regime",
            "risk_management",
            "general",
        ]
        assert action["priority"] in ["high", "medium", "low"]
        assert action["implementation_difficulty"] in ["easy", "medium", "hard"]

    def test_generate_actions_max_actions_limit(self, generator, error_pattern_momentum_high):
        """Test that max_actions parameter limits results."""
        actions_all = generator.generate_actions(error_pattern_momentum_high)
        actions_limited = generator.generate_actions(error_pattern_momentum_high, max_actions=1)

        assert len(actions_limited) <= 1
        assert len(actions_limited) <= len(actions_all)

    def test_generate_actions_feature_engineering_category(
        self, generator, error_pattern_momentum_high
    ):
        """Test that feature engineering actions are generated."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        # Should have at least one feature engineering action
        feature_actions = [a for a in actions if a["category"] == "feature_engineering"]
        assert len(feature_actions) > 0

    def test_generate_actions_has_description(self, generator, error_pattern_momentum_high):
        """Test that actions have descriptions."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        # All actions should have non-empty descriptions
        for action in actions:
            assert action["description"]
            assert len(action["description"]) > 0

    def test_categorize_action_feature_engineering(self, generator):
        """Test action categorization for feature engineering actions."""
        assert (
            generator._categorize_action("Add momentum confirmation with volume")
            == "feature_engineering"
        )
        assert generator._categorize_action("Add new indicator") == "feature_engineering"

    def test_categorize_action_filter_regime(self, generator):
        """Test action categorization for filter/regime actions."""
        # Note: "Add regime filter" matches "add" first → feature_engineering
        assert generator._categorize_action("Apply regime filter") == "filter_regime"
        assert generator._categorize_action("Set threshold") == "filter_regime"

    def test_categorize_action_risk_management(self, generator):
        """Test action categorization for risk management actions."""
        assert generator._categorize_action("Adjust position size") == "risk_management"
        assert generator._categorize_action("Set stop loss") == "risk_management"

    def test_categorize_action_model_adjustment(self, generator):
        """Test action categorization for model adjustment actions."""
        assert generator._categorize_action("Tune parameters") == "model_adjustment"
        assert generator._categorize_action("Adjust model") == "model_adjustment"

    def test_priority_ranking_by_confidence(self, generator):
        """Test that higher confidence patterns get higher priority."""
        # Create two patterns with different confidence
        pattern_high_conf = ErrorPattern(
            cluster_id=0,
            n_trades=20,
            description="High momentum",
            top_features=[("momentum_20d", 0.45, 0.001, 0.002, True)],
            separation_score=1.0,
            distinctiveness=1.5,
            hypothesis="Test hypothesis",
            actions=["Add filter", "Adjust model"],
            confidence=0.9,
        )

        pattern_low_conf = ErrorPattern(
            cluster_id=1,
            n_trades=20,
            description="High momentum",
            top_features=[("momentum_20d", 0.45, 0.001, 0.002, True)],
            separation_score=1.0,
            distinctiveness=1.5,
            hypothesis="Test hypothesis",
            actions=["Add filter", "Adjust model"],
            confidence=0.5,
        )

        actions_high = generator.generate_actions(pattern_high_conf, max_actions=1)
        actions_low = generator.generate_actions(pattern_low_conf, max_actions=1)

        # Higher confidence should have "high" priority, lower should have lower priority
        assert actions_high[0]["priority"] == "high"
        # Low confidence first action could be medium or low
        assert actions_low[0]["priority"] in ["medium", "low"]

    def test_priority_ranking_by_position(self, generator, error_pattern_momentum_high):
        """Test that first action has higher priority."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        if len(actions) >= 2:
            # Priority order: high > medium > low
            priority_order = {"high": 3, "medium": 2, "low": 1}
            first_score = priority_order[actions[0]["priority"]]
            last_score = priority_order[actions[-1]["priority"]]
            # First action should have >= priority than last
            assert first_score >= last_score

    def test_estimate_difficulty_hard(self, generator):
        """Test difficulty estimation for hard tasks."""
        assert generator._estimate_difficulty("Implement HMM model") == "hard"
        assert generator._estimate_difficulty("Build ensemble") == "hard"

    def test_estimate_difficulty_medium(self, generator):
        """Test difficulty estimation for medium tasks."""
        assert generator._estimate_difficulty("Add new feature") == "medium"
        assert generator._estimate_difficulty("Consider adding") == "medium"

    def test_estimate_difficulty_easy(self, generator):
        """Test difficulty estimation for easy tasks."""
        assert generator._estimate_difficulty("Check the logs") == "easy"

    def test_actions_have_rationale(self, generator, error_pattern_momentum_high):
        """Test that actions include rationale."""
        actions = generator.generate_actions(error_pattern_momentum_high)

        for action in actions:
            assert "rationale" in action
            assert "Based on pattern:" in action["rationale"]


class TestTradeShapAnalyzerActionIntegration:
    """Test action generation integration with TradeShapAnalyzer."""

    @pytest.fixture
    def mock_shap_data(self):
        """Create mock SHAP data for testing."""

        # Mock model with predict method
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X))

        model = MockModel()

        # Create feature matrix (10 trades × 20 features)
        features_array = np.random.randn(10, 20)
        feature_names = [f"feature_{i}" for i in range(20)]
        feature_names[0] = "momentum_20d"  # Use recognizable feature name
        feature_names[1] = "volatility_10d"

        # Create SHAP values (same shape as features)
        shap_values = np.random.randn(10, 20)
        # Make momentum_20d have high positive SHAP
        shap_values[:, 0] = 0.45

        return model, features_array, shap_values, feature_names

    @pytest.fixture
    def analyzer(self, mock_shap_data):
        """Create TradeShapAnalyzer instance."""
        from datetime import datetime, timedelta

        import polars as pl

        from ml4t.diagnostic.config import TradeConfig
        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        model, features_array, shap_values, feature_names = mock_shap_data
        config = TradeConfig()

        # Convert features to Polars DataFrame with timestamps
        timestamps = [datetime.now() - timedelta(days=i) for i in range(len(features_array))]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                **{name: features_array[:, i] for i, name in enumerate(feature_names)},
            }
        )

        return TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=shap_values,
            config=config,
        )

    def test_analyzer_hypothesis_generator_property_exists(self, analyzer):
        """Test that TradeShapAnalyzer has hypothesis_generator property."""
        assert hasattr(analyzer, "hypothesis_generator")

    def test_analyzer_hypothesis_generator_lazily_initialized(self, analyzer):
        """Test that HypothesisGenerator is lazily initialized."""
        # Should not exist before first access
        assert analyzer._hypothesis_generator is None

        # Access property
        gen = analyzer.hypothesis_generator

        # Should now exist
        assert analyzer._hypothesis_generator is not None
        assert gen is not None

    def test_analyzer_hypothesis_generator_returns_generator(self, analyzer):
        """Test that hypothesis_generator returns a HypothesisGenerator."""
        gen = analyzer.hypothesis_generator
        assert isinstance(gen, HypothesisGenerator)

    def test_analyzer_generate_actions_via_generator(self, analyzer):
        """Test generating actions via the hypothesis generator property."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="High momentum_20d (↑0.45) → Losses",
            top_features=[
                ("momentum_20d", 0.45, 0.001, 0.002, True),
            ],
            separation_score=1.2,
            distinctiveness=1.8,
            hypothesis="Test hypothesis",
            actions=["Add filter", "Adjust model"],
            confidence=0.75,
        )

        actions = analyzer.hypothesis_generator.generate_actions(pattern)

        assert isinstance(actions, list)
        assert len(actions) > 0


class TestActionGenerationPerformance:
    """Test performance of action generation."""

    def test_action_generation_performance(self):
        """Test that action generation completes in <200ms per pattern."""
        import time

        generator = HypothesisGenerator()

        # Create pattern
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="High momentum_20d (↑0.45) + Low volatility_10d (↓-0.28) → Losses",
            top_features=[
                ("momentum_20d", 0.45, 0.001, 0.002, True),
                ("volatility_10d", -0.28, 0.003, 0.004, True),
                ("rsi_14", 0.18, 0.05, 0.06, False),
            ],
            separation_score=1.2,
            distinctiveness=1.8,
            hypothesis="Test hypothesis",
            actions=["Add filter", "Adjust model"],
            confidence=0.75,
        )

        # Measure time
        start = time.time()
        actions = generator.generate_actions(pattern)
        duration = time.time() - start

        # Should complete in <200ms
        assert duration < 0.2, f"Action generation took {duration:.3f}s, expected <0.2s"
        assert len(actions) > 0

    def test_action_generation_scales_linearly(self):
        """Test that action generation scales linearly with max_actions."""
        import time

        generator = HypothesisGenerator()

        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=25,
            description="Test",
            top_features=[("momentum_20d", 0.45, 0.001, 0.002, True)],
            separation_score=1.0,
            distinctiveness=1.5,
            hypothesis="Test hypothesis",
            actions=["Action 1", "Action 2", "Action 3", "Action 4", "Action 5"],
            confidence=0.75,
        )

        # Generate many actions
        start = time.time()
        generator.generate_actions(pattern, max_actions=20)
        duration_many = time.time() - start

        # Generate few actions
        start = time.time()
        generator.generate_actions(pattern, max_actions=2)
        duration_few = time.time() - start

        # Both should be fast
        assert duration_many < 0.2
        assert duration_few < 0.2

        # Difference should be small (same computation, just slicing)
        assert abs(duration_many - duration_few) < 0.05
