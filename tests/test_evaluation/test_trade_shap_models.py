"""Tests for Trade SHAP diagnostic models."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.trade_shap.models import (
    ClusteringResult,
    ErrorPattern,
    TradeShapExplanation,
    TradeShapResult,
)


class TestTradeShapExplanation:
    """Tests for TradeShapExplanation model."""

    def test_basic_creation(self):
        """Test creating a TradeShapExplanation."""
        explanation = TradeShapExplanation(
            trade_id="AAPL_2024-01-15T10:30:00",
            timestamp=datetime(2024, 1, 15, 10, 30),
            top_features=[
                ("momentum_20d", 0.342),
                ("volatility_10d", -0.215),
                ("rsi_14d", 0.108),
            ],
            feature_values={"momentum_20d": 1.235, "volatility_10d": 0.025, "rsi_14d": 65.0},
            shap_vector=np.array([0.342, -0.215, 0.108, 0.05, -0.02]),
        )

        assert explanation.trade_id == "AAPL_2024-01-15T10:30:00"
        assert len(explanation.top_features) == 3
        assert explanation.top_features[0][0] == "momentum_20d"
        assert explanation.shap_vector.shape == (5,)

    def test_feature_values_dict(self):
        """Test feature values dictionary access."""
        explanation = TradeShapExplanation(
            trade_id="test_trade",
            timestamp=datetime(2024, 1, 15),
            top_features=[("feature_a", 0.5)],
            feature_values={"feature_a": 100.0, "feature_b": 200.0},
            shap_vector=np.array([0.5, 0.3]),
        )

        assert explanation.feature_values["feature_a"] == 100.0
        assert explanation.feature_values["feature_b"] == 200.0

    def test_numpy_array_in_model(self):
        """Test that numpy arrays are properly handled."""
        shap_vec = np.random.randn(50)

        explanation = TradeShapExplanation(
            trade_id="test",
            timestamp=datetime.now(),
            top_features=[("f1", 0.1)],
            feature_values={"f1": 1.0},
            shap_vector=shap_vec,
        )

        assert np.allclose(explanation.shap_vector, shap_vec)


class TestClusteringResult:
    """Tests for ClusteringResult model."""

    def test_basic_creation(self):
        """Test creating a ClusteringResult."""
        result = ClusteringResult(
            n_clusters=3,
            cluster_assignments=[0, 1, 2, 0, 1],
            linkage_matrix=np.array([[0, 1, 0.5, 2], [2, 3, 0.8, 3]]),
            centroids=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            silhouette_score=0.65,
            cluster_sizes=[2, 2, 1],
            distance_metric="euclidean",
            linkage_method="ward",
        )

        assert result.n_clusters == 3
        assert len(result.cluster_assignments) == 5
        assert result.silhouette_score == 0.65
        assert result.distance_metric == "euclidean"

    def test_optional_scores(self):
        """Test optional clustering quality scores."""
        result = ClusteringResult(
            n_clusters=2,
            cluster_assignments=[0, 1],
            linkage_matrix=np.array([[0, 1, 0.5, 2]]),
            centroids=np.array([[0.1], [0.2]]),
            silhouette_score=0.5,
            davies_bouldin_score=0.8,
            calinski_harabasz_score=150.5,
            cluster_sizes=[1, 1],
            distance_metric="cosine",
            linkage_method="average",
        )

        assert result.davies_bouldin_score == 0.8
        assert result.calinski_harabasz_score == 150.5

    def test_none_optional_scores(self):
        """Test with optional scores set to None."""
        result = ClusteringResult(
            n_clusters=2,
            cluster_assignments=[0, 1],
            linkage_matrix=np.array([[0, 1, 0.5, 2]]),
            centroids=np.array([[0.1], [0.2]]),
            silhouette_score=0.5,
            cluster_sizes=[1, 1],
            distance_metric="euclidean",
            linkage_method="ward",
        )

        assert result.davies_bouldin_score is None
        assert result.calinski_harabasz_score is None


class TestErrorPattern:
    """Tests for ErrorPattern model."""

    def test_basic_creation(self):
        """Test creating an ErrorPattern."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="High momentum (up 0.45) + Low volatility (down -0.32) -> Losses",
            top_features=[
                ("momentum_20d", 0.45, 0.001, 0.002, True),
                ("volatility_10d", -0.32, 0.003, 0.004, True),
            ],
            separation_score=1.2,
            distinctiveness=1.8,
        )

        assert pattern.cluster_id == 0
        assert pattern.n_trades == 15
        assert len(pattern.top_features) == 2
        assert pattern.separation_score == 1.2

    def test_with_hypothesis_and_actions(self):
        """Test pattern with hypothesis and actions."""
        pattern = ErrorPattern(
            cluster_id=1,
            n_trades=22,
            description="High RSI pattern",
            top_features=[("rsi_14", 0.38, 0.001, 0.001, True)],
            separation_score=0.9,
            distinctiveness=1.5,
            hypothesis="Trades entering overbought conditions",
            actions=[
                "Add overbought filter: skip trades when RSI > 70",
                "Consider volume profile",
            ],
            confidence=0.85,
        )

        assert pattern.hypothesis is not None
        assert len(pattern.actions) == 2
        assert pattern.confidence == 0.85

    def test_to_dict(self):
        """Test converting ErrorPattern to dictionary."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=10,
            description="Test pattern",
            top_features=[("feature_a", 0.5, 0.01, 0.02, True)],
            separation_score=1.0,
            distinctiveness=1.5,
        )

        result = pattern.to_dict()

        assert result["cluster_id"] == 0
        assert result["n_trades"] == 10
        assert result["description"] == "Test pattern"
        assert len(result["top_features"]) == 1
        assert result["top_features"][0]["feature_name"] == "feature_a"

    def test_to_dict_with_actions(self):
        """Test to_dict with actions."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=5,
            description="Test",
            top_features=[],
            separation_score=1.0,
            distinctiveness=1.0,
            actions=["Action 1", "Action 2"],
        )

        result = pattern.to_dict()
        assert result["actions"] == ["Action 1", "Action 2"]

    def test_to_dict_empty_actions(self):
        """Test to_dict with no actions."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=5,
            description="Test",
            top_features=[],
            separation_score=1.0,
            distinctiveness=1.0,
        )

        result = pattern.to_dict()
        assert result["actions"] == []

    def test_summary_simple(self):
        """Test simple summary generation."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=15,
            description="High momentum pattern",
            top_features=[],
            separation_score=1.0,
            distinctiveness=1.5,
        )

        summary = pattern.summary()

        assert "Pattern 0" in summary
        assert "15 trades" in summary
        assert "High momentum pattern" in summary

    def test_summary_with_actions(self):
        """Test detailed summary with actions."""
        pattern = ErrorPattern(
            cluster_id=1,
            n_trades=20,
            description="RSI overbought pattern",
            top_features=[],
            separation_score=1.0,
            distinctiveness=1.5,
            hypothesis="Trading into overbought conditions",
            actions=["Add RSI filter", "Reduce position size"],
            confidence=0.75,
        )

        summary = pattern.summary(include_actions=True)

        assert "Pattern 1" in summary
        assert "20 trades" in summary
        assert "Hypothesis:" in summary
        assert "Actions:" in summary
        assert "75%" in summary

    def test_summary_no_hypothesis(self):
        """Test summary without hypothesis returns simple format."""
        pattern = ErrorPattern(
            cluster_id=0,
            n_trades=10,
            description="Simple pattern",
            top_features=[],
            separation_score=1.0,
            distinctiveness=1.0,
        )

        # With include_actions=True but no hypothesis
        summary = pattern.summary(include_actions=True)

        # Should still be simple one-liner when no hypothesis
        assert "Pattern 0: 10 trades - Simple pattern" == summary

    def test_validation_cluster_id_non_negative(self):
        """Test that cluster_id must be non-negative."""
        with pytest.raises(ValueError):
            ErrorPattern(
                cluster_id=-1,
                n_trades=10,
                description="Test",
                top_features=[],
                separation_score=1.0,
                distinctiveness=1.0,
            )

    def test_validation_n_trades_positive(self):
        """Test that n_trades must be positive."""
        with pytest.raises(ValueError):
            ErrorPattern(
                cluster_id=0,
                n_trades=0,
                description="Test",
                top_features=[],
                separation_score=1.0,
                distinctiveness=1.0,
            )

    def test_validation_confidence_range(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            ErrorPattern(
                cluster_id=0,
                n_trades=10,
                description="Test",
                top_features=[],
                separation_score=1.0,
                distinctiveness=1.0,
                confidence=1.5,  # Invalid: > 1
            )


class TestTradeShapResult:
    """Tests for TradeShapResult model."""

    def test_basic_creation(self):
        """Test creating a TradeShapResult."""
        result = TradeShapResult(
            n_trades_analyzed=20,
            n_trades_explained=18,
            n_trades_failed=2,
        )

        assert result.n_trades_analyzed == 20
        assert result.n_trades_explained == 18
        assert result.n_trades_failed == 2
        assert len(result.explanations) == 0
        assert len(result.error_patterns) == 0

    def test_with_explanations(self):
        """Test result with explanations."""
        explanations = [
            TradeShapExplanation(
                trade_id=f"trade_{i}",
                timestamp=datetime(2024, 1, 15),
                top_features=[("f1", 0.1)],
                feature_values={"f1": 1.0},
                shap_vector=np.array([0.1]),
            )
            for i in range(3)
        ]

        result = TradeShapResult(
            n_trades_analyzed=3,
            n_trades_explained=3,
            n_trades_failed=0,
            explanations=explanations,
        )

        assert len(result.explanations) == 3
        assert result.explanations[0].trade_id == "trade_0"

    def test_with_failed_trades(self):
        """Test result with failed trades."""
        result = TradeShapResult(
            n_trades_analyzed=5,
            n_trades_explained=3,
            n_trades_failed=2,
            failed_trades=[
                ("trade_3", "Missing feature data"),
                ("trade_4", "SHAP computation failed"),
            ],
        )

        assert len(result.failed_trades) == 2
        assert result.failed_trades[0][0] == "trade_3"
        assert "Missing" in result.failed_trades[0][1]

    def test_with_error_patterns(self):
        """Test result with error patterns."""
        patterns = [
            ErrorPattern(
                cluster_id=i,
                n_trades=10,
                description=f"Pattern {i}",
                top_features=[],
                separation_score=1.0,
                distinctiveness=1.0,
            )
            for i in range(2)
        ]

        result = TradeShapResult(
            n_trades_analyzed=20,
            n_trades_explained=20,
            n_trades_failed=0,
            error_patterns=patterns,
        )

        assert len(result.error_patterns) == 2
        assert result.error_patterns[0].cluster_id == 0

    def test_empty_defaults(self):
        """Test that lists default to empty."""
        result = TradeShapResult(
            n_trades_analyzed=0,
            n_trades_explained=0,
            n_trades_failed=0,
        )

        assert result.explanations == []
        assert result.failed_trades == []
        assert result.error_patterns == []
