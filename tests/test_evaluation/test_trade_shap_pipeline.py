"""Tests for Trade-SHAP analysis pipeline.

Tests cover TradeShapPipeline initialization, individual/batch explanation
delegation, and the full analyze_worst_trades workflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config import (
    TradeAlignmentSettings,
    TradeClusteringSettings,
    TradeConfig,
)
from ml4t.diagnostic.evaluation.trade_shap.models import (
    TradeExplainFailure,
    TradeShapExplanation,
    TradeShapResult,
)
from ml4t.diagnostic.evaluation.trade_shap.pipeline import TradeShapPipeline


class MockTradeMetrics:
    """Mock TradeMetrics for testing without full dependency."""

    def __init__(self, timestamp: datetime, symbol: str = "TEST", pnl: float = -5.0) -> None:
        self.timestamp = timestamp
        self.symbol = symbol
        self.entry_price = 100.0
        self.exit_price = 100.0 + pnl
        self.pnl = pnl
        self.duration = timedelta(hours=1)
        self.direction = "long"


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Create sample features DataFrame with timestamps."""
    n_samples = 100
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]

    np.random.seed(42)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "momentum": np.random.randn(n_samples),
            "volatility": np.random.randn(n_samples),
            "trend": np.random.randn(n_samples),
            "volume": np.random.randn(n_samples),
            "spread": np.random.randn(n_samples),
        }
    )


@pytest.fixture
def sample_shap_values() -> np.ndarray:
    """Create SHAP values matching sample_features."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names matching sample data."""
    return ["momentum", "volatility", "trend", "volume", "spread"]


@pytest.fixture
def pipeline(sample_features, sample_shap_values, feature_names) -> TradeShapPipeline:
    """Create a default pipeline for testing."""
    return TradeShapPipeline(
        features_df=sample_features,
        shap_values=sample_shap_values,
        feature_names=feature_names,
    )


class TestTradeConfigForPipeline:
    """Tests for canonical pipeline config model."""

    def test_default_values(self):
        """Default config should have sensible defaults."""
        config = TradeConfig()

        assert config.alignment.tolerance == 300
        assert config.alignment.mode == "entry"
        assert config.alignment.missing_strategy == "skip"
        assert config.alignment.top_n_features is None
        assert config.clustering.normalization == "l2"
        assert config.clustering.min_cluster_size == 5

    def test_custom_values(self):
        """Custom config values should be stored."""
        config = TradeConfig(
            alignment=TradeAlignmentSettings(
                tolerance=60,
                mode="nearest",
                missing_strategy="zero",
                top_n_features=5,
            ),
            clustering=TradeClusteringSettings(normalization="l1"),
        )

        assert config.alignment.tolerance == 60
        assert config.alignment.mode == "nearest"
        assert config.alignment.missing_strategy == "zero"
        assert config.alignment.top_n_features == 5
        assert config.clustering.normalization == "l1"


class TestTradeShapPipelineInit:
    """Tests for TradeShapPipeline initialization."""

    def test_default_config_creates_pipeline(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Pipeline with default config should initialize correctly."""
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
        )

        assert pipeline.features_df is sample_features
        assert pipeline.feature_names == feature_names
        assert pipeline.config is not None
        assert pipeline.explainer is not None
        assert pipeline.clusterer is not None
        assert pipeline.characterizer is not None
        assert pipeline.hypothesis_generator is not None

    def test_custom_config_applied(self, sample_features, sample_shap_values, feature_names):
        """Custom config should be applied to pipeline."""
        config = TradeConfig(
            alignment=TradeAlignmentSettings(top_n_features=3),
            clustering=TradeClusteringSettings(normalization="l1"),
        )

        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        assert pipeline.config.alignment.top_n_features == 3
        assert pipeline.config.clustering.normalization == "l1"
        assert pipeline.explainer.top_n_features == 3


class TestExplainTrade:
    """Tests for explain_trade() delegation."""

    def test_explain_trade_delegates_to_explainer(self, pipeline):
        """explain_trade should delegate to explainer."""
        trade_time = datetime(2024, 1, 1, 19, 0, 0)  # Row 10
        trade = MockTradeMetrics(timestamp=trade_time)

        result = pipeline.explain_trade(trade)

        assert isinstance(result, TradeShapExplanation)
        assert result.timestamp == trade_time

    def test_explain_trade_failure_returns_failure(self, pipeline):
        """Non-matching trade should return TradeExplainFailure."""
        trade_time = datetime(2025, 6, 1, 0, 0, 0)  # Outside range
        trade = MockTradeMetrics(timestamp=trade_time)

        result = pipeline.explain_trade(trade)

        assert isinstance(result, TradeExplainFailure)


class TestExplainTrades:
    """Tests for explain_trades() delegation."""

    def test_explain_trades_delegates_to_explainer(self, pipeline):
        """explain_trades should delegate to explainer."""
        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0)),
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 15, 0, 0)),
        ]

        explanations, failures = pipeline.explain_trades(trades)

        assert len(explanations) == 2
        assert len(failures) == 0


class TestAnalyzeWorstTrades:
    """Tests for analyze_worst_trades() pipeline method."""

    def test_empty_trades_returns_empty_result(self, pipeline):
        """Empty trades list should return empty result."""
        result = pipeline.analyze_worst_trades([])

        assert isinstance(result, TradeShapResult)
        assert result.n_trades_analyzed == 0
        assert result.n_trades_explained == 0
        assert result.n_trades_failed == 0
        assert result.explanations == []
        assert result.error_patterns == []

    def test_n_zero_returns_empty_result(self, pipeline):
        """n=0 should return empty result."""
        trades = [MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0))]

        result = pipeline.analyze_worst_trades(trades, n=0)

        assert result.n_trades_analyzed == 0
        assert result.explanations == []

    def test_all_failures_returns_result_without_patterns(self, pipeline):
        """All alignment failures should return result without patterns."""
        trades = [MockTradeMetrics(timestamp=datetime(2025, 6, i, 0, 0, 0)) for i in range(1, 6)]

        result = pipeline.analyze_worst_trades(trades)

        assert result.n_trades_analyzed == 5
        assert result.n_trades_explained == 0
        assert result.n_trades_failed == 5
        assert result.explanations == []
        assert len(result.failed_trades) == 5
        assert result.error_patterns == []

    def test_successful_analysis_with_patterns(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Successful analysis should produce patterns."""
        # Lower min_trades for testing
        config = TradeConfig(min_trades_for_clustering=5)
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        # Create trades matching timestamps (use timedelta to avoid hour overflow)
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        trades = [
            MockTradeMetrics(timestamp=base_time + timedelta(hours=i), pnl=-float(i))
            for i in range(20)
        ]

        result = pipeline.analyze_worst_trades(trades)

        assert result.n_trades_analyzed == 20
        assert result.n_trades_explained == 20
        assert result.n_trades_failed == 0
        assert len(result.explanations) == 20
        # Patterns may or may not be generated depending on clustering
        assert isinstance(result.error_patterns, list)

    def test_n_limits_trades_analyzed(self, pipeline):
        """n parameter should limit trades analyzed."""
        trades = [MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10 + i, 0, 0)) for i in range(10)]

        result = pipeline.analyze_worst_trades(trades, n=5)

        assert result.n_trades_analyzed == 5
        assert result.n_trades_explained == 5

    def test_insufficient_trades_for_clustering(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Fewer trades than min_trades_for_clustering should skip clustering."""
        config = TradeConfig(min_trades_for_clustering=50)
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9 + i, 0, 0))
            for i in range(10)  # Less than 50
        ]

        result = pipeline.analyze_worst_trades(trades)

        assert result.n_trades_explained == 10
        assert result.error_patterns == []  # No patterns due to insufficient trades

    def test_normalization_applied(self, sample_features, sample_shap_values, feature_names):
        """Normalization should be applied when configured."""
        config = TradeConfig(
            min_trades_for_clustering=5,
            clustering=TradeClusteringSettings(normalization="l2"),
        )
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        trades = [MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9 + i, 0, 0)) for i in range(10)]

        # Should not raise - normalization is applied internally
        result = pipeline.analyze_worst_trades(trades)
        assert result.n_trades_explained == 10

    def test_normalization_skipped_when_none(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Normalization should be skipped when None."""
        config = TradeConfig(
            min_trades_for_clustering=5,
            clustering=TradeClusteringSettings(normalization=None),
        )
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        trades = [MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9 + i, 0, 0)) for i in range(10)]

        result = pipeline.analyze_worst_trades(trades)
        assert result.n_trades_explained == 10

    def test_mixed_success_and_failure(self, pipeline):
        """Mixed matching/non-matching trades should be handled."""
        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0)),  # Match
            MockTradeMetrics(timestamp=datetime(2025, 6, 1, 0, 0, 0)),  # No match
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 15, 0, 0)),  # Match
        ]

        result = pipeline.analyze_worst_trades(trades)

        assert result.n_trades_analyzed == 3
        assert result.n_trades_explained == 2
        assert result.n_trades_failed == 1


class TestGenerateActions:
    """Tests for generate_actions() method."""

    def test_generate_actions_raises_not_implemented(self, pipeline):
        """generate_actions should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Use hypothesis_generator"):
            pipeline.generate_actions()


class TestEdgeCases:
    """Additional edge case tests."""

    def test_single_trade_analysis(self, pipeline):
        """Single trade should work correctly."""
        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0))

        result = pipeline.analyze_worst_trades([trade])

        assert result.n_trades_analyzed == 1
        assert result.n_trades_explained == 1
        assert result.error_patterns == []  # Single trade can't cluster

    def test_result_contains_trade_ids(self, pipeline):
        """Result should contain correct trade IDs."""
        trades = [
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 0, 0), symbol="BTC"),
            MockTradeMetrics(timestamp=datetime(2024, 1, 1, 11, 0, 0), symbol="ETH"),
        ]

        result = pipeline.analyze_worst_trades(trades)

        assert len(result.explanations) == 2
        trade_ids = [exp.trade_id for exp in result.explanations]
        assert "BTC_" in trade_ids[0]
        assert "ETH_" in trade_ids[1]

    def test_failed_trades_contains_reason(self, pipeline):
        """Failed trades should contain reason."""
        trade = MockTradeMetrics(timestamp=datetime(2025, 6, 1, 0, 0, 0))

        result = pipeline.analyze_worst_trades([trade])

        assert len(result.failed_trades) == 1
        trade_id, reason = result.failed_trades[0]
        assert reason == "alignment_missing"

    def test_different_normalization_methods(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Different normalization methods should work."""
        for method in ["l1", "l2", "standardize", None]:
            config = TradeConfig(
                min_trades_for_clustering=5,
                clustering=TradeClusteringSettings(normalization=method),
            )
            pipeline = TradeShapPipeline(
                features_df=sample_features,
                shap_values=sample_shap_values,
                feature_names=feature_names,
                config=config,
            )

            trades = [
                MockTradeMetrics(timestamp=datetime(2024, 1, 1, 9 + i, 0, 0)) for i in range(10)
            ]

            result = pipeline.analyze_worst_trades(trades)
            assert result.n_trades_explained == 10, f"Failed for method={method}"

    def test_pipeline_with_nearest_alignment(
        self, sample_features, sample_shap_values, feature_names
    ):
        """Pipeline with nearest alignment should find nearby timestamps."""
        config = TradeConfig(
            alignment=TradeAlignmentSettings(
                tolerance=1800,  # 30 minutes
                mode="nearest",
            )
        )
        pipeline = TradeShapPipeline(
            features_df=sample_features,
            shap_values=sample_shap_values,
            feature_names=feature_names,
            config=config,
        )

        # Trade 15 minutes after a known timestamp
        trade = MockTradeMetrics(timestamp=datetime(2024, 1, 1, 10, 15, 0))

        result = pipeline.analyze_worst_trades([trade])

        assert result.n_trades_explained == 1
        assert result.n_trades_failed == 0

    def test_pipeline_preserves_explanation_order(self, pipeline):
        """Explanations should preserve trade order."""
        timestamps = [
            datetime(2024, 1, 1, 20, 0, 0),
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 1, 15, 0, 0),
        ]
        trades = [MockTradeMetrics(timestamp=ts) for ts in timestamps]

        result = pipeline.analyze_worst_trades(trades)

        # Order should be preserved
        for i, exp in enumerate(result.explanations):
            assert exp.timestamp == timestamps[i]
