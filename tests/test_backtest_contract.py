"""Tests for ML4T Backtest integration contract."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from ml4t.diagnostic.integration import (
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    EnvironmentType,
    EvaluationExport,
    PromotionWorkflow,
    StrategyMetadata,
)


class TestEnvironmentType:
    """Test EnvironmentType enum."""

    def test_all_values(self):
        """Test all environment types are defined."""
        expected = {"backtest", "paper", "live"}
        actual = {e.value for e in EnvironmentType}
        assert actual == expected

    def test_enum_is_string(self):
        """Test EnvironmentType inherits from str."""
        assert isinstance(EnvironmentType.BACKTEST, str)
        assert EnvironmentType.BACKTEST == "backtest"


class TestComparisonType:
    """Test ComparisonType enum."""

    def test_all_values(self):
        """Test all comparison types are defined."""
        expected = {"bayesian", "bootstrap", "parametric", "cusum"}
        actual = {c.value for c in ComparisonType}
        assert actual == expected

    def test_enum_is_string(self):
        """Test ComparisonType inherits from str."""
        assert isinstance(ComparisonType.BAYESIAN, str)
        assert ComparisonType.BAYESIAN == "bayesian"


class TestStrategyMetadata:
    """Test StrategyMetadata schema."""

    def test_basic_metadata(self):
        """Test creating basic metadata."""
        start = datetime(2020, 1, 1)
        end = datetime(2023, 12, 31)
        metadata = StrategyMetadata(
            strategy_id="momentum_v1",
            environment=EnvironmentType.BACKTEST,
            start_date=start,
            end_date=end,
        )
        assert metadata.strategy_id == "momentum_v1"
        assert metadata.environment == EnvironmentType.BACKTEST
        assert metadata.start_date == start
        assert metadata.end_date == end
        assert metadata.version is None
        assert metadata.config_hash is None

    def test_full_metadata(self):
        """Test metadata with all fields."""
        metadata = StrategyMetadata(
            strategy_id="momentum_rsi",
            version="1.2.3",
            environment=EnvironmentType.LIVE,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            config_hash="abc123",
            description="Momentum strategy with RSI filter",
            tags={"asset_class": "crypto", "timeframe": "1h"},
        )
        assert metadata.version == "1.2.3"
        assert metadata.config_hash == "abc123"
        assert metadata.description == "Momentum strategy with RSI filter"
        assert metadata.tags == {"asset_class": "crypto", "timeframe": "1h"}

    def test_metadata_serialization(self):
        """Test metadata can be serialized to dict."""
        metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.PAPER,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        data = metadata.model_dump()
        assert data["strategy_id"] == "test"
        assert data["environment"] == "paper"


class TestEvaluationExport:
    """Test EvaluationExport schema."""

    def test_basic_export(self):
        """Test creating basic export."""
        metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        metrics = {
            "sharpe_ratio": 1.85,
            "cagr": 0.24,
            "max_drawdown": -0.18,
        }
        export = EvaluationExport(metadata=metadata, metrics=metrics)
        assert export.metadata == metadata
        assert export.metrics == metrics
        assert export.diagnostics is None
        assert export.sharpe_framework is None
        assert isinstance(export.timestamp, datetime)

    def test_full_export(self):
        """Test export with all fields."""
        metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        metrics = {"sharpe_ratio": 1.85}
        diagnostics = {
            "stationarity": {"adf_pvalue": 0.01, "stationary": True},
            "correlation": {"max_correlation": 0.65},
        }
        sharpe_framework = {
            "psr": 0.92,
            "dsr": 1.45,
            "min_trl": 250,
        }
        export = EvaluationExport(
            metadata=metadata,
            metrics=metrics,
            diagnostics=diagnostics,
            sharpe_framework=sharpe_framework,
            diagnostic_version="2.0.0",
        )
        assert export.diagnostics == diagnostics
        assert export.sharpe_framework == sharpe_framework
        assert export.diagnostic_version == "2.0.0"

    def test_to_dict(self):
        """Test export to dictionary."""
        metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        export = EvaluationExport(metadata=metadata, metrics={"sharpe_ratio": 1.85})
        data = export.to_dict()
        assert isinstance(data, dict)
        assert "metadata" in data
        assert "metrics" in data
        assert data["metrics"]["sharpe_ratio"] == 1.85

    def test_to_json(self):
        """Test export to JSON string."""
        metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        export = EvaluationExport(metadata=metadata, metrics={"sharpe_ratio": 1.85})
        json_str = export.to_json()
        assert isinstance(json_str, str)
        assert "sharpe_ratio" in json_str
        assert "1.85" in json_str


class TestComparisonRequest:
    """Test ComparisonRequest schema."""

    def test_basic_request(self):
        """Test creating basic comparison request."""
        bt_metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        live_metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.LIVE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
        )
        backtest_export = EvaluationExport(metadata=bt_metadata, metrics={"sharpe_ratio": 1.85})
        live_export = EvaluationExport(metadata=live_metadata, metrics={"sharpe_ratio": 1.72})
        request = ComparisonRequest(
            strategy_id="test",
            backtest_export=backtest_export,
            live_export=live_export,
        )
        assert request.strategy_id == "test"
        assert request.backtest_export == backtest_export
        assert request.live_export == live_export
        assert request.comparison_type == ComparisonType.BAYESIAN
        assert request.confidence_level == 0.95

    def test_request_with_options(self):
        """Test request with custom options."""
        bt_metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        live_metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.LIVE,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
        )
        request = ComparisonRequest(
            strategy_id="test",
            backtest_export=EvaluationExport(metadata=bt_metadata, metrics={"sharpe_ratio": 1.85}),
            live_export=EvaluationExport(metadata=live_metadata, metrics={"sharpe_ratio": 1.72}),
            comparison_type=ComparisonType.BOOTSTRAP,
            confidence_level=0.99,
            hypothesis="live >= backtest",
        )
        assert request.comparison_type == ComparisonType.BOOTSTRAP
        assert request.confidence_level == 0.99
        assert request.hypothesis == "live >= backtest"

    def test_confidence_level_validation(self):
        """Test confidence level must be in [0.5, 0.99]."""
        bt_metadata = StrategyMetadata(
            strategy_id="test",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )
        # Invalid confidence level
        with pytest.raises(ValidationError):
            ComparisonRequest(
                strategy_id="test",
                backtest_export=EvaluationExport(metadata=bt_metadata, metrics={}),
                live_export=EvaluationExport(metadata=bt_metadata, metrics={}),
                confidence_level=1.5,
            )


class TestComparisonResult:
    """Test ComparisonResult schema."""

    def test_basic_result(self):
        """Test creating basic comparison result."""
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.92,
            metrics_comparison={
                "sharpe_ratio": {
                    "backtest": 1.85,
                    "live": 1.72,
                    "diff": -0.13,
                }
            },
            statistical_tests={"bayesian": {"bayes_factor": 3.2, "posterior_prob": 0.92}},
            recommendation="Live performance consistent with backtest",
        )
        assert result.strategy_id == "test"
        assert result.decision == "PROMOTE"
        assert result.confidence == 0.92
        assert result.metrics_comparison["sharpe_ratio"]["backtest"] == 1.85
        assert result.recommendation == "Live performance consistent with backtest"

    def test_result_with_bayesian_evidence(self):
        """Test result with Bayesian evidence."""
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.95,
            metrics_comparison={"sharpe_ratio": {"backtest": 1.85, "live": 1.80}},
            statistical_tests={},
            bayesian_evidence={
                "bayes_factor": 5.2,
                "posterior_prob": 0.95,
                "prior_prob": 0.5,
            },
            recommendation="Strong evidence for promotion",
        )
        assert result.bayesian_evidence is not None
        assert result.bayesian_evidence["bayes_factor"] == 5.2
        assert result.bayesian_evidence["posterior_prob"] == 0.95

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="MONITOR",
            confidence=0.65,
            metrics_comparison={"sharpe_ratio": {"backtest": 1.85, "live": 1.45}},
            statistical_tests={},
            recommendation="Monitor for another period",
            warnings=[
                "Sharpe ratio declined by 21%",
                "Sample size may be too small (n=30)",
            ],
        )
        assert result.warnings is not None
        assert len(result.warnings) == 2
        assert "Sharpe ratio declined" in result.warnings[0]

    def test_to_dict(self):
        """Test result to dictionary."""
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.92,
            metrics_comparison={"sharpe_ratio": {"backtest": 1.85, "live": 1.72}},
            statistical_tests={},
            recommendation="Good to go",
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["strategy_id"] == "test"
        assert data["decision"] == "PROMOTE"
        assert data["confidence"] == 0.92

    def test_summary(self):
        """Test result summary generation."""
        result = ComparisonResult(
            strategy_id="momentum_v1",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.92,
            metrics_comparison={
                "sharpe_ratio": {
                    "backtest": 1.85,
                    "live": 1.72,
                    "diff": -0.13,
                },
                "cagr": {"backtest": 0.24, "live": 0.22, "diff": -0.02},
            },
            statistical_tests={},
            recommendation="Live performance within expected range",
            warnings=["Minor drawdown increase observed"],
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "momentum_v1" in summary
        assert "PROMOTE" in summary
        assert "0.92" in summary
        assert "Sharpe Ratio" in summary
        assert "1.85" in summary
        assert "1.72" in summary
        assert "⚠️" in summary  # Warning symbol


class TestPromotionWorkflow:
    """Test PromotionWorkflow schema."""

    def test_basic_workflow(self):
        """Test creating basic promotion workflow."""
        workflow = PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=30,
            promotion_criteria={
                "min_sharpe": 1.5,
                "max_drawdown": -0.15,
            },
        )
        assert workflow.strategy_id == "test"
        assert workflow.paper_duration_days == 30
        assert workflow.promotion_criteria["min_sharpe"] == 1.5
        assert workflow.approval_required is True
        assert workflow.risk_limits is None

    def test_workflow_with_risk_limits(self):
        """Test workflow with risk limits."""
        workflow = PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=60,
            promotion_criteria={"min_sharpe": 2.0},
            approval_required=False,
            risk_limits={
                "max_position_size": 0.05,
                "max_leverage": 2.0,
            },
        )
        assert workflow.approval_required is False
        assert workflow.risk_limits is not None
        assert workflow.risk_limits["max_position_size"] == 0.05

    def test_evaluate_promotion_success(self):
        """Test promotion evaluation - success case."""
        workflow = PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=30,
            promotion_criteria={
                "bayesian_confidence": 0.90,
            },
        )
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.95,
            metrics_comparison={"sharpe_ratio": {"backtest": 1.85, "live": 1.80}},
            statistical_tests={},
            recommendation="Good",
        )
        assert workflow.evaluate_promotion(result) is True

    def test_evaluate_promotion_failure_decision(self):
        """Test promotion evaluation - wrong decision."""
        workflow = PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=30,
            promotion_criteria={"bayesian_confidence": 0.90},
        )
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="REJECT",
            confidence=0.95,
            metrics_comparison={},
            statistical_tests={},
            recommendation="Not good",
        )
        assert workflow.evaluate_promotion(result) is False

    def test_evaluate_promotion_failure_confidence(self):
        """Test promotion evaluation - low confidence."""
        workflow = PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=30,
            promotion_criteria={"bayesian_confidence": 0.95},
        )
        result = ComparisonResult(
            strategy_id="test",
            comparison_type=ComparisonType.BAYESIAN,
            decision="PROMOTE",
            confidence=0.85,  # Below threshold
            metrics_comparison={},
            statistical_tests={},
            recommendation="Marginal",
        )
        assert workflow.evaluate_promotion(result) is False

    def test_paper_duration_validation(self):
        """Test paper duration must be >= 1."""
        # Valid
        PromotionWorkflow(
            strategy_id="test",
            paper_duration_days=1,
            promotion_criteria={},
        )
        # Invalid
        with pytest.raises(ValidationError):
            PromotionWorkflow(
                strategy_id="test",
                paper_duration_days=0,
                promotion_criteria={},
            )
