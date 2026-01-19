"""Tests for the main Evaluator framework.

These tests validate the complete Three-Tier Validation Framework integration,
ensuring that the Evaluator correctly orchestrates splitters, metrics, and
statistical tests.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml4t.diagnostic.evaluation.framework import EvaluationResult, Evaluator
from ml4t.diagnostic.splitters import CombinatorialPurgedCV, PurgedWalkForwardCV


class TestEvaluationResult:
    """Test EvaluationResult container class."""

    def test_basic_creation(self):
        """Test basic EvaluationResult creation."""
        metrics_results = {
            "sharpe": {"mean": 1.5, "std": 0.3},
            "ic": {"mean": 0.1, "std": 0.05},
        }

        result = EvaluationResult(
            tier=2,
            splitter_name="PurgedWalkForwardCV",
            metrics_results=metrics_results,
        )

        assert result.tier == 2
        assert result.splitter_name == "PurgedWalkForwardCV"
        assert result.metrics_results == metrics_results
        assert len(result.statistical_tests) == 0
        assert len(result.fold_results) == 0

    def test_summary_generation(self):
        """Test summary generation."""
        metrics_results = {
            "sharpe": {"mean": 1.5, "std": 0.3},
            "ic": {"mean": 0.1, "std": 0.05},
        }

        statistical_tests = {"dsr": {"dsr": 0.8, "p_value": 0.2}}

        result = EvaluationResult(
            tier=1,
            splitter_name="CombinatorialPurgedCV",
            metrics_results=metrics_results,
            statistical_tests=statistical_tests,
        )

        summary = result.summary()

        assert summary["tier"] == 1
        assert summary["splitter"] == "CombinatorialPurgedCV"
        assert "timestamp" in summary
        assert summary["metrics"]["sharpe"]["mean"] == 1.5
        assert summary["statistical_tests"]["dsr"]["test_statistic"] == 0.8

    def test_repr(self):
        """Test string representation."""
        metrics_results = {"sharpe": {"mean": 1.5}, "ic": 0.1}

        result = EvaluationResult(
            tier=3,
            splitter_name="TestSplitter",
            metrics_results=metrics_results,
        )

        repr_str = repr(result)
        assert "EvaluationResult" in repr_str
        assert "tier=3" in repr_str
        assert "TestSplitter" in repr_str


class TestEvaluator:
    """Test main Evaluator class."""

    def test_tier_inference(self):
        """Test automatic tier inference."""
        # Tier 1: CPCV splitter
        evaluator1 = Evaluator(splitter=CombinatorialPurgedCV(n_groups=4))
        assert evaluator1.tier == 1

        # Tier 1: DSR statistical test
        evaluator2 = Evaluator(statistical_tests=["dsr"])
        assert evaluator2.tier == 1

        # Tier 2: HAC statistical test
        evaluator3 = Evaluator(statistical_tests=["hac_ic"])
        assert evaluator3.tier == 2

        # Tier 3: Default
        evaluator4 = Evaluator()
        assert evaluator4.tier == 3

    def test_default_configurations(self):
        """Test default configurations for each tier."""
        # Tier 3
        evaluator3 = Evaluator(tier=3)
        assert isinstance(evaluator3.splitter, PurgedWalkForwardCV)
        assert evaluator3.splitter.n_splits == 3
        assert set(evaluator3.metrics) == {"ic", "hit_rate"}
        assert evaluator3.statistical_tests == []

        # Tier 2
        evaluator2 = Evaluator(tier=2)
        assert isinstance(evaluator2.splitter, PurgedWalkForwardCV)
        assert evaluator2.splitter.n_splits == 5
        assert set(evaluator2.metrics) == {"ic", "sharpe", "hit_rate"}
        assert evaluator2.statistical_tests == ["hac_ic"]

        # Tier 1
        evaluator1 = Evaluator(tier=1)
        assert isinstance(evaluator1.splitter, CombinatorialPurgedCV)
        assert set(evaluator1.metrics) == {
            "ic",
            "sharpe",
            "sortino",
            "max_drawdown",
            "hit_rate",
        }
        assert set(evaluator1.statistical_tests) == {"dsr", "fdr"}

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Invalid tier - now uses Pydantic validation messages
        with pytest.raises(ValueError, match="Configuration validation failed.*tier"):
            Evaluator(tier=4)

        # Invalid metric - still uses registry validation
        with pytest.raises(ValueError, match="Unknown metrics"):
            Evaluator(metrics=["invalid_metric"])

        # Invalid statistical test - still uses registry validation
        with pytest.raises(ValueError, match="Unknown statistical tests"):
            Evaluator(statistical_tests=["invalid_test"])

    def test_basic_evaluation(self):
        """Test basic model evaluation."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        y = 0.1 * X[:, 0] + 0.05 * X[:, 1] + np.random.randn(n_samples) * 0.02

        # Simple linear model
        model = LinearRegression()

        # Tier 3 evaluation
        evaluator = Evaluator(tier=3)
        result = evaluator.evaluate(X, y, model)

        assert isinstance(result, EvaluationResult)
        assert result.tier == 3
        assert "ic" in result.metrics_results
        assert "hit_rate" in result.metrics_results
        assert len(result.fold_results) == 3  # Default n_splits for tier 3

        # Check that metrics are reasonable
        ic_mean = result.metrics_results["ic"]["mean"]
        assert not np.isnan(ic_mean)
        assert -1 <= ic_mean <= 1  # IC should be in valid range

        hit_rate_mean = result.metrics_results["hit_rate"]["mean"]
        assert 0 <= hit_rate_mean <= 100  # Hit rate should be percentage

    def test_evaluation_with_callable_model(self):
        """Test evaluation with callable model instead of sklearn."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples) * 0.02

        # Simple callable model (returns mean of features)
        def simple_model(X_train, y_train, X_test):
            return np.mean(X_test, axis=1)

        evaluator = Evaluator(tier=3, metrics=["ic"])
        result = evaluator.evaluate(X, y, simple_model)

        assert isinstance(result, EvaluationResult)
        assert "ic" in result.metrics_results

    def test_evaluation_with_strategy_function(self):
        """Test evaluation with custom strategy function."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples) * 0.02

        model = LinearRegression()

        # Custom strategy function (threshold-based)
        def threshold_strategy(predictions, returns):
            # Only trade when prediction magnitude > threshold
            threshold = np.std(predictions) * 0.5
            positions = np.where(
                np.abs(predictions) > threshold,
                np.sign(predictions),
                0,
            )
            return positions * returns

        evaluator = Evaluator(tier=3, metrics=["sharpe"])
        result = evaluator.evaluate(X, y, model, strategy_func=threshold_strategy)

        assert isinstance(result, EvaluationResult)
        assert "sharpe" in result.metrics_results

    def test_tier_2_evaluation(self):
        """Test Tier 2 evaluation with statistical tests."""
        # Generate synthetic data with some signal
        np.random.seed(42)
        n_samples = 300
        X = np.random.randn(n_samples, 4)
        # Add some signal
        y = 0.2 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(n_samples) * 0.05

        model = RandomForestRegressor(n_estimators=20, random_state=42)

        evaluator = Evaluator(tier=2, random_state=42)
        result = evaluator.evaluate(X, y, model)

        assert result.tier == 2
        assert "hac_ic" in result.statistical_tests

        # Check HAC test results
        hac_result = result.statistical_tests["hac_ic"]
        assert "ic" in hac_result
        assert "t_stat" in hac_result
        assert "p_value" in hac_result

    def test_tier_1_evaluation_small_dataset(self):
        """Test Tier 1 evaluation with smaller dataset (to avoid long runtime)."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 150  # Smaller for faster CPCV
        X = np.random.randn(n_samples, 3)
        y = 0.15 * X[:, 0] + np.random.randn(n_samples) * 0.03

        model = LinearRegression()

        # Use smaller CPCV configuration for speed
        small_cpcv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
        evaluator = Evaluator(
            splitter=small_cpcv,
            tier=1,
            metrics=["ic", "sharpe"],
            statistical_tests=["dsr"],
            random_state=42,
        )

        result = evaluator.evaluate(X, y, model)

        assert result.tier == 1
        assert len(result.fold_results) == 4  # C(4,1) = 4 combinations
        assert "dsr" in result.statistical_tests

        # Check DSR results
        dsr_result = result.statistical_tests["dsr"]
        assert "dsr" in dsr_result
        assert "p_value" in dsr_result

    def test_batch_evaluation(self):
        """Test batch evaluation of multiple models."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 120
        X = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples) * 0.02

        # Multiple models
        models = [
            LinearRegression(),
            RandomForestRegressor(n_estimators=10, random_state=42),
        ]
        model_names = ["Linear", "RandomForest"]

        evaluator = Evaluator(tier=3, random_state=42)
        results = evaluator.batch_evaluate(models, X, y, model_names)

        assert len(results) == 2
        assert "Linear" in results
        assert "RandomForest" in results

        for _name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.tier == 3

    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create mock results
        result1 = EvaluationResult(
            tier=3,
            splitter_name="TestSplitter",
            metrics_results={"sharpe": {"mean": 1.5, "std": 0.2}},
        )

        result2 = EvaluationResult(
            tier=3,
            splitter_name="TestSplitter",
            metrics_results={"sharpe": {"mean": 1.2, "std": 0.3}},
        )

        batch_results = {"Model1": result1, "Model2": result2}

        evaluator = Evaluator(tier=3)
        comparison = evaluator.compare_models(batch_results, primary_metric="sharpe")

        assert comparison["primary_metric"] == "sharpe"
        assert comparison["n_models"] == 2
        assert comparison["best_model"] == "Model1"  # Higher Sharpe
        assert len(comparison["ranking"]) == 2
        assert comparison["ranking"][0]["model"] == "Model1"

    def test_error_handling(self):
        """Test error handling in evaluation."""
        # Mismatched X and y dimensions
        X = np.random.randn(100, 3)
        y = np.random.randn(50)  # Wrong size

        model = LinearRegression()
        evaluator = Evaluator(tier=3)

        with pytest.raises(
            ValueError,
            match="x and y must have the same number of samples",
        ):
            evaluator.evaluate(X, y, model)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small dataset
        X = np.random.randn(10, 2)
        y = np.random.randn(10) * 0.01

        model = LinearRegression()
        evaluator = Evaluator(
            splitter=PurgedWalkForwardCV(n_splits=2),  # Minimal splits
            tier=3,
            metrics=["ic"],
        )

        # Should not crash even with small dataset
        result = evaluator.evaluate(X, y, model)
        assert isinstance(result, EvaluationResult)

        # Check that some results might be NaN but evaluation completes
        ic_mean = result.metrics_results["ic"]["mean"]
        # Could be NaN due to small sample size, but that's okay
        assert isinstance(ic_mean, float | np.floating) or np.isnan(ic_mean)

    def test_reproducibility(self):
        """Test that results are reproducible with random_state."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100) * 0.02

        model1 = RandomForestRegressor(n_estimators=10, random_state=1)
        model2 = RandomForestRegressor(n_estimators=10, random_state=1)

        evaluator1 = Evaluator(tier=3, random_state=42)
        evaluator2 = Evaluator(tier=3, random_state=42)

        result1 = evaluator1.evaluate(X, y, model1)
        result2 = evaluator2.evaluate(X, y, model2)

        # Results should be very similar (allowing for small numerical differences)
        ic1 = result1.metrics_results["ic"]["mean"]
        ic2 = result2.metrics_results["ic"]["mean"]

        if not (np.isnan(ic1) and np.isnan(ic2)):
            assert abs(ic1 - ic2) < 1e-10
