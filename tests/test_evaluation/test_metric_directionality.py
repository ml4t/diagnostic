"""Tests for metric directionality in model comparisons.

This test verifies that the compare_models function correctly sorts
models based on whether each metric should be maximized or minimized.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.framework import (
    EvaluationResult,
    Evaluator,
    get_metric_directionality,
)
from ml4t.diagnostic.evaluation.metric_registry import MetricRegistry


class TestMetricDirectionality:
    """Test metric directionality registry and comparison logic."""

    def test_metric_registry_completeness(self):
        """Test that common metrics are in the registry."""
        registry = MetricRegistry.default()

        # Performance metrics (should be maximized)
        assert registry.is_maximize("sharpe") is True
        assert registry.is_maximize("sortino") is True
        assert registry.is_maximize("information_coefficient") is True
        assert registry.is_maximize("total_return") is True

        # Risk metrics (should be minimized)
        assert registry.is_maximize("max_drawdown") is False
        assert registry.is_maximize("volatility") is False
        assert registry.is_maximize("value_at_risk") is False

        # Statistical metrics
        assert registry.is_maximize("p_value") is False  # Lower is better

    def test_get_metric_directionality(self):
        """Test the directionality helper function."""
        # Known metrics
        assert get_metric_directionality("sharpe") is True
        assert get_metric_directionality("max_drawdown") is False

        # Case insensitive
        assert get_metric_directionality("SHARPE") is True
        assert get_metric_directionality("Max_Drawdown") is False

        # With hyphens/spaces
        assert get_metric_directionality("sharpe-ratio") is True
        assert get_metric_directionality("max drawdown") is False

        # Unknown metrics with heuristics
        assert get_metric_directionality("custom_return") is True  # Has "return"
        assert get_metric_directionality("custom_loss") is False  # Has "loss"
        assert get_metric_directionality("tracking_error_custom") is False  # Has "error"

        # Unknown metric without heuristics (defaults to maximize)
        assert get_metric_directionality("mystery_metric") is True

    def test_compare_models_sharpe_ratio(self):
        """Test model comparison with Sharpe ratio (higher is better)."""
        evaluator = Evaluator()

        # Create mock evaluation results with different Sharpe ratios
        results = {
            "model_a": self._create_mock_result({"sharpe": {"mean": 1.5}}),
            "model_b": self._create_mock_result({"sharpe": {"mean": 2.0}}),
            "model_c": self._create_mock_result({"sharpe": {"mean": 0.8}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="sharpe")

        # Check ranking (higher Sharpe is better, so descending order)
        ranking = comparison["ranking"]
        assert len(ranking) == 3
        assert ranking[0]["model"] == "model_b"  # Highest Sharpe (2.0)
        assert ranking[1]["model"] == "model_a"  # Middle Sharpe (1.5)
        assert ranking[2]["model"] == "model_c"  # Lowest Sharpe (0.8)

        assert comparison["best_model"] == "model_b"

    def test_compare_models_max_drawdown(self):
        """Test model comparison with max drawdown (lower is better)."""
        evaluator = Evaluator()

        # Create mock results with different max drawdowns
        results = {
            "model_a": self._create_mock_result({"max_drawdown": {"mean": -0.15}}),
            "model_b": self._create_mock_result({"max_drawdown": {"mean": -0.05}}),
            "model_c": self._create_mock_result({"max_drawdown": {"mean": -0.25}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="max_drawdown")

        # Check ranking (smaller absolute drawdown is better)
        # abs(-0.05) < abs(-0.15) < abs(-0.25)
        ranking = comparison["ranking"]
        assert len(ranking) == 3
        assert ranking[0]["model"] == "model_b"  # Smallest abs drawdown (|-0.05|)
        assert ranking[1]["model"] == "model_a"  # Middle abs drawdown (|-0.15|)
        assert ranking[2]["model"] == "model_c"  # Largest abs drawdown (|-0.25|)

        assert comparison["best_model"] == "model_b"

    def test_compare_models_p_value(self):
        """Test model comparison with p-value (lower is better)."""
        evaluator = Evaluator()

        # Create mock results with different p-values
        results = {
            "model_a": self._create_mock_result({"p_value": {"mean": 0.05}}),
            "model_b": self._create_mock_result({"p_value": {"mean": 0.01}}),
            "model_c": self._create_mock_result({"p_value": {"mean": 0.10}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="p_value")

        # Check ranking (lower p-value is better)
        ranking = comparison["ranking"]
        assert len(ranking) == 3
        assert ranking[0]["model"] == "model_b"  # Lowest p-value (0.01)
        assert ranking[1]["model"] == "model_a"  # Middle p-value (0.05)
        assert ranking[2]["model"] == "model_c"  # Highest p-value (0.10)

        assert comparison["best_model"] == "model_b"

    def test_compare_models_information_coefficient(self):
        """Test model comparison with IC (higher is better)."""
        evaluator = Evaluator()

        # Create mock results with different ICs
        results = {
            "model_a": self._create_mock_result({"information_coefficient": {"mean": 0.03}}),
            "model_b": self._create_mock_result({"information_coefficient": {"mean": 0.08}}),
            "model_c": self._create_mock_result({"information_coefficient": {"mean": -0.02}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="information_coefficient")

        # Check ranking (higher IC is better)
        ranking = comparison["ranking"]
        assert len(ranking) == 3
        assert ranking[0]["model"] == "model_b"  # Highest IC (0.08)
        assert ranking[1]["model"] == "model_a"  # Middle IC (0.03)
        assert ranking[2]["model"] == "model_c"  # Lowest IC (-0.02)

        assert comparison["best_model"] == "model_b"

    def test_compare_models_mixed_metrics(self):
        """Test that different metrics sort correctly in their own comparisons."""
        evaluator = Evaluator()

        # Create mock results with various metrics
        results = {
            "model_a": self._create_mock_result(
                {
                    "sharpe": {"mean": 1.5},
                    "max_drawdown": {"mean": -0.15},
                    "volatility": {"mean": 0.20},
                }
            ),
            "model_b": self._create_mock_result(
                {
                    "sharpe": {"mean": 2.0},
                    "max_drawdown": {"mean": -0.25},  # Worse drawdown
                    "volatility": {"mean": 0.15},  # Better volatility
                }
            ),
        }

        # Compare by Sharpe (higher is better)
        sharpe_comp = evaluator.compare_models(results, primary_metric="sharpe")
        assert sharpe_comp["best_model"] == "model_b"  # Higher Sharpe wins

        # Compare by max drawdown (lower is better)
        dd_comp = evaluator.compare_models(results, primary_metric="max_drawdown")
        assert dd_comp["best_model"] == "model_a"  # Smaller drawdown wins

        # Compare by volatility (lower is better)
        vol_comp = evaluator.compare_models(results, primary_metric="volatility")
        assert vol_comp["best_model"] == "model_b"  # Lower volatility wins

    def test_compare_models_with_nan_values(self):
        """Test that NaN values are handled correctly."""
        evaluator = Evaluator()

        # Create mock results with some NaN values
        results = {
            "model_a": self._create_mock_result({"sharpe": {"mean": 1.5}}),
            "model_b": self._create_mock_result({"sharpe": {"mean": np.nan}}),
            "model_c": self._create_mock_result({"sharpe": {"mean": 2.0}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="sharpe")

        # Only valid models should be ranked
        ranking = comparison["ranking"]
        assert len(ranking) == 2  # model_b excluded due to NaN
        assert ranking[0]["model"] == "model_c"
        assert ranking[1]["model"] == "model_a"

    def test_compare_models_unknown_metric(self):
        """Test comparison with unknown metric uses default heuristics."""
        evaluator = Evaluator()

        # Create mock results with custom metric
        results = {
            "model_a": self._create_mock_result({"custom_score": {"mean": 0.5}}),
            "model_b": self._create_mock_result({"custom_score": {"mean": 0.8}}),
        }

        # Should default to "higher is better" for unknown metric with "score" in name
        comparison = evaluator.compare_models(results, primary_metric="custom_score")
        assert comparison["best_model"] == "model_b"

        # Test with "error" metric (should default to lower is better)
        results = {
            "model_a": self._create_mock_result({"custom_error": {"mean": 0.5}}),
            "model_b": self._create_mock_result({"custom_error": {"mean": 0.3}}),
        }

        comparison = evaluator.compare_models(results, primary_metric="custom_error")
        assert comparison["best_model"] == "model_b"  # Lower error is better

    def test_all_registry_metrics_have_valid_directionality(self):
        """Test that all metrics in registry have boolean directionality."""
        registry = MetricRegistry.default()
        for metric in registry.list_metrics():
            directionality = registry.is_maximize(metric)
            assert isinstance(directionality, bool), f"{metric} has non-boolean directionality"

    def _create_mock_result(self, metrics_dict):
        """Helper to create mock EvaluationResult with specified metrics."""
        result = MagicMock(spec=EvaluationResult)
        result.metrics_results = metrics_dict
        result.summary = MagicMock(return_value={"metrics": metrics_dict})
        return result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
