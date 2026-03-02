"""Tests for canonical feature-outcome metrics helpers."""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.metrics.feature_outcome import analyze_feature_outcome_series


class TestAnalyzeFeatureOutcomeSeries:
    """Test single-series feature-outcome analysis helper."""

    def test_rejects_misaligned_inputs(self) -> None:
        with pytest.raises(ValueError, match="must have same length"):
            analyze_feature_outcome_series(
                predictions=np.array([1.0, 2.0, 3.0]),
                outcomes=np.array([1.0, 2.0]),
            )

    def test_handles_empty_after_nan_filter(self) -> None:
        result = analyze_feature_outcome_series(
            predictions=np.array([np.nan, np.nan]),
            outcomes=np.array([np.nan, np.nan]),
        )

        assert result["n_observations"] == 0
        assert np.isnan(result["ic_mean"])
        assert result["monotonicity_analysis"] is None

    def test_small_sample_uses_stable_contract(self) -> None:
        result = analyze_feature_outcome_series(
            predictions=np.array([0.1, 0.2]),
            outcomes=np.array([0.2, 0.3]),
        )

        assert result["n_observations"] == 2
        assert result["p_value"] == 1.0
        assert result["monotonicity_analysis"] is None

    def test_filters_nan_and_keeps_valid_observation_count(self) -> None:
        result = analyze_feature_outcome_series(
            predictions=np.array([1.0, np.nan, 3.0, 4.0, np.nan]),
            outcomes=np.array([1.0, 2.0, 2.0, 4.0, np.nan]),
        )

        assert result["n_observations"] == 3
        assert np.isfinite(result["ic_mean"])
        assert np.isfinite(result["ic_ir"])

    def test_can_include_monotonicity_analysis(self) -> None:
        rng = np.random.default_rng(42)
        predictions = np.linspace(-1.0, 1.0, 200)
        outcomes = 0.5 * predictions + rng.normal(0.0, 0.1, size=predictions.shape[0])

        result = analyze_feature_outcome_series(
            predictions=predictions,
            outcomes=outcomes,
            include_monotonicity=True,
            n_quantiles=5,
        )

        monotonicity = result["monotonicity_analysis"]
        assert monotonicity is not None
        assert "is_monotonic" in monotonicity
        assert "monotonicity_score" in monotonicity
