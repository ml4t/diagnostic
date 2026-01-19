"""Tests for Wasserstein distance drift detection."""

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.drift import (
    WassersteinResult,
    compute_wasserstein_distance,
)


class TestWassersteinBasic:
    """Basic tests for Wasserstein distance computation."""

    def test_identical_distributions(self):
        """Wasserstein distance should be near zero for identical distributions."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = reference.copy()  # Exact same data

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        assert isinstance(result, WassersteinResult)
        assert result.distance < 1e-10  # Should be essentially zero
        assert not result.drifted  # No drift detected
        assert result.p == 1  # Default is W_1
        assert result.n_reference == 1000
        assert result.n_test == 1000

    def test_mean_shift_detected(self):
        """Wasserstein should detect mean shift in distribution."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(1.0, 1, 1000)  # Mean shifted by 1

        result = compute_wasserstein_distance(reference, test, random_state=42)

        # For same variance, W_1 ≈ |mean_diff|
        assert result.distance > 0.8  # Should be close to 1.0
        assert result.drifted  # Drift should be detected
        assert result.p_value is not None  # Threshold calibration done

    def test_variance_shift_detected(self):
        """Wasserstein should detect variance change."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 2, 1000)  # Same mean, 2x variance

        result = compute_wasserstein_distance(reference, test, random_state=42)

        assert result.distance > 0.3  # Should detect spread difference
        # May or may not be flagged depending on permutation test

    def test_small_shift(self):
        """Wasserstein should be sensitive to small shifts."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.2, 1, 1000)  # Small mean shift

        result = compute_wasserstein_distance(reference, test, random_state=42)

        assert 0.1 < result.distance < 0.4
        # Small shift should be detectable with permutation test

    def test_stable_distribution(self):
        """Independent samples from same distribution should not show drift."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        np.random.seed(43)  # Different seed but same distribution
        test = np.random.normal(0, 1, 1000)

        result = compute_wasserstein_distance(reference, test, random_state=42)

        # Small distance due to sampling variability
        assert result.distance < 0.2
        # Should not be flagged as drift (p-value should be large)


class TestWassersteinParameters:
    """Tests for different parameter configurations."""

    def test_wasserstein_1_vs_2(self):
        """Test Wasserstein-1 vs Wasserstein-2."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(1, 1, 500)

        result_1 = compute_wasserstein_distance(reference, test, p=1, threshold_calibration=False)
        result_2 = compute_wasserstein_distance(reference, test, p=2, threshold_calibration=False)

        # Both should detect drift
        assert result_1.distance > 0
        assert result_2.distance > 0
        assert result_1.p == 1
        assert result_2.p == 2
        # W_2 can be different from W_1
        assert result_1.distance != result_2.distance

    def test_threshold_calibration_enabled(self):
        """Test with threshold calibration enabled."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(0.3, 1, 500)

        result = compute_wasserstein_distance(
            reference,
            test,
            threshold_calibration=True,
            n_permutations=100,  # Reduced for speed
            alpha=0.05,
            random_state=42,
        )

        assert result.threshold is not None
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1
        assert result.threshold_calibration_config is not None
        assert result.threshold_calibration_config["n_permutations"] == 100
        assert result.threshold_calibration_config["alpha"] == 0.05

    def test_threshold_calibration_disabled(self):
        """Test with threshold calibration disabled (faster)."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(0.5, 1, 500)

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Should still have threshold (heuristic)
        assert result.threshold is not None
        # But no p-value
        assert result.p_value is None
        assert result.threshold_calibration_config is None

    def test_custom_alpha(self):
        """Test custom significance level."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(0.2, 1, 500)

        result_05 = compute_wasserstein_distance(
            reference, test, alpha=0.05, n_permutations=100, random_state=42
        )
        result_01 = compute_wasserstein_distance(
            reference, test, alpha=0.01, n_permutations=100, random_state=42
        )

        # More stringent alpha should have higher threshold
        assert result_01.threshold >= result_05.threshold

    def test_subsampling(self):
        """Test subsampling for large datasets."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 10000)
        test = np.random.normal(0.5, 1, 10000)

        result = compute_wasserstein_distance(
            reference, test, n_samples=500, threshold_calibration=False, random_state=42
        )

        # Should have used 500 samples
        assert result.n_reference == 500
        assert result.n_test == 500
        # Should still detect drift
        assert result.distance > 0.3

    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(0.3, 1, 500)

        result1 = compute_wasserstein_distance(reference, test, n_permutations=100, random_state=42)
        result2 = compute_wasserstein_distance(reference, test, n_permutations=100, random_state=42)

        # Should get same results
        assert result1.distance == result2.distance
        assert result1.p_value == result2.p_value
        assert result1.threshold == result2.threshold


class TestWassersteinInputTypes:
    """Tests for different input types."""

    def test_numpy_array_input(self):
        """Test with numpy arrays."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 100)
        test = np.random.normal(0.5, 1, 100)

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        assert isinstance(result, WassersteinResult)
        assert result.distance > 0

    def test_polars_series_input(self):
        """Test with Polars Series."""
        np.random.seed(42)
        reference_array = np.random.normal(0, 1, 100)
        test_array = np.random.normal(0.5, 1, 100)

        reference_pl = pl.Series(reference_array)
        test_pl = pl.Series(test_array)

        result = compute_wasserstein_distance(reference_pl, test_pl, threshold_calibration=False)

        assert isinstance(result, WassersteinResult)
        assert result.distance > 0

    def test_mixed_input_types(self):
        """Test with mixed input types."""
        np.random.seed(42)
        reference_array = np.random.normal(0, 1, 100)
        test_array = np.random.normal(0.5, 1, 100)

        reference_pl = pl.Series(reference_array)

        result = compute_wasserstein_distance(reference_pl, test_array, threshold_calibration=False)

        assert isinstance(result, WassersteinResult)
        assert result.distance > 0


class TestWassersteinEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Should raise error for empty arrays."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_wasserstein_distance(np.array([]), np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must not be empty"):
            compute_wasserstein_distance(np.array([1, 2, 3]), np.array([]))

    def test_invalid_p_value(self):
        """Should raise error for invalid p value."""
        reference = np.random.normal(0, 1, 100)
        test = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError, match="p must be 1 or 2"):
            compute_wasserstein_distance(reference, test, p=3)

    def test_constant_feature(self):
        """Should handle constant features gracefully."""
        reference = np.ones(100) * 5.0
        test = np.ones(100) * 5.0

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        assert result.distance < 1e-10
        assert not result.drifted

    def test_perfectly_separated_distributions(self):
        """Should handle non-overlapping distributions."""
        np.random.seed(42)
        reference = np.random.uniform(0, 1, 1000)
        test = np.random.uniform(10, 11, 1000)  # Completely different range

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Distance should be around 10 (the gap)
        assert result.distance > 9
        assert result.drifted

    def test_different_sample_sizes(self):
        """Should work with different sample sizes."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 100)
        test = np.random.normal(0.5, 1, 1000)  # 10x larger

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        assert result.n_reference == 100
        assert result.n_test == 1000
        assert result.distance > 0.3

    def test_single_value(self):
        """Should handle single value arrays."""
        reference = np.array([1.0])
        test = np.array([2.0])

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Distance should be |1 - 2| = 1
        assert abs(result.distance - 1.0) < 1e-10


class TestWassersteinStatistics:
    """Tests for distribution statistics in results."""

    def test_reference_stats(self):
        """Test that reference statistics are correctly computed."""
        np.random.seed(42)
        reference = np.random.normal(5, 2, 1000)
        test = np.random.normal(5, 2, 1000)

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Check reference stats are reasonable
        assert 4.5 < result.reference_stats["mean"] < 5.5
        assert 1.8 < result.reference_stats["std"] < 2.2
        assert "min" in result.reference_stats
        assert "max" in result.reference_stats
        assert "median" in result.reference_stats
        assert "q25" in result.reference_stats
        assert "q75" in result.reference_stats

    def test_test_stats(self):
        """Test that test statistics are correctly computed."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(2, 1, 1000)  # Mean shifted

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Test mean should be ~2
        assert 1.8 < result.test_stats["mean"] < 2.2
        # Reference mean should be ~0
        assert -0.2 < result.reference_stats["mean"] < 0.2


class TestWassersteinInterpretation:
    """Tests for interpretation and reporting."""

    def test_interpretation_messages(self):
        """Interpretation should match drift status."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)

        # No drift case
        test_nodrift = np.random.normal(0, 1, 500)
        result = compute_wasserstein_distance(
            reference, test_nodrift, random_state=42, n_permutations=100
        )
        if not result.drifted:
            assert "No significant drift" in result.interpretation
            assert "consistent" in result.interpretation.lower()

        # Drift case
        test_drift = np.random.normal(2, 1, 500)
        result = compute_wasserstein_distance(
            reference, test_drift, random_state=42, n_permutations=100
        )
        if result.drifted:
            assert "drift detected" in result.interpretation.lower()
            assert "differs" in result.interpretation.lower()

    def test_summary_formatting(self):
        """Summary should be readable and contain key information."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 100)
        test = np.random.normal(0.5, 1, 100)

        result = compute_wasserstein_distance(reference, test, n_permutations=50, random_state=42)
        summary = result.summary()

        # Should contain key information
        assert "Wasserstein Distance" in summary
        assert f"Wasserstein-{result.p} Distance" in summary
        assert "Drift Detected" in summary
        assert "Sample Sizes" in summary
        assert "Distribution Statistics" in summary
        assert "Interpretation" in summary
        assert "Computation Time" in summary

    def test_computation_time_tracked(self):
        """Computation time should be tracked."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        test = np.random.normal(0.5, 1, 500)

        result = compute_wasserstein_distance(reference, test, n_permutations=100, random_state=42)

        assert result.computation_time > 0
        assert result.computation_time < 60  # Should complete in reasonable time


class TestWassersteinComparison:
    """Tests comparing Wasserstein to PSI and other metrics."""

    def test_sensitivity_to_small_shifts(self):
        """Wasserstein should be sensitive to small distribution shifts."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)

        # Very small shift
        test = np.random.normal(0.1, 1, 1000)

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Should detect small shift (distance should be ~0.1)
        assert 0.05 < result.distance < 0.2

    def test_distribution_shape_sensitivity(self):
        """Test sensitivity to distribution shape changes."""
        np.random.seed(42)
        # Uniform distribution
        reference = np.random.uniform(-1, 1, 1000)
        # Normal distribution with same mean and variance
        test = np.random.normal(0, 1 / np.sqrt(3), 1000)

        result = compute_wasserstein_distance(reference, test, threshold_calibration=False)

        # Should detect shape difference even with similar moments
        assert result.distance > 0


class TestWassersteinIntegration:
    """Integration tests for realistic use cases."""

    def test_model_monitoring_workflow(self):
        """Test typical model monitoring scenario."""
        # Simulate training data
        np.random.seed(42)
        train_feature = np.random.normal(0, 1, 5000)

        # Simulate production data with increasing drift
        week1 = np.random.normal(0, 1, 1000)
        week2 = np.random.normal(0.1, 1, 1000)  # Small drift
        week3 = np.random.normal(0.3, 1, 1000)  # Moderate drift
        week4 = np.random.normal(0.8, 1.1, 1000)  # Significant drift

        # Monitor each week
        w1 = compute_wasserstein_distance(train_feature, week1, random_state=42, n_permutations=50)
        w2 = compute_wasserstein_distance(train_feature, week2, random_state=42, n_permutations=50)
        w3 = compute_wasserstein_distance(train_feature, week3, random_state=42, n_permutations=50)
        w4 = compute_wasserstein_distance(train_feature, week4, random_state=42, n_permutations=50)

        # Wasserstein distance should increase over time
        assert w1.distance < w2.distance
        assert w2.distance < w3.distance
        assert w3.distance < w4.distance

        # Week 4 should trigger drift alert
        assert w4.drifted

    def test_ab_test_comparison(self):
        """Test A/B test scenario where distributions should match."""
        np.random.seed(42)

        # A/B test with same treatment (null hypothesis)
        group_a = np.random.normal(100, 15, 1000)
        group_b = np.random.normal(100, 15, 1000)

        result = compute_wasserstein_distance(group_a, group_b, random_state=42, n_permutations=100)

        # Should not detect drift (distributions are the same)
        # P-value should be reasonably large
        assert result.p_value > 0.05 or not result.drifted

    def test_feature_engineering_validation(self):
        """Test validating feature transformations are stable."""
        np.random.seed(42)

        # Original feature
        original = np.random.exponential(2, 1000)

        # Log transform
        log_transformed = np.log(original + 1)

        # Simulate time-shifted data
        original_t1 = np.random.exponential(2, 1000)
        log_transformed_t1 = np.log(original_t1 + 1)

        # Original features should be similar across time
        result_original = compute_wasserstein_distance(
            original, original_t1, random_state=42, n_permutations=100
        )

        # Log-transformed features should also be similar
        result_log = compute_wasserstein_distance(
            log_transformed, log_transformed_t1, random_state=42, n_permutations=100
        )

        # Neither should show significant drift
        # (allowing for sampling variability)
        assert result_original.p_value > 0.01
        assert result_log.p_value > 0.01


class TestWassersteinPermutationTest:
    """Tests specifically for permutation test calibration."""

    def test_permutation_null_distribution(self):
        """Test that permutation test correctly estimates null distribution."""
        np.random.seed(42)
        # Same distribution - should not reject null
        reference = np.random.normal(0, 1, 200)
        test = np.random.normal(0, 1, 200)

        result = compute_wasserstein_distance(reference, test, n_permutations=200, random_state=42)

        # P-value should be reasonably large (> 0.05)
        assert result.p_value > 0.05
        # Should not detect drift
        assert not result.drifted

    def test_permutation_alternative_distribution(self):
        """Test that permutation test correctly rejects null for shifted distributions."""
        np.random.seed(42)
        # Different distributions - should reject null
        reference = np.random.normal(0, 1, 200)
        test = np.random.normal(1.5, 1, 200)  # Large shift

        result = compute_wasserstein_distance(reference, test, n_permutations=200, random_state=42)

        # P-value should be small
        assert result.p_value < 0.05
        # Should detect drift
        assert result.drifted

    def test_permutation_count(self):
        """Test different permutation counts."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 200)
        test = np.random.normal(0.5, 1, 200)

        # More permutations should give more stable estimates
        result_50 = compute_wasserstein_distance(
            reference, test, n_permutations=50, random_state=42
        )
        result_500 = compute_wasserstein_distance(
            reference, test, n_permutations=500, random_state=42
        )

        # Observed distance should be same
        assert result_50.distance == result_500.distance
        # Thresholds and p-values may differ slightly
        # but should be in same ballpark
        assert abs(result_50.p_value - result_500.p_value) < 0.1
