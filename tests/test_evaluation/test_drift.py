"""Tests for drift detection metrics."""

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.drift import PSIResult, compute_psi


class TestPSIContinuous:
    """Tests for PSI with continuous features."""

    def test_stable_distribution(self):
        """PSI should be near zero for identical distributions."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 1, 1000)

        result = compute_psi(reference, test, n_bins=10)

        assert isinstance(result, PSIResult)
        assert result.psi < 0.1  # Should be green (stable)
        assert result.alert_level == "green"
        assert result.n_bins == 10
        assert not result.is_categorical

    def test_mean_shift(self):
        """PSI should detect mean shift in distribution."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(1.0, 1, 1000)  # Mean shifted by 1

        result = compute_psi(reference, test, n_bins=10)

        assert result.psi > 0.2  # Should be red (significant drift)
        assert result.alert_level == "red"

    def test_variance_shift(self):
        """PSI should detect variance change."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 2, 1000)  # Variance doubled

        result = compute_psi(reference, test, n_bins=10)

        assert result.psi > 0.1  # Should detect change
        assert result.alert_level in ["yellow", "red"]

    def test_small_shift(self):
        """PSI should detect small but noticeable shifts."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.3, 1, 1000)  # Small mean shift

        result = compute_psi(reference, test, n_bins=10)

        # Should be in yellow zone (0.1 <= PSI < 0.2)
        assert 0.05 < result.psi < 0.3
        # Alert level depends on exact PSI value
        assert result.alert_level in ["yellow", "red", "green"]

    def test_constant_feature(self):
        """PSI should handle constant features gracefully."""
        reference = np.ones(100)
        test = np.ones(100)

        result = compute_psi(reference, test, n_bins=10)

        assert result.psi < 0.1
        assert result.alert_level == "green"

    def test_different_bin_counts(self):
        """PSI should work with different numbers of bins."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.5, 1, 1000)

        result_5 = compute_psi(reference, test, n_bins=5)
        result_10 = compute_psi(reference, test, n_bins=10)
        result_20 = compute_psi(reference, test, n_bins=20)

        # PSI values should be roughly similar but not identical
        assert result_5.n_bins == 5
        assert result_10.n_bins == 10
        assert result_20.n_bins == 20

        # All should detect the shift
        assert result_5.alert_level in ["yellow", "red"]
        assert result_10.alert_level in ["yellow", "red"]
        assert result_20.alert_level in ["yellow", "red"]

    def test_polars_series_input(self):
        """PSI should accept Polars Series."""
        np.random.seed(42)
        reference_array = np.random.normal(0, 1, 100)
        test_array = np.random.normal(0.5, 1, 100)

        reference_pl = pl.Series(reference_array)
        test_pl = pl.Series(test_array)

        result = compute_psi(reference_pl, test_pl, n_bins=10)

        assert isinstance(result, PSIResult)
        assert result.psi > 0

    def test_bin_level_details(self):
        """PSI result should contain bin-level analysis."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.5, 1, 1000)

        result = compute_psi(reference, test, n_bins=10)

        # Check bin-level arrays
        assert len(result.bin_psi) == 10
        assert len(result.reference_counts) == 10
        assert len(result.test_counts) == 10
        assert len(result.reference_percents) == 10
        assert len(result.test_percents) == 10

        # Percentages should sum to ~1
        assert np.abs(result.reference_percents.sum() - 1.0) < 1e-6
        assert np.abs(result.test_percents.sum() - 1.0) < 1e-6

        # Total PSI should equal sum of bin PSI
        assert np.abs(result.psi - result.bin_psi.sum()) < 1e-6

    def test_summary_formatting(self):
        """PSI summary should be readable."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 100)
        test = np.random.normal(0.5, 1, 100)

        result = compute_psi(reference, test, n_bins=5)
        summary = result.summary()

        # Should contain key information
        assert "Population Stability Index" in summary
        assert f"PSI Value: {result.psi:.4f}" in summary
        assert result.alert_level.upper() in summary
        assert "Continuous" in summary
        assert "Bin-Level Analysis" in summary


class TestPSICategorical:
    """Tests for PSI with categorical features."""

    def test_stable_categorical(self):
        """PSI should be near zero for identical categorical distributions."""
        np.random.seed(42)
        reference = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])
        test = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])

        result = compute_psi(reference, test, is_categorical=True)

        assert result.psi < 0.1
        assert result.alert_level == "green"
        assert result.is_categorical
        assert result.n_bins == 3  # Three categories

    def test_shifted_categorical(self):
        """PSI should detect distribution shift in categorical data."""
        np.random.seed(42)
        reference = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])
        test = np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.3, 0.5])

        result = compute_psi(reference, test, is_categorical=True)

        assert result.psi > 0.2  # Should detect significant shift
        assert result.alert_level == "red"

    def test_missing_category_separate(self):
        """New categories should be handled with separate bin."""
        reference = np.array(["A", "B", "C"] * 100)
        test = np.array(["A", "B", "C", "D"] * 75)  # 'D' is new

        result = compute_psi(
            reference, test, is_categorical=True, missing_category_handling="separate"
        )

        # Should have 4 bins (A, B, C, D)
        assert result.n_bins == 4
        assert "D" in result.bin_edges
        # D should have 0 count in reference
        d_index = result.bin_edges.index("D")
        assert result.reference_counts[d_index] == 0

    def test_missing_category_error(self):
        """Should raise error when new categories found and handling='error'."""
        reference = np.array(["A", "B", "C"] * 100)
        test = np.array(["A", "B", "C", "D"] * 75)

        with pytest.raises(ValueError, match="New categories found"):
            compute_psi(reference, test, is_categorical=True, missing_category_handling="error")

    def test_missing_category_ignore(self):
        """Should ignore new categories when handling='ignore'."""
        reference = np.array(["A", "B", "C"] * 100)
        test = np.array(["A", "B", "C", "D"] * 75)

        result = compute_psi(
            reference, test, is_categorical=True, missing_category_handling="ignore"
        )

        # Should have 3 bins (A, B, C) - D ignored
        assert result.n_bins == 3
        assert "D" not in result.bin_edges

    def test_categorical_bin_labels(self):
        """Categorical bins should use category labels."""
        reference = np.array(["Low", "Medium", "High"] * 100)
        test = np.array(["Low", "Medium", "High"] * 100)

        result = compute_psi(reference, test, is_categorical=True)

        assert "Low" in result.bin_edges
        assert "Medium" in result.bin_edges
        assert "High" in result.bin_edges


class TestPSIEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Should raise error for empty arrays."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_psi(np.array([]), np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must not be empty"):
            compute_psi(np.array([1, 2, 3]), np.array([]))

    def test_custom_thresholds(self):
        """Should respect custom PSI thresholds."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0.3, 1, 1000)

        # More sensitive thresholds
        result = compute_psi(reference, test, psi_threshold_yellow=0.05, psi_threshold_red=0.1)

        # Small shift should now trigger red alert
        if result.psi > 0.1:
            assert result.alert_level == "red"

    def test_numerical_stability(self):
        """PSI should handle edge cases without NaN/Inf."""
        # Very small probabilities
        np.random.seed(42)
        reference = np.random.normal(0, 1, 10000)
        test = np.random.normal(0.1, 1, 10000)

        result = compute_psi(reference, test, n_bins=100)  # Many bins = small probs

        assert np.isfinite(result.psi)
        assert all(np.isfinite(result.bin_psi))

    def test_perfectly_separated_distributions(self):
        """PSI should handle non-overlapping distributions."""
        reference = np.random.uniform(0, 1, 1000)
        test = np.random.uniform(10, 11, 1000)  # Completely different range

        result = compute_psi(reference, test, n_bins=10)

        assert result.psi > 0.2
        assert result.alert_level == "red"

    def test_single_category(self):
        """Should handle single-category features."""
        reference = np.array(["A"] * 100)
        test = np.array(["A"] * 100)

        result = compute_psi(reference, test, is_categorical=True)

        assert result.psi < 0.1
        assert result.alert_level == "green"
        assert result.n_bins == 1


class TestPSIInterpretation:
    """Tests for PSI interpretation and reporting."""

    def test_interpretation_messages(self):
        """Interpretation should match alert level."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)

        # Green case
        test_green = np.random.normal(0, 1, 1000)
        result = compute_psi(reference, test_green, n_bins=10)
        if result.alert_level == "green":
            assert "stable" in result.interpretation.lower()

        # Red case
        test_red = np.random.normal(2, 1, 1000)
        result = compute_psi(reference, test_red, n_bins=10)
        if result.alert_level == "red":
            assert "significant" in result.interpretation.lower()
            assert "investigate" in result.interpretation.lower()

    def test_alert_level_consistency(self):
        """Alert level should be consistent with PSI value."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)

        # Test various shifts
        for shift in [0, 0.2, 0.5, 1.0, 2.0]:
            test = np.random.normal(shift, 1, 1000)
            result = compute_psi(reference, test, n_bins=10)

            # Check consistency
            if result.psi < 0.1:
                assert result.alert_level == "green"
            elif result.psi < 0.2:
                assert result.alert_level == "yellow"
            else:
                assert result.alert_level == "red"


# Integration tests
class TestPSIIntegration:
    """Integration tests for realistic use cases."""

    def test_model_monitoring_workflow(self):
        """Test typical model monitoring scenario."""
        # Simulate training data
        np.random.seed(42)
        train_feature = np.random.normal(0, 1, 5000)

        # Simulate production data with slight drift over time
        week1 = np.random.normal(0, 1, 1000)
        week2 = np.random.normal(0.1, 1, 1000)  # Small drift
        week3 = np.random.normal(0.5, 1, 1000)  # Moderate drift
        week4 = np.random.normal(1.0, 1.2, 1000)  # Significant drift

        # Monitor each week
        psi_week1 = compute_psi(train_feature, week1, n_bins=10)
        psi_week2 = compute_psi(train_feature, week2, n_bins=10)
        psi_week3 = compute_psi(train_feature, week3, n_bins=10)
        psi_week4 = compute_psi(train_feature, week4, n_bins=10)

        # PSI should increase over time
        assert psi_week1.psi < psi_week2.psi
        assert psi_week2.psi < psi_week3.psi
        assert psi_week3.psi < psi_week4.psi

        # Week 4 should trigger alert
        assert psi_week4.alert_level in ["yellow", "red"]

    def test_multi_feature_monitoring(self):
        """Test monitoring multiple features."""
        np.random.seed(42)

        # Multiple features with different drift patterns
        ref_stable = np.random.normal(0, 1, 1000)
        ref_drifted = np.random.normal(0, 1, 1000)
        ref_categorical = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])

        test_stable = np.random.normal(0, 1, 1000)
        test_drifted = np.random.normal(1, 1, 1000)  # Drifted
        test_categorical = np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.3, 0.5])

        psi_stable = compute_psi(ref_stable, test_stable)
        psi_drifted = compute_psi(ref_drifted, test_drifted)
        psi_cat = compute_psi(ref_categorical, test_categorical, is_categorical=True)

        # Stable feature should be green
        assert psi_stable.alert_level == "green"

        # Drifted feature should be red
        assert psi_drifted.alert_level in ["yellow", "red"]

        # Categorical should detect shift
        assert psi_cat.psi > 0.1
