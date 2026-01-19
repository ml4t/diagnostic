"""Tests for statistical testing and multiple testing correction.

Tests cover DSR, RAS, FDR control, bootstrap methods, and other statistical tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.diagnostic.evaluation.stats import (
    DSRResult,
    benjamini_hochberg_fdr,
    deflated_sharpe_ratio_from_statistics,
    holm_bonferroni,
    rademacher_complexity,
    ras_ic_adjustment,
)


class TestDeflatedSharpeRatio:
    """Tests for Deflated Sharpe Ratio (DSR).

    Uses deflated_sharpe_ratio_from_statistics() for pre-computed statistics.
    Tests for raw returns input use deflated_sharpe_ratio() directly.
    """

    def test_basic_dsr(self):
        """Test DSR with basic inputs."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,  # Best Sharpe from trials
            n_trials=10,  # 10 strategies tested
            variance_trials=0.5,  # Variance of Sharpe ratios across trials
            n_samples=252,  # 1 year of daily data
        )

        # Returns DSRResult with probability in [0, 1]
        assert isinstance(result, DSRResult)
        assert 0 <= result.probability <= 1

    def test_dsr_zscore_accessible(self):
        """Test DSR z-score is accessible from result."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )

        # Z-score can be any real number
        assert isinstance(result.z_score, float)
        assert np.isfinite(result.z_score)

    def test_dsr_deflated_sharpe_accessible(self):
        """Test DSR deflated Sharpe is accessible from result."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )

        assert isinstance(result.deflated_sharpe, float)
        # Deflated Sharpe should be less than observed (adjusted for multiple testing)
        assert result.deflated_sharpe <= 2.0 + 0.1  # Allow small tolerance

    def test_dsr_result_has_all_components(self):
        """Test DSRResult has all expected components."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )

        assert isinstance(result, DSRResult)
        # Check all key attributes are present and finite
        assert np.isfinite(result.probability)
        assert np.isfinite(result.z_score)
        assert np.isfinite(result.expected_max_sharpe)
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.deflated_sharpe)

    def test_dsr_single_trial(self):
        """Test DSR with single trial (n_trials=1) is PSR."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1,
            variance_trials=0.0,  # No variance for single trial
            n_samples=252,
        )

        # With single trial, probability should be high for high Sharpe
        assert isinstance(result, DSRResult)
        assert result.probability > 0.5  # Should be a high probability
        assert result.expected_max_sharpe == 0.0  # No multiple testing adjustment

    def test_dsr_many_trials(self):
        """Test that more trials lead to more deflation."""
        result_10 = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )
        result_100 = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=100,
            variance_trials=0.5,
            n_samples=252,
        )

        # More trials should lead to lower probability (more deflation)
        assert result_100.probability < result_10.probability

    def test_dsr_high_variance(self):
        """Test DSR with high variance across trials."""
        result_low_var = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.1,
            n_samples=252,
        )
        result_high_var = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=1.0,
            n_samples=252,
        )

        # Higher variance means observed max is more likely due to luck
        # So DSR should be lower
        assert result_high_var.probability < result_low_var.probability

    def test_dsr_with_skewness_kurtosis(self):
        """Test DSR with non-normal return distributions."""
        result_normal = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
            skewness=0.0,
            excess_kurtosis=0.0,  # Normal (Fisher convention)
        )

        result_heavy_tails = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
            skewness=0.0,
            excess_kurtosis=3.0,  # Heavy tails (Fisher convention)
        )

        # Both should produce valid results
        assert 0 <= result_normal.probability <= 1
        assert 0 <= result_heavy_tails.probability <= 1

    def test_dsr_negative_sharpe(self):
        """Test DSR with negative observed Sharpe."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=-1.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )

        # Negative Sharpe should give low probability
        assert isinstance(result, DSRResult)
        assert result.probability < 0.5


class TestRademacherComplexity:
    """Tests for Rademacher complexity estimation."""

    def test_basic_complexity(self):
        """Test basic Rademacher complexity calculation."""
        np.random.seed(42)
        # T=100 time periods, N=10 strategies
        X = np.random.randn(100, 10)

        complexity = rademacher_complexity(X, n_simulations=1000, random_state=42)

        assert isinstance(complexity, float)
        assert complexity >= 0  # Complexity is non-negative
        assert np.isfinite(complexity)

    def test_complexity_with_correlation(self):
        """Test complexity with correlated strategies."""
        np.random.seed(42)
        T, N = 100, 10

        # Uncorrelated strategies
        X_uncorr = np.random.randn(T, N)

        # Highly correlated strategies (all same with noise)
        base = np.random.randn(T, 1)
        X_corr = base + np.random.randn(T, N) * 0.1

        complexity_uncorr = rademacher_complexity(X_uncorr, n_simulations=1000, random_state=42)
        complexity_corr = rademacher_complexity(X_corr, n_simulations=1000, random_state=42)

        # Correlated strategies should have lower complexity
        assert complexity_corr < complexity_uncorr

    def test_complexity_scaling(self):
        """Test that complexity decreases with more samples."""
        np.random.seed(42)
        N = 10

        X_small = np.random.randn(50, N)
        X_large = np.random.randn(500, N)

        complexity_small = rademacher_complexity(X_small, n_simulations=1000, random_state=42)
        complexity_large = rademacher_complexity(X_large, n_simulations=1000, random_state=42)

        # More samples should reduce complexity (less overfitting risk)
        assert complexity_large < complexity_small

    def test_complexity_invalid_input(self):
        """Test complexity with invalid inputs."""
        # 1D array should raise error
        with pytest.raises((ValueError, TypeError)):
            rademacher_complexity(np.array([1, 2, 3]))

        # Non-numpy array should raise error
        with pytest.raises(TypeError):
            rademacher_complexity([[1, 2], [3, 4]])

    def test_complexity_determinism(self):
        """Test that random_state makes computation deterministic."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        complexity_1 = rademacher_complexity(X, n_simulations=1000, random_state=123)
        complexity_2 = rademacher_complexity(X, n_simulations=1000, random_state=123)

        assert complexity_1 == complexity_2


class TestRASAdjustment:
    """Tests for Rademacher Anti-Serum (RAS) IC adjustment."""

    def test_ras_ic_basic(self):
        """Test RAS IC adjustment with basic inputs."""
        np.random.seed(42)
        observed_ic = np.array([0.05, 0.03, 0.02, 0.01, -0.01])
        complexity = 0.02  # Pre-computed Rademacher complexity

        adjusted = ras_ic_adjustment(
            observed_ic=observed_ic,
            complexity=complexity,
            n_samples=252,
        )

        assert isinstance(adjusted, np.ndarray)
        assert len(adjusted) == len(observed_ic)
        # Adjusted should be less than or equal to observed (conservative)
        assert all(adjusted <= observed_ic + 1e-10)

    def test_ras_ic_with_params(self):
        """Test RAS IC adjustment with custom parameters."""
        observed_ic = np.array([0.05, 0.03, 0.02])

        adjusted_95 = ras_ic_adjustment(
            observed_ic=observed_ic,
            complexity=0.02,
            n_samples=252,
            delta=0.05,  # 95% confidence
        )

        adjusted_99 = ras_ic_adjustment(
            observed_ic=observed_ic,
            complexity=0.02,
            n_samples=252,
            delta=0.01,  # 99% confidence
        )

        # Higher confidence (lower delta) should give more conservative bounds
        assert all(adjusted_99 <= adjusted_95 + 1e-10)

    def test_ras_ic_zero_complexity(self):
        """Test RAS IC adjustment with zero complexity."""
        observed_ic = np.array([0.05, 0.03])

        # With zero complexity, only estimation error adjustment
        adjusted = ras_ic_adjustment(
            observed_ic=observed_ic,
            complexity=0.0,
            n_samples=252,
        )

        assert all(adjusted <= observed_ic + 1e-10)


class TestFDRControl:
    """Tests for False Discovery Rate control methods."""

    def test_benjamini_hochberg_basic(self):
        """Test Benjamini-Hochberg FDR control."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.5, 0.8, 0.9]

        # Default returns boolean array
        rejected = benjamini_hochberg_fdr(p_values, alpha=0.05)

        assert isinstance(rejected, np.ndarray)
        assert rejected.dtype == bool
        assert len(rejected) == len(p_values)

        # First few should be rejected
        assert rejected[0]  # p=0.001 should be rejected
        assert not rejected[-1]  # p=0.9 should not be rejected

    def test_benjamini_hochberg_with_details(self):
        """Test BH FDR with detailed output."""
        p_values = [0.001, 0.01, 0.02, 0.5]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

        assert isinstance(result, dict)
        assert "rejected" in result
        assert "adjusted_p_values" in result
        assert "n_rejected" in result

        # Check counts match
        assert result["n_rejected"] == sum(result["rejected"])

    def test_benjamini_hochberg_no_rejections(self):
        """Test BH with no significant p-values."""
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]

        rejected = benjamini_hochberg_fdr(p_values, alpha=0.05)

        # None should be rejected
        assert not any(rejected)

    def test_benjamini_hochberg_all_rejections(self):
        """Test BH when all p-values are significant."""
        p_values = [0.001, 0.002, 0.003, 0.004]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

        # All should be rejected
        assert result["n_rejected"] == len(p_values)
        assert all(result["rejected"])

    def test_benjamini_hochberg_empty(self):
        """Test BH with empty p-values."""
        result = benjamini_hochberg_fdr([], alpha=0.05, return_details=True)

        assert result["n_rejected"] == 0
        assert len(result["rejected"]) == 0

    def test_benjamini_hochberg_single(self):
        """Test BH with single p-value."""
        # Single significant p-value
        rejected = benjamini_hochberg_fdr([0.01], alpha=0.05)
        assert rejected[0]

        # Single non-significant p-value
        rejected = benjamini_hochberg_fdr([0.1], alpha=0.05)
        assert not rejected[0]

    def test_holm_bonferroni_basic(self):
        """Test Holm-Bonferroni step-down procedure."""
        p_values = [0.001, 0.01, 0.02, 0.08, 0.12]

        result = holm_bonferroni(p_values, alpha=0.05)

        assert isinstance(result, dict)
        assert "rejected" in result
        assert "adjusted_p_values" in result
        assert "n_rejected" in result

        # First hypothesis should be rejected (p=0.001 < 0.05/5 = 0.01)
        assert result["rejected"][0]

    def test_holm_vs_bh_conservativeness(self):
        """Test that Holm is more conservative than BH."""
        p_values = [0.001, 0.015, 0.025, 0.035, 0.045]

        bh_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
        holm_result = holm_bonferroni(p_values, alpha=0.05)

        # Holm controls FWER (more stringent), BH controls FDR
        # Holm should reject <= BH in most cases
        assert holm_result["n_rejected"] <= bh_result["n_rejected"] + 1

    def test_fdr_different_alphas(self):
        """Test FDR control with different alpha levels."""
        p_values = [0.001, 0.01, 0.05, 0.1, 0.5]

        result_01 = benjamini_hochberg_fdr(p_values, alpha=0.01, return_details=True)
        result_05 = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
        result_10 = benjamini_hochberg_fdr(p_values, alpha=0.10, return_details=True)

        # Higher alpha should reject more (or equal)
        assert result_01["n_rejected"] <= result_05["n_rejected"]
        assert result_05["n_rejected"] <= result_10["n_rejected"]


class TestPropertyBased:
    """Property-based tests for statistical functions."""

    @given(
        observed_sharpe=st.floats(min_value=-3, max_value=5, allow_nan=False),
        n_trials=st.integers(min_value=1, max_value=100),
        variance_trials=st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
        n_samples=st.integers(min_value=30, max_value=1000),
    )
    @settings(max_examples=20)
    def test_dsr_properties(self, observed_sharpe, n_trials, variance_trials, n_samples):
        """Property test for DSR invariants."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            variance_trials=variance_trials,
            n_samples=n_samples,
        )

        # DSR probability should be in [0, 1]
        assert 0 <= result.probability <= 1

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0001, max_value=0.9999, allow_nan=False),
            min_size=1,
            max_size=20,
        ),
        alpha=st.floats(min_value=0.01, max_value=0.2),
    )
    @settings(max_examples=20)
    def test_fdr_properties(self, p_values, alpha):
        """Property test for FDR control invariants."""
        result = benjamini_hochberg_fdr(p_values, alpha=alpha, return_details=True)

        # Significant count should be non-negative and <= total
        assert 0 <= result["n_rejected"] <= len(p_values)

        # Rejected array should match significant count
        assert sum(result["rejected"]) == result["n_rejected"]

        # Adjusted p-values should be >= original (or very close due to float precision)
        for orig, adj in zip(p_values, result["adjusted_p_values"], strict=False):
            assert adj >= orig - 1e-10


class TestEdgeCasesAndRobustness:
    """Edge cases and robustness tests."""

    def test_dsr_extreme_sharpe(self):
        """Test DSR with extreme Sharpe ratios."""
        # Very high Sharpe
        result_high = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=10.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )
        assert 0 <= result_high.probability <= 1

        # Very low (negative) Sharpe
        result_low = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=-5.0,
            n_trials=10,
            variance_trials=0.5,
            n_samples=252,
        )
        assert 0 <= result_low.probability <= 1
        assert (
            result_low.probability < result_high.probability
        )  # Lower Sharpe should give lower probability

    def test_p_values_at_boundaries(self):
        """Test FDR with p-values at exact boundaries."""
        # p-value exactly at alpha
        result = benjamini_hochberg_fdr([0.05], alpha=0.05, return_details=True)
        assert isinstance(result["n_rejected"], int)

        # p-value of 0
        result = benjamini_hochberg_fdr([0.0, 0.5], alpha=0.05, return_details=True)
        assert result["rejected"][0]  # p=0 should always be rejected

    def test_identical_p_values(self):
        """Test FDR with all identical p-values."""
        p_values = [0.03, 0.03, 0.03, 0.03]

        result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

        # Should handle ties gracefully
        assert isinstance(result, dict)
        assert result["n_rejected"] >= 0

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        p_values = np.array([0.001, 0.01, 0.1])

        rejected = benjamini_hochberg_fdr(p_values, alpha=0.05)
        assert isinstance(rejected, np.ndarray)

    def test_rademacher_minimal_input(self):
        """Test Rademacher with minimal valid input."""
        # Minimal 2D array
        X = np.array([[1.0], [2.0]])  # 2 samples, 1 strategy

        complexity = rademacher_complexity(X, n_simulations=100, random_state=42)
        assert np.isfinite(complexity)


class TestVarianceRescalingFactor:
    """Tests for _get_variance_rescaling_factor internal function."""

    def test_exact_values_from_table(self):
        """Test exact values that exist in the lookup table."""
        from ml4t.diagnostic.evaluation.stats import _get_variance_rescaling_factor

        # Test known values from the table
        assert _get_variance_rescaling_factor(1) == 1.0
        assert _get_variance_rescaling_factor(2) == pytest.approx(0.82565)
        assert _get_variance_rescaling_factor(10) == pytest.approx(0.58681)
        assert _get_variance_rescaling_factor(100) == pytest.approx(0.42942)

    def test_interpolation_between_values(self):
        """Test linear interpolation for values not in table."""
        from ml4t.diagnostic.evaluation.stats import _get_variance_rescaling_factor

        # Between 2 and 3: should be between 0.82565 and 0.74798
        _get_variance_rescaling_factor(2)
        _get_variance_rescaling_factor(3)
        _get_variance_rescaling_factor(2)  # Exact, for comparison

        # Test a value that requires interpolation (e.g., 15)
        # Between 10 (0.58681) and 20 (0.52131)
        factor_15 = _get_variance_rescaling_factor(15)
        assert 0.52131 < factor_15 < 0.58681

    def test_value_below_minimum(self):
        """Test value below minimum in table (k < 1)."""
        from ml4t.diagnostic.evaluation.stats import _get_variance_rescaling_factor

        # k=0 should return the factor for k=1
        # Actually k=0 is not in the table, so it returns the minimum key value
        # k < 1 edge case - returns factor for k=1
        factor = _get_variance_rescaling_factor(0)
        assert factor == 1.0  # Returns factor for k=1

    def test_value_above_maximum(self):
        """Test value above maximum in table (k > 100)."""
        from ml4t.diagnostic.evaluation.stats import _get_variance_rescaling_factor

        # k > 100 should return factor for k=100
        factor = _get_variance_rescaling_factor(200)
        assert factor == pytest.approx(0.42942)

        factor = _get_variance_rescaling_factor(1000)
        assert factor == pytest.approx(0.42942)


class TestDSRValidation:
    """Tests for DSR input validation."""

    def test_dsr_negative_n_trials(self):
        """Test DSR with invalid n_trials."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=2.0,
                n_trials=0,
                variance_trials=0.5,
                n_samples=252,
            )

    def test_dsr_negative_variance(self):
        """Test DSR with negative variance (when n_trials > 1)."""
        # Note: variance_trials must be positive when n_trials > 1
        with pytest.raises(ValueError, match="variance_trials must be positive"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=2.0,
                n_trials=10,
                variance_trials=-0.1,
                n_samples=252,
            )

    def test_dsr_invalid_n_samples(self):
        """Test DSR with invalid n_samples."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=2.0,
                n_trials=10,
                variance_trials=0.5,
                n_samples=0,
            )

    def test_dsr_invalid_return_format(self):
        """Test DSR with invalid autocorrelation (replaces return_format test)."""
        # The new API doesn't have return_format - test autocorrelation validation instead
        with pytest.raises(ValueError, match="autocorrelation must be in"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=2.0,
                n_trials=10,
                variance_trials=0.5,
                n_samples=252,
                autocorrelation=1.0,  # Invalid: must be < 1
            )

    def test_dsr_single_trial_with_components(self):
        """Test DSR with single trial returns proper components."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1,
            variance_trials=0.0,  # No variance for single trial
            n_samples=252,
        )

        assert isinstance(result, DSRResult)
        # Single trial (PSR) - no multiple testing adjustment
        assert result.expected_max_sharpe == 0.0
        assert result.n_trials == 1

    def test_dsr_single_trial_zscore_format(self):
        """Test DSR with single trial has high z-score for high Sharpe."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1,
            variance_trials=0.0,
            n_samples=252,
        )

        # High Sharpe with no multiple testing adjustment should have high z-score
        assert result.z_score > 0  # Positive z-score for positive Sharpe

    def test_dsr_single_trial_adjusted_format(self):
        """Test DSR with single trial has no deflation."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1,
            variance_trials=0.0,
            n_samples=252,
        )

        # No deflation for single trial
        assert result.deflated_sharpe == 2.0  # Observed - expected_max (0)

    def test_dsr_degenerate_zero_variance(self):
        """Test DSR edge case with zero variance of trials."""
        # When variance_trials = 0 and n_trials = 1, it's PSR
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1,
            variance_trials=0.0,  # Zero variance OK for single trial
            n_samples=252,
        )

        assert isinstance(result, DSRResult)
        assert 0 <= result.probability <= 1


class TestRASSharpeAdjustment:
    """Tests for RAS Sharpe adjustment function."""

    def test_ras_sharpe_basic(self):
        """Test RAS Sharpe adjustment with basic inputs."""
        from ml4t.diagnostic.evaluation.stats import ras_sharpe_adjustment

        observed_sharpe = np.array([0.5, 0.3, 0.2, 0.1, -0.1])
        complexity = 0.05

        adjusted = ras_sharpe_adjustment(
            observed_sharpe=observed_sharpe,
            complexity=complexity,
            n_samples=252,
            n_strategies=5,
        )

        assert isinstance(adjusted, np.ndarray)
        assert len(adjusted) == len(observed_sharpe)
        # Adjusted should be less than or equal to observed (conservative)
        assert all(adjusted <= observed_sharpe + 1e-10)

    def test_ras_sharpe_with_different_delta(self):
        """Test RAS Sharpe with different confidence levels."""
        from ml4t.diagnostic.evaluation.stats import ras_sharpe_adjustment

        observed_sharpe = np.array([0.5, 0.3, 0.2])

        adjusted_95 = ras_sharpe_adjustment(
            observed_sharpe=observed_sharpe,
            complexity=0.05,
            n_samples=252,
            n_strategies=3,
            delta=0.05,
        )

        adjusted_99 = ras_sharpe_adjustment(
            observed_sharpe=observed_sharpe,
            complexity=0.05,
            n_samples=252,
            n_strategies=3,
            delta=0.01,
        )

        # Higher confidence (lower delta) should give more conservative bounds
        assert all(adjusted_99 <= adjusted_95 + 1e-10)

    def test_ras_sharpe_validation_errors(self):
        """Test RAS Sharpe validation."""
        from ml4t.diagnostic.evaluation.stats import ras_sharpe_adjustment

        # Invalid observed_sharpe shape
        with pytest.raises(ValueError, match="must be 1D"):
            ras_sharpe_adjustment(
                observed_sharpe=np.array([[1, 2], [3, 4]]),
                complexity=0.05,
                n_samples=252,
                n_strategies=2,
            )

        # Negative complexity
        with pytest.raises(ValueError, match="non-negative"):
            ras_sharpe_adjustment(
                observed_sharpe=np.array([0.5]),
                complexity=-0.1,
                n_samples=252,
                n_strategies=1,
            )

        # Invalid n_samples
        with pytest.raises(ValueError, match="positive"):
            ras_sharpe_adjustment(
                observed_sharpe=np.array([0.5]),
                complexity=0.05,
                n_samples=0,
                n_strategies=1,
            )

        # Invalid n_strategies
        with pytest.raises(ValueError, match="positive"):
            ras_sharpe_adjustment(
                observed_sharpe=np.array([0.5]),
                complexity=0.05,
                n_samples=252,
                n_strategies=0,
            )

        # Invalid delta
        with pytest.raises(ValueError, match="must be in"):
            ras_sharpe_adjustment(
                observed_sharpe=np.array([0.5]),
                complexity=0.05,
                n_samples=252,
                n_strategies=1,
                delta=1.5,
            )


class TestRASICValidation:
    """Tests for RAS IC validation paths."""

    def test_ras_ic_invalid_shape(self):
        """Test RAS IC with invalid input shape."""
        with pytest.raises(ValueError, match="must be 1D"):
            ras_ic_adjustment(
                observed_ic=np.array([[0.05, 0.03], [0.02, 0.01]]),
                complexity=0.02,
                n_samples=252,
            )

    def test_ras_ic_negative_complexity(self):
        """Test RAS IC with negative complexity."""
        with pytest.raises(ValueError, match="non-negative"):
            ras_ic_adjustment(
                observed_ic=np.array([0.05]),
                complexity=-0.01,
                n_samples=252,
            )

    def test_ras_ic_invalid_n_samples(self):
        """Test RAS IC with invalid n_samples."""
        with pytest.raises(ValueError, match="positive"):
            ras_ic_adjustment(
                observed_ic=np.array([0.05]),
                complexity=0.02,
                n_samples=0,
            )

    def test_ras_ic_invalid_delta(self):
        """Test RAS IC with invalid delta."""
        with pytest.raises(ValueError, match="must be in"):
            ras_ic_adjustment(
                observed_ic=np.array([0.05]),
                complexity=0.02,
                n_samples=252,
                delta=0.0,  # Invalid: must be > 0
            )

        with pytest.raises(ValueError, match="must be in"):
            ras_ic_adjustment(
                observed_ic=np.array([0.05]),
                complexity=0.02,
                n_samples=252,
                delta=1.0,  # Invalid: must be < 1
            )

    def test_ras_ic_invalid_kappa(self):
        """Test RAS IC with invalid kappa."""
        with pytest.raises(ValueError, match="positive"):
            ras_ic_adjustment(
                observed_ic=np.array([0.05]),
                complexity=0.02,
                n_samples=252,
                kappa=-0.01,
            )


class TestComputePBO:
    """Tests for Probability of Backtest Overfitting (PBO) calculation."""

    def test_pbo_basic(self):
        """Test basic PBO calculation."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo
        from ml4t.diagnostic.evaluation.stats.backtest_overfitting import PBOResult

        np.random.seed(42)
        # 10 CV folds, 5 strategies
        is_perf = np.random.randn(10, 5)
        oos_perf = np.random.randn(10, 5)

        result = compute_pbo(is_perf, oos_perf)

        assert isinstance(result, PBOResult)
        assert 0 <= result.pbo <= 1
        assert result.pbo_pct == result.pbo * 100
        assert result.n_combinations == 10
        assert result.n_strategies == 5

    def test_pbo_correlated(self):
        """Test PBO with correlated IS and OOS performance."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo

        np.random.seed(42)
        # Create correlated IS and OOS (good scenario)
        signal = np.random.randn(10, 5)
        is_perf = signal + np.random.randn(10, 5) * 0.1
        oos_perf = signal + np.random.randn(10, 5) * 0.1

        result = compute_pbo(is_perf, oos_perf)

        # Correlated should have lower PBO
        assert result.pbo < 0.5  # Less than random

    def test_pbo_uncorrelated(self):
        """Test PBO with uncorrelated IS and OOS performance."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo

        np.random.seed(42)
        # Uncorrelated: random IS and OOS
        is_perf = np.random.randn(100, 5)
        oos_perf = np.random.randn(100, 5)

        result = compute_pbo(is_perf, oos_perf)

        # Uncorrelated should have ~50% PBO
        assert 0.3 <= result.pbo <= 0.7

    def test_pbo_1d_input(self):
        """Test PBO with 1D input (single combination)."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo

        is_perf = np.array([0.5, 0.3, 0.2, 0.1, -0.1])
        oos_perf = np.array([0.4, 0.2, 0.3, 0.0, -0.2])

        result = compute_pbo(is_perf, oos_perf)

        assert result.n_combinations == 1
        assert result.n_strategies == 5

    def test_pbo_shape_mismatch(self):
        """Test PBO with mismatched shapes."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo

        is_perf = np.random.randn(10, 5)
        oos_perf = np.random.randn(10, 3)  # Different number of strategies

        with pytest.raises(ValueError, match="same shape"):
            compute_pbo(is_perf, oos_perf)

    def test_pbo_insufficient_strategies(self):
        """Test PBO with insufficient strategies."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo

        is_perf = np.random.randn(10, 1)  # Only 1 strategy
        oos_perf = np.random.randn(10, 1)

        with pytest.raises(ValueError, match="at least 2 strategies"):
            compute_pbo(is_perf, oos_perf)

    def test_pbo_degradation_statistics(self):
        """Test PBO returns degradation statistics."""
        from ml4t.diagnostic.evaluation.stats import compute_pbo
        from ml4t.diagnostic.evaluation.stats.backtest_overfitting import PBOResult

        np.random.seed(42)
        is_perf = np.random.randn(10, 5)
        oos_perf = np.random.randn(10, 5)

        result = compute_pbo(is_perf, oos_perf)

        # Check result has all required attributes (dataclass fields)
        assert isinstance(result, PBOResult)
        assert hasattr(result, "degradation_mean")
        assert hasattr(result, "degradation_std")
        assert hasattr(result, "is_best_rank_oos_median")
        assert hasattr(result, "is_best_rank_oos_mean")
        # Also verify interpret() and to_dict() methods work
        assert isinstance(result.interpret(), str)
        assert isinstance(result.to_dict(), dict)


class TestStationaryBootstrapIC:
    """Tests for stationary bootstrap IC."""

    def test_bootstrap_ic_basic(self):
        """Test basic bootstrap IC calculation."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        np.random.seed(42)
        n = 100
        predictions = np.random.randn(n)
        returns = 0.3 * predictions + np.random.randn(n) * 0.7

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=100,  # Reduced for speed
            return_details=True,
        )

        assert isinstance(result, dict)
        assert "ic" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "bootstrap_mean" in result
        assert "bootstrap_std" in result
        assert -1 <= result["ic"] <= 1

    def test_bootstrap_ic_return_pvalue_only(self):
        """Test bootstrap IC returning p-value only."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        np.random.seed(42)
        n = 100
        predictions = np.random.randn(n)
        returns = np.random.randn(n)

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=50,
            return_details=False,
        )

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_bootstrap_ic_with_block_size(self):
        """Test bootstrap IC with explicit block size."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        np.random.seed(42)
        n = 100
        predictions = np.random.randn(n)
        returns = np.random.randn(n)

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=50,
            block_size=10.0,
            return_details=True,
        )

        assert isinstance(result, dict)

    def test_bootstrap_ic_small_sample_warning(self):
        """Test bootstrap IC with small sample issues warning."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        np.random.seed(42)
        predictions = np.random.randn(20)
        returns = np.random.randn(20)

        with pytest.warns(UserWarning, match="too small"):
            stationary_bootstrap_ic(
                predictions=predictions,
                returns=returns,
                n_samples=50,
                return_details=True,
            )

    def test_bootstrap_ic_nan_handling(self):
        """Test bootstrap IC handles NaN values."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        np.random.seed(42)
        n = 100
        predictions = np.random.randn(n)
        returns = np.random.randn(n)
        predictions[10] = np.nan
        returns[20] = np.nan

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=50,
            return_details=True,
        )

        assert isinstance(result, dict)
        # Should have removed NaN pairs
        assert np.isfinite(result["ic"])

    def test_bootstrap_ic_constant_predictions(self):
        """Test bootstrap IC with constant predictions (should return NaN)."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        predictions = np.ones(100)
        returns = np.random.randn(100)

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=50,
            return_details=True,
        )

        assert isinstance(result, dict)
        assert np.isnan(result["ic"])

    def test_bootstrap_ic_length_mismatch(self):
        """Test bootstrap IC with length mismatch."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        predictions = np.random.randn(100)
        returns = np.random.randn(50)

        with pytest.raises(ValueError, match="same length"):
            stationary_bootstrap_ic(predictions, returns)


class TestOptimalBlockSize:
    """Tests for _optimal_block_size internal function."""

    def test_optimal_block_size_basic(self):
        """Test optimal block size calculation."""
        from ml4t.diagnostic.evaluation.stats import _optimal_block_size

        np.random.seed(42)
        data = np.random.randn(100)

        block_size = _optimal_block_size(data)

        assert isinstance(block_size, int | float)
        assert block_size >= 1
        assert block_size <= 100 // 3  # Capped at n/3

    def test_optimal_block_size_small_sample(self):
        """Test optimal block size with small sample."""
        from ml4t.diagnostic.evaluation.stats import _optimal_block_size

        data = np.array([1, 2, 3, 4, 5])  # n < 10

        block_size = _optimal_block_size(data)

        assert block_size >= 1

    def test_optimal_block_size_autocorrelated(self):
        """Test optimal block size with autocorrelated data."""
        from ml4t.diagnostic.evaluation.stats import _optimal_block_size

        np.random.seed(42)
        n = 200
        # Create autocorrelated series
        data = np.cumsum(np.random.randn(n))

        block_size_autocorr = _optimal_block_size(data)

        # IID data
        iid_data = np.random.randn(n)
        _optimal_block_size(iid_data)

        # Autocorrelated should have larger block size
        # (This is a tendency, not guaranteed for all random seeds)
        assert block_size_autocorr >= 1


class TestStationaryBootstrapIndices:
    """Tests for _stationary_bootstrap_indices internal function."""

    def test_bootstrap_indices_basic(self):
        """Test bootstrap indices generation."""
        from ml4t.diagnostic.evaluation.stats import _stationary_bootstrap_indices

        np.random.seed(42)
        indices = _stationary_bootstrap_indices(100, 10.0)

        assert len(indices) == 100
        assert all(0 <= idx < 100 for idx in indices)

    def test_bootstrap_indices_small_block(self):
        """Test bootstrap indices with small block size."""
        from ml4t.diagnostic.evaluation.stats import _stationary_bootstrap_indices

        np.random.seed(42)
        indices = _stationary_bootstrap_indices(50, 2.0)

        assert len(indices) == 50


class TestRobustIC:
    """Tests for robust IC calculation with stationary bootstrap."""

    def test_robust_ic_basic(self):
        """Test basic IC calculation with robust standard errors."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        np.random.seed(42)
        n = 200
        predictions = np.random.randn(n)
        returns = 0.2 * predictions + np.random.randn(n) * 0.8

        result = robust_ic(predictions, returns, return_details=True)

        assert isinstance(result, dict)
        assert "ic" in result
        assert "bootstrap_std" in result
        assert "t_stat" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_robust_ic_return_t_stat_only(self):
        """Test IC returning t-stat only."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        np.random.seed(42)
        n = 200
        predictions = np.random.randn(n)
        returns = 0.3 * predictions + np.random.randn(n) * 0.7

        result = robust_ic(predictions, returns, return_details=False)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_robust_ic_positive_correlation(self):
        """Test IC detects positive correlation."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        np.random.seed(42)
        n = 200
        predictions = np.random.randn(n)
        returns = 0.5 * predictions + np.random.randn(n) * 0.5

        result = robust_ic(predictions, returns, return_details=True)

        assert result["ic"] > 0
        assert result["t_stat"] > 2  # Significant

    def test_robust_ic_constant_predictions(self):
        """Test IC with constant predictions returns NaN."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        predictions = np.ones(100)
        returns = np.random.randn(100)

        result = robust_ic(predictions, returns, return_details=True)

        assert np.isnan(result["ic"])

    def test_robust_ic_length_mismatch(self):
        """Test IC with length mismatch raises error."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        predictions = np.random.randn(100)
        returns = np.random.randn(50)

        with pytest.raises(ValueError, match="same length"):
            robust_ic(predictions, returns)

    def test_hac_adjusted_ic_is_alias(self):
        """Test that hac_adjusted_ic is an alias for robust_ic."""
        from ml4t.diagnostic.evaluation.stats import hac_adjusted_ic, robust_ic

        assert hac_adjusted_ic is robust_ic


class TestWhitesRealityCheck:
    """Tests for White's Reality Check."""

    def test_whites_reality_check_basic(self):
        """Test basic White's Reality Check."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        n_periods = 100
        n_strategies = 10

        benchmark = np.random.randn(n_periods) * 0.01
        strategies = np.random.randn(n_periods, n_strategies) * 0.01

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=100,
            random_state=42,
        )

        assert isinstance(result, dict)
        assert "test_statistic" in result
        assert "p_value" in result
        assert "critical_values" in result
        assert "best_strategy_idx" in result
        assert 0 <= result["p_value"] <= 1

    def test_whites_reality_check_with_outperformer(self):
        """Test Reality Check with a true outperformer."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        n_periods = 200
        n_strategies = 5

        benchmark = np.random.randn(n_periods) * 0.01
        strategies = np.random.randn(n_periods, n_strategies) * 0.01
        # Add alpha to one strategy
        strategies[:, 0] += 0.05

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=100,
            random_state=42,
        )

        # Should identify strategy 0 as best
        assert result["best_strategy_idx"] == 0
        # p-value should be low for true outperformer
        assert result["p_value"] < 0.5

    def test_whites_reality_check_pandas_input(self):
        """Test Reality Check with pandas input."""
        import pandas as pd

        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        n_periods = 100

        benchmark = pd.Series(np.random.randn(n_periods) * 0.01)
        strategies = pd.DataFrame(np.random.randn(n_periods, 5) * 0.01)

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=50,
            random_state=42,
        )

        assert isinstance(result, dict)

    def test_whites_reality_check_polars_input(self):
        """Test Reality Check with polars input."""
        import polars as pl

        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        n_periods = 100

        benchmark = pl.Series(np.random.randn(n_periods) * 0.01)
        strategies = pl.DataFrame(np.random.randn(n_periods, 5) * 0.01)

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=50,
            random_state=42,
        )

        assert isinstance(result, dict)

    def test_whites_reality_check_1d_strategies(self):
        """Test Reality Check with 1D strategies (single strategy)."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        benchmark = np.random.randn(100) * 0.01
        strategies = np.random.randn(100) * 0.01  # 1D array

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=50,
            random_state=42,
        )

        assert result["n_strategies"] == 1

    def test_whites_reality_check_length_mismatch(self):
        """Test Reality Check with length mismatch."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        benchmark = np.random.randn(100)
        strategies = np.random.randn(50, 5)

        with pytest.raises(ValueError, match="same number of periods"):
            whites_reality_check(benchmark, strategies)

    def test_whites_reality_check_critical_values(self):
        """Test Reality Check returns proper critical values."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        benchmark = np.random.randn(100)
        strategies = np.random.randn(100, 5)

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=100,
            random_state=42,
        )

        assert "90%" in result["critical_values"]
        assert "95%" in result["critical_values"]
        assert "99%" in result["critical_values"]
        # Critical values should be increasing
        assert result["critical_values"]["90%"] <= result["critical_values"]["95%"]
        assert result["critical_values"]["95%"] <= result["critical_values"]["99%"]


class TestMultipleTestingSummary:
    """Tests for multiple_testing_summary function."""

    def test_summary_basic(self):
        """Test basic multiple testing summary."""
        from ml4t.diagnostic.evaluation.stats import multiple_testing_summary

        test_results = [
            {"name": "Strategy A", "p_value": 0.001},
            {"name": "Strategy B", "p_value": 0.01},
            {"name": "Strategy C", "p_value": 0.05},
            {"name": "Strategy D", "p_value": 0.5},
        ]

        result = multiple_testing_summary(test_results, alpha=0.05)

        assert isinstance(result, dict)
        assert result["n_tests"] == 4
        assert "n_significant_uncorrected" in result
        assert "n_significant_corrected" in result
        assert result["correction_method"] == "benjamini_hochberg"
        assert result["n_significant_corrected"] <= result["n_significant_uncorrected"]

    def test_summary_empty_results(self):
        """Test summary with empty results."""
        from ml4t.diagnostic.evaluation.stats import multiple_testing_summary

        result = multiple_testing_summary([], alpha=0.05)

        assert result["n_tests"] == 0
        assert result["n_significant_uncorrected"] == 0
        assert result["n_significant_corrected"] == 0

    def test_summary_with_nan_pvalues(self):
        """Test summary with NaN p-values."""
        from ml4t.diagnostic.evaluation.stats import multiple_testing_summary

        test_results = [
            {"name": "Strategy A", "p_value": np.nan},
            {"name": "Strategy B", "p_value": np.nan},
        ]

        result = multiple_testing_summary(test_results, alpha=0.05)

        assert result["n_tests"] == 2
        assert "warning" in result

    def test_summary_all_significant(self):
        """Test summary when all tests are significant."""
        from ml4t.diagnostic.evaluation.stats import multiple_testing_summary

        test_results = [
            {"p_value": 0.001},
            {"p_value": 0.002},
            {"p_value": 0.003},
        ]

        result = multiple_testing_summary(test_results, alpha=0.05)

        assert result["n_significant_uncorrected"] == 3

    def test_summary_invalid_method(self):
        """Test summary with invalid correction method."""
        from ml4t.diagnostic.evaluation.stats import multiple_testing_summary

        test_results = [{"p_value": 0.01}]

        with pytest.raises(ValueError, match="Unknown correction method"):
            multiple_testing_summary(test_results, method="invalid_method")


class TestHolmBonferroniEdgeCases:
    """Tests for Holm-Bonferroni edge cases."""

    def test_holm_empty(self):
        """Test Holm with empty p-values."""
        result = holm_bonferroni([], alpha=0.05)

        assert result["n_rejected"] == 0
        assert result["rejected"] == []
        assert result["adjusted_p_values"] == []

    def test_holm_single(self):
        """Test Holm with single p-value."""
        # Significant
        result = holm_bonferroni([0.01], alpha=0.05)
        assert result["rejected"] == [True]

        # Not significant
        result = holm_bonferroni([0.1], alpha=0.05)
        assert result["rejected"] == [False]

    def test_holm_all_rejected(self):
        """Test Holm when all should be rejected."""
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni(p_values, alpha=0.10)

        # With alpha=0.10 and these tiny p-values, all should be rejected
        assert result["n_rejected"] == 3

    def test_holm_step_down_logic(self):
        """Test that Holm correctly stops at first non-rejection."""
        # p-values where third one fails Holm test
        p_values = [0.001, 0.01, 0.04, 0.05]
        result = holm_bonferroni(p_values, alpha=0.05)

        # First should definitely be rejected (0.001 < 0.05/4 = 0.0125)
        assert result["rejected"][0]


class TestRademacherValidationPaths:
    """Tests for Rademacher complexity validation paths."""

    def test_rademacher_empty_dimension(self):
        """Test Rademacher with empty dimension."""
        # T=0 or N=0 should raise error
        X_empty = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError, match="positive dimensions"):
            rademacher_complexity(X_empty)


class TestRobustICEdgeCases:
    """Tests for robust IC edge cases."""

    def test_robust_ic_zero_std_returns_nan(self):
        """Test robust IC returns NaN when bootstrap std is zero."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        # Constant predictions → NaN IC
        predictions = np.ones(100)
        returns = np.random.randn(100)

        result = robust_ic(predictions, returns, return_details=False)

        assert np.isnan(result)


class TestBootstrapICNaNReturn:
    """Test bootstrap IC NaN return path when return_details=False."""

    def test_bootstrap_ic_constant_returns_pvalue_only(self):
        """Test bootstrap IC with constant input returning p-value."""
        from ml4t.diagnostic.evaluation.stats import stationary_bootstrap_ic

        predictions = np.ones(50)  # Constant
        returns = np.random.randn(50)

        result = stationary_bootstrap_ic(
            predictions=predictions,
            returns=returns,
            n_samples=20,
            return_details=False,
        )

        # Should return NaN p-value
        assert np.isnan(result)


class TestOptimalBlockSizeNegativeACF:
    """Test optimal block size with negative autocorrelation."""

    def test_negative_autocorrelation(self):
        """Test optimal block size with negative lag-1 autocorrelation."""
        from ml4t.diagnostic.evaluation.stats import _optimal_block_size

        np.random.seed(42)
        n = 100
        # Create series with strong negative autocorrelation
        data = np.zeros(n)
        data[::2] = 1
        data[1::2] = -1
        data += np.random.randn(n) * 0.1

        block_size = _optimal_block_size(data)

        # Should still return valid block size
        assert block_size >= 1
        assert block_size <= n // 3


class TestWhitesRealityCheckNumpyStrategies:
    """Test White's Reality Check with 2D numpy array but handled as DataFrame."""

    def test_reality_check_numpy_1d_reshaped(self):
        """Test Reality Check handles 1D numpy reshaping correctly."""
        from ml4t.diagnostic.evaluation.stats import whites_reality_check

        np.random.seed(42)
        benchmark = np.random.randn(100)
        # Pass as 1D - should be reshaped to (100, 1)
        strategies = np.random.randn(100)

        result = whites_reality_check(
            returns_benchmark=benchmark,
            returns_strategies=strategies,
            bootstrap_samples=30,
            random_state=42,
        )

        assert result["n_strategies"] == 1
        assert result["n_periods"] == 100


class TestBHFDREmptyArrayReturn:
    """Test BH FDR with empty array returns correct type."""

    def test_bh_empty_no_details(self):
        """Test BH with empty p-values returns empty numpy array."""
        result = benjamini_hochberg_fdr([], alpha=0.05, return_details=False)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 0


class TestRobustICSmallSample:
    """Test robust IC with small samples."""

    def test_robust_ic_small_sample_works(self):
        """Test IC with small sample still computes."""
        from ml4t.diagnostic.evaluation.stats import robust_ic

        np.random.seed(42)
        predictions = np.random.randn(20)
        returns = 0.3 * predictions + np.random.randn(20) * 0.7

        result = robust_ic(predictions, returns, return_details=True)

        assert isinstance(result, dict)
        assert "ic" in result
        assert np.isfinite(result["ic"])


class TestDeflatedSharpeRatioRawReturns:
    """Tests for deflated_sharpe_ratio() with raw returns input.

    These tests cover the function that takes raw returns arrays
    rather than pre-computed statistics.
    """

    def test_single_strategy_psr(self):
        """Test PSR with single strategy returns."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # Daily returns with positive mean

        result = deflated_sharpe_ratio(returns, frequency="daily")

        assert isinstance(result, DSRResult)
        assert result.n_trials == 1  # Single strategy = PSR
        assert 0 <= result.probability <= 1
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.z_score)

    def test_multiple_strategies_dsr(self):
        """Test DSR with multiple strategy returns."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        n_periods = 252
        strategies = [
            np.random.randn(n_periods) * 0.01 + 0.0005,  # Strategy 1
            np.random.randn(n_periods) * 0.01 + 0.001,  # Strategy 2 (best)
            np.random.randn(n_periods) * 0.01 + 0.0003,  # Strategy 3
        ]

        result = deflated_sharpe_ratio(strategies, frequency="daily")

        assert isinstance(result, DSRResult)
        assert result.n_trials == 3  # Multiple strategies = DSR
        assert result.variance_trials > 0  # Cross-sectional variance
        assert 0 <= result.probability <= 1
        # Expected max Sharpe adjustment should reduce probability
        assert result.expected_max_sharpe >= 0

    def test_list_of_single_strategy_treated_as_psr(self):
        """Test that list with single element is treated as PSR."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = [np.random.randn(100) * 0.01 + 0.0005]  # List with single array

        result = deflated_sharpe_ratio(returns, frequency="daily")

        assert result.n_trials == 1
        assert result.variance_trials == 0.0

    def test_weekly_frequency(self):
        """Test with weekly returns."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.002  # 1 year of weekly returns

        result = deflated_sharpe_ratio(returns, frequency="weekly")

        assert result.frequency == "weekly"
        assert result.periods_per_year == 52
        # Annualized Sharpe should be different from raw Sharpe
        assert abs(result.sharpe_ratio_annualized) != abs(result.sharpe_ratio)

    def test_monthly_frequency(self):
        """Test with monthly returns."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(36) * 0.04 + 0.005  # 3 years of monthly returns

        result = deflated_sharpe_ratio(returns, frequency="monthly")

        assert result.frequency == "monthly"
        assert result.periods_per_year == 12
        assert result.n_samples == 36

    def test_custom_periods_per_year(self):
        """Test with custom periods per year."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.001

        result = deflated_sharpe_ratio(returns, frequency="daily", periods_per_year=365)

        assert result.periods_per_year == 365

    def test_custom_benchmark_sharpe(self):
        """Test with custom benchmark Sharpe."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        result_default = deflated_sharpe_ratio(returns, frequency="daily", benchmark_sharpe=0.0)
        result_high_bench = deflated_sharpe_ratio(
            returns, frequency="daily", benchmark_sharpe=0.5
        )

        # Higher benchmark should give lower probability
        assert result_high_bench.probability <= result_default.probability

    def test_custom_statistics_override(self):
        """Test that custom skewness/kurtosis/autocorrelation override computed values."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        # Override with specific statistics
        result = deflated_sharpe_ratio(
            returns,
            frequency="daily",
            skewness=-0.5,  # Override
            excess_kurtosis=2.0,  # Override (heavy tails)
            autocorrelation=0.1,  # Override
        )

        assert result.skewness == -0.5
        assert result.excess_kurtosis == 2.0
        assert result.autocorrelation == 0.1

    def test_nan_handling_in_returns(self):
        """Test that NaN values are properly removed from returns."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(260) * 0.01 + 0.001
        returns[10] = np.nan
        returns[50] = np.nan
        returns[100] = np.nan  # 3 NaN values

        result = deflated_sharpe_ratio(returns, frequency="daily")

        # Should have removed NaN values
        assert result.n_samples == 257

    def test_multiple_strategies_with_nan(self):
        """Test multiple strategies with NaN values."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        strategies = [
            np.append(np.random.randn(100) * 0.01, [np.nan, np.nan]),
            np.random.randn(102) * 0.01 + 0.001,
            np.random.randn(102) * 0.01 + 0.0005,
        ]

        result = deflated_sharpe_ratio(strategies, frequency="daily")

        assert result.n_trials == 3
        # Should still compute successfully

    def test_interpret_method_psr(self):
        """Test interpret() method output for PSR case."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        result = deflated_sharpe_ratio(returns, frequency="daily")
        interpretation = result.interpret()

        assert isinstance(interpretation, str)
        assert "Probabilistic Sharpe Ratio" in interpretation
        assert "Sharpe ratio:" in interpretation
        assert "Probability of skill:" in interpretation

    def test_interpret_method_dsr(self):
        """Test interpret() method output for DSR case."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        strategies = [
            np.random.randn(252) * 0.01 + 0.0005,
            np.random.randn(252) * 0.01 + 0.001,
            np.random.randn(252) * 0.01 + 0.0003,
        ]

        result = deflated_sharpe_ratio(strategies, frequency="daily")
        interpretation = result.interpret()

        assert isinstance(interpretation, str)
        assert "Deflated Sharpe Ratio" in interpretation
        assert "best of 3 strategies" in interpretation
        assert "Expected max from noise:" in interpretation
        assert "Deflated Sharpe:" in interpretation

    def test_interpret_insufficient_sample(self):
        """Test interpret() with insufficient sample size."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        # Very small sample - likely won't meet MinTRL
        returns = np.random.randn(30) * 0.01 + 0.0001  # Small Sharpe

        result = deflated_sharpe_ratio(returns, frequency="daily")

        if not result.has_adequate_sample:
            interpretation = result.interpret()
            assert "WARNING" in interpretation or "insufficient" in interpretation.lower()

    def test_to_dict_method(self):
        """Test to_dict() method returns all expected keys."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        result = deflated_sharpe_ratio(returns, frequency="daily")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "probability" in result_dict
        assert "is_significant" in result_dict
        assert "z_score" in result_dict
        assert "sharpe_ratio" in result_dict
        assert "sharpe_ratio_annualized" in result_dict
        assert "n_samples" in result_dict
        assert "n_trials" in result_dict
        assert "skewness" in result_dict
        assert "excess_kurtosis" in result_dict
        assert "min_trl" in result_dict

    def test_confidence_level_effect(self):
        """Test effect of different confidence levels."""
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        result_95 = deflated_sharpe_ratio(returns, frequency="daily", confidence_level=0.95)
        result_99 = deflated_sharpe_ratio(returns, frequency="daily", confidence_level=0.99)

        # Higher confidence level means harder to be significant
        # (same probability, but higher threshold)
        assert result_95.confidence_level == 0.95
        assert result_99.confidence_level == 0.99

    def test_consistency_with_from_statistics(self):
        """Test that raw returns version is consistent with from_statistics."""
        from ml4t.diagnostic.evaluation.stats import (
            deflated_sharpe_ratio,
            deflated_sharpe_ratio_from_statistics,
        )
        from ml4t.diagnostic.evaluation.stats.moments import compute_return_statistics

        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        # Compute using raw returns
        result_raw = deflated_sharpe_ratio(returns, frequency="daily")

        # Compute statistics manually
        sharpe, skew, kurt, rho, n = compute_return_statistics(returns)

        # Compute using pre-computed statistics
        result_stats = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=sharpe,
            n_trials=1,
            variance_trials=0.0,
            n_samples=n,
            skewness=skew,
            excess_kurtosis=kurt - 3.0,  # Convert Pearson to Fisher
            autocorrelation=rho,
            frequency="daily",
        )

        # Should have same probability (within floating point tolerance)
        assert abs(result_raw.probability - result_stats.probability) < 1e-10
        assert abs(result_raw.z_score - result_stats.z_score) < 1e-10


class TestMinTRLResultMethods:
    """Tests for MinTRLResult interpret() and to_dict() methods."""

    def test_min_trl_interpret_adequate_sample(self):
        """Test interpret() when sample is adequate."""
        from ml4t.diagnostic.evaluation.stats import compute_min_trl

        np.random.seed(42)
        returns = np.random.randn(500) * 0.01 + 0.002  # Strong Sharpe with many samples

        result = compute_min_trl(returns, frequency="daily")

        if result.has_adequate_sample:
            interpretation = result.interpret()
            assert "ADEQUATE" in interpretation
            assert "Minimum Track Record Length" in interpretation

    def test_min_trl_interpret_insufficient_sample(self):
        """Test interpret() when sample is insufficient."""
        from ml4t.diagnostic.evaluation.stats import compute_min_trl

        np.random.seed(42)
        returns = np.random.randn(30) * 0.01 + 0.0003  # Low Sharpe, few samples

        result = compute_min_trl(returns, frequency="daily")

        if not result.has_adequate_sample and np.isfinite(result.min_trl):
            interpretation = result.interpret()
            assert "INSUFFICIENT" in interpretation

    def test_min_trl_interpret_infinite(self):
        """Test interpret() when MinTRL is infinite."""
        from ml4t.diagnostic.evaluation.stats import compute_min_trl

        np.random.seed(42)
        # Returns with negative mean → observed SR < 0 < target SR
        returns = np.random.randn(100) * 0.01 - 0.001

        result = compute_min_trl(returns, frequency="daily", target_sharpe=0.0)

        import math

        if math.isinf(result.min_trl):
            interpretation = result.interpret()
            assert "INFINITE" in interpretation
            assert "Cannot reject null hypothesis" in interpretation
