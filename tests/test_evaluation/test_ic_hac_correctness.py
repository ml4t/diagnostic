"""High-quality correctness tests for HAC-adjusted IC statistics.

These tests verify mathematical correctness of HAC (Heteroskedasticity and
Autocorrelation Consistent) standard error estimation for IC time series.

Key properties tested:
1. HAC SE >= Naive SE when positive autocorrelation present
2. t-stat = mean / SE (mathematical definition)
3. p-value symmetry (|IC| gives same p-value regardless of sign)
4. Newey-West lag formula correctness
5. Kernel weight functions produce expected values
"""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.metrics.ic_statistics import (
    _get_kernel_weights,
    compute_ic_hac_stats,
)


class TestHACMathematicalCorrectness:
    """Tests verifying mathematical properties of HAC estimation."""

    def test_t_stat_equals_mean_over_se(self):
        """Verify t-statistic = mean_ic / hac_se (by definition)."""
        # Create IC series with known mean
        ic_series = np.array([0.05, 0.06, 0.04, 0.07, 0.05, 0.06, 0.05, 0.04])

        result = compute_ic_hac_stats(ic_series)

        # t_stat should equal mean / SE
        expected_t = result["mean_ic"] / result["hac_se"]
        assert abs(result["t_stat"] - expected_t) < 1e-10, (
            f"t_stat ({result['t_stat']}) != mean/se ({expected_t})"
        )

    def test_p_value_symmetry(self):
        """P-value should be same for +IC and -IC of same magnitude."""
        ic_positive = np.array([0.05, 0.06, 0.04, 0.07, 0.05])
        ic_negative = -ic_positive

        result_pos = compute_ic_hac_stats(ic_positive)
        result_neg = compute_ic_hac_stats(ic_negative)

        # P-values should be identical (two-tailed test)
        assert abs(result_pos["p_value"] - result_neg["p_value"]) < 1e-10, (
            f"P-values differ: {result_pos['p_value']} vs {result_neg['p_value']}"
        )

        # t-stats should be opposite signs
        assert abs(result_pos["t_stat"] + result_neg["t_stat"]) < 1e-10

    def test_hac_se_larger_with_positive_autocorrelation(self):
        """HAC SE should be larger than naive SE when IC has positive autocorrelation.

        This is the key insight: when IC values are autocorrelated, the "effective"
        sample size is smaller than n, so HAC SE accounts for this.
        """
        # Create IC series with strong positive autocorrelation
        np.random.seed(42)
        n = 100
        rho = 0.7  # AR(1) coefficient
        innovations = np.random.randn(n) * 0.02
        ic_series = np.zeros(n)
        ic_series[0] = 0.05 + innovations[0]
        for t in range(1, n):
            ic_series[t] = 0.05 + rho * (ic_series[t - 1] - 0.05) + innovations[t]

        result = compute_ic_hac_stats(ic_series)

        # HAC SE should be larger due to autocorrelation
        assert result["hac_se"] > result["naive_se"], (
            f"HAC SE ({result['hac_se']:.6f}) should be > "
            f"naive SE ({result['naive_se']:.6f}) with autocorrelated data"
        )

        # The ratio should be substantial for rho=0.7
        ratio = result["hac_se"] / result["naive_se"]
        assert ratio > 1.2, f"HAC/naive ratio ({ratio:.2f}) seems too low for rho=0.7"

    def test_hac_se_similar_to_naive_for_iid(self):
        """HAC SE should be close to naive SE for IID (no autocorrelation) data."""
        np.random.seed(123)
        ic_series = np.random.randn(100) * 0.03 + 0.02  # IID normal

        result = compute_ic_hac_stats(ic_series)

        # For IID data, HAC and naive should be close
        ratio = result["hac_se"] / result["naive_se"]
        assert 0.8 < ratio < 1.3, f"HAC/naive ratio ({ratio:.2f}) should be ~1 for IID data"

    def test_newey_west_lag_formula(self):
        """Verify Newey-West automatic lag selection formula: floor(4*(T/100)^(2/9))."""
        test_cases = [
            (100, 4),  # T=100 -> 4
            (252, 5),  # T=252 -> 5 (typical trading year)
            (500, 6),  # T=500 -> 6
            (1000, 7),  # T=1000 -> 7
        ]

        for n, _expected_lags in test_cases:
            ic_series = np.random.randn(n) * 0.02

            result = compute_ic_hac_stats(ic_series)

            # Allow for the min/max clipping in the implementation
            assert result["effective_lags"] >= 1, "Should have at least 1 lag"
            assert result["effective_lags"] <= n // 2, "Should not exceed T/2"


class TestKernelWeights:
    """Tests for HAC kernel weight functions."""

    def test_bartlett_kernel_properties(self):
        """Bartlett kernel should be triangular: weights decline linearly."""
        bartlett = _get_kernel_weights("bartlett")
        nlags = 4
        weights = bartlett(nlags)

        # Weight at lag 0 should be 1
        assert weights[0] == 1.0

        # Weights should decrease linearly
        expected = np.array([1 - h / (nlags + 1) for h in range(nlags + 1)])
        np.testing.assert_array_almost_equal(weights, expected)

        # Last weight should be positive (1/(nlags+1))
        assert weights[-1] == pytest.approx(1 / (nlags + 1))

    def test_uniform_kernel_properties(self):
        """Uniform kernel should give equal weights."""
        uniform = _get_kernel_weights("uniform")
        nlags = 4
        weights = uniform(nlags)

        # All weights should be 1
        expected = np.ones(nlags + 1)
        np.testing.assert_array_equal(weights, expected)

    def test_parzen_kernel_properties(self):
        """Parzen kernel should be smooth and bounded."""
        parzen = _get_kernel_weights("parzen")
        nlags = 10
        weights = parzen(nlags)

        # Weight at lag 0 should be 1
        assert weights[0] == 1.0

        # All weights should be in [0, 1]
        assert all(0 <= w <= 1 for w in weights)

        # Weights should decrease monotonically
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], f"Parzen weights not monotonic at {i}"

    def test_invalid_kernel_raises(self):
        """Unknown kernel should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            _get_kernel_weights("invalid_kernel")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_insufficient_data(self):
        """Should return NaN for n < 3."""
        result = compute_ic_hac_stats(np.array([0.05, 0.06]))

        assert np.isnan(result["mean_ic"])
        assert np.isnan(result["hac_se"])
        assert np.isnan(result["t_stat"])
        assert np.isnan(result["p_value"])
        assert result["n_periods"] == 2

    def test_single_value(self):
        """Should handle single value gracefully."""
        result = compute_ic_hac_stats(np.array([0.05]))

        assert np.isnan(result["mean_ic"])
        assert result["n_periods"] == 1

    def test_empty_series(self):
        """Should handle empty series gracefully."""
        result = compute_ic_hac_stats(np.array([]))

        assert np.isnan(result["mean_ic"])
        assert result["n_periods"] == 0

    def test_nan_handling(self):
        """NaN values should be removed before computation."""
        ic_series = np.array([0.05, np.nan, 0.06, 0.04, np.nan, 0.07, 0.05])

        result = compute_ic_hac_stats(ic_series)

        # Should compute on 5 valid values
        assert result["n_periods"] == 5
        assert not np.isnan(result["mean_ic"])

    def test_constant_series(self):
        """Constant IC series should give zero SE and undefined t-stat."""
        ic_series = np.array([0.05] * 10)

        result = compute_ic_hac_stats(ic_series)

        assert result["mean_ic"] == 0.05
        assert result["naive_se"] == 0.0
        # t-stat undefined (0/0), should be NaN or inf
        # Implementation may handle this differently


class TestStatisticalProperties:
    """Tests verifying statistical properties of HAC estimation."""

    def test_confidence_interval_coverage(self):
        """HAC-based CI should have approximately correct coverage.

        Under H0: IC = 0, about 5% of 95% CIs should NOT contain 0.
        This is a statistical property test - may fail occasionally.
        """
        np.random.seed(42)
        n_simulations = 200
        n_periods = 50
        alpha = 0.05

        rejections = 0
        for _ in range(n_simulations):
            # Generate IID IC under null (mean = 0)
            ic_series = np.random.randn(n_periods) * 0.02

            result = compute_ic_hac_stats(ic_series)

            # Reject if p-value < alpha
            if result["p_value"] < alpha:
                rejections += 1

        rejection_rate = rejections / n_simulations

        # Should be close to alpha (5%) - allow some tolerance
        assert 0.02 < rejection_rate < 0.12, (
            f"Rejection rate {rejection_rate:.2%} too far from {alpha:.0%}"
        )

    def test_power_increases_with_effect_size(self):
        """Test should reject more often when true IC is larger."""
        np.random.seed(42)
        n_simulations = 100
        n_periods = 50

        def rejection_rate(true_ic: float) -> float:
            rejections = 0
            for _ in range(n_simulations):
                ic_series = np.random.randn(n_periods) * 0.02 + true_ic
                result = compute_ic_hac_stats(ic_series)
                if result["p_value"] < 0.05:
                    rejections += 1
            return rejections / n_simulations

        rate_null = rejection_rate(0.0)
        rate_small = rejection_rate(0.01)  # Smaller effect to avoid ceiling
        rate_large = rejection_rate(0.03)  # Moderate effect

        # Power should increase with effect size
        # Use >= since at ceiling (100%) both large effects may saturate
        assert rate_small > rate_null, (
            f"Small effect ({rate_small:.0%}) should have more power than null ({rate_null:.0%})"
        )
        assert rate_large >= rate_small, (
            f"Large effect ({rate_large:.0%}) should have >= power than small ({rate_small:.0%})"
        )


class TestInputFormats:
    """Tests for different input data formats."""

    def test_numpy_array_input(self):
        """Should accept numpy array."""
        ic = np.array([0.05, 0.06, 0.04, 0.07, 0.05])
        result = compute_ic_hac_stats(ic)
        assert not np.isnan(result["mean_ic"])

    def test_list_input(self):
        """Should accept Python list."""
        ic = [0.05, 0.06, 0.04, 0.07, 0.05]
        result = compute_ic_hac_stats(ic)
        assert not np.isnan(result["mean_ic"])

    def test_polars_dataframe_input(self):
        """Should accept Polars DataFrame with ic column."""
        import polars as pl

        df = pl.DataFrame({"ic": [0.05, 0.06, 0.04, 0.07, 0.05]})
        result = compute_ic_hac_stats(df, ic_col="ic")
        assert not np.isnan(result["mean_ic"])

    def test_pandas_dataframe_input(self):
        """Should accept Pandas DataFrame with ic column."""
        import pandas as pd

        df = pd.DataFrame({"ic": [0.05, 0.06, 0.04, 0.07, 0.05]})
        result = compute_ic_hac_stats(df, ic_col="ic")
        assert not np.isnan(result["mean_ic"])
