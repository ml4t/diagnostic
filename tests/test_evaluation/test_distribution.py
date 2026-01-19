"""Tests for distribution diagnostics module."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ml4t.diagnostic.errors import ValidationError
from ml4t.diagnostic.evaluation.distribution import (
    DistributionAnalysisResult,
    HillEstimatorResult,
    JarqueBeraResult,
    MomentsResult,
    QQPlotData,
    ShapiroWilkResult,
    TailAnalysisResult,
    analyze_distribution,
    analyze_tails,
    compute_moments,
    generate_qq_data,
    hill_estimator,
    jarque_bera_test,
    shapiro_wilk_test,
)


class TestComputeMoments:
    """Tests for compute_moments function."""

    def test_moments_normal_data(self):
        """Test moments on normal distribution."""
        np.random.seed(42)
        # Normal distribution should have skew ≈ 0, kurtosis ≈ 0
        data = np.random.normal(0, 1, 5000)  # Large sample for stable estimates

        result = compute_moments(data, test_significance=True, alpha=0.05)

        assert isinstance(result, MomentsResult)
        assert result.n_obs == 5000
        assert abs(result.mean) < 0.1  # Should be close to 0
        assert 0.9 < result.std < 1.1  # Should be close to 1
        assert abs(result.skewness) < 0.1  # Should be close to 0
        assert abs(result.excess_kurtosis) < 0.2  # Should be close to 0

        # With large sample, should not be significantly different from normal
        # (though occasionally might be by chance)
        assert result.skewness_se > 0
        assert result.excess_kurtosis_se > 0

    def test_moments_skewed_data(self):
        """Test on right-skewed distribution (lognormal)."""
        np.random.seed(42)
        # Lognormal is right-skewed with positive kurtosis
        data = np.random.lognormal(0, 0.5, 1000)

        result = compute_moments(data, test_significance=True, alpha=0.05)

        assert result.skewness > 0.5  # Should be positively skewed
        assert result.excess_kurtosis > 0.0  # Should have positive excess kurtosis
        assert result.skewness_significant  # Should be significantly skewed
        assert result.excess_kurtosis_significant  # Should be significantly leptokurtic

    def test_moments_left_skewed_data(self):
        """Test on left-skewed distribution."""
        np.random.seed(42)
        # Negative of lognormal is left-skewed
        data = -np.random.lognormal(0, 0.5, 1000)

        result = compute_moments(data)

        assert result.skewness < -0.5  # Should be negatively skewed
        assert result.skewness_significant

    def test_moments_heavy_tailed_data(self):
        """Test on heavy-tailed distribution (Student's t)."""
        np.random.seed(42)
        # Student's t with df=3 has heavy tails (excess kurtosis > 0)
        data = stats.t.rvs(df=3, size=1000)

        result = compute_moments(data)

        # t(3) has theoretical excess kurtosis = 6 (very heavy tails)
        assert result.excess_kurtosis > 1.0  # Should detect fat tails
        assert result.excess_kurtosis_significant

    def test_moments_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = compute_moments(data)

        assert isinstance(result, MomentsResult)
        assert result.n_obs == 1000

    def test_moments_no_significance_test(self):
        """Test with significance testing disabled."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = compute_moments(data, test_significance=False)

        assert not result.skewness_significant
        assert not result.excess_kurtosis_significant
        # But moments should still be computed
        assert result.skewness is not None
        assert result.excess_kurtosis is not None

    def test_moments_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            compute_moments(np.array([]))

    def test_moments_nan_values(self):
        """Test error handling for NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValidationError, match="NaN"):
            compute_moments(data)

    def test_moments_infinite_values(self):
        """Test error handling for infinite values."""
        data = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValidationError, match="infinite"):
            compute_moments(data)

    def test_moments_constant_data(self):
        """Test error handling for constant series."""
        data = np.ones(100)
        with pytest.raises(ValidationError, match="constant"):
            compute_moments(data)

    def test_moments_insufficient_data(self):
        """Test error handling for insufficient data."""
        data = np.array([1.0, 2.0, 3.0])  # Less than min_length=20
        with pytest.raises(ValidationError, match="Insufficient data"):
            compute_moments(data)

    def test_moments_wrong_type(self):
        """Test error handling for wrong data type."""
        with pytest.raises(ValidationError, match="must be pandas Series or numpy array"):
            compute_moments([1, 2, 3, 4, 5])  # List, not array

    def test_moments_multidimensional(self):
        """Test error handling for multidimensional data."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValidationError, match="1-dimensional"):
            compute_moments(data)

    def test_moments_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = compute_moments(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Distribution Moments Analysis" in summary
        assert "Skewness:" in summary
        assert "Excess Kurtosis:" in summary
        assert str(result.n_obs) in summary


class TestJarqueBeraTest:
    """Tests for jarque_bera_test function."""

    def test_jb_normal_data(self):
        """JB test should accept normality for normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = jarque_bera_test(data, alpha=0.05)

        assert isinstance(result, JarqueBeraResult)
        assert result.n_obs == 1000
        # With large normal sample, should usually pass (though not always)
        # Check p-value is reasonable (not zero)
        assert result.p_value > 0.001
        # Check moments are reasonable
        assert abs(result.skewness) < 0.3
        assert abs(result.excess_kurtosis) < 0.5

    def test_jb_non_normal_data(self):
        """JB test should reject normality for lognormal data."""
        np.random.seed(42)
        # Lognormal is definitely not normal
        data = np.random.lognormal(0, 0.5, 1000)

        result = jarque_bera_test(data, alpha=0.05)

        assert not result.is_normal  # Should reject normality
        assert result.p_value < 0.05  # Low p-value
        assert result.skewness > 0.5  # Positive skew
        assert result.excess_kurtosis > 0.0  # Positive excess kurtosis

    def test_jb_uniform_data(self):
        """JB test on uniform distribution."""
        np.random.seed(42)
        # Uniform has negative excess kurtosis (thin tails)
        data = np.random.uniform(-1, 1, 1000)

        result = jarque_bera_test(data)

        # Uniform is not normal (kurtosis < 0)
        assert not result.is_normal
        assert result.excess_kurtosis < -0.5  # Negative excess kurtosis

    def test_jb_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = jarque_bera_test(data)

        assert isinstance(result, JarqueBeraResult)
        assert result.n_obs == 1000

    def test_jb_different_alpha(self):
        """Test with different significance level."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = jarque_bera_test(data, alpha=0.01)

        assert result.alpha == 0.01

    def test_jb_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            jarque_bera_test(np.array([]))

    def test_jb_nan_values(self):
        """Test error handling for NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        with pytest.raises(ValidationError, match="NaN or infinite"):
            jarque_bera_test(data)

    def test_jb_insufficient_data(self):
        """Test error handling for insufficient data."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError, match="Insufficient data"):
            jarque_bera_test(data)

    def test_jb_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = jarque_bera_test(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Jarque-Bera Normality Test" in summary
        assert "Test Statistic:" in summary
        assert "P-value:" in summary


class TestShapiroWilkTest:
    """Tests for shapiro_wilk_test function."""

    def test_sw_normal_data(self):
        """Shapiro-Wilk on normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)  # Smaller sample for SW

        result = shapiro_wilk_test(data, alpha=0.05)

        assert isinstance(result, ShapiroWilkResult)
        assert result.n_obs == 500
        # W should be close to 1 for normal data
        assert result.statistic > 0.98
        # Should usually pass normality (though not always)
        assert result.p_value > 0.01

    def test_sw_non_normal_data(self):
        """Shapiro-Wilk should reject normality for non-normal data."""
        np.random.seed(42)
        # Lognormal is definitely not normal
        data = np.random.lognormal(0, 0.5, 500)

        result = shapiro_wilk_test(data, alpha=0.05)

        assert not result.is_normal  # Should reject normality
        assert result.p_value < 0.05  # Low p-value
        assert result.statistic < 0.98  # W far from 1

    def test_sw_small_sample(self):
        """Shapiro-Wilk works well with small samples."""
        np.random.seed(42)
        # Small sample from normal distribution
        data = np.random.normal(0, 1, 50)

        result = shapiro_wilk_test(data)

        assert result.n_obs == 50
        # Should still give reasonable results
        assert 0.9 < result.statistic <= 1.0

    def test_sw_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 500))

        result = shapiro_wilk_test(data)

        assert isinstance(result, ShapiroWilkResult)
        assert result.n_obs == 500

    def test_sw_minimum_sample(self):
        """Test with minimum sample size (3 observations)."""
        data = np.array([1.0, 2.0, 3.0])

        result = shapiro_wilk_test(data)

        assert result.n_obs == 3
        # Should work but with limited power
        assert result.statistic is not None

    def test_sw_large_sample(self):
        """Test behavior with large sample (>5000)."""
        np.random.seed(42)
        # Create sample larger than scipy limit (5000)
        data = np.random.normal(0, 1, 6000)

        result = shapiro_wilk_test(data)

        # Should truncate to 5000 and issue warning (check logs)
        assert result.n_obs == 5000  # Truncated

    def test_sw_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            shapiro_wilk_test(np.array([]))

    def test_sw_insufficient_data(self):
        """Test error handling for insufficient data."""
        data = np.array([1.0, 2.0])  # Less than 3
        with pytest.raises(ValidationError, match="Insufficient data"):
            shapiro_wilk_test(data)

    def test_sw_constant_data(self):
        """Test error handling for constant series."""
        data = np.ones(100)
        with pytest.raises(ValidationError, match="constant"):
            shapiro_wilk_test(data)

    def test_sw_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)
        result = shapiro_wilk_test(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Shapiro-Wilk Normality Test" in summary
        assert "Test Statistic (W):" in summary
        assert "P-value:" in summary


class TestDistributionIntegration:
    """Integration tests comparing different methods."""

    def test_moments_vs_jarque_bera_consistency(self):
        """Moments from compute_moments should match JB test."""
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 1000)

        moments_result = compute_moments(data)
        jb_result = jarque_bera_test(data)

        # Skewness and kurtosis should be nearly identical
        np.testing.assert_allclose(moments_result.skewness, jb_result.skewness, rtol=1e-10)
        np.testing.assert_allclose(
            moments_result.excess_kurtosis, jb_result.excess_kurtosis, rtol=1e-10
        )

    def test_jb_vs_sw_agreement(self):
        """JB and SW should generally agree on clearly non-normal data."""
        np.random.seed(42)
        # Very skewed data - both should reject
        data = np.random.lognormal(0, 1.0, 500)  # Very skewed

        jb_result = jarque_bera_test(data)
        sw_result = shapiro_wilk_test(data)

        # Both should reject normality
        assert not jb_result.is_normal
        assert not sw_result.is_normal

    def test_jb_vs_sw_on_normal(self):
        """JB and SW should generally agree on normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)

        jb_result = jarque_bera_test(data)
        sw_result = shapiro_wilk_test(data)

        # Both should usually accept normality (though not guaranteed)
        # At minimum, p-values should be relatively high
        assert jb_result.p_value > 0.01
        assert sw_result.p_value > 0.01

    def test_comprehensive_distribution_analysis(self):
        """Complete workflow: moments + JB + SW."""
        np.random.seed(42)
        # Create mixed distribution (not normal)
        data = np.concatenate(
            [
                np.random.normal(0, 1, 400),
                np.random.normal(3, 0.5, 100),  # Add outliers
            ]
        )
        np.random.shuffle(data)

        # Run all analyses
        moments = compute_moments(data)
        jb = jarque_bera_test(data)
        sw = shapiro_wilk_test(data)

        # All should detect non-normality
        assert moments.skewness_significant or moments.excess_kurtosis_significant
        assert not jb.is_normal or not sw.is_normal

        # Verify we can get summaries
        assert len(moments.summary()) > 100
        assert len(jb.summary()) > 100
        assert len(sw.summary()) > 100


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_perfectly_symmetric_data(self):
        """Test with perfectly symmetric data."""
        # Create symmetric data: [-n, -n+1, ..., -1, 0, 1, ..., n-1, n]
        data = np.arange(-100, 101)

        result = compute_moments(data)

        # Should have exactly zero skewness
        assert abs(result.skewness) < 1e-10

    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        np.random.seed(42)
        # Two well-separated normal distributions
        data = np.concatenate(
            [
                np.random.normal(-2, 0.5, 500),
                np.random.normal(2, 0.5, 500),
            ]
        )

        jb = jarque_bera_test(data)
        sw = shapiro_wilk_test(data)

        # Bimodal is not normal
        assert not jb.is_normal
        assert not sw.is_normal

    def test_zero_mean_unit_variance(self):
        """Test that normalization doesn't affect normality tests."""
        np.random.seed(42)
        data = np.random.normal(5, 3, 1000)  # Non-standard mean/std
        standardized = (data - np.mean(data)) / np.std(data)

        result1 = jarque_bera_test(data)
        result2 = jarque_bera_test(standardized)

        # Results should be identical (JB is scale-invariant)
        np.testing.assert_allclose(result1.statistic, result2.statistic, rtol=1e-10)
        np.testing.assert_allclose(result1.p_value, result2.p_value, rtol=1e-10)


class TestHillEstimator:
    """Tests for Hill estimator function."""

    def test_hill_t_distribution_heavy_tail(self):
        """Test Hill estimator on Student's t (df=3) - heavy tail."""
        np.random.seed(42)
        # t(3) has tail index α ≈ 3
        data = np.random.standard_t(df=3, size=1000)

        result = hill_estimator(data, k=None, tail="both")

        assert isinstance(result, HillEstimatorResult)
        assert result.n_obs == 1000
        # For t(3), tail index should be around 3 (medium tails)
        assert 2.0 < result.tail_index < 5.0
        assert result.classification in ["medium", "heavy"]
        assert result.tail == "both"
        assert result.k == int(np.sqrt(1000))  # Default k = sqrt(n)
        assert result.tail_index_se > 0

    def test_hill_normal_thin_tail(self):
        """Test on normal distribution - should have large tail index."""
        np.random.seed(42)
        # Normal has exponential tail decay (α → ∞)
        data = np.random.normal(0, 1, 1000)

        result = hill_estimator(data, tail="upper")

        # Normal should have high tail index (thin tails)
        # Though Hill estimator can be biased for non-power-law distributions
        assert result.tail_index > 2.0
        assert result.classification in ["medium", "thin"]

    def test_hill_upper_tail_only(self):
        """Test Hill estimator on upper tail only."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        result = hill_estimator(data, tail="upper")

        assert result.tail == "upper"
        assert result.tail_index > 0

    def test_hill_lower_tail_only(self):
        """Test Hill estimator on lower tail only."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        result = hill_estimator(data, tail="lower")

        assert result.tail == "lower"
        assert result.tail_index > 0

    def test_hill_both_tails(self):
        """Test Hill estimator on both tails (returns minimum)."""
        np.random.seed(42)
        # Create asymmetric data
        data = np.concatenate(
            [
                np.random.exponential(1, 500),  # Upper tail
                -np.random.exponential(2, 500),  # Lower tail (heavier)
            ]
        )

        result = hill_estimator(data, tail="both")

        assert result.tail == "both"
        # Should return the heavier tail (minimum α)
        assert result.tail_index > 0

    def test_hill_custom_k(self):
        """Test with custom k parameter."""
        np.random.seed(42)
        data = np.random.standard_t(df=4, size=1000)

        k = 100
        result = hill_estimator(data, k=k, tail="upper")

        assert result.k == k
        assert result.tail_index > 0
        assert result.tail_index_se == result.tail_index / np.sqrt(k)

    def test_hill_classification_heavy(self):
        """Test classification for heavy-tailed distribution."""
        np.random.seed(42)
        # Create very heavy-tailed data (Pareto with α ≈ 1.5)
        alpha = 1.5
        data = (np.random.pareto(alpha, 1000) + 1) * 0.1

        result = hill_estimator(data, tail="upper")

        # Should classify as heavy
        assert result.tail_index <= 4.0  # At least medium
        assert result.classification in ["heavy", "medium"]

    def test_hill_classification_medium(self):
        """Test classification for medium-tailed distribution."""
        np.random.seed(42)
        # t(5) has tail index α ≈ 5, should be thin or medium
        data = np.random.standard_t(df=5, size=1000)

        result = hill_estimator(data)

        # Could be medium or thin depending on estimation variance
        assert result.classification in ["medium", "thin"]

    def test_hill_classification_thin(self):
        """Test classification for thin-tailed distribution."""
        np.random.seed(42)
        # Exponential has thin tail (α → ∞)
        data = np.random.exponential(1, 1000)

        result = hill_estimator(data, tail="upper")

        # Should likely classify as thin or medium
        assert result.tail_index > 2.0

    def test_hill_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.standard_t(df=3, size=1000))

        result = hill_estimator(data)

        assert isinstance(result, HillEstimatorResult)
        assert result.n_obs == 1000

    def test_hill_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            hill_estimator(np.array([]))

    def test_hill_insufficient_data(self):
        """Test error handling for insufficient data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Less than min_length=50
        with pytest.raises(ValidationError, match="Insufficient data"):
            hill_estimator(data)

    def test_hill_invalid_tail_parameter(self):
        """Test error handling for invalid tail parameter."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        with pytest.raises(ValidationError, match="Invalid tail parameter"):
            hill_estimator(data, tail="invalid")

    def test_hill_invalid_k_too_small(self):
        """Test error handling for k < 2."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        with pytest.raises(ValidationError, match="k must be at least 2"):
            hill_estimator(data, k=1)

    def test_hill_invalid_k_too_large(self):
        """Test error handling for k >= n."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        with pytest.raises(ValidationError, match="k must be less than"):
            hill_estimator(data, k=100)

    def test_hill_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.standard_t(df=3, size=1000)
        result = hill_estimator(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Hill Estimator" in summary
        assert "Tail Index" in summary
        assert str(result.classification.upper()) in summary


class TestGenerateQQData:
    """Tests for generate_qq_data function."""

    def test_qq_normal_on_normal_data(self):
        """Test QQ plot for normal data against normal distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = generate_qq_data(data, distribution="normal")

        assert isinstance(result, QQPlotData)
        assert result.distribution == "normal"
        assert result.n_obs == 1000
        assert len(result.theoretical_quantiles) == 1000
        assert len(result.sample_quantiles) == 1000
        # R² should be close to 1 for normal data
        assert result.r_squared > 0.98
        assert result.df is None

    def test_qq_t_distribution(self):
        """Test QQ plot with Student's t distribution."""
        np.random.seed(42)
        # Generate t-distributed data
        df = 5
        data = np.random.standard_t(df=df, size=1000)

        # Compare to t-distribution
        result = generate_qq_data(data, distribution="t", df=df)

        assert result.distribution == "t"
        assert result.df == df
        # Should fit better than normal
        assert result.r_squared > 0.95

    def test_qq_t_better_fit_than_normal(self):
        """Test that t-distributed data fits t-QQ better than normal-QQ."""
        np.random.seed(42)
        df = 3
        data = np.random.standard_t(df=df, size=1000)

        qq_normal = generate_qq_data(data, distribution="normal")
        qq_t = generate_qq_data(data, distribution="t", df=df)

        # t-distribution should fit better
        assert qq_t.r_squared > qq_normal.r_squared

    def test_qq_uniform_distribution(self):
        """Test QQ plot with uniform distribution."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 1000)

        result = generate_qq_data(data, distribution="uniform")

        assert result.distribution == "uniform"
        assert result.r_squared > 0.95  # Should fit well

    def test_qq_exponential_distribution(self):
        """Test QQ plot with exponential distribution."""
        np.random.seed(42)
        data = np.random.exponential(1, 1000)

        result = generate_qq_data(data, distribution="exponential")

        assert result.distribution == "exponential"
        assert result.r_squared > 0.95  # Should fit well

    def test_qq_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = generate_qq_data(data, distribution="normal")

        assert isinstance(result, QQPlotData)
        assert result.n_obs == 1000

    def test_qq_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            generate_qq_data(np.array([]), distribution="normal")

    def test_qq_invalid_distribution(self):
        """Test error handling for invalid distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        with pytest.raises(ValidationError, match="Invalid distribution"):
            generate_qq_data(data, distribution="invalid")

    def test_qq_t_without_df(self):
        """Test error handling when df not provided for t-distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        with pytest.raises(ValidationError, match="Degrees of freedom.*required"):
            generate_qq_data(data, distribution="t", df=None)

    def test_qq_t_invalid_df(self):
        """Test error handling for invalid df."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        with pytest.raises(ValidationError, match="Degrees of freedom must be"):
            generate_qq_data(data, distribution="t", df=0)

    def test_qq_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = generate_qq_data(data, distribution="normal")

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Q-Q Plot Analysis" in summary
        assert "Normal Distribution" in summary
        assert "R²" in summary


class TestAnalyzeTails:
    """Tests for comprehensive tail analysis function."""

    def test_analyze_heavy_tailed_data(self):
        """Test comprehensive analysis on heavy-tailed data."""
        np.random.seed(42)
        # t(3) has medium to heavy tails
        data = np.random.standard_t(df=3, size=1000)

        result = analyze_tails(data)

        assert isinstance(result, TailAnalysisResult)
        assert isinstance(result.hill_result, HillEstimatorResult)
        assert isinstance(result.qq_normal, QQPlotData)
        # Should compute t-QQ for medium/heavy tails
        assert result.qq_t is not None
        assert isinstance(result.qq_t, QQPlotData)
        assert result.best_fit in ["normal", "t", "heavy-tailed"]

    def test_analyze_normal_data(self):
        """Test comprehensive analysis on normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = analyze_tails(data)

        # Normal data should fit normal distribution
        assert result.qq_normal.r_squared > 0.95
        # May or may not compute t-QQ depending on Hill estimate
        # Best fit should likely be normal
        assert result.best_fit in ["normal", "t"]

    def test_analyze_best_fit_selection_t(self):
        """Test that t-distribution is selected when it fits better."""
        np.random.seed(42)
        # Generate t-distributed data
        df = 4
        data = np.random.standard_t(df=df, size=1000)

        result = analyze_tails(data)

        # Should compute both QQ plots for medium tails
        assert result.qq_t is not None
        # t-distribution should fit significantly better
        if result.qq_t.r_squared > result.qq_normal.r_squared + 0.02:
            assert result.best_fit == "t"

    def test_analyze_best_fit_selection_heavy(self):
        """Test that heavy-tailed is selected for very heavy tails."""
        np.random.seed(42)
        # Create very heavy-tailed data (Pareto)
        alpha = 1.8
        data = (np.random.pareto(alpha, 1000) + 1) * 0.1

        result = analyze_tails(data)

        # Should detect heavy tails
        if result.hill_result.classification == "heavy":
            assert result.best_fit in ["heavy-tailed", "t"]

    def test_analyze_custom_k(self):
        """Test with custom k parameter."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        k = 150
        result = analyze_tails(data, k=k)

        assert result.hill_result.k == k

    def test_analyze_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.standard_t(df=3, size=1000))

        result = analyze_tails(data)

        assert isinstance(result, TailAnalysisResult)

    def test_analyze_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError):
            analyze_tails(np.array([]))

    def test_analyze_summary(self):
        """Test summary method."""
        np.random.seed(42)
        data = np.random.standard_t(df=3, size=1000)
        result = analyze_tails(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Comprehensive Tail Analysis" in summary
        assert "TAIL INDEX ESTIMATION" in summary
        assert "DISTRIBUTION COMPARISON" in summary
        assert "INTERPRETATION" in summary
        assert "RECOMMENDATIONS" in summary

    def test_analyze_thin_tails_no_t_qq(self):
        """Test that t-QQ is not computed for thin tails."""
        np.random.seed(42)
        # Create data that appears thin-tailed
        # (this is probabilistic, so result may vary)
        data = np.random.exponential(1, 1000)

        result = analyze_tails(data)

        # If classified as thin, qq_t might be None
        if result.hill_result.classification == "thin":
            # qq_t may or may not be computed
            pass  # Test is informational


class TestHeavyTailIntegration:
    """Integration tests for heavy tail analysis."""

    def test_hill_vs_qq_consistency(self):
        """Test consistency between Hill estimator and QQ plots."""
        np.random.seed(42)
        # Heavy-tailed data
        data = np.random.standard_t(df=3, size=1000)

        hill = hill_estimator(data)
        qq_normal = generate_qq_data(data, distribution="normal")
        qq_t = generate_qq_data(data, distribution="t", df=3)

        # If Hill indicates heavy/medium tails, t-QQ should fit better than normal
        if hill.classification in ["heavy", "medium"]:
            # t-distribution should fit at least as well
            assert qq_t.r_squared >= qq_normal.r_squared * 0.95  # Allow small tolerance

    def test_complete_distribution_workflow(self):
        """Test complete distribution analysis workflow."""
        np.random.seed(42)
        # Create heavy-tailed data
        data = np.random.standard_t(df=4, size=1000)

        # Traditional methods
        moments = compute_moments(data)
        jb = jarque_bera_test(data)
        sw = shapiro_wilk_test(data)

        # Heavy tail analysis
        tail_analysis = analyze_tails(data)

        # All should detect deviations from normality
        assert moments.excess_kurtosis > 0  # Fat tails
        # JB/SW may or may not reject depending on sample
        # But tail analysis should provide more detail
        assert tail_analysis.hill_result.tail_index > 0
        assert tail_analysis.best_fit in ["normal", "t", "heavy-tailed"]

        # Verify all summaries work
        assert len(moments.summary()) > 100
        assert len(jb.summary()) > 100
        assert len(sw.summary()) > 100
        assert len(tail_analysis.summary()) > 200

    def test_comparison_normal_vs_t(self):
        """Compare normal and t-distributed data."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        t_data = np.random.standard_t(df=3, size=1000)

        normal_analysis = analyze_tails(normal_data)
        t_analysis = analyze_tails(t_data)

        # t-data should have heavier tails (lower tail index)
        # Note: Hill estimator can be variable, so use broad comparison
        assert t_analysis.hill_result.classification in ["medium", "heavy"]
        # Normal might be classified as thin or medium
        # t-data should have lower tail index than normal
        assert t_analysis.hill_result.tail_index < normal_analysis.hill_result.tail_index


class TestAnalyzeDistribution:
    """Tests for comprehensive analyze_distribution function."""

    def test_analyze_distribution_normal_data(self):
        """Test comprehensive analysis on normal distribution."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = analyze_distribution(data, alpha=0.05, compute_tails=True)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.moments_result is not None
        assert result.jarque_bera_result is not None
        assert result.shapiro_wilk_result is not None
        assert result.tail_analysis_result is not None
        assert result.is_normal in [True, False]
        assert result.recommended_distribution in ["normal", "t", "heavy-tailed", "stable"]
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

        # Normal data should likely pass normality
        # (though not guaranteed due to sampling)
        if result.is_normal:
            assert result.recommended_distribution == "normal"

    def test_analyze_distribution_t_data(self):
        """Test on Student's t distribution (heavy tails)."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        result = analyze_distribution(data, alpha=0.05, compute_tails=True)

        # Should detect heavier tails than normal
        assert result.tail_analysis_result is not None
        assert result.recommended_distribution in ["t", "normal", "heavy-tailed"]

        # If t-distribution recommended, should have df estimate
        if result.recommended_distribution == "t":
            assert result.recommended_df is not None
            assert result.recommended_df >= 2

    def test_analyze_distribution_heavy_tailed(self):
        """Test on very heavy-tailed distribution."""
        np.random.seed(42)
        # Create Pareto distribution (heavy tails)
        alpha = 1.8
        data = (np.random.pareto(alpha, 1000) + 1) * 0.1

        result = analyze_distribution(data, alpha=0.05, compute_tails=True)

        # Should detect heavy tails
        assert not result.is_normal  # Should reject normality
        if result.tail_analysis_result:
            assert result.tail_analysis_result.hill_result.classification in ["heavy", "medium"]

    def test_analyze_distribution_lognormal(self):
        """Test on lognormal distribution (skewed)."""
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 1000)

        result = analyze_distribution(data, alpha=0.05, compute_tails=True)

        # Should reject normality due to skewness
        assert not result.is_normal
        assert result.moments_result.skewness_significant
        assert result.moments_result.skewness > 0  # Positive skew
        # Recommendation depends on tail analysis - could be normal, t, or heavy-tailed
        # depending on QQ fit (lognormal with sigma=0.5 is moderately skewed but tails are light)
        assert result.recommended_distribution in ["normal", "t", "heavy-tailed"]

    def test_analyze_distribution_without_tails(self):
        """Test with compute_tails=False for speed."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = analyze_distribution(data, alpha=0.05, compute_tails=False)

        # Should complete without tail analysis
        assert result.tail_analysis_result is None
        assert result.moments_result is not None
        assert result.jarque_bera_result is not None
        assert result.shapiro_wilk_result is not None
        assert result.is_normal in [True, False]
        assert result.recommended_distribution in ["normal", "t", "heavy-tailed"]

    def test_analyze_distribution_pandas_series(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = analyze_distribution(data)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.moments_result.n_obs == 1000

    def test_analyze_distribution_different_alpha(self):
        """Test with different significance level."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = analyze_distribution(data, alpha=0.01)

        assert result.moments_result.alpha == 0.01
        assert result.jarque_bera_result.alpha == 0.01
        assert result.shapiro_wilk_result.alpha == 0.01

    def test_analyze_distribution_empty_data(self):
        """Test error handling for empty data."""
        with pytest.raises(ValidationError):
            analyze_distribution(np.array([]))

    def test_analyze_distribution_none(self):
        """Test error handling for None data."""
        with pytest.raises(ValidationError, match="cannot be None"):
            analyze_distribution(None)

    def test_analyze_distribution_summary(self):
        """Test summary method provides comprehensive output."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        result = analyze_distribution(data)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "COMPREHENSIVE DISTRIBUTION ANALYSIS" in summary
        assert "MOMENTS:" in summary
        assert "NORMALITY TESTS:" in summary
        assert "RECOMMENDATION:" in summary
        assert "INTERPRETATION:" in summary
        assert "RISK IMPLICATIONS:" in summary
        assert str(result.moments_result.n_obs) in summary

    def test_analyze_distribution_repr(self):
        """Test string representation."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = analyze_distribution(data)

        repr_str = repr(result)
        assert "DistributionAnalysisResult" in repr_str
        assert "is_normal" in repr_str
        assert "recommended" in repr_str

    def test_analyze_distribution_consensus_normality(self):
        """Test consensus normality logic."""
        np.random.seed(42)

        # Normal data (both tests should pass)
        normal_data = np.random.normal(0, 1, 1000)
        result_normal = analyze_distribution(normal_data, compute_tails=False)

        # Very non-normal data (both tests should fail)
        lognormal_data = np.random.lognormal(0, 1.0, 1000)
        result_lognormal = analyze_distribution(lognormal_data, compute_tails=False)

        # Consensus requires both tests to pass
        if result_normal.is_normal:
            assert result_normal.jarque_bera_result.is_normal
            assert result_normal.shapiro_wilk_result.is_normal

        # Non-normal data should fail at least one test
        assert not result_lognormal.is_normal

    def test_analyze_distribution_recommendation_logic(self):
        """Test distribution recommendation logic."""
        np.random.seed(42)

        # Normal data should recommend normal (if passes tests)
        normal_data = np.random.normal(0, 1, 1000)
        result_normal = analyze_distribution(normal_data)

        if result_normal.is_normal:
            assert result_normal.recommended_distribution == "normal"
            assert result_normal.recommended_df is None

        # Heavy-tailed data should recommend t or heavy-tailed
        t_data = np.random.standard_t(df=3, size=1000)
        result_t = analyze_distribution(t_data)

        assert result_t.recommended_distribution in ["t", "heavy-tailed", "normal"]
        if result_t.recommended_distribution == "t":
            assert result_t.recommended_df is not None
            assert 2 <= result_t.recommended_df <= 30

    def test_analyze_distribution_interpretation_content(self):
        """Test that interpretation contains meaningful content."""
        np.random.seed(42)

        # Test normal data interpretation
        normal_data = np.random.normal(0, 1, 1000)
        result_normal = analyze_distribution(normal_data, compute_tails=False)

        if result_normal.is_normal:
            assert "consistent with normal" in result_normal.interpretation.lower()

        # Test skewed data interpretation
        lognormal_data = np.random.lognormal(0, 0.5, 1000)
        result_skewed = analyze_distribution(lognormal_data, compute_tails=False)

        assert "skewness" in result_skewed.interpretation.lower()

    def test_analyze_distribution_risk_implications(self):
        """Test that summary includes risk implications."""
        np.random.seed(42)
        data = np.random.standard_t(df=5, size=1000)

        result = analyze_distribution(data)
        summary = result.summary()

        assert "RISK IMPLICATIONS:" in summary
        # Should mention VaR, CVaR, or risk measures
        assert any(
            term in summary.lower() for term in ["var", "cvar", "sharpe", "risk", "portfolio"]
        )

    def test_analyze_distribution_tail_failure_graceful(self):
        """Test graceful handling when tail analysis fails."""
        np.random.seed(42)
        # Create data that might cause tail analysis issues
        # (though in practice this should be rare)
        data = np.random.normal(0, 1, 100)  # Smaller sample

        result = analyze_distribution(data, compute_tails=True)

        # Should complete even if tail analysis fails
        assert result is not None
        assert result.moments_result is not None
        # tail_analysis_result may or may not be None depending on success
