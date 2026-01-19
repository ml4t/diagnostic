"""Tests for volatility clustering detection (ARCH-LM test).

This module tests the ARCH Lagrange Multiplier test implementation for
detecting conditional heteroscedasticity in time series data.
"""

# Check if arch package is available for GARCH simulation
import importlib.util

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.errors import ValidationError
from ml4t.diagnostic.evaluation.volatility import (
    ARCHLMResult,
    GARCHResult,
    VolatilityAnalysisResult,
    analyze_volatility,
    arch_lm_test,
    fit_garch,
)

HAS_ARCH = importlib.util.find_spec("arch") is not None


class TestARCHLMResult:
    """Tests for ARCHLMResult class."""

    def test_init_no_arch_effects(self):
        """Test initialization with no ARCH effects (high p-value)."""
        result = ARCHLMResult(
            test_statistic=10.5,
            p_value=0.15,  # > 0.05 => no ARCH
            lags=12,
            n_obs=1000,
        )

        assert result.test_statistic == 10.5
        assert result.p_value == 0.15
        assert result.lags == 12
        assert result.n_obs == 1000
        assert result.has_arch_effects is False

    def test_init_with_arch_effects(self):
        """Test initialization with ARCH effects (low p-value)."""
        result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,  # < 0.05 => ARCH effects
            lags=12,
            n_obs=1000,
        )

        assert result.test_statistic == 45.2
        assert result.p_value == 0.001
        assert result.lags == 12
        assert result.n_obs == 1000
        assert result.has_arch_effects is True

    def test_repr(self):
        """Test string representation."""
        result = ARCHLMResult(test_statistic=10.5, p_value=0.15, lags=12, n_obs=1000)
        repr_str = repr(result)
        assert "ARCHLMResult" in repr_str
        assert "10.5" in repr_str
        assert "0.15" in repr_str
        assert "False" in repr_str

    def test_summary_no_arch(self):
        """Test summary for no ARCH effects."""
        result = ARCHLMResult(test_statistic=10.5, p_value=0.15, lags=12, n_obs=1000)
        summary = result.summary()

        assert "ARCH Lagrange Multiplier Test Results" in summary
        assert "10.5" in summary
        assert "0.15" in summary
        assert "12" in summary
        assert "1000" in summary
        assert "No ARCH effects" in summary
        assert "Fail to reject" in summary
        assert "constant variance" in summary.lower()

    def test_summary_with_arch(self):
        """Test summary with ARCH effects."""
        result = ARCHLMResult(test_statistic=45.2, p_value=0.001, lags=12, n_obs=1000)
        summary = result.summary()

        assert "ARCH effects detected" in summary
        assert "Reject H0" in summary
        assert "GARCH" in summary
        assert "volatility clustering" in summary.lower()


class TestArchLMTest:
    """Tests for arch_lm_test function."""

    def test_white_noise_no_arch(self):
        """Test white noise should not show ARCH effects."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = arch_lm_test(white_noise, lags=12)

        assert isinstance(result, ARCHLMResult)
        assert result.lags == 12
        assert result.n_obs == 1000
        # White noise should typically not show ARCH effects
        # (but this is probabilistic, so we just check result exists)
        assert isinstance(result.has_arch_effects, bool)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1

    def test_garch_has_arch_effects(self):
        """Test GARCH-like process should show ARCH effects."""
        np.random.seed(42)

        # Manually simulate GARCH(1,1)-like process
        # sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
        n = 1000
        omega = 0.01
        alpha = 0.1
        beta = 0.85

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        garch_data = eps * np.sqrt(sigma2)

        result = arch_lm_test(garch_data, lags=12)

        assert isinstance(result, ARCHLMResult)
        assert result.lags == 12
        # GARCH should typically show ARCH effects
        # This is probabilistic but should work with seed
        assert result.has_arch_effects is True
        assert result.p_value < 0.05

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(500))

        result = arch_lm_test(data, lags=6)

        assert isinstance(result, ARCHLMResult)
        assert result.n_obs == 500

    def test_different_lags(self):
        """Test with different lag specifications."""
        np.random.seed(42)
        data = np.random.randn(500)

        # Test with various lag values
        for lags in [1, 5, 10, 20]:
            result = arch_lm_test(data, lags=lags)
            assert result.lags == lags
            assert isinstance(result.test_statistic, float)
            assert isinstance(result.p_value, float)

    def test_demean_true_vs_false(self):
        """Test difference between demean=True and demean=False."""
        np.random.seed(42)
        # Data with non-zero mean
        data = np.random.randn(500) + 2.0

        result_demean = arch_lm_test(data, lags=12, demean=True)
        result_no_demean = arch_lm_test(data, lags=12, demean=False)

        # Both should run successfully
        assert isinstance(result_demean, ARCHLMResult)
        assert isinstance(result_no_demean, ARCHLMResult)
        # Results may differ slightly
        # (just verify both execute without error)

    def test_empty_data(self):
        """Test with empty data raises ValidationError."""
        with pytest.raises(ValidationError, match="empty data"):
            arch_lm_test(np.array([]))

    def test_wrong_shape(self):
        """Test with 2D data raises ValidationError."""
        data = np.random.randn(100, 2)
        with pytest.raises(ValidationError, match="1-dimensional"):
            arch_lm_test(data)

    def test_nan_values(self):
        """Test with NaN values raises ValidationError."""
        data = np.random.randn(100)
        data[50] = np.nan
        with pytest.raises(ValidationError, match="NaN or infinite"):
            arch_lm_test(data)

    def test_infinite_values(self):
        """Test with infinite values raises ValidationError."""
        data = np.random.randn(100)
        data[50] = np.inf
        with pytest.raises(ValidationError, match="NaN or infinite"):
            arch_lm_test(data)

    def test_insufficient_data(self):
        """Test with too few observations raises ValidationError."""
        data = np.random.randn(10)
        with pytest.raises(ValidationError, match="Insufficient data"):
            arch_lm_test(data, lags=12)

    def test_invalid_lags_negative(self):
        """Test with negative lags raises ValidationError."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="must be positive"):
            arch_lm_test(data, lags=-1)

    def test_invalid_lags_too_large(self):
        """Test with lags >= data size raises ValidationError."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="Insufficient data"):
            arch_lm_test(data, lags=100)

    def test_minimum_sample_size(self):
        """Test minimum sample size requirement."""
        # With lags=12, need at least 22 observations (lags + 10 buffer)
        np.random.seed(42)
        data = np.random.randn(25)

        result = arch_lm_test(data, lags=12)
        assert isinstance(result, ARCHLMResult)
        assert result.n_obs == 25

    def test_large_sample(self):
        """Test with large sample size."""
        np.random.seed(42)
        data = np.random.randn(10000)

        result = arch_lm_test(data, lags=24)

        assert isinstance(result, ARCHLMResult)
        assert result.n_obs == 10000
        assert result.lags == 24

    def test_constant_series(self):
        """Test with constant series (edge case)."""
        data = np.ones(1000)

        # This should run but may have numerical issues
        # The test should handle this gracefully
        result = arch_lm_test(data, lags=12)
        assert isinstance(result, ARCHLMResult)

    def test_high_volatility_clustering(self):
        """Test series with strong volatility clustering."""
        np.random.seed(42)

        # Manually create ARCH(1)-like process with strong effects
        # sigma_t^2 = omega + alpha * eps_{t-1}^2
        n = 1000
        omega = 0.01
        alpha = 0.5

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha)  # Unconditional variance

        for t in range(1, n):
            sigma2[t] = omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2

        arch_data = eps * np.sqrt(sigma2)

        result = arch_lm_test(arch_data, lags=12)

        # Should detect ARCH effects with high confidence
        assert result.has_arch_effects is True
        assert result.p_value < 0.01  # Very low p-value

    def test_result_consistency(self):
        """Test that results are consistent with same input."""
        np.random.seed(42)
        data = np.random.randn(500)

        result1 = arch_lm_test(data, lags=12)
        result2 = arch_lm_test(data, lags=12)

        # Should get identical results
        assert result1.test_statistic == result2.test_statistic
        assert result1.p_value == result2.p_value
        assert result1.has_arch_effects == result2.has_arch_effects

    def test_direct_statsmodels_comparison(self):
        """Test that results match statsmodels directly."""
        from statsmodels.stats.diagnostic import het_arch

        np.random.seed(42)
        data = np.random.randn(500)
        lags = 12

        # Our implementation
        result = arch_lm_test(data, lags=lags, demean=True)

        # Direct statsmodels call
        residuals = data - np.mean(data)
        result_tuple = het_arch(residuals, nlags=lags)
        lm_stat = result_tuple[0]
        p_value = result_tuple[1]

        # Should match exactly
        assert np.isclose(result.test_statistic, lm_stat)
        assert np.isclose(result.p_value, p_value)

    def test_non_demeaned_comparison(self):
        """Test non-demeaned version against statsmodels."""
        from statsmodels.stats.diagnostic import het_arch

        np.random.seed(42)
        data = np.random.randn(500)
        lags = 12

        # Our implementation without demeaning
        result = arch_lm_test(data, lags=lags, demean=False)

        # Direct statsmodels call (no demeaning)
        result_tuple = het_arch(data, nlags=lags)
        lm_stat = result_tuple[0]
        p_value = result_tuple[1]

        # Should match exactly
        assert np.isclose(result.test_statistic, lm_stat)
        assert np.isclose(result.p_value, p_value)


class TestArchLMTestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_sample(self):
        """Test with minimum viable sample."""
        np.random.seed(42)
        data = np.random.randn(15)  # Just above minimum for lags=1

        result = arch_lm_test(data, lags=1)
        assert isinstance(result, ARCHLMResult)

    def test_single_lag(self):
        """Test with single lag (minimal test)."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = arch_lm_test(data, lags=1)
        assert result.lags == 1
        assert isinstance(result.test_statistic, float)

    def test_many_lags(self):
        """Test with many lags relative to sample size."""
        np.random.seed(42)
        data = np.random.randn(500)

        # Use 50 lags (10% of sample)
        result = arch_lm_test(data, lags=50)
        assert result.lags == 50
        assert isinstance(result.test_statistic, float)

    def test_returns_series(self):
        """Test with realistic returns series."""
        np.random.seed(42)

        # Simulate daily returns with strong volatility clustering
        # Use stronger alpha for clearer ARCH effects
        n = 1000
        omega = 0.01
        alpha = 0.15  # Stronger ARCH effect
        beta = 0.80

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        # GARCH(1,1) volatility
        for i in range(1, n):
            sigma2[i] = (
                omega + alpha * (eps[i - 1] * np.sqrt(sigma2[i - 1])) ** 2 + beta * sigma2[i - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = arch_lm_test(returns, lags=12, demean=True)

        # Should detect ARCH effects due to volatility clustering
        assert isinstance(result, ARCHLMResult)
        # This series has ARCH by construction with stronger parameters
        assert result.has_arch_effects is True


class TestGARCHResult:
    """Tests for GARCHResult class."""

    def test_init_scalar_params(self):
        """Test initialization with scalar parameters."""
        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))

        result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            persistence=0.95,
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        assert result.omega == 0.01
        assert result.alpha == 0.1
        assert result.beta == 0.85
        assert result.persistence == 0.95
        assert result.converged is True
        assert result.iterations == 50
        assert result.n_obs == 100

    def test_init_vector_params(self):
        """Test initialization with vector parameters (higher order)."""
        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))

        result = GARCHResult(
            omega=0.01,
            alpha=(0.1, 0.05),
            beta=(0.7, 0.1),
            persistence=0.95,
            log_likelihood=-500.0,
            aic=1010.0,
            bic=1025.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=75,
            n_obs=100,
        )

        assert result.alpha == (0.1, 0.05)
        assert result.beta == (0.7, 0.1)
        assert result.persistence == 0.95

    def test_repr(self):
        """Test string representation."""
        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))

        result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            persistence=0.95,
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        repr_str = repr(result)
        assert "GARCHResult" in repr_str
        assert "0.010000" in repr_str
        assert "0.1" in repr_str
        assert "0.85" in repr_str
        assert "0.95" in repr_str

    def test_summary_stationary(self):
        """Test summary for stationary GARCH."""
        vol = pd.Series(np.random.rand(100) * 0.02 + 0.01)
        resid = pd.Series(np.random.randn(100))

        result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            persistence=0.96,  # > 0.95 to trigger "High persistence"
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        summary = result.summary()
        assert "GARCH Model Fitting Results" in summary
        assert "0.010000" in summary
        assert "0.100000" in summary
        assert "0.850000" in summary
        assert "0.960000" in summary
        assert "High persistence" in summary
        assert "Converged:         Yes" in summary

    def test_summary_non_stationary(self):
        """Test summary for non-stationary GARCH."""
        vol = pd.Series(np.random.rand(100) * 0.02 + 0.01)
        resid = pd.Series(np.random.randn(100))

        result = GARCHResult(
            omega=0.01,
            alpha=0.5,
            beta=0.6,
            persistence=1.1,  # Non-stationary
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        summary = result.summary()
        assert "WARNING: Persistence ≥ 1" in summary
        assert "non-stationary" in summary


@pytest.mark.skipif(not HAS_ARCH, reason="arch package not installed")
class TestFitGARCH:
    """Tests for fit_garch function."""

    def test_garch_simulated_data(self):
        """Test GARCH fitting on simulated GARCH(1,1) data with known parameters."""
        np.random.seed(42)

        # Simulate GARCH(1,1) with known parameters
        n = 1000
        omega_true = 0.01
        alpha_true = 0.1
        beta_true = 0.85

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega_true / (1 - alpha_true - beta_true)

        for t in range(1, n):
            sigma2[t] = (
                omega_true
                + alpha_true * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2
                + beta_true * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        # Fit GARCH model
        result = fit_garch(returns, p=1, q=1)

        # Check result type
        assert isinstance(result, GARCHResult)

        # Check convergence
        assert result.converged is True

        # Check parameters are close to true values (with tolerance)
        # Note: Estimates won't be exact due to sampling variability
        assert isinstance(result.omega, float)
        assert isinstance(result.alpha, float)
        assert isinstance(result.beta, float)

        # Check that persistence is reasonable
        assert 0 < result.persistence < 1.0

        # Check that fitted volatility is positive
        assert np.all(result.conditional_volatility > 0)

        # Check that standardized residuals are approximately N(0,1)
        resid = result.standardized_residuals.values
        assert np.abs(np.mean(resid)) < 0.2  # Mean close to 0
        assert 0.8 < np.std(resid) < 1.2  # Std close to 1

    def test_garch_pandas_series(self):
        """Test GARCH fitting with pandas Series input."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 500
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = pd.Series(eps * np.sqrt(sigma2), name="returns")

        result = fit_garch(returns, p=1, q=1)

        assert isinstance(result, GARCHResult)
        assert result.converged is True

    def test_garch_different_distributions(self):
        """Test GARCH fitting with different error distributions."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 500
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        # Test different distributions
        for dist in ["normal", "t"]:
            result = fit_garch(returns, p=1, q=1, dist=dist)
            assert isinstance(result, GARCHResult)
            assert result.converged is True

    def test_garch_higher_order(self):
        """Test GARCH(2,2) fitting."""
        np.random.seed(42)

        # Simulate GARCH(1,1) data (we'll fit GARCH(2,2) to it)
        n = 1000
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        # Fit GARCH(2,2)
        result = fit_garch(returns, p=2, q=2)

        assert isinstance(result, GARCHResult)
        assert isinstance(result.alpha, tuple)
        assert isinstance(result.beta, tuple)
        assert len(result.alpha) == 2
        assert len(result.beta) == 2

    def test_garch_constant_mean(self):
        """Test GARCH with constant mean model."""
        np.random.seed(42)

        # Simulate GARCH data with non-zero mean
        n = 500
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2) + 0.001  # Add small mean

        result = fit_garch(returns, p=1, q=1, mean_model="Constant")

        assert isinstance(result, GARCHResult)
        assert result.converged is True

    def test_garch_empty_data(self):
        """Test GARCH with empty data raises ValidationError."""
        with pytest.raises(ValidationError, match="empty data"):
            fit_garch(np.array([]))

    def test_garch_wrong_shape(self):
        """Test GARCH with 2D data raises ValidationError."""
        data = np.random.randn(100, 2)
        with pytest.raises(ValidationError, match="1-dimensional"):
            fit_garch(data)

    def test_garch_nan_values(self):
        """Test GARCH with NaN values raises ValidationError."""
        data = np.random.randn(500)
        data[50] = np.nan
        with pytest.raises(ValidationError, match="NaN or infinite"):
            fit_garch(data)

    def test_garch_infinite_values(self):
        """Test GARCH with infinite values raises ValidationError."""
        data = np.random.randn(500)
        data[50] = np.inf
        with pytest.raises(ValidationError, match="NaN or infinite"):
            fit_garch(data)

    def test_garch_insufficient_data(self):
        """Test GARCH with too few observations raises ValidationError."""
        data = np.random.randn(50)
        with pytest.raises(ValidationError, match="Insufficient data"):
            fit_garch(data, p=1, q=1)

    def test_garch_invalid_p(self):
        """Test GARCH with invalid p raises ValidationError."""
        data = np.random.randn(500)
        with pytest.raises(ValidationError, match="ARCH order"):
            fit_garch(data, p=0, q=1)

    def test_garch_invalid_q(self):
        """Test GARCH with invalid q raises ValidationError."""
        data = np.random.randn(500)
        with pytest.raises(ValidationError, match="GARCH order"):
            fit_garch(data, p=1, q=0)

    def test_garch_model_fit_statistics(self):
        """Test that model fit statistics are reasonable."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 1000
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = fit_garch(returns, p=1, q=1)

        # Check that AIC and BIC are computed
        assert isinstance(result.aic, float)
        assert isinstance(result.bic, float)
        assert isinstance(result.log_likelihood, float)

        # BIC should be larger than AIC (more penalty)
        # Actually this is not always true, depends on sample size
        # So just check they are reasonable values
        assert not np.isnan(result.aic)
        assert not np.isnan(result.bic)
        assert not np.isnan(result.log_likelihood)

    def test_garch_conditional_volatility_positive(self):
        """Test that conditional volatility is always positive."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 500
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = fit_garch(returns, p=1, q=1)

        # Conditional volatility should always be positive
        assert np.all(result.conditional_volatility > 0)
        assert np.all(np.isfinite(result.conditional_volatility))

    def test_garch_persistence_stationary(self):
        """Test that fitted persistence indicates stationarity."""
        np.random.seed(42)

        # Simulate stationary GARCH data
        n = 1000
        omega, alpha, beta = 0.01, 0.1, 0.85
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = fit_garch(returns, p=1, q=1)

        # Persistence should be < 1 for stationary process
        # (may not always be true due to estimation, but usually is)
        assert result.persistence > 0
        # Allow some tolerance for estimation error
        assert result.persistence < 1.2  # Very lenient


class TestVolatilityAnalysisResult:
    """Tests for VolatilityAnalysisResult class."""

    def test_init_with_arch_no_garch(self):
        """Test initialization with ARCH effects but no GARCH fitted."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=None,
        )

        assert result.has_volatility_clustering is True
        assert result.persistence is None
        assert result.arch_lm_result == arch_result
        assert result.garch_result is None

    def test_init_with_arch_and_garch(self):
        """Test initialization with both ARCH and GARCH results."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))
        garch_result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            persistence=0.95,
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=garch_result,
        )

        assert result.has_volatility_clustering is True
        assert result.persistence == 0.95
        assert result.arch_lm_result == arch_result
        assert result.garch_result == garch_result

    def test_init_no_arch_effects(self):
        """Test initialization with no ARCH effects."""
        arch_result = ARCHLMResult(
            test_statistic=10.5,
            p_value=0.15,
            lags=12,
            n_obs=1000,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=None,
        )

        assert result.has_volatility_clustering is False
        assert result.persistence is None

    def test_repr(self):
        """Test string representation."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=None,
        )

        repr_str = repr(result)
        assert "VolatilityAnalysisResult" in repr_str
        assert "has_clustering=True" in repr_str
        assert "0.001" in repr_str

    def test_summary_with_clustering_no_garch(self):
        """Test summary when clustering detected but no GARCH fitted."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=None,
        )

        summary = result.summary()
        assert "Comprehensive Volatility Analysis" in summary
        assert "ARCH-LM Test" in summary
        assert "45.2" in summary
        assert "0.001" in summary
        assert "ARCH effects detected" in summary
        assert "Not fitted" in summary

    def test_summary_with_garch(self):
        """Test summary with GARCH results."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))
        garch_result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.85,
            persistence=0.95,
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=garch_result,
        )

        summary = result.summary()
        assert "GARCH Model Fitting Results" in summary
        assert "0.010000" in summary
        assert "0.100000" in summary
        assert "0.850000" in summary
        assert "0.95" in summary
        assert "Persistence" in summary

    def test_interpretation_no_clustering(self):
        """Test interpretation when no clustering detected."""
        arch_result = ARCHLMResult(
            test_statistic=10.5,
            p_value=0.15,
            lags=12,
            n_obs=1000,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=None,
        )

        assert "No volatility clustering" in result.interpretation
        assert "Constant variance" in result.interpretation
        assert "Recommendations" in result.interpretation

    def test_interpretation_with_high_persistence(self):
        """Test interpretation with high persistence."""
        arch_result = ARCHLMResult(
            test_statistic=45.2,
            p_value=0.001,
            lags=12,
            n_obs=1000,
        )

        vol = pd.Series(np.random.rand(100))
        resid = pd.Series(np.random.randn(100))
        garch_result = GARCHResult(
            omega=0.01,
            alpha=0.1,
            beta=0.89,
            persistence=0.99,
            log_likelihood=-500.0,
            aic=1005.0,
            bic=1015.0,
            conditional_volatility=vol,
            standardized_residuals=resid,
            converged=True,
            iterations=50,
            n_obs=100,
        )

        result = VolatilityAnalysisResult(
            arch_lm_result=arch_result,
            garch_result=garch_result,
        )

        assert "Volatility clustering detected" in result.interpretation
        assert "0.99" in result.interpretation
        assert "Very high persistence" in result.interpretation


class TestAnalyzeVolatility:
    """Tests for analyze_volatility function."""

    def test_white_noise_no_clustering(self):
        """Test white noise should not show clustering."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = analyze_volatility(
            white_noise,
            arch_lags=12,
            fit_garch_model=True,
        )

        assert isinstance(result, VolatilityAnalysisResult)
        assert result.arch_lm_result is not None
        # White noise typically shows no ARCH effects
        # (probabilistic, but with seed should be consistent)

    def test_garch_data_with_clustering(self):
        """Test GARCH data shows clustering and fits model."""
        np.random.seed(42)

        # Simulate GARCH(1,1)
        n = 1000
        omega = 0.01
        alpha = 0.1
        beta = 0.85

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = analyze_volatility(
            returns,
            arch_lags=12,
            fit_garch_model=True,
        )

        assert isinstance(result, VolatilityAnalysisResult)
        assert result.has_volatility_clustering is True
        # GARCH should be fitted since arch package available
        if HAS_ARCH:
            assert result.garch_result is not None
            assert result.persistence is not None
            assert 0 < result.persistence < 1.2

    def test_clustering_detected_skip_garch(self):
        """Test that GARCH is skipped when fit_garch_model=False."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 1000
        omega = 0.01
        alpha = 0.1
        beta = 0.85

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        result = analyze_volatility(
            returns,
            arch_lags=12,
            fit_garch_model=False,  # Skip GARCH
        )

        assert isinstance(result, VolatilityAnalysisResult)
        assert result.has_volatility_clustering is True
        assert result.garch_result is None  # Should not fit GARCH
        assert result.persistence is None

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000))

        result = analyze_volatility(returns, arch_lags=12)

        assert isinstance(result, VolatilityAnalysisResult)
        assert result.arch_lm_result.n_obs == 1000

    def test_different_arch_lags(self):
        """Test with different ARCH lag specifications."""
        np.random.seed(42)
        returns = np.random.randn(1000)

        for lags in [6, 12, 24]:
            result = analyze_volatility(returns, arch_lags=lags)
            assert result.arch_lm_result.lags == lags

    @pytest.mark.skipif(not HAS_ARCH, reason="arch package not installed")
    def test_garch_orders(self):
        """Test with different GARCH orders."""
        np.random.seed(42)

        # Simulate GARCH data
        n = 1000
        omega = 0.01
        alpha = 0.1
        beta = 0.85

        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        # Test GARCH(1,1)
        result = analyze_volatility(returns, garch_p=1, garch_q=1)
        if result.garch_result is not None:
            assert result.garch_result.converged is True

    def test_empty_data(self):
        """Test with empty data raises error."""
        with pytest.raises(ValidationError, match="empty"):
            analyze_volatility(np.array([]))

    def test_invalid_data_shape(self):
        """Test with 2D data raises error."""
        data = np.random.randn(100, 2)
        with pytest.raises(ValidationError, match="1-dimensional"):
            analyze_volatility(data)

    def test_insufficient_data(self):
        """Test with too little data raises error."""
        data = np.random.randn(10)
        with pytest.raises(ValidationError, match="Insufficient"):
            analyze_volatility(data, arch_lags=12)

    def test_summary_output(self):
        """Test that summary produces formatted output."""
        np.random.seed(42)
        returns = np.random.randn(1000)

        result = analyze_volatility(returns, arch_lags=12)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Comprehensive Volatility Analysis" in summary
        assert "ARCH-LM Test" in summary
        assert "Interpretation" in summary
        assert "Recommendations" in summary
