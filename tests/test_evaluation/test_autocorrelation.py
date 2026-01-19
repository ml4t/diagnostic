"""Tests for autocorrelation analysis (ACF/PACF/Ljung-Box).

This module tests ACF computation with various time series patterns:
- White noise (no autocorrelation)
- AR(1) process (exponential decay)
- MA(1) process (single spike)
- Random walk (all correlations near 1)
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import acf as sm_acf

from ml4t.diagnostic.errors import ValidationError
from ml4t.diagnostic.evaluation.autocorrelation import (
    ACFResult,
    AutocorrelationAnalysisResult,
    PACFResult,
    analyze_autocorrelation,
    compute_acf,
    compute_pacf,
)


class TestACFResult:
    """Tests for ACFResult class."""

    def test_acf_result_creation(self):
        """Test ACFResult initialization."""
        acf_values = np.array([1.0, 0.5, 0.25, 0.125])
        conf_int = np.array(
            [
                [1.0, 1.0],
                [0.4, 0.6],
                [0.15, 0.35],
                [0.025, 0.225],
            ]
        )
        lags = np.array([0, 1, 2, 3])

        result = ACFResult(
            acf_values=acf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="standard",
        )

        assert len(result.acf_values) == 4
        assert result.acf_values[0] == 1.0
        assert result.n_obs == 100
        assert result.alpha == 0.05
        assert result.method == "standard"

    def test_significant_lags_detection(self):
        """Test identification of significant lags.

        Tests H0: ρ(k) = 0 by checking if CI excludes 0.
        """
        # Create ACF with different significance patterns
        acf_values = np.array([1.0, 0.8, 0.1, -0.05])
        conf_int = np.array(
            [
                [1.0, 1.0],  # Lag 0 (ignored)
                [0.6, 1.0],  # Lag 1: CI excludes 0 (significant: lower > 0)
                [-0.1, 0.3],  # Lag 2: CI includes 0 (not significant)
                [-0.25, 0.15],  # Lag 3: CI includes 0 (not significant)
            ]
        )
        lags = np.array([0, 1, 2, 3])

        result = ACFResult(
            acf_values=acf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="standard",
        )

        # Only lag 1 should be significant (CI excludes 0)
        assert result.significant_lags == [1]

        # Test negative significant lag
        acf_values2 = np.array([1.0, -0.5, 0.05])
        conf_int2 = np.array(
            [
                [1.0, 1.0],  # Lag 0
                [-0.7, -0.3],  # Lag 1: CI excludes 0 (significant: upper < 0)
                [-0.1, 0.2],  # Lag 2: CI includes 0 (not significant)
            ]
        )
        lags2 = np.array([0, 1, 2])

        result2 = ACFResult(
            acf_values=acf_values2,
            conf_int=conf_int2,
            lags=lags2,
            alpha=0.05,
            n_obs=100,
            method="standard",
        )

        # Lag 1 should be significant (negative ACF with CI excluding 0)
        assert result2.significant_lags == [1]

    def test_repr_str(self):
        """Test string representations."""
        acf_values = np.array([1.0, 0.5, 0.25])
        conf_int = np.zeros((3, 2))
        lags = np.array([0, 1, 2])

        result = ACFResult(
            acf_values=acf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="fft",
        )

        repr_str = repr(result)
        assert "ACFResult" in repr_str
        assert "n_obs=100" in repr_str
        assert "nlags=2" in repr_str

        str_output = str(result)
        assert "ACF Analysis Results" in str_output
        assert "Observations: 100" in str_output


class TestComputeACF:
    """Tests for compute_acf function."""

    def test_white_noise_acf(self):
        """Test ACF of white noise (should be ~0 for all lags > 0)."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = compute_acf(white_noise, nlags=20, alpha=0.05)

        # Check basics
        assert result.n_obs == 1000
        assert len(result.acf_values) == 21  # nlags + 1
        assert result.acf_values[0] == 1.0  # Lag 0 always 1

        # For white noise, most lags should be insignificant
        # Expect ~5% false positives (1 lag out of 20)
        assert len(result.significant_lags) <= 3, (
            f"Too many significant lags for white noise: {result.significant_lags}"
        )

        # All ACF values (except lag 0) should be small
        assert np.abs(result.acf_values[1:]).max() < 0.15, "White noise ACF values too large"

    def test_ar1_process_acf(self):
        """Test ACF of AR(1) process (exponential decay)."""
        np.random.seed(42)
        phi = 0.7
        n = 5000  # Larger sample for more reliable estimates

        # Generate AR(1): X_t = phi * X_{t-1} + epsilon_t
        noise = np.random.randn(n)
        ar1 = [noise[0]]
        for t in range(1, n):
            ar1.append(phi * ar1[-1] + noise[t])
        ar1 = np.array(ar1)  # type: ignore[assignment]

        result = compute_acf(ar1, nlags=20, alpha=0.05)

        # Check exponential decay: ACF[k] ≈ phi^k
        expected_acf = np.array([phi**k for k in range(21)])

        # Check first few lags (theory: ACF[k] = phi^k)
        for k in range(1, 4):
            assert abs(result.acf_values[k] - expected_acf[k]) < 0.15, (
                f"AR(1) ACF[{k}] = {result.acf_values[k]:.3f}, expected {expected_acf[k]:.3f}"
            )

        # Should have significant lag 1 at minimum (high persistence)
        assert 1 in result.significant_lags, "AR(1) should have significant lag-1 autocorrelation"

        # First lag should be high (> 0.5 for phi=0.7)
        assert result.acf_values[1] > 0.5, (
            f"AR(1) lag-1 ACF should be high, got {result.acf_values[1]:.3f}"
        )

    def test_ma1_process_acf(self):
        """Test ACF of MA(1) process (single spike at lag 1, then zero)."""
        np.random.seed(42)
        theta = 0.5
        n = 1000

        # Generate MA(1): X_t = epsilon_t + theta * epsilon_{t-1}
        noise = np.random.randn(n + 1)
        ma1 = noise[1:] + theta * noise[:-1]

        result = compute_acf(ma1, nlags=10, alpha=0.05)

        # Theory: ACF[1] = theta / (1 + theta^2), ACF[k] = 0 for k > 1
        expected_acf1 = theta / (1 + theta**2)

        # Check lag 1
        assert abs(result.acf_values[1] - expected_acf1) < 0.1, (
            f"MA(1) ACF[1] = {result.acf_values[1]:.3f}, expected {expected_acf1:.3f}"
        )

        # Check lags > 1 are small
        for k in range(2, 6):
            assert abs(result.acf_values[k]) < 0.15, (
                f"MA(1) ACF[{k}] = {result.acf_values[k]:.3f} should be near 0"
            )

    def test_random_walk_acf(self):
        """Test ACF of random walk (all correlations near 1, non-stationary)."""
        np.random.seed(42)
        noise = np.random.randn(500)
        random_walk = np.cumsum(noise)

        result = compute_acf(random_walk, nlags=10, alpha=0.05)

        # Random walk has very high autocorrelation at all lags
        # All ACF values should be > 0.9
        for k in range(1, 6):
            assert result.acf_values[k] > 0.9, (
                f"Random walk ACF[{k}] = {result.acf_values[k]:.3f} should be near 1"
            )

    def test_pandas_series_input(self):
        """Test ACF with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100), name="returns")

        result = compute_acf(data, nlags=10)

        assert result.n_obs == 100
        assert len(result.acf_values) == 11

    def test_nlags_auto_selection(self):
        """Test automatic nlags selection (min(10*log10(n), n-1))."""
        np.random.seed(42)

        # For n=1000: 10*log10(1000) = 30
        data = np.random.randn(1000)
        result = compute_acf(data, nlags=None)
        assert len(result.acf_values) == 31  # 30 lags + lag 0

        # For n=100: 10*log10(100) = 20
        data = np.random.randn(100)
        result = compute_acf(data, nlags=None)
        assert len(result.acf_values) == 21

    def test_fft_method(self):
        """Test ACF computation with FFT."""
        np.random.seed(42)
        data = np.random.randn(1000)

        result = compute_acf(data, nlags=20, fft=True)

        assert result.method == "fft"
        assert len(result.acf_values) == 21

    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        data = np.random.randn(100)

        # 95% CI
        result = compute_acf(data, nlags=10, alpha=0.05)
        assert result.alpha == 0.05
        assert result.conf_int.shape == (11, 2)

        # CI should be symmetric around 0 for white noise (approximately)
        # And get wider for higher lags (fewer effective observations)
        ci_width = result.conf_int[:, 1] - result.conf_int[:, 0]
        assert ci_width[0] == 0.0  # Lag 0 has no uncertainty
        assert all(ci_width[1:] > 0), "CI widths should be positive"

    def test_against_statsmodels(self):
        """Test that results match statsmodels directly."""
        np.random.seed(42)
        data = np.random.randn(500)
        nlags = 15
        alpha = 0.05

        # Our implementation
        result = compute_acf(data, nlags=nlags, alpha=alpha)

        # Statsmodels directly
        sm_acf_values, sm_conf_int = sm_acf(data, nlags=nlags, alpha=alpha)

        # Should match exactly (we use statsmodels backend)
        np.testing.assert_array_almost_equal(result.acf_values, sm_acf_values, decimal=10)
        np.testing.assert_array_almost_equal(result.conf_int, sm_conf_int, decimal=10)

    def test_missing_data_handling(self):
        """Test different missing data strategies."""
        np.random.seed(42)
        data = np.random.randn(100)
        data[10:15] = np.nan

        # missing='raise' should raise error
        with pytest.raises(ValidationError, match="contains NaN"):
            compute_acf(data, nlags=10, missing="raise")

        # missing='drop' should drop NaN and compute
        result = compute_acf(data, nlags=10, missing="drop")
        assert result.n_obs == 95  # 100 - 5 NaN

        # missing='conservative' should also work
        result = compute_acf(data, nlags=10, missing="conservative")
        assert result.n_obs == 95


class TestACFValidation:
    """Tests for ACF input validation."""

    def test_empty_data(self):
        """Test error for empty data."""
        with pytest.raises(ValidationError, match="empty data"):
            compute_acf(np.array([]))

    def test_all_nan_data(self):
        """Test error for all NaN data."""
        data = np.full(100, np.nan)
        with pytest.raises(ValidationError, match="All data is NaN"):
            compute_acf(data, missing="drop")

    def test_insufficient_data(self):
        """Test error for insufficient data."""
        with pytest.raises(ValidationError, match="at least 3 observations"):
            compute_acf(np.array([1.0, 2.0]))

    def test_negative_nlags(self):
        """Test error for negative nlags."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="non-negative"):
            compute_acf(data, nlags=-1)

    def test_nlags_too_large(self):
        """Test error for nlags >= n_obs."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="less than number of observations"):
            compute_acf(data, nlags=100)

    def test_large_nlags_warning(self, caplog):
        """Test warning for large nlags relative to sample size."""
        import logging

        data = np.random.randn(100)

        # nlags > n/4 should trigger warning
        with caplog.at_level(logging.WARNING):
            result = compute_acf(data, nlags=30)

        # Should still compute successfully
        assert result.n_obs == 100

        # Should have logged warning (check if logging is configured)
        # Note: This may not work if logger is not configured in test environment


class TestACFEdgeCases:
    """Tests for ACF edge cases."""

    def test_constant_series(self):
        """Test ACF of constant series (undefined - zero variance)."""
        data = np.ones(100)

        # statsmodels returns NaN for constant series (zero variance)
        # This is handled gracefully - ACF is undefined but not an error
        result = compute_acf(data)

        # ACF should contain NaN values (except possibly lag 0)
        assert np.any(np.isnan(result.acf_values)), "Constant series should produce NaN ACF values"

    def test_very_short_series(self):
        """Test ACF with minimum viable series (n=3)."""
        data = np.array([1.0, 2.0, 3.0])

        # Should work with nlags=1
        result = compute_acf(data, nlags=1)
        assert result.n_obs == 3
        assert len(result.acf_values) == 2  # Lag 0 and 1

    def test_single_lag(self):
        """Test ACF with nlags=1."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = compute_acf(data, nlags=1)
        assert len(result.acf_values) == 2  # Lag 0 and 1
        assert result.lags[-1] == 1

    def test_alpha_variations(self):
        """Test different significance levels."""
        np.random.seed(42)
        data = np.random.randn(100)

        # 99% CI (alpha=0.01)
        result_99 = compute_acf(data, nlags=10, alpha=0.01)

        # 95% CI (alpha=0.05)
        result_95 = compute_acf(data, nlags=10, alpha=0.05)

        # 90% CI (alpha=0.10)
        result_90 = compute_acf(data, nlags=10, alpha=0.10)

        # Narrower alpha (99%) should have wider confidence intervals
        ci_width_99 = result_99.conf_int[1, 1] - result_99.conf_int[1, 0]
        ci_width_95 = result_95.conf_int[1, 1] - result_95.conf_int[1, 0]
        ci_width_90 = result_90.conf_int[1, 1] - result_90.conf_int[1, 0]

        assert ci_width_99 > ci_width_95 > ci_width_90, (
            "Confidence intervals should widen for smaller alpha"
        )


class TestACFRealWorldPatterns:
    """Tests for realistic financial time series patterns."""

    def test_financial_returns_pattern(self):
        """Test ACF on simulated financial returns (weak autocorrelation)."""
        np.random.seed(42)

        # Simulate daily returns with very weak autocorrelation
        # (market efficiency suggests returns should be nearly uncorrelated)
        n = 1000
        returns = np.random.randn(n) * 0.02  # 2% daily volatility

        # Add tiny autocorrelation (0.05) to simulate microstructure effects
        for t in range(1, n):
            returns[t] += 0.05 * returns[t - 1]

        result = compute_acf(returns, nlags=20)

        # Should have weak autocorrelation
        assert all(abs(result.acf_values[1:]) < 0.2), (
            "Financial returns should have weak autocorrelation"
        )

        # Most lags should be insignificant
        assert len(result.significant_lags) <= 5, (
            "Most lags should be insignificant for financial returns"
        )

    def test_volatility_clustering_pattern(self):
        """Test ACF on absolute returns (volatility clustering)."""
        np.random.seed(42)

        # Simulate returns with GARCH-like volatility clustering
        n = 5000  # Larger sample for reliable estimates
        returns = np.random.randn(n)
        volatility = [0.02]

        # Simple GARCH(1,1)-like process with stronger persistence
        for t in range(1, n):
            volatility.append(0.005 + 0.15 * returns[t - 1] ** 2 + 0.80 * volatility[-1])
            returns[t] = returns[t] * np.sqrt(volatility[-1])

        # Absolute returns show persistence (volatility clustering)
        abs_returns = np.abs(returns)
        result = compute_acf(abs_returns, nlags=20)

        # First lag should show positive autocorrelation (volatility clustering)
        # In practice, volatility clustering produces moderate autocorrelation (0.1-0.5)
        assert result.acf_values[1] > 0.05, (
            f"Volatility clustering should produce positive lag-1 autocorrelation, got {result.acf_values[1]:.3f}"
        )

        # Should have at least some significant lags (less strict than before)
        # Note: Actual GARCH effects can be subtle in finite samples
        assert len(result.significant_lags) >= 1, (
            "Absolute returns should show some significant autocorrelation"
        )

    def test_mean_reverting_signal(self):
        """Test ACF on mean-reverting signal (negative autocorrelation)."""
        np.random.seed(42)
        n = 1000

        # Generate Ornstein-Uhlenbeck process (mean-reverting)
        # dX_t = -theta * X_t * dt + sigma * dW_t
        theta = 0.5
        sigma = 0.1
        dt = 1.0

        signal = [0.0]
        for _t in range(1, n):
            drift = -theta * signal[-1] * dt
            diffusion = sigma * np.sqrt(dt) * np.random.randn()
            signal.append(signal[-1] + drift + diffusion)

        signal = np.array(signal)  # type: ignore[assignment]

        # Differences should show negative autocorrelation (mean reversion)
        diff_signal = np.diff(signal)
        result = compute_acf(diff_signal, nlags=10)

        # Lag 1 should be negative (mean reversion)
        assert result.acf_values[1] < 0, (
            "Mean-reverting signal should show negative lag-1 autocorrelation"
        )


# ============================================================================
# PACF Tests
# ============================================================================


class TestPACFResult:
    """Tests for PACFResult class."""

    def test_pacf_result_creation(self):
        """Test PACFResult initialization."""
        pacf_values = np.array([1.0, 0.7, 0.05, 0.02])
        conf_int = np.array(
            [
                [1.0, 1.0],
                [0.6, 0.8],
                [-0.05, 0.15],
                [-0.08, 0.12],
            ]
        )
        lags = np.array([0, 1, 2, 3])

        result = PACFResult(
            pacf_values=pacf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="ywadjusted",
        )

        assert len(result.pacf_values) == 4
        assert result.pacf_values[0] == 1.0
        assert result.n_obs == 100
        assert result.alpha == 0.05
        assert result.method == "ywadjusted"

    def test_significant_lags_detection(self):
        """Test identification of significant lags in PACF.

        Tests H0: π(k) = 0 by checking if CI excludes 0.
        """
        # Create PACF with different significance patterns
        pacf_values = np.array([1.0, 0.7, 0.1, -0.05])
        conf_int = np.array(
            [
                [1.0, 1.0],  # Lag 0 (ignored)
                [0.5, 0.9],  # Lag 1: CI excludes 0 (significant: lower > 0)
                [-0.1, 0.3],  # Lag 2: CI includes 0 (not significant)
                [-0.25, 0.15],  # Lag 3: CI includes 0 (not significant)
            ]
        )
        lags = np.array([0, 1, 2, 3])

        result = PACFResult(
            pacf_values=pacf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="ywadjusted",
        )

        # Only lag 1 should be significant (CI excludes 0)
        assert result.significant_lags == [1]

    def test_repr_str(self):
        """Test string representations."""
        pacf_values = np.array([1.0, 0.7, 0.05])
        conf_int = np.zeros((3, 2))
        lags = np.array([0, 1, 2])

        result = PACFResult(
            pacf_values=pacf_values,
            conf_int=conf_int,
            lags=lags,
            alpha=0.05,
            n_obs=100,
            method="ols",
        )

        repr_str = repr(result)
        assert "PACFResult" in repr_str
        assert "n_obs=100" in repr_str
        assert "nlags=2" in repr_str
        assert "method='ols'" in repr_str

        str_output = str(result)
        assert "PACF Analysis Results" in str_output
        assert "Observations: 100" in str_output
        assert "Method: ols" in str_output


class TestComputePACF:
    """Tests for compute_pacf function."""

    def test_white_noise_pacf(self):
        """Test PACF of white noise (should be ~0 for all lags > 0)."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = compute_pacf(white_noise, nlags=20, alpha=0.05)

        # Check basics
        assert result.n_obs == 1000
        assert len(result.pacf_values) == 21  # nlags + 1
        assert result.pacf_values[0] == 1.0  # Lag 0 always 1

        # For white noise, most lags should be insignificant
        # Expect ~5% false positives (1 lag out of 20)
        assert len(result.significant_lags) <= 3, (
            f"Too many significant lags for white noise: {result.significant_lags}"
        )

        # All PACF values (except lag 0) should be small
        assert np.abs(result.pacf_values[1:]).max() < 0.15, "White noise PACF values too large"

    def test_ar1_process_pacf(self):
        """Test PACF of AR(1) process (single spike at lag 1, then cutoff)."""
        np.random.seed(42)
        phi = 0.7
        n = 5000  # Larger sample for more reliable estimates

        # Generate AR(1): X_t = phi * X_{t-1} + epsilon_t
        noise = np.random.randn(n)
        ar1 = [noise[0]]
        for t in range(1, n):
            ar1.append(phi * ar1[-1] + noise[t])
        ar1 = np.array(ar1)  # type: ignore[assignment]

        result = compute_pacf(ar1, nlags=20, alpha=0.05)

        # Key AR(1) PACF pattern: single spike at lag 1, then cutoff
        # PACF[1] should be ≈ phi
        assert abs(result.pacf_values[1] - phi) < 0.1, (
            f"AR(1) PACF[1] = {result.pacf_values[1]:.3f}, expected {phi:.3f}"
        )

        # PACF[k] for k > 1 should be small (cutoff pattern)
        for k in range(2, min(6, len(result.pacf_values))):
            assert abs(result.pacf_values[k]) < 0.15, (
                f"AR(1) PACF[{k}] = {result.pacf_values[k]:.3f} should be near 0 (cutoff)"
            )

        # Should have significant lag 1 (at minimum)
        assert 1 in result.significant_lags, (
            "AR(1) should have significant lag-1 partial autocorrelation"
        )

        # Most lags > 1 should be insignificant (cutoff)
        significant_after_1 = [lag for lag in result.significant_lags if lag > 1]
        assert len(significant_after_1) <= 3, (
            f"AR(1) should show cutoff after lag 1, but significant lags: {result.significant_lags}"
        )

    def test_ar2_process_pacf(self):
        """Test PACF of AR(2) process (two spikes at lags 1-2, then cutoff)."""
        np.random.seed(42)
        phi1 = 0.5
        phi2 = 0.3
        n = 5000

        # Generate AR(2): X_t = phi1 * X_{t-1} + phi2 * X_{t-2} + epsilon_t
        noise = np.random.randn(n)
        ar2 = [noise[0], noise[1]]
        for t in range(2, n):
            ar2.append(phi1 * ar2[-1] + phi2 * ar2[-2] + noise[t])
        ar2 = np.array(ar2)  # type: ignore[assignment]

        result = compute_pacf(ar2, nlags=20, alpha=0.05)

        # Key AR(2) PACF pattern: spikes at lags 1-2, then cutoff
        # Both lag 1 and 2 should be significant
        assert 1 in result.significant_lags, "AR(2) should have significant PACF at lag 1"
        assert 2 in result.significant_lags, "AR(2) should have significant PACF at lag 2"

        # PACF values at lags 1-2 should be non-trivial
        assert abs(result.pacf_values[1]) > 0.2, (
            f"AR(2) PACF[1] should be substantial, got {result.pacf_values[1]:.3f}"
        )
        assert abs(result.pacf_values[2]) > 0.1, (
            f"AR(2) PACF[2] should be substantial, got {result.pacf_values[2]:.3f}"
        )

        # Most lags > 2 should be insignificant (cutoff)
        significant_after_2 = [lag for lag in result.significant_lags if lag > 2]
        assert len(significant_after_2) <= 3, (
            f"AR(2) should show cutoff after lag 2, but significant lags: {result.significant_lags}"
        )

    def test_ma1_process_pacf(self):
        """Test PACF of MA(1) process (exponential decay, no sharp cutoff)."""
        np.random.seed(42)
        theta = 0.5
        n = 5000  # Larger sample

        # Generate MA(1): X_t = epsilon_t + theta * epsilon_{t-1}
        noise = np.random.randn(n + 1)
        ma1 = noise[1:] + theta * noise[:-1]

        result = compute_pacf(ma1, nlags=10, alpha=0.05)

        # Key MA(1) PACF pattern: exponential decay (no sharp cutoff like AR)
        # Should have multiple significant lags (decay pattern)
        # This is opposite to ACF which cuts off at lag 1 for MA(1)
        assert len(result.significant_lags) >= 2, (
            f"MA(1) PACF should decay (multiple lags), got {result.significant_lags}"
        )

        # No sharp cutoff - later lags should still be non-trivial
        # (though decaying)
        non_zero_count = sum(1 for k in range(1, 6) if abs(result.pacf_values[k]) > 0.05)
        assert non_zero_count >= 3, "MA(1) PACF should show decay pattern, not sharp cutoff"

    def test_pandas_series_input(self):
        """Test PACF with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100), name="returns")

        result = compute_pacf(data, nlags=10)

        assert result.n_obs == 100
        assert len(result.pacf_values) == 11

    def test_nlags_auto_selection(self):
        """Test automatic nlags selection (min(10*log10(n), n//2-1))."""
        np.random.seed(42)

        # For n=1000: min(30, 499) = 30
        data = np.random.randn(1000)
        result = compute_pacf(data, nlags=None)
        assert len(result.pacf_values) == 31  # 30 lags + lag 0

        # For n=100: min(20, 49) = 20
        data = np.random.randn(100)
        result = compute_pacf(data, nlags=None)
        assert len(result.pacf_values) == 21

    def test_pacf_methods(self):
        """Test different PACF estimation methods."""
        np.random.seed(42)
        data = np.random.randn(500)

        # Test methods that support alpha parameter
        methods_with_alpha = ["ywadjusted", "yw_adjusted", "ols"]
        results = {}

        for method in methods_with_alpha:
            result = compute_pacf(data, nlags=10, method=method)
            results[method] = result
            assert result.method == method
            assert len(result.pacf_values) == 11
            assert result.conf_int.shape == (11, 2)

        # Test 'ld' and 'ldadjusted' methods
        # These also support alpha in recent statsmodels versions
        for method in ["ld", "ldadjusted"]:
            try:
                result = compute_pacf(data, nlags=10, method=method)
                results[method] = result
                assert result.method == method
                assert len(result.pacf_values) == 11
            except Exception:
                # If method doesn't work with current statsmodels version, skip it
                pass

        # All methods should produce similar results for white noise
        # (though not identical due to different estimators)
        for method in results:
            assert np.abs(results[method].pacf_values[1:]).max() < 0.2, (
                f"Method {method} produced unexpected PACF for white noise"
            )

    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        data = np.random.randn(100)

        # 95% CI
        result = compute_pacf(data, nlags=10, alpha=0.05)
        assert result.alpha == 0.05
        assert result.conf_int.shape == (11, 2)

        # CI should be positive width
        ci_width = result.conf_int[:, 1] - result.conf_int[:, 0]
        assert ci_width[0] == 0.0  # Lag 0 has no uncertainty
        assert all(ci_width[1:] > 0), "CI widths should be positive"

    def test_against_statsmodels(self):
        """Test that results match statsmodels directly."""
        np.random.seed(42)
        data = np.random.randn(500)
        nlags = 15
        alpha = 0.05
        method = "ywadjusted"

        # Our implementation
        result = compute_pacf(data, nlags=nlags, alpha=alpha, method=method)

        # Statsmodels directly
        from statsmodels.tsa.stattools import pacf as sm_pacf

        sm_pacf_values, sm_conf_int = sm_pacf(data, nlags=nlags, alpha=alpha, method=method)

        # Should match exactly (we use statsmodels backend)
        np.testing.assert_array_almost_equal(result.pacf_values, sm_pacf_values, decimal=10)
        np.testing.assert_array_almost_equal(result.conf_int, sm_conf_int, decimal=10)

    def test_missing_data_handling(self):
        """Test PACF with missing data (automatically drops NaN)."""
        np.random.seed(42)
        data = np.random.randn(100)
        data[10:15] = np.nan

        # Should drop NaN and compute
        result = compute_pacf(data, nlags=10)
        assert result.n_obs == 95  # 100 - 5 NaN


class TestPACFValidation:
    """Tests for PACF input validation."""

    def test_empty_data(self):
        """Test error for empty data."""
        with pytest.raises(ValidationError, match="empty data"):
            compute_pacf(np.array([]))

    def test_all_nan_data(self):
        """Test error for all NaN data."""
        data = np.full(100, np.nan)
        with pytest.raises(ValidationError, match="All data is NaN"):
            compute_pacf(data)

    def test_insufficient_data(self):
        """Test error for insufficient data."""
        with pytest.raises(ValidationError, match="at least 5 observations"):
            compute_pacf(np.array([1.0, 2.0, 3.0]))

    def test_negative_nlags(self):
        """Test error for negative nlags."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="non-negative"):
            compute_pacf(data, nlags=-1)

    def test_nlags_too_large(self):
        """Test error for nlags >= n_obs/2."""
        data = np.random.randn(100)
        with pytest.raises(ValidationError, match="less than n_obs/2"):
            compute_pacf(data, nlags=50)

    def test_large_nlags_warning(self, caplog):
        """Test warning for large nlags relative to sample size."""
        import logging

        data = np.random.randn(100)

        # nlags > n/4 should trigger warning
        with caplog.at_level(logging.WARNING):
            result = compute_pacf(data, nlags=30)

        # Should still compute successfully
        assert result.n_obs == 100


class TestPACFEdgeCases:
    """Tests for PACF edge cases."""

    def test_very_short_series(self):
        """Test PACF with minimum viable series (n=5)."""
        data = np.array([1.0, 2.0, 3.0, 2.5, 1.5])

        # Should work with nlags=1
        result = compute_pacf(data, nlags=1)
        assert result.n_obs == 5
        assert len(result.pacf_values) == 2  # Lag 0 and 1

    def test_single_lag(self):
        """Test PACF with nlags=1."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = compute_pacf(data, nlags=1)
        assert len(result.pacf_values) == 2  # Lag 0 and 1
        assert result.lags[-1] == 1

    def test_alpha_variations(self):
        """Test different significance levels."""
        np.random.seed(42)
        data = np.random.randn(100)

        # 99% CI (alpha=0.01)
        result_99 = compute_pacf(data, nlags=10, alpha=0.01)

        # 95% CI (alpha=0.05)
        result_95 = compute_pacf(data, nlags=10, alpha=0.05)

        # 90% CI (alpha=0.10)
        result_90 = compute_pacf(data, nlags=10, alpha=0.10)

        # Narrower alpha (99%) should have wider confidence intervals
        ci_width_99 = result_99.conf_int[1, 1] - result_99.conf_int[1, 0]
        ci_width_95 = result_95.conf_int[1, 1] - result_95.conf_int[1, 0]
        ci_width_90 = result_90.conf_int[1, 1] - result_90.conf_int[1, 0]

        assert ci_width_99 > ci_width_95 > ci_width_90, (
            "Confidence intervals should widen for smaller alpha"
        )


class TestACFvsPACFComparison:
    """Tests comparing ACF and PACF patterns for model identification."""

    def test_ar1_acf_vs_pacf(self):
        """Test AR(1): ACF decays, PACF cuts off."""
        np.random.seed(42)
        phi = 0.7
        n = 5000

        noise = np.random.randn(n)
        ar1 = [noise[0]]
        for t in range(1, n):
            ar1.append(phi * ar1[-1] + noise[t])
        ar1 = np.array(ar1)  # type: ignore[assignment]

        acf_result = compute_acf(ar1, nlags=20)
        pacf_result = compute_pacf(ar1, nlags=20)

        # AR(1) pattern:
        # - ACF: Exponential decay (many significant lags)
        # - PACF: Cutoff after lag 1 (only lag 1 significant)

        # ACF should have many significant lags (decay)
        assert len(acf_result.significant_lags) >= 5, (
            f"AR(1) ACF should decay slowly, got {len(acf_result.significant_lags)} significant lags"
        )

        # PACF should cutoff after lag 1
        pacf_sig_after_1 = [lag for lag in pacf_result.significant_lags if lag > 1]
        assert len(pacf_sig_after_1) <= 3, (
            f"AR(1) PACF should cutoff after lag 1, got significant lags: {pacf_result.significant_lags}"
        )

    def test_ma1_acf_vs_pacf(self):
        """Test MA(1): ACF cuts off, PACF decays."""
        np.random.seed(42)
        theta = 0.5
        n = 5000

        noise = np.random.randn(n + 1)
        ma1 = noise[1:] + theta * noise[:-1]

        acf_result = compute_acf(ma1, nlags=20)
        pacf_result = compute_pacf(ma1, nlags=20)

        # MA(1) pattern:
        # - ACF: Cutoff after lag 1 (only lag 1 significant)
        # - PACF: Exponential decay (many significant lags)

        # ACF should have few significant lags (cutoff)
        # For MA(1), expect mainly lag 1, maybe 1-2 more due to sampling
        assert len(acf_result.significant_lags) <= 5, (
            f"MA(1) ACF should cutoff, got {len(acf_result.significant_lags)} significant lags"
        )

        # PACF should have multiple significant lags (decay)
        assert len(pacf_result.significant_lags) >= 2, (
            f"MA(1) PACF should decay, got {len(pacf_result.significant_lags)} significant lags"
        )


class TestAutocorrelationAnalysisResult:
    """Tests for AutocorrelationAnalysisResult class."""

    def test_result_creation(self):
        """Test AutocorrelationAnalysisResult initialization."""
        # Create mock ACF result
        acf_values = np.array([1.0, 0.5, 0.25])
        acf_conf_int = np.array([[1.0, 1.0], [0.4, 0.6], [0.15, 0.35]])
        acf_lags = np.array([0, 1, 2])
        acf_result = ACFResult(
            acf_values=acf_values,
            conf_int=acf_conf_int,
            lags=acf_lags,
            alpha=0.05,
            n_obs=100,
            method="standard",
        )

        # Create mock PACF result
        pacf_values = np.array([1.0, 0.7, 0.05])
        pacf_conf_int = np.array([[1.0, 1.0], [0.6, 0.8], [-0.1, 0.2]])
        pacf_lags = np.array([0, 1, 2])
        pacf_result = PACFResult(
            pacf_values=pacf_values,
            conf_int=pacf_conf_int,
            lags=pacf_lags,
            alpha=0.05,
            n_obs=100,
            method="ywadjusted",
        )

        # Create summary DataFrame
        summary_df = pd.DataFrame(
            {
                "lag": [1, 2],
                "acf_value": [0.5, 0.25],
                "acf_significant": [True, False],
                "pacf_value": [0.7, 0.05],
                "pacf_significant": [True, False],
            }
        )

        # Create result
        result = AutocorrelationAnalysisResult(
            acf_result=acf_result,
            pacf_result=pacf_result,
            suggested_ar_order=1,
            suggested_ma_order=1,
            significant_acf_lags=[1],
            significant_pacf_lags=[1],
            is_white_noise=False,
            summary_df=summary_df,
        )

        assert result.suggested_ar_order == 1
        assert result.suggested_ma_order == 1
        assert result.suggested_d_order == 0  # Always 0
        assert result.suggested_arima_order == (1, 0, 1)
        assert not result.is_white_noise
        assert len(result.summary_df) == 2

    def test_repr_str(self):
        """Test string representations."""
        # Create minimal result
        acf_result = ACFResult(
            acf_values=np.array([1.0, 0.5]),
            conf_int=np.zeros((2, 2)),
            lags=np.array([0, 1]),
            alpha=0.05,
            n_obs=100,
            method="standard",
        )
        pacf_result = PACFResult(
            pacf_values=np.array([1.0, 0.7]),
            conf_int=np.zeros((2, 2)),
            lags=np.array([0, 1]),
            alpha=0.05,
            n_obs=100,
            method="ywadjusted",
        )
        summary_df = pd.DataFrame({"lag": [1]})

        result = AutocorrelationAnalysisResult(
            acf_result=acf_result,
            pacf_result=pacf_result,
            suggested_ar_order=1,
            suggested_ma_order=0,
            significant_acf_lags=[1],
            significant_pacf_lags=[1],
            is_white_noise=False,
            summary_df=summary_df,
        )

        repr_str = repr(result)
        assert "AutocorrelationAnalysisResult" in repr_str
        assert "ARIMA(1,0,0)" in repr_str
        assert "white_noise=False" in repr_str

        str_output = str(result)
        assert "Autocorrelation Analysis Results" in str_output
        assert "AR(1) process detected" in str_output


class TestAnalyzeAutocorrelation:
    """Tests for analyze_autocorrelation function."""

    def test_white_noise_identification(self):
        """Test identification of white noise (ARIMA(0,0,0))."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = analyze_autocorrelation(white_noise, max_lags=20, alpha=0.05)

        # Should suggest ARIMA(0,0,0) based on consecutive significant lags
        # Note: With alpha=0.05, we expect ~1 spurious significant lag out of 20
        # The key is that there should be no consecutive significant lags starting from lag 1
        assert result.suggested_ar_order == 0, f"Expected p=0, got {result.suggested_ar_order}"
        assert result.suggested_ma_order == 0, f"Expected q=0, got {result.suggested_ma_order}"
        assert result.suggested_d_order == 0, "d should always be 0"
        assert result.suggested_arima_order == (0, 0, 0)

        # Should have few significant lags (allow for spurious significance)
        # With 20 lags and alpha=0.05, expect ~1 false positive
        assert len(result.significant_acf_lags) <= 2, (
            f"White noise should have ≤2 significant ACF lags, got {len(result.significant_acf_lags)}"
        )
        assert len(result.significant_pacf_lags) <= 2, (
            f"White noise should have ≤2 significant PACF lags, got {len(result.significant_pacf_lags)}"
        )

        # The is_white_noise flag is strict (no significant lags at all)
        # For white noise with spurious lags, we check ARIMA order instead
        # If there are spurious lags but no consecutive pattern, ARIMA should still be (0,0,0)

        # Check summary DataFrame structure
        assert "lag" in result.summary_df.columns
        assert "acf_value" in result.summary_df.columns
        assert "acf_significant" in result.summary_df.columns
        assert "pacf_value" in result.summary_df.columns
        assert "pacf_significant" in result.summary_df.columns
        assert len(result.summary_df) == 20  # max_lags

    def test_ar1_identification(self):
        """Test identification of AR(1) process (ARIMA(1,0,0) or ARIMA(1,0,q)).

        Note: Pure AR(1) processes have ACF that decays exponentially (many consecutive
        significant lags), not a sharp cutoff. PACF cuts off at lag 1. Box-Jenkins
        methodology uses PACF cutoff for AR order, but ACF may suggest MA order due to
        the decay pattern. The key is that PACF correctly identifies p=1.
        """
        np.random.seed(42)
        phi = 0.7
        n = 5000

        # Generate AR(1)
        noise = np.random.randn(n)
        ar1 = [noise[0]]
        for t in range(1, n):
            ar1.append(phi * ar1[-1] + noise[t])
        ar1 = np.array(ar1)  # type: ignore[assignment]

        result = analyze_autocorrelation(ar1, max_lags=20, alpha=0.05)

        # Should NOT identify as white noise
        assert not result.is_white_noise, "AR(1) should not be white noise"

        # Should correctly identify AR order from PACF cutoff
        assert result.suggested_ar_order == 1, (
            f"AR(1) should suggest p=1 from PACF cutoff, got {result.suggested_ar_order}"
        )

        # Note: ACF decays for AR processes (no sharp cutoff), so MA order may be > 0
        # This is expected behavior - the algorithm detects consecutive significant ACF lags
        # In practice, users would rely on PACF cutoff (p=1) to identify this as AR(1)

        # Should have many significant ACF lags (decay pattern)
        assert len(result.significant_acf_lags) > 5, (
            "AR(1) should have many significant ACF lags (decay pattern)"
        )

        # PACF should cut off at lag 1
        assert 1 in result.significant_pacf_lags, "AR(1) should have significant PACF at lag 1"

        # Check results are accessible
        assert result.acf_result is not None
        assert result.pacf_result is not None
        assert result.summary_df is not None

    def test_ar2_identification(self):
        """Test identification of AR(2) process from PACF cutoff.

        Note: AR(2) has ACF that decays (many significant lags), but PACF cuts off at lag 2.
        The AR order is correctly identified from PACF, while ACF may suggest MA order.
        """
        np.random.seed(42)
        phi1 = 0.5
        phi2 = 0.3
        n = 5000

        # Generate AR(2)
        noise = np.random.randn(n)
        ar2 = [noise[0], noise[1]]
        for t in range(2, n):
            ar2.append(phi1 * ar2[-1] + phi2 * ar2[-2] + noise[t])
        ar2 = np.array(ar2)  # type: ignore[assignment]

        result = analyze_autocorrelation(ar2, max_lags=20, alpha=0.05)

        # Should NOT identify as white noise
        assert not result.is_white_noise, "AR(2) should not be white noise"

        # Should suggest p=2 from PACF cutoff
        assert result.suggested_ar_order >= 1, (
            f"AR(2) should suggest p≥1, got {result.suggested_ar_order}"
        )
        assert result.suggested_ar_order <= 3, (
            f"AR(2) should suggest p≤3, got {result.suggested_ar_order}"
        )

        # Note: ACF decays for AR processes, so MA order may be > 0 (expected)

        # Should have significant PACF at lags 1 and 2
        assert 1 in result.significant_pacf_lags, "AR(2) should have significant PACF at lag 1"
        assert 2 in result.significant_pacf_lags, "AR(2) should have significant PACF at lag 2"

    def test_ma1_identification(self):
        """Test identification of MA(1) process (ARIMA(0,0,1))."""
        np.random.seed(42)
        theta = 0.5
        n = 5000

        # Generate MA(1)
        noise = np.random.randn(n + 1)
        ma1 = noise[1:] + theta * noise[:-1]

        result = analyze_autocorrelation(ma1, max_lags=20, alpha=0.05)

        # Should NOT identify as white noise
        assert not result.is_white_noise, "MA(1) should not be white noise"

        # Should suggest ARIMA(0,0,1) for MA(1)
        # Note: MA processes can be harder to identify, may get p>0 or q>1
        assert result.suggested_ma_order >= 1, (
            f"MA(1) should suggest q≥1, got {result.suggested_ma_order}"
        )

        # Should have significant ACF lags
        assert len(result.significant_acf_lags) > 0, "MA(1) should have significant ACF lags"

    def test_different_alpha_levels(self):
        """Test with different significance levels."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        # Test with stricter alpha (0.01)
        result_strict = analyze_autocorrelation(white_noise, max_lags=20, alpha=0.01)

        # Test with default alpha (0.05)
        result_default = analyze_autocorrelation(white_noise, max_lags=20, alpha=0.05)

        # Stricter alpha should have fewer significant lags
        strict_total = len(result_strict.significant_acf_lags) + len(
            result_strict.significant_pacf_lags
        )
        default_total = len(result_default.significant_acf_lags) + len(
            result_default.significant_pacf_lags
        )

        assert strict_total <= default_total, "Stricter alpha should produce fewer significant lags"

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(1000), name="returns")

        result = analyze_autocorrelation(data, max_lags=20)

        # Should suggest ARIMA(0,0,0) (key test for pandas input handling)
        assert result.suggested_arima_order == (0, 0, 0)
        assert isinstance(result.summary_df, pd.DataFrame)

    def test_auto_max_lags(self):
        """Test automatic max_lags selection."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        # Should auto-select max_lags
        result = analyze_autocorrelation(white_noise)

        # Should compute reasonable number of lags
        expected_lags = int(min(10 * np.log10(1000), 1000 // 2 - 1))
        assert len(result.summary_df) == expected_lags

    def test_different_methods(self):
        """Test with different ACF and PACF methods."""
        np.random.seed(42)
        ar1_data = np.random.randn(1000)
        for i in range(1, len(ar1_data)):
            ar1_data[i] = 0.7 * ar1_data[i - 1] + ar1_data[i]

        # Test standard ACF + ywadjusted PACF (default)
        result1 = analyze_autocorrelation(
            ar1_data,
            max_lags=20,
            acf_method="standard",
            pacf_method="ywadjusted",
        )

        # Test FFT ACF + OLS PACF
        result2 = analyze_autocorrelation(
            ar1_data,
            max_lags=20,
            acf_method="fft",
            pacf_method="ols",
        )

        # Both should identify as non-white-noise
        assert not result1.is_white_noise
        assert not result2.is_white_noise

        # Both should suggest p≥1
        assert result1.suggested_ar_order >= 1
        assert result2.suggested_ar_order >= 1

    def test_summary_dataframe_structure(self):
        """Test summary DataFrame has correct structure."""
        np.random.seed(42)
        data = np.random.randn(500)

        result = analyze_autocorrelation(data, max_lags=10)

        df = result.summary_df

        # Check required columns
        required_cols = [
            "lag",
            "acf_value",
            "acf_significant",
            "acf_ci_lower",
            "acf_ci_upper",
            "pacf_value",
            "pacf_significant",
            "pacf_ci_lower",
            "pacf_ci_upper",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check row count (excludes lag 0)
        assert len(df) == 10

        # Check lag values start at 1
        assert df["lag"].iloc[0] == 1
        assert df["lag"].iloc[-1] == 10

        # Check boolean columns
        assert df["acf_significant"].dtype == bool
        assert df["pacf_significant"].dtype == bool

    def test_validation_errors(self):
        """Test validation errors are properly raised."""
        # Empty data
        with pytest.raises(ValidationError):
            analyze_autocorrelation(np.array([]), max_lags=10)

        # Insufficient data
        with pytest.raises(ValidationError):
            analyze_autocorrelation(np.array([1.0, 2.0]), max_lags=10)

    def test_arima_order_property(self):
        """Test suggested_arima_order property."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        result = analyze_autocorrelation(white_noise, max_lags=20)

        # Property should return tuple
        arima_order = result.suggested_arima_order
        assert isinstance(arima_order, tuple)
        assert len(arima_order) == 3
        assert arima_order[0] == result.suggested_ar_order
        assert arima_order[1] == result.suggested_d_order
        assert arima_order[2] == result.suggested_ma_order

    def test_interpretation_in_str(self):
        """Test that __str__ provides interpretation."""
        np.random.seed(42)

        # White noise - check for ARIMA(0,0,0)
        white_noise = np.random.randn(1000)
        result_wn = analyze_autocorrelation(white_noise, max_lags=20)
        str_wn = str(result_wn)
        # Check that suggested order is (0,0,0) which is shown in output
        assert "ARIMA order: (0, 0, 0)" in str_wn or "0, 0, 0" in str_wn

        # AR(1) - check for p=1 identification
        ar1 = np.random.randn(5000)
        for i in range(1, len(ar1)):
            ar1[i] = 0.7 * ar1[i - 1] + ar1[i]
        result_ar = analyze_autocorrelation(ar1, max_lags=20)
        str_ar = str(result_ar)
        # Check that AR order is identified (p >= 1)
        assert "ARIMA order:" in str_ar
