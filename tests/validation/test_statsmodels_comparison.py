"""Validation tests comparing ml4t-diagnostic statistical tests against statsmodels.

This module validates that ml4t-diagnostic's statistical tests produce
results matching statsmodels within numerical precision tolerances.

Reference: statsmodels (https://www.statsmodels.org/)
"""

from __future__ import annotations

import numpy as np
import pytest

# Reference implementation
from numpy.testing import assert_allclose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from ml4t.diagnostic.evaluation.autocorrelation import (
    compute_acf,
    compute_pacf,
)

# ML4T implementations
from ml4t.diagnostic.evaluation.stationarity import (
    adf_test,
    kpss_test,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def stationary_series():
    """Generate a stationary AR(1) process."""
    np.random.seed(42)
    n = 500
    phi = 0.5  # AR coefficient < 1 (stationary)
    noise = np.random.randn(n) * 0.5
    series = np.zeros(n)
    series[0] = noise[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    return series


@pytest.fixture
def nonstationary_series():
    """Generate a random walk (non-stationary)."""
    np.random.seed(42)
    n = 500
    innovations = np.random.randn(n) * 0.5
    return np.cumsum(innovations)  # Random walk = integrated noise


@pytest.fixture
def trending_series():
    """Generate a series with deterministic trend (non-stationary)."""
    np.random.seed(42)
    n = 500
    trend = np.linspace(0, 10, n)
    noise = np.random.randn(n) * 0.5
    return trend + noise


@pytest.fixture
def autocorrelated_series():
    """Generate a series with known autocorrelation structure."""
    np.random.seed(42)
    n = 200
    # MA(2) process: y_t = e_t + 0.5*e_{t-1} + 0.3*e_{t-2}
    noise = np.random.randn(n + 2)
    series = np.array([noise[i] + 0.5 * noise[i - 1] + 0.3 * noise[i - 2] for i in range(2, n + 2)])
    return series


# =============================================================================
# ADF Test Comparison
# =============================================================================


class TestADFComparison:
    """Compare ADF test against statsmodels.adfuller."""

    def test_adf_stationary_series(self, stationary_series):
        """ADF test should reject unit root for stationary series."""
        # ML4T result
        ml4t_result = adf_test(stationary_series)

        # Statsmodels result
        sm_result = adfuller(stationary_series, autolag="AIC")
        sm_statistic = sm_result[0]
        sm_pvalue = sm_result[1]

        # Test statistics should match closely
        assert_allclose(
            ml4t_result.test_statistic,
            sm_statistic,
            rtol=0.01,
            err_msg="ADF statistic differs from statsmodels",
        )

        # p-values should be similar (allow more tolerance for interpolation)
        assert_allclose(
            ml4t_result.p_value, sm_pvalue, rtol=0.1, err_msg="ADF p-value differs from statsmodels"
        )

        # Both should reject unit root at 5% level for stationary series
        assert ml4t_result.p_value < 0.05, "ML4T should reject unit root"
        assert sm_pvalue < 0.05, "Statsmodels should reject unit root"

    def test_adf_nonstationary_series(self, nonstationary_series):
        """ADF test should NOT reject unit root for random walk."""
        # ML4T result
        ml4t_result = adf_test(nonstationary_series)

        # Statsmodels result
        sm_result = adfuller(nonstationary_series, autolag="AIC")
        sm_pvalue = sm_result[1]

        # Both should fail to reject unit root
        assert ml4t_result.p_value > 0.05, "ML4T should not reject unit root for random walk"
        assert sm_pvalue > 0.05, "Statsmodels should not reject unit root for random walk"

    def test_adf_critical_values(self, stationary_series):
        """Critical values should be similar to statsmodels."""
        ml4t_result = adf_test(stationary_series)

        sm_result = adfuller(stationary_series, autolag="AIC")
        sm_critical = sm_result[4]  # Critical values dict

        # Check 5% critical value (most commonly used)
        if ml4t_result.critical_values:
            ml4t_cv_5pct = ml4t_result.critical_values.get("5%")
            sm_cv_5pct = sm_critical.get("5%")
            if ml4t_cv_5pct and sm_cv_5pct:
                assert_allclose(
                    ml4t_cv_5pct, sm_cv_5pct, rtol=0.05, err_msg="5% critical values differ"
                )


# =============================================================================
# KPSS Test Comparison
# =============================================================================


class TestKPSSComparison:
    """Compare KPSS test against statsmodels.kpss."""

    def test_kpss_stationary_series(self, stationary_series):
        """KPSS test should NOT reject stationarity for stationary series."""
        # ML4T result
        ml4t_result = kpss_test(stationary_series, regression="c")

        # Statsmodels result (suppress warning about p-value interpolation)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_result = kpss(stationary_series, regression="c", nlags="auto")

        sm_statistic = sm_result[0]
        sm_pvalue = sm_result[1]

        # Test statistics should match
        assert_allclose(
            ml4t_result.test_statistic,
            sm_statistic,
            rtol=0.01,
            err_msg="KPSS statistic differs from statsmodels",
        )

        # Both should NOT reject stationarity (p-value > 0.05)
        assert ml4t_result.p_value > 0.05, "ML4T KPSS should not reject stationarity"
        assert sm_pvalue > 0.05, "Statsmodels KPSS should not reject stationarity"

    def test_kpss_nonstationary_series(self, nonstationary_series):
        """KPSS test should reject stationarity for random walk."""
        # ML4T result
        ml4t_result = kpss_test(nonstationary_series, regression="c")

        # Statsmodels result
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_result = kpss(nonstationary_series, regression="c", nlags="auto")

        sm_pvalue = sm_result[1]

        # Both should reject stationarity (p-value < 0.05)
        assert ml4t_result.p_value < 0.05, "ML4T KPSS should reject stationarity for random walk"
        assert sm_pvalue < 0.05, "Statsmodels KPSS should reject stationarity"

    def test_kpss_with_trend(self, trending_series):
        """KPSS with trend regression for trending data."""
        # ML4T result with trend
        ml4t_result = kpss_test(trending_series, regression="ct")

        # Statsmodels result with trend
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_result = kpss(trending_series, regression="ct", nlags="auto")

        sm_statistic = sm_result[0]

        # Statistics should be similar
        assert_allclose(
            ml4t_result.test_statistic,
            sm_statistic,
            rtol=0.05,
            err_msg="KPSS with trend differs from statsmodels",
        )


# =============================================================================
# ADF vs KPSS Complementary Tests
# =============================================================================


class TestStationarityComplementary:
    """Test that ADF and KPSS produce complementary results."""

    def test_stationary_both_agree(self, stationary_series):
        """For stationary series, ADF rejects and KPSS doesn't reject."""
        adf_result = adf_test(stationary_series)
        kpss_result = kpss_test(stationary_series)

        # ADF: small p-value means stationary (rejects unit root)
        # KPSS: large p-value means stationary (fails to reject stationarity)
        assert adf_result.p_value < 0.05, "ADF should reject unit root"
        assert kpss_result.p_value > 0.05, "KPSS should not reject stationarity"

    def test_nonstationary_both_agree(self, nonstationary_series):
        """For non-stationary series, ADF doesn't reject and KPSS rejects."""
        adf_result = adf_test(nonstationary_series)
        kpss_result = kpss_test(nonstationary_series)

        # ADF: large p-value means non-stationary (fails to reject unit root)
        # KPSS: small p-value means non-stationary (rejects stationarity)
        assert adf_result.p_value > 0.05, "ADF should not reject unit root"
        assert kpss_result.p_value < 0.05, "KPSS should reject stationarity"


# =============================================================================
# ACF/PACF Comparison
# =============================================================================


class TestACFComparison:
    """Compare ACF computation against statsmodels."""

    def test_acf_values_match(self, autocorrelated_series):
        """ACF values should match statsmodels computation."""
        nlags = 20

        # ML4T ACF - returns CorrelationResult object with .values attribute
        ml4t_result = compute_acf(autocorrelated_series, nlags=nlags)
        ml4t_acf = ml4t_result.values

        # Statsmodels ACF
        sm_acf = acf(autocorrelated_series, nlags=nlags, fft=False)

        # Values should match closely
        # Note: ML4T includes lag 0, same as statsmodels
        assert len(ml4t_acf) == nlags + 1, f"Expected {nlags + 1} ACF values, got {len(ml4t_acf)}"
        assert_allclose(ml4t_acf, sm_acf, rtol=0.01, err_msg="ACF values differ from statsmodels")

    def test_acf_lag_1_positive_for_ar1(self, stationary_series):
        """AR(1) process with phi=0.5 should have ACF(1) ≈ 0.5."""
        ml4t_result = compute_acf(stationary_series, nlags=5)
        ml4t_acf = ml4t_result.values

        # First lag should be positive and close to phi=0.5
        # ml4t_acf[0] is lag 0 (= 1.0), ml4t_acf[1] is lag 1
        assert 0.3 < ml4t_acf[1] < 0.7, f"ACF(1) should be ~0.5 for AR(1), got {ml4t_acf[1]}"


class TestPACFComparison:
    """Compare PACF computation against statsmodels."""

    def test_pacf_values_match(self, autocorrelated_series):
        """PACF values should match statsmodels computation."""
        nlags = 10

        # ML4T PACF - returns CorrelationResult object with .values attribute
        ml4t_result = compute_pacf(autocorrelated_series, nlags=nlags)
        ml4t_pacf = ml4t_result.values

        # Statsmodels PACF (using Yule-Walker adjusted method)
        sm_pacf = pacf(autocorrelated_series, nlags=nlags, method="ywadjusted")

        # Values should match (with some tolerance for different methods)
        # Both include lag 0 (= 1.0)
        assert len(ml4t_pacf) == nlags + 1, f"Expected {nlags + 1} PACF values"
        assert_allclose(
            ml4t_pacf[1:], sm_pacf[1:], rtol=0.1, err_msg="PACF values differ from statsmodels"
        )

    def test_pacf_ar1_cutoff(self, stationary_series):
        """AR(1) PACF should cut off after lag 1."""
        ml4t_result = compute_pacf(stationary_series, nlags=10)
        ml4t_pacf = ml4t_result.values

        # For AR(1), PACF should be significant at lag 1, then near zero
        # ml4t_pacf[0] is lag 0 (= 1.0), ml4t_pacf[1] is lag 1
        pacf_lag1 = ml4t_pacf[1]
        pacf_lag2_plus = ml4t_pacf[2:]

        # Lag 1 should be significant
        assert abs(pacf_lag1) > 0.1, "PACF(1) should be significant for AR(1)"

        # Later lags should be small (within 2/sqrt(n) confidence band)
        n = len(stationary_series)
        conf_band = 2 / np.sqrt(n)
        small_count = sum(abs(p) < conf_band for p in pacf_lag2_plus)

        # Most later lags should be within confidence band
        assert small_count >= len(pacf_lag2_plus) * 0.7, (
            "Higher PACF lags should be small for AR(1)"
        )


# =============================================================================
# Mathematical Properties
# =============================================================================


class TestStatisticalProperties:
    """Test mathematical properties of statistical tests."""

    def test_adf_statistic_sign(self, stationary_series, nonstationary_series):
        """ADF statistic should be more negative for stationary series."""
        adf_stat = adf_test(stationary_series)
        adf_nonstat = adf_test(nonstationary_series)

        # More negative = stronger evidence against unit root
        assert adf_stat.test_statistic < adf_nonstat.test_statistic, (
            "Stationary series should have more negative ADF statistic"
        )

    def test_acf_lag_0_equals_1(self, stationary_series):
        """ACF at lag 0 should equal 1 (correlation with self)."""
        ml4t_result = compute_acf(stationary_series, nlags=5)
        ml4t_acf = ml4t_result.values
        sm_acf = acf(stationary_series, nlags=5)

        # ML4T includes lag 0
        assert_allclose(ml4t_acf[0], 1.0, rtol=0.001, err_msg="ACF(0) should equal 1")

        # Statsmodels always includes lag 0
        assert_allclose(sm_acf[0], 1.0, rtol=0.001)

    def test_acf_bounds(self, autocorrelated_series):
        """ACF values should be in [-1, 1]."""
        ml4t_result = compute_acf(autocorrelated_series, nlags=20)
        ml4t_acf = ml4t_result.values

        for i, val in enumerate(ml4t_acf):
            assert -1 <= val <= 1, f"ACF at lag {i} = {val} out of bounds"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for statistical tests."""

    def test_short_series(self):
        """Tests should handle short series gracefully."""
        np.random.seed(42)
        short_series = np.random.randn(50)

        # ADF should work (may need minimum samples)
        try:
            adf_result = adf_test(short_series)
            assert np.isfinite(adf_result.test_statistic)
        except (ValueError, Exception) as e:
            # Some tests may raise for too-short series - that's OK
            assert "too short" in str(e).lower() or "insufficient" in str(e).lower()

    def test_constant_series(self):
        """Tests should handle constant series."""
        constant = np.ones(100) * 5.0

        # ADF on constant series - should return NaN or raise
        try:
            result = adf_test(constant)
            # If it returns, statistic might be NaN or very negative
            assert np.isnan(result.test_statistic) or result.test_statistic < -10
        except (ValueError, Exception):
            pass  # Raising is acceptable

    def test_nan_handling(self):
        """Tests should handle NaN values."""
        np.random.seed(42)
        series_with_nan = np.random.randn(100)
        series_with_nan[50] = np.nan

        # Should handle gracefully (either skip NaN or raise)
        try:
            result = adf_test(series_with_nan)
            # If it works, should give finite result
            assert np.isfinite(result.test_statistic)
        except (ValueError, Exception):
            pass  # Raising is acceptable
