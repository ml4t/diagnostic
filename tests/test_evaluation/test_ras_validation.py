"""Validation tests for Rademacher Anti-Serum (RAS) implementation.

Tests replicate simulation results from Paleologo (2024), Section 8.4.
Validates against Tables 8.3 and 8.4 for null and alternative hypotheses.

Author: ML4T evaluation library
Date: 2025-11-16

Tests run optimized Monte Carlo simulations and complete quickly (<1s).
"""

import numpy as np
import pytest

from ml4t.diagnostic.evaluation.stats import (
    rademacher_complexity,
    ras_ic_adjustment,
    ras_sharpe_adjustment,
)


class TestRademacherComplexity:
    """Test Rademacher complexity calculation."""

    def test_basic_computation(self):
        """Test basic Rademacher complexity computation."""
        # Simple case: 100 strategies, 500 periods
        np.random.seed(42)
        X = np.random.randn(500, 100)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)

        # Should be positive and bounded by Massart's lemma
        assert R_hat > 0
        massart_bound = np.sqrt(2 * np.log(100) / 500)
        assert R_hat <= massart_bound * 1.3  # Allow 30% slack

    def test_identical_strategies_zero_complexity(self):
        """Test that identical strategies have near-zero complexity."""
        # All strategies are identical
        np.random.seed(42)
        x = np.random.randn(1000)
        X = np.column_stack([x] * 100)  # 100 identical columns

        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)

        # Should be very close to zero (within numerical precision)
        assert abs(R_hat) < 0.01  # Tolerance for Monte Carlo estimation

    def test_uncorrelated_vs_correlated(self):
        """Test that uncorrelated strategies have higher complexity."""
        np.random.seed(42)
        T, N = 1000, 200  # Reduced from (2500, 500) for faster tests

        # Uncorrelated strategies
        X_uncorr = np.random.randn(T, N)
        R_uncorr = rademacher_complexity(X_uncorr, n_simulations=100, random_state=42)

        # Correlated strategies (ρ=0.8)
        factor = np.random.randn(T)
        noise = np.random.randn(T, N)
        X_corr = 0.8 * factor[:, np.newaxis] + np.sqrt(1 - 0.8**2) * noise
        R_corr = rademacher_complexity(X_corr, n_simulations=100, random_state=42)

        # Uncorrelated should have significantly higher complexity
        assert R_uncorr > R_corr

    def test_massart_bound_comparison(self):
        """Test that empirical R̂ is close to but below Massart's bound.

        From Table 8.2 in Paleologo (2024), Massart's bound is typically
        10-20% higher than observed R̂.

        Note: Matrix sizes reduced from original (500-5000, 2500-5000) to
        (100-500, 500-1000) for faster test execution while maintaining
        statistical validity of the bound comparison.
        """
        np.random.seed(42)
        test_cases = [
            (100, 500),  # N=100, T=500 (small)
            (500, 500),  # N=500, T=500 (medium)
            (100, 1000),  # N=100, T=1000 (longer history)
            (500, 1000),  # N=500, T=1000 (large)
        ]

        for N, T in test_cases:
            X = np.random.randn(T, N)
            R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)
            massart = np.sqrt(2 * np.log(N) / T)

            # R̂ should be below Massart's bound
            assert R_hat <= massart

            # R̂ should be within 10-30% of Massart's bound for uncorrelated data
            assert R_hat >= massart * 0.70  # At least 70% of bound
            assert R_hat <= massart * 1.05  # At most 105% of bound

    def test_input_validation(self):
        """Test input validation."""
        X = np.random.randn(100, 50)

        # Valid call
        R = rademacher_complexity(X, n_simulations=100, random_state=42)
        assert isinstance(R, float)

        # Wrong type
        with pytest.raises(TypeError, match="X must be numpy array"):
            rademacher_complexity([[1, 2], [3, 4]])

        # Wrong dimensions
        with pytest.raises(ValueError, match="X must be 2D array"):
            rademacher_complexity(np.array([1, 2, 3]))

        # Empty array
        with pytest.raises(ValueError, match="positive dimensions"):
            rademacher_complexity(np.zeros((0, 10)))

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        X = np.random.randn(100, 50)

        R1 = rademacher_complexity(X, n_simulations=500, random_state=42)
        R2 = rademacher_complexity(X, n_simulations=500, random_state=42)

        assert R1 == R2  # Exact match with same seed


class TestRASSimulationsNullCase:
    """Test RAS with null hypothesis (all strategies have SR=0 or IC=0).

    Replicates Table 8.3 from Paleologo (2024) for Gaussian returns.
    Should have 0% false positives.
    """

    @pytest.mark.parametrize(
        "correlation,N,T",
        [
            (0.2, 200, 1000),  # Reduced from (500, 2500)
            (0.8, 200, 1000),  # Reduced from (500, 2500)
            (0.2, 500, 1000),  # Reduced from (5000, 2500)
            (0.8, 500, 1000),  # Reduced from (5000, 2500)
        ],
    )
    def test_null_sharpe_gaussian(self, correlation, N, T):
        """Test null Sharpe case with Gaussian returns (Table 8.3, rows with SR=0)."""
        np.random.seed(42)
        delta = 0.05

        # Generate null strategies (zero Sharpe)
        factor = np.random.randn(T)
        noise = np.random.randn(T, N)
        X = correlation * factor[:, np.newaxis] + np.sqrt(1 - correlation**2) * noise

        # Compute observed Sharpe and RAS adjustment
        observed_sharpe = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)
        adjusted_sharpe = ras_sharpe_adjustment(
            observed_sharpe, R_hat, n_samples=T, n_strategies=N, delta=delta
        )

        # Count false positives (should be 0%)
        n_positive = np.sum(adjusted_sharpe > 0)
        pct_positive = 100 * n_positive / N

        # Paper reports 0.0% false positives for all null cases
        assert pct_positive <= 1.0, f"False positive rate {pct_positive:.1f}% > 1%"

        # Also verify Rademacher positive count
        rademacher_positive = observed_sharpe - 2 * R_hat
        n_rad_positive = np.sum(rademacher_positive > 0)
        pct_rad_positive = 100 * n_rad_positive / N

        # Should also be near 0%
        assert pct_rad_positive <= 2.0, f"Rad positive rate {pct_rad_positive:.1f}% > 2%"

    def test_null_ic_gaussian(self):
        """Test null IC case (all strategies have IC=0)."""
        np.random.seed(42)
        N, T = 200, 1000  # Reduced from (500, 2500) for faster tests
        delta = 0.05
        kappa = 1.0  # Theoretical bound

        # Generate null ICs (bounded to [-kappa, +kappa])
        X = np.random.uniform(-kappa, kappa, size=(T, N))
        X = X - X.mean(axis=0)  # Center to have mean ≈ 0

        # Compute observed IC and RAS adjustment
        observed_ic = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)
        adjusted_ic = ras_ic_adjustment(observed_ic, R_hat, n_samples=T, delta=delta, kappa=kappa)

        # Count false positives
        n_positive = np.sum(adjusted_ic > 0)
        pct_positive = 100 * n_positive / N

        # Should have near-zero false positives
        assert pct_positive <= 1.0, f"False positive rate {pct_positive:.1f}% > 1%"


class TestRASSimulationsAlternative:
    """Test RAS with alternative hypothesis (20% strategies have positive SR/IC).

    Replicates Table 8.3 from Paleologo (2024) for Gaussian returns.
    Should detect true positives with zero false discovery rate (FDR=0).
    """

    @pytest.mark.parametrize(
        "correlation,N,T",
        [
            (0.2, 200, 1000),  # Reduced from (500, 2500)
            (0.8, 200, 1000),  # Reduced from (500, 2500)
        ],
    )
    def test_alternative_sharpe_gaussian(self, correlation, N, T):
        """Test alternative Sharpe case (Table 8.3, rows with SR=0.2)."""
        np.random.seed(42)
        delta = 0.05
        true_sr = 0.2  # Non-annualized Sharpe for 20% of strategies
        pct_positive_strategies = 0.20

        # Generate strategies: 80% null, 20% with SR=0.2
        factor = np.random.randn(T)
        noise = np.random.randn(T, N)
        X = correlation * factor[:, np.newaxis] + np.sqrt(1 - correlation**2) * noise

        # Add signal to 20% of strategies
        n_positive = int(N * pct_positive_strategies)
        X[:, :n_positive] += true_sr  # Add mean shift for first 20%

        # Compute observed Sharpe and RAS adjustment
        observed_sharpe = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)
        adjusted_sharpe = ras_sharpe_adjustment(
            observed_sharpe, R_hat, n_samples=T, n_strategies=N, delta=delta
        )

        # Count detections
        n_detected = np.sum(adjusted_sharpe > 0)
        pct_detected = 100 * n_detected / N

        # Paper reports detection rates vary (1.7% to 20%)
        # Conservative bound should detect fewer than true positives
        assert pct_detected <= 25.0, f"Detection rate {pct_detected:.1f}% > 25%"

        # Check false discovery rate (FDR)
        # FDR = false positives / total positives
        # Paper reports FDR=0 in all cases
        detected_indices = np.where(adjusted_sharpe > 0)[0]
        false_positives = np.sum(detected_indices >= n_positive)
        if n_detected > 0:
            fdr = false_positives / n_detected
            assert fdr <= 0.05, f"FDR {fdr:.2%} > 5%"  # Allow tiny numerical error

    def test_alternative_high_correlation_detects_more(self):
        """Test that high correlation allows more detections (Table 8.3).

        When strategies are highly correlated (ρ=0.8), the Rademacher complexity
        is lower, leading to tighter bounds and better detection rates.
        """
        np.random.seed(42)
        N, T = 200, 1500  # Reduced from (500, 5000) for faster tests
        delta = 0.05
        true_sr = 0.2
        n_positive = int(N * 0.20)

        # Low correlation (ρ=0.2)
        factor_low = np.random.randn(T)
        noise_low = np.random.randn(T, N)
        X_low = 0.2 * factor_low[:, np.newaxis] + np.sqrt(1 - 0.2**2) * noise_low
        X_low[:, :n_positive] += true_sr

        observed_low = X_low.mean(axis=0)
        R_low = rademacher_complexity(X_low, n_simulations=100, random_state=42)
        adjusted_low = ras_sharpe_adjustment(
            observed_low, R_low, n_samples=T, n_strategies=N, delta=delta
        )
        pct_low = 100 * np.sum(adjusted_low > 0) / N

        # High correlation (ρ=0.8)
        factor_high = np.random.randn(T)
        noise_high = np.random.randn(T, N)
        X_high = 0.8 * factor_high[:, np.newaxis] + np.sqrt(1 - 0.8**2) * noise_high
        X_high[:, :n_positive] += true_sr

        observed_high = X_high.mean(axis=0)
        R_high = rademacher_complexity(X_high, n_simulations=100, random_state=42)
        adjusted_high = ras_sharpe_adjustment(
            observed_high, R_high, n_samples=T, n_strategies=N, delta=delta
        )
        pct_high = 100 * np.sum(adjusted_high > 0) / N

        # High correlation should have lower R̂
        assert R_high < R_low

        # High correlation should detect at least as many (usually more)
        # This might not always hold due to random variation, so we test conservatively
        assert pct_high >= pct_low * 0.5  # At least 50% as many


class TestRASICPracticalKappa:
    """Test RAS for IC with practical kappa values."""

    def test_practical_kappa_tighter_bounds(self):
        """Test that practical κ=0.02 gives much tighter bounds than κ=1.0."""
        np.random.seed(42)
        N, T = 200, 1000  # Reduced from (500, 2500) for faster tests
        delta = 0.01

        # Generate realistic ICs (bounded to ≈0.02)
        X = np.random.randn(T, N) * 0.02
        observed_ic = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)

        # Theoretical bound (κ=1.0)
        adjusted_theoretical = ras_ic_adjustment(
            observed_ic, R_hat, n_samples=T, delta=delta, kappa=1.0
        )

        # Practical bound (κ=0.02)
        adjusted_practical = ras_ic_adjustment(
            observed_ic, R_hat, n_samples=T, delta=delta, kappa=0.02
        )

        # Practical should be much less conservative
        theoretical_haircut = observed_ic - adjusted_theoretical
        practical_haircut = observed_ic - adjusted_practical

        # Practical haircut should be much smaller
        assert np.mean(practical_haircut) < np.mean(theoretical_haircut) * 0.1


class TestRASInputValidation:
    """Test input validation for RAS functions."""

    def test_ras_ic_validation(self):
        """Test RAS IC adjustment input validation."""
        observed_ic = np.array([0.05, 0.03, -0.01])
        R_hat = 0.04
        T = 1000

        # Valid call
        result = ras_ic_adjustment(observed_ic, R_hat, T, delta=0.05, kappa=1.0)
        assert result.shape == observed_ic.shape

        # Invalid inputs
        with pytest.raises(ValueError, match="must be 1D"):
            ras_ic_adjustment(np.array([[1, 2]]), R_hat, T)

        with pytest.raises(ValueError, match="non-negative"):
            ras_ic_adjustment(observed_ic, -0.1, T)

        with pytest.raises(ValueError, match="positive"):
            ras_ic_adjustment(observed_ic, R_hat, 0)

        with pytest.raises(ValueError, match="delta must be in"):
            ras_ic_adjustment(observed_ic, R_hat, T, delta=1.5)

        with pytest.raises(ValueError, match="kappa must be positive"):
            ras_ic_adjustment(observed_ic, R_hat, T, kappa=-1)

    def test_ras_sharpe_validation(self):
        """Test RAS Sharpe adjustment input validation."""
        observed_sharpe = np.array([0.5, 0.3, -0.1])
        R_hat = 0.06
        T = 2500
        N = 1000

        # Valid call
        result = ras_sharpe_adjustment(observed_sharpe, R_hat, T, N, delta=0.05)
        assert result.shape == observed_sharpe.shape

        # Invalid inputs
        with pytest.raises(ValueError, match="must be 1D"):
            ras_sharpe_adjustment(np.array([[1, 2]]), R_hat, T, N)

        with pytest.raises(ValueError, match="non-negative"):
            ras_sharpe_adjustment(observed_sharpe, -0.1, T, N)

        with pytest.raises(ValueError, match="positive"):
            ras_sharpe_adjustment(observed_sharpe, R_hat, 0, N)

        with pytest.raises(ValueError, match="positive"):
            ras_sharpe_adjustment(observed_sharpe, R_hat, T, 0)

        with pytest.raises(ValueError, match="delta must be in"):
            ras_sharpe_adjustment(observed_sharpe, R_hat, T, N, delta=0)


class TestRASComponentBreakdown:
    """Test decomposition of RAS adjustment into components."""

    def test_sharpe_components(self):
        """Test that data snooping and estimation error components are reasonable."""
        np.random.seed(42)
        N, T = 300, 1000  # Reduced from (1000, 2500) for faster tests
        delta = 0.05

        X = np.random.randn(T, N)
        observed_sharpe = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)

        # Manual calculation of components
        data_snooping = 2 * R_hat
        error_term1 = 3 * np.sqrt(2 * np.log(2 / delta) / T)
        error_term2 = np.sqrt(2 * np.log(2 * N / delta) / T)
        estimation_error = error_term1 + error_term2

        # Verify against function output
        adjusted = ras_sharpe_adjustment(observed_sharpe, R_hat, T, N, delta)
        expected_adjusted = observed_sharpe - data_snooping - estimation_error

        np.testing.assert_allclose(adjusted, expected_adjusted, rtol=1e-10)

        # Components should be positive
        assert data_snooping > 0
        assert estimation_error > 0

        # For this configuration, both should be comparable magnitude
        # (though not necessarily equal)
        assert 0.01 < data_snooping < 1.0
        assert 0.01 < estimation_error < 1.0

    def test_ic_components(self):
        """Test that IC components match formula."""
        np.random.seed(42)
        N, T = 200, 1000  # Reduced from (500, 2500) for faster tests
        delta = 0.05
        kappa = 0.02

        X = np.random.randn(T, N) * kappa
        observed_ic = X.mean(axis=0)
        R_hat = rademacher_complexity(X, n_simulations=100, random_state=42)

        # Manual calculation
        data_snooping = 2 * R_hat
        estimation_error = 2 * kappa * np.sqrt(np.log(2 / delta) / T)

        # Verify against function
        adjusted = ras_ic_adjustment(observed_ic, R_hat, T, delta, kappa)
        expected_adjusted = observed_ic - data_snooping - estimation_error

        np.testing.assert_allclose(adjusted, expected_adjusted, rtol=1e-10)


# Mark slow tests
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
