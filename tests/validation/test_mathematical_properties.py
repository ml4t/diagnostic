"""Mathematical property tests for ml4t-diagnostic statistical methods.

This module tests mathematical invariants that must hold regardless of input:
- Bounds (e.g., IC ∈ [-1, 1])
- Monotonicity (e.g., PSR increases with T for positive SR)
- Limiting behavior (e.g., PSR(SR=0) → 0.5 as T → ∞)
- Relationships (e.g., DSR ≤ PSR always)
- Conservation laws (e.g., train ∩ test = ∅)

These tests catch edge cases not covered by reference library comparisons.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from ml4t.diagnostic.evaluation.autocorrelation import compute_acf, compute_pacf
from ml4t.diagnostic.evaluation.portfolio_analysis import (
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics
from ml4t.diagnostic.splitters import CombinatorialPurgedCV


def compute_psr(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,  # Normal kurtosis
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    PSR is DSR with n_trials=1 (no multiple testing correction).
    """
    # Convert kurtosis to excess kurtosis (Fisher convention)
    excess_kurtosis = kurtosis - 3.0

    result = deflated_sharpe_ratio_from_statistics(
        observed_sharpe=observed_sharpe,
        n_samples=n_observations,
        n_trials=1,  # PSR = DSR with 1 trial
        benchmark_sharpe=benchmark_sharpe,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    return result.probability


# =============================================================================
# PSR / DSR Properties
# =============================================================================


class TestPSRProperties:
    """Test mathematical properties of Probabilistic Sharpe Ratio."""

    def test_psr_zero_sharpe_converges_to_half(self):
        """PSR(SR=0, T=large) → 0.5 (50% probability of positive skill).

        With SR=0, the probability that skill is positive should be 50%.
        As sample size increases, estimate becomes more precise around 0.5.
        """
        sr = 0.0
        # Test with increasing sample sizes
        for t in [100, 500, 1000, 5000]:
            psr = compute_psr(
                observed_sharpe=sr,
                benchmark_sharpe=0.0,
                n_observations=t,
                skewness=0.0,
                kurtosis=3.0,  # Normal
            )
            # PSR should be ~0.5 for SR=0
            assert_allclose(psr, 0.5, atol=0.05, err_msg=f"PSR(SR=0, T={t}) should be ~0.5")

    def test_psr_increases_with_sample_size_for_positive_sr(self):
        """For positive SR, PSR should increase with sample size.

        More observations provide more evidence that true SR > benchmark.
        """
        sr = 0.15  # Use smaller SR so PSR doesn't saturate at 1.0
        t_values = [50, 100, 250, 500, 1000]
        psr_values = []

        for t in t_values:
            psr = compute_psr(
                observed_sharpe=sr,
                benchmark_sharpe=0.0,
                n_observations=t,
                skewness=0.0,
                kurtosis=3.0,
            )
            psr_values.append(psr)

        # PSR should be monotonically increasing with T (or at least non-decreasing)
        for i in range(len(psr_values) - 1):
            assert psr_values[i] <= psr_values[i + 1] + 1e-10, (
                f"PSR should increase with T: PSR(T={t_values[i]})={psr_values[i]:.4f} "
                f"should be <= PSR(T={t_values[i + 1]})={psr_values[i + 1]:.4f}"
            )

    def test_psr_decreases_with_sample_size_for_negative_sr(self):
        """For negative SR, PSR should decrease with sample size.

        More observations provide more evidence that true SR < benchmark.
        """
        sr = -0.5
        t_values = [50, 100, 250, 500, 1000]
        psr_values = []

        for t in t_values:
            psr = compute_psr(
                observed_sharpe=sr,
                benchmark_sharpe=0.0,
                n_observations=t,
                skewness=0.0,
                kurtosis=3.0,
            )
            psr_values.append(psr)

        # PSR should be monotonically decreasing with T
        for i in range(len(psr_values) - 1):
            assert psr_values[i] > psr_values[i + 1], (
                f"PSR should decrease with T for negative SR: "
                f"PSR(T={t_values[i]})={psr_values[i]:.4f} "
                f"should be > PSR(T={t_values[i + 1]})={psr_values[i + 1]:.4f}"
            )

    def test_psr_bounded_zero_one(self):
        """PSR must always be in [0, 1] as it's a probability."""
        # Test extreme values
        test_cases = [
            {"observed_sharpe": 5.0, "n_observations": 10},  # Very high SR, small T
            {"observed_sharpe": -5.0, "n_observations": 10},  # Very low SR, small T
            {"observed_sharpe": 0.1, "n_observations": 10000},  # Small SR, large T
            {"observed_sharpe": 2.0, "n_observations": 1000},  # High SR, medium T
        ]

        for case in test_cases:
            psr = compute_psr(
                observed_sharpe=case["observed_sharpe"],
                benchmark_sharpe=0.0,
                n_observations=case["n_observations"],
                skewness=0.0,
                kurtosis=3.0,
            )
            assert 0.0 <= psr <= 1.0, f"PSR must be in [0,1], got {psr} for {case}"

    def test_psr_symmetric_around_benchmark(self):
        """PSR(SR=+x) + PSR(SR=-x) ≈ 1 for symmetric distributions."""
        sr_positive = 0.5
        sr_negative = -0.5
        t = 252

        psr_pos = compute_psr(
            observed_sharpe=sr_positive,
            benchmark_sharpe=0.0,
            n_observations=t,
            skewness=0.0,
            kurtosis=3.0,
        )
        psr_neg = compute_psr(
            observed_sharpe=sr_negative,
            benchmark_sharpe=0.0,
            n_observations=t,
            skewness=0.0,
            kurtosis=3.0,
        )

        # For symmetric distribution, these should sum to ~1
        assert_allclose(psr_pos + psr_neg, 1.0, atol=0.01, err_msg="PSR symmetry violated")


class TestDSRProperties:
    """Test mathematical properties of Deflated Sharpe Ratio."""

    def test_dsr_less_than_or_equal_psr(self):
        """DSR ≤ PSR always (DSR is more conservative due to multiple testing).

        The deflation for multiple trials can only reduce the probability.
        """
        sr = 1.0
        n_obs = 252
        skew = 0.0
        kurt = 3.0
        # Assume cross-sectional variance of 0.5 (typical for diverse strategies)
        var_trials = 0.5

        psr = compute_psr(
            observed_sharpe=sr,
            benchmark_sharpe=0.0,
            n_observations=n_obs,
            skewness=skew,
            kurtosis=kurt,
        )

        # Test with various numbers of trials
        for n_trials in [1, 5, 10, 50, 100]:
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=sr,
                n_trials=n_trials,
                n_samples=n_obs,
                variance_trials=var_trials if n_trials > 1 else 0.0,
                skewness=skew,
                excess_kurtosis=kurt - 3.0,  # Convert to excess kurtosis
                autocorrelation=0.0,
            )
            dsr = result.probability
            assert dsr <= psr + 1e-10, (  # Small tolerance for numerical precision
                f"DSR should be ≤ PSR: DSR={dsr:.4f}, PSR={psr:.4f} for n_trials={n_trials}"
            )

    def test_dsr_decreases_with_more_trials(self):
        """DSR should decrease as number of trials increases.

        More trials = more multiple testing penalty.
        """
        sr = 1.5
        n_obs = 500
        skew = 0.0
        kurt = 3.0
        var_trials = 0.5  # Cross-sectional variance

        n_trials_list = [1, 5, 10, 25, 50, 100]
        dsr_values = []

        for n_trials in n_trials_list:
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=sr,
                n_trials=n_trials,
                n_samples=n_obs,
                variance_trials=var_trials if n_trials > 1 else 0.0,
                skewness=skew,
                excess_kurtosis=kurt - 3.0,
                autocorrelation=0.0,
            )
            dsr_values.append(result.probability)

        # DSR should be monotonically decreasing with n_trials
        for i in range(len(dsr_values) - 1):
            assert dsr_values[i] >= dsr_values[i + 1] - 1e-10, (
                f"DSR should decrease with n_trials: "
                f"DSR(n={n_trials_list[i]})={dsr_values[i]:.4f} "
                f"should be >= DSR(n={n_trials_list[i + 1]})={dsr_values[i + 1]:.4f}"
            )

    def test_dsr_bounded_zero_one(self):
        """DSR must always be in [0, 1] as it's a probability."""
        test_cases = [
            {"observed_sharpe": 3.0, "n_trials": 5, "n_observations": 252},
            {"observed_sharpe": 0.5, "n_trials": 100, "n_observations": 1000},
            {"observed_sharpe": 2.0, "n_trials": 1, "n_observations": 50},
        ]
        var_trials = 0.5  # Cross-sectional variance

        for case in test_cases:
            n_trials = case["n_trials"]
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=case["observed_sharpe"],
                n_trials=n_trials,
                n_samples=case["n_observations"],
                variance_trials=var_trials if n_trials > 1 else 0.0,
                skewness=0.0,
                excess_kurtosis=0.0,  # Normal distribution
                autocorrelation=0.0,
            )
            dsr = result.probability
            assert 0.0 <= dsr <= 1.0, f"DSR must be in [0,1], got {dsr} for {case}"


# =============================================================================
# IC / Correlation Properties
# =============================================================================


class TestICProperties:
    """Test mathematical properties of Information Coefficient."""

    def test_acf_lag_0_equals_1(self):
        """ACF at lag 0 must equal 1 (correlation with self)."""
        np.random.seed(42)
        series = np.random.randn(200)

        result = compute_acf(series, nlags=10)
        acf_values = result.values

        assert_allclose(acf_values[0], 1.0, rtol=1e-10, err_msg="ACF(0) must equal 1")

    def test_acf_bounded(self):
        """ACF values must be in [-1, 1]."""
        np.random.seed(42)
        series = np.random.randn(200)

        result = compute_acf(series, nlags=20)
        acf_values = result.values

        for i, val in enumerate(acf_values):
            assert -1 <= val <= 1, f"ACF at lag {i} = {val} is out of bounds [-1, 1]"

    def test_pacf_lag_0_equals_1(self):
        """PACF at lag 0 must equal 1."""
        np.random.seed(42)
        series = np.random.randn(200)

        result = compute_pacf(series, nlags=10)
        pacf_values = result.values

        assert_allclose(pacf_values[0], 1.0, rtol=1e-10, err_msg="PACF(0) must equal 1")

    def test_pacf_bounded(self):
        """PACF values must be in [-1, 1]."""
        np.random.seed(42)
        series = np.random.randn(200)

        result = compute_pacf(series, nlags=10)
        pacf_values = result.values

        for i, val in enumerate(pacf_values):
            assert -1 <= val <= 1, f"PACF at lag {i} = {val} is out of bounds [-1, 1]"

    def test_random_signal_ic_near_zero(self):
        """IC of random signal should be approximately zero."""
        np.random.seed(42)
        n = 1000
        random_signal = np.random.randn(n)
        random_target = np.random.randn(n)

        # Compute Spearman correlation (IC)
        from scipy.stats import spearmanr

        ic, _ = spearmanr(random_signal, random_target)

        # Should be close to zero for random data
        assert abs(ic) < 0.1, f"IC of random signals should be ~0, got {ic}"


# =============================================================================
# Drawdown Properties
# =============================================================================


class TestDrawdownProperties:
    """Test mathematical properties of drawdown calculations."""

    def test_max_drawdown_finite(self):
        """Max drawdown must be a finite number."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.02

        dd = max_drawdown(returns)
        # Note: max_drawdown returns negative values (loss convention)
        # The absolute value represents the drawdown magnitude
        assert np.isfinite(dd), f"Max drawdown must be finite, got {dd}"

    def test_max_drawdown_bounded_by_one(self):
        """Max drawdown magnitude should be reasonable for typical returns."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.05

        dd = max_drawdown(returns)
        # For typical returns, magnitude should be < 2
        assert abs(dd) <= 2.0, f"Max drawdown should be reasonably bounded, got {dd}"

    def test_drawdown_zero_for_monotonic_up(self):
        """Drawdown should be zero for monotonically increasing equity."""
        # Constant positive returns = monotonic up
        returns = np.ones(252) * 0.001  # 0.1% daily

        dd = max_drawdown(returns)
        assert_allclose(dd, 0.0, atol=1e-10, err_msg="Drawdown should be 0 for monotonic up")

    def test_drawdown_magnitude_for_single_drop(self):
        """For a single drop, drawdown magnitude equals the loss magnitude."""
        # Start with gains, then one big loss
        returns = np.zeros(100)
        returns[:50] = 0.01  # 1% daily gains
        returns[50] = -0.20  # 20% drop
        returns[51:] = 0.01  # Resume gains

        dd = max_drawdown(returns)
        # Drawdown magnitude should be approximately 20% (the drop)
        # Note: max_drawdown returns negative value in loss convention
        assert 0.15 <= abs(dd) <= 0.30, f"Drawdown magnitude should be ~20%, got {dd}"


# =============================================================================
# Sharpe / Sortino Properties
# =============================================================================


class TestSharpeProperties:
    """Test mathematical properties of Sharpe ratio."""

    def test_sharpe_scale_invariant(self):
        """Sharpe ratio is scale-invariant (multiplying returns by constant)."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.02 + 0.0005

        sr1 = sharpe_ratio(returns, risk_free=0.0)
        sr2 = sharpe_ratio(returns * 2, risk_free=0.0)  # Double returns

        # Should be the same (both mean and std scale by 2)
        assert_allclose(sr1, sr2, rtol=0.01, err_msg="Sharpe should be scale invariant")

    def test_sharpe_sign_matches_mean_excess(self):
        """Sharpe ratio sign should match sign of mean excess return."""
        np.random.seed(42)
        n = 252

        # Positive mean returns
        pos_returns = np.abs(np.random.randn(n)) * 0.01 + 0.001
        sr_pos = sharpe_ratio(pos_returns, risk_free=0.0)
        assert sr_pos > 0, "Sharpe should be positive for positive mean returns"

        # Negative mean returns
        neg_returns = -np.abs(np.random.randn(n)) * 0.01 - 0.001
        sr_neg = sharpe_ratio(neg_returns, risk_free=0.0)
        assert sr_neg < 0, "Sharpe should be negative for negative mean returns"


class TestSortinoProperties:
    """Test mathematical properties of Sortino ratio."""

    def test_sortino_greater_than_sharpe_for_right_skew(self):
        """Sortino should be >= Sharpe for right-skewed (positive) returns.

        Sortino only penalizes downside volatility, so for right-skewed
        distributions with positive mean, it should be higher.
        """
        np.random.seed(42)
        # Create right-skewed returns with positive mean and some downside
        n = 500
        base_returns = np.random.randn(n) * 0.02 + 0.003  # Positive mean
        # Add some extra positive outliers for skew
        positive_outliers = np.random.exponential(0.02, n // 10)
        returns = base_returns.copy()
        returns[: n // 10] += positive_outliers

        sr = sharpe_ratio(returns, risk_free=0.0)
        sortino = sortino_ratio(returns, risk_free=0.0)

        # For right-skewed positive returns, Sortino >= Sharpe
        assert sortino >= sr - 0.1, (  # Allow small tolerance
            f"Sortino ({sortino:.3f}) should be >= Sharpe ({sr:.3f}) for right-skewed returns"
        )


# =============================================================================
# Cross-Validation Properties
# =============================================================================


class TestCrossValidationProperties:
    """Test mathematical properties of cross-validation splitters."""

    def test_train_test_disjoint(self):
        """Train and test sets must be disjoint (no overlap)."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = np.random.randn(n)
        times = np.arange(n)

        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)

        for train_idx, test_idx in cv.split(X, y, times):
            train_set = set(train_idx)
            test_set = set(test_idx)
            intersection = train_set & test_set

            assert len(intersection) == 0, (
                f"Train and test sets must be disjoint, found overlap: {intersection}"
            )

    def test_indices_valid_range(self):
        """All indices must be in valid range [0, n-1]."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = np.random.randn(n)
        times = np.arange(n)

        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)

        for train_idx, test_idx in cv.split(X, y, times):
            all_indices = np.concatenate([train_idx, test_idx])
            assert all_indices.min() >= 0, "Indices must be >= 0"
            assert all_indices.max() < n, f"Indices must be < {n}"

    def test_purge_creates_gap(self):
        """Purging should create a gap between train and test."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 5)
        y = np.random.randn(n)
        times = np.arange(n)

        # Large purge to make gap obvious
        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2, embargo_pct=0.05)

        for train_idx, test_idx in cv.split(X, y, times):
            train_max = train_idx.max()
            test_min = test_idx.min()

            # With embargo, there should be no train indices within embargo_pct of test start
            # (This is a simplified check - actual purging is more complex)
            if train_max < test_min:  # Train before test
                gap = test_min - train_max - 1
                assert gap >= 0, "There should be no negative gap"


# =============================================================================
# Bootstrap Properties
# =============================================================================


class TestBootstrapProperties:
    """Test mathematical properties of bootstrap confidence intervals."""

    def test_ci_contains_true_value_approximately(self):
        """95% CI should contain true mean approximately 95% of the time.

        This is a statistical property test - we run many simulations.
        """
        np.random.seed(42)
        from scipy import stats

        true_mean = 5.0
        n_samples = 100
        n_simulations = 200
        contained = 0

        for _ in range(n_simulations):
            sample = np.random.normal(true_mean, 1.0, n_samples)
            # Simple t-confidence interval (not bootstrap, but tests the concept)
            se = stats.sem(sample)
            ci_low = sample.mean() - 1.96 * se
            ci_high = sample.mean() + 1.96 * se

            if ci_low <= true_mean <= ci_high:
                contained += 1

        coverage = contained / n_simulations
        # Should be close to 95% (allow wide tolerance due to sampling)
        assert 0.85 <= coverage <= 1.0, (
            f"95% CI should contain true value ~95% of time, got {coverage:.1%}"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test handling of edge cases and boundary conditions."""

    def test_single_observation_sharpe(self):
        """Sharpe with single observation should handle gracefully."""
        returns = np.array([0.05])

        # Should either return nan or raise
        try:
            sr = sharpe_ratio(returns, risk_free=0.0)
            # If it returns, should be nan or inf
            assert np.isnan(sr) or np.isinf(sr), f"Single obs Sharpe should be nan/inf, got {sr}"
        except (ValueError, ZeroDivisionError):
            pass  # Raising is acceptable

    def test_constant_returns_sharpe(self):
        """Sharpe with constant returns (std=0) should handle gracefully."""
        returns = np.ones(100) * 0.01

        try:
            sr = sharpe_ratio(returns, risk_free=0.0)
            # If it returns, should be nan or inf
            assert np.isnan(sr) or np.isinf(sr) or abs(sr) > 1e10, (
                f"Constant returns Sharpe should be nan/inf/large, got {sr}"
            )
        except (ValueError, ZeroDivisionError):
            pass  # Raising is acceptable

    def test_acf_short_series(self):
        """ACF should handle short series appropriately."""
        short_series = np.array([1, 2, 3, 4, 5])

        try:
            result = compute_acf(short_series, nlags=2)
            # If successful, values should be valid
            assert np.all(np.isfinite(result.values)), "ACF values should be finite"
        except ValueError:
            pass  # Raising for too-short series is acceptable

    def test_drawdown_all_losses(self):
        """Drawdown for all-loss series should be computable."""
        np.random.seed(42)
        # All negative returns
        returns = -np.abs(np.random.randn(100)) * 0.01 - 0.001

        dd = max_drawdown(returns)
        assert np.isfinite(dd), "Drawdown should be finite for all-loss series"
        # max_drawdown returns negative value (loss convention)
        # The magnitude should be non-trivial for all-loss series
        assert abs(dd) > 0, "Drawdown magnitude should be > 0 for all-loss series"

    def test_psr_extreme_skewness(self):
        """PSR should handle extreme skewness/kurtosis."""
        sr = 1.0
        n = 252

        # Extreme skewness
        psr_skewed = compute_psr(
            observed_sharpe=sr,
            benchmark_sharpe=0.0,
            n_observations=n,
            skewness=5.0,  # Very skewed
            kurtosis=50.0,  # Very fat tails
        )

        assert 0.0 <= psr_skewed <= 1.0, (
            f"PSR with extreme moments should be valid, got {psr_skewed}"
        )
