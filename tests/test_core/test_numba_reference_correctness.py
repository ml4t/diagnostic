"""Reference correctness tests for Numba-optimized functions.

These tests verify Numba implementations match reference implementations
from scipy/numpy, ensuring mathematical correctness is not sacrificed
for performance.

Key verifications:
1. Drawdown matches vectorized numpy implementation
2. Rolling Sharpe matches manual calculation
3. IC matches scipy.stats correlations
4. Ranking matches scipy.stats.rankdata
5. Purge/embargo indices are mathematically correct
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats

from ml4t.diagnostic.core.numba_utils import (
    _rank_data_numba,
    calculate_drawdown_numba,
    calculate_ic_vectorized,
    embargo_indices_numba,
    purge_indices_numba,
    rolling_sharpe_numba,
)


class TestDrawdownReferenceCorrectness:
    """Verify Numba drawdown matches reference implementation."""

    def _reference_drawdown(self, cum_returns: np.ndarray) -> tuple[float, int, int, int]:
        """Reference implementation using pure numpy."""
        if len(cum_returns) == 0:
            return np.nan, -1, -1, -1

        # Wealth, including the 1.0 that precedes the first return, so a first-period loss is
        # a drawdown rather than a new peak.
        wealth = 1.0 + cum_returns
        running_max = np.maximum.accumulate(np.maximum(wealth, 1.0))

        # Fractional drawdown at each point. This used to be `cum_returns - running_max`, the
        # same absolute subtraction the kernel made, so the test agreed with the bug instead of
        # catching it.
        drawdowns = wealth / running_max - 1.0

        # Find max drawdown
        max_dd = drawdowns.min()

        if max_dd >= 0:
            return 0.0, 0, 0, 0

        # Find trough (where max drawdown occurred)
        trough_idx = int(np.argmin(drawdowns))

        # Find peak (running max at trough point)
        peak_idx = int(np.argmax(np.maximum(wealth, 1.0)[: trough_idx + 1]))

        # Duration
        duration = trough_idx - peak_idx

        return float(max_dd), duration, peak_idx, trough_idx

    def test_matches_reference_simple(self):
        """Simple case should match reference."""
        cum_returns = np.array([0.0, 0.1, 0.05, 0.15, 0.08, 0.12])

        numba_dd, numba_dur, numba_peak, numba_trough = calculate_drawdown_numba(cum_returns)
        ref_dd, ref_dur, ref_peak, ref_trough = self._reference_drawdown(cum_returns)

        assert numba_dd == pytest.approx(ref_dd, rel=1e-10)
        assert numba_peak == ref_peak
        assert numba_trough == ref_trough

    def test_matches_reference_random(self):
        """Random data should match reference."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02
        cum_returns = np.cumprod(1 + returns) - 1

        numba_dd, numba_dur, numba_peak, numba_trough = calculate_drawdown_numba(cum_returns)
        ref_dd, ref_dur, ref_peak, ref_trough = self._reference_drawdown(cum_returns)

        assert numba_dd == pytest.approx(ref_dd, rel=1e-10)
        assert numba_peak == ref_peak
        assert numba_trough == ref_trough

    def test_matches_reference_monotonic_increase(self):
        """Monotonic increase: no drawdown."""
        cum_returns = np.cumsum(np.abs(np.random.randn(100) * 0.01))

        numba_dd, _, _, _ = calculate_drawdown_numba(cum_returns)
        ref_dd, _, _, _ = self._reference_drawdown(cum_returns)

        assert numba_dd == ref_dd == 0.0

    def test_drawdown_formula_exact(self):
        """Verify drawdown = trough_wealth / peak_wealth - 1.

        This asserted `trough_value - peak_value`, which is the defect it was meant to guard:
        an absolute distance below the high-water mark rather than a fractional decline.
        """
        cum_returns = np.array([0.0, 0.1, 0.05, 0.15, 0.08, 0.2])

        max_dd, duration, peak_idx, trough_idx = calculate_drawdown_numba(cum_returns)

        expected_dd = (1.0 + cum_returns[trough_idx]) / (1.0 + cum_returns[peak_idx]) - 1.0
        assert max_dd == pytest.approx(expected_dd, rel=1e-10)
        # 1.08 / 1.15 - 1, stated as a number so this cannot drift with the implementation.
        assert max_dd == pytest.approx(-0.0608695652, abs=1e-9)


class TestRollingSharpeReferenceCorrectness:
    """Verify Numba rolling Sharpe matches reference implementation."""

    def _reference_rolling_sharpe(
        self,
        returns: np.ndarray,
        window: int,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Reference implementation using numpy."""
        n = len(returns)
        if n < window:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)
        daily_rf = risk_free_rate / periods_per_year
        sqrt_periods = np.sqrt(periods_per_year)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1 : i + 1]
            excess_returns = window_returns - daily_rf

            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns)

            if std_excess > 0:
                result[i] = mean_excess / std_excess * sqrt_periods
            elif abs(mean_excess) < 1e-10:
                result[i] = 0.0
            else:
                result[i] = np.nan

        return result

    def test_matches_reference_basic(self):
        """Basic case should match reference."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        numba_sharpe = rolling_sharpe_numba(returns, 20)
        ref_sharpe = self._reference_rolling_sharpe(returns, 20)

        np.testing.assert_allclose(numba_sharpe, ref_sharpe, rtol=1e-10)

    def test_matches_reference_with_risk_free(self):
        """With risk-free rate should match reference."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.0005  # Slight positive drift

        numba_sharpe = rolling_sharpe_numba(returns, 20, risk_free_rate=0.02)
        ref_sharpe = self._reference_rolling_sharpe(returns, 20, risk_free_rate=0.02)

        np.testing.assert_allclose(numba_sharpe, ref_sharpe, rtol=1e-10)

    def test_sharpe_formula_exact(self):
        """Verify Sharpe = mean(excess) / std(excess) * sqrt(periods)."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        window = 5
        risk_free_rate = 0.0
        periods_per_year = 252

        result = rolling_sharpe_numba(returns, window, risk_free_rate, periods_per_year)

        # Manual calculation for last point
        excess = returns - (risk_free_rate / periods_per_year)
        expected = np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)

        assert result[-1] == pytest.approx(expected, rel=1e-6)


class TestICReferenceCorrectness:
    """Verify Numba IC matches scipy correlation functions."""

    def test_pearson_matches_scipy(self):
        """Pearson IC should match scipy.stats.pearsonr."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = predictions * 0.5 + np.random.randn(100) * 0.5

        numba_ic = calculate_ic_vectorized(predictions, returns, method=0)
        scipy_ic, _ = scipy_stats.pearsonr(predictions, returns)

        assert numba_ic == pytest.approx(scipy_ic, rel=1e-6)

    def test_spearman_matches_scipy(self):
        """Spearman IC should match scipy.stats.spearmanr."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = predictions * 0.5 + np.random.randn(100) * 0.5

        numba_ic = calculate_ic_vectorized(predictions, returns, method=1)
        scipy_ic, _ = scipy_stats.spearmanr(predictions, returns)

        assert numba_ic == pytest.approx(scipy_ic, rel=1e-6)

    def test_pearson_perfect_correlation(self):
        """Perfect correlation should return exactly 1.0."""
        predictions = np.arange(10, dtype=float)
        returns = predictions * 2 + 5

        ic = calculate_ic_vectorized(predictions, returns, method=0)

        assert ic == pytest.approx(1.0, rel=1e-10)

    def test_pearson_formula_exact(self):
        """Verify Pearson formula: cov(X,Y) / (std(X) * std(Y))."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([2.0, 4.0, 5.0, 4.0, 5.0])

        ic = calculate_ic_vectorized(predictions, returns, method=0)

        # Manual calculation
        pred_mean = np.mean(predictions)
        ret_mean = np.mean(returns)
        numerator = np.sum((predictions - pred_mean) * (returns - ret_mean))
        denominator = np.sqrt(
            np.sum((predictions - pred_mean) ** 2) * np.sum((returns - ret_mean) ** 2)
        )
        expected = numerator / denominator

        assert ic == pytest.approx(expected, rel=1e-10)

    def test_spearman_monotonic_relationship(self):
        """Spearman should be 1.0 for any monotonic relationship."""
        predictions = np.arange(10, dtype=float)
        returns = predictions**3  # Non-linear but monotonic

        ic = calculate_ic_vectorized(predictions, returns, method=1)

        assert ic == pytest.approx(1.0, rel=1e-6)


class TestRankDataReferenceCorrectness:
    """Verify Numba ranking matches scipy.stats.rankdata."""

    def test_matches_scipy_no_ties(self):
        """Without ties should match scipy."""
        data = np.array([3.0, 1.0, 4.0, 1.5, 5.0])

        numba_ranks = _rank_data_numba(data)
        scipy_ranks = scipy_stats.rankdata(data, method="average")

        np.testing.assert_allclose(numba_ranks, scipy_ranks, rtol=1e-10)

    def test_matches_scipy_with_ties(self):
        """With ties should match scipy (average method)."""
        data = np.array([1.0, 2.0, 2.0, 3.0, 2.0])

        numba_ranks = _rank_data_numba(data)
        scipy_ranks = scipy_stats.rankdata(data, method="average")

        np.testing.assert_allclose(numba_ranks, scipy_ranks, rtol=1e-10)

    def test_matches_scipy_all_same(self):
        """All same values should match scipy."""
        data = np.array([5.0, 5.0, 5.0, 5.0])

        numba_ranks = _rank_data_numba(data)
        scipy_ranks = scipy_stats.rankdata(data, method="average")

        np.testing.assert_allclose(numba_ranks, scipy_ranks, rtol=1e-10)

    def test_rank_properties(self):
        """Verify ranking properties: sum of ranks = n*(n+1)/2."""
        data = np.random.randn(50)

        ranks = _rank_data_numba(data)
        n = len(data)

        expected_sum = n * (n + 1) / 2
        assert np.sum(ranks) == pytest.approx(expected_sum, rel=1e-10)

    def test_rank_preserves_order(self):
        """Higher values should have higher ranks."""
        data = np.array([10.0, 20.0, 30.0, 40.0])

        ranks = _rank_data_numba(data)

        # Ranks should be in order for sorted data
        np.testing.assert_array_equal(ranks, [1.0, 2.0, 3.0, 4.0])


class TestPurgeIndicesCorrectness:
    """Verify purge indices are mathematically correct."""

    def test_purge_indices_formula(self):
        """Purge indices should be [max(0, test_start-horizon), test_start)."""
        test_start = 50
        test_end = 60
        label_horizon = 10
        n_samples = 100

        indices = purge_indices_numba(test_start, test_end, label_horizon, n_samples)

        expected_start = max(0, test_start - label_horizon)
        expected_end = test_start
        expected = np.arange(expected_start, expected_end)

        np.testing.assert_array_equal(indices, expected)

    def test_purge_at_boundary_start(self):
        """Purge should not go below index 0."""
        test_start = 5
        label_horizon = 10  # Would go to -5, but should stop at 0

        indices = purge_indices_numba(test_start, 10, label_horizon, 100)

        expected = np.arange(0, test_start)
        np.testing.assert_array_equal(indices, expected)

    def test_purge_empty_when_no_horizon(self):
        """Zero horizon should return empty purge."""
        indices = purge_indices_numba(50, 60, 0, 100)

        assert len(indices) == 0

    def test_purge_never_includes_test(self):
        """Purge indices should never include test indices."""
        test_start = 50
        test_end = 60

        for horizon in [5, 10, 20]:
            indices = purge_indices_numba(test_start, test_end, horizon, 100)

            # All purge indices should be < test_start
            if len(indices) > 0:
                assert np.all(indices < test_start)


class TestEmbargoIndicesCorrectness:
    """Verify embargo indices are mathematically correct."""

    def test_embargo_indices_formula(self):
        """Embargo indices should be [test_end, min(test_end+embargo, n))."""
        test_end = 60
        embargo_size = 10
        n_samples = 100

        indices = embargo_indices_numba(test_end, embargo_size, n_samples)

        expected_start = test_end
        expected_end = min(test_end + embargo_size, n_samples)
        expected = np.arange(expected_start, expected_end)

        np.testing.assert_array_equal(indices, expected)

    def test_embargo_at_boundary_end(self):
        """Embargo should not go beyond n_samples."""
        test_end = 95
        embargo_size = 10  # Would go to 105, but should stop at 100
        n_samples = 100

        indices = embargo_indices_numba(test_end, embargo_size, n_samples)

        expected = np.arange(95, 100)
        np.testing.assert_array_equal(indices, expected)

    def test_embargo_empty_when_test_at_end(self):
        """Embargo should be empty when test ends at n_samples."""
        indices = embargo_indices_numba(100, 10, 100)

        assert len(indices) == 0

    def test_embargo_empty_when_size_zero(self):
        """Zero embargo size should return empty."""
        indices = embargo_indices_numba(50, 0, 100)

        assert len(indices) == 0

    def test_embargo_never_includes_test(self):
        """Embargo indices should never include test indices."""
        test_end = 60

        for embargo in [5, 10, 20]:
            indices = embargo_indices_numba(test_end, embargo, 100)

            # All embargo indices should be >= test_end
            if len(indices) > 0:
                assert np.all(indices >= test_end)


class TestPurgeEmbargoTogether:
    """Test purge and embargo work correctly together."""

    def test_purge_embargo_no_overlap(self):
        """Purge and embargo should never overlap."""
        test_start = 40
        test_end = 60
        horizon = 10
        embargo = 5
        n_samples = 100

        purge = purge_indices_numba(test_start, test_end, horizon, n_samples)
        embargo_idx = embargo_indices_numba(test_end, embargo, n_samples)

        # No overlap
        if len(purge) > 0 and len(embargo_idx) > 0:
            assert purge.max() < embargo_idx.min()

    def test_purge_embargo_test_complete_separation(self):
        """Purge, test, and embargo should be completely separate."""
        test_start = 40
        test_end = 60
        horizon = 10
        embargo = 5
        n_samples = 100

        purge = set(purge_indices_numba(test_start, test_end, horizon, n_samples).tolist())
        test_indices = set(range(test_start, test_end))
        embargo_idx = set(embargo_indices_numba(test_end, embargo, n_samples).tolist())

        # All three should be disjoint
        assert purge.isdisjoint(test_indices)
        assert purge.isdisjoint(embargo_idx)
        assert test_indices.isdisjoint(embargo_idx)


class TestNumericalStability:
    """Test numerical stability of Numba functions."""

    def test_ic_with_near_zero_variance(self):
        """IC should handle near-zero variance gracefully."""
        predictions = np.array([1.0, 1.0 + 1e-15, 1.0, 1.0 + 1e-15, 1.0])
        returns = np.random.randn(5)

        ic = calculate_ic_vectorized(predictions, returns, method=0)

        # Should return 0 or a valid number, not inf/nan
        assert np.isfinite(ic) or ic == 0.0

    def test_drawdown_with_tiny_values(self):
        """Drawdown should handle tiny values correctly."""
        cum_returns = np.array([1e-15, 1e-14, 1e-15, 1e-13, 1e-15])

        max_dd, _, _, _ = calculate_drawdown_numba(cum_returns)

        assert np.isfinite(max_dd)

    def test_rolling_sharpe_with_tiny_std(self):
        """Rolling Sharpe should handle tiny std correctly."""
        returns = np.array([0.01, 0.01 + 1e-12, 0.01, 0.01 + 1e-12, 0.01])

        result = rolling_sharpe_numba(returns, 5)

        # Should be very high (large mean / tiny std) but finite
        # or handled as a special case
        assert np.isfinite(result[-1]) or result[-1] > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDrawdownAgainstKnownTruth:
    """Values computed by hand, so the kernel is checked against arithmetic rather than against
    a second implementation of itself. The reference in this module reimplemented the kernel's
    own `cum_returns - running_max` subtraction, so it agreed with the bug for as long as the
    bug existed."""

    def test_a_compounding_series_reports_a_fractional_drawdown(self):
        from ml4t.diagnostic.core.numba_utils import calculate_drawdown_numba

        # 24 gains of 10% take wealth to 1.1**24, then a 30% loss. The drawdown is the 30% loss,
        # whatever the level. The absolute subtraction reported -295%.
        returns = np.array([0.10] * 24 + [-0.30])
        cum = np.cumprod(1.0 + returns) - 1.0
        dd, _, peak, trough = calculate_drawdown_numba(cum)
        assert dd == pytest.approx(-0.30, abs=1e-12)
        assert (peak, trough) == (23, 24)

    def test_a_first_period_loss_is_a_drawdown(self):
        from ml4t.diagnostic.core.numba_utils import calculate_drawdown_numba

        # Wealth starts at 1.0, so a 50% first-period loss is a 50% drawdown. Seeding the peak
        # from cum_returns[0] made it the peak and reported 0.0.
        cum = np.cumprod(1.0 + np.array([-0.5, 0.1])) - 1.0
        dd, _, _, trough = calculate_drawdown_numba(cum)
        assert dd == pytest.approx(-0.5, abs=1e-12)
        assert trough == 0

    def test_the_public_metric_matches_its_own_docstring(self):
        from ml4t.diagnostic.metrics.risk_adjusted import maximum_drawdown

        # Wealth peaks at 1.1286 and troughs at 0.99317, one -12% return apart.
        returns = np.array([0.10, -0.05, 0.08, -0.12, 0.03])
        assert maximum_drawdown(returns)["max_drawdown"] == pytest.approx(-0.120, abs=1e-9)

    def test_a_monotonically_rising_series_has_no_drawdown(self):
        from ml4t.diagnostic.core.numba_utils import calculate_drawdown_numba

        cum = np.cumprod(1.0 + np.array([0.01] * 10)) - 1.0
        assert calculate_drawdown_numba(cum)[0] == 0.0
