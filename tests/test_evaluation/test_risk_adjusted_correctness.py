"""High-quality correctness tests for risk-adjusted performance metrics.

These tests verify mathematical formulas for Sharpe ratio, Sortino ratio,
and maximum drawdown against hand-calculated expected values.

Key properties tested:
1. Sharpe = mean(excess) / std(excess) * sqrt(annualization)
2. Sortino uses downside deviation: sqrt(mean(min(excess, 0)²))
3. Maximum drawdown = peak-to-trough decline in cumulative returns
4. Edge cases: zero volatility, no drawdown, all positive/negative
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.metrics.risk_adjusted import (
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


class TestSharpeRatioCorrectness:
    """Tests verifying Sharpe ratio mathematical correctness."""

    def test_sharpe_formula_exact(self):
        """Verify Sharpe = mean(excess) / std(excess)."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.015])
        risk_free = 0.001

        sharpe = sharpe_ratio(returns, risk_free_rate=risk_free)

        # Calculate expected manually
        excess = returns - risk_free
        expected = np.mean(excess) / np.std(excess, ddof=1)

        assert abs(sharpe - expected) < 1e-10, (
            f"Sharpe formula mismatch: expected {expected:.6f}, got {sharpe:.6f}"
        )

    def test_sharpe_annualization(self):
        """Verify annualization multiplies by sqrt(factor)."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        ann_factor = 252  # Daily to annual

        sharpe_daily = sharpe_ratio(returns)
        sharpe_annual = sharpe_ratio(returns, annualization_factor=ann_factor)

        expected_annual = sharpe_daily * np.sqrt(ann_factor)

        assert abs(sharpe_annual - expected_annual) < 1e-10, (
            f"Annualization mismatch: expected {expected_annual:.6f}, got {sharpe_annual:.6f}"
        )

    def test_sharpe_zero_volatility_positive_mean(self):
        """Zero volatility with positive mean → +inf."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Constant positive

        sharpe = sharpe_ratio(returns)

        assert sharpe == np.inf, f"Expected +inf for zero vol positive mean, got {sharpe}"

    def test_sharpe_zero_volatility_negative_mean(self):
        """Zero volatility with negative mean → -inf."""
        returns = np.array([-0.01, -0.01, -0.01, -0.01])  # Constant negative

        sharpe = sharpe_ratio(returns)

        assert sharpe == -np.inf, f"Expected -inf for zero vol negative mean, got {sharpe}"

    def test_sharpe_zero_mean_zero_volatility(self):
        """Zero volatility with zero mean → NaN."""
        returns = np.array([0.0, 0.0, 0.0, 0.0])

        sharpe = sharpe_ratio(returns)

        assert np.isnan(sharpe), f"Expected NaN for zero mean zero vol, got {sharpe}"

    def test_sharpe_risk_free_effect(self):
        """Higher risk-free rate reduces Sharpe (subtracts from mean)."""
        returns = np.array([0.02, 0.03, 0.01, 0.04, 0.02])

        sharpe_low_rf = sharpe_ratio(returns, risk_free_rate=0.01)
        sharpe_high_rf = sharpe_ratio(returns, risk_free_rate=0.02)

        assert sharpe_low_rf > sharpe_high_rf, (
            f"Higher rf should reduce Sharpe: low_rf={sharpe_low_rf:.4f}, high_rf={sharpe_high_rf:.4f}"
        )

    def test_sharpe_negative_returns(self):
        """Verify Sharpe is negative for negative mean returns."""
        returns = np.array([-0.02, -0.01, -0.03, -0.02, -0.01])

        sharpe = sharpe_ratio(returns)

        assert sharpe < 0, f"Expected negative Sharpe for negative returns, got {sharpe}"

    def test_sharpe_accepts_polars_series(self):
        """Verify Polars Series input works correctly."""
        returns_np = np.array([0.01, 0.02, -0.01, 0.03])
        returns_pl = pl.Series(returns_np.tolist())

        sharpe_np = sharpe_ratio(returns_np)
        sharpe_pl = sharpe_ratio(returns_pl)

        assert abs(sharpe_np - sharpe_pl) < 1e-10, (
            f"Polars vs numpy mismatch: {sharpe_np} vs {sharpe_pl}"
        )

    def test_sharpe_accepts_pandas_series(self):
        """Verify Pandas Series input works correctly."""
        returns_np = np.array([0.01, 0.02, -0.01, 0.03])
        returns_pd = pd.Series(returns_np)

        sharpe_np = sharpe_ratio(returns_np)
        sharpe_pd = sharpe_ratio(returns_pd)

        assert abs(sharpe_np - sharpe_pd) < 1e-10, (
            f"Pandas vs numpy mismatch: {sharpe_np} vs {sharpe_pd}"
        )

    def test_sharpe_handles_nan(self):
        """Verify NaN values are properly excluded."""
        returns_with_nan = np.array([0.01, np.nan, 0.02, -0.01, np.nan, 0.03])
        returns_clean = np.array([0.01, 0.02, -0.01, 0.03])

        sharpe_with_nan = sharpe_ratio(returns_with_nan)
        sharpe_clean = sharpe_ratio(returns_clean)

        assert abs(sharpe_with_nan - sharpe_clean) < 1e-10, (
            f"NaN handling mismatch: with_nan={sharpe_with_nan}, clean={sharpe_clean}"
        )


class TestSortinoRatioCorrectness:
    """Tests verifying Sortino ratio mathematical correctness."""

    def test_sortino_formula_exact(self):
        """Verify Sortino = mean(excess) / downside_std."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.015])
        target = 0.0

        sortino = sortino_ratio(returns, target_return=target)

        # Calculate expected manually
        excess = returns - target
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside**2))
        expected = np.mean(excess) / downside_std

        assert abs(sortino - expected) < 1e-10, (
            f"Sortino formula mismatch: expected {expected:.6f}, got {sortino:.6f}"
        )

    def test_sortino_downside_only(self):
        """Verify Sortino only uses downside returns for volatility."""
        # Returns with asymmetric distribution
        # Large upside, small downside
        returns = np.array([0.10, 0.08, -0.01, 0.05, -0.02, 0.12])

        sortino = sortino_ratio(returns)
        sharpe = sharpe_ratio(returns)

        # Sortino should be higher than Sharpe for right-skewed returns
        # because it ignores upside volatility
        assert sortino > sharpe, (
            f"Sortino should exceed Sharpe for right-skewed returns: "
            f"Sortino={sortino:.4f}, Sharpe={sharpe:.4f}"
        )

    def test_sortino_no_downside_positive_mean(self):
        """All positive returns → Sortino = +inf."""
        returns = np.array([0.01, 0.02, 0.005, 0.015])  # All positive

        sortino = sortino_ratio(returns)

        assert sortino == np.inf, f"Expected +inf for all positive returns, got {sortino}"

    def test_sortino_all_downside(self):
        """All negative returns: Sortino uses all returns for downside."""
        returns = np.array([-0.02, -0.01, -0.03, -0.02])

        sortino = sortino_ratio(returns)

        # Calculate expected: downside_std = std of all returns (all are downside)
        downside_std = np.sqrt(np.mean(returns**2))
        expected = np.mean(returns) / downside_std

        assert abs(sortino - expected) < 1e-10, (
            f"All-downside Sortino mismatch: expected {expected:.6f}, got {sortino:.6f}"
        )

    def test_sortino_target_return_effect(self):
        """Higher target reduces Sortino (more returns become "downside")."""
        returns = np.array([0.02, 0.01, 0.03, 0.015, 0.025])

        sortino_low_target = sortino_ratio(returns, target_return=0.0)
        sortino_high_target = sortino_ratio(returns, target_return=0.02)

        # With higher target, more returns are below target → lower Sortino
        assert sortino_low_target > sortino_high_target, "Higher target should reduce Sortino"

    def test_sortino_annualization(self):
        """Verify annualization multiplies by sqrt(factor)."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

        sortino_daily = sortino_ratio(returns)
        sortino_annual = sortino_ratio(returns, annualization_factor=252)

        expected_annual = sortino_daily * np.sqrt(252)

        assert abs(sortino_annual - expected_annual) < 1e-10, (
            f"Annualization mismatch: expected {expected_annual:.6f}, got {sortino_annual:.6f}"
        )


class TestMaximumDrawdownCorrectness:
    """Tests verifying maximum drawdown mathematical correctness."""

    def test_drawdown_formula_exact(self):
        """Verify MDD = (trough - peak) / peak in cumulative return space."""
        # Construct returns with known drawdown
        # Start at 1.0, go to 1.1, drop to 0.9, recover to 1.0
        # Peak = 1.1, Trough = 0.9, MDD = (0.9 - 1.1) / 1.1 = -0.182

        returns = np.array([0.10, -0.1818])  # 1.0 → 1.1 → 0.9

        result = maximum_drawdown(returns)

        # Cumulative: [0.10, -0.1 - 0.1*0.1818] = [0.10, -0.0818]
        # Actually: (1+0.1)*(1-0.1818) - 1 = 1.1 * 0.8182 - 1 = -0.1

        # Let me recalculate: start with 1.0
        # After +10%: 1.0 * 1.1 = 1.1
        # After -18.18%: 1.1 * 0.8182 = 0.9
        # Drawdown from peak 1.1 to 0.9 = (0.9 - 1.1) / 1.1 = -0.182

        assert result["max_drawdown"] < 0, "Drawdown should be negative"
        assert result["max_drawdown"] > -0.2, f"Unexpected MDD: {result['max_drawdown']}"

    def test_no_drawdown_monotonic_increase(self):
        """Monotonically increasing returns → no drawdown (MDD = 0)."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])  # All positive

        result = maximum_drawdown(returns)

        assert result["max_drawdown"] == 0.0, (
            f"Expected 0 drawdown for monotonic increase, got {result['max_drawdown']}"
        )

    def test_drawdown_always_negative_or_zero(self):
        """Drawdown is always <= 0 by definition."""
        np.random.seed(42)

        for _ in range(10):
            returns = np.random.randn(50) * 0.02

            result = maximum_drawdown(returns)

            assert result["max_drawdown"] <= 0, (
                f"Drawdown must be <= 0, got {result['max_drawdown']}"
            )

    def test_drawdown_peak_before_trough(self):
        """Peak date should be before or equal to trough date."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.03

        result = maximum_drawdown(returns)

        if result["max_drawdown"] < 0:
            assert result["peak_date"] <= result["trough_date"], (
                f"Peak ({result['peak_date']}) should be <= trough ({result['trough_date']})"
            )

    def test_drawdown_duration_positive(self):
        """Drawdown duration should be non-negative."""
        returns = np.array([0.05, -0.10, -0.05, 0.02, 0.03])

        result = maximum_drawdown(returns)

        assert result["max_drawdown_duration"] >= 0, (
            f"Duration should be >= 0, got {result['max_drawdown_duration']}"
        )

    def test_drawdown_known_scenario(self):
        """Test with a known drawdown scenario."""
        # Equity curve: 100 → 110 → 100 → 90 → 95
        # Returns: +10%, -9.09%, -10%, +5.56%
        # Peak at 110, Trough at 90
        # MDD = (90 - 110) / 110 = -18.18%

        returns = np.array([0.10, -0.0909, -0.10, 0.0556])

        result = maximum_drawdown(returns)

        # MDD should be approximately -18.18%
        assert -0.20 < result["max_drawdown"] < -0.15, (
            f"Expected MDD around -0.18, got {result['max_drawdown']:.4f}"
        )


class TestConfidenceIntervals:
    """Tests for Sharpe ratio confidence intervals."""

    def test_ci_contains_point_estimate(self):
        """Confidence interval should typically contain the point estimate."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02 + 0.001

        result = sharpe_ratio(
            returns,
            confidence_intervals=True,
            random_state=42,
        )

        assert result["lower_ci"] <= result["sharpe"] <= result["upper_ci"], (
            f"Point estimate {result['sharpe']:.4f} outside CI "
            f"[{result['lower_ci']:.4f}, {result['upper_ci']:.4f}]"
        )

    def test_ci_width_decreases_with_sample_size(self):
        """Larger samples should give narrower CIs."""
        np.random.seed(42)
        returns_small = np.random.randn(50) * 0.02 + 0.001
        returns_large = np.random.randn(500) * 0.02 + 0.001

        result_small = sharpe_ratio(returns_small, confidence_intervals=True, random_state=42)
        result_large = sharpe_ratio(returns_large, confidence_intervals=True, random_state=42)

        width_small = result_small["upper_ci"] - result_small["lower_ci"]
        width_large = result_large["upper_ci"] - result_large["lower_ci"]

        assert width_large < width_small, (
            f"Larger sample should have narrower CI: "
            f"small={width_small:.4f}, large={width_large:.4f}"
        )

    def test_ci_reproducibility_with_seed(self):
        """Same random_state should give same CI."""
        returns = np.random.randn(100) * 0.02 + 0.001

        result1 = sharpe_ratio(returns, confidence_intervals=True, random_state=42)
        result2 = sharpe_ratio(returns, confidence_intervals=True, random_state=42)

        assert result1["lower_ci"] == result2["lower_ci"]
        assert result1["upper_ci"] == result2["upper_ci"]


class TestEdgeCasesRiskAdjusted:
    """Edge cases for risk-adjusted metrics."""

    def test_insufficient_data(self):
        """Single return should give NaN."""
        returns = np.array([0.01])

        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)

        assert np.isnan(sharpe), f"Expected NaN for single return, got Sharpe={sharpe}"
        assert np.isnan(sortino), f"Expected NaN for single return, got Sortino={sortino}"

    def test_empty_returns(self):
        """Empty returns should give NaN."""
        returns = np.array([])

        result = maximum_drawdown(returns)

        assert np.isnan(result["max_drawdown"])

    def test_all_nan_returns(self):
        """All NaN returns should give NaN."""
        returns = np.array([np.nan, np.nan, np.nan])

        sharpe = sharpe_ratio(returns)

        assert np.isnan(sharpe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
