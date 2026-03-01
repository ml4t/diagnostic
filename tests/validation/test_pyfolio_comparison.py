"""Validation tests comparing ml4t-diagnostic portfolio metrics against empyrical.

This module validates that ml4t-diagnostic's portfolio metrics produce
results matching the industry-standard empyrical library within numerical
precision tolerances.

Reference: empyrical-reloaded (https://github.com/stefan-jansen/empyrical-reloaded)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Reference implementation (optional dependency for validation parity tests)
empyrical = pytest.importorskip(
    "empyrical",
    reason="empyrical is required for validation-against-empyrical tests",
)

# ML4T implementation
from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import (
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    conditional_var,
    information_ratio,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_returns():
    """Simple returns for basic validation."""
    np.random.seed(42)
    return np.random.randn(252) * 0.02  # 2% daily vol, 1 year


@pytest.fixture
def positive_returns():
    """Returns with positive drift (winning strategy)."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0005  # 1% vol, 0.05% daily return
    return returns


@pytest.fixture
def negative_returns():
    """Returns with negative drift (losing strategy)."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 - 0.0005
    return returns


@pytest.fixture
def volatile_returns():
    """High volatility returns."""
    np.random.seed(42)
    returns = np.random.randn(504) * 0.04  # 4% daily vol, 2 years
    return returns


@pytest.fixture
def fat_tailed_returns():
    """Returns from Student's t distribution (fat tails)."""
    np.random.seed(42)
    returns = np.random.standard_t(df=3, size=252) * 0.01
    return returns


@pytest.fixture
def trending_returns():
    """Strongly trending returns (for drawdown testing)."""
    np.random.seed(42)
    # Create a returns series with a significant drawdown
    returns = np.concatenate(
        [
            np.random.randn(100) * 0.01 + 0.002,  # Uptrend
            np.random.randn(50) * 0.02 - 0.01,  # Drawdown
            np.random.randn(102) * 0.01 + 0.001,  # Recovery
        ]
    )
    return returns


@pytest.fixture
def benchmark_returns():
    """Benchmark returns for relative metrics."""
    np.random.seed(123)
    return np.random.randn(252) * 0.015 + 0.0003


# =============================================================================
# Sharpe Ratio Tests
# =============================================================================


class TestSharpeRatioComparison:
    """Compare Sharpe ratio calculation against empyrical."""

    def test_sharpe_simple_returns(self, simple_returns):
        """Test Sharpe ratio with simple returns."""
        ml4t_sharpe = sharpe_ratio(simple_returns, risk_free=0.0, periods_per_year=252)
        emp_sharpe = empyrical.sharpe_ratio(simple_returns, risk_free=0.0, period="daily")

        assert_allclose(
            ml4t_sharpe, emp_sharpe, rtol=0.01, err_msg="Sharpe ratio differs from empyrical"
        )

    def test_sharpe_with_risk_free(self, positive_returns):
        """Test Sharpe ratio with non-zero risk-free rate."""
        rf_annual = 0.02  # 2% annual

        ml4t_sharpe = sharpe_ratio(positive_returns, risk_free=rf_annual, periods_per_year=252)

        # empyrical expects daily risk-free rate
        rf_daily = (1 + rf_annual) ** (1 / 252) - 1
        emp_sharpe = empyrical.sharpe_ratio(positive_returns, risk_free=rf_daily, period="daily")

        assert_allclose(
            ml4t_sharpe, emp_sharpe, rtol=0.01, err_msg="Sharpe ratio with risk-free differs"
        )

    def test_sharpe_negative_strategy(self, negative_returns):
        """Test Sharpe ratio for losing strategy."""
        ml4t_sharpe = sharpe_ratio(negative_returns, risk_free=0.0, periods_per_year=252)
        emp_sharpe = empyrical.sharpe_ratio(negative_returns, risk_free=0.0, period="daily")

        # Both should be negative
        assert ml4t_sharpe < 0
        assert emp_sharpe < 0
        assert_allclose(ml4t_sharpe, emp_sharpe, rtol=0.01)


# =============================================================================
# Sortino Ratio Tests
# =============================================================================


class TestSortinoRatioComparison:
    """Compare Sortino ratio calculation against empyrical."""

    def test_sortino_basic(self, simple_returns):
        """Test Sortino ratio basic calculation."""
        ml4t_sortino = sortino_ratio(simple_returns, risk_free=0.0, periods_per_year=252)
        emp_sortino = empyrical.sortino_ratio(simple_returns, required_return=0.0, period="daily")

        # Note: Sortino implementations may differ slightly in downside calculation
        # Allow 5% relative tolerance
        assert_allclose(
            ml4t_sortino,
            emp_sortino,
            rtol=0.05,
            err_msg="Sortino ratio differs significantly from empyrical",
        )

    def test_sortino_positive_strategy(self, positive_returns):
        """Test Sortino for positive returns - should be higher than Sharpe."""
        ml4t_sortino = sortino_ratio(positive_returns, risk_free=0.0, periods_per_year=252)
        ml4t_sharpe = sharpe_ratio(positive_returns, risk_free=0.0, periods_per_year=252)

        # For positive-mean returns, Sortino >= Sharpe (downside < total vol)
        assert ml4t_sortino >= ml4t_sharpe - 0.1  # Allow small margin


# =============================================================================
# Max Drawdown Tests
# =============================================================================


class TestMaxDrawdownComparison:
    """Compare max drawdown calculation against empyrical."""

    def test_max_drawdown_basic(self, simple_returns):
        """Test max drawdown calculation."""
        ml4t_dd = max_drawdown(simple_returns)
        emp_dd = empyrical.max_drawdown(simple_returns)

        assert_allclose(ml4t_dd, emp_dd, rtol=0.001, err_msg="Max drawdown differs from empyrical")

    def test_max_drawdown_trending(self, trending_returns):
        """Test max drawdown with clear trend changes."""
        ml4t_dd = max_drawdown(trending_returns)
        emp_dd = empyrical.max_drawdown(trending_returns)

        # Should both identify significant drawdown
        assert ml4t_dd < -0.05  # At least 5% drawdown
        assert_allclose(ml4t_dd, emp_dd, rtol=0.001)

    def test_max_drawdown_monotonic_up(self):
        """Test max drawdown for monotonically increasing equity."""
        returns = np.array([0.01] * 100)  # 1% daily return
        ml4t_dd = max_drawdown(returns)
        emp_dd = empyrical.max_drawdown(returns)

        # Should be very close to 0 (no drawdown)
        assert ml4t_dd > -0.001
        assert_allclose(ml4t_dd, emp_dd, atol=1e-10)


# =============================================================================
# Annual Return Tests
# =============================================================================


class TestAnnualReturnComparison:
    """Compare annual return (CAGR) calculation against empyrical."""

    def test_annual_return_basic(self, simple_returns):
        """Test annual return calculation."""
        ml4t_ret = annual_return(simple_returns, periods_per_year=252)
        emp_ret = empyrical.annual_return(simple_returns, period="daily")

        assert_allclose(
            ml4t_ret, emp_ret, rtol=0.01, err_msg="Annual return differs from empyrical"
        )

    def test_annual_return_multi_year(self, volatile_returns):
        """Test annual return over multiple years."""
        ml4t_ret = annual_return(volatile_returns, periods_per_year=252)
        emp_ret = empyrical.annual_return(volatile_returns, period="daily")

        assert_allclose(ml4t_ret, emp_ret, rtol=0.01)


# =============================================================================
# Annual Volatility Tests
# =============================================================================


class TestAnnualVolatilityComparison:
    """Compare annual volatility calculation against empyrical."""

    def test_annual_volatility_basic(self, simple_returns):
        """Test annual volatility calculation."""
        ml4t_vol = annual_volatility(simple_returns, periods_per_year=252)
        emp_vol = empyrical.annual_volatility(simple_returns, period="daily")

        assert_allclose(
            ml4t_vol, emp_vol, rtol=0.001, err_msg="Annual volatility differs from empyrical"
        )


# =============================================================================
# Calmar Ratio Tests
# =============================================================================


class TestCalmarRatioComparison:
    """Compare Calmar ratio calculation against empyrical."""

    def test_calmar_basic(self, trending_returns):
        """Test Calmar ratio calculation."""
        ml4t_calmar = calmar_ratio(trending_returns, periods_per_year=252)
        emp_calmar = empyrical.calmar_ratio(trending_returns, period="daily")

        # Calmar implementations can differ - allow 10% tolerance
        assert_allclose(
            ml4t_calmar,
            emp_calmar,
            rtol=0.1,
            err_msg="Calmar ratio differs significantly from empyrical",
        )


# =============================================================================
# Omega Ratio Tests
# =============================================================================


class TestOmegaRatioComparison:
    """Compare Omega ratio calculation against empyrical."""

    def test_omega_basic(self, simple_returns):
        """Test Omega ratio calculation."""
        ml4t_omega = omega_ratio(simple_returns, threshold=0.0)
        emp_omega = empyrical.omega_ratio(simple_returns, required_return=0.0)

        assert_allclose(
            ml4t_omega, emp_omega, rtol=0.001, err_msg="Omega ratio differs from empyrical"
        )

    def test_omega_with_threshold(self, positive_returns):
        """Test Omega ratio with non-zero threshold.

        Note: empyrical expects required_return as an ANNUAL rate and converts
        to daily. ml4t expects the threshold already in period terms (daily).
        Use annualization=1 to disable empyrical's conversion.
        """
        threshold = 0.001  # 0.1% daily

        ml4t_omega = omega_ratio(positive_returns, threshold=threshold)
        # Use annualization=1 to tell empyrical threshold is already in daily terms
        emp_omega = empyrical.omega_ratio(
            positive_returns, required_return=threshold, annualization=1
        )

        assert_allclose(ml4t_omega, emp_omega, rtol=0.01)


# =============================================================================
# Tail Ratio Tests
# =============================================================================


class TestTailRatioComparison:
    """Compare tail ratio calculation against empyrical."""

    def test_tail_ratio_basic(self, simple_returns):
        """Test tail ratio calculation."""
        ml4t_tail = tail_ratio(simple_returns)
        emp_tail = empyrical.tail_ratio(simple_returns)

        assert_allclose(
            ml4t_tail, emp_tail, rtol=0.001, err_msg="Tail ratio differs from empyrical"
        )

    def test_tail_ratio_fat_tails(self, fat_tailed_returns):
        """Test tail ratio with fat-tailed distribution."""
        ml4t_tail = tail_ratio(fat_tailed_returns)
        emp_tail = empyrical.tail_ratio(fat_tailed_returns)

        assert_allclose(ml4t_tail, emp_tail, rtol=0.01)


# =============================================================================
# VaR and CVaR Tests
# =============================================================================


class TestVaRComparison:
    """Compare Value at Risk calculation against empyrical."""

    def test_var_95(self, simple_returns):
        """Test 95% VaR calculation."""
        ml4t_var = value_at_risk(simple_returns, confidence=0.95)
        emp_var = empyrical.value_at_risk(simple_returns, cutoff=0.05)

        assert_allclose(ml4t_var, emp_var, rtol=0.001, err_msg="VaR differs from empyrical")

    def test_var_99(self, volatile_returns):
        """Test 99% VaR calculation."""
        ml4t_var = value_at_risk(volatile_returns, confidence=0.99)
        emp_var = empyrical.value_at_risk(volatile_returns, cutoff=0.01)

        assert_allclose(ml4t_var, emp_var, rtol=0.001)


class TestCVaRComparison:
    """Compare Conditional VaR calculation against empyrical."""

    def test_cvar_95(self, simple_returns):
        """Test 95% CVaR calculation."""
        ml4t_cvar = conditional_var(simple_returns, confidence=0.95)
        emp_cvar = empyrical.conditional_value_at_risk(simple_returns, cutoff=0.05)

        assert_allclose(ml4t_cvar, emp_cvar, rtol=0.001, err_msg="CVaR differs from empyrical")


# =============================================================================
# Alpha/Beta Tests
# =============================================================================


class TestAlphaBetaComparison:
    """Compare alpha/beta calculation against empyrical."""

    def test_alpha_beta_basic(self, simple_returns, benchmark_returns):
        """Test alpha and beta calculation."""
        ml4t_alpha, ml4t_beta = alpha_beta(
            simple_returns, benchmark_returns, risk_free=0.0, periods_per_year=252
        )

        emp_alpha = empyrical.alpha(
            simple_returns, benchmark_returns, risk_free=0.0, period="daily"
        )
        emp_beta = empyrical.beta(simple_returns, benchmark_returns, risk_free=0.0)

        assert_allclose(ml4t_beta, emp_beta, rtol=0.01, err_msg="Beta differs from empyrical")
        # Alpha may differ more due to annualization methods
        assert_allclose(
            ml4t_alpha, emp_alpha, rtol=0.1, err_msg="Alpha differs significantly from empyrical"
        )


# =============================================================================
# Information Ratio Tests
# =============================================================================


class TestInformationRatioComparison:
    """Compare information ratio calculation."""

    def test_information_ratio_basic(self, simple_returns, benchmark_returns):
        """Test information ratio calculation."""
        ml4t_ir = information_ratio(simple_returns, benchmark_returns, periods_per_year=252)

        # empyrical uses tracking error in calculation
        active_returns = simple_returns - benchmark_returns[: len(simple_returns)]
        emp_ir = empyrical.sharpe_ratio(active_returns, risk_free=0.0, period="daily")

        # IR and Sharpe of active returns should be close
        assert_allclose(
            ml4t_ir, emp_ir, rtol=0.1, err_msg="Information ratio differs from active Sharpe"
        )


# =============================================================================
# Mathematical Property Tests
# =============================================================================


class TestMathematicalProperties:
    """Test mathematical properties that must hold regardless of library."""

    def test_sharpe_scales_with_leverage(self):
        """Sharpe is scale-invariant (leverage doesn't change Sharpe).

        Multiplying returns by constant k scales both mean and std by k,
        so Sharpe = (k*mean)/(k*std) = mean/std remains unchanged.
        """
        np.random.seed(42)
        base_returns = np.random.randn(252) * 0.01
        leveraged_returns = base_returns * 2

        sharpe_base = sharpe_ratio(base_returns)
        sharpe_leveraged = sharpe_ratio(leveraged_returns)

        # Sharpe should be the same (scale-invariant)
        assert_allclose(sharpe_leveraged, sharpe_base, rtol=0.01)

    def test_sharpe_increases_with_mean_shift(self):
        """Adding constant to returns increases Sharpe (mean increases, vol unchanged)."""
        np.random.seed(42)
        base_returns = np.random.randn(252) * 0.01  # Mean ~0
        shifted_returns = base_returns + 0.001  # Mean ~0.001

        sharpe_base = sharpe_ratio(base_returns)
        sharpe_shifted = sharpe_ratio(shifted_returns)

        # Shifted should have higher Sharpe (higher mean, same vol)
        assert sharpe_shifted > sharpe_base

    def test_max_drawdown_bounds(self, simple_returns):
        """Max drawdown should be between -1 and 0."""
        dd = max_drawdown(simple_returns)
        assert -1 <= dd <= 0

    def test_var_less_than_cvar(self, simple_returns):
        """CVaR (expected shortfall) should be <= VaR."""
        var = value_at_risk(simple_returns, confidence=0.95)
        cvar = conditional_var(simple_returns, confidence=0.95)

        # CVaR is the average of tail, VaR is the cutoff
        assert cvar <= var

    def test_omega_ratio_one_at_mean(self, simple_returns):
        """Omega ratio at the mean return should be close to 1."""
        mean_return = np.mean(simple_returns)
        omega = omega_ratio(simple_returns, threshold=mean_return)

        # At mean, gains and losses should roughly balance
        assert 0.8 <= omega <= 1.2

    def test_sortino_greater_than_sharpe_for_positive_skew(self):
        """For positive-skew returns, Sortino >= Sharpe typically.

        Sortino only penalizes downside volatility, so for strategies
        with more upside than downside volatility, Sortino > Sharpe.
        """
        np.random.seed(42)
        # Create returns with positive mean and positive skew
        # (some downside required for valid Sortino calculation)
        returns = np.random.randn(252) * 0.01 + 0.002  # Positive mean

        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)

        # Both should be positive and finite
        assert sharpe > 0
        assert np.isfinite(sortino)
        # Sortino typically >= Sharpe for positive-mean returns
        # (downside vol <= total vol)
        assert sortino >= sharpe * 0.8  # Allow margin for sampling noise


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_return(self):
        """Test with single return value."""
        returns = np.array([0.01])

        # Should return NaN for metrics requiring multiple observations
        assert np.isnan(sharpe_ratio(returns))
        assert np.isnan(sortino_ratio(returns))

    def test_constant_returns(self):
        """Test with constant returns (zero volatility).

        For constant returns, std ≈ 0 (at machine epsilon), so Sharpe can be:
        - inf (if std exactly 0)
        - very large finite (if std near machine epsilon)
        - nan (if implementation explicitly handles this case)
        """
        returns = np.array([0.01] * 100)

        # Sharpe is effectively undefined when std ≈ 0
        sharpe = sharpe_ratio(returns)
        # Accept inf, nan, or very large absolute value
        assert np.isinf(sharpe) or np.isnan(sharpe) or np.abs(sharpe) > 1e10

    def test_all_negative_returns(self):
        """Test with all negative returns."""
        returns = np.array([-0.01] * 100)

        sharpe = sharpe_ratio(returns)
        assert sharpe < 0

        dd = max_drawdown(returns)
        assert dd < -0.5  # Significant drawdown

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        returns = np.array([0.01, 0.02, np.nan, -0.01, 0.03])

        # Should not raise, should use nanmean/nanstd
        sharpe = sharpe_ratio(returns)
        assert np.isfinite(sharpe)
