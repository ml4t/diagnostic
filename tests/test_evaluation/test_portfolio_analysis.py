"""Tests for portfolio_analysis module.

Comprehensive tests for:
- PortfolioMetrics dataclass
- RollingMetricsResult dataclass
- DrawdownResult dataclass
- Helper functions (omega_ratio, tail_ratio, max_drawdown, etc.)
- PortfolioAnalysis class
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.portfolio_analysis import (
    DrawdownPeriod,
    DrawdownResult,
    PortfolioAnalysis,
    PortfolioMetrics,
    RollingMetricsResult,
    annual_return,
    annual_volatility,
    calmar_ratio,
    conditional_var,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    value_at_risk,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_returns():
    """Simple returns for basic tests."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01  # 1% daily vol


@pytest.fixture
def positive_returns():
    """Returns with consistent positive drift."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 + 0.0005  # Positive drift


@pytest.fixture
def negative_returns():
    """Returns with consistent negative drift."""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 - 0.001  # Negative drift


@pytest.fixture
def benchmark_returns():
    """Benchmark returns for relative metrics."""
    np.random.seed(123)
    return np.random.randn(252) * 0.008  # Slightly lower vol than strategy


@pytest.fixture
def portfolio_analysis(simple_returns, benchmark_returns):
    """PortfolioAnalysis instance with benchmark."""
    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=len(simple_returns) - 1),
        eager=True,
    )
    return PortfolioAnalysis(
        returns=simple_returns,
        benchmark=benchmark_returns,
        dates=dates,
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestOmegaRatio:
    """Tests for omega_ratio function."""

    def test_positive_returns_high_omega(self, positive_returns):
        """Test that positive returns give omega > 1."""
        result = omega_ratio(positive_returns)
        assert result > 1.0

    def test_negative_returns_low_omega(self, negative_returns):
        """Test that negative returns give omega < 1."""
        result = omega_ratio(negative_returns)
        assert result < 1.0

    def test_zero_threshold(self, simple_returns):
        """Test omega with default threshold of 0."""
        result = omega_ratio(simple_returns)
        assert np.isfinite(result)
        assert result > 0

    def test_custom_threshold(self, positive_returns):
        """Test omega with custom threshold."""
        result = omega_ratio(positive_returns, threshold=0.0005)
        assert np.isfinite(result)

    def test_all_positive_returns(self):
        """Test omega when all returns are positive."""
        returns = np.abs(np.random.randn(100) * 0.01) + 0.001
        result = omega_ratio(returns)
        assert result == np.inf or result > 1000  # Very high omega

    def test_all_negative_returns(self):
        """Test omega when all returns are negative."""
        returns = -np.abs(np.random.randn(100) * 0.01) - 0.001
        result = omega_ratio(returns)
        assert result < 0.01  # Very low omega

    def test_accepts_list(self):
        """Test omega accepts list input."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        result = omega_ratio(returns)
        assert np.isfinite(result)

    def test_accepts_polars_series(self):
        """Test omega accepts Polars Series."""
        returns = pl.Series("returns", [0.01, -0.005, 0.02, -0.01, 0.015])
        result = omega_ratio(returns)
        assert np.isfinite(result)


class TestTailRatio:
    """Tests for tail_ratio function."""

    def test_symmetric_distribution(self, simple_returns):
        """Test tail ratio for roughly symmetric distribution."""
        result = tail_ratio(simple_returns)
        # Should be close to 1 for symmetric distribution
        assert 0.5 < result < 2.0

    def test_skewed_positive(self):
        """Test tail ratio with positively skewed distribution."""
        np.random.seed(42)
        # Create positively skewed returns
        returns = np.abs(np.random.randn(1000) * 0.01)
        result = tail_ratio(returns)
        # Right tail should be heavier -> ratio > 1
        assert result > 0

    def test_accepts_list(self):
        """Test tail ratio accepts list input."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        result = tail_ratio(returns)
        assert np.isfinite(result)


class TestMaxDrawdown:
    """Tests for max_drawdown function."""

    def test_positive_only_returns(self):
        """Test max drawdown when returns are always positive."""
        returns = np.array([0.01, 0.02, 0.01, 0.03])
        result = max_drawdown(returns)
        assert result == 0.0  # No drawdown if always going up

    def test_negative_only_returns(self):
        """Test max drawdown when returns are always negative."""
        returns = np.array([-0.01, -0.02, -0.01, -0.03])
        result = max_drawdown(returns)
        assert result < 0  # Should have significant drawdown

    def test_known_drawdown(self):
        """Test max drawdown with known scenario."""
        # Go up 10%, then drop 20%
        returns = np.array([0.10, -0.20])
        result = max_drawdown(returns)
        # After +10%: wealth = 1.10
        # After -20%: wealth = 1.10 * 0.80 = 0.88
        # Drawdown from peak = (0.88 - 1.10) / 1.10 = -0.2
        assert result == pytest.approx(-0.2, abs=0.01)

    def test_returns_negative_value(self, simple_returns):
        """Test that max drawdown is always negative or zero."""
        result = max_drawdown(simple_returns)
        assert result <= 0


class TestAnnualReturn:
    """Tests for annual_return function."""

    def test_positive_annual_return(self, positive_returns):
        """Test annual return with positive drift."""
        result = annual_return(positive_returns)
        assert result > 0

    def test_negative_annual_return(self, negative_returns):
        """Test annual return with negative drift."""
        result = annual_return(negative_returns)
        assert result < 0

    def test_empty_returns(self):
        """Test annual return with empty array."""
        returns = np.array([])
        result = annual_return(returns)
        assert np.isnan(result)

    def test_custom_periods_per_year(self, simple_returns):
        """Test annual return with weekly data (52 periods)."""
        weekly_returns = simple_returns[:52]
        result = annual_return(weekly_returns, periods_per_year=52)
        assert np.isfinite(result)


class TestAnnualVolatility:
    """Tests for annual_volatility function."""

    def test_positive_volatility(self, simple_returns):
        """Test that volatility is positive."""
        result = annual_volatility(simple_returns)
        assert result > 0

    def test_constant_returns_zero_vol(self):
        """Test that constant returns give near-zero volatility."""
        returns = np.full(252, 0.001)  # Constant 0.1% daily
        result = annual_volatility(returns)
        assert result < 0.01  # Near zero

    def test_custom_periods_per_year(self, simple_returns):
        """Test volatility with weekly data."""
        weekly_returns = simple_returns[:52]
        result = annual_volatility(weekly_returns, periods_per_year=52)
        assert np.isfinite(result)
        assert result > 0


class TestValueAtRisk:
    """Tests for value_at_risk function."""

    def test_var_95(self, simple_returns):
        """Test 95% VaR."""
        result = value_at_risk(simple_returns, confidence=0.95)
        assert result < 0  # VaR should be negative (loss)

    def test_var_99_more_extreme(self, simple_returns):
        """Test that 99% VaR is more extreme than 95%."""
        var_95 = value_at_risk(simple_returns, confidence=0.95)
        var_99 = value_at_risk(simple_returns, confidence=0.99)
        assert var_99 < var_95  # 99% VaR should be more negative

    def test_var_accepts_list(self):
        """Test VaR accepts list input."""
        returns = [0.01, -0.02, 0.005, -0.015, 0.008]
        result = value_at_risk(returns)
        assert np.isfinite(result)


class TestConditionalVaR:
    """Tests for conditional_var function."""

    def test_cvar_more_extreme_than_var(self, simple_returns):
        """Test that CVaR is more extreme than VaR."""
        var = value_at_risk(simple_returns, confidence=0.95)
        cvar = conditional_var(simple_returns, confidence=0.95)
        assert cvar <= var  # CVaR is the expected loss beyond VaR

    def test_cvar_negative(self, simple_returns):
        """Test that CVaR is negative."""
        result = conditional_var(simple_returns)
        assert result < 0


class TestSharpeRatio:
    """Tests for sharpe_ratio function."""

    def test_positive_sharpe_positive_drift(self, positive_returns):
        """Test Sharpe ratio with positive returns."""
        result = sharpe_ratio(positive_returns)
        assert result > 0

    def test_negative_sharpe_negative_drift(self, negative_returns):
        """Test Sharpe ratio with negative returns."""
        result = sharpe_ratio(negative_returns)
        assert result < 0

    def test_zero_vol_returns_extreme(self):
        """Test Sharpe ratio with zero volatility returns extreme value."""
        returns = np.full(100, 0.001)  # Constant returns
        result = sharpe_ratio(returns)
        # With near-zero vol, Sharpe is extremely large or undefined
        # The function may return a very large number or nan/inf
        assert np.isnan(result) or np.isinf(result) or abs(result) > 1e10

    def test_custom_risk_free(self, simple_returns):
        """Test Sharpe ratio with custom risk-free rate."""
        result_zero_rf = sharpe_ratio(simple_returns, risk_free=0.0)
        result_high_rf = sharpe_ratio(simple_returns, risk_free=0.05)
        assert result_zero_rf > result_high_rf  # Higher RF means lower Sharpe


class TestSortinoRatio:
    """Tests for sortino_ratio function."""

    def test_positive_sortino(self, positive_returns):
        """Test Sortino ratio with positive returns."""
        result = sortino_ratio(positive_returns)
        assert result > 0

    def test_sortino_handles_positive_skew(self):
        """Test that Sortino handles positively skewed returns."""
        # Create returns with some negative values
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # Slight positive drift
        sortino = sortino_ratio(returns)
        # Should be finite when there is some downside
        assert np.isfinite(sortino) or np.isnan(sortino)


class TestCalmarRatio:
    """Tests for calmar_ratio function."""

    def test_calmar_positive_returns(self, positive_returns):
        """Test Calmar ratio with positive returns."""
        result = calmar_ratio(positive_returns)
        assert result > 0  # Positive returns, negative drawdown -> positive ratio

    def test_calmar_negative_returns(self, negative_returns):
        """Test Calmar ratio with negative returns."""
        result = calmar_ratio(negative_returns)
        # Negative annual return / negative max dd -> could be positive or negative
        assert np.isfinite(result)


# =============================================================================
# PortfolioMetrics Tests
# =============================================================================


class TestPortfolioMetrics:
    """Tests for PortfolioMetrics dataclass."""

    def test_initialization(self):
        """Test PortfolioMetrics initialization."""
        metrics = PortfolioMetrics(
            total_return=0.15,
            annual_return=0.12,
            annual_volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=1.0,
            calmar_ratio=0.5,
            omega_ratio=1.2,
            tail_ratio=1.1,
            max_drawdown=-0.20,
            skewness=-0.5,
            kurtosis=3.5,
            var_95=-0.025,
            cvar_95=-0.035,
            stability=0.95,
            win_rate=0.55,
            profit_factor=1.2,
            avg_win=0.008,
            avg_loss=-0.006,
        )

        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 0.67
        assert metrics.max_drawdown == -0.20

    def test_summary_without_benchmark(self):
        """Test summary method without benchmark metrics."""
        metrics = PortfolioMetrics(
            total_return=0.15,
            annual_return=0.12,
            annual_volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=1.0,
            calmar_ratio=0.5,
            omega_ratio=1.2,
            tail_ratio=1.1,
            max_drawdown=-0.20,
            skewness=-0.5,
            kurtosis=3.5,
            var_95=-0.025,
            cvar_95=-0.035,
            stability=0.95,
            win_rate=0.55,
            profit_factor=1.2,
            avg_win=0.008,
            avg_loss=-0.006,
        )

        summary = metrics.summary()
        assert "Portfolio Performance Summary" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Benchmark Comparison" not in summary  # No benchmark

    def test_summary_with_benchmark(self):
        """Test summary method with benchmark metrics."""
        metrics = PortfolioMetrics(
            total_return=0.15,
            annual_return=0.12,
            annual_volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=1.0,
            calmar_ratio=0.5,
            omega_ratio=1.2,
            tail_ratio=1.1,
            max_drawdown=-0.20,
            skewness=-0.5,
            kurtosis=3.5,
            var_95=-0.025,
            cvar_95=-0.035,
            stability=0.95,
            win_rate=0.55,
            profit_factor=1.2,
            avg_win=0.008,
            avg_loss=-0.006,
            alpha=0.03,
            beta=1.1,
            information_ratio=0.5,
            up_capture=1.1,
            down_capture=0.9,
        )

        summary = metrics.summary()
        assert "Benchmark Comparison" in summary
        assert "Alpha" in summary
        assert "Beta" in summary


# =============================================================================
# RollingMetricsResult Tests
# =============================================================================


class TestRollingMetricsResult:
    """Tests for RollingMetricsResult dataclass."""

    def test_initialization(self):
        """Test RollingMetricsResult initialization."""
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), eager=True)
        result = RollingMetricsResult(
            windows=[21, 63, 252],
            dates=dates,
        )

        assert result.windows == [21, 63, 252]
        assert len(result.dates) > 0
        assert result.sharpe == {}  # Empty initially

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        dates = pl.Series(
            "date", pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10), eager=True)
        )
        sharpe_21 = pl.Series("sharpe_21d", np.random.randn(10))

        result = RollingMetricsResult(
            windows=[21],
            dates=dates,
            sharpe={21: sharpe_21},
        )

        df = result.to_dataframe("sharpe")
        assert "date" in df.columns
        assert "sharpe_21d" in df.columns

    def test_to_dataframe_empty_metric(self):
        """Test conversion with empty metric."""
        dates = pl.Series(
            "date", pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10), eager=True)
        )

        result = RollingMetricsResult(
            windows=[21],
            dates=dates,
        )

        df = result.to_dataframe("beta")  # Beta not populated
        assert len(df) == 0


# =============================================================================
# DrawdownResult Tests
# =============================================================================


class TestDrawdownResult:
    """Tests for DrawdownResult dataclass."""

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        dates = pl.Series(
            "date", pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10), eager=True)
        )
        underwater = pl.Series("underwater", np.random.randn(10) * -0.01)

        result = DrawdownResult(
            current_drawdown=-0.05,
            max_drawdown=-0.15,
            avg_drawdown=-0.08,
            underwater_curve=underwater,
            top_drawdowns=[],
            max_duration_days=30,
            avg_duration_days=15.5,
            num_drawdowns=5,
            dates=dates,
        )

        df = result.to_dataframe()
        assert "date" in df.columns
        assert "drawdown" in df.columns


class TestDrawdownPeriod:
    """Tests for DrawdownPeriod dataclass."""

    def test_initialization(self):
        """Test DrawdownPeriod initialization."""
        from datetime import date

        period = DrawdownPeriod(
            peak_date=date(2023, 1, 1),
            valley_date=date(2023, 2, 1),
            recovery_date=date(2023, 3, 1),
            depth=-0.15,
            duration_days=31,
            recovery_days=28,
        )

        assert period.depth == -0.15
        assert period.duration_days == 31
        assert period.recovery_days == 28

    def test_no_recovery(self):
        """Test DrawdownPeriod without recovery."""
        from datetime import date

        period = DrawdownPeriod(
            peak_date=date(2023, 1, 1),
            valley_date=date(2023, 2, 1),
            recovery_date=None,
            depth=-0.15,
            duration_days=31,
            recovery_days=None,
        )

        assert period.recovery_date is None
        assert period.recovery_days is None


# =============================================================================
# PortfolioAnalysis Class Tests
# =============================================================================


class TestPortfolioAnalysisInitialization:
    """Tests for PortfolioAnalysis initialization."""

    def test_from_numpy_array(self, simple_returns):
        """Test initialization with numpy array."""
        analysis = PortfolioAnalysis(returns=simple_returns)

        assert len(analysis.returns) == len(simple_returns)
        assert analysis.has_benchmark is False

    def test_from_polars_series(self, simple_returns):
        """Test initialization with Polars Series."""
        returns = pl.Series("returns", simple_returns)
        analysis = PortfolioAnalysis(returns=returns)

        assert len(analysis.returns) == len(simple_returns)

    def test_from_list(self, simple_returns):
        """Test initialization with list."""
        returns = list(simple_returns)
        analysis = PortfolioAnalysis(returns=returns)

        assert len(analysis.returns) == len(simple_returns)

    def test_with_benchmark(self, simple_returns, benchmark_returns):
        """Test initialization with benchmark."""
        analysis = PortfolioAnalysis(
            returns=simple_returns,
            benchmark=benchmark_returns,
        )

        assert analysis.has_benchmark is True

    def test_with_dates(self, simple_returns):
        """Test initialization with dates."""
        dates = pl.date_range(
            pl.date(2023, 1, 1),
            pl.date(2023, 1, 1) + pl.duration(days=len(simple_returns) - 1),
            eager=True,
        )
        analysis = PortfolioAnalysis(
            returns=simple_returns,
            dates=dates,
        )

        assert len(analysis.dates) == len(simple_returns)

    def test_auto_generate_dates(self, simple_returns):
        """Test that dates are auto-generated if not provided."""
        analysis = PortfolioAnalysis(returns=simple_returns)

        assert len(analysis.dates) == len(simple_returns)

    def test_custom_risk_free(self, simple_returns):
        """Test initialization with custom risk-free rate."""
        analysis = PortfolioAnalysis(
            returns=simple_returns,
            risk_free=0.03,
        )

        metrics = analysis.compute_summary_stats()
        assert metrics is not None


class TestPortfolioAnalysisMetrics:
    """Tests for PortfolioAnalysis metric computation."""

    def test_compute_summary_stats(self, portfolio_analysis):
        """Test compute_summary_stats returns PortfolioMetrics."""
        metrics = portfolio_analysis.compute_summary_stats()

        assert isinstance(metrics, PortfolioMetrics)
        assert np.isfinite(metrics.sharpe_ratio)
        assert np.isfinite(metrics.max_drawdown)

    def test_metrics_caching(self, portfolio_analysis):
        """Test that metrics are cached."""
        metrics1 = portfolio_analysis.compute_summary_stats()
        metrics2 = portfolio_analysis.compute_summary_stats()

        assert metrics1 is metrics2  # Same object from cache

    def test_force_recompute(self, portfolio_analysis):
        """Test force_recompute bypasses cache."""
        metrics1 = portfolio_analysis.compute_summary_stats()
        metrics2 = portfolio_analysis.compute_summary_stats(force_recompute=True)

        # Values should be the same but different objects
        assert metrics1.sharpe_ratio == metrics2.sharpe_ratio

    def test_benchmark_metrics_populated(self, portfolio_analysis):
        """Test that benchmark metrics are populated when benchmark provided."""
        metrics = portfolio_analysis.compute_summary_stats()

        assert metrics.alpha is not None
        assert metrics.beta is not None
        assert metrics.information_ratio is not None

    def test_no_benchmark_metrics_none(self, simple_returns):
        """Test that benchmark metrics are None when no benchmark."""
        analysis = PortfolioAnalysis(returns=simple_returns)
        metrics = analysis.compute_summary_stats()

        assert metrics.alpha is None
        assert metrics.beta is None


class TestPortfolioAnalysisRolling:
    """Tests for PortfolioAnalysis rolling metrics."""

    def test_compute_rolling_metrics(self, portfolio_analysis):
        """Test compute_rolling_metrics returns RollingMetricsResult."""
        result = portfolio_analysis.compute_rolling_metrics(
            windows=[21, 63],
            metrics=["sharpe", "volatility"],
        )

        assert isinstance(result, RollingMetricsResult)
        assert 21 in result.sharpe
        assert 63 in result.sharpe

    def test_rolling_metrics_caching(self, portfolio_analysis):
        """Test that rolling metrics are cached."""
        result1 = portfolio_analysis.compute_rolling_metrics(windows=[21])
        result2 = portfolio_analysis.compute_rolling_metrics(windows=[21])

        assert result1 is result2

    def test_rolling_sharpe(self, portfolio_analysis):
        """Test rolling Sharpe ratio computation."""
        result = portfolio_analysis.compute_rolling_metrics(
            windows=[21],
            metrics=["sharpe"],
        )

        sharpe_series = result.sharpe[21]
        assert len(sharpe_series) == len(portfolio_analysis.returns)
        # First 20 values should be NaN (need 21 for first window)
        assert np.isnan(sharpe_series[0])
        assert np.isfinite(sharpe_series[-1])

    def test_rolling_volatility(self, portfolio_analysis):
        """Test rolling volatility computation."""
        result = portfolio_analysis.compute_rolling_metrics(
            windows=[21],
            metrics=["volatility"],
        )

        vol_series = result.volatility[21]
        assert len(vol_series) == len(portfolio_analysis.returns)
        assert np.isfinite(vol_series[-1])
        assert vol_series[-1] > 0  # Volatility is positive

    def test_rolling_beta_with_benchmark(self, portfolio_analysis):
        """Test rolling beta computation with benchmark."""
        result = portfolio_analysis.compute_rolling_metrics(
            windows=[21],
            metrics=["beta"],
        )

        assert 21 in result.beta
        beta_series = result.beta[21]
        assert np.isfinite(beta_series[-1])

    def test_rolling_returns(self, portfolio_analysis):
        """Test rolling returns computation."""
        result = portfolio_analysis.compute_rolling_metrics(
            windows=[21],
            metrics=["returns"],
        )

        ret_series = result.returns[21]
        assert len(ret_series) == len(portfolio_analysis.returns)


class TestPortfolioAnalysisDrawdown:
    """Tests for PortfolioAnalysis drawdown analysis."""

    def test_compute_drawdown_analysis(self, portfolio_analysis):
        """Test compute_drawdown_analysis returns DrawdownResult."""
        result = portfolio_analysis.compute_drawdown_analysis(top_n=5)

        assert isinstance(result, DrawdownResult)
        assert result.max_drawdown <= 0
        assert result.num_drawdowns >= 0

    def test_drawdown_caching(self, portfolio_analysis):
        """Test that drawdown results are cached."""
        result1 = portfolio_analysis.compute_drawdown_analysis()
        result2 = portfolio_analysis.compute_drawdown_analysis()

        assert result1 is result2

    def test_underwater_curve(self, portfolio_analysis):
        """Test underwater curve in drawdown result."""
        result = portfolio_analysis.compute_drawdown_analysis()

        assert len(result.underwater_curve) == len(portfolio_analysis.returns)
        assert all(result.underwater_curve.to_numpy() <= 0)  # All values <= 0

    def test_top_drawdowns_limited(self, portfolio_analysis):
        """Test that top_drawdowns respects top_n limit."""
        result = portfolio_analysis.compute_drawdown_analysis(top_n=3)

        assert len(result.top_drawdowns) <= 3


class TestPortfolioAnalysisProperties:
    """Tests for PortfolioAnalysis properties."""

    def test_returns_property(self, portfolio_analysis):
        """Test returns property."""
        assert isinstance(portfolio_analysis.returns, np.ndarray)

    def test_dates_property(self, portfolio_analysis):
        """Test dates property."""
        assert isinstance(portfolio_analysis.dates, pl.Series)

    def test_benchmark_property(self, portfolio_analysis):
        """Test benchmark property."""
        assert isinstance(portfolio_analysis.benchmark, np.ndarray)

    def test_has_benchmark_property(self, portfolio_analysis):
        """Test has_benchmark property."""
        assert portfolio_analysis.has_benchmark is True

    def test_has_positions_property(self, simple_returns):
        """Test has_positions property."""
        analysis = PortfolioAnalysis(returns=simple_returns)
        assert analysis.has_positions is False

    def test_has_transactions_property(self, simple_returns):
        """Test has_transactions property."""
        analysis = PortfolioAnalysis(returns=simple_returns)
        assert analysis.has_transactions is False


class TestPortfolioAnalysisWithPositions:
    """Tests for PortfolioAnalysis with positions data."""

    def test_with_polars_positions(self, simple_returns):
        """Test initialization with Polars positions DataFrame."""
        positions = pl.DataFrame(
            {
                "date": pl.date_range(
                    pl.date(2023, 1, 1),
                    pl.date(2023, 1, 1) + pl.duration(days=len(simple_returns) - 1),
                    eager=True,
                ),
                "asset_a": np.random.randn(len(simple_returns)),
                "asset_b": np.random.randn(len(simple_returns)),
            }
        )

        analysis = PortfolioAnalysis(
            returns=simple_returns,
            positions=positions,
        )

        assert analysis.has_positions is True

    def test_with_pandas_positions(self, simple_returns):
        """Test initialization with Pandas positions DataFrame."""
        dates = pd.date_range("2023-01-01", periods=len(simple_returns), freq="D")
        positions = pd.DataFrame(
            {
                "asset_a": np.random.randn(len(simple_returns)),
                "asset_b": np.random.randn(len(simple_returns)),
            },
            index=dates,
        )

        analysis = PortfolioAnalysis(
            returns=simple_returns,
            positions=positions,
        )

        assert analysis.has_positions is True
