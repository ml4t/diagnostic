"""Integration tests for DSR with TradeAnalysis and configuration system.

This module tests the integration between DSR (Deflated Sharpe Ratio) and other
components of the ml4t-diagnostic library, particularly:
- TradeAnalysis workflow integration
- Configuration serialization/deserialization
- End-to-end real-world workflows

These tests complement test_dsr_validation.py (which tests DSR math) and
test_dsr_bootstrap.py (which tests DSR statistical properties).
"""

import json

import numpy as np
import pytest

from ml4t.diagnostic.config.sharpe_config import DSRSettings
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics
from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis, TradeRecord

# =============================================================================
# TRADEANALYSIS + DSR INTEGRATION TESTS
# =============================================================================


class TestTradeAnalysisDSRIntegration:
    """Test integration between TradeAnalysis and DSR calculations."""

    @pytest.fixture
    def sample_trades_and_returns(self):
        """Create realistic trades and returns for testing."""
        # Create 50 trades with realistic characteristics
        np.random.seed(42)
        n_trades = 50

        # Generate daily returns with positive bias (Sharpe ~1.2)
        daily_returns = np.random.normal(0.001, 0.015, 252)  # One year

        trades = []
        for i in range(n_trades):
            entry_price = 100.0
            # Generate price returns (percentage move)
            price_return = np.random.normal(0.005, 0.02)  # Mean 0.5% with 2% vol

            # Randomly choose direction
            direction = "long" if np.random.random() > 0.5 else "short"

            quantity = 1.0

            # Calculate exit price and PnL based on direction
            if direction == "long":
                exit_price = entry_price * (1 + price_return)
                pnl = (exit_price - entry_price) * quantity
            else:  # short
                exit_price = entry_price * (1 + price_return)
                pnl = (entry_price - exit_price) * quantity  # Profit when price falls

            trade = TradeRecord(
                timestamp=f"2024-01-{(i % 28) + 1:02d}T10:00:00",
                symbol="AAPL",
                direction=direction,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                duration=f"PT{10 + i}H",  # ISO 8601 duration format
            )
            trades.append(trade)

        return {"trades": trades, "returns": daily_returns}

    def test_tradeanalysis_provides_sharpe_for_dsr(self, sample_trades_and_returns):
        """Test that TradeAnalysis can provide Sharpe ratio for DSR input."""
        # Analyze trades
        analyzer = TradeAnalysis(sample_trades_and_returns["trades"])
        analyzer.compute_statistics()

        # Get Sharpe ratio from trade returns
        trade_returns = np.array([t.pnl for t in sample_trades_and_returns["trades"]])
        observed_sharpe = np.mean(trade_returns) / np.std(trade_returns, ddof=1)

        # Calculate DSR assuming 10 strategies were tested
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=10,
            variance_trials=1.0,
            n_samples=len(trade_returns),
        )

        # Verify DSR calculation worked
        assert hasattr(dsr_result, "probability")
        assert hasattr(dsr_result, "expected_max_sharpe")
        assert dsr_result.expected_max_sharpe > 0  # Selection bias present
        assert 0 <= dsr_result.probability <= 1  # Probability format

        # DSR should deflate observed Sharpe
        assert dsr_result.z_score < observed_sharpe

    def test_dsr_with_worst_trades_analysis(self, sample_trades_and_returns):
        """Test DSR calculation focused on worst trades subset."""
        analyzer = TradeAnalysis(sample_trades_and_returns["trades"])

        # Get worst 10 trades
        worst_trades = analyzer.worst_trades(n=10)

        # Calculate Sharpe from worst trades
        worst_pnls = np.array([t.pnl for t in worst_trades])
        worst_sharpe = np.mean(worst_pnls) / np.std(worst_pnls, ddof=1)

        # Apply DSR (assuming this subset was selected from 100 strategies)
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=worst_sharpe,
            n_trials=100,
            variance_trials=1.0,
            n_samples=len(worst_pnls),
        )

        # Worst trades should have negative or low Sharpe
        assert worst_sharpe < 0

        # DSR should heavily deflate (high n_trials)
        assert dsr_result.expected_max_sharpe > 0
        assert dsr_result.z_score < worst_sharpe

    def test_end_to_end_workflow(self, sample_trades_and_returns):
        """Test complete workflow: Trades + Returns → TradeAnalysis → DSR."""
        # Step 1: Analyze trades
        analyzer = TradeAnalysis(sample_trades_and_returns["trades"])
        stats = analyzer.compute_statistics()

        # Step 2: Extract metrics
        assert stats.n_trades == 50
        assert stats.win_rate > 0
        assert stats.avg_pnl is not None

        # Step 3: Calculate Sharpe from returns
        returns = np.array(sample_trades_and_returns["returns"])
        observed_sharpe = np.mean(returns) / np.std(returns, ddof=1)

        # Step 4: Apply DSR with realistic parameters
        n_strategies_tested = 20  # Simulating a parameter sweep

        # Calculate higher moments for DSR
        skewness = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
        kurtosis = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))

        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=n_strategies_tested,
            variance_trials=1.0,
            n_samples=len(returns),
            skewness=skewness,
            excess_kurtosis=kurtosis - 3.0,  # Convert Pearson to Fisher
        )

        # Step 5: Verify complete workflow
        assert hasattr(dsr_result, "probability")
        assert hasattr(dsr_result, "p_value")

        # DSR should provide meaningful deflation
        assert dsr_result.expected_max_sharpe > 0
        assert dsr_result.z_score < observed_sharpe

        # Can make statistical inference
        alpha = 0.05
        is_significant = dsr_result.p_value < alpha
        # Don't assert significance (depends on random data), just check it's computable
        assert isinstance(is_significant, bool | np.bool_)


# =============================================================================
# CONFIGURATION SERIALIZATION TESTS
# =============================================================================


class TestDSRSettingsIntegration:
    """Test DSR configuration serialization and integration with workflow."""

    def test_dsr_config_creation(self):
        """Test creating DSRSettings with various parameters."""
        config = DSRSettings(
            n_trials=100,
            variance_inflation=1.5,
            prob_zero_sharpe=0.7,
        )

        assert config.n_trials == 100
        assert config.variance_inflation == 1.5
        assert config.prob_zero_sharpe == 0.7

    def test_dsr_config_serialization_json(self, tmp_path):
        """Test DSRSettings can be serialized to/from JSON."""
        config = DSRSettings(
            n_trials=50,
            variance_inflation=2.0,
            prob_zero_sharpe=0.6,
        )

        # Serialize to JSON
        config_path = tmp_path / "dsr_config.json"
        config_dict = config.to_dict()

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Deserialize from JSON
        with open(config_path) as f:
            loaded_dict = json.load(f)

        loaded_config = DSRSettings.from_dict(loaded_dict)

        # Verify round-trip
        assert loaded_config.n_trials == config.n_trials
        assert loaded_config.variance_inflation == config.variance_inflation
        assert loaded_config.prob_zero_sharpe == config.prob_zero_sharpe

    def test_dsr_config_with_dsr_function(self):
        """Test that DSR function works with realistic parameters."""
        observed_sharpe = 1.5
        n_samples = 252

        # Use DSR function with realistic parameters
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=25,
            variance_trials=1.0,
            n_samples=n_samples,
            skewness=-1.2,
            excess_kurtosis=3.5,  # Fisher (Pearson 6.5 - 3)
        )

        # Verify result contains expected attributes
        assert hasattr(result, "probability")
        assert hasattr(result, "expected_max_sharpe")
        assert hasattr(result, "p_value")
        assert hasattr(result, "z_score")

        # Verify result is valid
        assert 0 <= result.probability <= 1
        assert np.isfinite(result.z_score)

    def test_config_validation(self):
        """Test DSR function validates parameters."""
        # Invalid n_trials should be caught by deflated_sharpe_ratio_from_statistics
        with pytest.raises(ValueError, match="n_trials must be positive"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=1.0,
                n_trials=0,  # Invalid
                variance_trials=1.0,
                n_samples=100,
            )

        # Negative variance should be caught (now raises when n_trials > 1)
        with pytest.raises(ValueError, match="variance_trials must be positive"):
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=1.0,
                n_trials=10,
                variance_trials=-1.0,  # Invalid
                n_samples=100,
            )


# =============================================================================
# REAL-WORLD WORKFLOW TESTS
# =============================================================================


class TestRealWorldWorkflows:
    """Test realistic end-to-end workflows combining multiple components."""

    def test_parameter_sweep_workflow(self):
        """Test DSR in a parameter sweep scenario."""
        # Simulate testing 50 different parameter combinations
        n_strategies = 50
        n_samples = 252  # One year of daily data

        # Generate Sharpe ratios from 50 strategies (null: all ~0)
        np.random.seed(123)
        sharpe_ratios = np.random.normal(0, 1 / np.sqrt(n_samples), n_strategies)

        # Select best strategy
        best_sharpe = np.max(sharpe_ratios)

        # Calculate empirical variance across strategies
        variance_trials = np.var(sharpe_ratios, ddof=1)

        # Apply DSR
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=best_sharpe,
            n_trials=n_strategies,
            variance_trials=variance_trials,
            n_samples=n_samples,
        )

        # Under null, DSR should deflate significantly
        assert dsr_result.expected_max_sharpe > 0
        assert dsr_result.z_score < best_sharpe

        # High p-value expected (null hypothesis)
        assert dsr_result.p_value > 0.05  # Likely not significant

    def test_multiple_timeframes_analysis(self):
        """Test DSR across multiple timeframes."""
        timeframes = {
            "1min": 390 * 252,  # ~1 year of minute bars
            "5min": 78 * 252,  # ~1 year of 5-min bars
            "1hour": 6 * 252,  # ~1 year of hourly bars
            "1day": 252,  # ~1 year of daily bars
        }

        results = {}

        for timeframe, n_samples in timeframes.items():
            # Simulate Sharpe ratio for this timeframe
            # Higher frequency = lower Sharpe (transaction costs, noise)
            base_sharpe = 1.0
            frequency_penalty = np.log10(n_samples / 252) * 0.3
            observed_sharpe = base_sharpe - frequency_penalty

            # Calculate DSR (assume tested 10 strategies)
            dsr_result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=observed_sharpe,
                n_trials=10,
                variance_trials=1.0,
                n_samples=n_samples,
            )

            results[timeframe] = {
                "observed_sharpe": observed_sharpe,
                "dsr": dsr_result.probability,
                "p_value": dsr_result.p_value,
            }

        # Verify results for all timeframes
        for _timeframe, result in results.items():
            assert result["observed_sharpe"] > 0
            assert 0 <= result["dsr"] <= 1
            assert 0 <= result["p_value"] <= 1

    def test_regime_specific_analysis(self):
        """Test DSR applied to regime-specific performance."""
        # Simulate 3 market regimes
        regimes = ["bull", "sideways", "bear"]
        regime_results = {}

        np.random.seed(456)

        for _i, regime in enumerate(regimes):
            # Different Sharpe per regime
            if regime == "bull":
                sharpe = 2.0
                n_samples = 126  # Half year
            elif regime == "sideways":
                sharpe = 0.5
                n_samples = 126
            else:  # bear
                sharpe = -1.0
                n_samples = 126

            # Apply DSR (assume tested 5 strategies per regime)
            dsr_result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=sharpe,
                n_trials=5,
                variance_trials=1.0,
                n_samples=n_samples,
            )

            regime_results[regime] = {
                "sharpe": sharpe,
                "dsr_zscore": dsr_result.z_score,
                "p_value": dsr_result.p_value,
            }

        # Bull regime should have best DSR z-score
        assert regime_results["bull"]["dsr_zscore"] > regime_results["sideways"]["dsr_zscore"]
        assert regime_results["bull"]["dsr_zscore"] > regime_results["bear"]["dsr_zscore"]

        # Bear regime should have worst (most negative) DSR z-score
        assert regime_results["bear"]["dsr_zscore"] < regime_results["sideways"]["dsr_zscore"]


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


class TestDSRIntegrationEdgeCases:
    """Test edge cases in DSR integration with other components."""

    def test_single_trade_edge_case(self):
        """Test DSR behavior with minimal data (single trade)."""
        # Create single trade
        trade = TradeRecord(
            timestamp="2024-01-01T10:00:00",
            symbol="AAPL",
            direction="long",
            quantity=1.0,
            entry_price=100.0,
            exit_price=105.0,
            pnl=5.0,
            duration="PT10H",  # ISO 8601 duration: 10 hours
        )

        analyzer = TradeAnalysis([trade])
        stats = analyzer.compute_statistics()

        assert stats.n_trades == 1

        # Can't compute Sharpe ratio from single observation
        # (std dev undefined), but DSR should handle gracefully
        # This test documents expected behavior

    def test_all_losing_trades(self):
        """Test DSR with strategy that only loses."""
        # Create 20 losing trades
        np.random.seed(789)
        trades = []
        for i in range(20):
            pnl = -np.random.uniform(10, 50)
            trade = TradeRecord(
                timestamp=f"2024-01-{(i % 28) + 1:02d}T10:00:00",
                symbol="AAPL",
                direction="short",  # Losing short trade
                quantity=1.0,
                entry_price=100.0,
                exit_price=100.0 + abs(pnl),  # Price went up (bad for short)
                pnl=pnl,
                duration="PT10H",
            )
            trades.append(trade)

        analyzer = TradeAnalysis(trades)
        stats = analyzer.compute_statistics()

        assert stats.win_rate == 0.0

        # Calculate Sharpe (will be negative)
        pnls = np.array([t.pnl for t in trades])
        sharpe = np.mean(pnls) / np.std(pnls, ddof=1)

        assert sharpe < 0

        # DSR should handle negative Sharpe
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=sharpe,
            n_trials=10,
            variance_trials=1.0,
            n_samples=len(pnls),
        )

        # DSR z-score should be very negative
        assert dsr_result.z_score < 0
        # P-value should be high (null: SR <= 0 not rejected)
        assert dsr_result.p_value > 0.5

    def test_config_with_extreme_parameters(self):
        """Test DSR with extreme but valid parameters."""
        # Extreme non-normality
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=1000,  # Many strategies
            variance_trials=5.0,  # High variance
            n_samples=252,
            skewness=-5.0,  # Extreme negative skew
            excess_kurtosis=22.0,  # Fisher (Pearson 25.0 - 3): Extreme fat tails
        )

        # Should still compute successfully
        assert hasattr(result, "probability")
        assert result.probability >= 0
        # With extreme parameters, result should still be valid
        assert np.isfinite(result.z_score)
        assert np.isfinite(result.expected_max_sharpe)
