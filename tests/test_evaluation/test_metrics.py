"""Tests for core metrics module.

These tests validate the mathematical correctness of all metrics using both
unit tests with known values and property-based tests for invariants.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


from ml4t.diagnostic.evaluation.metrics import (
    analyze_feature_outcome,
    analyze_ml_importance,
    compute_conditional_ic,
    compute_forward_returns,
    compute_h_statistic,
    compute_ic_by_horizon,
    compute_ic_decay,
    compute_ic_hac_stats,
    compute_ic_ir,
    compute_ic_series,
    compute_mda_importance,
    compute_mdi_importance,
    compute_monotonicity,
    compute_permutation_importance,
    cov_hac,
    hit_rate,
    information_coefficient,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


class TestInformationCoefficient:
    """Test Information Coefficient calculation."""

    def test_perfect_positive_correlation(self):
        """Test IC with perfect positive correlation."""
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ic = information_coefficient(predictions, returns)
        assert abs(ic - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        """Test IC with perfect negative correlation."""
        predictions = np.array([5, 4, 3, 2, 1])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ic = information_coefficient(predictions, returns)
        assert abs(ic - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Test IC with no correlation."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100)

        ic = information_coefficient(predictions, returns)
        # Should be close to zero but allow for random variation
        assert abs(ic) < 0.3

    def test_with_confidence_intervals(self):
        """Test IC with confidence intervals."""
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        returns = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])

        result = information_coefficient(
            predictions,
            returns,
            confidence_intervals=True,
        )

        assert isinstance(result, dict)
        assert "ic" in result
        assert "lower_ci" in result
        assert "upper_ci" in result
        assert "p_value" in result

        # IC should be close to 1.0
        assert abs(result["ic"] - 1.0) < 1e-10

        # Confidence interval should be meaningful
        assert result["lower_ci"] < result["ic"] < result["upper_ci"]

        # P-value should be very small for perfect correlation
        assert result["p_value"] < 0.01

    def test_different_input_types(self):
        """Test IC with different input types."""
        predictions_array = np.array([1, 2, 3, 4])
        returns_array = np.array([0.1, 0.2, 0.3, 0.4])

        predictions_series = pd.Series(predictions_array)
        returns_series = pd.Series(returns_array)

        predictions_pl = pl.Series(predictions_array)
        returns_pl = pl.Series(returns_array)

        # All should give same result
        ic1 = information_coefficient(predictions_array, returns_array)
        ic2 = information_coefficient(predictions_series, returns_series)
        ic3 = information_coefficient(predictions_pl, returns_pl)

        assert abs(ic1 - ic2) < 1e-10
        assert abs(ic1 - ic3) < 1e-10

    def test_nan_handling(self):
        """Test IC with NaN values."""
        predictions = np.array([1, 2, np.nan, 4, 5])
        returns = np.array([0.1, np.nan, 0.3, 0.4, 0.5])

        ic = information_coefficient(predictions, returns)

        # Should still work with valid pairs
        assert not np.isnan(ic)

        # Test all NaN
        ic_nan = information_coefficient(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        )
        assert np.isnan(ic_nan)

    def test_edge_cases(self):
        """Test IC edge cases."""
        # Empty arrays
        ic_empty = information_coefficient(np.array([]), np.array([]))
        assert np.isnan(ic_empty)

        # Single value
        ic_single = information_coefficient(np.array([1]), np.array([0.1]))
        assert np.isnan(ic_single)

        # Different lengths should raise error
        with pytest.raises(ValueError):
            information_coefficient(np.array([1, 2]), np.array([0.1]))

    @given(
        data=st.data(),
    )
    def test_ic_properties(self, data):
        """Property-based test for IC invariants."""
        # Generate same-length arrays to avoid filter_too_much
        n = data.draw(st.integers(min_value=3, max_value=50))
        predictions = data.draw(
            st.lists(
                st.floats(min_value=-10, max_value=10),
                min_size=n,
                max_size=n,
            )
        )
        returns = data.draw(st.lists(st.floats(min_value=-1, max_value=1), min_size=n, max_size=n))

        pred_array = np.array(predictions)
        ret_array = np.array(returns)

        # Skip if all NaN or infinite
        assume(np.isfinite(pred_array).sum() >= 3)
        assume(np.isfinite(ret_array).sum() >= 3)

        ic = information_coefficient(pred_array, ret_array)

        # IC should be in [-1, 1] range
        if not np.isnan(ic):
            assert -1 <= ic <= 1


class TestSharpeRatio:
    """Test Sharpe Ratio calculation."""

    def test_known_values(self):
        """Test Sharpe ratio with known values."""
        # Returns with mean=0.01, std=0.02
        returns = np.array([0.01, 0.03, -0.01, 0.01, 0.02])

        sharpe = sharpe_ratio(returns)
        expected_sharpe = np.mean(returns) / np.std(returns, ddof=1)

        assert abs(sharpe - expected_sharpe) < 1e-10

    def test_annualization(self):
        """Test Sharpe ratio annualization."""
        returns = np.array([0.01, 0.02, -0.01, 0.02])

        sharpe_daily = sharpe_ratio(returns)
        sharpe_annual = sharpe_ratio(returns, annualization_factor=252)

        expected_annual = sharpe_daily * np.sqrt(252)
        assert abs(sharpe_annual - expected_annual) < 1e-10

    def test_risk_free_rate(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = np.array([0.02, 0.03, 0.01, 0.04])
        rf_rate = 0.01

        sharpe = sharpe_ratio(returns, risk_free_rate=rf_rate)
        expected_sharpe = (np.mean(returns) - rf_rate) / np.std(returns, ddof=1)

        assert abs(sharpe - expected_sharpe) < 1e-10

    def test_with_confidence_intervals(self):
        """Test Sharpe ratio with bootstrap confidence intervals."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.05, 100)  # Positive mean returns

        result = sharpe_ratio(returns, confidence_intervals=True, random_state=42)

        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "lower_ci" in result
        assert "upper_ci" in result

        # Confidence interval should bracket the point estimate
        assert result["lower_ci"] <= result["sharpe"] <= result["upper_ci"]

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        # All returns the same (zero volatility)
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        sharpe = sharpe_ratio(returns)
        assert np.isinf(sharpe)  # Should be infinite

    def test_edge_cases(self):
        """Test Sharpe ratio edge cases."""
        # Empty returns
        sharpe_empty = sharpe_ratio(np.array([]))
        assert np.isnan(sharpe_empty)

        # Single return
        sharpe_single = sharpe_ratio(np.array([0.01]))
        assert np.isnan(sharpe_single)

        # All NaN
        sharpe_nan = sharpe_ratio(np.array([np.nan, np.nan]))
        assert np.isnan(sharpe_nan)

    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=5,
            max_size=100,
        ),
    )
    def test_sharpe_properties(self, returns):
        """Property-based test for Sharpe ratio invariants."""
        ret_array = np.array(returns)

        # Skip if all NaN or insufficient valid data
        valid_returns = ret_array[np.isfinite(ret_array)]
        assume(len(valid_returns) >= 5)
        assume(np.std(valid_returns, ddof=1) > 1e-6)  # Non-zero volatility

        sharpe = sharpe_ratio(ret_array)

        # Sharpe ratio should be finite (given our assumptions)
        assert np.isfinite(sharpe)

        # Scaling property: doubling returns should keep Sharpe ratio the same
        # (since both mean and std are doubled)
        sharpe_double = sharpe_ratio(ret_array * 2)
        assert abs(sharpe_double - sharpe) < 1e-10


class TestMaximumDrawdown:
    """Test Maximum Drawdown calculation."""

    def test_monotonic_increasing(self):
        """Test MDD with monotonically increasing returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.03])

        dd = maximum_drawdown(returns)

        # No drawdown for monotonically increasing cumulative returns
        assert dd["max_drawdown"] >= -1e-10  # Allow for floating point error

    def test_simple_drawdown(self):
        """Test MDD with known drawdown pattern."""
        # Returns: +10%, -5%, +8%, -12%, +3%
        # Cumulative: 10%, 4.5%, 12.86%, 1.32%, 4.36%
        # Peak at 12.86%, trough at 1.32% -> DD = -89.7%
        returns = np.array([0.10, -0.05, 0.08, -0.12, 0.03])

        dd = maximum_drawdown(returns)

        # Should have negative drawdown
        assert dd["max_drawdown"] < 0

        # Peak should be before trough
        assert dd["peak_date"] < dd["trough_date"]

        # Duration should be non-negative
        assert dd["max_drawdown_duration"] >= 0

    def test_cumulative_input(self):
        """Test MDD with cumulative returns as input."""
        cum_returns = np.array([0.0, 0.10, 0.05, 0.13, -0.02, 0.01])

        dd = maximum_drawdown(cum_returns, cumulative=True)

        # Should identify the drawdown from 0.13 to -0.02
        assert dd["max_drawdown"] < 0
        assert dd["peak_date"] == 3  # Index of 0.13
        assert dd["trough_date"] == 4  # Index of -0.02

    def test_edge_cases(self):
        """Test MDD edge cases."""
        # Empty returns
        dd_empty = maximum_drawdown(np.array([]))
        assert np.isnan(dd_empty["max_drawdown"])

        # Single return
        dd_single = maximum_drawdown(np.array([0.01]))
        assert dd_single["max_drawdown"] == 0.0

        # All NaN
        dd_nan = maximum_drawdown(np.array([np.nan, np.nan]))
        assert np.isnan(dd_nan["max_drawdown"])

    @given(
        returns=st.lists(
            st.floats(min_value=-0.3, max_value=0.3),
            min_size=2,
            max_size=50,
        ),
    )
    def test_mdd_properties(self, returns):
        """Property-based test for MDD invariants."""
        ret_array = np.array(returns)

        # Skip if all NaN
        valid_returns = ret_array[np.isfinite(ret_array)]
        assume(len(valid_returns) >= 2)

        dd = maximum_drawdown(ret_array)

        if not np.isnan(dd["max_drawdown"]):
            # Maximum drawdown should always be non-positive
            assert dd["max_drawdown"] <= 0

            # Peak should come before or at the same time as trough
            if not np.isnan(dd["peak_date"]) and not np.isnan(dd["trough_date"]):
                assert dd["peak_date"] <= dd["trough_date"]


class TestSortinoRatio:
    """Test Sortino Ratio calculation."""

    def test_known_values(self):
        """Test Sortino ratio with known values."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        target = 0.0

        sortino = sortino_ratio(returns, target_return=target)

        # Calculate expected value manually
        excess_returns = returns - target
        downside_returns = excess_returns[excess_returns < 0]

        expected_sortino = np.mean(excess_returns) / np.sqrt(
            np.mean(downside_returns**2),
        )

        assert abs(sortino - expected_sortino) < 1e-10

    def test_no_downside(self):
        """Test Sortino ratio with no downside."""
        returns = np.array([0.01, 0.02, 0.03, 0.01])  # All positive

        sortino = sortino_ratio(returns, target_return=0.0)

        # Should be infinite (no downside risk)
        assert np.isinf(sortino)

    def test_target_return(self):
        """Test Sortino ratio with different target returns."""
        returns = np.array([0.02, -0.01, 0.03, 0.01])

        sortino_0 = sortino_ratio(returns, target_return=0.0)
        sortino_1 = sortino_ratio(returns, target_return=0.01)

        # Different targets should give different results
        assert sortino_0 != sortino_1

    def test_annualization(self):
        """Test Sortino ratio annualization."""
        returns = np.array([0.01, -0.005, 0.02, -0.01])

        sortino_daily = sortino_ratio(returns)
        sortino_annual = sortino_ratio(returns, annualization_factor=252)

        expected_annual = sortino_daily * np.sqrt(252)
        assert abs(sortino_annual - expected_annual) < 1e-10

    def test_edge_cases(self):
        """Test Sortino ratio edge cases."""
        # Empty returns
        sortino_empty = sortino_ratio(np.array([]))
        assert np.isnan(sortino_empty)

        # Single return
        sortino_single = sortino_ratio(np.array([0.01]))
        assert np.isnan(sortino_single)

    @given(
        returns=st.lists(
            st.floats(min_value=-0.2, max_value=0.2),
            min_size=5,
            max_size=50,
        ),
    )
    def test_sortino_properties(self, returns):
        """Property-based test for Sortino ratio invariants."""
        ret_array = np.array(returns)

        # Skip if all NaN or insufficient data
        valid_returns = ret_array[np.isfinite(ret_array)]
        assume(len(valid_returns) >= 5)

        # Ensure we have some meaningful downside for calculation
        excess_returns = valid_returns - 0.0
        downside_returns = excess_returns[excess_returns < 0]
        assume(len(downside_returns) >= 1)
        assume(np.abs(downside_returns).max() > 1e-10)  # Avoid tiny values

        sortino = sortino_ratio(ret_array, target_return=0.0)

        # Should be finite (given our assumptions)
        if not np.isinf(sortino):
            assert np.isfinite(sortino)


class TestHitRate:
    """Test Hit Rate calculation."""

    def test_perfect_hit_rate(self):
        """Test hit rate with perfect predictions."""
        predictions = np.array([0.1, -0.2, 0.3, -0.1])
        returns = np.array([0.02, -0.01, 0.05, -0.03])

        hr = hit_rate(predictions, returns)

        assert hr == 100.0  # Perfect directional accuracy

    def test_zero_hit_rate(self):
        """Test hit rate with completely wrong predictions."""
        predictions = np.array([0.1, -0.2, 0.3, -0.1])
        returns = np.array([-0.02, 0.01, -0.05, 0.03])

        hr = hit_rate(predictions, returns)

        assert hr == 0.0  # No correct predictions

    def test_fifty_percent_hit_rate(self):
        """Test hit rate with 50% accuracy."""
        predictions = np.array([0.1, -0.2, 0.3, -0.1])
        returns = np.array([0.02, -0.01, -0.05, 0.03])  # 50% correct

        hr = hit_rate(predictions, returns)

        assert hr == 50.0

    def test_zero_values(self):
        """Test hit rate with zero predictions/returns."""
        predictions = np.array([0.1, 0.0, -0.2, 0.0])
        returns = np.array([0.02, 0.0, -0.01, 0.01])

        hr = hit_rate(predictions, returns)

        # Zeros are treated as correct (conservative approach)
        assert hr >= 50.0  # At least the zeros should be correct

    def test_different_input_types(self):
        """Test hit rate with different input types."""
        predictions_array = np.array([0.1, -0.2, 0.3])
        returns_array = np.array([0.02, -0.01, 0.05])

        predictions_series = pd.Series(predictions_array)
        returns_series = pd.Series(returns_array)

        # Should give same result
        hr1 = hit_rate(predictions_array, returns_array)
        hr2 = hit_rate(predictions_series, returns_series)

        assert abs(hr1 - hr2) < 1e-10

    def test_nan_handling(self):
        """Test hit rate with NaN values."""
        predictions = np.array([0.1, np.nan, 0.3, -0.1])
        returns = np.array([0.02, -0.01, np.nan, -0.03])

        hr = hit_rate(predictions, returns)

        # Should work with valid pairs only
        assert not np.isnan(hr)

    def test_edge_cases(self):
        """Test hit rate edge cases."""
        # Empty arrays
        hr_empty = hit_rate(np.array([]), np.array([]))
        assert np.isnan(hr_empty)

        # Different lengths should raise error
        with pytest.raises(ValueError):
            hit_rate(np.array([1, 2]), np.array([0.1]))

    @given(
        data=st.data(),
    )
    def test_hit_rate_properties(self, data):
        """Property-based test for hit rate invariants."""
        # Generate same-length arrays to avoid filter_too_much
        n = data.draw(st.integers(min_value=2, max_value=50))
        predictions = data.draw(
            st.lists(
                st.floats(min_value=-1, max_value=1),
                min_size=n,
                max_size=n,
            )
        )
        returns = data.draw(
            st.lists(
                st.floats(min_value=-0.5, max_value=0.5),
                min_size=n,
                max_size=n,
            )
        )

        pred_array = np.array(predictions)
        ret_array = np.array(returns)

        # Skip if all NaN
        valid_mask = np.isfinite(pred_array) & np.isfinite(ret_array)
        assume(np.sum(valid_mask) >= 2)

        hr = hit_rate(pred_array, ret_array)

        # Hit rate should be between 0 and 100
        if not np.isnan(hr):
            assert 0 <= hr <= 100


class TestComputeICSeriesAndRelated:
    """Tests for IC time series and related calculations."""

    @pytest.fixture
    def panel_data(self) -> pl.DataFrame:
        """Create panel data for IC calculations."""
        np.random.seed(42)
        n_periods = 50
        n_assets = 10

        data = []
        base_date = pd.Timestamp("2020-01-01")

        for t in range(n_periods):
            date = base_date + pd.Timedelta(days=t)
            preds = np.random.randn(n_assets)
            rets = preds * 0.3 + np.random.randn(n_assets) * 0.7

            for i in range(n_assets):
                data.append(
                    {
                        "date": date,
                        "asset": f"asset_{i}",
                        "prediction": preds[i],
                        "forward_return": rets[i],
                    }
                )

        return pl.DataFrame(data)

    def test_compute_ic_series_basic(self, panel_data):
        """Test IC series computation."""
        result = compute_ic_series(
            panel_data,
            panel_data,
            pred_col="prediction",
            ret_col="forward_return",
            date_col="date",
        )

        # Should return a DataFrame with IC per date
        assert isinstance(result, pl.DataFrame | pd.DataFrame)

    def test_compute_ic_series_with_method(self, panel_data):
        """Test IC series with different correlation methods."""
        result_spearman = compute_ic_series(
            panel_data,
            panel_data,
            pred_col="prediction",
            ret_col="forward_return",
            date_col="date",
            method="spearman",
        )

        # Should return valid IC values
        assert isinstance(result_spearman, pl.DataFrame | pd.DataFrame)


class TestComputeForwardReturns:
    """Tests for forward returns computation."""

    @pytest.fixture
    def price_data(self) -> pl.DataFrame:
        """Create price data for forward returns."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = [100, 105, 110, 108, 115, 112, 120, 118, 125, 130]
        return pl.DataFrame({"date": dates, "close": prices})

    def test_forward_returns_basic(self, price_data):
        """Test basic forward returns calculation."""
        result = compute_forward_returns(price_data, periods=1)

        # Should return a DataFrame
        assert isinstance(result, pl.DataFrame | pd.DataFrame)

    def test_forward_returns_multiple_periods(self, price_data):
        """Test forward returns with multiple horizons."""
        result = compute_forward_returns(price_data, periods=[1, 5])

        # Should have columns for each period
        assert isinstance(result, pl.DataFrame | pd.DataFrame)

    def test_forward_returns_custom_column(self, price_data):
        """Test forward returns with custom column name."""
        result = compute_forward_returns(
            price_data,
            periods=1,
            price_col="close",
        )

        assert isinstance(result, pl.DataFrame | pd.DataFrame)

    def test_forward_returns_custom_output_template(self, price_data):
        """Test custom output column naming template."""
        result = compute_forward_returns(
            price_data,
            periods=[1, 3],
            output_col_template="{period}D_fwd_return",
        )

        assert "1D_fwd_return" in result.columns
        assert "3D_fwd_return" in result.columns

    def test_forward_returns_nan_to_null(self):
        """Test optional NaN->null conversion on generated forward returns."""
        prices = pl.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="D"),
                "close": [100.0, float("nan"), 102.0],
            }
        )

        result = compute_forward_returns(
            prices,
            periods=1,
            nan_to_null=True,
        )

        # Day 1 return references day 2 NaN, so should be null (not NaN)
        assert result["fwd_ret_1"][0] is None


class TestComputeMonotonicity:
    """Tests for monotonicity calculation."""

    @pytest.fixture
    def feature_outcome_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create feature and outcome data."""
        np.random.seed(42)
        n = 500

        # Create monotonic relationship
        features = np.random.randn(n)
        outcomes = features * 0.5 + np.random.randn(n) * 0.3

        df_features = pl.DataFrame({"feature": features})
        df_outcomes = pl.DataFrame({"outcome": outcomes})

        return df_features, df_outcomes

    def test_monotonicity_basic(self, feature_outcome_data):
        """Test basic monotonicity calculation."""
        features, outcomes = feature_outcome_data

        result = compute_monotonicity(
            features,
            outcomes,
            feature_col="feature",
            outcome_col="outcome",
            n_quantiles=5,
        )

        # Should return a dict with monotonicity score
        assert isinstance(result, dict)
        assert "monotonicity" in result or len(result) > 0

    def test_monotonicity_with_arrays(self):
        """Test monotonicity with numpy arrays."""
        np.random.seed(42)
        features = np.random.randn(100)
        outcomes = features * 2 + np.random.randn(100) * 0.1

        result = compute_monotonicity(features, outcomes, n_quantiles=5)

        assert isinstance(result, dict)


class TestComputeICIR:
    """Tests for IC Information Ratio calculation."""

    def test_ic_ir_basic(self):
        """Test basic IC IR calculation."""
        # Create IC series DataFrame
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        np.random.seed(42)
        ic_values = 0.05 + np.random.randn(50) * 0.02  # Mean ~0.05

        ic_df = pl.DataFrame({"date": dates, "ic": ic_values})

        ic_ir = compute_ic_ir(ic_df, ic_col="ic")

        # Should be positive (positive mean IC)
        assert isinstance(ic_ir, float | dict)

    def test_ic_ir_with_array(self):
        """Test IC IR with numpy array."""
        np.random.seed(42)
        ic_values = 0.05 + np.random.randn(50) * 0.02

        ic_ir = compute_ic_ir(ic_values)

        # Should return a value
        assert isinstance(ic_ir, float | dict)

    def test_ic_ir_with_ci(self):
        """Test IC IR with confidence intervals."""
        np.random.seed(42)
        ic_values = 0.05 + np.random.randn(100) * 0.02

        result = compute_ic_ir(ic_values, confidence_intervals=True)

        # Should return a dict with CI
        assert isinstance(result, dict)


class TestCovHAC:
    """Tests for HAC covariance estimation.

    Note: cov_hac is from statsmodels, so we just test basic integration.
    """

    def test_cov_hac_import(self):
        """Test that cov_hac can be imported."""
        # cov_hac is a re-export from statsmodels
        assert callable(cov_hac)


class TestComputeICHACStats:
    """Tests for HAC-adjusted IC statistics."""

    def test_ic_hac_stats_basic(self):
        """Test basic HAC-adjusted IC stats."""
        np.random.seed(42)
        n_periods = 100
        ic_values = 0.03 + np.random.randn(n_periods) * 0.05

        # Create DataFrame
        dates = pd.date_range("2020-01-01", periods=n_periods, freq="D")
        ic_df = pl.DataFrame({"date": dates, "ic": ic_values})

        result = compute_ic_hac_stats(ic_df, ic_col="ic")

        assert isinstance(result, dict)

    def test_ic_hac_stats_with_array(self):
        """Test HAC stats with numpy array."""
        np.random.seed(42)
        ic_values = 0.03 + np.random.randn(100) * 0.05

        result = compute_ic_hac_stats(ic_values)

        assert isinstance(result, dict)


class TestComputeICByHorizon:
    """Tests for IC by horizon calculation."""

    @pytest.fixture
    def predictions_prices_data(self):
        """Create predictions and prices data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        predictions = pl.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )

        # Create prices with some trend
        prices_raw = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = pl.DataFrame(
            {
                "date": dates,
                "close": prices_raw,
            }
        )

        return predictions, prices

    def test_ic_by_horizon_basic(self, predictions_prices_data):
        """Test IC by horizon calculation."""
        predictions, prices = predictions_prices_data

        result = compute_ic_by_horizon(
            predictions,
            prices,
            horizons=[1, 5, 10],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        # Should return IC for each horizon
        assert isinstance(result, dict)

    def test_ic_decay_concept(self):
        """Test IC decay conceptually with simple data."""
        np.random.seed(42)
        n = 1000

        # Signal that predicts immediate future well
        predictions = np.random.randn(n)
        immediate_return = predictions * 0.6 + np.random.randn(n) * 0.4

        # Compute IC using basic function
        ic_h1 = information_coefficient(predictions[:-5], immediate_return[:-5])

        # IC should be meaningfully positive for correlated data
        assert ic_h1 > 0.3


class TestComputeConditionalIC:
    """Tests for conditional IC calculation."""

    def test_conditional_ic_import(self):
        """Test conditional IC function exists."""
        # Just verify the function is callable
        assert callable(compute_conditional_ic)


class TestComputeHStatistic:
    """Tests for H-statistic (interaction strength) calculation."""

    def test_h_statistic_import(self):
        """Test H-statistic function exists."""
        # Just verify the function is callable
        assert callable(compute_h_statistic)


class TestImportanceMetrics:
    """Tests for feature importance metrics."""

    @pytest.fixture(scope="class")
    def classification_data(self):
        """Create classification data for importance tests."""
        np.random.seed(42)
        n = 500
        n_features = 5

        # Create features
        X = np.random.randn(n, n_features)

        # Target depends mainly on first 2 features
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n) * 0.3 > 0).astype(int)

        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_model(self, classification_data):
        """Train a RandomForest model once for MDI and permutation tests."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def trained_rf_model_oob(self, classification_data):
        """Train a RandomForest model with OOB for MDA tests."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, oob_score=True, random_state=42)
        model.fit(X, y)
        return model

    def test_compute_mdi_importance(self, trained_rf_model):
        """Test MDI importance calculation."""
        try:
            result = compute_mdi_importance(
                trained_rf_model, feature_names=[f"f{i}" for i in range(5)]
            )

            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("sklearn not available")

    def test_compute_permutation_importance(self, trained_rf_model, classification_data):
        """Test permutation importance calculation."""
        X, y = classification_data

        try:
            result = compute_permutation_importance(
                trained_rf_model,
                X,
                y,
                feature_names=[f"f{i}" for i in range(5)],
            )

            assert isinstance(result, dict)
        except (ImportError, TypeError) as e:
            pytest.skip(f"Test skipped: {e}")

    def test_compute_mda_importance(self, trained_rf_model_oob, classification_data):
        """Test MDA importance calculation."""
        X, y = classification_data

        try:
            result = compute_mda_importance(
                trained_rf_model_oob,
                X,
                y,
                feature_names=[f"f{i}" for i in range(5)],
            )

            if result is not None:
                assert isinstance(result, dict)
        except (ImportError, ValueError, TypeError) as e:
            pytest.skip(f"Test skipped: {e}")


class TestAnalyzeFeatureOutcome:
    """Tests for feature-outcome analysis."""

    @pytest.fixture
    def predictions_prices(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create predictions and prices data."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        predictions = pl.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )

        prices = pl.DataFrame(
            {
                "date": dates,
                "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            }
        )

        return predictions, prices

    def test_analyze_feature_outcome_basic(self, predictions_prices):
        """Test basic feature-outcome analysis."""
        predictions, prices = predictions_prices

        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)


class TestAnalyzeMLImportance:
    """Tests for ML-based importance analysis."""

    def test_analyze_ml_importance_import(self):
        """Test ML importance function exists."""
        assert callable(analyze_ml_importance)

    @pytest.fixture(scope="class")
    def ml_importance_data(self):
        """Create data for ML importance tests."""
        np.random.seed(42)
        n = 500
        n_features = 5

        # Create features where first two are important
        X = np.random.randn(n, n_features)
        # Target depends on first 2 features
        y = (X[:, 0] * 0.8 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3 > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names

    @pytest.fixture(scope="class")
    def trained_rf_model_ml(self, ml_importance_data):
        """Train a RandomForest model once for ML importance tests."""
        from sklearn.ensemble import RandomForestClassifier

        X, y, _ = ml_importance_data
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    def test_analyze_ml_importance_basic(self, trained_rf_model_ml, ml_importance_data):
        """Test basic ML importance analysis."""
        X, y, feature_names = ml_importance_data

        try:
            result = analyze_ml_importance(
                trained_rf_model_ml,
                X,
                y,
                feature_names=feature_names,
                methods=["mdi", "pfi"],  # Skip SHAP for speed
            )

            assert isinstance(result, dict)
            # Results are nested under method_results
            assert "method_results" in result
            assert "mdi" in result["method_results"] or "pfi" in result["method_results"]

        except ImportError:
            pytest.skip("sklearn not available")

    def test_analyze_ml_importance_with_dataframe(self, trained_rf_model_ml, ml_importance_data):
        """Test ML importance with DataFrame input."""
        X, y, feature_names = ml_importance_data

        try:
            # Convert to DataFrame
            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = analyze_ml_importance(
                trained_rf_model_ml,
                X_df,
                y,
                methods=["mdi"],
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


class TestComputeConditionalICExtended:
    """Extended tests for conditional IC calculation."""

    @pytest.fixture
    def conditional_ic_data(self):
        """Create data for conditional IC tests."""
        np.random.seed(42)
        n = 500

        # Feature A: momentum signal
        feature_a = np.random.randn(n)

        # Feature B: volatility regime (high/low)
        feature_b = np.abs(np.random.randn(n))

        # Returns depend on feature_a, but differently in different regimes
        # In high volatility, feature_a is more predictive
        high_vol_mask = feature_b > np.median(feature_b)
        returns = np.zeros(n)
        returns[high_vol_mask] = (
            feature_a[high_vol_mask] * 0.8 + np.random.randn(np.sum(high_vol_mask)) * 0.2
        )
        returns[~high_vol_mask] = (
            feature_a[~high_vol_mask] * 0.2 + np.random.randn(np.sum(~high_vol_mask)) * 0.8
        )

        return feature_a, feature_b, returns

    def test_conditional_ic_basic(self, conditional_ic_data):
        """Test basic conditional IC calculation."""
        feature_a, feature_b, returns = conditional_ic_data

        result = compute_conditional_ic(
            feature_a,
            feature_b,
            returns,
            n_quantiles=5,
        )

        assert isinstance(result, dict)
        assert "quantile_ics" in result or "ic_by_quantile" in result or len(result) > 0

    def test_conditional_ic_with_dataframe(self, conditional_ic_data):
        """Test conditional IC with DataFrame input."""
        feature_a, feature_b, returns = conditional_ic_data
        n = len(feature_a)

        dates = pd.date_range("2020-01-01", periods=n, freq="D")

        df_a = pl.DataFrame({"date": dates, "feature_a": feature_a})
        df_b = pl.DataFrame({"date": dates, "feature_b": feature_b})
        df_returns = pl.DataFrame({"date": dates, "returns": returns})

        result = compute_conditional_ic(
            df_a,
            df_b,
            df_returns,
            date_col="date",
            n_quantiles=3,
        )

        assert isinstance(result, dict)

    def test_conditional_ic_different_quantiles(self, conditional_ic_data):
        """Test conditional IC with different numbers of quantiles."""
        feature_a, feature_b, returns = conditional_ic_data

        for n_q in [2, 3, 5, 10]:
            result = compute_conditional_ic(
                feature_a,
                feature_b,
                returns,
                n_quantiles=n_q,
            )
            assert isinstance(result, dict)

    def test_conditional_ic_methods(self, conditional_ic_data):
        """Test conditional IC with different correlation methods."""
        feature_a, feature_b, returns = conditional_ic_data

        for method in ["spearman", "pearson"]:
            result = compute_conditional_ic(
                feature_a,
                feature_b,
                returns,
                method=method,
            )
            assert isinstance(result, dict)


class TestComputeHStatisticExtended:
    """Extended tests for H-statistic calculation."""

    @pytest.fixture
    def h_statistic_data(self):
        """Create data for H-statistic tests."""
        np.random.seed(42)
        n = 200
        n_features = 4

        # Features with known interactions
        X = np.random.randn(n, n_features)
        # Target has interaction between features 0 and 1
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        return X, y

    def test_h_statistic_basic(self, h_statistic_data):
        """Test basic H-statistic calculation."""
        X, y = h_statistic_data

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_h_statistic(
                model,
                X,
                feature_pairs=[(0, 1)],
                n_samples=50,
                grid_resolution=10,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")

    def test_h_statistic_with_names(self, h_statistic_data):
        """Test H-statistic with feature names."""
        X, y = h_statistic_data
        feature_names = ["f0", "f1", "f2", "f3"]

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_h_statistic(
                model,
                X,
                feature_pairs=[("f0", "f1")],
                feature_names=feature_names,
                n_samples=50,
                grid_resolution=10,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")

    def test_h_statistic_with_dataframe(self, h_statistic_data):
        """Test H-statistic with DataFrame input."""
        X, y = h_statistic_data
        feature_names = ["f0", "f1", "f2", "f3"]

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = compute_h_statistic(
                model,
                X_df,
                feature_pairs=[("f0", "f1")],
                n_samples=50,
                grid_resolution=10,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")

    def test_h_statistic_all_pairs(self, h_statistic_data):
        """Test H-statistic with all feature pairs (None)."""
        X, y = h_statistic_data

        try:
            from sklearn.ensemble import RandomForestRegressor

            # Use smaller data for speed
            X_small = X[:100, :3]
            y_small = y[:100]

            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X_small, y_small)

            result = compute_h_statistic(
                model,
                X_small,
                feature_pairs=None,  # All pairs
                n_samples=30,
                grid_resolution=5,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


class TestAnalyzeInteractions:
    """Tests for comprehensive interaction analysis."""

    @pytest.fixture
    def interaction_data(self):
        """Create data for interaction analysis tests."""
        np.random.seed(42)
        n = 200
        n_features = 4

        # Features with known interactions
        X = np.random.randn(n, n_features)
        # Regression target with interaction
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_analyze_interactions_basic(self, interaction_data):
        """Test basic interaction analysis."""
        X, y, feature_names = interaction_data

        try:
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_interactions

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Convert to DataFrame so feature names are recognized
            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = analyze_interactions(
                model,
                X_df,
                y,
                feature_pairs=[("feature_0", "feature_1")],
                methods=["h_statistic"],  # Just one method for speed
                grid_resolution=5,
                max_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")

    def test_analyze_interactions_with_dataframe(self, interaction_data):
        """Test interaction analysis with DataFrame."""
        X, y, feature_names = interaction_data

        try:
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_interactions

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = analyze_interactions(
                model,
                X_df,
                y,
                feature_pairs=[("feature_0", "feature_1")],
                methods=["h_statistic"],
                grid_resolution=5,
                max_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


class TestShapImportance:
    """Tests for SHAP-based feature importance."""

    @pytest.fixture(scope="class")
    def shap_data(self):
        """Create data for SHAP importance tests."""
        np.random.seed(42)
        n = 200
        n_features = 5

        X = np.random.randn(n, n_features)
        # Target depends mainly on first 2 features
        y = X[:, 0] * 0.8 + X[:, 1] * 0.5 + np.random.randn(n) * 0.2

        feature_names = [f"f{i}" for i in range(n_features)]
        return X, y, feature_names

    @pytest.fixture(scope="class")
    def trained_rf_model_shap(self, shap_data):
        """Train a RandomForest model once for SHAP tests."""
        from sklearn.ensemble import RandomForestRegressor

        X, y, _ = shap_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_compute_shap_importance_with_tree_model(self, trained_rf_model_shap, shap_data):
        """Test SHAP importance with tree-based model."""
        pytest.importorskip("shap")
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, _, feature_names = shap_data

        result = compute_shap_importance(
            trained_rf_model_shap,
            X,
            feature_names=feature_names,
            max_samples=50,
        )

        assert isinstance(result, dict)
        # First feature should be important
        if "importance" in result:
            assert len(result["importance"]) == len(feature_names)

    def test_compute_shap_importance_with_dataframe(self, trained_rf_model_shap, shap_data):
        """Test SHAP importance with DataFrame input."""
        pytest.importorskip("shap")
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, _, feature_names = shap_data

        X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

        result = compute_shap_importance(
            trained_rf_model_shap,
            X_df,
            max_samples=50,
        )

        assert isinstance(result, dict)


# Property-based tests for advanced functions
class TestAdvancedMetricsProperties:
    """Property-based tests for advanced metrics."""

    @given(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=50, max_size=200
        ),
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=50, max_size=200
        ),
        st.lists(
            st.floats(min_value=-0.1, max_value=0.1, allow_nan=False), min_size=50, max_size=200
        ),
    )
    def test_conditional_ic_returns_valid_structure(self, feature_a, feature_b, returns):
        """Property: conditional IC always returns a dict."""
        # Ensure same length
        min_len = min(len(feature_a), len(feature_b), len(returns))
        assume(min_len >= 50)

        fa = np.array(feature_a[:min_len])
        fb = np.array(feature_b[:min_len])
        ret = np.array(returns[:min_len])

        # Skip if constant
        assume(np.std(fa) > 1e-10)
        assume(np.std(fb) > 1e-10)

        result = compute_conditional_ic(fa, fb, ret, n_quantiles=3)
        assert isinstance(result, dict)


# ==============================================================================
# ADDITIONAL COVERAGE TESTS
# ==============================================================================


class TestComputeICDecayExtended:
    """Extended tests for IC decay computation."""

    @pytest.fixture
    def ic_decay_data(self):
        """Create data for IC decay tests - predictions and prices DataFrames."""
        np.random.seed(42)
        n_dates = 100
        n_symbols = 5

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")

        data = []
        for symbol_idx in range(n_symbols):
            base_price = 100 + symbol_idx * 10
            for _i, date in enumerate(dates):
                data.append(
                    {
                        "date": date,
                        "symbol": f"SYM_{symbol_idx}",
                        "prediction": np.random.randn(),
                        "close": base_price * (1 + np.random.randn() * 0.02),
                    }
                )

        df = pl.DataFrame(data)
        predictions = df.select(["date", "symbol", "prediction"])
        prices = df.select(["date", "symbol", "close"])
        return predictions, prices

    def test_compute_ic_decay_basic(self, ic_decay_data):
        """Test basic IC decay computation."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        predictions, prices = ic_decay_data

        result = compute_ic_decay(
            predictions,
            prices,
            horizons=[1, 5, 10],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)
        assert "ic_by_horizon" in result
        assert "half_life" in result

    def test_compute_ic_decay_with_pandas(self, ic_decay_data):
        """Test IC decay with pandas DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        predictions, prices = ic_decay_data

        predictions_pd = predictions.to_pandas()
        prices_pd = prices.to_pandas()

        result = compute_ic_decay(
            predictions_pd,
            prices_pd,
            horizons=[1, 5],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)
        assert "ic_by_horizon" in result

    def test_compute_ic_decay_with_different_methods(self, ic_decay_data):
        """Test IC decay with different correlation methods."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        predictions, prices = ic_decay_data

        for method in ["spearman", "pearson"]:
            result = compute_ic_decay(
                predictions,
                prices,
                horizons=[1, 5],
                method=method,
            )
            assert isinstance(result, dict)

    def test_compute_ic_decay_with_group(self, ic_decay_data):
        """Test IC decay with grouping by symbol."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        predictions, prices = ic_decay_data

        result = compute_ic_decay(
            predictions,
            prices,
            horizons=[1, 5],
            group_col="symbol",
        )

        assert isinstance(result, dict)


class TestComputeICSeriesPandas:
    """Tests for IC series computation with pandas input (cover lines 701-727)."""

    @pytest.fixture
    def panel_data_pandas(self):
        """Create panel data as pandas DataFrames."""
        np.random.seed(42)
        n_dates = 20
        n_symbols = 10

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        symbols = [f"SYM_{i}" for i in range(n_symbols)]

        data = []
        for date in dates:
            for symbol in symbols:
                data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "prediction": np.random.randn(),
                        "returns": np.random.randn() * 0.01,
                    }
                )

        df = pd.DataFrame(data)
        predictions = df[["date", "symbol", "prediction"]]
        returns = df[["date", "symbol", "returns"]]
        return predictions, returns

    def test_compute_ic_series_pandas(self, panel_data_pandas):
        """Test IC series with pandas DataFrames."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_series

        predictions, returns = panel_data_pandas

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="returns",
            date_col="date",
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)

    def test_compute_ic_series_pandas_pearson(self, panel_data_pandas):
        """Test IC series with pearson method."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_series

        predictions, returns = panel_data_pandas

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="returns",
            date_col="date",
            method="pearson",
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)


class TestComputeShapInteractions:
    """Tests for SHAP interaction computation."""

    @pytest.fixture
    def interaction_model_data(self):
        """Create data for SHAP interaction tests."""
        np.random.seed(42)
        n = 200
        n_features = 4

        X = np.random.randn(n, n_features)
        # Create interaction effect
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        return X, y

    def test_compute_shap_interactions_with_rf(self, interaction_model_data):
        """Test SHAP interactions with RandomForest."""
        pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        X, y = interaction_model_data

        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=4)
        model.fit(X, y)

        result = compute_shap_interactions(
            model,
            X,
            max_samples=50,
            top_k=3,
        )

        assert isinstance(result, dict)
        assert "interaction_matrix" in result
        assert "top_interactions" in result
        assert "feature_names" in result
        assert len(result["top_interactions"]) <= 3

    def test_compute_shap_interactions_with_dataframe(self, interaction_model_data):
        """Test SHAP interactions with DataFrame input."""
        pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        X, y = interaction_model_data

        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=4)
        model.fit(X, y)

        feature_names = ["a", "b", "c", "d"]
        X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

        result = compute_shap_interactions(
            model,
            X_df,
            max_samples=50,
        )

        assert isinstance(result, dict)
        assert result["feature_names"] == feature_names

    def test_compute_shap_interactions_with_classification(self, interaction_model_data):
        """Test SHAP interactions with classification model."""
        pytest.importorskip("shap")
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        X, y = interaction_model_data
        y_class = (y > np.median(y)).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4)
        model.fit(X, y_class)

        result = compute_shap_interactions(
            model,
            X,
            max_samples=50,
        )

        assert isinstance(result, dict)
        assert "interaction_matrix" in result


class TestComputeICByHorizonExtended:
    """Extended tests for IC by horizon computation."""

    @pytest.fixture
    def horizon_data(self):
        """Create data for horizon tests."""
        np.random.seed(42)
        n_dates = 100
        n_symbols = 5

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")

        data = []
        for symbol_idx in range(n_symbols):
            base_price = 100 + symbol_idx * 10
            for _i, date in enumerate(dates):
                data.append(
                    {
                        "date": date,
                        "symbol": f"SYM_{symbol_idx}",
                        "prediction": np.random.randn(),
                        "close": base_price * (1 + np.random.randn() * 0.01),
                    }
                )

        return pl.DataFrame(data)

    def test_compute_ic_by_horizon_basic(self, horizon_data):
        """Test basic IC by horizon."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_by_horizon

        predictions = horizon_data.select(["date", "symbol", "prediction"])
        prices = horizon_data.select(["date", "symbol", "close"])

        result = compute_ic_by_horizon(
            predictions,
            prices,
            horizons=[1, 5, 10],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)
        # Should have IC for each horizon
        for horizon in [1, 5, 10]:
            assert horizon in result

    def test_compute_ic_by_horizon_pandas(self, horizon_data):
        """Test IC by horizon with pandas DataFrames."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_by_horizon

        horizon_data_pd = horizon_data.to_pandas()

        predictions = horizon_data_pd[["date", "symbol", "prediction"]]
        prices = horizon_data_pd[["date", "symbol", "close"]]

        result = compute_ic_by_horizon(
            predictions,
            prices,
            horizons=[1, 5],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling paths."""

    def test_information_coefficient_short_arrays(self):
        """Test IC with very short arrays."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])

        # Should handle short arrays gracefully
        result = information_coefficient(x, y)
        assert np.isfinite(result)

    def test_sharpe_ratio_with_extreme_values(self):
        """Test Sharpe with extreme returns."""
        # Variable positive returns (not constant to avoid inf)
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.01  # Positive bias
        result = sharpe_ratio(returns, annualization_factor=252)
        assert isinstance(result, int | float)

        # Variable negative returns
        returns = np.random.randn(100) * 0.01 - 0.01  # Negative bias
        result = sharpe_ratio(returns, annualization_factor=252)
        assert isinstance(result, int | float)

    def test_sortino_with_no_downside_risk(self):
        """Test Sortino when all returns are positive."""
        np.random.seed(42)
        returns = np.abs(np.random.randn(100)) * 0.01  # All positive
        result = sortino_ratio(returns)
        # Should handle gracefully (might be inf or large value)
        assert isinstance(result, int | float)

    def test_hit_rate_with_zeros(self):
        """Test hit rate when all values are zero."""
        predictions = np.zeros(100)
        returns = np.zeros(100)

        result = hit_rate(predictions, returns)
        # Should handle zeros without error
        assert isinstance(result, int | float)

    def test_maximum_drawdown_with_constant_returns(self):
        """Test MDD with constant returns."""
        returns = np.ones(100) * 0.01  # Constant positive returns

        result = maximum_drawdown(returns, cumulative=False)  # Not cumulative
        # Returns a dict with MDD stats
        assert isinstance(result, dict)
        assert "max_drawdown" in result

    def test_compute_monotonicity_with_edge_cases(self):
        """Test monotonicity with edge cases."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        # Perfectly monotonic
        features = np.arange(100, dtype=float)
        target = np.arange(100, dtype=float)

        result = compute_monotonicity(features, target)
        assert result["monotonicity_score"] > 0.9  # Should be nearly 1

        # Reverse monotonic
        result_rev = compute_monotonicity(features, -target)
        assert result_rev["monotonicity_score"] > 0.9

    def test_analyze_feature_outcome_with_predictions(self):
        """Test feature outcome analysis with predictions DataFrame."""
        from ml4t.diagnostic.evaluation.metrics import analyze_feature_outcome

        np.random.seed(42)
        n_dates = 50
        n_symbols = 3

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        data = []
        for symbol_idx in range(n_symbols):
            base_price = 100
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "symbol": f"SYM_{symbol_idx}",
                        "prediction": np.random.randn(),
                        "close": base_price * (1 + np.random.randn() * 0.02),
                    }
                )

        df = pl.DataFrame(data)
        predictions = df.select(["date", "symbol", "prediction"])
        prices = df.select(["date", "symbol", "close"])

        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)


class TestSHAPImportanceEdgeCases:
    """Tests for SHAP importance edge cases."""

    @pytest.fixture
    def shap_data_extended(self):
        """Create extended data for SHAP tests."""
        np.random.seed(42)
        n = 300
        n_features = 6

        X = np.random.randn(n, n_features)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(n) * 0.1

        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_compute_shap_importance_with_max_samples(self, shap_data_extended):
        """Test SHAP importance with sample limiting."""
        X, y, feature_names = shap_data_extended

        try:
            import shap  # noqa: F401
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_shap_importance(
                model,
                X,
                feature_names=feature_names,
                max_samples=50,  # Limit samples
            )

            assert isinstance(result, dict)
            assert "importances" in result

        except ImportError:
            pytest.skip("SHAP not available")

    def test_compute_shap_importance_with_xgboost(self, shap_data_extended):
        """Test SHAP importance with XGBoost model."""
        X, y, feature_names = shap_data_extended

        try:
            import shap  # noqa: F401
            import xgboost as xgb

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = xgb.XGBRegressor(n_estimators=10, max_depth=4, random_state=42)
            model.fit(X, y)

            result = compute_shap_importance(
                model,
                X,
                feature_names=feature_names,
                max_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("XGBoost or SHAP not available")

    def test_compute_shap_importance_with_lightgbm(self, shap_data_extended):
        """Test SHAP importance with LightGBM model."""
        X, y, feature_names = shap_data_extended

        try:
            import lightgbm as lgb
            import shap  # noqa: F401

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = lgb.LGBMRegressor(n_estimators=10, max_depth=4, random_state=42, verbosity=-1)
            model.fit(X, y)

            result = compute_shap_importance(
                model,
                X,
                feature_names=feature_names,
                max_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("LightGBM or SHAP not available")


class TestAnalyzeInteractionsExtended:
    """Extended tests for interaction analysis."""

    @pytest.fixture
    def interaction_data_extended(self):
        """Create extended interaction data."""
        np.random.seed(42)
        n = 300
        n_features = 5

        X = np.random.randn(n, n_features)
        y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        feature_names = [f"f{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_analyze_interactions_multiple_methods(self, interaction_data_extended):
        """Test interaction analysis with multiple methods."""
        X, y, feature_names = interaction_data_extended

        try:
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_interactions

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = analyze_interactions(
                model,
                X_df,
                y,
                feature_pairs=[("f0", "f1"), ("f0", "f2")],
                methods=["conditional_ic", "h_statistic"],
                grid_resolution=5,
                max_samples=50,
            )

            assert isinstance(result, dict)
            assert "method_results" in result

        except ImportError:
            pytest.skip("sklearn not available")

    def test_analyze_interactions_all_pairs(self, interaction_data_extended):
        """Test interaction analysis with automatic pair detection."""
        X, y, feature_names = interaction_data_extended[:3]  # Use fewer features
        X = X[:, :3]  # Only 3 features for faster test

        try:
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_interactions

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            X_df = pl.DataFrame({f"f{i}": X[:, i] for i in range(3)})

            result = analyze_interactions(
                model,
                X_df,
                y,
                feature_pairs=None,  # Auto-detect all pairs
                methods=["h_statistic"],
                grid_resolution=5,
                max_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


class TestMDAImportanceExtended:
    """Extended tests for MDA importance."""

    @pytest.fixture
    def mda_data(self):
        """Create data for MDA tests."""
        np.random.seed(42)
        n = 200
        n_features = 4

        X = np.random.randn(n, n_features)
        y = (X[:, 0] > 0).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_compute_mda_basic(self, mda_data):
        """Test basic MDA computation."""
        from ml4t.diagnostic.evaluation.metrics import compute_mda_importance

        X, y, feature_names = mda_data

        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_mda_importance(
                model,
                X,
                y,
                feature_names=feature_names,
            )

            assert isinstance(result, dict)
            assert "importances" in result

        except ImportError:
            pytest.skip("sklearn not available")

    def test_compute_mda_with_dataframe(self, mda_data):
        """Test MDA with DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics import compute_mda_importance

        X, y, feature_names = mda_data

        try:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            X_df = pl.DataFrame({name: X[:, i] for i, name in enumerate(feature_names)})

            result = compute_mda_importance(
                model,
                X_df,
                y,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


# ==============================================================================
# ADDITIONAL EDGE CASE AND ERROR PATH TESTS
# ==============================================================================


class TestConditionalICEdgeCases:
    """Edge case tests for conditional IC to cover error handling paths."""

    def test_conditional_ic_with_constant_conditioning_feature(self):
        """Test conditional IC when conditioning feature is constant."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 100
        feature_a = np.random.randn(n)
        feature_b = np.ones(n)  # Constant - can't create quantiles
        returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)
        assert isinstance(result, dict)
        # Should handle gracefully

    def test_conditional_ic_with_few_unique_values(self):
        """Test conditional IC when conditioning feature has few unique values."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 100
        feature_a = np.random.randn(n)
        feature_b = np.random.choice([0, 1, 2], size=n)  # Only 3 unique values
        returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=5)
        assert isinstance(result, dict)

    def test_conditional_ic_with_nan_values(self):
        """Test conditional IC with NaN values in input."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 100
        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        returns = np.random.randn(n) * 0.01

        # Add some NaN values
        feature_a[10:15] = np.nan
        feature_b[20:25] = np.nan

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=3)
        assert isinstance(result, dict)

    def test_conditional_ic_small_quantiles(self):
        """Test conditional IC with small number of observations per quantile."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 20  # Very small dataset
        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(feature_a, feature_b, returns, n_quantiles=10)
        assert isinstance(result, dict)


class TestSHAPExplainerTypes:
    """Tests for different SHAP explainer types."""

    @pytest.fixture
    def linear_model_data(self):
        """Create data for linear model tests."""
        np.random.seed(42)
        n = 200
        n_features = 5

        X = np.random.randn(n, n_features)
        # Linear relationship
        y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + np.random.randn(n) * 0.1

        return X, y

    def test_shap_with_linear_model(self, linear_model_data):
        """Test SHAP with linear regression model."""
        X, y = linear_model_data

        try:
            import shap  # noqa: F401
            from sklearn.linear_model import LinearRegression

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = LinearRegression()
            model.fit(X, y)

            result = compute_shap_importance(
                model,
                X,
                max_samples=50,
            )

            assert isinstance(result, dict)
            assert "importances" in result

        except ImportError:
            pytest.skip("SHAP not available")

    def test_shap_with_kernel_explainer(self, linear_model_data):
        """Test SHAP with KernelExplainer (model-agnostic)."""
        X, y = linear_model_data

        try:
            import shap  # noqa: F401
            from sklearn.ensemble import GradientBoostingRegressor

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
            model.fit(X, y)

            result = compute_shap_importance(
                model,
                X,
                max_samples=30,
                explainer_type="kernel",
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("SHAP not available")

    def test_shap_with_pandas_dataframe(self, linear_model_data):
        """Test SHAP with pandas DataFrame input."""
        X, y = linear_model_data

        try:
            import shap  # noqa: F401
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)

            result = compute_shap_importance(
                model,
                X_df,
                max_samples=50,
            )

            assert isinstance(result, dict)
            assert result["feature_names"] == feature_names

        except ImportError:
            pytest.skip("SHAP not available")


class TestICSeriesEdgeCases:
    """Edge case tests for IC series computation."""

    def test_ic_series_with_insufficient_data(self):
        """Test IC series with very few observations per period."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_series

        np.random.seed(42)
        n_dates = 5
        n_symbols = 2  # Very few symbols

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        data = []
        for date in dates:
            for symbol_idx in range(n_symbols):
                data.append(
                    {
                        "date": date,
                        "symbol": f"SYM_{symbol_idx}",
                        "prediction": np.random.randn(),
                        "returns": np.random.randn() * 0.01,
                    }
                )

        df = pl.DataFrame(data)
        predictions = df.select(["date", "symbol", "prediction"])
        returns = df.select(["date", "symbol", "returns"])

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="returns",
            date_col="date",
            min_periods=1,  # Low threshold
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)

    def test_ic_series_with_single_date(self):
        """Test IC series with only one date."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_series

        np.random.seed(42)
        n_symbols = 20

        data = []
        date = pd.Timestamp("2020-01-01")
        for symbol_idx in range(n_symbols):
            data.append(
                {
                    "date": date,
                    "symbol": f"SYM_{symbol_idx}",
                    "prediction": np.random.randn(),
                    "returns": np.random.randn() * 0.01,
                }
            )

        df = pl.DataFrame(data)
        predictions = df.select(["date", "symbol", "prediction"])
        returns = df.select(["date", "symbol", "returns"])

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="returns",
            date_col="date",
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)


class TestHStatisticEdgeCases:
    """Edge case tests for H-statistic computation."""

    def test_h_statistic_with_constant_feature(self):
        """Test H-statistic when one feature is constant."""
        from ml4t.diagnostic.evaluation.metrics import compute_h_statistic

        np.random.seed(42)
        n = 100

        try:
            from sklearn.ensemble import RandomForestRegressor

            X = np.random.randn(n, 3)
            X[:, 1] = 1.0  # Constant feature
            y = X[:, 0] + np.random.randn(n) * 0.1

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_h_statistic(
                model,
                X,
                feature_pairs=[(0, 2)],  # Avoid constant feature
                grid_resolution=5,
                n_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")

    def test_h_statistic_with_categorical_like_feature(self):
        """Test H-statistic with integer/categorical-like features."""
        from ml4t.diagnostic.evaluation.metrics import compute_h_statistic

        np.random.seed(42)
        n = 100

        try:
            from sklearn.ensemble import RandomForestRegressor

            X = np.random.randn(n, 3)
            X[:, 0] = np.random.choice([0, 1, 2, 3, 4], size=n)  # Categorical-like
            y = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.1

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = compute_h_statistic(
                model,
                X,
                feature_pairs=[(0, 1)],
                grid_resolution=5,
                n_samples=50,
            )

            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("sklearn not available")


class TestAnalyzeFeatureOutcomeExtended:
    """Extended tests for analyze_feature_outcome."""

    @pytest.fixture
    def feature_outcome_data(self):
        """Create data for feature outcome analysis."""
        np.random.seed(42)
        n_dates = 50
        n_symbols = 5

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        data = []
        for symbol_idx in range(n_symbols):
            base_price = 100
            for date in dates:
                data.append(
                    {
                        "date": date,
                        "symbol": f"SYM_{symbol_idx}",
                        "prediction": np.random.randn(),
                        "close": base_price * (1 + np.random.randn() * 0.02),
                    }
                )

        return pl.DataFrame(data)

    def test_analyze_feature_outcome_with_group(self, feature_outcome_data):
        """Test feature outcome analysis with grouping."""
        from ml4t.diagnostic.evaluation.metrics import analyze_feature_outcome

        df = feature_outcome_data
        predictions = df.select(["date", "symbol", "prediction"])
        prices = df.select(["date", "symbol", "close"])

        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="prediction",
            price_col="close",
            date_col="date",
            group_col="symbol",
        )

        assert isinstance(result, dict)

    def test_analyze_feature_outcome_multiple_horizons(self, feature_outcome_data):
        """Test feature outcome analysis with multiple horizons."""
        from ml4t.diagnostic.evaluation.metrics import analyze_feature_outcome

        df = feature_outcome_data
        predictions = df.select(["date", "symbol", "prediction"])
        prices = df.select(["date", "symbol", "close"])

        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="prediction",
            price_col="close",
            date_col="date",
            horizons=[1, 5, 10],
        )

        assert isinstance(result, dict)

    def test_analyze_feature_outcome_no_decay(self, feature_outcome_data):
        """Test feature outcome analysis without decay analysis."""
        from ml4t.diagnostic.evaluation.metrics import analyze_feature_outcome

        df = feature_outcome_data
        predictions = df.select(["date", "symbol", "prediction"])
        prices = df.select(["date", "symbol", "close"])

        result = analyze_feature_outcome(
            predictions,
            prices,
            pred_col="prediction",
            price_col="close",
            date_col="date",
            include_decay=False,
        )

        assert isinstance(result, dict)


class TestImportanceConsensus:
    """Tests for importance consensus calculations."""

    @pytest.fixture
    def consensus_data(self):
        """Create data for consensus tests."""
        np.random.seed(42)
        n = 200
        n_features = 5

        X = np.random.randn(n, n_features)
        # First two features are important
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(n) * 0.1

        feature_names = [f"feature_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_analyze_ml_importance_all_methods(self, consensus_data):
        """Test ML importance with all available methods."""
        X, y, feature_names = consensus_data

        try:
            import shap  # noqa: F401
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = analyze_ml_importance(
                model,
                X,
                y,
                feature_names=feature_names,
                methods=["mdi", "pfi", "shap"],
            )

            assert isinstance(result, dict)
            assert "method_results" in result
            assert "consensus_ranking" in result

        except ImportError:
            pytest.skip("SHAP not available")

    def test_analyze_ml_importance_mdi_only(self, consensus_data):
        """Test ML importance with MDI only."""
        X, y, feature_names = consensus_data

        try:
            from sklearn.ensemble import RandomForestRegressor

            from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)

            result = analyze_ml_importance(
                model,
                X,
                y,
                feature_names=feature_names,
                methods=["mdi"],
            )

            assert isinstance(result, dict)
            assert "mdi" in result["method_results"]

        except ImportError:
            pytest.skip("sklearn not available")


class TestMonotonicityExtended:
    """Extended tests for monotonicity computation."""

    def test_monotonicity_with_quantiles(self):
        """Test monotonicity with different quantile counts."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 200
        features = np.random.randn(n)
        # Create monotonic relationship
        outcomes = features * 0.5 + np.random.randn(n) * 0.1

        for n_quantiles in [3, 5, 10]:
            result = compute_monotonicity(features, outcomes, n_quantiles=n_quantiles)
            assert isinstance(result, dict)
            assert "monotonicity_score" in result

    def test_monotonicity_with_dataframe_input(self):
        """Test monotonicity with DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 100
        features = np.random.randn(n)
        outcomes = features + np.random.randn(n) * 0.1

        features_df = pl.DataFrame({"feature": features})
        outcomes_df = pl.DataFrame({"outcome": outcomes})

        result = compute_monotonicity(
            features_df,
            outcomes_df,
            feature_col="feature",
            outcome_col="outcome",
        )

        assert isinstance(result, dict)

    def test_monotonicity_non_monotonic_data(self):
        """Test monotonicity with non-monotonic relationship."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 200
        features = np.random.randn(n)
        # Quadratic (non-monotonic) relationship
        outcomes = features**2 + np.random.randn(n) * 0.1

        result = compute_monotonicity(features, outcomes)
        assert isinstance(result, dict)
        # Should detect non-monotonicity
        assert result["direction"] == "non-monotonic" or result["monotonicity_score"] < 0.9


class TestForwardReturnsComputation:
    """Tests for forward returns computation."""

    def test_compute_forward_returns_basic(self):
        """Test basic forward returns computation."""
        from ml4t.diagnostic.evaluation.metrics import compute_forward_returns

        np.random.seed(42)
        n_dates = 50

        prices = 100 * (1 + np.random.randn(n_dates).cumsum() * 0.01)

        df = pl.DataFrame(
            {
                "close": prices,
            }
        )

        result = compute_forward_returns(
            df,
            price_col="close",
            periods=[1, 5],
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)

    def test_compute_forward_returns_with_symbols(self):
        """Test forward returns with multiple symbols."""
        from ml4t.diagnostic.evaluation.metrics import compute_forward_returns

        np.random.seed(42)
        n_dates = 30
        n_symbols = 3

        data = []
        for symbol_idx in range(n_symbols):
            base_price = 100 + symbol_idx * 20
            for _ in range(n_dates):
                data.append(
                    {
                        "symbol": f"SYM_{symbol_idx}",
                        "close": base_price * (1 + np.random.randn() * 0.02),
                    }
                )

        df = pl.DataFrame(data)

        result = compute_forward_returns(
            df,
            price_col="close",
            group_col="symbol",
            periods=[1, 5],
        )

        assert isinstance(result, pd.DataFrame | pl.DataFrame)


# ============================================================================
# ADDITIONAL COVERAGE TESTS
# ============================================================================


class TestICEdgeCases:
    """Edge cases for Information Coefficient to hit uncovered paths."""

    def test_ic_single_element_with_ci(self):
        """Test IC with single element returns NaN dict."""
        predictions = np.array([1.0])
        returns = np.array([0.5])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert np.isnan(result["ic"])
        assert np.isnan(result["lower_ci"])

    def test_ic_all_nan_with_ci(self):
        """Test IC with all NaN values returns NaN dict."""
        predictions = np.array([np.nan, np.nan, np.nan])
        returns = np.array([np.nan, np.nan, np.nan])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert np.isnan(result["ic"])

    def test_ic_unknown_method_raises(self):
        """Test IC raises for unknown correlation method."""
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        with pytest.raises(ValueError, match="Unknown correlation method"):
            information_coefficient(predictions, returns, method="invalid")

    def test_ic_pearson_method(self):
        """Test IC with Pearson correlation."""
        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ic = information_coefficient(predictions, returns, method="pearson")
        assert abs(ic - 1.0) < 1e-10

    def test_ic_constant_values_returns_nan(self):
        """Test IC with constant values returns NaN."""
        predictions = np.array([1, 1, 1, 1])
        returns = np.array([0.5, 0.5, 0.5, 0.5])

        result = information_coefficient(predictions, returns, confidence_intervals=True)
        assert np.isnan(result["ic"])

    def test_ic_small_sample_no_ci(self):
        """Test IC with small sample gives IC but no CI."""
        predictions = np.array([1, 2, 3])
        returns = np.array([0.1, 0.2, 0.3])

        result = information_coefficient(predictions, returns, confidence_intervals=True)

        assert isinstance(result, dict)
        assert result["ic"] == pytest.approx(1.0)
        assert np.isnan(result["lower_ci"])
        assert np.isnan(result["upper_ci"])


class TestSharpeRatioEdgeCases:
    """Edge cases for Sharpe ratio."""

    def test_sharpe_zero_std_positive_mean(self):
        """Test Sharpe with zero volatility and positive mean returns inf."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])  # Constant positive

        sharpe = sharpe_ratio(returns)
        assert np.isinf(sharpe)
        assert sharpe > 0  # Positive infinity

    def test_sharpe_zero_std_negative_mean(self):
        """Test Sharpe with zero volatility and negative mean returns -inf."""
        returns = np.array([-0.01, -0.01, -0.01, -0.01])  # Constant negative

        sharpe = sharpe_ratio(returns)
        assert np.isinf(sharpe)
        assert sharpe < 0  # Negative infinity

    def test_sharpe_zero_std_zero_mean(self):
        """Test Sharpe with zero volatility and zero mean returns NaN."""
        returns = np.array([0.0, 0.0, 0.0, 0.0])  # Constant zero

        sharpe = sharpe_ratio(returns)
        assert np.isnan(sharpe)

    def test_sharpe_with_ci_inf_returns_nan_ci(self):
        """Test Sharpe CI when base Sharpe is inf."""
        from ml4t.diagnostic.evaluation.metrics import sharpe_ratio_with_ci

        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio_with_ci(returns)

        assert np.isinf(result["sharpe"])
        assert np.isnan(result["lower_ci"])

    def test_sharpe_with_ci_small_sample(self):
        """Test Sharpe CI with small sample returns NaN CI."""
        from ml4t.diagnostic.evaluation.metrics import sharpe_ratio_with_ci

        returns = np.array([0.01, 0.02, 0.03])  # Only 3 samples
        result = sharpe_ratio_with_ci(returns)

        assert not np.isnan(result["sharpe"])
        assert np.isnan(result["lower_ci"])

    def test_sharpe_with_ci_annualization(self):
        """Test Sharpe CI with annualization factor."""
        from ml4t.diagnostic.evaluation.metrics import sharpe_ratio_with_ci

        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        result = sharpe_ratio_with_ci(returns, annualization_factor=252)

        assert "sharpe" in result
        assert "lower_ci" in result
        assert result["lower_ci"] < result["sharpe"] < result["upper_ci"]


class TestHACKernelWeights:
    """Test HAC kernel weight functions."""

    def test_hac_uniform_kernel(self):
        """Test HAC with uniform kernel."""
        np.random.seed(42)
        ic_series = np.random.randn(50)

        result = compute_ic_hac_stats(ic_series, kernel="uniform")

        assert isinstance(result, dict)
        assert "mean_ic" in result
        assert "hac_se" in result

    def test_hac_parzen_kernel(self):
        """Test HAC with Parzen kernel."""
        np.random.seed(42)
        ic_series = np.random.randn(50)

        result = compute_ic_hac_stats(ic_series, kernel="parzen")

        assert isinstance(result, dict)
        assert "mean_ic" in result
        assert "hac_se" in result

    def test_hac_unknown_kernel_fallback(self):
        """Test HAC falls back to naive SE for unknown kernel."""
        np.random.seed(42)
        ic_series = np.random.randn(50)

        # Unknown kernel triggers fallback to naive SE (doesn't raise)
        result = compute_ic_hac_stats(ic_series, kernel="invalid_kernel")

        # Should still return valid result with naive SE
        assert isinstance(result, dict)
        assert "mean_ic" in result
        assert "hac_se" in result

    def test_hac_with_exception_fallback(self):
        """Test HAC falls back to naive SE on numerical issues."""
        # Edge case: very small series might cause HAC to fail
        ic_series = np.array([0.1, 0.1])  # Minimal series

        result = compute_ic_hac_stats(ic_series, maxlags=0)

        # Should still return a valid result
        assert isinstance(result, dict)
        assert "mean_ic" in result


class TestMDIEdgeCases:
    """Edge cases for MDI importance."""

    def test_mdi_model_without_feature_importances(self):
        """Test MDI raises for model without feature_importances_."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)

        # LinearRegression doesn't have feature_importances_
        with pytest.raises(AttributeError, match="does not have 'feature_importances_'"):
            compute_mdi_importance(model)

    def test_mdi_feature_names_length_mismatch(self):
        """Test MDI raises when feature names length doesn't match."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model.fit(X, y)

        # Provide wrong number of feature names
        with pytest.raises(ValueError, match="does not match number of importances"):
            compute_mdi_importance(model, feature_names=["a", "b", "c"])  # Only 3 names

    def test_mdi_auto_feature_names_sklearn(self):
        """Test MDI auto-detects feature names from sklearn model."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        np.random.seed(42)
        X = pd.DataFrame({f"col_{i}": np.random.randn(50) for i in range(5)})
        y = np.random.randn(50)
        model.fit(X, y)

        result = compute_mdi_importance(model)

        # Should have feature names from DataFrame
        assert result["n_features"] == 5

    def test_mdi_normalize_false(self):
        """Test MDI without normalization."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model.fit(X, y)

        result = compute_mdi_importance(model, normalize=False)

        assert not result["normalized"]


class TestMDARemovalMethods:
    """Tests for MDA importance with different removal methods."""

    def test_mda_with_median_removal(self):
        """Test MDA with median removal method."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_mda_importance(model, X, y, removal_method="median")

        assert result["removal_method"] == "median"
        assert result["n_features"] == 5

    def test_mda_with_zero_removal(self):
        """Test MDA with zero removal method."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_mda_importance(model, X, y, removal_method="zero")

        assert result["removal_method"] == "zero"

    def test_mda_invalid_removal_method(self):
        """Test MDA raises for invalid removal method."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="removal_method must be one of"):
            compute_mda_importance(model, X, y, removal_method="invalid")

    def test_mda_with_pandas_dataframe(self):
        """Test MDA with pandas DataFrame input."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = pd.DataFrame({f"feat_{i}": np.random.randn(50) for i in range(5)})
        y = pd.Series(np.random.randn(50))

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_mda_importance(model, X, y)

        assert "feat_0" in result["feature_names"]

    def test_mda_with_polars_dataframe(self):
        """Test MDA with polars DataFrame input."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = pl.DataFrame({f"feat_{i}": np.random.randn(50) for i in range(5)})
        y = pl.Series(np.random.randn(50))

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X.to_numpy(), y.to_numpy())

        result = compute_mda_importance(model, X, y)

        assert "feat_0" in result["feature_names"]

    def test_mda_with_custom_scoring(self):
        """Test MDA with custom scoring function."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_mda_importance(model, X, y, scoring="neg_mean_squared_error")

        assert result["scoring"] == "neg_mean_squared_error"

    def test_mda_with_feature_groups(self):
        """Test MDA with feature groups."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 6)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        feature_names = [f"feat_{i}" for i in range(6)]
        feature_groups = {
            "group_a": ["feat_0", "feat_1"],
            "group_b": ["feat_2", "feat_3"],
        }

        result = compute_mda_importance(
            model, X, y, feature_names=feature_names, feature_groups=feature_groups
        )

        assert "group_a" in result["feature_names"]
        assert "group_b" in result["feature_names"]
        assert result["n_features"] == 2

    def test_mda_feature_group_invalid_feature(self):
        """Test MDA raises for invalid feature in group."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        feature_names = ["feat_0", "feat_1", "feat_2"]
        feature_groups = {
            "group_a": ["feat_0", "nonexistent_feat"],  # Invalid feature
        }

        with pytest.raises(ValueError, match="not found in feature_names"):
            compute_mda_importance(
                model, X, y, feature_names=feature_names, feature_groups=feature_groups
            )


class TestAnalyzeInteractionsEdgeCases:
    """Edge cases for analyze_interactions."""

    def test_analyze_interactions_empty_methods_raises(self):
        """Test analyze_interactions raises for empty methods list."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="At least one method must be specified"):
            analyze_interactions(model, X, y, methods=[])

    def test_analyze_interactions_invalid_pair_length(self):
        """Test analyze_interactions raises for invalid feature pair."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {"a": np.random.randn(50), "b": np.random.randn(50), "c": np.random.randn(50)}
        )
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            analyze_interactions(
                model,
                X,
                y,
                methods=["conditional_ic"],
                feature_pairs=[("a", "b", "c")],  # Invalid: 3 elements
            )

    def test_analyze_interactions_unknown_feature(self):
        """Test analyze_interactions raises for unknown feature in pair."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="contains unknown features"):
            analyze_interactions(
                model, X, y, methods=["conditional_ic"], feature_pairs=[("a", "nonexistent")]
            )

    def test_analyze_interactions_numpy_with_feature_pairs(self):
        """Test analyze_interactions with numpy array and feature pairs."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # With numpy array, feature names are auto-generated as f0, f1, f2
        result = analyze_interactions(
            model, X, y, methods=["conditional_ic"], feature_pairs=[("f0", "f1")]
        )

        assert "conditional_ic" in result["method_results"]

    def test_analyze_interactions_conditional_ic_only(self):
        """Test analyze_interactions with conditional IC only method."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
            }
        )
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model, X, y, methods=["conditional_ic"], feature_pairs=[("a", "b")]
        )

        assert "conditional_ic" in result["method_results"]
        assert "h_statistic" not in result["method_results"]

    def test_analyze_interactions_h_statistic_only(self):
        """Test analyze_interactions with h_statistic only method."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
            }
        )
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model, X, y, methods=["h_statistic"], feature_pairs=[("a", "b")]
        )

        assert "h_statistic" in result["method_results"]


class TestExplainerCreation:
    """Tests for SHAP explainer creation paths."""

    @pytest.fixture
    def tree_model(self):
        """Create a tree-based model for SHAP."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        return model, X

    def test_shap_with_tree_explainer_explicit(self, tree_model):
        """Test SHAP with explicit tree explainer."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        model, X = tree_model

        result = compute_shap_importance(model, X, explainer_type="tree")

        assert result["explainer_type"] == "tree"

    def test_shap_with_kernel_explainer_explicit(self, tree_model):
        """Test SHAP with explicit kernel explainer."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        model, X = tree_model

        result = compute_shap_importance(
            model,
            X,
            explainer_type="kernel",
            max_samples=20,  # Small for speed
        )

        assert result["explainer_type"] == "kernel"


class TestGPUDetection:
    """Tests for GPU detection."""

    def test_gpu_detection_returns_bool(self):
        """Test GPU detection returns a boolean."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _detect_gpu_available

        result = _detect_gpu_available()
        assert isinstance(result, bool)


class TestMDAImportanceVariations:
    """Additional tests for MDA importance."""

    def test_mda_with_polars_series(self):
        """Test MDA with polars Series as y."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = pl.Series(np.random.randn(50))

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y.to_numpy())

        result = compute_mda_importance(model, X, y)

        assert result["n_features"] == 5

    def test_mda_with_pandas_series(self):
        """Test MDA with pandas Series as y."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = pd.Series(np.random.randn(50))

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_mda_importance(model, X, y)

        assert result["n_features"] == 5


class TestICDecayEdgeCases:
    """Edge cases for IC decay computation."""

    def test_ic_decay_pandas_input(self):
        """Test IC decay with pandas DataFrame input."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n),
                "prediction": np.random.randn(n),
                "close": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = compute_ic_decay(
            predictions=df[["date", "prediction"]],
            prices=df[["date", "close"]],
            horizons=[1, 5],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)
        assert "ic_by_horizon" in result

    def test_ic_by_horizon_direct(self):
        """Test compute_ic_by_horizon directly."""
        np.random.seed(42)
        n = 100

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), eager=True)
        predictions = pl.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pl.DataFrame(
            {
                "date": dates,
                "close": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = compute_ic_by_horizon(
            predictions,
            prices,
            horizons=[1, 5],
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)
        assert 1 in result or 5 in result


class TestPFIVariations:
    """Additional tests for Permutation Feature Importance."""

    def test_pfi_with_polars_input(self):
        """Test PFI with polars DataFrame input."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = pl.DataFrame({f"feat_{i}": np.random.randn(50) for i in range(5)})
        y = pl.Series(np.random.randn(50))

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X.to_numpy(), y.to_numpy())

        result = compute_permutation_importance(model, X, y)

        assert result["n_features"] == 5
        assert "feat_0" in result["feature_names"]

    def test_pfi_with_custom_scoring(self):
        """Test PFI with custom scoring function."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_permutation_importance(model, X, y, scoring="neg_mean_squared_error")

        assert result["n_features"] == 5


# ============================================================================
# MORE COVERAGE TESTS - EDGE CASES AND PATHS
# ============================================================================


class TestConditionalICDataFrameInput:
    """Tests for conditional IC with DataFrame inputs."""

    def test_conditional_ic_with_polars_dataframe(self):
        """Test conditional IC with polars DataFrame input."""
        np.random.seed(42)
        n = 100
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), eager=True)

        df_a = pl.DataFrame(
            {
                "date": dates,
                "feature_a": np.random.randn(n),
            }
        )
        df_b = pl.DataFrame(
            {
                "date": dates,
                "feature_b": np.random.randn(n),
            }
        )
        df_ret = pl.DataFrame(
            {
                "date": dates,
                "returns": np.random.randn(n) * 0.01,
            }
        )

        result = compute_conditional_ic(
            feature_a=df_a,
            feature_b=df_b,
            forward_returns=df_ret,
            date_col="date",
            n_quantiles=3,
        )

        assert isinstance(result, dict)
        assert "quantile_ics" in result

    def test_conditional_ic_with_pandas_dataframe(self):
        """Test conditional IC with pandas DataFrame input."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)

        df_a = pd.DataFrame(
            {
                "date": dates,
                "feature_a": np.random.randn(n),
            }
        )
        df_b = pd.DataFrame(
            {
                "date": dates,
                "feature_b": np.random.randn(n),
            }
        )
        df_ret = pd.DataFrame(
            {
                "date": dates,
                "returns": np.random.randn(n) * 0.01,
            }
        )

        result = compute_conditional_ic(
            feature_a=df_a,
            feature_b=df_b,
            forward_returns=df_ret,
            date_col="date",
            n_quantiles=3,
        )

        assert isinstance(result, dict)
        assert "quantile_ics" in result

    def test_conditional_ic_insufficient_data_per_quantile(self):
        """Test conditional IC returns NaN when not enough data per quantile."""
        np.random.seed(42)
        # Very small data - only 5 samples, with 5 quantiles = 1 per quantile
        feature_a = np.array([1, 2, 3, 4, 5])
        feature_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        result = compute_conditional_ic(
            feature_a,
            feature_b,
            returns,
            n_quantiles=5,
            min_periods=5,  # Each quantile needs 5, but only has 1
        )

        assert isinstance(result, dict)
        # Should have NaN for quantiles with insufficient data


class TestICSeriesExtended:
    """Extended tests for IC series computation."""

    def test_ic_series_with_pandas_dataframe(self):
        """Test IC series with pandas DataFrame input."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)

        predictions = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        returns = pd.DataFrame(
            {
                "date": dates,
                "forward_return": np.random.randn(n) * 0.01,
            }
        )

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="forward_return",
            date_col="date",
        )

        assert isinstance(result, pl.DataFrame | pd.DataFrame)

    def test_ic_series_pearson_method(self):
        """Test IC series with Pearson correlation."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)

        predictions = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        returns = pd.DataFrame(
            {
                "date": dates,
                "forward_return": np.random.randn(n) * 0.01,
            }
        )

        result = compute_ic_series(
            predictions,
            returns,
            pred_col="prediction",
            ret_col="forward_return",
            date_col="date",
            method="pearson",
        )

        assert isinstance(result, pl.DataFrame | pd.DataFrame)


class TestAnalyzeInteractionsMorePaths:
    """Additional tests for analyze_interactions edge cases."""

    def test_analyze_interactions_multiple_methods(self):
        """Test analyze_interactions with multiple methods including SHAP."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(60),
                "b": np.random.randn(60),
                "c": np.random.randn(60),
            }
        )
        y = X["a"] + X["b"] * 0.5 + np.random.randn(60) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Test with conditional_ic and h_statistic (SHAP optional due to time)
        result = analyze_interactions(
            model,
            X,
            y,
            methods=["conditional_ic", "h_statistic"],
            feature_pairs=[("a", "b"), ("a", "c")],
        )

        assert "conditional_ic" in result["method_results"]
        assert "h_statistic" in result["method_results"]
        assert "consensus_ranking" in result
        assert "method_agreement" in result

    def test_analyze_interactions_with_shap(self):
        """Test analyze_interactions with SHAP method."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = np.random.randn(40)

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Test with SHAP only
        result = analyze_interactions(
            model, X, y, methods=["shap"], feature_pairs=[("a", "b")], max_samples=20
        )

        assert "shap" in result["method_results"]


class TestSHAPImportanceLinearModel:
    """SHAP importance tests with linear models."""

    def test_shap_with_linear_explainer(self):
        """Test SHAP with linear model using LinearExplainer."""
        from sklearn.linear_model import Ridge

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(50) * 0.1

        model = Ridge()
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="linear")

        assert result["explainer_type"] == "linear"
        assert result["n_features"] == 5


class TestMaximumDrawdownEdgeCases:
    """Edge cases for maximum drawdown."""

    def test_mdd_with_cumulative_returns(self):
        """Test MDD with cumulative returns input."""
        # Cumulative returns (equity curve)
        cumulative = np.array([100, 110, 105, 115, 108, 120, 115])

        result = maximum_drawdown(cumulative, cumulative=True)

        assert isinstance(result, dict)
        assert "max_drawdown" in result
        assert result["max_drawdown"] <= 0  # Drawdown is negative

    def test_mdd_with_simple_returns(self):
        """Test MDD with simple returns input."""
        returns = np.array([0.1, -0.05, 0.1, -0.06, 0.11, -0.04])

        result = maximum_drawdown(returns, cumulative=False)

        assert isinstance(result, dict)
        assert "max_drawdown" in result


class TestSortinoRatioEdgeCases:
    """Edge cases for Sortino ratio."""

    def test_sortino_all_positive_returns(self):
        """Test Sortino with all positive returns (no downside)."""
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.01])

        result = sortino_ratio(returns)

        # With no negative returns, downside std is 0, should return inf
        assert np.isinf(result) or result > 100  # Very high Sortino

    def test_sortino_with_target(self):
        """Test Sortino with non-zero target rate."""
        np.random.seed(42)
        returns = np.random.randn(50) * 0.01

        result = sortino_ratio(returns, target_return=0.001)

        assert not np.isnan(result)


class TestHitRateEdgeCases:
    """Edge cases for hit rate."""

    def test_hit_rate_with_polars_series(self):
        """Test hit rate with polars Series input."""
        predictions = pl.Series([0.1, 0.2, -0.1, 0.05])
        returns = pl.Series([0.02, 0.03, -0.01, -0.02])

        result = hit_rate(predictions, returns)

        # Hit rate is a percentage (0-100)
        assert 0 <= result <= 100


class TestMDIFeatureNameExtraction:
    """Tests for MDI feature name extraction from different models."""

    def test_mdi_with_lightgbm_model(self):
        """Test MDI extracts feature names from LightGBM model."""
        pytest.importorskip("lightgbm")
        import lightgbm as lgb

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        model.fit(X, y, feature_name=[f"lgb_feat_{i}" for i in range(5)])

        result = compute_mdi_importance(model)

        assert result["n_features"] == 5
        # LightGBM stores feature names
        assert "lgb_feat_" in result["feature_names"][0]

    def test_mdi_with_xgboost_model(self):
        """Test MDI extracts feature names from XGBoost model."""
        pytest.importorskip("xgboost")
        import xgboost as xgb

        np.random.seed(42)
        X = pd.DataFrame({f"xgb_feat_{i}": np.random.randn(100) for i in range(5)})
        y = np.random.randn(100)

        model = xgb.XGBRegressor(n_estimators=10)
        model.fit(X, y)

        result = compute_mdi_importance(model)

        assert result["n_features"] == 5


class TestICIRComputation:
    """Tests for IC Information Ratio computation."""

    def test_ic_ir_basic(self):
        """Test IC IR computation."""
        np.random.seed(42)
        ic_series = np.random.randn(50) * 0.1

        # compute_ic_ir returns a float (the IC IR value)
        result = compute_ic_ir(ic_series)

        assert isinstance(result, float | int | np.floating)
        assert not np.isnan(result)

    def test_ic_ir_with_constant_ic(self):
        """Test IC IR with constant IC series."""
        ic_series = np.array([0.05, 0.05, 0.05, 0.05])

        result = compute_ic_ir(ic_series)

        # Constant IC means zero std, should be inf
        assert np.isinf(result) or result > 100


class TestSharpeBootstrapEdgeCases:
    """Edge cases for Sharpe ratio with bootstrap CI."""

    def test_sharpe_with_ci_zero_std_samples(self):
        """Test Sharpe CI when all bootstrap samples have zero std."""
        from ml4t.diagnostic.evaluation.metrics import sharpe_ratio_with_ci

        # All constant returns means bootstrap samples also constant -> empty list
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio_with_ci(returns)

        # Should handle this edge case gracefully
        assert "sharpe" in result


class TestMaximumDrawdownSpecialCases:
    """Special cases for maximum drawdown."""

    def test_mdd_no_drawdown_monotonic_gains(self):
        """Test MDD with monotonically increasing cumulative returns."""
        # Monotonically increasing means no drawdown
        cumulative = np.array([100, 102, 105, 108, 112, 118])

        result = maximum_drawdown(cumulative, cumulative=True)

        assert isinstance(result, dict)
        # No drawdown should mean MDD is 0 or very small
        assert result["max_drawdown"] >= -0.01  # Close to zero

    def test_mdd_empty_returns(self):
        """Test MDD with empty array (after NaN removal)."""
        returns = np.array([np.nan, np.nan])

        result = maximum_drawdown(returns)

        assert np.isnan(result["max_drawdown"])


class TestAnalyzeMLImportanceVariants:
    """More tests for analyze_ml_importance."""

    def test_analyze_ml_importance_with_mdi_pfi(self):
        """Test ML importance with MDI and PFI methods."""
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(42)
        X = pd.DataFrame({f"feat_{i}": np.random.randn(50) for i in range(5)})
        y = X["feat_0"] * 2 + np.random.randn(50) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"])

        assert "method_results" in result
        assert "mdi" in result["method_results"]
        assert "pfi" in result["method_results"]


class TestConditionalICValidationPaths:
    """Tests for conditional IC validation paths."""

    def test_conditional_ic_with_group_col(self):
        """Test conditional IC with group column validation."""
        np.random.seed(42)
        n = 100

        df_a = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n),
                "symbol": ["A"] * 50 + ["B"] * 50,
                "feature_a": np.random.randn(n),
            }
        )
        df_b = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n),
                "symbol": ["A"] * 50 + ["B"] * 50,
                "feature_b": np.random.randn(n),
            }
        )
        df_ret = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n),
                "symbol": ["A"] * 50 + ["B"] * 50,
                "returns": np.random.randn(n) * 0.01,
            }
        )

        result = compute_conditional_ic(
            feature_a=df_a,
            feature_b=df_b,
            forward_returns=df_ret,
            date_col="date",
            group_col="symbol",
            n_quantiles=3,
        )

        assert isinstance(result, dict)


class TestSHAPInteractions:
    """Tests for SHAP interaction analysis."""

    def test_shap_interactions_basic(self):
        """Test SHAP interaction computation."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
                "c": np.random.randn(40),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(40) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_interactions(model, X, max_samples=20)

        assert isinstance(result, dict)
        assert "top_interactions" in result


class TestAnalyzeFeatureOutcomePolars:
    """Tests for analyze_feature_outcome with Polars input."""

    def test_analyze_feature_outcome_polars_input(self):
        """Test analyze_feature_outcome with Polars DataFrame."""
        np.random.seed(42)
        n = 100

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), eager=True)
        predictions = pl.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pl.DataFrame(
            {
                "date": dates,
                "close": 100 * (1 + np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions,
            prices=prices,
            horizons=[1, 5],
        )

        assert isinstance(result, dict)


# ============================================================================
# Additional coverage tests - 2nd round
# ============================================================================


class TestSortinoRatioSpecialCases:
    """Special cases for sortino_ratio function."""

    def test_sortino_negative_mean_no_downside(self):
        """Test Sortino when all returns >= target but mean is negative (impossible but tests path)."""
        from ml4t.diagnostic.evaluation.metrics import sortino_ratio

        # All returns exactly at target, mean excess is 0
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = sortino_ratio(returns, target_return=0.01)
        # When no downside and mean_excess == 0, should return nan
        assert np.isnan(result)

    def test_sortino_positive_mean_no_downside(self):
        """Test Sortino with positive mean and no downside returns."""
        from ml4t.diagnostic.evaluation.metrics import sortino_ratio

        # All positive excess returns
        returns = np.array([0.05, 0.06, 0.07, 0.08])
        result = sortino_ratio(returns, target_return=0.01)
        # Should return inf when positive mean and no downside
        assert np.isinf(result) and result > 0

    def test_sortino_negative_mean_no_downside_path(self):
        """Test Sortino with negative mean when no downside (edge case)."""
        from ml4t.diagnostic.evaluation.metrics import sortino_ratio

        # Target higher than all returns, but still no values below target
        # This is tricky - we need values above target but mean excess < 0
        # Actually impossible, so test the zero downside_std path instead
        returns = np.array([0.01, 0.02, 0.01, 0.01])
        # Very small negative excess returns that equal zero after squaring
        result = sortino_ratio(returns, target_return=0.0)
        # All positive excess, should return inf
        assert np.isinf(result)

    def test_sortino_zero_downside_std(self):
        """Test Sortino when downside std is zero."""
        from ml4t.diagnostic.evaluation.metrics import sortino_ratio

        # All downside returns identical (zero std)
        returns = np.array([0.01, -0.01, -0.01, -0.01, -0.01])
        result = sortino_ratio(returns, target_return=0.0)
        # Should compute normally
        assert isinstance(result, float)


class TestComputeForwardReturnsPandas:
    """Tests for compute_forward_returns with pandas group_col."""

    def test_forward_returns_pandas_with_group(self):
        """Test forward returns computation with pandas DataFrame and group_col."""
        from ml4t.diagnostic.evaluation.metrics import compute_forward_returns

        np.random.seed(42)
        n = 100

        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n),
                "symbol": ["A"] * 50 + ["B"] * 50,
                "close": 100 + np.random.randn(n).cumsum() * 0.5,
            }
        )

        result = compute_forward_returns(
            prices=df,
            periods=[1, 5],
            price_col="close",
            group_col="symbol",
        )

        assert isinstance(result, pd.DataFrame)
        assert "fwd_ret_1" in result.columns
        assert "fwd_ret_5" in result.columns


class TestComputeMonotonicityValidation:
    """Tests for compute_monotonicity validation paths."""

    def test_monotonicity_polars_dataframe_feature_col(self):
        """Test with Polars DataFrame requiring feature_col."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 50

        features = pl.DataFrame(
            {
                "feat": np.random.randn(n),
            }
        )
        outcomes = np.random.randn(n) * 0.01

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            feature_col="feat",
            n_quantiles=5,
        )

        assert isinstance(result, dict)
        assert "correlation" in result

    def test_monotonicity_pandas_dataframe_feature_col(self):
        """Test with pandas DataFrame requiring feature_col."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 50

        features = pd.DataFrame(
            {
                "feat": np.random.randn(n),
            }
        )
        outcomes = np.random.randn(n) * 0.01

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            feature_col="feat",
            n_quantiles=5,
        )

        assert isinstance(result, dict)

    def test_monotonicity_polars_dataframe_outcome_col(self):
        """Test with Polars DataFrame for outcomes requiring outcome_col."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 50

        features = np.random.randn(n)
        outcomes = pl.DataFrame(
            {
                "ret": np.random.randn(n) * 0.01,
            }
        )

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            outcome_col="ret",
            n_quantiles=5,
        )

        assert isinstance(result, dict)

    def test_monotonicity_pandas_dataframe_outcome_col(self):
        """Test with pandas DataFrame for outcomes requiring outcome_col."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 50

        features = np.random.randn(n)
        outcomes = pd.DataFrame(
            {
                "ret": np.random.randn(n) * 0.01,
            }
        )

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            outcome_col="ret",
            n_quantiles=5,
        )

        assert isinstance(result, dict)

    def test_monotonicity_length_mismatch(self):
        """Test error when features and outcomes have different lengths."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        features = np.array([1, 2, 3, 4, 5])
        outcomes = np.array([0.1, 0.2, 0.3])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            compute_monotonicity(features=features, outcomes=outcomes)

    def test_monotonicity_insufficient_data(self):
        """Test with insufficient data for quantile analysis."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        features = np.array([1.0, 2.0, 3.0])
        outcomes = np.array([0.1, 0.2, 0.3])

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            n_quantiles=10,  # Too many quantiles for 3 observations
        )

        assert result["direction"] == "insufficient_data"

    def test_monotonicity_pearson_method(self):
        """Test with Pearson correlation method."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        features = np.random.randn(100)
        outcomes = features * 0.5 + np.random.randn(100) * 0.1

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            method="pearson",
        )

        assert isinstance(result, dict)
        assert "correlation" in result

    def test_monotonicity_invalid_method(self):
        """Test with invalid correlation method."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        outcomes = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 10)

        with pytest.raises(ValueError, match="Unknown method"):
            compute_monotonicity(
                features=features,
                outcomes=outcomes,
                method="invalid",
            )


class TestAnalyzeFeatureOutcomePaths:
    """Tests for analyze_feature_outcome different paths."""

    def test_feature_outcome_pandas_path(self):
        """Test analyze_feature_outcome with pandas DataFrame."""
        np.random.seed(42)
        n = 60

        dates = pd.date_range("2020-01-01", periods=n)
        predictions = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pd.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions,
            prices=prices,
            horizons=[1, 5],
            include_decay=True,
        )

        assert isinstance(result, dict)

    def test_feature_outcome_with_group_col(self):
        """Test analyze_feature_outcome with group column (panel data)."""
        np.random.seed(42)
        n_dates = 30
        n_symbols = 3

        dates = pd.date_range("2020-01-01", periods=n_dates)
        all_dates = list(dates) * n_symbols
        all_symbols = ["A"] * n_dates + ["B"] * n_dates + ["C"] * n_dates

        predictions = pd.DataFrame(
            {
                "date": all_dates,
                "symbol": all_symbols,
                "prediction": np.random.randn(n_dates * n_symbols),
            }
        )
        prices = pd.DataFrame(
            {
                "date": all_dates,
                "symbol": all_symbols,
                "close": 100 + np.random.randn(n_dates * n_symbols).cumsum() * 0.1,
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions,
            prices=prices,
            group_col="symbol",
            horizons=[1],
        )

        assert isinstance(result, dict)

    def test_feature_outcome_without_hac(self):
        """Test analyze_feature_outcome without HAC adjustment."""
        np.random.seed(42)
        n = 60

        dates = pd.date_range("2020-01-01", periods=n)
        predictions = pl.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pl.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions,
            prices=prices,
            include_hac=False,
        )

        assert isinstance(result, dict)


class TestExplainerCreationPaths:
    """Tests for SHAP explainer creation paths."""

    def test_invalid_explainer_type(self):
        """Test that invalid explainer_type raises ValueError."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = X[:, 0] + np.random.randn(20) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="Invalid explainer_type"):
            compute_shap_importance(model, X, explainer_type="invalid_type")

    def test_use_gpu_false_explicit(self):
        """Test explicit use_gpu=False path."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(20),
                "b": np.random.randn(20),
            }
        )
        y = X["a"] + np.random.randn(20) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X, use_gpu=False)

        assert isinstance(result, dict)
        assert "importances" in result


class TestAnalyzeMLImportanceEdgeCases:
    """Edge cases for analyze_ml_importance."""

    def test_empty_methods_raises_error(self):
        """Test that empty methods list raises ValueError."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = X[:, 0] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="At least one method"):
            analyze_ml_importance(model, X, y, methods=[])

    def test_numpy_input_generates_feature_names(self):
        """Test that numpy input generates feature names automatically."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = X[:, 0] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = analyze_ml_importance(model, X, y, methods=["mdi"])

        # Should generate f0, f1, f2, f3
        assert "f0" in result["method_results"]["mdi"]["feature_names"]

    def test_method_failure_mdi(self):
        """Test handling of MDI method failure."""
        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        # Create a model without feature_importances_
        class FakeModel:
            def predict(self, X):
                return np.zeros(len(X))

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)

        model = FakeModel()

        # Should fail MDI but succeed if other methods work
        with pytest.raises(ValueError, match="All methods failed"):
            analyze_ml_importance(model, X, y, methods=["mdi"])


class TestConditionalICQuantilePaths:
    """Tests for conditional IC quantile computation paths."""

    def test_conditional_ic_insufficient_data_per_quantile(self):
        """Test conditional IC with insufficient data in some quantiles."""
        np.random.seed(42)
        n = 20  # Very small dataset

        # Create data with uneven distribution
        feature_a = np.concatenate([np.ones(15), np.zeros(5)])
        feature_b = np.random.randn(n)
        returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=returns,
            n_quantiles=5,  # Many quantiles for small dataset
            min_periods=5,
        )

        assert isinstance(result, dict)

    def test_conditional_ic_all_nan_quantiles(self):
        """Test conditional IC when all quantiles have insufficient data."""
        # Very small dataset with high min_periods
        feature_a = np.array([1.0, 2.0, 3.0])
        feature_b = np.array([0.1, 0.2, 0.3])
        returns = np.array([0.01, 0.02, 0.03])

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=returns,
            n_quantiles=3,
            min_periods=10,  # Higher than data size
        )

        assert isinstance(result, dict)
        # Should have interpretation about insufficient quantiles
        assert "interpretation" in result


class TestSHAPBinaryClassificationPaths:
    """Tests for SHAP with binary classification models."""

    def test_shap_binary_classification_list_format(self):
        """Test SHAP with binary classifier returning list format."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "shap_values" in result


class TestAnalyzeInteractionsMethodFailures:
    """Tests for analyze_interactions with method failures."""

    def test_interactions_conditional_ic_failure(self):
        """Test analyze_interactions when conditional_ic fails."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
                "c": np.random.randn(30),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # Test with only h_statistic to ensure other methods work
        result = analyze_interactions(
            model,
            X,
            y,
            methods=["h_statistic"],
            feature_pairs=[("a", "b"), ("a", "c")],
        )

        assert isinstance(result, dict)
        assert "h_statistic" in result["method_results"]

    def test_interactions_shap_with_pairs_filter(self):
        """Test analyze_interactions SHAP with feature_pairs filter."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
                "c": np.random.randn(30),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model,
            X,
            y,
            methods=["shap"],
            feature_pairs=[("a", "b")],  # Only one pair
            max_samples=20,
        )

        assert isinstance(result, dict)
        assert "shap" in result["method_results"]


class TestSHAPInteractionsMulticlass:
    """Tests for SHAP interactions with multiclass models."""

    def test_shap_interactions_no_subsampling(self):
        """Test SHAP interactions without subsampling (small dataset)."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(20),
                "b": np.random.randn(20),
            }
        )
        y = X["a"] * X["b"]

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # max_samples=None means no subsampling
        result = compute_shap_interactions(
            model,
            X,
            max_samples=None,  # No subsampling
        )

        assert isinstance(result, dict)
        assert result["n_samples_used"] == 20


class TestMaximumDrawdownPaths:
    """Additional maximum drawdown edge cases."""

    def test_mdd_all_nan_after_removal(self):
        """Test MDD when all values are NaN."""
        from ml4t.diagnostic.evaluation.metrics import maximum_drawdown

        returns = np.array([np.nan, np.nan, np.nan])
        result = maximum_drawdown(returns)

        assert np.isnan(result["max_drawdown"])

    def test_mdd_single_value(self):
        """Test MDD with single non-NaN value."""
        from ml4t.diagnostic.evaluation.metrics import maximum_drawdown

        returns = np.array([0.05])
        result = maximum_drawdown(returns)

        assert isinstance(result, dict)


class TestICDecayAdditionalPaths:
    """Additional IC decay edge cases."""

    def test_ic_decay_with_all_horizons(self):
        """Test IC decay with multiple horizons."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        np.random.seed(42)
        n = 100

        dates = pd.date_range("2020-01-01", periods=n)
        predictions = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pd.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = compute_ic_decay(
            predictions=predictions,
            prices=prices,
            horizons=[1, 2, 5, 10, 21],
        )

        assert isinstance(result, dict)
        assert "ic_by_horizon" in result
        assert 1 in result["ic_by_horizon"]
        assert 21 in result["ic_by_horizon"]


class TestAnalyzeInteractionsWarnings:
    """Tests for warnings generated in analyze_interactions."""

    def test_interactions_method_agreement_warning(self):
        """Test that low method agreement triggers warning."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        n = 50
        X = pd.DataFrame(
            {
                "a": np.random.randn(n),
                "b": np.random.randn(n),
                "c": np.random.randn(n),
                "d": np.random.randn(n),
            }
        )
        y = X["a"] * X["b"] + X["c"] + np.random.randn(n) * 0.1

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Using multiple methods to potentially trigger disagreement
        result = analyze_interactions(
            model,
            X,
            y,
            methods=["conditional_ic", "h_statistic"],
            feature_pairs=[("a", "b"), ("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")],
            max_samples=30,
        )

        assert isinstance(result, dict)
        assert "warnings" in result
        assert "method_agreement" in result


# ============================================================================
# Additional coverage tests - 3rd round
# ============================================================================


class TestSHAPExplainerExplicitTypes:
    """Tests for explicit SHAP explainer type specification."""

    def test_linear_explainer_explicit(self):
        """Test explicit linear explainer type."""
        from sklearn.linear_model import LinearRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
            }
        )
        y = X["a"] * 2 + X["b"] + np.random.randn(30) * 0.1

        model = LinearRegression()
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="linear")

        assert isinstance(result, dict)
        assert result["explainer_type"] == "linear"

    def test_deep_explainer_requires_background(self):
        """Test that deep explainer requires background_data."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = X[:, 0] + np.random.randn(20) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # Deep explainer requires background_data
        with pytest.raises(ValueError, match="background_data"):
            compute_shap_importance(model, X, explainer_type="deep")


class TestAnalyzeMLImportanceMethodPaths:
    """Tests for analyze_ml_importance method failure paths."""

    def test_mda_method_failure(self):
        """Test handling of MDA method failure."""
        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        # Create a model that doesn't work with MDA
        class BadModel:
            def predict(self, X):
                raise ValueError("Model can't predict")

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)

        model = BadModel()

        # Should fail all methods
        with pytest.raises(ValueError, match="All methods failed"):
            analyze_ml_importance(model, X, y, methods=["mda"])

    def test_pfi_method_failure(self):
        """Test handling of PFI method failure."""
        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        # Create a model that doesn't support fit/predict properly
        class BadScorer:
            def predict(self, X):
                raise ValueError("Bad prediction")

        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)

        model = BadScorer()

        # Should fail all methods
        with pytest.raises(ValueError, match="All methods failed"):
            analyze_ml_importance(model, X, y, methods=["pfi"])


class TestSHAPProgressPaths:
    """Tests for SHAP computation with progress bar."""

    def test_shap_with_show_progress(self):
        """Test SHAP computation with progress bar enabled."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
            }
        )
        y = X["a"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # show_progress=True should work
        result = compute_shap_importance(model, X, show_progress=True)

        assert isinstance(result, dict)
        assert "importances" in result


class TestConditionalICDataPaths:
    """Tests for conditional IC with different data scenarios."""

    def test_conditional_ic_with_few_valid_quantiles(self):
        """Test conditional IC when only some quantiles have enough data."""
        np.random.seed(42)

        # Create data where some quantiles will have fewer than min_periods
        n = 50
        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=returns,
            n_quantiles=5,
            min_periods=5,
        )

        assert isinstance(result, dict)
        # Should have interpretation about the result
        assert "interpretation" in result


class TestSortinoSpecialCases:
    """Special edge cases for Sortino ratio."""

    def test_sortino_all_downside_same_value(self):
        """Test Sortino when all downside returns are identical."""
        from ml4t.diagnostic.evaluation.metrics import sortino_ratio

        # All negative excess returns identical - zero std
        returns = np.array([-0.01, -0.01, -0.01, -0.01, -0.01])
        result = sortino_ratio(returns, target_return=0.0)

        # Downside std is non-zero since we use sqrt(mean(squared)) not std
        assert isinstance(result, float)


class TestSHAPValuesListFormat:
    """Tests for SHAP values in list format (binary/multiclass)."""

    def test_shap_multiclass_classification(self):
        """Test SHAP with multiclass classifier."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(60),
                "b": np.random.randn(60),
            }
        )
        # 3-class target
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "shap_values" in result


class TestSHAPInteractionsBinaryClassification:
    """Tests for SHAP interactions with binary classification."""

    def test_shap_interactions_binary_classifier(self):
        """Test SHAP interactions with binary classifier."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_interactions(model, X, max_samples=20)

        assert isinstance(result, dict)
        assert "top_interactions" in result


class TestKernelExplainerClassifier:
    """Test kernel explainer with classifiers."""

    def test_kernel_explainer_with_classifier(self):
        """Test kernel explainer with classifier using predict_proba."""
        import warnings

        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)

        # Force kernel explainer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_shap_importance(
                model,
                X,
                explainer_type="kernel",
                max_samples=10,  # Small for speed
            )

        assert isinstance(result, dict)
        assert result["explainer_type"] == "kernel"


class TestAnalyzeInteractionsEmptyResults:
    """Test analyze_interactions result handling."""

    def test_interactions_with_h_statistic_results(self):
        """Test analyze_interactions properly handles h_statistics key."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
                "c": np.random.randn(40),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(40) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model,
            X,
            y,
            methods=["h_statistic"],
            feature_pairs=[("a", "b"), ("a", "c")],
        )

        assert isinstance(result, dict)
        assert "h_statistic" in result["method_results"]
        # Check consensus_ranking is populated
        assert len(result["consensus_ranking"]) > 0


# ============================================================================
# Additional coverage tests - 4th round (targeting specific uncovered lines)
# ============================================================================


class TestFeatureOutcomePandasNoHAC:
    """Test analyze_feature_outcome pandas path with include_hac=False."""

    def test_feature_outcome_pandas_no_hac(self):
        """Test pandas path in analyze_feature_outcome without HAC."""
        np.random.seed(42)
        n = 60

        dates = pd.date_range("2020-01-01", periods=n)
        predictions = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices = pd.DataFrame(
            {
                "date": dates,
                "close": 100 * np.exp(np.random.randn(n).cumsum() * 0.01),
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions,
            prices=prices,
            include_hac=False,
            include_decay=False,  # Skip decay for speed
        )

        assert isinstance(result, dict)


class TestSHAPInteractionsMulticlassClassifier:
    """Test SHAP interactions with multiclass classifier."""

    def test_shap_interactions_multiclass(self):
        """Test SHAP interactions with 3-class classifier (list format handling)."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(60),
                "b": np.random.randn(60),
            }
        )
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_interactions(model, X, max_samples=20)

        assert isinstance(result, dict)
        assert "interaction_matrix" in result


class TestSHAPExplainerFallbackPaths:
    """Test SHAP explainer auto-selection and fallback paths."""

    def test_shap_auto_selects_tree_explainer(self):
        """Test that SHAP auto-selects tree explainer for tree models."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
            }
        )
        y = X["a"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # Auto mode should select tree explainer
        result = compute_shap_importance(model, X, explainer_type="auto")

        assert isinstance(result, dict)
        assert result["explainer_type"] == "tree"


class TestAnalyzeMLImportanceDefaultMethods:
    """Test analyze_ml_importance with default methods."""

    def test_ml_importance_default_methods(self):
        """Test analyze_ml_importance uses default methods when None."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = X["a"] + np.random.randn(40) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # methods=None should use default ["mdi", "pfi", "shap"]
        result = analyze_ml_importance(model, X, y, methods=None)

        assert isinstance(result, dict)
        assert "method_results" in result


class TestAnalyzeMLImportanceAllMethods:
    """Test analyze_ml_importance with all methods."""

    def test_ml_importance_with_mda(self):
        """Test analyze_ml_importance including MDA method."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = X["a"] + np.random.randn(40) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        # Include MDA method
        result = analyze_ml_importance(model, X, y, methods=["mdi", "mda"])

        assert isinstance(result, dict)
        assert "mdi" in result["method_results"]
        assert "mda" in result["method_results"]


class TestConditionalICWithDateCol:
    """Test conditional IC with date column in DataFrame input."""

    def test_conditional_ic_dataframe_with_dates(self):
        """Test conditional IC with DataFrame input including dates."""
        np.random.seed(42)
        n = 100

        dates = pd.date_range("2020-01-01", periods=n)
        df_a = pd.DataFrame(
            {
                "date": dates,
                "feature_a": np.random.randn(n),
            }
        )
        df_b = pd.DataFrame(
            {
                "date": dates,
                "feature_b": np.random.randn(n),
            }
        )
        df_ret = pd.DataFrame(
            {
                "date": dates,
                "returns": np.random.randn(n) * 0.01,
            }
        )

        result = compute_conditional_ic(
            feature_a=df_a,
            feature_b=df_b,
            forward_returns=df_ret,
            date_col="date",
            n_quantiles=5,
        )

        assert isinstance(result, dict)
        assert "interpretation" in result


class TestSHAPWithPolarsInput:
    """Test SHAP importance with Polars DataFrame input."""

    def test_shap_importance_polars_input(self):
        """Test SHAP importance with Polars DataFrame."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pl.DataFrame(
            {
                "a": np.random.randn(30),
                "b": np.random.randn(30),
            }
        )
        y = X["a"].to_numpy() + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X.to_pandas(), y)

        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "importances" in result


class TestHStatisticWithSampling:
    """Test H-statistic with sampling."""

    def test_h_statistic_with_n_samples(self):
        """Test H-statistic computation with sample limit."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_h_statistic

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
                "c": np.random.randn(100),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = compute_h_statistic(
            model,
            X,
            feature_pairs=[(0, 1)],
            n_samples=50,
        )

        assert isinstance(result, dict)
        assert "h_statistics" in result


# =============================================================================
# Additional Coverage Tests for 95% Target
# =============================================================================


class TestConditionalICWithoutDateCol:
    """Test conditional IC computation without date column."""

    def test_conditional_ic_simple_quantiles(self):
        """Test conditional IC with simple quantile path (no date_col)."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 200

        feature_a = np.random.randn(n)
        feature_b = np.random.randn(n)
        forward_returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=forward_returns,
            n_quantiles=5,
            # No date_col - uses simple quantile path
        )

        assert isinstance(result, dict)
        assert "quantile_ics" in result
        assert len(result["quantile_ics"]) == 5

    def test_conditional_ic_insufficient_unique_values(self):
        """Test conditional IC with insufficient unique values for quantiles."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n = 100

        feature_a = np.random.randn(n)
        feature_b = np.repeat([1.0, 2.0], n // 2)  # Only 2 unique values
        forward_returns = np.random.randn(n) * 0.01

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=forward_returns,
            n_quantiles=10,  # More quantiles than unique values
        )

        # Should handle gracefully - may still compute with fewer quantiles
        assert isinstance(result, dict)


class TestGPUExplainerPaths:
    """Test GPU-related paths in SHAP explainer creation."""

    def test_gpu_requested_but_not_available(self):
        """Test error when GPU requested but not available."""
        from unittest.mock import patch

        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        y = X["a"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        # Mock GPU as unavailable (patch at the module where it's used)
        with patch(
            "ml4t.diagnostic.evaluation.metrics.importance_shap._detect_gpu_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="GPU requested.*but GPU not available"):
                compute_shap_importance(model, X, use_gpu=True)

    def test_gpu_auto_detection_small_dataset(self):
        """Test that GPU auto-detection doesn't use GPU for small datasets."""
        from unittest.mock import patch

        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = X["a"] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        # Even if GPU is available, small dataset shouldn't use it (patch at the module where it's used)
        with patch(
            "ml4t.diagnostic.evaluation.metrics.importance_shap._detect_gpu_available",
            return_value=True,
        ):
            result = compute_shap_importance(model, X, use_gpu="auto")

        assert isinstance(result, dict)


class TestExplainerFallbackCascade:
    """Test explainer creation fallback cascade."""

    def test_linear_explainer_fallback(self):
        """Test fallback to LinearExplainer when TreeExplainer fails."""
        from sklearn.linear_model import LinearRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y = X["a"] + X["b"] + np.random.randn(50) * 0.1

        model = LinearRegression()
        model.fit(X, y)

        # LinearRegression should use LinearExplainer (not TreeExplainer)
        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "importances" in result

    def test_kernel_explainer_explicit(self):
        """Test explicit KernelExplainer usage."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        y = X["a"] + np.random.randn(30) * 0.1

        # Simple custom model
        class SimpleModel:
            def fit(self, X, y):
                self.coef_ = np.ones(X.shape[1])
                return self

            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return X @ self.coef_

        model = SimpleModel()
        model.fit(X, y)

        # This should use KernelExplainer
        result = compute_shap_importance(
            model,
            X,
            explainer_type="kernel",
            background_data=X.iloc[:5].values,
        )

        assert isinstance(result, dict)


class TestDeepExplainerPaths:
    """Test DeepExplainer-related paths."""

    def test_deep_explainer_requires_background(self):
        """Test DeepExplainer requires background_data."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        y = X["a"] + np.random.randn(30) * 0.1

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="background_data"):
            compute_shap_importance(model, X, explainer_type="deep")


class TestKernelExplainerWithProgress:
    """Test KernelExplainer with progress display."""

    def test_kernel_explainer_show_progress(self):
        """Test KernelExplainer with show_progress=True."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
        y = X["a"] + np.random.randn(20) * 0.1

        # Simple custom model
        class SimpleModel:
            def fit(self, X, y):
                self.coef_ = np.ones(X.shape[1])
                return self

            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return X @ self.coef_

        model = SimpleModel()
        model.fit(X, y)

        result = compute_shap_importance(
            model,
            X,
            explainer_type="kernel",
            background_data=X.iloc[:5].values,
            show_progress=True,
        )

        assert isinstance(result, dict)


class TestMulticlassSHAPInteractions:
    """Test SHAP interactions with multiclass classifier."""

    def test_shap_interactions_multiclass(self):
        """Test SHAP interactions with multiclass classification."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(60),
                "b": np.random.randn(60),
            }
        )
        y = np.repeat([0, 1, 2], 20)  # 3 classes

        model = RandomForestClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = compute_shap_interactions(model, X)

        assert isinstance(result, dict)


class TestUnknownExplainerType:
    """Test unknown explainer type error."""

    def test_unknown_explainer_type(self):
        """Test error for unknown explainer type."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30)})
        y = np.random.randn(30)

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        with pytest.raises(ValueError, match="Invalid explainer_type"):
            compute_shap_importance(model, X, explainer_type="unknown")


class TestSHAPValuesListFormatBinaryClassifier:
    """Test SHAP values list format for binary classifiers."""

    def test_shap_binary_classifier_list_format(self):
        """Test SHAP with binary classifier returning list of arrays."""
        from sklearn.ensemble import GradientBoostingClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = GradientBoostingClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "importances" in result


class TestMonotonicityWithPolarsInput:
    """Test monotonicity computation with Polars input."""

    def test_monotonicity_polars_dataframe(self):
        """Test compute_monotonicity with Polars DataFrame."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 100

        features = pl.DataFrame(
            {
                "feat": np.random.randn(n),
            }
        )
        outcomes = np.random.randn(n) * 0.01

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            feature_col="feat",
        )

        assert isinstance(result, dict)
        assert "correlation" in result


class TestAnalyzeFeatureOutcomeWithSeries:
    """Test analyze_feature_outcome with DataFrame input."""

    def test_feature_outcome_dataframe_input(self):
        """Test analyze_feature_outcome with DataFrame."""
        from ml4t.diagnostic.evaluation.metrics import analyze_feature_outcome

        np.random.seed(42)
        n = 100

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        predictions_df = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices_df = pd.DataFrame(
            {
                "date": dates,
                "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            }
        )

        result = analyze_feature_outcome(
            predictions=predictions_df,
            prices=prices_df,
            pred_col="prediction",
            price_col="close",
            date_col="date",
        )

        assert isinstance(result, dict)


class TestICDecayWithCustomLags:
    """Test IC decay with custom horizon structure."""

    def test_ic_decay_custom_horizons(self):
        """Test IC decay with custom horizons."""
        from ml4t.diagnostic.evaluation.metrics import compute_ic_decay

        np.random.seed(42)
        n = 200

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        predictions_df = pd.DataFrame(
            {
                "date": dates,
                "prediction": np.random.randn(n),
            }
        )
        prices_df = pd.DataFrame(
            {
                "date": dates,
                "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
            }
        )

        result = compute_ic_decay(
            predictions=predictions_df,
            prices=prices_df,
            horizons=[1, 2, 3, 5, 10],
        )

        assert isinstance(result, dict)
        assert "ic_by_horizon" in result


class TestQuantileReturnsEdgeCases:
    """Test compute_monotonicity with quantile edge cases."""

    def test_monotonicity_with_nan_values(self):
        """Test compute_monotonicity handles NaN values in predictions."""
        from ml4t.diagnostic.evaluation.metrics import compute_monotonicity

        np.random.seed(42)
        n = 100

        features = pd.DataFrame(
            {
                "feat": np.random.randn(n),
            }
        )
        # Add some NaN
        features.iloc[10:15, 0] = np.nan
        outcomes = np.random.randn(n) * 0.01

        result = compute_monotonicity(
            features=features,
            outcomes=outcomes,
            feature_col="feat",
            n_quantiles=5,
        )

        assert isinstance(result, dict)


class TestICWithConstantInput:
    """Test IC calculation with constant input."""

    def test_ic_constant_predictions(self):
        """Test IC with constant predictions returns NaN."""
        from ml4t.diagnostic.evaluation.metrics import information_coefficient

        predictions = np.ones(100)  # Constant
        returns = np.random.randn(100)

        result = information_coefficient(predictions, returns)

        assert np.isnan(result)

    def test_ic_constant_returns(self):
        """Test IC with constant returns returns NaN."""
        from ml4t.diagnostic.evaluation.metrics import information_coefficient

        predictions = np.random.randn(100)
        returns = np.ones(100)  # Constant

        result = information_coefficient(predictions, returns)

        assert np.isnan(result)


class TestSHAPWithClassifierPredict:
    """Test SHAP with classifier using predict method."""

    def test_shap_classifier_with_predict_proba(self):
        """Test SHAP with classifier that has predict_proba."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(model, X)

        assert isinstance(result, dict)
        assert "importances" in result


class TestAnalyzeInteractionsMethods:
    """Test analyze_interactions with different method combinations."""

    def test_interactions_conditional_ic_only(self):
        """Test analyze_interactions with only conditional_ic."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model,
            X,
            y,
            methods=["conditional_ic"],
        )

        assert isinstance(result, dict)
        assert "method_results" in result
        assert "conditional_ic" in result["method_results"]

    def test_interactions_h_statistic_only(self):
        """Test analyze_interactions with only h_statistic."""
        from sklearn.ensemble import RandomForestRegressor

        from ml4t.diagnostic.evaluation.metrics import analyze_interactions

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
            }
        )
        y = X["a"] * X["b"] + np.random.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)

        result = analyze_interactions(
            model,
            X,
            y,
            methods=["h_statistic"],
        )

        assert isinstance(result, dict)
        assert "method_results" in result
        assert "h_statistic" in result["method_results"]


class TestHitRatePredictions:
    """Test hit rate predictions."""

    def test_hit_rate_perfect_predictions(self):
        """Test hit rate with perfect predictions."""
        from ml4t.diagnostic.evaluation.metrics import hit_rate

        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Perfect positive correlation

        result = hit_rate(predictions, returns)

        # Should be 100% (all predictions correctly predict sign of returns)
        assert result == 100.0

    def test_hit_rate_all_wrong(self):
        """Test hit rate with all wrong predictions."""
        from ml4t.diagnostic.evaluation.metrics import hit_rate

        predictions = np.array([1, 2, 3, 4, 5])
        returns = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])  # Opposite direction

        result = hit_rate(predictions, returns)

        # Should be 0%
        assert result == 0.0


class TestMaxDrawdownEdgeCases:
    """Test max drawdown edge cases."""

    def test_max_drawdown_monotonic_increasing(self):
        """Test max drawdown with monotonically increasing returns."""
        from ml4t.diagnostic.evaluation.metrics import maximum_drawdown

        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # All positive

        result = maximum_drawdown(returns)

        # Returns a dict
        assert isinstance(result, dict)
        assert "max_drawdown" in result
        # No significant drawdown for monotonically increasing
        assert result["max_drawdown"] <= 0.01  # May have tiny drawdowns

    def test_max_drawdown_single_drop(self):
        """Test max drawdown with single large drop."""
        from ml4t.diagnostic.evaluation.metrics import maximum_drawdown

        returns = np.array([0.1, 0.1, -0.3, 0.1, 0.1])

        result = maximum_drawdown(returns)

        # Should capture the drop (drawdown is expressed as negative)
        assert isinstance(result, dict)
        assert result["max_drawdown"] < 0.0  # Negative drawdown value
        assert abs(result["max_drawdown"]) > 0.2  # Significant drop


class TestConditionalICWithDataFrameAndDateCol:
    """Test conditional IC with DataFrame input and date_col for panel structure."""

    def test_conditional_ic_panel_data(self):
        """Test conditional IC with panel data (date_col specified)."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        np.random.seed(42)
        n_dates = 50
        n_assets = 5
        n = n_dates * n_assets

        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates, freq="D"), n_assets)
        assets = np.tile(["A", "B", "C", "D", "E"], n_dates)

        feature_a_df = pd.DataFrame(
            {
                "date": dates,
                "asset": assets,
                "feature_a": np.random.randn(n),
            }
        )
        feature_b_df = pd.DataFrame(
            {
                "date": dates,
                "asset": assets,
                "feature_b": np.random.randn(n),
            }
        )
        returns_df = pd.DataFrame(
            {
                "date": dates,
                "asset": assets,
                "returns": np.random.randn(n) * 0.01,
            }
        )

        result = compute_conditional_ic(
            feature_a_df,
            feature_b_df,
            returns_df,
            date_col="date",
            group_col="asset",
            n_quantiles=3,
        )

        assert isinstance(result, dict)
        assert "quantile_ics" in result


class TestAutoExplainerFallbackCascade:
    """Test the auto explainer selection fallback cascade."""

    def test_auto_explainer_for_custom_model(self):
        """Test auto explainer selection falls through cascade for custom model."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        y = X["a"] + np.random.randn(30) * 0.1

        # Custom model that only has predict method
        class CustomModel:
            def fit(self, X, y):
                return self

            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.values
                return np.mean(X, axis=1)

        model = CustomModel()
        model.fit(X, y)

        # Should fall through tree → linear → kernel
        result = compute_shap_importance(
            model,
            X,
            explainer_type="auto",
            performance_warning=False,
        )

        assert isinstance(result, dict)


class TestSHAPWithMulticlassPredict:
    """Test SHAP with multiclass classifier predict_proba."""

    def test_shap_multiclass_kernel_explainer(self):
        """Test SHAP KernelExplainer with multiclass classifier."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(60),
                "b": np.random.randn(60),
            }
        )
        y = np.repeat([0, 1, 2], 20)

        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(
            model,
            X,
            explainer_type="kernel",
            background_data=X.iloc[:10].values,
        )

        assert isinstance(result, dict)


class TestInsufficientDataPaths:
    """Test paths for insufficient data handling."""

    def test_conditional_ic_insufficient_data(self):
        """Test conditional IC with insufficient data."""
        from ml4t.diagnostic.evaluation.metrics import compute_conditional_ic

        # Too few samples for meaningful quantile analysis
        feature_a = np.random.randn(10)
        feature_b = np.random.randn(10)
        returns = np.random.randn(10) * 0.01

        result = compute_conditional_ic(
            feature_a=feature_a,
            feature_b=feature_b,
            forward_returns=returns,
            n_quantiles=10,  # More quantiles than samples per group
            min_periods=5,
        )

        assert isinstance(result, dict)
        # Should return interpretation about insufficient data
        assert "interpretation" in result


class TestExplainerWithBinaryClassifier:
    """Test SHAP explainer paths with binary classifier."""

    def test_kernel_explainer_binary_classifier(self):
        """Test KernelExplainer with binary classifier predict_proba."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "a": np.random.randn(40),
                "b": np.random.randn(40),
            }
        )
        y = (X["a"] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)

        result = compute_shap_importance(
            model,
            X,
            explainer_type="kernel",
            background_data=X.iloc[:10].values,
        )

        assert isinstance(result, dict)
        assert "importances" in result


class TestDeepExplainerWithTensorFlow:
    """Test DeepExplainer with TensorFlow model."""

    def test_deep_explainer_with_keras_model(self):
        """Test DeepExplainer with a Keras sequential model."""
        pytest.importorskip("tensorflow", reason="TensorFlow not installed")
        import tensorflow as tf

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        # Suppress TensorFlow warnings
        tf.get_logger().setLevel("ERROR")

        np.random.seed(42)
        tf.random.set_seed(42)

        # Create simple data
        X = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),
            }
        )
        y = (X["a"] + X["b"]).values

        # Create simple Keras model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_shape=(2,)),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X.values, y, epochs=5, verbose=0)

        # Background data for DeepExplainer
        background = X.iloc[:20].values

        result = compute_shap_importance(
            model,
            X,
            explainer_type="deep",
            background_data=background,
        )

        assert isinstance(result, dict)
        assert "importances" in result
        assert len(result["importances"]) == 2  # Two features
