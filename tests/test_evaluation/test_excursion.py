"""Tests for price excursion analysis module."""

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.excursion import (
    ExcursionAnalysisResult,
    analyze_excursions,
    compute_excursions,
)


class TestComputeExcursions:
    """Tests for compute_excursions function."""

    def test_basic_computation(self):
        """Test basic MFE/MAE computation with known values."""
        # Simple price series: 100 -> 105 -> 95 -> 110
        prices = pl.Series([100.0, 105.0, 95.0, 110.0, 100.0])

        result = compute_excursions(prices, horizons=[2])

        # For horizon=2 starting at index 0 (price=100):
        # Window is [100, 105, 95]
        # MFE = (105-100)/100 = 0.05
        # MAE = (95-100)/100 = -0.05
        assert "mfe_2" in result.columns
        assert "mae_2" in result.columns

        # First row
        mfe_0 = result["mfe_2"][0]
        mae_0 = result["mae_2"][0]
        assert pytest.approx(mfe_0, rel=0.01) == 0.05
        assert pytest.approx(mae_0, rel=0.01) == -0.05

    def test_multiple_horizons(self):
        """Test computation with multiple horizons."""
        prices = pl.Series([100.0] * 100)  # Flat prices
        prices = pl.Series(np.linspace(100, 110, 100))  # Upward trend

        result = compute_excursions(prices, horizons=[5, 10, 20])

        assert "mfe_5" in result.columns
        assert "mae_5" in result.columns
        assert "mfe_10" in result.columns
        assert "mae_10" in result.columns
        assert "mfe_20" in result.columns
        assert "mae_20" in result.columns

    def test_return_types(self):
        """Test different return type calculations."""
        prices = pl.Series([100.0, 110.0, 90.0, 100.0])

        # Percentage returns
        result_pct = compute_excursions(prices, horizons=[1], return_type="pct")
        assert pytest.approx(result_pct["mfe_1"][0], rel=0.01) == 0.10  # (110-100)/100

        # Log returns
        result_log = compute_excursions(prices, horizons=[1], return_type="log")
        assert pytest.approx(result_log["mfe_1"][0], rel=0.01) == np.log(110 / 100)

        # Absolute returns
        result_abs = compute_excursions(prices, horizons=[1], return_type="abs")
        assert pytest.approx(result_abs["mfe_1"][0], rel=0.01) == 10.0

    def test_short_series_error(self):
        """Test that short series raises appropriate error."""
        prices = pl.Series([100.0, 105.0])

        with pytest.raises(ValueError, match="too short"):
            compute_excursions(prices, horizons=[10])

    def test_handles_nan(self):
        """Test that NaN values are handled gracefully."""
        prices = pl.Series([100.0, np.nan, 105.0, 110.0, 100.0])

        result = compute_excursions(prices, horizons=[2])

        # First row should be NaN due to NaN in window
        assert np.isnan(result["mfe_2"][0])

    def test_numpy_input(self):
        """Test with numpy array input."""
        prices = np.array([100.0, 105.0, 95.0, 110.0, 100.0])

        result = compute_excursions(prices, horizons=[2])

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

    def test_pandas_input(self):
        """Test with pandas Series input."""
        import pandas as pd

        prices = pd.Series([100.0, 105.0, 95.0, 110.0, 100.0])

        result = compute_excursions(prices, horizons=[2])

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0


class TestAnalyzeExcursions:
    """Tests for analyze_excursions function."""

    @pytest.fixture
    def random_walk_prices(self):
        """Generate a random walk price series."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.01  # 1% daily vol
        prices = 100 * np.exp(np.cumsum(returns))
        return pl.Series(prices)

    def test_basic_analysis(self, random_walk_prices):
        """Test basic analysis with default parameters."""
        result = analyze_excursions(random_walk_prices)

        assert isinstance(result, ExcursionAnalysisResult)
        assert result.horizons == [15, 30, 60]
        assert result.n_samples > 0
        assert len(result.statistics) == 3
        assert result.percentile_matrix is not None

    def test_custom_horizons(self, random_walk_prices):
        """Test with custom horizons."""
        result = analyze_excursions(random_walk_prices, horizons=[5, 10, 20, 50])

        assert result.horizons == [5, 10, 20, 50]
        assert len(result.statistics) == 4

    def test_custom_percentiles(self, random_walk_prices):
        """Test with custom percentiles."""
        result = analyze_excursions(
            random_walk_prices, horizons=[30], percentiles=[5, 25, 50, 75, 95]
        )

        stats = result.statistics[30]
        assert 5 in stats.mfe_percentiles
        assert 95 in stats.mfe_percentiles

    def test_get_percentile(self, random_walk_prices):
        """Test get_percentile method."""
        result = analyze_excursions(random_walk_prices, horizons=[30, 60])

        # Should return float
        mfe_75 = result.get_percentile(horizon=30, percentile=75, side="mfe")
        assert isinstance(mfe_75, float)

        mae_25 = result.get_percentile(horizon=30, percentile=25, side="mae")
        assert isinstance(mae_25, float)

        # MFE should be positive (max is >= entry)
        assert mfe_75 >= 0

        # MAE should be negative (min is <= entry)
        assert mae_25 <= 0

    def test_get_percentile_invalid_horizon(self, random_walk_prices):
        """Test get_percentile with invalid horizon."""
        result = analyze_excursions(random_walk_prices, horizons=[30])

        with pytest.raises(ValueError, match="not in analysis"):
            result.get_percentile(horizon=60, percentile=50, side="mfe")

    def test_summary(self, random_walk_prices):
        """Test summary generation."""
        result = analyze_excursions(random_walk_prices, horizons=[30, 60])

        summary = result.summary()

        assert "Price Excursion Analysis Summary" in summary
        assert "MFE" in summary
        assert "MAE" in summary
        assert "30 bars" in summary
        assert "60 bars" in summary

    def test_keep_raw(self, random_walk_prices):
        """Test keeping raw excursion values."""
        result_no_raw = analyze_excursions(random_walk_prices, horizons=[30], keep_raw=False)
        result_with_raw = analyze_excursions(random_walk_prices, horizons=[30], keep_raw=True)

        assert result_no_raw.excursions is None
        assert result_with_raw.excursions is not None
        assert isinstance(result_with_raw.excursions, pl.DataFrame)

    def test_rolling_stats(self, random_walk_prices):
        """Test rolling statistics computation."""
        result = analyze_excursions(random_walk_prices, horizons=[30], rolling_window=100)

        assert result.rolling_stats is not None
        assert "mfe_30_median" in result.rolling_stats.columns
        assert "mae_30_median" in result.rolling_stats.columns

    def test_statistics_content(self, random_walk_prices):
        """Test that statistics are reasonable."""
        result = analyze_excursions(random_walk_prices, horizons=[30])

        stats = result.statistics[30]

        # MFE mean should be positive (max >= entry always gives non-negative excursion)
        assert stats.mfe_mean >= 0

        # MAE mean should be negative
        assert stats.mae_mean <= 0

        # Std should be positive
        assert stats.mfe_std > 0
        assert stats.mae_std > 0

    def test_longer_horizon_larger_excursion(self, random_walk_prices):
        """Test that longer horizons have larger average excursions."""
        result = analyze_excursions(random_walk_prices, horizons=[10, 30, 60])

        # MFE should generally increase with horizon (more time = more opportunity)
        mfe_10 = result.statistics[10].mfe_mean
        mfe_30 = result.statistics[30].mfe_mean
        mfe_60 = result.statistics[60].mfe_mean

        # This should generally hold for random walk
        assert mfe_30 > mfe_10 * 0.8  # Some tolerance
        assert mfe_60 > mfe_30 * 0.8

    def test_percentile_ordering(self, random_walk_prices):
        """Test that percentiles are correctly ordered."""
        result = analyze_excursions(
            random_walk_prices, horizons=[30], percentiles=[10, 25, 50, 75, 90]
        )

        stats = result.statistics[30]

        # MFE percentiles should be increasing
        assert stats.mfe_percentiles[10] <= stats.mfe_percentiles[25]
        assert stats.mfe_percentiles[25] <= stats.mfe_percentiles[50]
        assert stats.mfe_percentiles[50] <= stats.mfe_percentiles[75]
        assert stats.mfe_percentiles[75] <= stats.mfe_percentiles[90]

        # MAE percentiles should also be increasing (less negative to more negative)
        assert stats.mae_percentiles[10] <= stats.mae_percentiles[25]
        assert stats.mae_percentiles[25] <= stats.mae_percentiles[50]


class TestExcursionUseCases:
    """Tests for typical use cases."""

    def test_parameter_selection_workflow(self):
        """Test typical workflow for TP/SL parameter selection."""
        # Generate trending price series
        np.random.seed(123)
        n = 500
        trend = np.linspace(0, 0.5, n)  # Upward drift
        noise = np.random.randn(n) * 0.02
        prices = 100 * np.exp(trend + np.cumsum(noise))

        # Analyze excursions
        result = analyze_excursions(
            pl.Series(prices), horizons=[20, 40, 60], percentiles=[25, 50, 75, 90]
        )

        # Get suggested parameters
        horizon = 40
        tp_level = result.get_percentile(horizon=horizon, percentile=75, side="mfe")
        sl_level = abs(result.get_percentile(horizon=horizon, percentile=25, side="mae"))

        # TP and SL should be reasonable percentages
        assert 0 < tp_level < 0.5  # Less than 50%
        assert 0 < sl_level < 0.5  # Less than 50%

        # TP should generally be positive, SL should be derived from negative MAE
        print(f"Suggested TP: {tp_level:.2%}, SL: {sl_level:.2%}")

    def test_regime_comparison(self):
        """Test comparing excursions across different market regimes."""
        np.random.seed(456)

        # Low volatility regime
        low_vol = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.005))

        # High volatility regime
        high_vol = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))

        result_low = analyze_excursions(pl.Series(low_vol), horizons=[20])
        result_high = analyze_excursions(pl.Series(high_vol), horizons=[20])

        # High vol should have larger excursions
        assert result_high.statistics[20].mfe_std > result_low.statistics[20].mfe_std
        assert abs(result_high.statistics[20].mae_std) > abs(result_low.statistics[20].mae_std)
