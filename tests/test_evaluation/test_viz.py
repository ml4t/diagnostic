"""Tests for visualization module."""

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation import visualization as viz


class TestICHeatmap:
    """Test IC heatmap visualization."""

    def test_basic_ic_heatmap(self):
        """Test basic IC heatmap creation."""
        # Generate test data
        np.random.seed(42)
        n_samples = 200
        predictions = pd.Series(np.random.randn(n_samples))
        returns = pd.DataFrame(
            {
                "1d": predictions + 0.1 * np.random.randn(n_samples),
                "5d": predictions + 0.3 * np.random.randn(n_samples),
                "10d": predictions + 0.5 * np.random.randn(n_samples),
            },
        )

        fig = viz.plot_ic_heatmap(predictions, returns)

        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == "heatmap"

    def test_ic_heatmap_with_time_index(self):
        """Test IC heatmap with datetime index."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        predictions = pd.Series(np.random.randn(100), index=dates)
        returns = pd.DataFrame(
            np.random.randn(100, 3),
            index=dates,
            columns=["1d", "5d", "10d"],
        )

        fig = viz.plot_ic_heatmap(predictions, returns, time_index=dates)

        assert fig is not None
        assert "Date" in fig.layout.xaxis.title.text

    def test_ic_heatmap_numpy_inputs(self):
        """Test IC heatmap with numpy array inputs."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        returns = np.random.randn(100, 3)

        fig = viz.plot_ic_heatmap(predictions, returns, horizons=[1, 5, 10])

        assert fig is not None
        assert len(fig.data[0].y) == 3  # 3 horizons


class TestQuantileReturns:
    """Test quantile returns visualization."""

    def test_basic_quantile_plot(self):
        """Test basic quantile returns plot."""
        np.random.seed(42)
        predictions = pd.Series(np.random.randn(1000))
        returns = pd.Series(0.01 * predictions + 0.02 * np.random.randn(1000))

        fig = viz.plot_quantile_returns(predictions, returns)

        assert fig is not None
        assert any("bar" in str(trace.type).lower() for trace in fig.data)

    def test_quantile_plot_with_cumulative(self):
        """Test quantile plot with cumulative returns."""
        np.random.seed(42)
        predictions = np.random.randn(500)
        returns = 0.005 * predictions + 0.01 * np.random.randn(500)

        fig = viz.plot_quantile_returns(
            predictions,
            returns,
            n_quantiles=4,
            show_cumulative=True,
        )

        assert fig is not None
        # Should have bar chart and line charts
        assert any("bar" in str(trace.type).lower() for trace in fig.data)
        assert any("scatter" in str(trace.type).lower() for trace in fig.data)

    def test_quantile_monotonicity(self):
        """Test that quantile returns show expected monotonicity."""
        np.random.seed(42)
        # Create data with clear monotonic relationship
        predictions = np.random.randn(1000)
        returns = 0.05 * predictions + 0.01 * np.random.randn(1000)

        fig = viz.plot_quantile_returns(predictions, returns, n_quantiles=5)

        # Extract quantile returns from bar chart
        bar_data = next(trace for trace in fig.data if trace.type == "bar")
        quantile_returns = bar_data.y

        # Should be mostly increasing (allowing for some noise)
        increases = sum(
            quantile_returns[i] < quantile_returns[i + 1] for i in range(len(quantile_returns) - 1)
        )
        assert increases >= 2  # At least 2 increases out of 4 transitions


class TestTurnoverDecay:
    """Test turnover and decay visualization."""

    def test_basic_turnover_plot(self):
        """Test basic turnover decay plot."""
        np.random.seed(42)
        # Create factor values with some persistence
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        n_assets = 50

        factor_values = pd.DataFrame(
            np.random.randn(100, n_assets),
            index=dates,
            columns=[f"asset_{i}" for i in range(n_assets)],
        )
        # Add some autocorrelation
        for col in factor_values.columns:
            factor_values[col] = factor_values[col].rolling(5).mean().fillna(0)

        fig = viz.plot_turnover_decay(factor_values)

        assert fig is not None
        # Should have multiple subplots
        assert len(fig.data) > 3

    def test_turnover_with_custom_lags(self):
        """Test turnover plot with custom lag specification."""
        np.random.seed(42)
        factor_values = pd.DataFrame(
            np.random.randn(50, 20),
            columns=[f"asset_{i}" for i in range(20)],
        )

        fig = viz.plot_turnover_decay(factor_values, quantiles=3, lags=[1, 2, 5])

        assert fig is not None
        # Check that autocorrelation plot has correct number of points
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert any(len(t.x) == 3 for t in scatter_traces)  # 3 lags


class TestFeatureDistributions:
    """Test feature distribution visualization."""

    def test_basic_distribution_plot(self):
        """Test basic feature distribution plot."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(200),
                "feature2": np.random.exponential(1, 200),
                "feature3": np.random.uniform(-1, 1, 200),
            },
            index=dates,
        )

        fig = viz.plot_feature_distributions(features)

        assert fig is not None
        assert len(fig.data) > 0

    def test_distribution_methods(self):
        """Test different distribution visualization methods."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
            },
        )

        # Test box plot
        fig_box = viz.plot_feature_distributions(features, method="box")
        assert any("box" in str(trace.type).lower() for trace in fig_box.data)

        # Test violin plot
        fig_violin = viz.plot_feature_distributions(features, method="violin")
        assert any("violin" in str(trace.type).lower() for trace in fig_violin.data)

        # Test histogram
        fig_hist = viz.plot_feature_distributions(features, method="hist")
        assert any("histogram" in str(trace.type).lower() for trace in fig_hist.data)

    def test_distribution_time_evolution(self):
        """Test that distributions show time evolution."""
        np.random.seed(42)
        # Create features that change over time
        n_samples = 400
        time_trend = np.linspace(0, 2, n_samples)

        features = pd.DataFrame(
            {
                "trending": np.random.randn(n_samples) + time_trend,
                "stable": np.random.randn(n_samples),
            },
        )

        fig = viz.plot_feature_distributions(features, n_periods=4)

        assert fig is not None
        # Should show multiple periods
        box_traces = [t for t in fig.data if t.type == "box"]
        unique_names = {t.name for t in box_traces}
        assert len(unique_names) == 4  # 4 periods


class TestThemeApplication:
    """Test theme application to plots."""

    def test_theme_application(self):
        """Test that themes can be applied to figures."""
        from ml4t.diagnostic.evaluation.themes import apply_theme

        np.random.seed(42)
        predictions = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))

        fig = viz.plot_quantile_returns(predictions, returns)

        # Apply different themes
        fig_default = apply_theme(fig, "default")
        assert fig_default is not None

        fig_dark = apply_theme(fig, "dark")
        assert fig_dark.layout.plot_bgcolor == "#1E1E1E"

        fig_print = apply_theme(fig, "print")
        assert fig_print.layout.plot_bgcolor == "white"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_series = pd.Series([])
        pd.DataFrame()

        # Should handle empty data gracefully
        fig = viz.plot_quantile_returns(empty_series, empty_series)
        assert fig is not None

    def test_single_value(self):
        """Test handling of single value inputs."""
        single_pred = pd.Series([1.0])
        single_ret = pd.Series([0.01])

        # Should handle single values
        fig = viz.plot_quantile_returns(single_pred, single_ret, n_quantiles=1)
        assert fig is not None

    def test_all_nan_values(self):
        """Test handling of all NaN values."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])

        # Should handle NaN values
        fig = viz.plot_quantile_returns(nan_series, nan_series)
        assert fig is not None
