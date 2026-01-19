"""Tests for diagnostic visualization functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.evaluation.diagnostic_plots import (
    export_static,
    get_figure_data,
    plot_acf_pacf,
    plot_distribution,
    plot_qq,
    plot_volatility_clustering,
)


class TestPlotACFPACF:
    """Tests for ACF/PACF plotting function."""

    def test_basic_plot(self):
        """Test basic ACF/PACF plot creation."""
        # White noise
        data = np.random.randn(500)
        fig = plot_acf_pacf(data, max_lags=20)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Has traces

    def test_with_pandas_series(self):
        """Test with pandas Series input."""
        data = pd.Series(np.random.randn(500))
        fig = plot_acf_pacf(data, max_lags=20)

        assert isinstance(fig, go.Figure)

    def test_ar_process(self):
        """Test with AR(1) process - should show significant PACF at lag 1."""
        # Generate AR(1) process
        n = 1000
        data = np.zeros(n)
        data[0] = np.random.randn()
        for i in range(1, n):
            data[i] = 0.7 * data[i - 1] + np.random.randn()

        fig = plot_acf_pacf(data, max_lags=20)

        assert isinstance(fig, go.Figure)
        # AR(1) should have significant PACF only at lag 1

    def test_custom_alpha(self):
        """Test with custom significance level."""
        data = np.random.randn(500)
        fig = plot_acf_pacf(data, max_lags=20, alpha=0.01)

        assert isinstance(fig, go.Figure)
        # Title should reflect 99% confidence
        assert "99" in fig.layout.title.text or "99" in str(fig.layout.annotations)

    def test_custom_title(self):
        """Test with custom title."""
        data = np.random.randn(500)
        title = "Custom ACF/PACF Analysis"
        fig = plot_acf_pacf(data, max_lags=20, title=title)

        assert isinstance(fig, go.Figure)
        assert title in fig.layout.title.text

    def test_max_lags_too_large(self):
        """Test when max_lags exceeds data length."""
        data = np.random.randn(50)
        # Should automatically adjust max_lags
        fig = plot_acf_pacf(data, max_lags=100)

        assert isinstance(fig, go.Figure)

    def test_empty_data(self):
        """Test with empty data raises error."""
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            plot_acf_pacf(data)

    def test_all_nan_data(self):
        """Test with all NaN data raises error."""
        data = np.full(100, np.nan)

        with pytest.raises(ValueError, match="empty"):
            plot_acf_pacf(data)

    def test_data_with_some_nans(self):
        """Test with data containing some NaNs."""
        data = np.random.randn(500)
        data[::10] = np.nan  # 10% NaN
        fig = plot_acf_pacf(data, max_lags=20)

        assert isinstance(fig, go.Figure)

    def test_custom_height(self):
        """Test with custom height parameter."""
        data = np.random.randn(500)
        fig = plot_acf_pacf(data, max_lags=20, height=600)

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 600


class TestPlotQQ:
    """Tests for QQ plot function."""

    def test_basic_qq_normal(self):
        """Test basic QQ plot against normal distribution."""
        data = np.random.randn(500)
        fig = plot_qq(data, distribution="norm")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Sample points + reference line

    def test_qq_with_pandas_series(self):
        """Test with pandas Series input."""
        data = pd.Series(np.random.randn(500))
        fig = plot_qq(data)

        assert isinstance(fig, go.Figure)

    def test_qq_t_distribution(self):
        """Test QQ plot against Student's t distribution."""
        # Generate t-distributed data
        data = np.random.standard_t(df=5, size=500)
        fig = plot_qq(data, distribution="t")

        assert isinstance(fig, go.Figure)
        assert "Student's t" in fig.layout.title.text

    def test_qq_uniform_distribution(self):
        """Test QQ plot against uniform distribution."""
        data = np.random.uniform(0, 1, size=500)
        fig = plot_qq(data, distribution="uniform")

        assert isinstance(fig, go.Figure)
        assert "Uniform" in fig.layout.title.text

    def test_qq_invalid_distribution(self):
        """Test with invalid distribution name."""
        data = np.random.randn(500)

        with pytest.raises(ValueError, match="Unknown distribution"):
            plot_qq(data, distribution="invalid")

    def test_qq_custom_title(self):
        """Test with custom title."""
        data = np.random.randn(500)
        title = "Custom QQ Plot"
        fig = plot_qq(data, title=title)

        assert isinstance(fig, go.Figure)
        assert title in fig.layout.title.text

    def test_qq_empty_data(self):
        """Test with empty data raises error."""
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            plot_qq(data)

    def test_qq_all_nan_data(self):
        """Test with all NaN data raises error."""
        data = np.full(100, np.nan)

        with pytest.raises(ValueError, match="empty"):
            plot_qq(data)

    def test_qq_heavy_tails(self):
        """Test QQ plot with heavy-tailed data."""
        # Generate data with heavy tails (Student's t with low df)
        data = np.random.standard_t(df=3, size=500)
        fig = plot_qq(data, distribution="norm")

        assert isinstance(fig, go.Figure)
        # With heavy tails, should see S-curve pattern

    def test_qq_custom_dimensions(self):
        """Test with custom width and height."""
        data = np.random.randn(500)
        fig = plot_qq(data, width=600, height=600)

        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 600
        assert fig.layout.height == 600


class TestPlotVolatilityClustering:
    """Tests for volatility clustering plot."""

    def test_basic_volatility_plot(self):
        """Test basic volatility clustering plot."""
        data = np.random.randn(500)
        fig = plot_volatility_clustering(data, window=20)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # At least 4 traces (one per panel)

    def test_with_pandas_series(self):
        """Test with pandas Series with datetime index."""
        index = pd.date_range("2020-01-01", periods=500, freq="D")
        data = pd.Series(np.random.randn(500), index=index)
        fig = plot_volatility_clustering(data, window=20)

        assert isinstance(fig, go.Figure)

    def test_garch_like_data(self):
        """Test with GARCH-like volatility clustering."""
        # Generate data with volatility clustering
        n = 1000
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.1

        for t in range(1, n):
            sigma[t] = np.sqrt(0.01 + 0.05 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2)
            returns[t] = sigma[t] * np.random.randn()

        fig = plot_volatility_clustering(returns, window=20)

        assert isinstance(fig, go.Figure)
        # Should show clear volatility clustering

    def test_custom_window(self):
        """Test with custom rolling window."""
        data = np.random.randn(500)
        fig = plot_volatility_clustering(data, window=50)

        assert isinstance(fig, go.Figure)

    def test_custom_title(self):
        """Test with custom title."""
        data = np.random.randn(500)
        title = "Custom Volatility Analysis"
        fig = plot_volatility_clustering(data, window=20, title=title)

        assert isinstance(fig, go.Figure)
        assert title in fig.layout.title.text

    def test_empty_data(self):
        """Test with empty data raises error."""
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            plot_volatility_clustering(data)

    def test_all_nan_data(self):
        """Test with all NaN data raises error."""
        data = np.full(100, np.nan)

        with pytest.raises(ValueError, match="empty"):
            plot_volatility_clustering(data)

    def test_data_with_some_nans(self):
        """Test with data containing some NaNs."""
        data = np.random.randn(500)
        data[::10] = np.nan
        fig = plot_volatility_clustering(data, window=20)

        assert isinstance(fig, go.Figure)

    def test_custom_height(self):
        """Test with custom height parameter."""
        data = np.random.randn(500)
        fig = plot_volatility_clustering(data, window=20, height=1000)

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 1000


class TestPlotDistribution:
    """Tests for distribution plotting function."""

    def test_basic_distribution_plot(self):
        """Test basic distribution plot with normal fit."""
        data = np.random.randn(500)
        fig = plot_distribution(data, bins=50, fit_normal=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Histogram + normal curve

    def test_with_pandas_series(self):
        """Test with pandas Series input."""
        data = pd.Series(np.random.randn(500))
        fig = plot_distribution(data)

        assert isinstance(fig, go.Figure)

    def test_fit_t_distribution(self):
        """Test with Student's t distribution fit."""
        data = np.random.standard_t(df=5, size=500)
        fig = plot_distribution(data, fit_normal=True, fit_t=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Histogram + normal + t

    def test_no_fitted_distributions(self):
        """Test with no fitted distributions."""
        data = np.random.randn(500)
        fig = plot_distribution(data, fit_normal=False, fit_t=False)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Only histogram

    def test_hide_moments(self):
        """Test with moments statistics hidden."""
        data = np.random.randn(500)
        fig = plot_distribution(data, show_moments=False)

        assert isinstance(fig, go.Figure)
        # No moments annotation

    def test_custom_bins(self):
        """Test with custom number of bins."""
        data = np.random.randn(500)
        fig = plot_distribution(data, bins=30)

        assert isinstance(fig, go.Figure)

    def test_custom_title(self):
        """Test with custom title."""
        data = np.random.randn(500)
        title = "Custom Distribution"
        fig = plot_distribution(data, title=title)

        assert isinstance(fig, go.Figure)
        assert title in fig.layout.title.text

    def test_empty_data(self):
        """Test with empty data raises error."""
        data = np.array([])

        with pytest.raises(ValueError, match="empty"):
            plot_distribution(data)

    def test_all_nan_data(self):
        """Test with all NaN data raises error."""
        data = np.full(100, np.nan)

        with pytest.raises(ValueError, match="empty"):
            plot_distribution(data)

    def test_skewed_data(self):
        """Test with skewed data."""
        # Generate skewed data
        data = np.random.exponential(scale=2.0, size=500)
        fig = plot_distribution(data, fit_normal=True)

        assert isinstance(fig, go.Figure)
        # Should show positive skewness in moments

    def test_heavy_tailed_data(self):
        """Test with heavy-tailed data."""
        data = np.random.standard_t(df=3, size=500)
        fig = plot_distribution(data, fit_normal=True, fit_t=True)

        assert isinstance(fig, go.Figure)
        # Should show high kurtosis in moments

    def test_custom_height(self):
        """Test with custom height parameter."""
        data = np.random.randn(500)
        fig = plot_distribution(data, height=600)

        assert isinstance(fig, go.Figure)
        assert fig.layout.height == 600


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_figure_data(self):
        """Test extracting data from Plotly figure."""
        data = np.random.randn(500)
        fig = plot_acf_pacf(data, max_lags=20)

        df = get_figure_data(fig)

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) > 0

    def test_get_figure_data_qq(self):
        """Test extracting data from QQ plot."""
        data = np.random.randn(500)
        fig = plot_qq(data)

        df = get_figure_data(fig)

        assert isinstance(df, pd.DataFrame)
        # Should have x and y for sample data and reference line

    def test_export_static_missing_kaleido(self):
        """Test export_static handles missing kaleido gracefully."""
        data = np.random.randn(500)
        fig = plot_acf_pacf(data, max_lags=20)

        # Should not raise, just print message
        # (may succeed if kaleido is installed)
        export_static(fig, "/tmp/test_plot", format="png")


class TestIntegration:
    """Integration tests for diagnostic plots."""

    def test_all_plots_for_same_data(self):
        """Test creating all diagnostic plots for the same dataset."""
        # Generate sample data
        data = np.random.randn(1000)

        # Create all plots
        fig_acf = plot_acf_pacf(data)
        fig_qq = plot_qq(data)
        fig_vol = plot_volatility_clustering(data)
        fig_dist = plot_distribution(data)

        assert all(isinstance(fig, go.Figure) for fig in [fig_acf, fig_qq, fig_vol, fig_dist])

    def test_garch_process_diagnostics(self):
        """Test full diagnostic suite on GARCH process."""
        # Generate GARCH(1,1) process
        n = 2000
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.1

        for t in range(1, n):
            sigma[t] = np.sqrt(0.01 + 0.05 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2)
            returns[t] = sigma[t] * np.random.randn()

        # All plots should work
        fig_acf = plot_acf_pacf(returns, max_lags=50)
        fig_qq = plot_qq(returns)
        fig_vol = plot_volatility_clustering(returns, window=20)
        fig_dist = plot_distribution(returns, fit_normal=True, fit_t=True)

        assert all(isinstance(fig, go.Figure) for fig in [fig_acf, fig_qq, fig_vol, fig_dist])

    def test_ar_process_diagnostics(self):
        """Test full diagnostic suite on AR process."""
        # Generate AR(2) process
        n = 1000
        data = np.zeros(n)
        data[0] = np.random.randn()
        data[1] = np.random.randn()

        for i in range(2, n):
            data[i] = 0.6 * data[i - 1] - 0.3 * data[i - 2] + np.random.randn()

        # All plots should work
        fig_acf = plot_acf_pacf(data, max_lags=30)
        fig_qq = plot_qq(data)
        fig_vol = plot_volatility_clustering(data, window=20)
        fig_dist = plot_distribution(data)

        assert all(isinstance(fig, go.Figure) for fig in [fig_acf, fig_qq, fig_vol, fig_dist])

    def test_real_world_scenario(self):
        """Test with realistic financial returns data."""
        # Simulate realistic stock returns
        # - Small mean
        # - Volatility clustering
        # - Slight negative skew
        # - Heavy tails

        n = 2000
        returns = np.zeros(n)
        sigma = np.zeros(n)
        sigma[0] = 0.02  # 2% daily vol

        for t in range(1, n):
            # GARCH(1,1) volatility
            sigma[t] = np.sqrt(0.0001 + 0.08 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2)
            # Use t-distribution for heavy tails
            returns[t] = sigma[t] * np.random.standard_t(df=5)

        # Add small positive drift
        returns += 0.0002

        # Create all diagnostic plots
        fig_acf = plot_acf_pacf(returns, max_lags=40)
        fig_qq = plot_qq(returns, distribution="norm")
        fig_vol = plot_volatility_clustering(returns, window=20)
        fig_dist = plot_distribution(returns, fit_normal=True, fit_t=True)

        assert all(isinstance(fig, go.Figure) for fig in [fig_acf, fig_qq, fig_vol, fig_dist])

    def test_pandas_series_with_datetime_index(self):
        """Test all plots work with pandas Series and datetime index."""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        data = pd.Series(np.random.randn(1000), index=dates)

        fig_acf = plot_acf_pacf(data)
        fig_qq = plot_qq(data)
        fig_vol = plot_volatility_clustering(data)
        fig_dist = plot_distribution(data)

        assert all(isinstance(fig, go.Figure) for fig in [fig_acf, fig_qq, fig_vol, fig_dist])

    def test_all_plots_are_interactive(self):
        """Test that all plots have interactive features enabled."""
        data = np.random.randn(500)

        fig_acf = plot_acf_pacf(data)
        fig_qq = plot_qq(data)
        fig_vol = plot_volatility_clustering(data)
        fig_dist = plot_distribution(data)

        # All should have hovermode set
        for fig in [fig_acf, fig_qq, fig_vol, fig_dist]:
            assert fig.layout.hovermode is not None

    def test_data_extraction_from_all_plots(self):
        """Test that data can be extracted from all plot types."""
        data = np.random.randn(500)

        fig_acf = plot_acf_pacf(data)
        fig_qq = plot_qq(data)
        fig_vol = plot_volatility_clustering(data)
        fig_dist = plot_distribution(data)

        # All should be able to export data
        for fig in [fig_acf, fig_qq, fig_vol, fig_dist]:
            df = get_figure_data(fig)
            assert isinstance(df, pd.DataFrame)
            assert len(df.columns) > 0
