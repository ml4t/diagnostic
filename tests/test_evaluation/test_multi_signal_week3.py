"""Week 3 Tests: Multi-Signal Visualization.

Tests for:
- plot_ic_ridge() - IC density ridge plot
- plot_signal_ranking_bar() - Horizontal bar chart
- plot_signal_correlation_heatmap() - Cluster heatmap
- plot_pareto_frontier() - Pareto frontier scatter
- MultiSignalDashboard - Complete dashboard

References
----------
Tufte, E. (1983). "The Visual Display of Quantitative Information"
Few, S. (2012). "Show Me the Numbers"
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
from scipy import special

from ml4t.diagnostic.results.multi_signal_results import (
    ComparisonResult,
    MultiSignalSummary,
)
from ml4t.diagnostic.visualization.signal import (
    MultiSignalDashboard,
    plot_ic_ridge,
    plot_pareto_frontier,
    plot_signal_correlation_heatmap,
    plot_signal_ranking_bar,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_summary() -> MultiSignalSummary:
    """Create sample MultiSignalSummary for testing."""
    np.random.seed(42)
    n_signals = 20

    signal_names = [f"signal_{i:02d}" for i in range(n_signals)]
    ic_means = np.random.uniform(-0.05, 0.15, n_signals).tolist()
    ic_stds = np.random.uniform(0.02, 0.08, n_signals).tolist()
    ic_irs = [m / s if s > 0 else 0 for m, s in zip(ic_means, ic_stds)]
    ic_t_stats = [m / (s / np.sqrt(252)) for m, s in zip(ic_means, ic_stds)]
    ic_p_values = [2 * (1 - 0.5 * (1 + special.erf(abs(t) / np.sqrt(2)))) for t in ic_t_stats]
    turnover_means = np.random.uniform(0.1, 0.5, n_signals).tolist()

    # FDR/FWER significance
    fdr_significant = [p < 0.05 for p in ic_p_values]
    fwer_significant = [p < (0.05 / n_signals) for p in ic_p_values]

    summary_data = {
        "signal_name": signal_names,
        "ic_mean": ic_means,
        "ic_std": ic_stds,
        "ic_ir": ic_irs,
        "ic_t_stat": ic_t_stats,
        "ic_p_value": ic_p_values,
        "turnover_mean": turnover_means,
        "fdr_significant": fdr_significant,
        "fwer_significant": fwer_significant,
    }

    return MultiSignalSummary(
        summary_data=summary_data,
        n_signals=n_signals,
        n_fdr_significant=sum(fdr_significant),
        n_fwer_significant=sum(fwer_significant),
        periods=(1, 5, 10),
        fdr_alpha=0.05,
        fwer_alpha=0.05,
    )


@pytest.fixture
def sample_correlation_matrix() -> pl.DataFrame:
    """Create sample correlation matrix for testing."""
    np.random.seed(42)
    n_signals = 20
    signal_names = [f"signal_{i:02d}" for i in range(n_signals)]

    # Create a valid correlation matrix
    # Start with random matrix and make it symmetric positive semi-definite
    random_matrix = np.random.randn(n_signals, n_signals) * 0.3
    corr_matrix = (random_matrix + random_matrix.T) / 2

    # Ensure diagonal is 1
    np.fill_diagonal(corr_matrix, 1.0)

    # Clip to valid correlation range
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    return pl.DataFrame({name: corr_matrix[:, i].tolist() for i, name in enumerate(signal_names)})


@pytest.fixture
def sample_comparison(sample_summary: MultiSignalSummary) -> ComparisonResult:
    """Create sample ComparisonResult for testing."""
    selected_signals = ["signal_00", "signal_05", "signal_10"]

    tear_sheets = {
        signal: {
            "signal_name": signal,
            "ic_analysis": {
                "ic_mean": {"1D": 0.05},
                "ic_ir": {"1D": 0.8},
            },
        }
        for signal in selected_signals
    }

    correlation_matrix = {
        "signal_00": [1.0, 0.3, 0.2],
        "signal_05": [0.3, 1.0, 0.15],
        "signal_10": [0.2, 0.15, 1.0],
    }

    return ComparisonResult(
        signals=selected_signals,
        selection_method="top_n",
        selection_params={"n": 3, "metric": "ic_ir"},
        tear_sheets=tear_sheets,
        correlation_matrix=correlation_matrix,
    )


# =============================================================================
# Test: plot_ic_ridge
# =============================================================================


class TestPlotICRidge:
    """Tests for plot_ic_ridge function."""

    def test_basic_ridge_plot(self, sample_summary: MultiSignalSummary) -> None:
        """Test basic IC ridge plot creation."""
        fig = plot_ic_ridge(sample_summary)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text is not None
        assert "IC" in fig.layout.title.text

    def test_ridge_plot_max_signals(self, sample_summary: MultiSignalSummary) -> None:
        """Test max_signals parameter limits displayed signals."""
        fig = plot_ic_ridge(sample_summary, max_signals=5)

        # Should have traces for range + point per signal = 2 * n_displayed
        # Limited to 5 signals, so max 10 traces
        assert len(fig.data) <= 10

    def test_ridge_plot_sort_by(self, sample_summary: MultiSignalSummary) -> None:
        """Test sorting by different metrics."""
        fig_by_mean = plot_ic_ridge(sample_summary, sort_by="ic_mean")
        fig_by_ir = plot_ic_ridge(sample_summary, sort_by="ic_ir")

        assert isinstance(fig_by_mean, go.Figure)
        assert isinstance(fig_by_ir, go.Figure)

    def test_ridge_plot_invalid_sort_by(self, sample_summary: MultiSignalSummary) -> None:
        """Test error on invalid sort_by metric."""
        with pytest.raises(ValueError, match="not found"):
            plot_ic_ridge(sample_summary, sort_by="nonexistent_metric")

    def test_ridge_plot_themes(self, sample_summary: MultiSignalSummary) -> None:
        """Test different themes."""
        fig_light = plot_ic_ridge(sample_summary, theme="default")
        fig_dark = plot_ic_ridge(sample_summary, theme="dark")

        assert isinstance(fig_light, go.Figure)
        assert isinstance(fig_dark, go.Figure)

    def test_ridge_plot_custom_dimensions(self, sample_summary: MultiSignalSummary) -> None:
        """Test custom width and height."""
        fig = plot_ic_ridge(sample_summary, width=1000, height=800)

        assert fig.layout.width == 1000
        assert fig.layout.height == 800


# =============================================================================
# Test: plot_signal_ranking_bar
# =============================================================================


class TestPlotSignalRankingBar:
    """Tests for plot_signal_ranking_bar function."""

    def test_basic_ranking_bar(self, sample_summary: MultiSignalSummary) -> None:
        """Test basic ranking bar chart creation."""
        fig = plot_signal_ranking_bar(sample_summary)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].type == "bar"

    def test_ranking_bar_top_n(self, sample_summary: MultiSignalSummary) -> None:
        """Test top_n parameter limits bars."""
        fig = plot_signal_ranking_bar(sample_summary, top_n=5)

        # Should have exactly 5 bars
        assert len(fig.data[0].y) == 5

    def test_ranking_bar_different_metrics(self, sample_summary: MultiSignalSummary) -> None:
        """Test ranking by different metrics."""
        fig_ir = plot_signal_ranking_bar(sample_summary, metric="ic_ir")
        fig_mean = plot_signal_ranking_bar(sample_summary, metric="ic_mean")

        assert isinstance(fig_ir, go.Figure)
        assert isinstance(fig_mean, go.Figure)

    def test_ranking_bar_invalid_metric(self, sample_summary: MultiSignalSummary) -> None:
        """Test error on invalid metric."""
        with pytest.raises(ValueError, match="not found"):
            plot_signal_ranking_bar(sample_summary, metric="nonexistent")

    def test_ranking_bar_color_by(self, sample_summary: MultiSignalSummary) -> None:
        """Test coloring by significance."""
        fig_fdr = plot_signal_ranking_bar(sample_summary, color_by="fdr_significant")
        fig_fwer = plot_signal_ranking_bar(sample_summary, color_by="fwer_significant")
        fig_none = plot_signal_ranking_bar(sample_summary, color_by=None)

        assert isinstance(fig_fdr, go.Figure)
        assert isinstance(fig_fwer, go.Figure)
        assert isinstance(fig_none, go.Figure)


# =============================================================================
# Test: plot_signal_correlation_heatmap
# =============================================================================


class TestPlotSignalCorrelationHeatmap:
    """Tests for plot_signal_correlation_heatmap function."""

    def test_basic_heatmap(self, sample_correlation_matrix: pl.DataFrame) -> None:
        """Test basic correlation heatmap creation."""
        fig = plot_signal_correlation_heatmap(sample_correlation_matrix)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "heatmap"

    def test_heatmap_with_clustering(self, sample_correlation_matrix: pl.DataFrame) -> None:
        """Test heatmap with hierarchical clustering."""
        fig_clustered = plot_signal_correlation_heatmap(sample_correlation_matrix, cluster=True)
        fig_unclustered = plot_signal_correlation_heatmap(sample_correlation_matrix, cluster=False)

        assert isinstance(fig_clustered, go.Figure)
        assert isinstance(fig_unclustered, go.Figure)
        # Clustered should have "Clustered" in title
        assert "Clustered" in fig_clustered.layout.title.text

    def test_heatmap_max_signals(self, sample_correlation_matrix: pl.DataFrame) -> None:
        """Test max_signals parameter."""
        fig = plot_signal_correlation_heatmap(sample_correlation_matrix, max_signals=10)

        # Heatmap should be limited to 10x10
        assert fig.data[0].z.shape[0] <= 10
        assert fig.data[0].z.shape[1] <= 10

    def test_heatmap_themes(self, sample_correlation_matrix: pl.DataFrame) -> None:
        """Test different themes."""
        fig_light = plot_signal_correlation_heatmap(sample_correlation_matrix, theme="default")
        fig_dark = plot_signal_correlation_heatmap(sample_correlation_matrix, theme="dark")

        assert isinstance(fig_light, go.Figure)
        assert isinstance(fig_dark, go.Figure)

    def test_heatmap_colorscale(self, sample_correlation_matrix: pl.DataFrame) -> None:
        """Test that colorscale is diverging (centered at 0)."""
        fig = plot_signal_correlation_heatmap(sample_correlation_matrix)

        # Check zmid is 0 (diverging at zero)
        assert fig.data[0].zmid == 0


# =============================================================================
# Test: plot_pareto_frontier
# =============================================================================


class TestPlotParetoFrontier:
    """Tests for plot_pareto_frontier function."""

    def test_basic_pareto_plot(self, sample_summary: MultiSignalSummary) -> None:
        """Test basic Pareto frontier plot creation."""
        fig = plot_pareto_frontier(sample_summary)

        assert isinstance(fig, go.Figure)
        # Should have at least scatter points
        assert len(fig.data) >= 1

    def test_pareto_highlights_frontier(self, sample_summary: MultiSignalSummary) -> None:
        """Test that Pareto frontier is highlighted."""
        fig = plot_pareto_frontier(sample_summary, highlight_pareto=True)

        # Should have scatter + line for frontier
        assert len(fig.data) >= 2

    def test_pareto_no_highlight(self, sample_summary: MultiSignalSummary) -> None:
        """Test without Pareto highlighting."""
        fig = plot_pareto_frontier(sample_summary, highlight_pareto=False)

        # Should have only scatter (no frontier line)
        assert len(fig.data) == 1

    def test_pareto_different_metrics(self, sample_summary: MultiSignalSummary) -> None:
        """Test with different x/y metrics."""
        fig = plot_pareto_frontier(
            sample_summary,
            x_metric="ic_std",
            y_metric="ic_mean",
        )

        assert isinstance(fig, go.Figure)
        assert "ic_std" in fig.layout.xaxis.title.text.lower().replace(" ", "_")

    def test_pareto_invalid_metric(self, sample_summary: MultiSignalSummary) -> None:
        """Test error on invalid metric."""
        with pytest.raises(ValueError, match="not found"):
            plot_pareto_frontier(sample_summary, x_metric="nonexistent")

    def test_pareto_minimize_maximize(self, sample_summary: MultiSignalSummary) -> None:
        """Test different minimize/maximize settings."""
        fig1 = plot_pareto_frontier(sample_summary, minimize_x=True, maximize_y=True)
        fig2 = plot_pareto_frontier(sample_summary, minimize_x=False, maximize_y=False)

        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)

    def test_pareto_has_annotation(self, sample_summary: MultiSignalSummary) -> None:
        """Test that Pareto annotation is present."""
        fig = plot_pareto_frontier(sample_summary)

        # Should have annotation with Pareto count
        assert len(fig.layout.annotations) >= 1
        assert "Pareto" in fig.layout.annotations[0].text


# =============================================================================
# Test: MultiSignalDashboard
# =============================================================================


class TestMultiSignalDashboard:
    """Tests for MultiSignalDashboard class."""

    def test_dashboard_initialization(self) -> None:
        """Test dashboard initialization."""
        dashboard = MultiSignalDashboard()
        assert dashboard.title == "Multi-Signal Analysis Dashboard"
        assert dashboard.theme == "light"

    def test_dashboard_custom_title(self) -> None:
        """Test dashboard with custom title."""
        dashboard = MultiSignalDashboard(title="Custom Title", theme="dark")
        assert dashboard.title == "Custom Title"
        assert dashboard.theme == "dark"

    def test_dashboard_generate_basic(self, sample_summary: MultiSignalSummary) -> None:
        """Test basic dashboard generation."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(sample_summary)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Multi-Signal Analysis Dashboard" in html
        assert "Summary" in html

    def test_dashboard_with_correlation(
        self,
        sample_summary: MultiSignalSummary,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Test dashboard with correlation matrix."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(
            sample_summary,
            correlation_matrix=sample_correlation_matrix,
        )

        assert "Correlation" in html
        assert "heatmap" in html.lower() or "plotly" in html.lower()

    def test_dashboard_with_comparison(
        self,
        sample_summary: MultiSignalSummary,
        sample_comparison: ComparisonResult,
    ) -> None:
        """Test dashboard with comparison result."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(
            sample_summary,
            comparison=sample_comparison,
        )

        assert "Comparison" in html
        assert "signal_00" in html

    def test_dashboard_full_features(
        self,
        sample_summary: MultiSignalSummary,
        sample_correlation_matrix: pl.DataFrame,
        sample_comparison: ComparisonResult,
    ) -> None:
        """Test dashboard with all features."""
        dashboard = MultiSignalDashboard(title="Full Dashboard")
        html = dashboard.generate(
            sample_summary,
            correlation_matrix=sample_correlation_matrix,
            comparison=sample_comparison,
        )

        # Check all tabs present
        assert "Summary" in html
        assert "Distribution" in html
        assert "Correlation" in html
        assert "Efficiency" in html
        assert "Comparison" in html

    def test_dashboard_dark_theme(self, sample_summary: MultiSignalSummary) -> None:
        """Test dashboard with dark theme."""
        dashboard = MultiSignalDashboard(theme="dark")
        html = dashboard.generate(sample_summary)

        # Dark theme should have dark background colors
        assert "#1e1e1e" in html or "dark" in html.lower()

    def test_dashboard_save(self, sample_summary: MultiSignalSummary, tmp_path) -> None:
        """Test saving dashboard to file."""
        dashboard = MultiSignalDashboard()
        output_path = str(tmp_path / "test_dashboard.html")

        saved_path = dashboard.save(output_path, sample_summary)

        assert saved_path == output_path
        assert (tmp_path / "test_dashboard.html").exists()

        # Read and verify content
        with open(saved_path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    def test_dashboard_metric_cards(self, sample_summary: MultiSignalSummary) -> None:
        """Test that metric cards are generated."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(sample_summary)

        # Should have metric grid and cards
        assert "metric-grid" in html
        assert "metric-card" in html
        assert "FDR Significant" in html
        assert "FWER Significant" in html

    def test_dashboard_searchable_table(self, sample_summary: MultiSignalSummary) -> None:
        """Test that signal table is searchable."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(sample_summary)

        # Should have search input
        assert "signal-search" in html
        assert "filterSignalTable" in html

    def test_dashboard_tab_navigation(self, sample_summary: MultiSignalSummary) -> None:
        """Test tab navigation is present."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(sample_summary)

        # Should have tab buttons
        assert "tab-button" in html
        assert "switchTab" in html

    def test_dashboard_plotly_cdn(self, sample_summary: MultiSignalSummary) -> None:
        """Test that Plotly.js is loaded from CDN."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(sample_summary)

        assert "cdn.plot.ly" in html


# =============================================================================
# Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests for multi-signal visualization."""

    def test_all_plots_compatible_with_summary(self, sample_summary: MultiSignalSummary) -> None:
        """Test all plot functions work with same summary."""
        fig_ridge = plot_ic_ridge(sample_summary)
        fig_bar = plot_signal_ranking_bar(sample_summary)
        fig_pareto = plot_pareto_frontier(sample_summary)

        assert all(isinstance(f, go.Figure) for f in [fig_ridge, fig_bar, fig_pareto])

    def test_dashboard_uses_all_plot_types(
        self,
        sample_summary: MultiSignalSummary,
        sample_correlation_matrix: pl.DataFrame,
    ) -> None:
        """Test dashboard generates all plot types."""
        dashboard = MultiSignalDashboard()
        html = dashboard.generate(
            sample_summary,
            correlation_matrix=sample_correlation_matrix,
        )

        # Should contain plotly plots
        assert "plotly" in html.lower()
        # At least one plot container
        assert html.count("plot-container") >= 1
        # Should have multiple tabs (distribution, correlation, efficiency)
        assert "Distribution" in html
        assert "Correlation" in html
        assert "Efficiency" in html

    def test_plots_serializable_to_html(self, sample_summary: MultiSignalSummary) -> None:
        """Test all plots can be serialized to HTML."""
        plots = [
            plot_ic_ridge(sample_summary),
            plot_signal_ranking_bar(sample_summary),
            plot_pareto_frontier(sample_summary),
        ]

        for fig in plots:
            html = fig.to_html(full_html=False, include_plotlyjs=False)
            assert isinstance(html, str)
            assert len(html) > 100
