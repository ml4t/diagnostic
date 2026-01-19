"""Tests for feature importance visualization functions."""

import numpy as np
import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.visualization.feature_plots import (
    plot_importance_bar,
    plot_importance_distribution,
    plot_importance_heatmap,
    plot_importance_summary,
)


@pytest.fixture
def mock_importance_results():
    """Create mock results from analyze_ml_importance()."""
    feature_names = [f"feature_{i}" for i in range(20)]

    # Create realistic importance scores for each method
    np.random.seed(42)

    # MDI: Tree-based importance
    mdi_importances = np.random.exponential(scale=0.05, size=20)
    mdi_importances = mdi_importances / mdi_importances.sum()

    # PFI: Permutation importance (similar pattern, different values)
    pfi_importances = np.random.exponential(scale=0.04, size=20)
    pfi_importances = pfi_importances / pfi_importances.sum()
    pfi_std = pfi_importances * 0.1  # Some variation

    # SHAP: SHAP values (correlated with others but not identical)
    shap_importances = np.random.exponential(scale=0.045, size=20)
    shap_importances = shap_importances / shap_importances.sum()

    # Sort features by importance for each method
    mdi_order = np.argsort(mdi_importances)[::-1]
    pfi_order = np.argsort(pfi_importances)[::-1]
    shap_order = np.argsort(shap_importances)[::-1]

    # Consensus ranking (average rank across methods)
    ranks = np.zeros(20)
    for i in range(20):
        ranks[i] = (
            np.where(mdi_order == i)[0][0]
            + np.where(pfi_order == i)[0][0]
            + np.where(shap_order == i)[0][0]
        ) / 3
    consensus_order = np.argsort(ranks)

    return {
        "method_results": {
            "mdi": {
                "feature_names": [feature_names[i] for i in mdi_order],
                "importances": mdi_importances[mdi_order],
                "method": "mdi",
            },
            "pfi": {
                "feature_names": [feature_names[i] for i in pfi_order],
                "importances_mean": pfi_importances[pfi_order],
                "importances_std": pfi_std[pfi_order],
                "method": "pfi",
            },
            "shap": {
                "feature_names": [feature_names[i] for i in shap_order],
                "importances": shap_importances[shap_order],
                "method": "shap",
            },
        },
        "consensus_ranking": [feature_names[i] for i in consensus_order],
        "method_agreement": {
            "mdi_vs_pfi": 0.85,
            "mdi_vs_shap": 0.72,
            "pfi_vs_shap": 0.78,
        },
        "top_features_consensus": [feature_names[i] for i in consensus_order[:5]],
        "warnings": [],
        "interpretation": "Strong consensus across methods",
        "methods_run": ["mdi", "pfi", "shap"],
        "methods_failed": [],
    }


@pytest.fixture
def mock_importance_results_two_methods():
    """Create mock results with only two methods."""
    feature_names = [f"feature_{i}" for i in range(10)]
    np.random.seed(42)

    mdi_importances = np.random.exponential(scale=0.1, size=10)
    pfi_importances = np.random.exponential(scale=0.09, size=10)

    mdi_order = np.argsort(mdi_importances)[::-1]
    pfi_order = np.argsort(pfi_importances)[::-1]

    ranks = np.zeros(10)
    for i in range(10):
        ranks[i] = (np.where(mdi_order == i)[0][0] + np.where(pfi_order == i)[0][0]) / 2
    consensus_order = np.argsort(ranks)

    return {
        "method_results": {
            "mdi": {
                "feature_names": [feature_names[i] for i in mdi_order],
                "importances": mdi_importances[mdi_order],
            },
            "pfi": {
                "feature_names": [feature_names[i] for i in pfi_order],
                "importances_mean": pfi_importances[pfi_order],
            },
        },
        "consensus_ranking": [feature_names[i] for i in consensus_order],
        "method_agreement": {"mdi_vs_pfi": 0.65},
        "top_features_consensus": [feature_names[i] for i in consensus_order[:3]],
        "methods_run": ["mdi", "pfi"],
        "methods_failed": [],
    }


# ===================================================================
# plot_importance_bar() tests
# ===================================================================


class TestPlotImportanceBar:
    """Tests for plot_importance_bar()."""

    def test_basic_usage(self, mock_importance_results):
        """Test basic bar plot creation."""
        fig = plot_importance_bar(mock_importance_results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert fig.data[0].orientation == "h"

        # Check data
        assert len(fig.data[0].y) == 20  # All features by default
        assert fig.data[0].y[0] == mock_importance_results["consensus_ranking"][0]

    def test_top_n_filtering(self, mock_importance_results):
        """Test top_n parameter."""
        fig = plot_importance_bar(mock_importance_results, top_n=10)

        assert len(fig.data[0].y) == 10
        assert fig.data[0].y[0] == mock_importance_results["consensus_ranking"][0]
        assert fig.data[0].y[9] == mock_importance_results["consensus_ranking"][9]

    def test_custom_title(self, mock_importance_results):
        """Test custom title."""
        custom_title = "My Custom Feature Importance"
        fig = plot_importance_bar(mock_importance_results, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_importance_results):
        """Test theme customization."""
        fig = plot_importance_bar(mock_importance_results, theme="dark")

        # Dark theme should have dark background (case-insensitive)
        assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d", "#000000"]

    def test_color_scheme(self, mock_importance_results):
        """Test color scheme parameter."""
        fig = plot_importance_bar(mock_importance_results, color_scheme="plasma")

        # Should have colorscale set
        assert fig.data[0].marker.colorscale is not None

    def test_show_values(self, mock_importance_results):
        """Test show_values parameter."""
        # With values
        fig = plot_importance_bar(mock_importance_results, show_values=True)
        assert fig.data[0].text is not None

        # Without values
        fig = plot_importance_bar(mock_importance_results, show_values=False)
        assert fig.data[0].text is None

    def test_custom_dimensions(self, mock_importance_results):
        """Test custom width and height."""
        fig = plot_importance_bar(mock_importance_results, width=1200, height=800, top_n=10)

        assert fig.layout.width == 1200
        assert fig.layout.height == 800

    def test_height_auto_sizing(self, mock_importance_results):
        """Test automatic height based on feature count."""
        # More features should mean taller plot
        fig_small = plot_importance_bar(mock_importance_results, top_n=5)
        fig_large = plot_importance_bar(mock_importance_results, top_n=20)

        assert fig_large.layout.height > fig_small.layout.height

    def test_missing_required_keys(self):
        """Test validation of required keys."""
        invalid_results: dict = {"consensus_ranking": []}  # Missing method_results

        with pytest.raises(ValueError, match="Missing keys"):
            plot_importance_bar(invalid_results)

    def test_invalid_top_n(self, mock_importance_results):
        """Test validation of top_n parameter."""
        with pytest.raises(ValueError):
            plot_importance_bar(mock_importance_results, top_n=0)

        with pytest.raises(ValueError):
            plot_importance_bar(mock_importance_results, top_n=-5)

    def test_invalid_theme(self, mock_importance_results):
        """Test validation of theme parameter."""
        with pytest.raises(ValueError, match="Unknown theme"):
            plot_importance_bar(mock_importance_results, theme="invalid_theme")

    def test_invalid_color_scheme(self, mock_importance_results):
        """Test validation of color_scheme parameter."""
        with pytest.raises(ValueError, match="Unknown color scheme"):
            plot_importance_bar(mock_importance_results, color_scheme="invalid_scheme")

    def test_colorbar_presence(self, mock_importance_results):
        """Test that colorbar is shown."""
        fig = plot_importance_bar(mock_importance_results)

        assert fig.data[0].marker.showscale is True
        assert fig.data[0].marker.colorbar is not None

    def test_hover_template(self, mock_importance_results):
        """Test hover information."""
        fig = plot_importance_bar(mock_importance_results)

        assert fig.data[0].hovertemplate is not None
        assert "%{y}" in fig.data[0].hovertemplate  # Feature name
        assert "%{x" in fig.data[0].hovertemplate  # Importance value


# ===================================================================
# plot_importance_heatmap() tests
# ===================================================================


class TestPlotImportanceHeatmap:
    """Tests for plot_importance_heatmap()."""

    def test_basic_usage(self, mock_importance_results):
        """Test basic heatmap creation."""
        fig = plot_importance_heatmap(mock_importance_results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

        # Check dimensions (3x3 for 3 methods)
        assert fig.data[0].z.shape == (3, 3)

    def test_correlation_matrix_symmetry(self, mock_importance_results):
        """Test that correlation matrix is symmetric."""
        fig = plot_importance_heatmap(mock_importance_results)
        matrix = fig.data[0].z

        assert np.allclose(matrix, matrix.T)

    def test_diagonal_is_one(self, mock_importance_results):
        """Test that diagonal is 1.0 (self-correlation)."""
        fig = plot_importance_heatmap(mock_importance_results)
        matrix = fig.data[0].z

        diagonal = np.diag(matrix)
        assert np.allclose(diagonal, 1.0)

    def test_correlation_values_in_range(self, mock_importance_results):
        """Test that correlations are in [-1, 1]."""
        fig = plot_importance_heatmap(mock_importance_results)
        matrix = fig.data[0].z

        assert np.all(matrix >= -1.0)
        assert np.all(matrix <= 1.0)

    def test_custom_title(self, mock_importance_results):
        """Test custom title."""
        custom_title = "Method Correlations"
        fig = plot_importance_heatmap(mock_importance_results, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_importance_results):
        """Test theme customization."""
        fig = plot_importance_heatmap(mock_importance_results, theme="print")

        # Print theme should have light background (case-insensitive)
        assert fig.layout.paper_bgcolor.lower() in ["#ffffff", "#fff", "white"]

    def test_diverging_colorscale(self, mock_importance_results):
        """Test diverging colorscale centered at 0."""
        fig = plot_importance_heatmap(mock_importance_results)

        assert fig.data[0].zmid == 0
        assert fig.data[0].zmin == -1
        assert fig.data[0].zmax == 1

    def test_show_values(self, mock_importance_results):
        """Test show_values parameter."""
        # With values
        fig = plot_importance_heatmap(mock_importance_results, show_values=True)
        assert fig.data[0].text is not None

        # Without values
        fig = plot_importance_heatmap(mock_importance_results, show_values=False)
        assert fig.data[0].text is None

    def test_custom_dimensions(self, mock_importance_results):
        """Test custom width and height."""
        fig = plot_importance_heatmap(mock_importance_results, width=1000, height=1000)

        assert fig.layout.width == 1000
        assert fig.layout.height == 1000

    def test_two_methods_only(self, mock_importance_results_two_methods):
        """Test with minimum (2) methods."""
        fig = plot_importance_heatmap(mock_importance_results_two_methods)

        assert fig.data[0].z.shape == (2, 2)

    def test_single_method_raises_error(self, mock_importance_results):
        """Test that single method raises error."""
        results = mock_importance_results.copy()
        results["methods_run"] = ["mdi"]
        results["method_agreement"] = {}

        with pytest.raises(ValueError, match="at least 2 methods"):
            plot_importance_heatmap(results)

    def test_missing_required_keys(self):
        """Test validation of required keys."""
        invalid_results = {"methods_run": ["mdi", "pfi"]}  # Missing method_agreement

        with pytest.raises(ValueError, match="Missing keys"):
            plot_importance_heatmap(invalid_results)

    def test_invalid_theme(self, mock_importance_results):
        """Test validation of theme parameter."""
        with pytest.raises(ValueError, match="Unknown theme"):
            plot_importance_heatmap(mock_importance_results, theme="invalid")

    def test_colorbar_presence(self, mock_importance_results):
        """Test that colorbar is shown."""
        fig = plot_importance_heatmap(mock_importance_results)

        assert fig.data[0].colorbar is not None
        assert fig.data[0].colorbar.title.text == "Correlation"


# ===================================================================
# plot_importance_distribution() tests
# ===================================================================


class TestPlotImportanceDistribution:
    """Tests for plot_importance_distribution()."""

    def test_basic_usage_subplots(self, mock_importance_results):
        """Test basic distribution with subplots."""
        fig = plot_importance_distribution(mock_importance_results)

        assert isinstance(fig, go.Figure)
        # 3 methods = 3 histogram traces
        assert len(fig.data) == 3

    def test_overlay_mode(self, mock_importance_results):
        """Test overlay mode."""
        fig = plot_importance_distribution(mock_importance_results, overlay=True)

        # Should have 3 traces (one per method)
        assert len(fig.data) == 3

        # Check opacity is set for overlay
        for trace in fig.data:
            assert trace.opacity == 0.7

    def test_single_method(self, mock_importance_results):
        """Test plotting single method."""
        fig = plot_importance_distribution(mock_importance_results, method="mdi")

        assert len(fig.data) == 1
        assert "MDI" in fig.data[0].name

    def test_invalid_method(self, mock_importance_results):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Method 'invalid' not found"):
            plot_importance_distribution(mock_importance_results, method="invalid")

    def test_custom_bins(self, mock_importance_results):
        """Test custom number of bins."""
        fig = plot_importance_distribution(mock_importance_results, bins=50)

        assert fig.data[0].nbinsx == 50

    def test_invalid_bins(self, mock_importance_results):
        """Test that invalid bins raises error."""
        with pytest.raises(ValueError):
            plot_importance_distribution(mock_importance_results, bins=0)

        with pytest.raises(ValueError):
            plot_importance_distribution(mock_importance_results, bins=-10)

    def test_custom_title(self, mock_importance_results):
        """Test custom title."""
        custom_title = "Score Distribution"
        fig = plot_importance_distribution(mock_importance_results, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_importance_results):
        """Test theme customization."""
        fig = plot_importance_distribution(mock_importance_results, theme="dark")

        # Dark theme should have dark background (case-insensitive)
        assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d", "#000000"]

    def test_custom_dimensions(self, mock_importance_results):
        """Test custom width and height."""
        fig = plot_importance_distribution(mock_importance_results, width=1200, height=800)

        assert fig.layout.width == 1200
        assert fig.layout.height == 800

    def test_height_auto_sizing_subplots(self, mock_importance_results):
        """Test automatic height for subplots."""
        fig_subplots = plot_importance_distribution(mock_importance_results)
        fig_overlay = plot_importance_distribution(mock_importance_results, overlay=True)

        # Subplots should be taller (3 methods * 400px each)
        assert fig_subplots.layout.height > fig_overlay.layout.height

    def test_missing_required_keys(self):
        """Test validation of required keys."""
        invalid_results: dict = {"method_results": {}}  # Missing methods_run

        with pytest.raises(ValueError, match="Missing keys"):
            plot_importance_distribution(invalid_results)

    def test_hover_template(self, mock_importance_results):
        """Test hover information."""
        fig = plot_importance_distribution(mock_importance_results)

        for trace in fig.data:
            assert trace.hovertemplate is not None
            assert "Importance:" in trace.hovertemplate
            assert "Count:" in trace.hovertemplate


# ===================================================================
# plot_importance_summary() tests
# ===================================================================


class TestPlotImportanceSummary:
    """Tests for plot_importance_summary()."""

    def test_basic_usage(self, mock_importance_results):
        """Test basic summary plot creation."""
        fig = plot_importance_summary(mock_importance_results)

        assert isinstance(fig, go.Figure)
        # Should have multiple traces: bar + heatmap + histograms
        assert len(fig.data) >= 4  # At least bar, heatmap, and 2+ histograms

    def test_has_all_components(self, mock_importance_results):
        """Test that all expected plot types are present."""
        fig = plot_importance_summary(mock_importance_results)

        # Check for different trace types
        trace_types = [trace.type for trace in fig.data]
        assert "bar" in trace_types
        assert "heatmap" in trace_types
        assert "histogram" in trace_types

    def test_custom_title(self, mock_importance_results):
        """Test custom title."""
        custom_title = "Comprehensive Feature Analysis"
        fig = plot_importance_summary(mock_importance_results, title=custom_title)

        assert custom_title in fig.layout.title.text

    def test_top_n_parameter(self, mock_importance_results):
        """Test top_n parameter for bar chart."""
        fig = plot_importance_summary(mock_importance_results, top_n=10)

        # Bar chart should have 10 features
        bar_trace = [t for t in fig.data if t.type == "bar"][0]
        assert len(bar_trace.y) == 10

    def test_invalid_top_n(self, mock_importance_results):
        """Test that invalid top_n raises error."""
        with pytest.raises(ValueError):
            plot_importance_summary(mock_importance_results, top_n=0)

    def test_theme_parameter(self, mock_importance_results):
        """Test theme customization."""
        fig = plot_importance_summary(mock_importance_results, theme="presentation")

        # Should have presentation theme colors
        assert fig.layout.paper_bgcolor is not None

    def test_custom_dimensions(self, mock_importance_results):
        """Test custom width and height."""
        fig = plot_importance_summary(mock_importance_results, width=1600, height=1200)

        assert fig.layout.width == 1600
        assert fig.layout.height == 1200

    def test_subplot_layout(self, mock_importance_results):
        """Test that subplots are properly arranged."""
        fig = plot_importance_summary(mock_importance_results)

        # Should have subplot annotations
        assert len(fig.layout.annotations) >= 3  # At least 3 subplot titles

    def test_missing_required_keys(self):
        """Test validation of required keys."""
        invalid_results = {  # type: ignore[var-annotated]
            "consensus_ranking": [],
            "method_results": {},
        }  # Missing other keys

        with pytest.raises(ValueError, match="Missing keys"):
            plot_importance_summary(invalid_results)

    def test_invalid_theme(self, mock_importance_results):
        """Test validation of theme parameter."""
        with pytest.raises(ValueError, match="Unknown theme"):
            plot_importance_summary(mock_importance_results, theme="invalid")

    def test_legend_present(self, mock_importance_results):
        """Test that legend is shown for histograms."""
        fig = plot_importance_summary(mock_importance_results)

        assert fig.layout.showlegend is True

    def test_histogram_overlay_mode(self, mock_importance_results):
        """Test that histograms use overlay mode."""
        fig = plot_importance_summary(mock_importance_results)

        assert fig.layout.barmode == "overlay"

    def test_heatmap_colorbar_position(self, mock_importance_results):
        """Test that heatmap colorbar doesn't overlap other plots."""
        fig = plot_importance_summary(mock_importance_results)

        # Find heatmap trace
        heatmap = [t for t in fig.data if t.type == "heatmap"][0]

        # Colorbar should be positioned to the right
        assert heatmap.colorbar.x > 1.0

    def test_two_methods_only(self, mock_importance_results_two_methods):
        """Test summary with minimum (2) methods."""
        fig = plot_importance_summary(mock_importance_results_two_methods)

        # Should still work with 2 methods
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Bar + heatmap + 2 histograms


# ===================================================================
# Integration tests
# ===================================================================


class TestFeaturePlotsIntegration:
    """Integration tests using all plot functions together."""

    def test_consistent_theming_across_plots(self, mock_importance_results):
        """Test that all plots respect theme setting."""
        theme = "dark"

        fig_bar = plot_importance_bar(mock_importance_results, theme=theme)
        fig_heatmap = plot_importance_heatmap(mock_importance_results, theme=theme)
        fig_dist = plot_importance_distribution(mock_importance_results, theme=theme)
        fig_summary = plot_importance_summary(mock_importance_results, theme=theme)

        # All should have dark background (case-insensitive)
        for fig in [fig_bar, fig_heatmap, fig_dist, fig_summary]:
            assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d", "#000000"]

    def test_all_plots_return_valid_figures(self, mock_importance_results):
        """Test that all plots return valid Plotly figures."""
        figs = [
            plot_importance_bar(mock_importance_results),
            plot_importance_heatmap(mock_importance_results),
            plot_importance_distribution(mock_importance_results),
            plot_importance_summary(mock_importance_results),
        ]

        for fig in figs:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
            assert fig.layout.title is not None

    def test_minimal_results(self, mock_importance_results_two_methods):
        """Test all plots work with minimal valid results."""
        # All plots should work with just 2 methods
        fig_bar = plot_importance_bar(mock_importance_results_two_methods)
        fig_heatmap = plot_importance_heatmap(mock_importance_results_two_methods)
        fig_dist = plot_importance_distribution(mock_importance_results_two_methods)
        fig_summary = plot_importance_summary(mock_importance_results_two_methods)

        for fig in [fig_bar, fig_heatmap, fig_dist, fig_summary]:
            assert isinstance(fig, go.Figure)
