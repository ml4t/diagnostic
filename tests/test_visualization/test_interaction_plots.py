"""Tests for feature interaction visualization functions."""

import numpy as np
import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.visualization.interaction_plots import (
    plot_interaction_bar,
    plot_interaction_heatmap,
    plot_interaction_network,
)


@pytest.fixture
def mock_shap_interactions():
    """Create mock results from compute_shap_interactions()."""
    np.random.seed(42)

    n_features = 10
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create symmetric interaction matrix
    matrix = np.random.exponential(scale=0.05, size=(n_features, n_features))
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    # Make diagonal larger (main effects)
    np.fill_diagonal(matrix, matrix.diagonal() * 2)

    # Extract top interactions
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((feature_names[i], feature_names[j], matrix[i, j]))

    # Sort by strength
    interactions.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "interaction_matrix": matrix,
        "feature_names": feature_names,
        "top_interactions": interactions[:20],  # Top 20
        "n_features": n_features,
        "n_samples_used": 100,
        "computation_time": 2.5,
    }


@pytest.fixture
def mock_analyze_interactions():
    """Create mock results from analyze_interactions()."""
    np.random.seed(42)

    feature_pairs = [
        ("momentum", "volatility"),
        ("size", "value"),
        ("momentum", "size"),
        ("volatility", "value"),
        ("momentum", "value"),
    ]

    # Create consensus ranking with scores dict
    consensus_ranking = [
        (
            pair[0],
            pair[1],
            np.random.exponential(0.1),
            {
                "conditional_ic": np.random.uniform(0, 1),
                "h_statistic": np.random.uniform(0, 1),
                "shap": np.random.uniform(0, 0.5),
            },
        )
        for pair in feature_pairs
    ]

    # Sort by average score
    consensus_ranking.sort(key=lambda x: x[2], reverse=True)

    return {
        "consensus_ranking": consensus_ranking,
        "method_results": {
            "conditional_ic": {"pairs": feature_pairs},
            "h_statistic": {"pairs": feature_pairs},
            "shap": {"pairs": feature_pairs},
        },
        "method_agreement": {
            ("conditional_ic", "h_statistic"): 0.75,
            ("conditional_ic", "shap"): 0.68,
            ("h_statistic", "shap"): 0.82,
        },
        "top_interactions_consensus": list(feature_pairs[:2]),
        "warnings": [],
        "interpretation": "Strong consensus across methods",
        "methods_run": ["conditional_ic", "h_statistic", "shap"],
        "methods_failed": [],
    }


# ===================================================================
# plot_interaction_bar() tests
# ===================================================================


class TestPlotInteractionBar:
    """Tests for plot_interaction_bar()."""

    def test_basic_usage_shap(self, mock_shap_interactions):
        """Test basic bar plot with SHAP results."""
        fig = plot_interaction_bar(mock_shap_interactions)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert fig.data[0].orientation == "h"

    def test_basic_usage_analyze(self, mock_analyze_interactions):
        """Test basic bar plot with analyze_interactions results."""
        fig = plot_interaction_bar(mock_analyze_interactions)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"

    def test_top_n_filtering(self, mock_shap_interactions):
        """Test top_n parameter."""
        fig = plot_interaction_bar(mock_shap_interactions, top_n=10)

        assert len(fig.data[0].y) == 10

    def test_custom_title(self, mock_shap_interactions):
        """Test custom title."""
        custom_title = "Strong Feature Interactions"
        fig = plot_interaction_bar(mock_shap_interactions, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_shap_interactions):
        """Test theme customization."""
        fig = plot_interaction_bar(mock_shap_interactions, theme="dark")

        # Should have dark theme applied
        assert isinstance(fig, go.Figure)

    def test_color_scheme(self, mock_shap_interactions):
        """Test color scheme parameter."""
        fig = plot_interaction_bar(mock_shap_interactions, color_scheme="plasma")

        assert fig.data[0].marker.colorscale is not None

    def test_show_values(self, mock_shap_interactions):
        """Test show_values parameter."""
        # With values
        fig = plot_interaction_bar(mock_shap_interactions, show_values=True)
        assert fig.data[0].text is not None

        # Without values
        fig = plot_interaction_bar(mock_shap_interactions, show_values=False)
        assert fig.data[0].text is None

    def test_custom_dimensions(self, mock_shap_interactions):
        """Test custom width and height."""
        fig = plot_interaction_bar(mock_shap_interactions, width=1200, height=800, top_n=10)

        assert fig.layout.width == 1200
        assert fig.layout.height == 800

    def test_invalid_results(self):
        """Test validation of results structure."""
        invalid_results = {"foo": "bar"}  # Missing required keys

        with pytest.raises(ValueError):
            plot_interaction_bar(invalid_results)

    def test_invalid_top_n(self, mock_shap_interactions):
        """Test validation of top_n parameter."""
        with pytest.raises(ValueError):
            plot_interaction_bar(mock_shap_interactions, top_n=0)

        with pytest.raises(ValueError):
            plot_interaction_bar(mock_shap_interactions, top_n=-5)

    def test_pair_labels_format(self, mock_shap_interactions):
        """Test that pair labels are formatted correctly."""
        fig = plot_interaction_bar(mock_shap_interactions, top_n=5)

        # Check labels contain × symbol
        for label in fig.data[0].y:
            assert " × " in label


# ===================================================================
# plot_interaction_heatmap() tests
# ===================================================================


class TestPlotInteractionHeatmap:
    """Tests for plot_interaction_heatmap()."""

    def test_basic_usage(self, mock_shap_interactions):
        """Test basic heatmap creation."""
        fig = plot_interaction_heatmap(mock_shap_interactions)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_matrix_dimensions(self, mock_shap_interactions):
        """Test that heatmap has correct dimensions."""
        fig = plot_interaction_heatmap(mock_shap_interactions)

        n_features = len(mock_shap_interactions["feature_names"])
        assert fig.data[0].z.shape == (n_features, n_features)

    def test_matrix_symmetry(self, mock_shap_interactions):
        """Test that interaction matrix is symmetric."""
        fig = plot_interaction_heatmap(mock_shap_interactions)
        matrix = fig.data[0].z

        assert np.allclose(matrix, matrix.T)

    def test_custom_title(self, mock_shap_interactions):
        """Test custom title."""
        custom_title = "Feature Interaction Strengths"
        fig = plot_interaction_heatmap(mock_shap_interactions, title=custom_title)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_shap_interactions):
        """Test theme customization."""
        fig = plot_interaction_heatmap(mock_shap_interactions, theme="print")

        assert isinstance(fig, go.Figure)

    def test_show_values(self, mock_shap_interactions):
        """Test show_values parameter."""
        # With values
        fig = plot_interaction_heatmap(mock_shap_interactions, show_values=True)
        assert fig.data[0].text is not None

        # Without values
        fig = plot_interaction_heatmap(mock_shap_interactions, show_values=False)
        assert fig.data[0].text is None

    def test_custom_dimensions(self, mock_shap_interactions):
        """Test custom width and height."""
        fig = plot_interaction_heatmap(mock_shap_interactions, width=1000, height=1000)

        assert fig.layout.width == 1000
        assert fig.layout.height == 1000

    def test_missing_required_keys(self):
        """Test validation of required keys."""
        invalid_results: dict = {"feature_names": []}  # Missing interaction_matrix

        with pytest.raises(ValueError, match="Missing keys"):
            plot_interaction_heatmap(invalid_results)

    def test_colorbar_presence(self, mock_shap_interactions):
        """Test that colorbar is shown."""
        fig = plot_interaction_heatmap(mock_shap_interactions)

        assert fig.data[0].colorbar is not None
        assert fig.data[0].colorbar.title.text == "Strength"

    def test_hover_text_diagonal(self, mock_shap_interactions):
        """Test hover text distinguishes main effects vs interactions."""
        fig = plot_interaction_heatmap(mock_shap_interactions)

        # Check that diagonal hover text mentions "Main Effect"
        hover_text = fig.data[0].hovertext
        assert "Main Effect" in hover_text[0][0]

    def test_hover_text_off_diagonal(self, mock_shap_interactions):
        """Test hover text for interactions."""
        fig = plot_interaction_heatmap(mock_shap_interactions)

        # Check that off-diagonal hover text mentions "Interaction"
        hover_text = fig.data[0].hovertext
        assert "Interaction" in hover_text[0][1]


# ===================================================================
# plot_interaction_network() tests
# ===================================================================


class TestPlotInteractionNetwork:
    """Tests for plot_interaction_network()."""

    def test_basic_usage_shap(self, mock_shap_interactions):
        """Test basic network with SHAP results."""
        fig = plot_interaction_network(mock_shap_interactions, threshold=0.01)

        assert isinstance(fig, go.Figure)
        # Should have edge traces and node trace
        assert len(fig.data) > 1

    def test_basic_usage_matrix(self, mock_shap_interactions):
        """Test network from interaction matrix."""
        # Remove top_interactions to test matrix path
        results = {
            "interaction_matrix": mock_shap_interactions["interaction_matrix"],
            "feature_names": mock_shap_interactions["feature_names"],
        }

        fig = plot_interaction_network(results, threshold=0.05)

        assert isinstance(fig, go.Figure)

    def test_threshold_filtering(self, mock_shap_interactions):
        """Test that threshold filters interactions."""
        # High threshold should result in fewer edges
        fig_high = plot_interaction_network(mock_shap_interactions, threshold=0.1)
        fig_low = plot_interaction_network(mock_shap_interactions, threshold=0.01)

        # High threshold should have fewer traces (fewer edges)
        assert len(fig_high.data) <= len(fig_low.data)

    def test_top_n_filtering(self, mock_shap_interactions):
        """Test top_n parameter."""
        fig = plot_interaction_network(mock_shap_interactions, top_n=5, threshold=0.0)

        # Should have 5 edges + 1 node trace = 6 traces (or more if nodes overlap)
        assert len(fig.data) >= 5

    def test_custom_title(self, mock_shap_interactions):
        """Test custom title."""
        custom_title = "Interaction Network"
        fig = plot_interaction_network(mock_shap_interactions, title=custom_title, threshold=0.01)

        assert fig.layout.title.text == custom_title

    def test_theme_parameter(self, mock_shap_interactions):
        """Test theme customization."""
        fig = plot_interaction_network(mock_shap_interactions, theme="dark", threshold=0.01)

        assert isinstance(fig, go.Figure)

    def test_node_size(self, mock_shap_interactions):
        """Test node_size parameter."""
        node_size = 50
        fig = plot_interaction_network(mock_shap_interactions, node_size=node_size, threshold=0.01)

        # Find node trace (last trace with markers)
        node_trace = [t for t in fig.data if "marker" in t and hasattr(t, "marker")][-1]
        assert node_trace.marker.size == node_size

    def test_edge_labels(self, mock_shap_interactions):
        """Test show_edge_labels parameter."""
        fig = plot_interaction_network(
            mock_shap_interactions, show_edge_labels=True, threshold=0.01, top_n=5
        )

        # Should have annotations for edge labels
        assert len(fig.layout.annotations) > 0

    def test_custom_dimensions(self, mock_shap_interactions):
        """Test custom width and height."""
        fig = plot_interaction_network(
            mock_shap_interactions, width=1200, height=900, threshold=0.01
        )

        assert fig.layout.width == 1200
        assert fig.layout.height == 900

    def test_invalid_results(self):
        """Test validation of results structure."""
        invalid_results = {"foo": "bar"}

        with pytest.raises(ValueError):
            plot_interaction_network(invalid_results)

    def test_invalid_top_n(self, mock_shap_interactions):
        """Test validation of top_n parameter."""
        with pytest.raises(ValueError):
            plot_interaction_network(mock_shap_interactions, top_n=0)

    def test_no_interactions_error(self, mock_shap_interactions):
        """Test error when threshold too high."""
        with pytest.raises(ValueError, match="No interactions above threshold"):
            plot_interaction_network(mock_shap_interactions, threshold=99.9)

    def test_adaptive_threshold(self, mock_shap_interactions):
        """Test that adaptive threshold works when threshold=None."""
        # Should not raise error with None threshold
        fig = plot_interaction_network(mock_shap_interactions, threshold=None)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_isolated_nodes_excluded(self, mock_shap_interactions):
        """Test that isolated nodes are not shown."""
        # Use very high threshold to get few interactions
        fig = plot_interaction_network(mock_shap_interactions, threshold=0.08, top_n=3)

        # Find node trace
        node_trace = [t for t in fig.data if "marker" in t and hasattr(t, "marker")][-1]

        # Number of nodes should be <= total features (some excluded)
        n_nodes_shown = len(node_trace.x)
        n_total_features = len(mock_shap_interactions["feature_names"])

        assert n_nodes_shown <= n_total_features


# ===================================================================
# Integration tests
# ===================================================================


class TestInteractionPlotsIntegration:
    """Integration tests using all plot functions together."""

    def test_all_plots_work_with_shap_results(self, mock_shap_interactions):
        """Test that all plots accept SHAP results."""
        fig_bar = plot_interaction_bar(mock_shap_interactions)
        fig_heatmap = plot_interaction_heatmap(mock_shap_interactions)
        fig_network = plot_interaction_network(mock_shap_interactions, threshold=0.01)

        for fig in [fig_bar, fig_heatmap, fig_network]:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_all_plots_work_with_analyze_results(self, mock_analyze_interactions):
        """Test that bar plot works with analyze_interactions results."""
        # Only bar plot supports analyze_interactions results
        fig_bar = plot_interaction_bar(mock_analyze_interactions)

        assert isinstance(fig_bar, go.Figure)
        assert len(fig_bar.data) > 0

    def test_consistent_theming_across_plots(self, mock_shap_interactions):
        """Test that all plots respect theme setting."""
        theme = "dark"

        fig_bar = plot_interaction_bar(mock_shap_interactions, theme=theme)
        fig_heatmap = plot_interaction_heatmap(mock_shap_interactions, theme=theme)
        fig_network = plot_interaction_network(mock_shap_interactions, theme=theme, threshold=0.01)

        # All should return valid figures
        for fig in [fig_bar, fig_heatmap, fig_network]:
            assert isinstance(fig, go.Figure)

    def test_complementary_views(self, mock_shap_interactions):
        """Test that different plots provide complementary information."""
        # Bar chart for top interactions
        fig_bar = plot_interaction_bar(mock_shap_interactions, top_n=10)

        # Heatmap for full matrix view
        fig_heatmap = plot_interaction_heatmap(mock_shap_interactions)

        # Network for relationship structure
        fig_network = plot_interaction_network(mock_shap_interactions, threshold=0.02)

        # All should be different visualizations
        assert fig_bar.data[0].type == "bar"
        assert fig_heatmap.data[0].type == "heatmap"
        # Network has scatter traces
        assert any(t.type == "scatter" for t in fig_network.data)
