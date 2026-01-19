"""Tests for ml4t-diagnostic.visualization.core module."""

import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.visualization.core import (
    AVAILABLE_THEMES,
    add_annotation,
    apply_responsive_layout,
    # Layout helpers
    create_base_figure,
    # Format helpers
    format_hover_template,
    format_number,
    format_percentage,
    # Color schemes
    get_color_scheme,
    get_colorscale,
    get_plot_theme,
    get_theme_config,
    # Theme management
    set_plot_theme,
    validate_color_scheme,
    # Validation
    validate_plot_results,
    validate_positive_int,
    validate_theme,
)

# =============================================================================
# Theme Management Tests
# =============================================================================


class TestThemeManagement:
    """Tests for theme management functions."""

    def test_default_theme(self):
        """Test default theme is 'default'."""
        assert get_plot_theme() == "default"

    def test_set_theme_valid(self):
        """Test setting valid themes."""
        for theme_name in AVAILABLE_THEMES.keys():
            set_plot_theme(theme_name)
            assert get_plot_theme() == theme_name

    def test_set_theme_invalid(self):
        """Test setting invalid theme raises error."""
        with pytest.raises(ValueError, match="Unknown theme"):
            set_plot_theme("nonexistent")

    def test_get_theme_config_default(self):
        """Test getting theme config for default theme."""
        config = get_theme_config("default")

        assert "layout" in config
        assert "colorway" in config
        assert "color_schemes" in config
        assert "defaults" in config

        # Check layout has required keys
        assert "paper_bgcolor" in config["layout"]
        assert "plot_bgcolor" in config["layout"]
        assert "font" in config["layout"]

    def test_get_theme_config_all_themes(self):
        """Test all themes have valid configurations."""
        for theme_name in AVAILABLE_THEMES.keys():
            config = get_theme_config(theme_name)

            # All themes must have these keys
            assert "name" in config
            assert "description" in config
            assert "layout" in config
            assert "colorway" in config
            assert "defaults" in config

    def test_get_theme_config_none_uses_global(self):
        """Test that None uses global theme."""
        set_plot_theme("dark")
        config = get_theme_config(None)
        assert config["name"] == "dark"

    def test_theme_colors_are_hex(self):
        """Test that all theme colors are valid hex codes."""
        for theme_name in AVAILABLE_THEMES.keys():
            config = get_theme_config(theme_name)

            # Check colorway
            for color in config["colorway"]:
                assert color.startswith("#")
                assert len(color) == 7  # #RRGGBB

    def test_theme_defaults_are_positive(self):
        """Test that all theme defaults are positive integers."""
        for theme_name in AVAILABLE_THEMES.keys():
            config = get_theme_config(theme_name)
            defaults = config["defaults"]

            assert defaults["bar_height"] > 0
            assert defaults["heatmap_height"] > 0
            assert defaults["scatter_height"] > 0
            assert defaults["line_height"] > 0
            assert defaults["width"] > 0


# =============================================================================
# Color Scheme Tests
# =============================================================================


class TestColorSchemes:
    """Tests for color scheme functions."""

    def test_get_color_scheme_valid(self):
        """Test getting valid color schemes."""
        schemes = ["blues", "viridis", "rdbu", "colorblind_safe"]

        for scheme in schemes:
            colors = get_color_scheme(scheme)
            assert isinstance(colors, list)
            assert len(colors) > 0

    def test_get_color_scheme_invalid(self):
        """Test getting invalid scheme raises error."""
        with pytest.raises(ValueError, match="Unknown color scheme"):
            get_color_scheme("nonexistent")

    def test_get_color_scheme_case_insensitive(self):
        """Test color scheme names are case insensitive."""
        colors1 = get_color_scheme("BLUES")
        colors2 = get_color_scheme("blues")
        colors3 = get_color_scheme("Blues")

        assert colors1 == colors2 == colors3

    def test_get_colorscale_continuous(self):
        """Test getting continuous colorscale."""
        scale = get_colorscale("viridis", n_colors=None)
        assert isinstance(scale, list)
        assert len(scale) > 0

    def test_get_colorscale_discrete(self):
        """Test getting discrete colors from colorscale."""
        n = 5
        colors = get_colorscale("viridis", n_colors=n)

        assert isinstance(colors, list)
        assert len(colors) == n

    def test_get_colorscale_reverse(self):
        """Test reversing colorscale."""
        colors_normal = get_colorscale("viridis", n_colors=5)
        colors_reversed = get_colorscale("viridis", n_colors=5, reverse=True)

        assert colors_normal[0] == colors_reversed[-1]
        assert colors_normal[-1] == colors_reversed[0]

    def test_colorblind_safe_scheme(self):
        """Test colorblind-safe scheme is available."""
        colors = get_color_scheme("colorblind_safe")
        assert len(colors) == 8  # Standard palette size

    def test_financial_schemes(self):
        """Test financial color schemes."""
        gains_losses = get_color_scheme("gains_losses")
        assert len(gains_losses) == 3  # Red, gray, green

        quantiles = get_color_scheme("quantiles")
        assert len(quantiles) == 5  # 5 quantiles


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for validation helper functions."""

    def test_validate_plot_results_valid(self):
        """Test validating valid results dict."""
        results = {
            "consensus_ranking": ["feat1", "feat2"],
            "method_results": {"mdi": {}, "pfi": {}},
        }

        # Should not raise
        validate_plot_results(
            results,
            required_keys=["consensus_ranking", "method_results"],
            function_name="test_function",
        )

    def test_validate_plot_results_not_dict(self):
        """Test validation fails for non-dict."""
        with pytest.raises(TypeError, match="requires dict"):
            validate_plot_results(
                "not a dict", required_keys=["key"], function_name="test_function"
            )

    def test_validate_plot_results_missing_keys(self):
        """Test validation fails for missing keys."""
        results = {"key1": "value1"}

        with pytest.raises(ValueError, match="Missing keys"):
            validate_plot_results(
                results, required_keys=["key1", "key2", "key3"], function_name="test_function"
            )

    def test_validate_positive_int_valid(self):
        """Test validating positive integers."""
        # Should not raise
        validate_positive_int(1, "param")
        validate_positive_int(100, "param")
        validate_positive_int(None, "param")  # None is allowed

    def test_validate_positive_int_invalid(self):
        """Test validation fails for non-positive integers."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_positive_int(0, "param")

        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_positive_int(-5, "param")

        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_positive_int(3.14, "param")

    def test_validate_theme_valid(self):
        """Test validating valid themes."""
        for theme_name in AVAILABLE_THEMES.keys():
            validated = validate_theme(theme_name)
            assert validated == theme_name

    def test_validate_theme_none_uses_global(self):
        """Test None returns global theme."""
        set_plot_theme("dark")
        validated = validate_theme(None)
        assert validated == "dark"

    def test_validate_theme_invalid(self):
        """Test validation fails for invalid theme."""
        with pytest.raises(ValueError, match="Unknown theme"):
            validate_theme("nonexistent")

    def test_validate_color_scheme_valid(self):
        """Test validating valid color schemes."""
        scheme = validate_color_scheme("viridis", "default")
        assert scheme == "viridis"

    def test_validate_color_scheme_none_uses_theme_default(self):
        """Test None uses theme's default color scheme."""
        scheme = validate_color_scheme(None, "default")
        assert scheme in ["blues", "rdbu", "set2"]  # One of the defaults

    def test_validate_color_scheme_invalid(self):
        """Test validation fails for invalid scheme."""
        with pytest.raises(ValueError, match="Unknown color scheme"):
            validate_color_scheme("nonexistent", "default")


# =============================================================================
# Layout Helper Tests
# =============================================================================


class TestLayoutHelpers:
    """Tests for layout helper functions."""

    def test_create_base_figure(self):
        """Test creating base figure."""
        fig = create_base_figure(
            title="Test Title", xaxis_title="X Axis", yaxis_title="Y Axis", theme="default"
        )

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Title"
        assert fig.layout.xaxis.title.text == "X Axis"
        assert fig.layout.yaxis.title.text == "Y Axis"

    def test_create_base_figure_applies_theme(self):
        """Test base figure applies theme correctly."""
        fig = create_base_figure(theme="dark")

        # Dark theme should have dark background (case-insensitive)
        assert fig.layout.paper_bgcolor.lower() == "#1e1e1e"
        assert fig.layout.plot_bgcolor.lower() == "#2d2d2d"

    def test_create_base_figure_custom_dimensions(self):
        """Test custom width and height."""
        width, height = 1200, 800
        fig = create_base_figure(width=width, height=height)

        assert fig.layout.width == width
        assert fig.layout.height == height

    def test_create_base_figure_custom_margin(self):
        """Test custom margin."""
        margin = {"l": 100, "r": 50, "t": 150, "b": 100}
        fig = create_base_figure(margin=margin)

        assert fig.layout.margin.l == 100
        assert fig.layout.margin.r == 50
        assert fig.layout.margin.t == 150
        assert fig.layout.margin.b == 100

    def test_apply_responsive_layout(self):
        """Test making figure responsive."""
        fig = create_base_figure(title="Test")
        fig = apply_responsive_layout(fig)

        assert fig.layout.autosize is True

    def test_add_annotation(self):
        """Test adding annotation to figure."""
        fig = create_base_figure(title="Test")
        fig = add_annotation(fig, text="Test annotation", x=0.5, y=0.95)

        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "Test annotation"
        assert fig.layout.annotations[0].x == 0.5
        assert fig.layout.annotations[0].y == 0.95


# =============================================================================
# Format Helper Tests
# =============================================================================


class TestFormatHelpers:
    """Tests for format helper functions."""

    def test_format_hover_template(self):
        """Test creating hover template."""
        template = format_hover_template(x_label="Feature", y_label="Importance", y_format=".4f")

        assert "Feature" not in template  # x uses %{x}
        assert "Importance" in template
        assert ".4f" in template
        assert "<extra></extra>" in template

    def test_format_number_integer(self):
        """Test formatting integers."""
        assert format_number(1234567, precision=0) == "1,234,567"
        assert format_number(42, precision=0) == "42"

    def test_format_number_float(self):
        """Test formatting floats."""
        assert format_number(3.14159, precision=2) == "3.14"
        assert format_number(0.123456, precision=4) == "0.1235"

    def test_format_percentage(self):
        """Test formatting percentages."""
        assert format_percentage(0.05, precision=1) == "5.0%"
        assert format_percentage(0.12345, precision=2) == "12.35%"
        assert format_percentage(1.0, precision=0) == "100%"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_theme_workflow(self):
        """Test complete theme workflow."""
        # Set theme
        set_plot_theme("dark")

        # Create figure with theme
        fig = create_base_figure(title="Test", theme=None)  # Should use global

        # Verify theme applied (case-insensitive)
        assert fig.layout.paper_bgcolor.lower() == "#1e1e1e"

        # Reset
        set_plot_theme("default")

    def test_color_scheme_workflow(self):
        """Test complete color scheme workflow."""
        # Get scheme
        scheme_name = "viridis"
        colors = get_colorscale(scheme_name, n_colors=5)

        # Validate
        assert len(colors) == 5

        # Use in figure (conceptually)
        create_base_figure(title="Test")
        # Would add traces with these colors

    def test_validation_workflow(self):
        """Test validation in typical plot function."""
        # Simulate plot function
        results = {"consensus_ranking": ["f1", "f2"], "method_results": {}}

        # Validate results
        validate_plot_results(
            results, required_keys=["consensus_ranking"], function_name="test_plot"
        )

        # Validate parameters
        top_n = 20
        validate_positive_int(top_n, "top_n")

        # Validate theme
        theme = validate_theme("dark")
        assert theme == "dark"

        # All validations passed
        assert True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_results_dict(self):
        """Test validation with empty dict."""
        with pytest.raises(ValueError):
            validate_plot_results({}, required_keys=["some_key"], function_name="test")

    def test_colorscale_single_color(self):
        """Test getting single color from scheme."""
        colors = get_colorscale("viridis", n_colors=1)
        assert len(colors) == 1

    def test_colorscale_many_colors(self):
        """Test getting many colors (more than scheme length)."""
        colors = get_colorscale("colorblind_safe", n_colors=20)
        assert len(colors) == 20

    def test_theme_with_none_values(self):
        """Test creating figure with all None optional params."""
        fig = create_base_figure(
            title=None,
            xaxis_title=None,
            yaxis_title=None,
            width=None,
            height=None,
            theme=None,
            margin=None,
        )

        assert isinstance(fig, go.Figure)

    def test_annotation_without_arrow(self):
        """Test annotation without arrow."""
        fig = create_base_figure(title="Test")
        fig = add_annotation(fig, text="No arrow", x=0.5, y=0.5, showarrow=False)

        assert fig.layout.annotations[0].showarrow is False
