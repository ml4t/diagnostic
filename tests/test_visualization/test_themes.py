"""Tests for evaluation/themes.py.

This module tests theming and styling functions for Plotly visualizations.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from ml4t.diagnostic.evaluation.themes import (
    COLORBLIND_SAFE,
    DARK_TEMPLATE,
    DEFAULT_TEMPLATE,
    FINANCIAL_COLORS,
    PRINT_TEMPLATE,
    add_pattern_overlay,
    apply_theme,
    format_currency,
    format_percentage,
    get_color_scale,
)


class TestColorDictionaries:
    """Tests for color constant dictionaries."""

    def test_financial_colors_completeness(self):
        """Test that FINANCIAL_COLORS has all expected keys."""
        required_keys = [
            "positive",
            "negative",
            "neutral",
            "primary",
            "secondary",
            "tertiary",
            "quaternary",
            "background",
            "paper",
            "grid",
            "text",
            "subtitle",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
        ]
        for key in required_keys:
            assert key in FINANCIAL_COLORS, f"Missing key: {key}"

    def test_financial_colors_hex_format(self):
        """Test that all FINANCIAL_COLORS values are valid hex colors."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for key, value in FINANCIAL_COLORS.items():
            assert hex_pattern.match(value), f"{key} has invalid hex: {value}"

    def test_colorblind_safe_completeness(self):
        """Test that COLORBLIND_SAFE has 8 colors."""
        assert len(COLORBLIND_SAFE) == 8

    def test_colorblind_safe_hex_format(self):
        """Test that all COLORBLIND_SAFE values are valid hex colors."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for key, value in COLORBLIND_SAFE.items():
            assert hex_pattern.match(value), f"{key} has invalid hex: {value}"

    def test_colorblind_safe_keys(self):
        """Test that COLORBLIND_SAFE has expected keys."""
        expected_keys = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
        for key in expected_keys:
            assert key in COLORBLIND_SAFE


class TestTemplates:
    """Tests for layout template dictionaries."""

    def test_default_template_structure(self):
        """Test that DEFAULT_TEMPLATE has required layout keys."""
        assert "layout" in DEFAULT_TEMPLATE
        layout = DEFAULT_TEMPLATE["layout"]
        assert "font" in layout
        assert "title" in layout
        assert "plot_bgcolor" in layout
        assert "paper_bgcolor" in layout
        assert "xaxis" in layout
        assert "yaxis" in layout
        assert "legend" in layout

    def test_dark_template_structure(self):
        """Test that DARK_TEMPLATE has required layout keys."""
        assert "layout" in DARK_TEMPLATE
        layout = DARK_TEMPLATE["layout"]
        assert "font" in layout
        assert "plot_bgcolor" in layout

    def test_print_template_structure(self):
        """Test that PRINT_TEMPLATE has required layout keys."""
        assert "layout" in PRINT_TEMPLATE
        layout = PRINT_TEMPLATE["layout"]
        assert "font" in layout
        assert "plot_bgcolor" in layout

    def test_default_template_font_settings(self):
        """Test that DEFAULT_TEMPLATE has font settings."""
        font = DEFAULT_TEMPLATE["layout"]["font"]
        assert "family" in font
        assert "size" in font
        assert "color" in font


class TestGetColorScale:
    """Tests for get_color_scale function."""

    def test_diverging_2_colors(self):
        """Test diverging scheme with 2 colors."""
        colors = get_color_scale(2, scheme="diverging")
        assert len(colors) == 2
        assert colors[0] == FINANCIAL_COLORS["negative"]
        assert colors[1] == FINANCIAL_COLORS["positive"]

    def test_diverging_3_colors(self):
        """Test diverging scheme with 3 colors."""
        colors = get_color_scale(3, scheme="diverging")
        assert len(colors) == 3
        assert colors[0] == FINANCIAL_COLORS["negative"]
        assert colors[1] == FINANCIAL_COLORS["neutral"]
        assert colors[2] == FINANCIAL_COLORS["positive"]

    def test_diverging_many_colors(self):
        """Test diverging scheme with many colors (uses plotly)."""
        colors = get_color_scale(7, scheme="diverging")
        assert len(colors) == 7

    def test_sequential_few_colors(self):
        """Test sequential scheme with few colors."""
        colors = get_color_scale(3, scheme="sequential")
        assert len(colors) == 3

    def test_sequential_5_colors(self):
        """Test sequential scheme with 5 colors."""
        colors = get_color_scale(5, scheme="sequential")
        assert len(colors) == 5
        # First should be lightest blue
        assert colors[0] == "#E3F2FD"

    def test_sequential_many_colors(self):
        """Test sequential scheme with many colors (uses plotly)."""
        colors = get_color_scale(10, scheme="sequential")
        assert len(colors) == 10

    def test_quantile_few_colors(self):
        """Test quantile scheme with few colors."""
        colors = get_color_scale(3, scheme="quantile")
        assert len(colors) == 3
        assert colors[0] == FINANCIAL_COLORS["q1"]

    def test_quantile_5_colors(self):
        """Test quantile scheme with 5 colors."""
        colors = get_color_scale(5, scheme="quantile")
        assert len(colors) == 5
        assert colors[0] == FINANCIAL_COLORS["q1"]
        assert colors[4] == FINANCIAL_COLORS["q5"]

    def test_quantile_many_colors(self):
        """Test quantile scheme with many colors (uses plotly)."""
        colors = get_color_scale(10, scheme="quantile")
        assert len(colors) == 10

    def test_colorblind_few_colors(self):
        """Test colorblind scheme with few colors."""
        colors = get_color_scale(4, scheme="colorblind")
        assert len(colors) == 4

    def test_colorblind_8_colors(self):
        """Test colorblind scheme with exactly 8 colors."""
        colors = get_color_scale(8, scheme="colorblind")
        assert len(colors) == 8
        # Should match COLORBLIND_SAFE values
        assert set(colors) == set(COLORBLIND_SAFE.values())

    def test_colorblind_many_colors_cycles(self):
        """Test colorblind scheme cycles when more than 8 colors requested."""
        colors = get_color_scale(12, scheme="colorblind")
        assert len(colors) == 12
        # Should cycle through the 8 colors

    def test_default_scheme_fallback(self):
        """Test that unknown scheme falls back to default (Set3)."""
        colors = get_color_scale(5, scheme="unknown_scheme")
        assert len(colors) == 5


class TestApplyTheme:
    """Tests for apply_theme function."""

    def test_apply_default_theme(self):
        """Test applying default theme."""
        fig = MagicMock()
        result = apply_theme(fig, theme="default")

        fig.update_layout.assert_called_once()
        assert result == fig

    def test_apply_dark_theme(self):
        """Test applying dark theme."""
        fig = MagicMock()
        result = apply_theme(fig, theme="dark")

        fig.update_layout.assert_called_once()
        assert result == fig

    def test_apply_print_theme(self):
        """Test applying print theme."""
        fig = MagicMock()
        result = apply_theme(fig, theme="print")

        fig.update_layout.assert_called_once()
        assert result == fig

    def test_apply_colorblind_theme(self):
        """Test applying colorblind theme."""
        fig = MagicMock()
        result = apply_theme(fig, theme="colorblind")

        fig.update_layout.assert_called_once()
        assert result == fig

    def test_invalid_theme_raises(self):
        """Test that invalid theme raises ValueError."""
        fig = MagicMock()

        with pytest.raises(ValueError, match="Unknown theme"):
            apply_theme(fig, theme="invalid_theme")

    def test_returns_same_figure(self):
        """Test that apply_theme returns the same figure object."""
        fig = MagicMock()
        result = apply_theme(fig)

        assert result is fig


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_basic_percentage(self):
        """Test basic percentage formatting."""
        assert format_percentage(0.05) == "5.0%"

    def test_larger_percentage(self):
        """Test larger percentage."""
        assert format_percentage(0.1234) == "12.3%"

    def test_custom_decimals(self):
        """Test custom decimal places."""
        assert format_percentage(0.12345, decimals=2) == "12.35%"
        assert format_percentage(0.12345, decimals=0) == "12%"

    def test_zero_percentage(self):
        """Test zero value."""
        assert format_percentage(0.0) == "0.0%"

    def test_100_percent(self):
        """Test 100%."""
        assert format_percentage(1.0) == "100.0%"

    def test_negative_percentage(self):
        """Test negative percentage."""
        assert format_percentage(-0.05) == "-5.0%"

    def test_very_small_percentage(self):
        """Test very small percentage."""
        result = format_percentage(0.0001, decimals=2)
        assert result == "0.01%"


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_basic_currency(self):
        """Test basic currency formatting."""
        assert format_currency(1234) == "$1,234"

    def test_with_decimals(self):
        """Test currency with decimal places."""
        assert format_currency(1234.5678, decimals=2) == "$1,234.57"

    def test_large_number(self):
        """Test large number with thousands separator."""
        assert format_currency(1234567) == "$1,234,567"

    def test_zero_value(self):
        """Test zero value."""
        assert format_currency(0) == "$0"

    def test_negative_value(self):
        """Test negative value."""
        assert format_currency(-1234) == "$-1,234"

    def test_custom_currency_symbol(self):
        """Test custom currency symbol."""
        assert format_currency(1234, currency="€") == "€1,234"
        assert format_currency(1234, currency="£") == "£1,234"

    def test_small_value(self):
        """Test small value with decimals."""
        assert format_currency(0.5, decimals=2) == "$0.50"


class TestAddPatternOverlay:
    """Tests for add_pattern_overlay function (currently stub)."""

    def test_returns_figure_unchanged(self):
        """Test that stub returns input figure unchanged."""
        fig = MagicMock()
        result = add_pattern_overlay(fig, 0, "diagonal")

        assert result is fig
