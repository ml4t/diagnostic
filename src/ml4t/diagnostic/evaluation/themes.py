"""Plotly themes and styling for ml4t-diagnostic visualizations.

This module provides consistent theming across all ml4t-diagnostic visualizations,
including color schemes, layout templates, and accessibility options.
"""

from typing import Any

# Financial color schemes
FINANCIAL_COLORS = {
    # Returns and performance
    "positive": "#00CC88",  # Green for gains
    "negative": "#FF4444",  # Red for losses
    "neutral": "#888888",  # Gray for neutral
    # Data series
    "primary": "#3366CC",  # Blue
    "secondary": "#FF9900",  # Orange
    "tertiary": "#109618",  # Dark green
    "quaternary": "#990099",  # Purple
    # UI elements
    "background": "#F8F9FA",  # Light gray
    "paper": "#FFFFFF",  # White
    "grid": "#E0E0E0",  # Grid lines
    "text": "#333333",  # Dark gray text
    "subtitle": "#666666",  # Medium gray
    # Quantiles (5-level)
    "q1": "#D32F2F",  # Dark red (lowest)
    "q2": "#F57C00",  # Orange
    "q3": "#FBC02D",  # Yellow
    "q4": "#689F38",  # Light green
    "q5": "#388E3C",  # Dark green (highest)
}

# Colorblind-friendly palette
COLORBLIND_SAFE = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "purple": "#5B4E96",
    "brown": "#A65628",
    "pink": "#F0E442",
    "gray": "#999999",
}

# Layout templates
DEFAULT_TEMPLATE = {
    "layout": {
        # Typography
        "font": {
            "family": "Arial, Helvetica, sans-serif",
            "size": 12,
            "color": FINANCIAL_COLORS["text"],
        },
        "title": {
            "font": {"size": 16, "color": FINANCIAL_COLORS["text"]},
            "x": 0.5,
            "xanchor": "center",
        },
        # Colors
        "plot_bgcolor": FINANCIAL_COLORS["background"],
        "paper_bgcolor": FINANCIAL_COLORS["paper"],
        # Margins
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        # Hover
        "hovermode": "closest",
        "hoverlabel": {"bgcolor": "white", "font_size": 12, "font_family": "Arial"},
        # Grid and axes
        "xaxis": {
            "gridcolor": FINANCIAL_COLORS["grid"],
            "zeroline": False,
            "showgrid": True,
            "showline": True,
            "linecolor": FINANCIAL_COLORS["grid"],
            "tickfont": {"size": 11},
        },
        "yaxis": {
            "gridcolor": FINANCIAL_COLORS["grid"],
            "zeroline": True,
            "zerolinecolor": FINANCIAL_COLORS["grid"],
            "showgrid": True,
            "showline": True,
            "linecolor": FINANCIAL_COLORS["grid"],
            "tickfont": {"size": 11},
        },
        # Legend
        "legend": {
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": FINANCIAL_COLORS["grid"],
            "borderwidth": 1,
        },
    },
}

# Dark theme for dashboards
DARK_TEMPLATE = {
    "layout": {
        # Typography
        "font": {"family": "Arial, Helvetica, sans-serif", "size": 12, "color": "#E0E0E0"},
        "title": {"font": {"size": 16, "color": "#FFFFFF"}, "x": 0.5, "xanchor": "center"},
        # Colors
        "plot_bgcolor": "#1E1E1E",
        "paper_bgcolor": "#121212",
        # Grid and axes
        "xaxis": {
            "gridcolor": "#333333",
            "zeroline": False,
            "showgrid": True,
            "showline": True,
            "linecolor": "#444444",
            "tickfont": {"color": "#B0B0B0"},
        },
        "yaxis": {
            "gridcolor": "#333333",
            "zeroline": True,
            "zerolinecolor": "#444444",
            "showgrid": True,
            "showline": True,
            "linecolor": "#444444",
            "tickfont": {"color": "#B0B0B0"},
        },
        # Legend
        "legend": {
            "bgcolor": "rgba(30,30,30,0.9)",
            "bordercolor": "#444444",
            "borderwidth": 1,
            "font": {"color": "#E0E0E0"},
        },
    },
}

# Print-friendly template
PRINT_TEMPLATE = {
    "layout": {
        # Black and white only
        "font": {"family": "Times New Roman, serif", "size": 10, "color": "black"},
        "title": {"font": {"size": 14, "color": "black"}, "x": 0.5, "xanchor": "center"},
        # White background
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        # Minimal margins for printing
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        # High contrast grid
        "xaxis": {
            "gridcolor": "#CCCCCC",
            "zeroline": True,
            "zerolinecolor": "black",
            "showgrid": True,
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
        },
        "yaxis": {
            "gridcolor": "#CCCCCC",
            "zeroline": True,
            "zerolinecolor": "black",
            "showgrid": True,
            "showline": True,
            "linecolor": "black",
            "linewidth": 1,
        },
        # No shadows or effects
        "legend": {"bgcolor": "white", "bordercolor": "black", "borderwidth": 1},
    },
}


def get_color_scale(n_colors: int, scheme: str = "diverging") -> list[str]:
    """Get a color scale for visualizations.

    Parameters
    ----------
    n_colors : int
        Number of colors needed
    scheme : str, default "diverging"
        Color scheme type: "diverging", "sequential", "quantile", "colorblind"

    Returns:
    -------
    list[str]
        List of hex color codes
    """
    if scheme == "diverging":
        # Red to green through white
        if n_colors == 2:
            return [FINANCIAL_COLORS["negative"], FINANCIAL_COLORS["positive"]]
        if n_colors == 3:
            return [
                FINANCIAL_COLORS["negative"],
                FINANCIAL_COLORS["neutral"],
                FINANCIAL_COLORS["positive"],
            ]
        # Use plotly's RdYlGn scale
        import plotly.colors as pc

        return pc.sample_colorscale("RdYlGn", n_colors)

    if scheme == "sequential":
        # Blue gradient
        if n_colors <= 5:
            return ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"][:n_colors]
        import plotly.colors as pc

        return pc.sample_colorscale("Blues", n_colors)

    if scheme == "quantile":
        # Specific colors for quantiles
        quantile_colors = [
            FINANCIAL_COLORS["q1"],
            FINANCIAL_COLORS["q2"],
            FINANCIAL_COLORS["q3"],
            FINANCIAL_COLORS["q4"],
            FINANCIAL_COLORS["q5"],
        ]
        if n_colors <= 5:
            return quantile_colors[:n_colors]
        # Interpolate if more than 5
        import plotly.colors as pc

        return pc.sample_colorscale("RdYlGn", n_colors)

    if scheme == "colorblind":
        # Colorblind-safe palette
        colors = list(COLORBLIND_SAFE.values())
        if n_colors <= len(colors):
            return colors[:n_colors]
        # Cycle through if need more
        return (colors * (n_colors // len(colors) + 1))[:n_colors]

    # Default categorical
    import plotly.express as px

    return px.colors.qualitative.Set3[:n_colors]


def apply_theme(fig: Any, theme: str = "default") -> Any:
    """Apply a theme to a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to apply theme to
    theme : str, default "default"
        Theme name: "default", "dark", "print", "colorblind"

    Returns:
    -------
    plotly.graph_objects.Figure
        Figure with theme applied
    """
    if theme == "default":
        template = DEFAULT_TEMPLATE
    elif theme == "dark":
        template = DARK_TEMPLATE
    elif theme == "print":
        template = PRINT_TEMPLATE
    elif theme == "colorblind":
        # Apply colorblind-safe colors to existing theme
        template = DEFAULT_TEMPLATE.copy()
        # Would need to update trace colors here
    else:
        raise ValueError(f"Unknown theme: {theme}")

    # Apply template
    fig.update_layout(template["layout"])

    return fig


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a value as percentage for display.

    Parameters
    ----------
    value : float
        Value to format (0.05 = 5%)
    decimals : int, default 1
        Number of decimal places

    Returns:
    -------
    str
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "$", decimals: int = 0) -> str:
    """Format a value as currency for display.

    Parameters
    ----------
    value : float
        Value to format
    currency : str, default "$"
        Currency symbol
    decimals : int, default 0
        Number of decimal places

    Returns:
    -------
    str
        Formatted currency string
    """
    if decimals == 0:
        return f"{currency}{value:,.0f}"
    return f"{currency}{value:,.{decimals}f}"


# Accessibility helpers
def add_pattern_overlay(fig: Any, _trace_index: int, _pattern: str = "diagonal") -> Any:
    """Add pattern overlay for better accessibility.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to modify
    trace_index : int
        Index of trace to add pattern to
    pattern : str, default "diagonal"
        Pattern type: "diagonal", "vertical", "horizontal", "dot"

    Returns:
    -------
    plotly.graph_objects.Figure
        Modified figure
    """
    # This would add SVG patterns for accessibility
    # Implementation depends on Plotly version and trace type
    return fig
