"""Core plotting utilities for ML4T Diagnostic visualizations.

Provides theme management, color schemes, validation helpers, and
common layout patterns used across all plot functions.

This module implements the standards defined in docs/plot_api_standards.md.
"""

from typing import Any

import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Global Theme State
# =============================================================================

_CURRENT_THEME = "default"  # Global theme setting


def set_plot_theme(theme: str) -> None:
    """Set the global plot theme for all subsequent visualizations.

    Parameters
    ----------
    theme : str
        Theme name: "default", "dark", "print", "presentation"

    Raises
    ------
    ValueError
        If theme name is not recognized

    Examples
    --------
    >>> import ml4t.diagnostic
    >>> ml4t-diagnostic.set_plot_theme("dark")
    >>> # All plots now use dark theme
    >>> fig = plot_importance_bar(results)
    """
    global _CURRENT_THEME

    if theme not in AVAILABLE_THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {', '.join(AVAILABLE_THEMES.keys())}"
        )

    _CURRENT_THEME = theme


def get_plot_theme() -> str:
    """Get the current global plot theme.

    Returns
    -------
    str
        Current theme name

    Examples
    --------
    >>> import ml4t.diagnostic
    >>> ml4t-diagnostic.get_plot_theme()
    'default'
    """
    return _CURRENT_THEME


# =============================================================================
# Theme Definitions
# =============================================================================

THEME_DEFAULT = {
    "name": "default",
    "description": "Clean, modern light theme for general use",
    "layout": {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F8F9FA",
        "font": {
            "family": "Inter, -apple-system, system-ui, sans-serif",
            "size": 12,
            "color": "#2C3E50",
        },
        "title_font": {
            "size": 18,
            "color": "#2C3E50",
            "family": "Inter, -apple-system, system-ui, sans-serif",
        },
        "margin": {"l": 80, "r": 20, "t": 100, "b": 80},
        "hovermode": "closest",
        "hoverlabel": {"bgcolor": "white", "font_size": 13, "font_family": "Inter, sans-serif"},
    },
    "colorway": [
        "#3498DB",  # Blue
        "#E74C3C",  # Red
        "#2ECC71",  # Green
        "#F39C12",  # Orange
        "#9B59B6",  # Purple
        "#1ABC9C",  # Teal
        "#E67E22",  # Dark orange
        "#95A5A6",  # Gray
    ],
    "color_schemes": {
        "sequential": "Blues",
        "diverging": "RdBu",
        "qualitative": "Set2",
    },
    "defaults": {
        "bar_height": 600,
        "heatmap_height": 800,
        "scatter_height": 700,
        "line_height": 500,
        "width": 1000,
    },
}

THEME_DARK = {
    "name": "dark",
    "description": "Dark mode theme for dashboards and presentations",
    "layout": {
        "paper_bgcolor": "#1E1E1E",
        "plot_bgcolor": "#2D2D2D",
        "font": {
            "family": "Inter, -apple-system, system-ui, sans-serif",
            "size": 12,
            "color": "#E0E0E0",
        },
        "title_font": {
            "size": 18,
            "color": "#FFFFFF",
            "family": "Inter, -apple-system, system-ui, sans-serif",
        },
        "margin": {"l": 80, "r": 20, "t": 100, "b": 80},
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "#3D3D3D",
            "font_size": 13,
            "font_family": "Inter, sans-serif",
            "font_color": "#FFFFFF",
        },
    },
    "colorway": [
        "#5DADE2",  # Light blue
        "#EC7063",  # Light red
        "#58D68D",  # Light green
        "#F5B041",  # Light orange
        "#AF7AC5",  # Light purple
        "#48C9B0",  # Light teal
        "#EB984E",  # Light dark orange
        "#AAB7B8",  # Light gray
    ],
    "color_schemes": {
        "sequential": "Blues",
        "diverging": "RdBu",
        "qualitative": "Set2",
    },
    "defaults": {
        "bar_height": 600,
        "heatmap_height": 800,
        "scatter_height": 700,
        "line_height": 500,
        "width": 1000,
    },
}

THEME_PRINT = {
    "name": "print",
    "description": "Publication-quality grayscale theme",
    "layout": {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Times New Roman, serif", "size": 11, "color": "#000000"},
        "title_font": {"size": 14, "color": "#000000", "family": "Times New Roman, serif"},
        "margin": {"l": 60, "r": 20, "t": 80, "b": 60},
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": "white",
            "font_size": 11,
            "font_family": "Times New Roman, serif",
        },
    },
    "colorway": [
        "#000000",  # Black
        "#444444",  # Dark gray
        "#888888",  # Medium gray
        "#BBBBBB",  # Light gray
    ],
    "color_schemes": {
        "sequential": "Greys",
        "diverging": "RdGy",
        "qualitative": "Greys",
    },
    "defaults": {
        "bar_height": 500,
        "heatmap_height": 700,
        "scatter_height": 600,
        "line_height": 450,
        "width": 800,
    },
}

THEME_PRESENTATION = {
    "name": "presentation",
    "description": "High-contrast theme for slides and presentations",
    "layout": {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#F0F0F0",
        "font": {
            "family": "Inter, -apple-system, system-ui, sans-serif",
            "size": 16,  # Larger fonts
            "color": "#000000",
        },
        "title_font": {
            "size": 24,  # Much larger title
            "color": "#000000",
            "family": "Inter, -apple-system, system-ui, sans-serif",
        },
        "margin": {"l": 100, "r": 40, "t": 120, "b": 100},
        "hovermode": "closest",
        "hoverlabel": {"bgcolor": "white", "font_size": 16, "font_family": "Inter, sans-serif"},
    },
    "colorway": [
        "#0066CC",  # Strong blue
        "#FF3333",  # Strong red
        "#00CC66",  # Strong green
        "#FF9900",  # Strong orange
        "#9933CC",  # Strong purple
        "#00CCCC",  # Strong teal
    ],
    "color_schemes": {
        "sequential": "Blues",
        "diverging": "RdBu",
        "qualitative": "Bold",
    },
    "defaults": {
        "bar_height": 700,
        "heatmap_height": 900,
        "scatter_height": 800,
        "line_height": 600,
        "width": 1200,
    },
}

AVAILABLE_THEMES = {
    "default": THEME_DEFAULT,
    "dark": THEME_DARK,
    "print": THEME_PRINT,
    "presentation": THEME_PRESENTATION,
}


def get_theme_config(theme: str | None = None) -> dict[str, Any]:
    """Get complete theme configuration.

    Parameters
    ----------
    theme : str | None, default None
        Theme name. If None, uses current global theme

    Returns
    -------
    dict[str, Any]
        Theme configuration dict with layout, colorway, defaults

    Raises
    ------
    ValueError
        If theme name is not recognized

    Examples
    --------
    >>> config = get_theme_config("dark")
    >>> config["layout"]["paper_bgcolor"]
    '#1E1E1E'
    """
    if theme is None:
        theme = get_plot_theme()

    if theme not in AVAILABLE_THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {', '.join(AVAILABLE_THEMES.keys())}"
        )

    return AVAILABLE_THEMES[theme]


# =============================================================================
# Color Schemes
# =============================================================================

COLOR_SCHEMES = {
    # Sequential (single hue, light to dark)
    "blues": px.colors.sequential.Blues,
    "greens": px.colors.sequential.Greens,
    "reds": px.colors.sequential.Reds,
    "oranges": px.colors.sequential.Oranges,
    "viridis": px.colors.sequential.Viridis,
    "cividis": px.colors.sequential.Cividis,
    "plasma": px.colors.sequential.Plasma,
    # Diverging (two hues with neutral center)
    "rdbu": px.colors.diverging.RdBu,
    "rdylgn": px.colors.diverging.RdYlGn,
    "brbg": px.colors.diverging.BrBG,
    "prgn": px.colors.diverging.PRGn,
    "blues_oranges": ["#0571B0", "#92C5DE", "#F7F7F7", "#F4A582", "#CA0020"],
    # Qualitative (distinct colors for categories)
    "set2": px.colors.qualitative.Set2,
    "set3": px.colors.qualitative.Set3,
    "pastel": px.colors.qualitative.Pastel,
    "dark2": px.colors.qualitative.Dark2,
    "bold": px.colors.qualitative.Bold,
    # Financial
    "gains_losses": ["#FF4444", "#CCCCCC", "#00CC88"],  # Red, gray, green
    "quantiles": ["#D32F2F", "#F57C00", "#FBC02D", "#689F38", "#388E3C"],
    # Color-blind safe
    "colorblind_safe": [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#CC78BC",
        "#5B4E96",
        "#A65628",
        "#F0E442",
        "#999999",
    ],
}


def get_color_scheme(name: str) -> list[str]:
    """Get a named color scheme.

    Parameters
    ----------
    name : str
        Color scheme name (see COLOR_SCHEMES for options)

    Returns
    -------
    list[str]
        List of hex color codes

    Raises
    ------
    ValueError
        If color scheme name is not recognized

    Examples
    --------
    >>> colors = get_color_scheme("viridis")
    >>> len(colors)
    11
    """
    name = name.lower()

    if name not in COLOR_SCHEMES:
        raise ValueError(
            f"Unknown color scheme '{name}'. Available: {', '.join(COLOR_SCHEMES.keys())}"
        )

    return COLOR_SCHEMES[name]


def get_colorscale(
    name: str, n_colors: int | None = None, reverse: bool = False
) -> list[str] | list[tuple[float, str]]:
    """Get a color scale for continuous or discrete coloring.

    Parameters
    ----------
    name : str
        Color scheme name
    n_colors : int | None, default None
        Number of discrete colors. If None, returns continuous colorscale
    reverse : bool, default False
        Reverse the color order

    Returns
    -------
    list[str] | list[tuple[float, str]]
        Discrete colors (if n_colors specified) or continuous colorscale

    Examples
    --------
    >>> # Continuous colorscale
    >>> scale = get_colorscale("viridis")
    >>> # Discrete colors
    >>> colors = get_colorscale("viridis", n_colors=5)
    >>> len(colors)
    5
    """
    colors = get_color_scheme(name)

    if reverse:
        colors = list(reversed(colors))

    if n_colors is None:
        # Return as continuous colorscale
        return colors

    # Sample n_colors from the scheme
    if n_colors <= len(colors):
        # Use evenly spaced colors including both endpoints
        import numpy as np

        indices = np.linspace(0, len(colors) - 1, n_colors, dtype=int)
        return [colors[i] for i in indices]
    else:
        # Need to interpolate
        import plotly.colors as pc

        return pc.sample_colorscale(colors, n_colors)


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_plot_results(
    results: dict[str, Any], required_keys: list[str], function_name: str
) -> None:
    """Validate that results dict has required structure.

    Parameters
    ----------
    results : dict[str, Any]
        Results dict from analyze_*() function
    required_keys : list[str]
        Keys that must be present in results
    function_name : str
        Name of calling function (for error messages)

    Raises
    ------
    TypeError
        If results is not a dict
    ValueError
        If required keys are missing

    Examples
    --------
    >>> validate_plot_results(
    ...     results,
    ...     required_keys=["consensus_ranking", "method_results"],
    ...     function_name="plot_importance_bar"
    ... )
    """
    if not isinstance(results, dict):
        raise TypeError(
            f"{function_name} requires dict from analyze_*() function, got {type(results).__name__}"
        )

    missing = [k for k in required_keys if k not in results]
    if missing:
        raise ValueError(
            f"Invalid results dict for {function_name}. "
            f"Missing keys: {missing}. "
            f"Expected output from corresponding analyze_*() function."
        )


def validate_positive_int(value: int | None, name: str) -> None:
    """Validate that value is a positive integer.

    Parameters
    ----------
    value : int | None
        Value to validate
    name : str
        Parameter name (for error messages)

    Raises
    ------
    ValueError
        If value is not a positive integer

    Examples
    --------
    >>> validate_positive_int(10, "top_n")  # OK
    >>> validate_positive_int(-5, "top_n")  # Raises ValueError
    """
    if value is not None and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_theme(theme: str | None) -> str:
    """Validate and resolve theme name.

    Parameters
    ----------
    theme : str | None
        Theme name or None (use global theme)

    Returns
    -------
    str
        Validated theme name

    Raises
    ------
    ValueError
        If theme name is not recognized

    Examples
    --------
    >>> validate_theme("dark")
    'dark'
    >>> validate_theme(None)  # Returns global theme
    'default'
    """
    if theme is None:
        theme = get_plot_theme()

    if theme not in AVAILABLE_THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {', '.join(AVAILABLE_THEMES.keys())}"
        )

    return theme


def validate_color_scheme(scheme: str | None, theme: str) -> str:
    """Validate and resolve color scheme name.

    Parameters
    ----------
    scheme : str | None
        Color scheme name or None (use theme default)
    theme : str
        Theme name (for default color scheme)

    Returns
    -------
    str
        Validated color scheme name

    Raises
    ------
    ValueError
        If color scheme name is not recognized

    Examples
    --------
    >>> validate_color_scheme("viridis", "default")
    'viridis'
    >>> validate_color_scheme(None, "default")  # Uses theme default
    'blues'
    """
    if scheme is None:
        # Use theme's default sequential scheme
        theme_config = get_theme_config(theme)
        scheme = theme_config["color_schemes"]["sequential"]

    scheme = scheme.lower()

    if scheme not in COLOR_SCHEMES:
        raise ValueError(
            f"Unknown color scheme '{scheme}'. Available: {', '.join(COLOR_SCHEMES.keys())}"
        )

    return scheme


# =============================================================================
# Layout Helpers
# =============================================================================


def create_base_figure(
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    theme: str | None = None,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    """Create a base figure with theme applied.

    Parameters
    ----------
    title : str | None, default None
        Figure title
    xaxis_title : str | None, default None
        X-axis label
    yaxis_title : str | None, default None
        Y-axis label
    width : int | None, default None
        Figure width in pixels
    height : int | None, default None
        Figure height in pixels
    theme : str | None, default None
        Theme name
    margin : dict[str, int] | None, default None
        Margin overrides

    Returns
    -------
    go.Figure
        Configured Plotly figure

    Examples
    --------
    >>> fig = create_base_figure(
    ...     title="Feature Importance",
    ...     xaxis_title="Features",
    ...     yaxis_title="Importance Score",
    ...     theme="dark"
    ... )
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    fig = go.Figure()

    # Build layout
    layout = {
        "title": title,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "width": width or theme_config["defaults"]["width"],
        "height": height,
        **theme_config["layout"],
    }

    if margin is not None:
        layout["margin"] = margin

    fig.update_layout(layout)

    return fig


def apply_responsive_layout(fig: go.Figure) -> go.Figure:
    """Make figure responsive (adapts to container size).

    Parameters
    ----------
    fig : go.Figure
        Figure to make responsive

    Returns
    -------
    go.Figure
        Modified figure

    Examples
    --------
    >>> fig = create_base_figure(title="Test")
    >>> fig = apply_responsive_layout(fig)
    """
    fig.update_layout(
        autosize=True,
        margin={"autoexpand": True},
    )

    return fig


def add_annotation(
    fig: go.Figure,
    text: str,
    x: float,
    y: float,
    xref: str = "paper",
    yref: str = "paper",
    showarrow: bool = False,
    **kwargs,
) -> go.Figure:
    """Add text annotation to figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to annotate
    text : str
        Annotation text
    x : float
        X position (0-1 for paper coordinates)
    y : float
        Y position (0-1 for paper coordinates)
    xref : str, default "paper"
        X reference: "paper" or "x"
    yref : str, default "paper"
        Y reference: "paper" or "y"
    showarrow : bool, default False
        Show arrow pointing to position
    **kwargs
        Additional annotation parameters

    Returns
    -------
    go.Figure
        Modified figure

    Examples
    --------
    >>> fig = create_base_figure(title="Test")
    >>> fig = add_annotation(
    ...     fig,
    ...     text="Key insight here",
    ...     x=0.5, y=0.95,
    ...     font=dict(size=14, color="red")
    ... )
    """
    fig.add_annotation(text=text, x=x, y=y, xref=xref, yref=yref, showarrow=showarrow, **kwargs)

    return fig


# =============================================================================
# Format Helpers
# =============================================================================


def format_hover_template(
    x_label: str = "x",
    y_label: str = "y",
    x_format: str = "",
    y_format: str = ".3f",
    extra: str = "",
) -> str:
    """Create a hover template string.

    Parameters
    ----------
    x_label : str, default "x"
        Label for x value
    y_label : str, default "y"
        Label for y value
    x_format : str, default ""
        Format string for x value
    y_format : str, default ".3f"
        Format string for y value
    extra : str, default ""
        Extra text to display

    Returns
    -------
    str
        Plotly hover template string

    Examples
    --------
    >>> template = format_hover_template(
    ...     x_label="Feature",
    ...     y_label="Importance",
    ...     y_format=".4f"
    ... )
    >>> template
    '<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
    """
    template = f"<b>%{{x{x_format}}}</b><br>{y_label}: %{{y{y_format}}}"

    if extra:
        template += f"<br>{extra}"

    template += "<extra></extra>"

    return template


def format_number(value: float, precision: int = 3) -> str:
    """Format number for display.

    Parameters
    ----------
    value : float
        Number to format
    precision : int, default 3
        Number of decimal places

    Returns
    -------
    str
        Formatted string

    Examples
    --------
    >>> format_number(0.123456, precision=2)
    '0.12'
    >>> format_number(1234567, precision=0)
    '1,234,567'
    """
    if precision == 0:
        return f"{value:,.0f}"
    return f"{value:,.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """Format value as percentage.

    Parameters
    ----------
    value : float
        Value to format (0.05 = 5%)
    precision : int, default 1
        Number of decimal places

    Returns
    -------
    str
        Formatted percentage string

    Examples
    --------
    >>> format_percentage(0.05, precision=1)
    '5.0%'
    >>> format_percentage(0.12345, precision=2)
    '12.35%'
    """
    return f"{value * 100:.{precision}f}%"


# =============================================================================
# Common Plot Elements
# =============================================================================


def add_threshold_line(
    fig: go.Figure,
    y: float,
    label: str | None = None,
    color: str = "gray",
    dash: str = "dash",
    line_width: float = 1,
    opacity: float = 0.8,
    row: int | None = None,
    col: int | None = None,
    annotation_position: str = "right",
) -> go.Figure:
    """Add a horizontal threshold line to a figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to modify
    y : float
        Y-axis value for the line
    label : str | None, default None
        Optional label/annotation for the line
    color : str, default "gray"
        Line color
    dash : str, default "dash"
        Line style: "solid", "dot", "dash", "longdash", "dashdot"
    line_width : float, default 1
        Line width in pixels
    opacity : float, default 0.8
        Line opacity (0-1)
    row : int | None, default None
        Subplot row (for subplots)
    col : int | None, default None
        Subplot column (for subplots)
    annotation_position : str, default "right"
        Label position: "left", "right"

    Returns
    -------
    go.Figure
        Modified figure

    Examples
    --------
    >>> fig = create_base_figure(title="Returns")
    >>> fig = add_threshold_line(fig, y=0, label="Zero line")
    >>> fig = add_threshold_line(fig, y=0.05, label="Target", color="green")
    """
    hline_kwargs = {
        "y": y,
        "line_dash": dash,
        "line_color": color,
        "line_width": line_width,
        "opacity": opacity,
    }

    if row is not None:
        hline_kwargs["row"] = row
    if col is not None:
        hline_kwargs["col"] = col

    fig.add_hline(**hline_kwargs)

    if label:
        x_pos = 0.98 if annotation_position == "right" else 0.02
        xanchor = "right" if annotation_position == "right" else "left"
        fig.add_annotation(
            text=label,
            x=x_pos,
            y=y,
            xref="paper",
            yref="y",
            showarrow=False,
            font={"size": 10, "color": color},
            xanchor=xanchor,
            yanchor="bottom",
        )

    return fig


def add_confidence_band(
    fig: go.Figure,
    x: list | Any,
    y_lower: list | Any,
    y_upper: list | Any,
    color: str = "blue",
    opacity: float = 0.2,
    name: str = "CI",
    showlegend: bool = False,
) -> go.Figure:
    """Add a shaded confidence band to a figure.

    Creates a filled area between y_lower and y_upper bounds.

    Parameters
    ----------
    fig : go.Figure
        Figure to modify
    x : array-like
        X-axis values
    y_lower : array-like
        Lower bound values
    y_upper : array-like
        Upper bound values
    color : str, default "blue"
        Fill color (name or hex)
    opacity : float, default 0.2
        Fill opacity (0-1)
    name : str, default "CI"
        Legend name
    showlegend : bool, default False
        Show in legend

    Returns
    -------
    go.Figure
        Modified figure

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(100)
    >>> y_mean = np.sin(x / 10)
    >>> y_lower = y_mean - 0.2
    >>> y_upper = y_mean + 0.2
    >>> fig = create_base_figure(title="Signal with CI")
    >>> fig = add_confidence_band(fig, x, y_lower, y_upper, color="#3498DB")
    """
    import numpy as np

    # Convert to lists if needed
    x = list(x) if hasattr(x, "__iter__") and not isinstance(x, str | list) else x
    y_lower = (
        list(y_lower)
        if hasattr(y_lower, "__iter__") and not isinstance(y_lower, str | list)
        else y_lower
    )
    y_upper = (
        list(y_upper)
        if hasattr(y_upper, "__iter__") and not isinstance(y_upper, str | list)
        else y_upper
    )

    # Convert named color to rgba
    if color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fillcolor = f"rgba({r}, {g}, {b}, {opacity})"
    elif color.startswith("rgb"):
        # Already rgb format, add alpha
        fillcolor = color.replace("rgb", "rgba").replace(")", f", {opacity})")
    else:
        # Named color - use a default mapping
        color_map = {
            "blue": (52, 152, 219),
            "red": (231, 76, 60),
            "green": (46, 204, 113),
            "orange": (243, 156, 18),
            "purple": (155, 89, 182),
            "gray": (128, 128, 128),
        }
        rgb = color_map.get(color.lower(), (128, 128, 128))
        fillcolor = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

    # Create the band using fill between traces
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself",
            fillcolor=fillcolor,
            line={"color": "rgba(0,0,0,0)"},  # Invisible line
            hoverinfo="skip",
            showlegend=showlegend,
            name=name,
        )
    )

    return fig


# =============================================================================
# Error Message Helpers
# =============================================================================


def require_plotly() -> None:
    """Check that Plotly is installed, raise helpful error if not.

    Raises
    ------
    ImportError
        If Plotly is not installed

    Examples
    --------
    >>> require_plotly()  # OK if plotly installed
    """
    try:
        import plotly.graph_objects as go  # noqa: F401 (availability check)
    except ImportError:
        raise ImportError(  # noqa: B904
            "Plotly is required for visualization. Install with:\n"
            "  pip install plotly\n"
            "Or install ML4T Diagnostic with viz extras:\n"
            "  pip install ml4t-diagnostic[viz]"
        )


def require_kaleido() -> None:
    """Check that kaleido is installed (for image export).

    Raises
    ------
    ImportError
        If kaleido is not installed

    Examples
    --------
    >>> require_kaleido()  # OK if kaleido installed
    """
    try:
        import kaleido  # noqa: F401 (availability check)
    except ImportError:
        raise ImportError(  # noqa: B904
            "Kaleido is required for image export. Install with:\n"
            "  pip install kaleido\n"
            "Or install ML4T Diagnostic with viz extras:\n"
            "  pip install ml4t-diagnostic[viz]"
        )


# Fix: Import plotly.express for color schemes
try:
    import plotly.express as px
except ImportError:
    # Plotly should be available if this module is imported
    raise ImportError(  # noqa: B904
        "Plotly is required for visualization. Install with:\n  pip install plotly"
    )
