"""ML4T color palette.

Canonical color definitions for all ml4t-diagnostic visualizations.
Aligned with the ml4t.io website identity.

If ml4t-style is ever published as a shared package, this module
can be replaced with a re-export from that package.
"""

COLORS = {
    # Primary blues (core identity)
    "blue": "#0a1628",  # Deep blue - primary emphasis, main data
    "blue_light": "#152238",  # Lighter blue - secondary elements
    "slate": "#1a2d4a",  # Mid-blue - tertiary, gridlines
    # Silver tones (backgrounds, text)
    "silver": "#F8F8F6",  # Light silver - text on dark, highlights
    "silver_muted": "#e8e8e6",  # Muted silver - borders, subtle elements
    # Warm accents (highlights, emphasis)
    "amber": "#D4A84B",  # Warm amber - CTAs, important highlights
    "amber_light": "#E4B85B",  # Lighter amber - hover states
    "copper": "#C87533",  # Copper - secondary accent
    # Semantic (for data meaning)
    "positive": "#10b981",  # Success green - profits, gains
    "negative": "#ef4444",  # Error red - losses (use sparingly!)
    "neutral": "#334155",  # Slate gray - neutral elements
    # Backgrounds
    "bg_light": "#FAFAF9",  # Warm off-white (light mode)
    "bg_dark": "#0a1628",  # Deep blue (dark mode)
}

__all__ = ["COLORS"]
