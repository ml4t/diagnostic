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
    "warning": "#f59e0b",  # Warning amber - caution
    "neutral": "#334155",  # Slate gray - neutral elements
    # Backgrounds
    "bg_light": "#FAFAF9",  # Warm off-white (light mode)
    "bg_dark": "#0a1628",  # Deep blue (dark mode)
    # Quantile ramp (5-step, red→green — visible on light and dark)
    "q1": "#ef4444",  # Worst quintile (red)
    "q2": "#C87533",  # Below average (copper)
    "q3": "#E4B85B",  # Neutral (amber light)
    "q4": "#6b9f9e",  # Above average (sage)
    "q5": "#10b981",  # Best quintile (green)
}

# Semantic aliases — use these for chart data series
SERIES_COLORS = {
    "strategy": COLORS["blue"],  # Primary data line
    "benchmark": COLORS["amber"],  # Benchmark overlay
    "fill": "rgba(10,22,40,0.06)",  # Light fill under curves
    "winner": COLORS["positive"],  # Winning trades/periods
    "loser": COLORS["negative"],  # Losing trades/periods
    "drawdown": "rgba(239,68,68,0.15)",  # Drawdown fill
    "drawdown_line": "rgba(239,68,68,0.6)",  # Drawdown line
}

# Fama-French factor palette — consistent across all factor charts
# Hierarchy: Alpha (bright) > Mkt-RF (dark anchor) > style factors (muted) > Residual (gray)
FACTOR_COLORS: dict[str, str] = {
    "Mkt-RF": "#1a2d4a",  # Dark navy — dominant, matches header identity
    "SMB": "#6b9f9e",  # Muted sage — blends in
    "HML": "#c4a872",  # Warm sand — blends in
    "RMW": "#8b7fb5",  # Muted lavender — blends in
    "CMA": "#b87d8b",  # Dusty rose — blends in
    "Alpha": "#2563eb",  # Bright blue — the star (manager skill)
    "Residual": "#94a3b8",  # Slate gray — unexplained noise
    "Total": "#3d3d3d",  # Dark charcoal — distinct from all factors
}

# Human-readable factor descriptions
FACTOR_DESCRIPTIONS: dict[str, str] = {
    "Mkt-RF": "Market excess return",
    "SMB": "Small Minus Big (size)",
    "HML": "High Minus Low (value)",
    "RMW": "Robust Minus Weak (profitability)",
    "CMA": "Conservative Minus Aggressive (investment)",
}


def get_factor_color(name: str) -> str:
    """Return the canonical color for a factor name, with fallback."""
    return FACTOR_COLORS.get(name, "#64748b")


__all__ = [
    "COLORS",
    "SERIES_COLORS",
    "FACTOR_COLORS",
    "FACTOR_DESCRIPTIONS",
    "get_factor_color",
]
