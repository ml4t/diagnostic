"""Conftest for visualization tests - set safe plotly renderer."""

import plotly.io as pio

pio.renderers.default = "json"
