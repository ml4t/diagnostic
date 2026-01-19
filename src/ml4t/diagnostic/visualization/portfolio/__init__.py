"""Portfolio visualization module.

Plotly-based interactive visualizations for portfolio analysis.
Replacement for pyfolio's matplotlib-based plots.
"""

from .dashboard import create_portfolio_dashboard
from .drawdown_plots import (
    plot_drawdown_periods,
    plot_drawdown_underwater,
)
from .returns_plots import (
    plot_annual_returns_bar,
    plot_cumulative_returns,
    plot_monthly_returns_heatmap,
    plot_returns_distribution,
    plot_rolling_returns,
)
from .risk_plots import (
    plot_rolling_beta,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)

__all__ = [
    # Returns
    "plot_cumulative_returns",
    "plot_rolling_returns",
    "plot_annual_returns_bar",
    "plot_monthly_returns_heatmap",
    "plot_returns_distribution",
    # Risk
    "plot_rolling_volatility",
    "plot_rolling_sharpe",
    "plot_rolling_beta",
    # Drawdown
    "plot_drawdown_underwater",
    "plot_drawdown_periods",
    # Dashboard
    "create_portfolio_dashboard",
]
