"""Signal Analysis Visualization Module.

Provides interactive Plotly visualizations for signal/factor analysis:

Dashboard (dashboard.py):
- SignalDashboard: Multi-tab HTML dashboard for signal analysis
- MultiSignalDashboard: Multi-tab dashboard for 50-200 signals

IC Plots (ic_plots.py):
- plot_ic_ts: IC time series with rolling mean
- plot_ic_histogram: IC distribution
- plot_ic_qq: Q-Q plot for normality
- plot_ic_heatmap: Monthly IC heatmap

Quantile Plots (quantile_plots.py):
- plot_quantile_returns_bar: Mean returns by quantile
- plot_quantile_returns_violin: Return distributions
- plot_cumulative_returns: Cumulative returns by quantile
- plot_spread_timeseries: Top-bottom spread over time

Turnover Plots (turnover_plots.py):
- plot_top_bottom_turnover: Extreme quantile turnover
- plot_autocorrelation: Signal rank autocorrelation

Event Study Plots (event_plots.py):
- plot_caar: CAAR time series with confidence bands
- plot_event_heatmap: Abnormal returns heatmap by event/day
- plot_ar_distribution: AR distribution for specific day
- plot_car_by_event: CAR bar chart by event

Multi-Signal Plots (multi_signal_plots.py):
- plot_ic_ridge: IC density ridge plot per signal
- plot_signal_ranking_bar: Horizontal bar chart by metric
- plot_signal_correlation_heatmap: Cluster heatmap
- plot_pareto_frontier: IC IR vs Turnover trade-off

All plots follow the ML4T Diagnostic visualization standards:
- Theme-aware (default, dark, print, presentation)
- Interactive with hover details
- Configurable colors and dimensions
"""

from ml4t.diagnostic.visualization.signal.dashboard import SignalDashboard
from ml4t.diagnostic.visualization.signal.event_plots import (
    plot_ar_distribution,
    plot_caar,
    plot_car_by_event,
    plot_event_heatmap,
)
from ml4t.diagnostic.visualization.signal.ic_plots import (
    plot_ic_heatmap,
    plot_ic_histogram,
    plot_ic_qq,
    plot_ic_ts,
)
from ml4t.diagnostic.visualization.signal.multi_signal_dashboard import (
    MultiSignalDashboard,
)
from ml4t.diagnostic.visualization.signal.multi_signal_plots import (
    plot_ic_ridge,
    plot_pareto_frontier,
    plot_signal_correlation_heatmap,
    plot_signal_ranking_bar,
)
from ml4t.diagnostic.visualization.signal.quantile_plots import (
    plot_cumulative_returns,
    plot_quantile_returns_bar,
    plot_quantile_returns_violin,
    plot_spread_timeseries,
)
from ml4t.diagnostic.visualization.signal.turnover_plots import (
    plot_autocorrelation,
    plot_top_bottom_turnover,
)

__all__ = [
    # Dashboards
    "SignalDashboard",
    "MultiSignalDashboard",
    # IC plots
    "plot_ic_ts",
    "plot_ic_histogram",
    "plot_ic_qq",
    "plot_ic_heatmap",
    # Quantile plots
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_cumulative_returns",
    "plot_spread_timeseries",
    # Turnover plots
    "plot_top_bottom_turnover",
    "plot_autocorrelation",
    # Event study plots
    "plot_caar",
    "plot_event_heatmap",
    "plot_ar_distribution",
    "plot_car_by_event",
    # Multi-signal plots
    "plot_ic_ridge",
    "plot_signal_ranking_bar",
    "plot_signal_correlation_heatmap",
    "plot_pareto_frontier",
]
