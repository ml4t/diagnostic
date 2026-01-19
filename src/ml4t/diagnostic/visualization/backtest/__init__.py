"""Backtest visualization module.

Plotly-based interactive visualizations for backtest analysis.
State-of-the-art tearsheet generation exceeding QuantStats.

This module provides:
- Executive summary with KPI cards and traffic lights
- Trade-level visualizations (MFE/MAE, exit reasons, waterfall)
- Cost attribution analysis (gross-to-net decomposition)
- Statistical validity displays (DSR gauge, confidence intervals)
- Unified tearsheet generation with template system
"""

from .cost_attribution import (
    plot_cost_by_asset,
    plot_cost_over_time,
    plot_cost_pie,
    plot_cost_sensitivity,
    plot_cost_waterfall,
)
from .executive_summary import (
    create_executive_summary,
    create_key_insights,
    create_metric_card,
    get_traffic_light_color,
)
from .interactive_controls import (
    get_date_range_html,
    get_drill_down_modal_html,
    get_interactive_toolbar_html,
    get_metric_filter_html,
    get_section_navigation_html,
    get_theme_switcher_html,
)
from .statistical_validity import (
    plot_confidence_intervals,
    plot_dsr_gauge,
    plot_minimum_track_record,
    plot_ras_analysis,
    plot_statistical_summary_card,
)
from .tearsheet import (
    BacktestTearsheet,
    generate_backtest_tearsheet,
)
from .template_system import (
    TearsheetSection,
    TearsheetTemplate,
    get_template,
)
from .trade_plots import (
    plot_consecutive_analysis,
    plot_exit_reason_breakdown,
    plot_mfe_mae_scatter,
    plot_trade_duration_distribution,
    plot_trade_size_vs_return,
    plot_trade_waterfall,
)

__all__ = [
    # Executive Summary
    "create_executive_summary",
    "create_key_insights",
    "create_metric_card",
    "get_traffic_light_color",
    # Trade Plots (Phase 2)
    "plot_mfe_mae_scatter",
    "plot_exit_reason_breakdown",
    "plot_trade_waterfall",
    "plot_trade_duration_distribution",
    "plot_trade_size_vs_return",
    "plot_consecutive_analysis",
    # Cost Attribution (Phase 3)
    "plot_cost_waterfall",
    "plot_cost_sensitivity",
    "plot_cost_over_time",
    "plot_cost_by_asset",
    "plot_cost_pie",
    # Statistical Validity (Phase 4)
    "plot_dsr_gauge",
    "plot_confidence_intervals",
    "plot_ras_analysis",
    "plot_minimum_track_record",
    "plot_statistical_summary_card",
    # Unified Tearsheet (Phase 5)
    "generate_backtest_tearsheet",
    "BacktestTearsheet",
    "get_template",
    "TearsheetTemplate",
    "TearsheetSection",
    # Interactive Controls (Phase 6)
    "get_date_range_html",
    "get_metric_filter_html",
    "get_section_navigation_html",
    "get_drill_down_modal_html",
    "get_interactive_toolbar_html",
    "get_theme_switcher_html",
]
