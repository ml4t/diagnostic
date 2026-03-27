"""Backtest visualization module.

Plotly-based interactive visualizations for backtest analysis.

This module provides:
- Executive summary surfaces for dense tearsheet reporting
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
    create_executive_summary_html,
    create_key_insights,
    create_key_metrics_table_html,
    create_metric_card,
    get_traffic_light_color,
)
from .export import EXPORT_BUNDLES, export_workspaces
from .html_tables import (
    create_cost_summary_line_html,
    create_credibility_box_html,
    create_top_drawdowns_table_html,
    create_worst_trades_table_html,
)
from .interactive_controls import (
    get_date_range_html,
    get_drill_down_modal_html,
    get_interactive_toolbar_html,
    get_metric_filter_html,
    get_section_navigation_html,
    get_theme_switcher_html,
)
from .ml_plots import plot_prediction_trade_alignment
from .profile_sections import (
    plot_activity_overview,
    plot_attribution_overview,
    plot_cost_bridge,
    plot_drawdown_anatomy,
    plot_occupancy_overview,
    plot_stability_overview,
)
from .shap_patterns import plot_shap_error_patterns, plot_shap_worst_trades
from .statistical_validity import (
    plot_confidence_intervals,
    plot_dsr_gauge,
    plot_minimum_track_record,
    plot_ras_analysis,
    plot_statistical_summary_card,
)
from .tail_risk import plot_tail_risk_analysis
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
    "create_executive_summary_html",
    "create_key_insights",
    "create_key_metrics_table_html",
    "create_metric_card",
    "get_traffic_light_color",
    # HTML Tables
    "create_credibility_box_html",
    "create_top_drawdowns_table_html",
    "create_worst_trades_table_html",
    "create_cost_summary_line_html",
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
    "plot_activity_overview",
    "plot_occupancy_overview",
    "plot_attribution_overview",
    "plot_drawdown_anatomy",
    "plot_cost_bridge",
    "plot_stability_overview",
    # Statistical Validity (Phase 4)
    "plot_dsr_gauge",
    "plot_confidence_intervals",
    "plot_ras_analysis",
    "plot_minimum_track_record",
    "plot_statistical_summary_card",
    # Tail Risk
    "plot_tail_risk_analysis",
    "plot_prediction_trade_alignment",
    # SHAP Error Patterns
    "plot_shap_error_patterns",
    "plot_shap_worst_trades",
    # Unified Tearsheet (Phase 5)
    "generate_backtest_tearsheet",
    "export_workspaces",
    "EXPORT_BUNDLES",
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
