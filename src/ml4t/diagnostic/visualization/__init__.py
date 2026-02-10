"""ML4T Diagnostic Visualization Module.

Provides interactive Plotly-based visualizations for all analysis results.

This module implements the four-tier visualization architecture:
- Layer 1: Analysis (compute_*, analyze_*) - Returns dicts
- Layer 2: Visualization (plot_*) - Returns Plotly Figures
- Layer 3: Reporting (generate_*_report) - HTML/PDF outputs
- Layer 4: Dashboard (Streamlit) - Separate package

All plot functions follow consistent patterns:
- Accept results dict from analyze_*() functions
- Return go.Figure objects
- Support theming and customization
- Interactive by default

Examples
--------
>>> from ml4t-diagnostic import analyze_ml_importance
>>> from ml4t.diagnostic.visualization import plot_importance_bar
>>>
>>> # Analyze
>>> results = analyze_ml_importance(model, X, y)
>>>
>>> # Visualize
>>> fig = plot_importance_bar(results)
>>> fig.show()
>>>
>>> # Or save
>>> fig.write_html("importance.html")
"""

from ml4t.diagnostic.visualization.barrier_plots import (
    # Barrier analysis plots (Phase 4)
    plot_hit_rate_heatmap,
    plot_precision_recall_curve,
    plot_profit_factor_bar,
    plot_time_to_target_box,
)
from ml4t.diagnostic.visualization.cv_plots import (
    # Cross-validation fold visualization
    plot_cv_folds,
)
from ml4t.diagnostic.visualization.core import (
    # Common plot elements
    add_annotation,
    add_confidence_band,
    add_threshold_line,
    apply_responsive_layout,
    # Layout helpers
    create_base_figure,
    # Color schemes
    get_color_scheme,
    get_colorscale,
    get_plot_theme,
    get_theme_config,
    # Theme management
    set_plot_theme,
    # Validation
    validate_plot_results,
    validate_positive_int,
    validate_theme,
)
from ml4t.diagnostic.visualization.dashboards import (
    # Dashboard base classes
    BaseDashboard,
    DashboardSection,
    # Interactive dashboards
    FeatureImportanceDashboard,
    FeatureInteractionDashboard,
)
from ml4t.diagnostic.visualization.data_extraction import (
    # TypedDict structures
    ImportanceVizData,
    InteractionVizData,
    # Data extraction functions
    extract_importance_viz_data,
    extract_interaction_viz_data,
)
from ml4t.diagnostic.visualization.feature_plots import (
    # Feature importance visualizations
    plot_importance_bar,
    plot_importance_distribution,
    plot_importance_heatmap,
    plot_importance_summary,
)
from ml4t.diagnostic.visualization.interaction_plots import (
    # Feature interaction visualizations
    plot_interaction_bar,
    plot_interaction_heatmap,
    plot_interaction_network,
)
from ml4t.diagnostic.visualization.portfolio import (
    create_portfolio_dashboard,
    plot_annual_returns_bar,
    plot_drawdown_periods,
    plot_drawdown_underwater,
    plot_monthly_returns_heatmap,
    plot_returns_distribution,
    plot_rolling_beta,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)
from ml4t.diagnostic.visualization.portfolio import (
    # Portfolio tear sheet (pyfolio replacement)
    plot_cumulative_returns as plot_portfolio_cumulative_returns,
)
from ml4t.diagnostic.visualization.portfolio import (
    plot_rolling_returns as plot_portfolio_rolling_returns,
)
from ml4t.diagnostic.visualization.report_generation import (
    combine_figures_to_html,
    # PDF export
    export_figures_to_pdf,
    generate_combined_report,
    # HTML report generation
    generate_importance_report,
    generate_interaction_report,
)
from ml4t.diagnostic.visualization.signal import (
    MultiSignalDashboard,
    # Dashboards
    SignalDashboard,
    # Turnover plots
    plot_autocorrelation,
    # Quantile plots
    plot_cumulative_returns,
    # IC plots
    plot_ic_heatmap,
    plot_ic_histogram,
    plot_ic_qq,
    # Multi-signal plots (Phase 3)
    plot_ic_ridge,
    plot_ic_ts,
    plot_pareto_frontier,
    plot_quantile_returns_bar,
    plot_quantile_returns_violin,
    plot_signal_correlation_heatmap,
    plot_signal_ranking_bar,
    plot_spread_timeseries,
    plot_top_bottom_turnover,
)

__all__ = [
    # Theme management
    "set_plot_theme",
    "get_plot_theme",
    "get_theme_config",
    # Color schemes
    "get_color_scheme",
    "get_colorscale",
    # Validation
    "validate_plot_results",
    "validate_positive_int",
    "validate_theme",
    # Layout helpers
    "create_base_figure",
    "apply_responsive_layout",
    # Common plot elements
    "add_annotation",
    "add_threshold_line",
    "add_confidence_band",
    # Feature importance plots
    "plot_importance_bar",
    "plot_importance_heatmap",
    "plot_importance_distribution",
    "plot_importance_summary",
    # Feature interaction plots
    "plot_interaction_bar",
    "plot_interaction_heatmap",
    "plot_interaction_network",
    # HTML report generation
    "generate_importance_report",
    "generate_interaction_report",
    "generate_combined_report",
    "combine_figures_to_html",
    # PDF export
    "export_figures_to_pdf",
    # Data extraction
    "extract_importance_viz_data",
    "extract_interaction_viz_data",
    "ImportanceVizData",
    "InteractionVizData",
    # Dashboard components
    "BaseDashboard",
    "DashboardSection",
    "FeatureImportanceDashboard",
    "FeatureInteractionDashboard",
    # Signal IC plots
    "plot_ic_ts",
    "plot_ic_histogram",
    "plot_ic_qq",
    "plot_ic_heatmap",
    # Signal quantile plots
    "plot_quantile_returns_bar",
    "plot_quantile_returns_violin",
    "plot_cumulative_returns",
    "plot_spread_timeseries",
    # Signal turnover plots
    "plot_top_bottom_turnover",
    "plot_autocorrelation",
    # Multi-signal plots (Phase 3)
    "plot_ic_ridge",
    "plot_signal_ranking_bar",
    "plot_signal_correlation_heatmap",
    "plot_pareto_frontier",
    # Signal dashboards
    "SignalDashboard",
    "MultiSignalDashboard",
    # Barrier analysis plots (Phase 4)
    "plot_hit_rate_heatmap",
    "plot_profit_factor_bar",
    "plot_precision_recall_curve",
    "plot_time_to_target_box",
    # Portfolio tear sheet (pyfolio replacement)
    "plot_portfolio_cumulative_returns",
    "plot_portfolio_rolling_returns",
    "plot_annual_returns_bar",
    "plot_monthly_returns_heatmap",
    "plot_returns_distribution",
    "plot_rolling_volatility",
    "plot_rolling_sharpe",
    "plot_rolling_beta",
    "plot_drawdown_underwater",
    "plot_drawdown_periods",
    "create_portfolio_dashboard",
    # Cross-validation fold visualization
    "plot_cv_folds",
]
