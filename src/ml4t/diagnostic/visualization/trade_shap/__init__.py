"""Trade SHAP visualization components.

This package contains visualization and dashboard functions for Trade SHAP analysis.

Main Functions:
    - run_diagnostics_dashboard: Run the interactive Streamlit dashboard
    - run_polished_dashboard: Run the polished version of the dashboard
    - extract_trade_returns: Extract trade returns from results
    - extract_trade_data: Extract trade data for visualization
    - export_trades_to_csv: Export trades to CSV format
    - export_patterns_to_csv: Export patterns to CSV format
    - export_full_report_html: Export full HTML report

Visualization Tabs:
    - render_statistical_validation_tab: Statistical validation visualizations
    - render_worst_trades_tab: Worst trades analysis visualizations
    - render_shap_analysis_tab: SHAP analysis visualizations
    - render_patterns_tab: Error pattern visualizations

Example:
    >>> from ml4t.diagnostic.visualization.trade_shap import run_diagnostics_dashboard
    >>> run_diagnostics_dashboard(result)

Note:
    All functions in this module require Streamlit to be installed. Functions are
    lazy-loaded to avoid slow import time when Streamlit is not needed.
"""

# Lazy loading to avoid Streamlit import overhead (~1.3s) at module load time
# Functions are imported on first access
_dashboard_module = None

# All public names for the module
_LAZY_IMPORTS = {
    "cached_extract_trade_data",
    "cached_extract_trade_returns",
    "display_data_summary",
    "export_full_report_html",
    "export_patterns_to_csv",
    "export_trades_to_csv",
    "extract_trade_data",
    "extract_trade_returns",
    "load_data_from_file",
    "render_patterns_tab",
    "render_shap_analysis_tab",
    "render_statistical_validation_tab",
    "render_worst_trades_tab",
    "run_diagnostics_dashboard",
    "run_polished_dashboard",
    "serialize_result_for_cache",
}


def __getattr__(name: str):
    """Lazy load dashboard functions to avoid importing Streamlit at module load."""
    global _dashboard_module

    if name in _LAZY_IMPORTS:
        if _dashboard_module is None:
            from ml4t.diagnostic.evaluation import trade_shap_dashboard as _mod

            _dashboard_module = _mod
        return getattr(_dashboard_module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main dashboard functions
    "run_diagnostics_dashboard",
    "run_polished_dashboard",
    # Data extraction
    "extract_trade_returns",
    "extract_trade_data",
    "load_data_from_file",
    "display_data_summary",
    # Visualization tabs
    "render_statistical_validation_tab",
    "render_worst_trades_tab",
    "render_shap_analysis_tab",
    "render_patterns_tab",
    # Export functions
    "export_trades_to_csv",
    "export_patterns_to_csv",
    "export_full_report_html",
    # Caching utilities
    "serialize_result_for_cache",
    "cached_extract_trade_returns",
    "cached_extract_trade_data",
]
