"""Dashboard application orchestrator.

Main entry point for the Trade-SHAP diagnostics dashboard.
Handles page configuration, data loading, and tab routing.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult

# Module-level load time tracking for styled mode
_LOAD_START_TIME: float | None = None


def _measure_load_time_start() -> None:
    """Start measuring load time."""
    global _LOAD_START_TIME
    _LOAD_START_TIME = time.time()


def _measure_load_time_end() -> float:
    """End load time measurement and return elapsed seconds."""
    global _LOAD_START_TIME
    if _LOAD_START_TIME is None:
        return 0.0
    elapsed = time.time() - _LOAD_START_TIME
    _LOAD_START_TIME = None
    return elapsed


def run_dashboard(
    result: TradeShapResult | dict[str, Any] | None = None,
    title: str = "Trade-SHAP Diagnostics Dashboard",
    styled: bool = False,
    allow_pickle_upload: bool = False,
) -> None:
    """Run the Streamlit diagnostics dashboard.

    This is the main entry point for the dashboard. It can be called
    programmatically with a result object, or run as a standalone app
    that allows file uploads.

    Parameters
    ----------
    result : TradeShapResult or dict, optional
        Pre-loaded analysis result. If None, dashboard will show file upload.
    title : str, default "Trade-SHAP Diagnostics Dashboard"
        Dashboard title.
    styled : bool, default False
        Enable professional styling with custom CSS, load time tracking,
        spinners, export buttons, and enhanced error handling.
    allow_pickle_upload : bool, default False
        Allow pickle file uploads. SECURITY WARNING: Pickle files can
        execute arbitrary code. Only enable for trusted sources.

    Examples
    --------
    Programmatic usage:
        >>> from ml4t.diagnostic.evaluation import TradeShapAnalyzer
        >>> from ml4t.diagnostic.evaluation.trade_dashboard import run_dashboard
        >>>
        >>> analyzer = TradeShapAnalyzer(model, features_df, shap_values)
        >>> result = analyzer.explain_worst_trades(worst_trades)
        >>> run_dashboard(result)

    Styled mode with professional theme:
        >>> run_dashboard(result, styled=True)

    Standalone app:
        $ streamlit run -m ml4t.diagnostic.evaluation.trade_dashboard.app
    """
    # Lazy import streamlit
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "streamlit is required for dashboard functionality. Install with: pip install streamlit"
        ) from None

    from ml4t.diagnostic.evaluation.trade_dashboard.io import (
        PickleDisabledError,
        load_result_from_upload,
    )
    from ml4t.diagnostic.evaluation.trade_dashboard.normalize import normalize_result
    from ml4t.diagnostic.evaluation.trade_dashboard.style import STYLED_CSS
    from ml4t.diagnostic.evaluation.trade_dashboard.tabs import (
        patterns,
        shap_analysis,
        stat_validation,
        worst_trades,
    )
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardConfig

    # Start load time measurement in styled mode
    if styled:
        _measure_load_time_start()

    # Page config
    page_config: dict[str, Any] = {
        "page_title": title,
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }
    if styled:
        page_config["menu_items"] = {
            "Get Help": "https://github.com/ml4t/ml4t-diagnostic",
            "Report a bug": "https://github.com/ml4t/ml4t-diagnostic/issues",
            "About": "# Trade-SHAP Diagnostics\nSystematic trade debugging for ML strategies",
        }
    st.set_page_config(**page_config)

    # Apply professional CSS in styled mode
    if styled:
        st.markdown(STYLED_CSS, unsafe_allow_html=True)

    # Title and description
    st.title(title)
    st.markdown(
        """
        **Systematic trade debugging and continuous improvement for ML trading strategies**

        This dashboard visualizes Trade-SHAP analysis results to help you:
        - Identify why specific trades failed
        - Discover recurring error patterns
        - Get actionable recommendations for improvement
        """
    )
    st.divider()

    # Sidebar for data loading
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Data Loading")

        if result is None:
            # File upload mode
            file_types = ["json"]
            if allow_pickle_upload:
                file_types.extend(["pkl", "pickle"])

            uploaded_file = st.file_uploader(
                "Upload TradeShapResult",
                type=file_types,
                help="Upload a JSON file containing TradeShapResult"
                + (" (or pickle if enabled)" if allow_pickle_upload else ""),
            )

            if uploaded_file is not None:
                try:
                    with st.spinner("Loading data..."):
                        result = load_result_from_upload(
                            uploaded_file,
                            allow_pickle=allow_pickle_upload,
                        )
                    st.success("Data loaded successfully!")
                except PickleDisabledError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.error(f"Failed to load data: {e}")
                    if styled:
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                    return
            else:
                st.info("Upload a file to get started")
                if not allow_pickle_upload:
                    st.caption(
                        "Pickle uploads disabled for security. Use JSON format for data transfer."
                    )
                st.stop()
        else:
            st.success("Data loaded programmatically")

    # Normalize result once for all tabs
    if result is not None:
        config = DashboardConfig(
            allow_pickle_upload=allow_pickle_upload,
            styled=styled,
            title=title,
        )
        bundle = normalize_result(result, config)

        # Display data summary in sidebar
        _display_sidebar_summary(st, bundle, styled)

        # Main content - tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Statistical Validation",
                "Worst Trades",
                "SHAP Analysis",
                "Patterns",
            ]
        )

        if styled:
            with tab1, st.spinner("Loading statistical validation..."):
                stat_validation.render_tab(st, bundle)

            with tab2, st.spinner("Loading worst trades..."):
                worst_trades.render_tab(st, bundle)

            with tab3, st.spinner("Loading SHAP analysis..."):
                shap_analysis.render_tab(st, bundle)

            with tab4, st.spinner("Loading error patterns..."):
                patterns.render_tab(st, bundle)

            # Show load time
            load_time = _measure_load_time_end()
            if load_time > 0:
                if load_time < 5.0:
                    st.sidebar.success(f"Loaded in {load_time:.2f}s")
                else:
                    st.sidebar.warning(f"Loaded in {load_time:.2f}s (>5s target)")
        else:
            with tab1:
                stat_validation.render_tab(st, bundle)

            with tab2:
                worst_trades.render_tab(st, bundle)

            with tab3:
                shap_analysis.render_tab(st, bundle)

            with tab4:
                patterns.render_tab(st, bundle)
    else:
        st.info("Please load data from the sidebar to begin analysis.")


def _display_sidebar_summary(st: Any, bundle: Any, styled: bool) -> None:
    """Display data summary in sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("Data Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trades Analyzed", bundle.n_trades_analyzed)
        with col2:
            st.metric("Trades Explained", bundle.n_trades_explained)

        if bundle.n_trades_failed > 0:
            st.warning(f"{bundle.n_trades_failed} trades failed explanation")

        n_patterns = len(bundle.patterns_df) if not bundle.patterns_df.empty else 0
        st.metric("Patterns Found", n_patterns)

        # Export buttons (styled mode)
        if styled:
            st.divider()
            st.subheader("Export")
            _render_export_buttons(st, bundle)

        # Display options
        st.divider()
        st.subheader("Display Options")
        st.checkbox("Show timestamps", value=True, key="show_timestamps")
        st.checkbox("Show confidence scores", value=True, key="show_confidence")

        # Footer
        st.divider()
        st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if styled:
            st.caption("Dashboard v2.0 (Modular)")


def _render_export_buttons(st: Any, bundle: Any) -> None:
    """Render export buttons in sidebar."""
    try:
        from ml4t.diagnostic.evaluation.trade_dashboard.export import (
            export_html_report,
            export_trades_csv,
        )

        # CSV export
        csv_data = export_trades_csv(bundle)
        if csv_data:
            st.download_button(
                label="Download Trades CSV",
                data=csv_data,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # HTML report export
        html_data = export_html_report(bundle)
        if html_data:
            st.download_button(
                label="Download HTML Report",
                data=html_data,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
            )
    except ImportError:
        st.caption("Export modules not available")
    except Exception as e:
        st.error(f"Export error: {e}")


# Allow running as a standalone Streamlit app
if __name__ == "__main__":
    run_dashboard()
