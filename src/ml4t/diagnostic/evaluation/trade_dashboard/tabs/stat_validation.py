"""Statistical Validation tab.

Displays PSR (Probabilistic Sharpe Ratio), distribution tests, and time-series tests.
Uses PSR instead of DSR because this dashboard analyzes a single strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def render_tab(st: Any, bundle: DashboardBundle) -> None:
    """Render the Statistical Validation tab.

    Parameters
    ----------
    st : streamlit
        Streamlit module instance.
    bundle : DashboardBundle
        Normalized dashboard data.
    """
    from ml4t.diagnostic.evaluation.trade_dashboard.stats import (
        compute_distribution_tests,
        compute_return_summary,
        compute_time_series_tests,
        probabilistic_sharpe_ratio,
    )

    st.header("Statistical Validation")

    st.info(
        "Statistical validation ensures that identified patterns are "
        "statistically significant and not due to random chance."
    )

    # Check if we have returns data
    if bundle.returns is None or len(bundle.returns) == 0:
        st.warning(
            "No trade returns available for statistical analysis. "
            "Ensure trade_metrics are attached to explanations."
        )
        return

    returns = bundle.returns
    summary = compute_return_summary(returns)

    # Show warning if using PnL instead of return_pct
    if bundle.returns_label == "pnl":
        st.caption(
            "Using PnL (dollar amounts) instead of normalized returns. "
            "Sharpe ratio interpretation is limited."
        )

    # PSR section (replaces incorrect DSR usage)
    st.subheader("Probabilistic Sharpe Ratio (PSR)")

    st.markdown(
        """
        **What is PSR?**
        The Probabilistic Sharpe Ratio (PSR) gives the probability that the true
        Sharpe ratio exceeds a benchmark (typically 0), accounting for sample size
        and return distribution characteristics.

        *Note: DSR (Deflated Sharpe Ratio) was previously shown here but is not
        applicable to single-strategy analysis. DSR requires K independent strategies
        to compute the variance across trials.*

        **Reference:** Bailey & Lopez de Prado (2012). "The Sharpe Ratio Efficient Frontier"
        """
    )

    # Calculate PSR
    psr_result = probabilistic_sharpe_ratio(
        observed_sharpe=summary.sharpe,
        benchmark_sharpe=0.0,
        n_samples=summary.n_samples,
        skewness=summary.skewness,
        kurtosis=summary.kurtosis,
        return_components=True,
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sharpe Ratio",
            f"{summary.sharpe:.3f}",
            help="Observed Sharpe ratio (mean / std)",
        )

    with col2:
        st.metric(
            "PSR (vs SR=0)",
            f"{psr_result['psr']:.3f}",
            help="Probability that true SR > 0",
        )

    with col3:
        p_value = 1 - psr_result["psr"]
        st.metric(
            "P-Value",
            f"{p_value:.4f}",
            help="1 - PSR: probability true SR <= 0",
        )

    with col4:
        st.metric("N Trades", summary.n_samples, help="Number of trades analyzed")

    # Interpretation
    psr = psr_result["psr"]
    if psr >= 0.99:
        st.success(f"Strong evidence SR > 0 (PSR = {psr:.3f} >= 0.99)")
    elif psr >= 0.95:
        st.success(f"Significant performance (PSR = {psr:.3f} >= 0.95)")
    elif psr >= 0.90:
        st.warning(f"Marginally significant (PSR = {psr:.3f} >= 0.90)")
    elif psr >= 0.50:
        st.warning(f"Weak evidence SR > 0 (PSR = {psr:.3f})")
    else:
        st.error(f"Evidence suggests SR <= 0 (PSR = {psr:.3f} < 0.50)")

    # Return statistics
    st.divider()
    st.subheader("Return Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{summary.mean:.4f}")

    with col2:
        st.metric("Std Dev", f"{summary.std:.4f}")

    with col3:
        st.metric("Win Rate", f"{summary.win_rate:.1%}")

    with col4:
        st.metric("Skewness", f"{summary.skewness:.3f}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Kurtosis", f"{summary.kurtosis:.3f}")

    with col2:
        st.metric("Min", f"{summary.min_val:.4f}")

    with col3:
        st.metric("Max", f"{summary.max_val:.4f}")

    with col4:
        pass  # Empty column for alignment

    # Distribution tests
    st.divider()
    st.subheader("Distribution Tests")

    dist_tests = compute_distribution_tests(returns)
    if not dist_tests.empty:
        st.dataframe(
            dist_tests,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Insufficient data for distribution tests.")

    # Time-series tests
    st.divider()
    st.subheader("Time-Series Tests")

    st.caption("These tests require chronologically ordered data. Trades are sorted by entry_time.")

    ts_tests = compute_time_series_tests(returns)
    if not ts_tests.empty:
        st.dataframe(
            ts_tests,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Insufficient data for time-series tests (need 20+ observations).")
