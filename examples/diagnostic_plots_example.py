"""Example usage of interactive diagnostic visualization functions.

This script demonstrates how to use the Plotly-based diagnostic plotting functions
for feature analysis. All visualizations are interactive (zoom, hover, pan) and can
be displayed in browsers or dashboards.

Examples:
    Run this script to generate interactive diagnostic plots:

        $ python examples/diagnostic_plots_example.py

    The plots will open in your browser automatically.

    For dashboard usage:

        >>> import streamlit as st
        >>> from ml4t.diagnostic.evaluation.diagnostic_plots import plot_acf_pacf
        >>> import numpy as np
        >>> data = np.random.randn(1000)
        >>> fig = plot_acf_pacf(data, max_lags=40)
        >>> st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.diagnostic_plots import (
    export_static,
    get_figure_data,
    plot_acf_pacf,
    plot_distribution,
    plot_qq,
    plot_volatility_clustering,
)


def example_1_white_noise():
    """Example 1: Interactive diagnostic plots for white noise (ideal case).

    White noise should show:
    - No significant autocorrelation (except at lag 0)
    - Normal distribution
    - QQ plot on diagonal
    - No volatility clustering

    All plots are interactive - try zooming and hovering!
    """
    print("=" * 80)
    print("Example 1: White Noise Analysis (Interactive)")
    print("=" * 80)

    # Generate white noise
    np.random.seed(42)
    data = np.random.randn(1000)

    # ACF/PACF plot - interactive!
    print("\n1. Creating interactive ACF/PACF plot...")
    fig_acf = plot_acf_pacf(data, max_lags=40, title="White Noise: ACF and PACF")
    fig_acf.show()  # Opens in browser
    print("   ✓ Opened in browser - zoom and hover to explore!")

    # QQ plot
    print("\n2. Creating interactive QQ plot...")
    fig_qq = plot_qq(data, title="White Noise: QQ Plot vs Normal")
    fig_qq.show()
    print("   ✓ Opened in browser")

    # Volatility clustering
    print("\n3. Creating interactive volatility clustering plot...")
    fig_vol = plot_volatility_clustering(data, window=20, title="White Noise: Volatility Analysis")
    fig_vol.show()
    print("   ✓ Opened in browser - linked x-axes for easy comparison")

    # Distribution
    print("\n4. Creating interactive distribution plot...")
    fig_dist = plot_distribution(data, bins=50, fit_normal=True, title="White Noise: Distribution")
    fig_dist.show()
    print("   ✓ Opened in browser - hover for exact values")

    # Optional: Export static image for presentation
    print("\n5. (Optional) Exporting static PNG for presentation...")
    export_static(fig_acf, "white_noise_acf_pacf", format="png", width=1200, height=400)

    # Optional: Extract data for custom analysis
    print("\n6. (Optional) Extracting data for custom analysis...")
    df = get_figure_data(fig_acf)
    print(f"   ✓ Extracted {len(df.columns)} data columns")
    print(f"   Columns: {list(df.columns)[:3]}...")

    print("\n✓ White noise analysis complete - check your browser!\n")


def example_2_garch_process():
    """Example 2: Interactive diagnostic plots for GARCH process.

    GARCH should show:
    - Little autocorrelation in returns
    - Significant autocorrelation in squared returns
    - Clear volatility clustering
    - Normal or heavy-tailed distribution
    """
    print("=" * 80)
    print("Example 2: GARCH(1,1) Process Analysis (Interactive)")
    print("=" * 80)

    # Generate GARCH(1,1) process
    np.random.seed(42)
    n = 2000
    returns = np.zeros(n)
    sigma = np.zeros(n)
    sigma[0] = 0.1

    for t in range(1, n):
        sigma[t] = np.sqrt(0.01 + 0.05 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2)
        returns[t] = sigma[t] * np.random.randn()

    # Create pandas Series with datetime index for better x-axis
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    returns_series = pd.Series(returns, index=dates, name="GARCH Returns")

    print("\n1. Creating ACF/PACF for returns...")
    fig_acf = plot_acf_pacf(returns_series, max_lags=40, title="GARCH Returns: ACF and PACF")
    fig_acf.show()
    print("   ✓ Little autocorrelation in returns")

    print("\n2. Creating ACF/PACF for squared returns...")
    squared_returns = returns_series**2
    fig_acf_sq = plot_acf_pacf(
        squared_returns, max_lags=40, title="GARCH Squared Returns: ACF and PACF"
    )
    fig_acf_sq.show()
    print("   ✓ Significant autocorrelation indicates volatility clustering!")

    print("\n3. Creating volatility clustering visualization...")
    fig_vol = plot_volatility_clustering(
        returns_series, window=20, title="GARCH Process: Volatility Clustering"
    )
    fig_vol.show()
    print("   ✓ Clear periods of high and low volatility - zoom to see details")

    print("\n4. Creating distribution plot...")
    fig_dist = plot_distribution(
        returns_series, bins=50, fit_normal=True, title="GARCH Returns: Distribution"
    )
    fig_dist.show()
    print("   ✓ Check moments statistics - hover over annotation")

    print("\n✓ GARCH process analysis complete\n")


def example_3_real_world_returns():
    """Example 3: Comprehensive diagnostic suite for realistic stock returns.

    Simulates realistic features:
    - Small positive drift
    - Volatility clustering (GARCH)
    - Heavy tails (t-distribution)

    Demonstrates full interactive workflow.
    """
    print("=" * 80)
    print("Example 3: Realistic Stock Returns Analysis (Full Interactive Suite)")
    print("=" * 80)

    # Generate realistic returns
    np.random.seed(42)
    n = 2000
    returns = np.zeros(n)
    sigma = np.zeros(n)
    sigma[0] = 0.02  # 2% daily volatility

    for t in range(1, n):
        # GARCH(1,1) volatility
        sigma[t] = np.sqrt(0.0001 + 0.08 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2)
        # Heavy tails via Student's t
        returns[t] = sigma[t] * np.random.standard_t(df=5)

    # Add small drift (~5% annualized)
    returns += 0.0002

    # Create pandas Series with datetime index
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    returns_series = pd.Series(returns, index=dates, name="Daily Returns")

    print(f"\nGenerated {n} days of returns")
    print(f"Mean: {returns.mean():.4f} ({returns.mean() * 252:.2%} annualized)")
    print(f"Volatility: {returns.std():.4f} ({returns.std() * np.sqrt(252):.2%} annualized)")

    print("\n1. ACF/PACF Analysis...")
    fig_acf = plot_acf_pacf(returns_series, max_lags=40, title="Stock Returns: ACF and PACF")
    fig_acf.show()

    print("\n2. Volatility Clustering Analysis...")
    fig_vol = plot_volatility_clustering(
        returns_series, window=20, title="Stock Returns: Volatility Clustering"
    )
    fig_vol.show()
    print("   ✓ Zoom into volatility episodes - drag to select time range")

    print("\n3. QQ Plot (Normality Assessment)...")
    fig_qq = plot_qq(returns_series, distribution="norm", title="Stock Returns: QQ Plot")
    fig_qq.show()
    print("   ✓ S-curve indicates heavy tails - zoom on extremes")

    print("\n4. Distribution with Multiple Fits...")
    fig_dist = plot_distribution(
        returns_series,
        bins=50,
        fit_normal=True,
        fit_t=True,
        title="Stock Returns: Distribution",
    )
    fig_dist.show()
    print("   ✓ Toggle legend to compare Normal vs Student's t fit")

    print("\n✓ Full diagnostic suite complete!")
    print("\nInteractive features demonstrated:")
    print("  • Zoom: Double-click on axis, or drag to select")
    print("  • Pan: Click and drag on chart")
    print("  • Hover: Detailed tooltips with exact values")
    print("  • Toggle: Click legend items to show/hide traces")
    print("  • Reset: Double-click anywhere to reset zoom")

    print("\n💡 Tip: All these plots work in dashboards too!")
    print("   See docstrings for Streamlit/Dash examples\n")


def example_4_dashboard_ready():
    """Example 4: Demonstrate dashboard-ready features.

    Shows how to prepare plots for use in Streamlit/Dash dashboards.
    """
    print("=" * 80)
    print("Example 4: Dashboard-Ready Features")
    print("=" * 80)

    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(1000)

    print("\n1. Create plot with custom dimensions for dashboard...")
    fig = plot_acf_pacf(data, max_lags=40, height=400)
    print(f"   ✓ Figure height: {fig.layout.height}px (responsive width)")

    print("\n2. Extract data for further analysis...")
    df = get_figure_data(fig)
    print(f"   ✓ Extracted {len(df)} data points")
    print(f"   ✓ Available columns: {list(df.columns)}")

    print("\n3. Save for presentations (optional)...")
    export_static(fig, "dashboard_plot", format="png", width=1200, height=400, scale=2)
    print("   ✓ High-res PNG saved")

    print("\n4. Dashboard integration example:")
    print("   ```python")
    print("   import streamlit as st")
    print("   from ml4t.diagnostic.evaluation import plot_acf_pacf")
    print("")
    print("   # In your Streamlit app:")
    print("   st.title('Feature Diagnostics Dashboard')")
    print("   ")
    print("   # User controls")
    print("   max_lags = st.slider('Max Lags', 10, 100, 40)")
    print("   alpha = st.selectbox('Significance', [0.01, 0.05, 0.10])")
    print("   ")
    print("   # Create interactive plot")
    print("   fig = plot_acf_pacf(data, max_lags=max_lags, alpha=alpha)")
    print("   st.plotly_chart(fig, use_container_width=True)")
    print("   ")
    print("   # Export underlying data")
    print("   if st.button('Download Data'):")
    print("       df = get_figure_data(fig)")
    print("       st.download_button('CSV', df.to_csv(), 'acf_data.csv')")
    print("   ```")

    print("\n✓ Dashboard features demonstrated\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Interactive Diagnostic Plots Examples")
    print("=" * 80)
    print("\nThis script demonstrates interactive Plotly-based visualizations.")
    print("All plots will open in your browser automatically.")
    print("\nPress Ctrl+C to skip examples and exit.\n")

    try:
        # Run examples
        example_1_white_noise()
        input("Press Enter to continue to next example...")

        example_2_garch_process()
        input("Press Enter to continue to next example...")

        example_3_real_world_returns()
        input("Press Enter to continue to next example...")

        example_4_dashboard_ready()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")

    print("=" * 80)
    print("All examples complete!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  • All plots are interactive (zoom, hover, pan)")
    print("  • Works in browsers, Jupyter, and dashboards (Streamlit/Dash)")
    print("  • Export to PNG/PDF/SVG for presentations")
    print("  • Extract underlying data as DataFrames")
    print("  • Consistent with ml4t.diagnostic.evaluation.viz (Plotly-based)")
    print("\n")


if __name__ == "__main__":
    main()
