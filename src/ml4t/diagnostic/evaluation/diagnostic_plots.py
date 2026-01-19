"""Interactive diagnostic visualizations for feature analysis.

This module provides Plotly-based interactive diagnostic plots for the Feature
Diagnostics framework (Module A), including:

- ACF/PACF plots with confidence bands
- QQ plots for normality assessment
- Volatility clustering visualizations
- Distribution analysis with fitted curves

All visualizations are interactive (zoom, hover, pan) and designed for
browser-based dashboards. Static exports (PNG, PDF) are available via
the export_static() function.

References
----------
.. [1] Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.
.. [2] Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
.. [3] Tsay, R. S. (2005). Analysis of Financial Time Series. Wiley.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Color scheme for financial diagnostics (consistent with viz.py)
COLORS = {
    "primary": "#3366CC",  # Blue
    "secondary": "#FF9900",  # Orange
    "positive": "#00CC88",  # Green
    "negative": "#FF4444",  # Red
    "neutral": "#888888",  # Gray
    "confidence": "rgba(255, 68, 68, 0.2)",  # Light red fill
}


def plot_acf_pacf(
    data: "NDArray | pd.Series",
    max_lags: int = 40,
    alpha: float = 0.05,
    title: str | None = None,
    height: int = 400,
) -> go.Figure:
    """Create interactive ACF and PACF plots with confidence bands.

    Creates a two-panel interactive figure showing:
    1. Autocorrelation Function (ACF) - correlation with lagged values
    2. Partial Autocorrelation Function (PACF) - correlation controlling for intermediate lags

    Includes confidence bands based on the specified significance level (alpha).
    Hover over bars to see exact values. Zoom and pan for detailed exploration.

    Parameters
    ----------
    data : ndarray or pd.Series
        Time series data to analyze
    max_lags : int, default 40
        Maximum number of lags to display
    alpha : float, default 0.05
        Significance level for confidence bands (default: 95% confidence)
    title : str, optional
        Figure title. If None, uses "ACF and PACF Analysis"
    height : int, default 400
        Figure height in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure with ACF and PACF plots

    Examples
    --------
    >>> import numpy as np
    >>> # AR(1) process
    >>> data = np.random.randn(1000)
    >>> for i in range(1, len(data)):
    ...     data[i] = 0.7 * data[i-1] + np.random.randn()
    >>> fig = plot_acf_pacf(data)
    >>> fig.show()  # Opens in browser
    >>> # Or in dashboard:
    >>> import streamlit as st
    >>> st.plotly_chart(fig)

    Notes
    -----
    The confidence bands are computed as ±z * sqrt(1/n) where z is the
    critical value for the specified alpha level and n is the sample size.
    This assumes the series is white noise under the null hypothesis.

    For ACF, significant lags indicate autocorrelation that may violate
    assumptions of many statistical tests.

    For PACF, the number of significant lags helps identify AR order:
    - PACF cuts off after lag p → AR(p) process
    - ACF cuts off after lag q → MA(q) process
    - Both decay gradually → ARMA process

    See Also
    --------
    ml4t-diagnostic.evaluation.autocorrelation : Statistical autocorrelation tests
    statsmodels.graphics.tsaplots : Alternative ACF/PACF plotting

    References
    ----------
    .. [1] Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis:
           Forecasting and Control.
    """
    # Convert to numpy array if pandas Series
    data_array: NDArray = data.to_numpy() if isinstance(data, pd.Series) else data

    # Remove NaN values
    data_array = data_array[~np.isnan(data_array)]

    if len(data_array) == 0:
        raise ValueError("Input data is empty after removing NaN values")

    n = len(data_array)
    if max_lags >= n:
        max_lags = n - 1

    # Compute ACF and PACF
    acf_values = _compute_acf(data_array, max_lags)
    pacf_values = _compute_pacf(data_array, max_lags)

    # Compute confidence bands
    z_crit = stats.norm.ppf(1 - alpha / 2)
    conf_level = z_crit / np.sqrt(n)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"ACF ({100 * (1 - alpha):.0f}% Confidence Band)",
            f"PACF ({100 * (1 - alpha):.0f}% Confidence Band)",
        ),
        horizontal_spacing=0.12,
    )

    lags = np.arange(max_lags + 1)

    # Plot ACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            marker_color=COLORS["primary"],
            name="ACF",
            hovertemplate="Lag: %{x}<br>ACF: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add ACF confidence bands
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[conf_level] * len(lags),
            mode="lines",
            line={"color": COLORS["negative"], "dash": "dash", "width": 1},
            name="Confidence Band",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[-conf_level] * len(lags),
            mode="lines",
            line={"color": COLORS["negative"], "dash": "dash", "width": 1},
            fill="tonexty",
            fillcolor=COLORS["confidence"],
            name="Confidence Band",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Plot PACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=pacf_values,
            marker_color=COLORS["secondary"],
            name="PACF",
            hovertemplate="Lag: %{x}<br>PACF: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add PACF confidence bands
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[conf_level] * len(lags),
            mode="lines",
            line={"color": COLORS["negative"], "dash": "dash", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=[-conf_level] * len(lags),
            mode="lines",
            line={"color": COLORS["negative"], "dash": "dash", "width": 1},
            fill="tonexty",
            fillcolor=COLORS["confidence"],
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Add zero lines
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=2)

    # Update layout
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
    fig.update_yaxes(title_text="Partial Autocorrelation", row=1, col=2)

    if title is None:
        title = "ACF and PACF Analysis"

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=height,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def plot_qq(
    data: "NDArray | pd.Series",
    distribution: str = "norm",
    title: str | None = None,
    height: int = 500,
    width: int = 500,
) -> go.Figure:
    """Create interactive QQ plot for assessing distributional assumptions.

    A Quantile-Quantile (QQ) plot compares the quantiles of the data against
    the quantiles of a theoretical distribution. Points falling along the
    diagonal line indicate the data follows the theoretical distribution.

    Interactive features: hover for exact values, zoom to focus on tails.

    Deviations from the diagonal indicate departures from the assumed distribution:
    - S-shaped curve: Heavy tails (fat-tailed distribution)
    - Inverted S: Light tails (thin-tailed distribution)
    - Points above/below line at extremes: Asymmetric tails

    Parameters
    ----------
    data : ndarray or pd.Series
        Data to assess
    distribution : str, default "norm"
        Theoretical distribution to compare against.
        Options: "norm" (normal), "t" (Student's t), "uniform"
    title : str, optional
        Plot title. If None, uses "QQ Plot vs {distribution}"
    height : int, default 500
        Figure height in pixels
    width : int, default 500
        Figure width in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure with QQ plot

    Examples
    --------
    >>> import numpy as np
    >>> # Normal data
    >>> data = np.random.randn(1000)
    >>> fig = plot_qq(data)
    >>> fig.show()

    >>> # Heavy-tailed data
    >>> data = np.random.standard_t(df=3, size=1000)
    >>> fig = plot_qq(data, distribution='t')
    >>> fig.show()

    Notes
    -----
    The QQ plot is a graphical complement to normality tests like Jarque-Bera
    or Shapiro-Wilk. It provides visual insight into *how* the data deviates
    from normality, not just whether it does.

    Common patterns:
    - Normal: Points on diagonal
    - Skewed: Curved pattern
    - Heavy-tailed: Points diverge at extremes (S-curve)
    - Light-tailed: Points converge at extremes (inverted S)

    For financial returns, heavy tails (leptokurtosis) are common, so observing
    departures at the extremes is typical.

    See Also
    --------
    ml4t-diagnostic.evaluation.distribution : Distribution diagnostic tests
    scipy.stats.probplot : Underlying QQ plot function

    References
    ----------
    .. [1] Wilk, M. B., & Gnanadesikan, R. (1968). "Probability plotting
           methods for the analysis of data." Biometrika, 55(1), 1-17.
    """
    # Convert to numpy array if pandas Series
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    # Remove NaN values
    data = data[~np.isnan(data)]

    if len(data) == 0:
        raise ValueError("Input data is empty after removing NaN values")

    # Generate QQ plot data based on distribution
    if distribution == "norm":
        (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(data, dist="norm")
        dist_name = "Normal"
    elif distribution == "t":
        # Estimate degrees of freedom
        params = stats.t.fit(data)
        df = params[0]
        (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(
            data, dist=stats.t, sparams=(df,)
        )
        dist_name = f"Student's t (df={df:.1f})"
    elif distribution == "uniform":
        (theoretical_q, sample_q), (slope, intercept, r) = stats.probplot(data, dist="uniform")
        dist_name = "Uniform"
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'norm', 't', or 'uniform'")

    # Create figure
    fig = go.Figure()

    # Add sample points
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=sample_q,
            mode="markers",
            marker={"color": COLORS["primary"], "size": 5, "opacity": 0.6},
            name="Sample Data",
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
        )
    )

    # Add reference line
    fitted_line = slope * theoretical_q + intercept
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=fitted_line,
            mode="lines",
            line={"color": COLORS["negative"], "dash": "dash", "width": 2},
            name="Reference Line",
            hovertemplate="Theoretical: %{x:.3f}<br>Expected: %{y:.3f}<extra></extra>",
        )
    )

    # Update layout
    if title is None:
        title = f"QQ Plot vs {dist_name} Distribution"

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=height,
        width=width,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
    )

    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add annotation for interpretation
    fig.add_annotation(
        text=(
            "Points on diagonal → data follows distribution<br>S-curve → heavy tails<br>Inverted S → light tails"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.10,
        showarrow=False,
        bgcolor="rgba(255, 248, 220, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        font={"size": 9},
        align="left",
    )

    return fig


def plot_volatility_clustering(
    data: "NDArray | pd.Series",
    window: int = 20,
    title: str | None = None,
    height: int = 800,
) -> go.Figure:
    """Create interactive volatility clustering visualization.

    Volatility clustering is a common feature in financial time series where
    large changes tend to be followed by large changes (of either sign), and
    small changes tend to be followed by small changes.

    This creates a 4-panel interactive figure showing:
    1. Original returns series
    2. Absolute returns (magnitude of changes)
    3. Squared returns (volatility proxy)
    4. Rolling volatility (rolling standard deviation)

    Hover for exact values, zoom to focus on volatility episodes, linked x-axes.

    Parameters
    ----------
    data : ndarray or pd.Series
        Time series data (typically returns)
    window : int, default 20
        Rolling window size for volatility calculation
    title : str, optional
        Figure title. If None, uses "Volatility Clustering Analysis"
    height : int, default 800
        Figure height in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure with 4-panel volatility analysis

    Examples
    --------
    >>> import numpy as np
    >>> # GARCH-like data
    >>> n = 1000
    >>> returns = np.zeros(n)
    >>> sigma = np.zeros(n)
    >>> sigma[0] = 0.1
    >>> for t in range(1, n):
    ...     sigma[t] = np.sqrt(0.01 + 0.05 * returns[t-1]**2 + 0.9 * sigma[t-1]**2)
    ...     returns[t] = sigma[t] * np.random.randn()
    >>> fig = plot_volatility_clustering(returns)
    >>> fig.show()

    Notes
    -----
    Volatility clustering violates the constant variance (homoscedasticity)
    assumption of many statistical models. If present, consider:
    - GARCH models for volatility forecasting
    - Robust standard errors in regressions
    - Volatility-adjusted metrics

    Visual signs of clustering:
    - Periods of high/low volatility in returns plot
    - Autocorrelation in squared returns (clustering persists)
    - Time-varying rolling volatility

    See Also
    --------
    ml4t-diagnostic.evaluation.volatility : ARCH/GARCH tests for volatility clustering

    References
    ----------
    .. [1] Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity
           with Estimates of the Variance of United Kingdom Inflation."
           Econometrica, 50(4), 987-1007.
    .. [2] Bollerslev, T. (1986). "Generalized autoregressive conditional
           heteroskedasticity." Journal of Econometrics, 31(3), 307-327.
    """
    # Convert to numpy array if pandas Series
    original_index: pd.Index | None
    if isinstance(data, pd.Series):
        original_index = data.index
        data_values = data.to_numpy()
    else:
        original_index = None
        data_values = data

    # Remove NaN values
    valid_idx = ~np.isnan(data_values)
    data_values = data_values[valid_idx]

    if len(data_values) == 0:
        raise ValueError("Input data is empty after removing NaN values")

    # Create time index - either filtered original index or sequential integers
    time_index: NDArray = (
        original_index[valid_idx].to_numpy()
        if original_index is not None
        else np.arange(len(data_values))
    )

    # Compute volatility measures
    abs_returns = np.abs(data_values)
    squared_returns = data_values**2
    rolling_vol = pd.Series(data_values).rolling(window=window, min_periods=1).std().values

    # Create 4-panel figure
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Returns Series",
            "Absolute Returns (Magnitude)",
            "Squared Returns (Volatility Proxy)",
            f"Rolling Volatility (window={window})",
        ),
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # 1. Original returns
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=data_values,
            mode="lines",
            line={"color": COLORS["primary"], "width": 0.8},
            name="Returns",
            hovertemplate="Time: %{x}<br>Return: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=1)

    # 2. Absolute returns
    mean_abs = np.mean(abs_returns)
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=abs_returns,
            mode="lines",
            line={"color": COLORS["secondary"], "width": 0.8},
            name="Absolute Returns",
            hovertemplate="Time: %{x}<br>|Return|: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=mean_abs,
        line_color=COLORS["negative"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"Mean: {mean_abs:.4f}",
        annotation_position="right",
        row=2,
        col=1,
    )

    # 3. Squared returns
    mean_sq = np.mean(squared_returns)
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=squared_returns,
            mode="lines",
            line={"color": COLORS["positive"], "width": 0.8},
            name="Squared Returns",
            hovertemplate="Time: %{x}<br>Return²: %{y:.6f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=mean_sq,
        line_color=COLORS["negative"],
        line_dash="dash",
        line_width=1.5,
        annotation_text=f"Mean: {mean_sq:.6f}",
        annotation_position="right",
        row=3,
        col=1,
    )

    # 4. Rolling volatility
    fig.add_trace(
        go.Scatter(
            x=time_index,
            y=rolling_vol,
            mode="lines",
            line={"color": COLORS["negative"], "width": 1.2},
            fill="tozeroy",
            fillcolor="rgba(255, 68, 68, 0.2)",
            name="Rolling Volatility",
            hovertemplate="Time: %{x}<br>Volatility: %{y:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    # Update axes
    fig.update_yaxes(title_text="Returns", row=1, col=1)
    fig.update_yaxes(title_text="|Returns|", row=2, col=1)
    fig.update_yaxes(title_text="Returns²", row=3, col=1)
    fig.update_yaxes(title_text="Volatility", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    # Update layout
    if title is None:
        title = "Volatility Clustering Analysis"

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=height,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig


def plot_distribution(
    data: "NDArray | pd.Series",
    bins: int = 50,
    fit_normal: bool = True,
    fit_t: bool = False,
    show_moments: bool = True,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Create interactive distribution histogram with fitted curves.

    Visualizes the empirical distribution of data with:
    - Interactive histogram of observed values
    - Fitted normal distribution (optional)
    - Fitted Student's t distribution (optional)
    - Moment statistics annotation (mean, std, skewness, kurtosis)

    Hover for bin details, toggle fitted distributions on/off.

    Useful for assessing normality and identifying distributional characteristics
    such as skewness and heavy tails.

    Parameters
    ----------
    data : ndarray or pd.Series
        Data to plot
    bins : int, default 50
        Number of histogram bins
    fit_normal : bool, default True
        Whether to overlay fitted normal distribution
    fit_t : bool, default False
        Whether to overlay fitted Student's t distribution
    show_moments : bool, default True
        Whether to display moment statistics on plot
    title : str, optional
        Plot title. If None, uses "Distribution Analysis"
    height : int, default 500
        Figure height in pixels

    Returns
    -------
    go.Figure
        Interactive Plotly figure with distribution plot

    Examples
    --------
    >>> import numpy as np
    >>> # Normal data
    >>> data = np.random.randn(1000)
    >>> fig = plot_distribution(data)
    >>> fig.show()

    >>> # Heavy-tailed data
    >>> data = np.random.standard_t(df=3, size=1000)
    >>> fig = plot_distribution(data, fit_t=True)
    >>> fig.show()

    Notes
    -----
    Financial returns typically exhibit:
    - Near-zero mean (if de-meaned)
    - Positive excess kurtosis (heavy tails)
    - Slight negative skewness (larger losses than gains)

    The fitted distributions help identify:
    - Normal: Good fit if kurtosis ≈ 3, skewness ≈ 0
    - Student's t: Better fit for heavy tails (kurtosis > 3)

    See Also
    --------
    ml4t-diagnostic.evaluation.distribution : Statistical distribution tests
    plot_qq : QQ plot for normality assessment

    References
    ----------
    .. [1] Mandelbrot, B. (1963). "The variation of certain speculative prices."
           Journal of Business, 36(4), 394-419.
    .. [2] Fama, E. F. (1965). "The behavior of stock-market prices."
           Journal of Business, 38(1), 34-105.
    """
    # Convert to numpy array if pandas Series
    if isinstance(data, pd.Series):
        data = data.to_numpy()

    # Remove NaN values
    data = data[~np.isnan(data)]

    if len(data) == 0:
        raise ValueError("Input data is empty after removing NaN values")

    # Compute moments
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=True)  # Excess kurtosis

    # Create figure
    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            histnorm="probability density",
            marker={
                "color": COLORS["primary"],
                "opacity": 0.6,
                "line": {"color": "black", "width": 0.5},
            },
            name="Empirical",
            hovertemplate="Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # Generate x values for fitted distributions
    x = np.linspace(data.min(), data.max(), 500)

    # Fit and plot normal distribution
    if fit_normal:
        normal_pdf = stats.norm.pdf(x, mean, std)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=normal_pdf,
                mode="lines",
                line={"color": COLORS["negative"], "width": 2},
                name=f"Normal(μ={mean:.3f}, σ={std:.3f})",
                hovertemplate="Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )

    # Fit and plot Student's t distribution
    if fit_t:
        # Fit t distribution
        params = stats.t.fit(data)
        df, loc, scale = params
        t_pdf = stats.t.pdf(x, df, loc, scale)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=t_pdf,
                mode="lines",
                line={"color": COLORS["positive"], "width": 2, "dash": "dash"},
                name=f"Student's t (df={df:.1f})",
                hovertemplate="Value: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )

    # Update layout
    if title is None:
        title = "Distribution Analysis"

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title="Value",
        yaxis_title="Density",
        height=height,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend={"x": 0.98, "y": 0.98, "xanchor": "right", "bgcolor": "rgba(255,255,255,0.8)"},
    )

    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add moment statistics
    if show_moments:
        textstr = (
            f"<b>Moments</b><br>"
            f"Mean: {mean:.4f}<br>"
            f"Std Dev: {std:.4f}<br>"
            f"Skewness: {skewness:.4f}<br>"
            f"Excess Kurtosis: {kurtosis:.4f}"
        )
        fig.add_annotation(
            text=textstr,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 248, 220, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font={"size": 10, "family": "monospace"},
            align="left",
            valign="top",
        )

    return fig


# Helper functions for ACF/PACF computation (unchanged from matplotlib version)


def _compute_acf(data: "NDArray", max_lags: int) -> "NDArray":
    """Compute autocorrelation function.

    Parameters
    ----------
    data : ndarray
        Time series data
    max_lags : int
        Maximum number of lags

    Returns
    -------
    ndarray
        ACF values for lags 0 to max_lags
    """
    data = data - np.mean(data)
    c0 = np.dot(data, data) / len(data)

    acf = np.zeros(max_lags + 1)
    acf[0] = 1.0  # Correlation with self is 1

    for k in range(1, max_lags + 1):
        ck = np.dot(data[:-k], data[k:]) / len(data)
        acf[k] = ck / c0

    return acf


def _compute_pacf(data: "NDArray", max_lags: int) -> "NDArray":
    """Compute partial autocorrelation function using Durbin-Levinson recursion.

    Parameters
    ----------
    data : ndarray
        Time series data
    max_lags : int
        Maximum number of lags

    Returns
    -------
    ndarray
        PACF values for lags 0 to max_lags

    References
    ----------
    .. [1] Durbin, J. (1960). "The fitting of time-series models."
           Revue de l'Institut International de Statistique, 233-244.
    """
    acf = _compute_acf(data, max_lags)

    pacf = np.zeros(max_lags + 1)
    pacf[0] = 1.0  # PACF at lag 0 is 1

    if max_lags == 0:
        return pacf

    # Durbin-Levinson recursion
    phi = np.zeros((max_lags + 1, max_lags + 1))
    phi[1, 1] = acf[1]
    pacf[1] = acf[1]

    for k in range(2, max_lags + 1):
        # Compute phi[k, k]
        numerator = acf[k]
        for j in range(1, k):
            numerator -= phi[k - 1, j] * acf[k - j]

        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k - 1, j] * acf[j]

        phi[k, k] = numerator / denominator
        pacf[k] = phi[k, k]

        # Update phi[k, j] for j < k
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

    return pacf


def export_static(fig: go.Figure, filename: str, format: str = "png", **kwargs) -> None:
    """Export Plotly figure as static image.

    Converts interactive Plotly figure to static format (PNG, PDF, SVG) for
    presentations, papers, or printable reports.

    Requires kaleido package: `pip install kaleido`

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to export
    filename : str
        Output filename (without extension)
    format : str, default "png"
        Export format: "png", "pdf", "svg", "jpeg"
    **kwargs
        Additional arguments passed to fig.write_image()
        Common options:
        - width: int, image width in pixels
        - height: int, image height in pixels
        - scale: float, image scale factor

    Examples
    --------
    >>> fig = plot_acf_pacf(data)
    >>> export_static(fig, "acf_pacf_report", format="pdf", width=1200, height=400)
    >>> # Creates: acf_pacf_report.pdf

    Notes
    -----
    For best quality PDFs:
    - Use format="pdf"
    - Set scale=2 or higher
    - Specify explicit width/height matching your document

    For web use:
    - Use format="png" or "svg"
    - SVG is vector (scales infinitely) but larger file size

    See Also
    --------
    plotly.graph_objects.Figure.write_image : Underlying export function
    """
    try:
        output_file = f"{filename}.{format}"
        fig.write_image(output_file, format=format, **kwargs)
        print(f"✓ Exported static image: {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("Install kaleido for static export: pip install kaleido")


def get_figure_data(fig: go.Figure) -> pd.DataFrame:
    """Extract underlying data from Plotly figure as DataFrame.

    Retrieves the numerical data used to create the visualization,
    enabling custom analysis or alternative visualizations.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure

    Returns
    -------
    pd.DataFrame
        Data from all traces in the figure

    Examples
    --------
    >>> fig = plot_acf_pacf(data)
    >>> df = get_figure_data(fig)
    >>> print(df.columns)
    >>> # Custom analysis on ACF/PACF values
    >>> significant_lags = df[df['acf'].abs() > 0.1]

    Notes
    -----
    The DataFrame structure depends on the plot type.
    Inspect df.columns to understand available data.
    """
    data_dict = {}

    for i, trace in enumerate(fig.data):
        trace_name = trace.name or f"trace_{i}"

        if hasattr(trace, "x") and trace.x is not None:
            data_dict[f"{trace_name}_x"] = trace.x

        if hasattr(trace, "y") and trace.y is not None:
            data_dict[f"{trace_name}_y"] = trace.y

    return pd.DataFrame(data_dict)
