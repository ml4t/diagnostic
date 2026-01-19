"""Interactive visualizations for ml4t-diagnostic evaluation results.

This module provides Plotly-based visualizations for the Three-Tier
Validation Framework, including IC heatmaps, quantile analysis,
and comprehensive evaluation dashboards.
"""

from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ml4t.diagnostic.backends.polars_backend import PolarsBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Color schemes for financial data
COLORS = {
    "positive": "#00CC88",  # Green for positive returns
    "negative": "#FF4444",  # Red for negative returns
    "neutral": "#888888",  # Gray for neutral
    "primary": "#3366CC",  # Blue for primary data
    "secondary": "#FF9900",  # Orange for secondary data
    "background": "#F8F9FA",
    "grid": "#E0E0E0",
}

# Plotly theme configuration
DEFAULT_LAYOUT = {
    "font": {"family": "Arial, sans-serif", "size": 12},
    "plot_bgcolor": COLORS["background"],
    "paper_bgcolor": "white",
    "hovermode": "closest",
    "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
    "xaxis": {"gridcolor": COLORS["grid"], "zeroline": False},
    "yaxis": {"gridcolor": COLORS["grid"], "zeroline": False},
}


def plot_ic_heatmap(
    predictions: Union[pd.DataFrame, "NDArray[Any]"],
    returns: Union[pd.DataFrame, "NDArray[Any]"],
    horizons: list[int] | None = None,
    time_index: pd.DatetimeIndex | None = None,
    regime_column: str | None = None,
    title: str = "Information Coefficient Term Structure",
    colorscale: str = "RdBu",
    _use_optimized: bool = True,
    use_streaming: bool = True,
) -> go.Figure:
    """Create interactive IC heatmap across multiple forward return horizons.

    This visualization shows how predictive power varies across different
    prediction horizons, helping identify the optimal holding period.

    Parameters
    ----------
    predictions : pd.DataFrame or ndarray
        Model predictions (same for all horizons)
    returns : pd.DataFrame or ndarray
        Forward returns for different horizons (columns = horizons)
    horizons : list[int], optional
        List of forward return horizons. If None, uses column names
    time_index : pd.DatetimeIndex, optional
        Time index for x-axis. If None, uses integer index
    regime_column : str, optional
        Column name for market regime filtering
    title : str, default "Information Coefficient Term Structure"
        Plot title
    colorscale : str, default "RdBu"
        Plotly colorscale name
    use_optimized : bool, default True
        Whether to use optimized Polars backend (always True for performance)

    Returns:
    -------
    go.Figure
        Interactive Plotly figure

    Examples:
    --------
    >>> # Simple usage
    >>> fig = plot_ic_heatmap(predictions, forward_returns)
    >>> fig.show()

    >>> # With custom horizons
    >>> fig = plot_ic_heatmap(
    ...     predictions,
    ...     returns_df,
    ...     horizons=[1, 5, 10, 20],
    ...     time_index=returns_df.index
    ... )
    """
    # Convert inputs to appropriate types
    predictions_data: pd.Series | pd.DataFrame | NDArray[Any]
    if isinstance(predictions, np.ndarray):
        predictions_data = pd.Series(predictions, name="predictions")
    else:
        predictions_data = predictions

    returns_data: pd.DataFrame | NDArray[Any]
    if isinstance(returns, np.ndarray):
        returns_data = (
            pd.DataFrame(returns, columns=cast(Any, ["returns"]))
            if returns.ndim == 1
            else pd.DataFrame(returns)
        )
    else:
        returns_data = returns

    # Determine horizons as strings for internal processing
    horizons_str: list[str]
    if horizons is None:
        if isinstance(returns_data, pd.DataFrame):
            horizons_str = [str(col) for col in returns_data.columns]
        else:
            horizons_str = ["1"]
    else:
        horizons_str = [str(h) for h in horizons]

    # Calculate rolling IC for each horizon
    window_size = min(60, len(predictions_data) // 4)  # Adaptive window

    # Convert Series to DataFrame for _compute_ic_matrix_optimized
    pred_for_ic: pd.DataFrame | NDArray[Any]
    if isinstance(predictions_data, pd.Series):
        pred_for_ic = predictions_data.to_frame()
    elif isinstance(predictions_data, pd.DataFrame):
        pred_for_ic = predictions_data
    else:
        pred_for_ic = predictions_data

    # Use optimized Polars implementation for all cases
    ic_matrix = _compute_ic_matrix_optimized(
        pred_for_ic,
        returns_data,
        horizons_str,
        window_size,
        use_streaming,
    )

    # Create time index
    x_values: pd.Index | pd.DatetimeIndex
    if time_index is not None:
        x_values = time_index[window_size:]
    else:
        x_values = pd.Index(list(range(window_size, len(predictions_data))))

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=ic_matrix,
            x=x_values,
            y=[f"{h}d" for h in horizons_str],
            colorscale=colorscale,
            zmid=0,
            text=np.round(ic_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Horizon: %{y}<br>Time: %{x}<br>IC: %{z:.3f}<extra></extra>",
            colorbar={"title": "IC", "tickmode": "linear", "tick0": -1, "dtick": 0.2},
        ),
    )

    # Update layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title="Date" if time_index is not None else "Time",
        yaxis_title="Forward Return Horizon",
        **DEFAULT_LAYOUT,
    )

    # Add regime filtering if specified
    if regime_column is not None:
        # This would add dropdown for regime filtering
        # Implementation depends on regime data structure
        pass

    return fig


def _compute_ic_matrix_optimized(
    predictions: Union[pd.DataFrame, "NDArray[Any]"],
    returns: Union[pd.DataFrame, "NDArray[Any]"],
    horizons: list[str],
    window_size: int,
    use_streaming: bool = True,
) -> list[list[float]]:
    """Compute IC matrix using optimized Polars operations with streaming for large datasets.

    Parameters
    ----------
    predictions : Union[pd.DataFrame, NDArray]
        Model predictions
    returns : Union[pd.DataFrame, NDArray]
        Returns data for different horizons
    horizons : list[str]
        List of horizon labels
    window_size : int
        Rolling window size for IC calculation
    use_streaming : bool, default True
        Whether to use streaming for large datasets (>100k samples)

    Returns
    -------
    list[list[float]]
        IC matrix with shape (n_horizons, n_time_points)
    """
    # Convert to Polars DataFrame
    data_dict: dict[str, NDArray[Any]] = {}

    # Handle predictions
    if isinstance(predictions, np.ndarray):
        pred_array = predictions.flatten()
    elif hasattr(predictions, "values"):
        pred_array = predictions.values.flatten()
    else:
        pred_array = np.array(predictions).flatten()

    data_dict["predictions"] = pred_array
    n_samples = len(pred_array)

    # Handle returns
    if isinstance(returns, pd.DataFrame):
        for i, horizon in enumerate(horizons):
            if i < returns.shape[1]:
                data_dict[f"returns_{horizon}"] = returns.iloc[:, i].to_numpy()
            else:
                data_dict[f"returns_{horizon}"] = returns.iloc[:, 0].to_numpy()
    elif isinstance(returns, np.ndarray):
        if returns.ndim == 2:
            for i, horizon in enumerate(horizons):
                if i < returns.shape[1]:
                    data_dict[f"returns_{horizon}"] = returns[:, i]
                else:
                    data_dict[f"returns_{horizon}"] = returns[:, 0]
        else:
            for horizon in horizons:
                data_dict[f"returns_{horizon}"] = returns
    else:
        # Assume single series
        ret_array = np.array(returns).flatten()
        for horizon in horizons:
            data_dict[f"returns_{horizon}"] = ret_array

    # Create Polars DataFrame
    df = pl.DataFrame(data_dict)

    # Choose appropriate method based on dataset size and streaming preference
    returns_matrix = df.select([f"returns_{h}" for h in horizons])
    min_periods = max(2, window_size // 2)

    if use_streaming and n_samples > 100000:
        # Use streaming method for large datasets
        ic_results = PolarsBackend.fast_multi_horizon_ic_streaming(
            df["predictions"],
            returns_matrix,
            window_size,
            min_periods=min_periods,
            chunk_size=PolarsBackend.adaptive_chunk_size(
                n_samples,
                len(horizons) + 1,
                target_memory_mb=500,
            ),
        )
    else:
        # Use standard method for smaller datasets
        ic_results = PolarsBackend.fast_multi_horizon_ic(
            df["predictions"],
            returns_matrix,
            window_size,
            min_periods=min_periods,
        )

    # Extract IC matrix
    ic_matrix = []
    for horizon in horizons:
        ic_series = ic_results[f"ic_returns_{horizon}"]
        # Remove initial NaN values and convert to list
        ic_values = ic_series.drop_nulls().to_list()
        # Trim to remove window startup
        if len(ic_values) > window_size:
            ic_values = ic_values[window_size:]
        ic_matrix.append(ic_values)

    return ic_matrix


def plot_quantile_returns(
    predictions: Union[pd.Series, "NDArray[Any]"],
    returns: Union[pd.Series, "NDArray[Any]"],
    n_quantiles: int = 5,
    show_cumulative: bool = True,
    title: str = "Returns by Prediction Quantile",
) -> go.Figure:
    """Create quantile bar chart with optional cumulative returns.

    This visualization shows average returns for each prediction quantile,
    helping validate monotonic relationships between predictions and outcomes.

    Parameters
    ----------
    predictions : pd.Series or ndarray
        Model predictions
    returns : pd.Series or ndarray
        Actual returns
    n_quantiles : int, default 5
        Number of quantiles to create
    show_cumulative : bool, default True
        Whether to show cumulative returns subplot
    title : str
        Plot title

    Returns:
    -------
    go.Figure
        Interactive Plotly figure with quantile analysis
    """
    # Store original index if available
    time_index = None
    if isinstance(returns, pd.Series):
        time_index = returns.index
    elif isinstance(predictions, pd.Series):
        time_index = predictions.index

    # Convert to numpy arrays for consistent processing
    pred_arr: NDArray[Any]
    ret_arr: NDArray[Any]
    if isinstance(predictions, pd.Series):
        pred_arr = predictions.to_numpy()
    else:
        pred_arr = predictions
    if isinstance(returns, pd.Series):
        ret_arr = returns.to_numpy()
    else:
        ret_arr = returns

    # Handle edge cases
    if len(pred_arr) == 0 or len(ret_arr) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    # Check for all NaN
    if np.all(np.isnan(pred_arr)) or np.all(np.isnan(ret_arr)):
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title)
        return fig

    # Create quantiles
    quantile_labels: NDArray[Any]
    try:
        quantile_result = pd.qcut(pred_arr, n_quantiles, labels=False, duplicates="drop") + 1
        quantile_labels = (
            quantile_result.to_numpy()
            if hasattr(quantile_result, "to_numpy")
            else np.array(quantile_result)
        )
    except ValueError:
        # If can't create quantiles, use equal splits
        quantile_labels = np.linspace(1, n_quantiles, len(pred_arr), dtype=int)

    # Calculate mean returns per quantile
    quantile_returns = []
    quantile_counts: list[int] = []
    std_errors = []

    for q in range(1, n_quantiles + 1):
        mask = quantile_labels == q
        q_returns = ret_arr[mask]
        quantile_returns.append(np.mean(q_returns))
        quantile_counts.append(np.sum(mask))
        std_errors.append(np.std(q_returns) / np.sqrt(len(q_returns)))

    # Create figure
    if show_cumulative:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.6, 0.4],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Mean Returns by Quantile", "Cumulative Returns"),
        )
    else:
        fig = go.Figure()

    # Colors based on return sign
    colors = [COLORS["positive"] if r > 0 else COLORS["negative"] for r in quantile_returns]

    # Add bar chart
    bar_trace = go.Bar(
        x=list(range(1, n_quantiles + 1)),
        y=quantile_returns,
        error_y={"type": "data", "array": std_errors, "visible": True},
        marker_color=colors,
        text=[f"{r:.2%}" for r in quantile_returns],
        textposition="outside",
        hovertemplate=(
            "Quantile %{x}<br>Mean Return: %{y:.2%}<br>Count: %{customdata}<extra></extra>"
        ),
        customdata=quantile_counts,
        name="Mean Return",
        showlegend=False,
    )

    if show_cumulative:
        fig.add_trace(bar_trace, row=1, col=1)

        # Calculate cumulative returns for each quantile with proper time alignment
        for q in range(1, n_quantiles + 1):
            mask = quantile_labels == q

            # If we have a time index, use it for proper alignment
            if time_index is not None:
                # Get returns and their corresponding times
                q_indices = np.where(mask)[0]
                # Convert to numpy to avoid pandas index issues with positional sorting
                q_returns_arr = ret_arr[mask]
                q_times = time_index[q_indices]

                # Sort by time
                time_order = np.argsort(q_times)
                q_returns_sorted = q_returns_arr[time_order]
                q_times_sorted = q_times[time_order]

                # Calculate cumulative returns on time-sorted data
                cumulative = np.cumprod(1 + q_returns_sorted) - 1

                fig.add_trace(
                    go.Scatter(
                        x=q_times_sorted,
                        y=cumulative,
                        mode="lines",
                        name=f"Q{q}",
                        line={"width": 2},
                        hovertemplate=(
                            "Quantile %{fullData.name}<br>Time: %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
            else:
                # Fallback to position-based if no time index
                q_returns_arr = ret_arr[mask]
                cumulative = np.cumprod(1 + q_returns_arr) - 1

                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(cumulative)),
                        y=cumulative,
                        mode="lines",
                        name=f"Q{q}",
                        line={"width": 2},
                        hovertemplate=(
                            "Quantile %{fullData.name}<br>Position: %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
    else:
        fig.add_trace(bar_trace)

    # Update layout
    fig.update_xaxes(
        title_text="Prediction Quantile",
        row=2 if show_cumulative else 1,
        col=1,
    )
    fig.update_yaxes(title_text="Mean Return", tickformat=".1%", row=1, col=1)

    if show_cumulative:
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".1%", row=2, col=1)
        # Update x-axis label based on whether we have time index
        x_label = "Time" if time_index is not None else "Position"
        fig.update_xaxes(title_text=x_label, row=2, col=1)

    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"}, **DEFAULT_LAYOUT)

    return fig


def plot_turnover_decay(
    factor_values: pd.DataFrame,
    quantiles: int = 5,
    lags: list[int] | None = None,
    title: str = "Factor Turnover and Decay Analysis",
) -> go.Figure:
    """Visualize factor stability through turnover and autocorrelation analysis.

    Parameters
    ----------
    factor_values : pd.DataFrame
        Time series of factor values (index = time, columns = assets)
    quantiles : int, default 5
        Number of quantiles for turnover calculation
    lags : list[int], optional
        Autocorrelation lags to compute. Default [1, 5, 10, 20]
    title : str
        Plot title

    Returns:
    -------
    go.Figure
        Multi-panel figure showing turnover and decay analysis
    """
    if lags is None:
        lags = [1, 5, 10, 20]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Quantile Turnover by Period",
            "Average Autocorrelation Decay",
            "Turnover Heatmap",
            "Signal Stability",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
        ],
    )

    # Calculate quantile assignments
    quantile_assignments = factor_values.apply(
        lambda x: pd.qcut(x, quantiles, labels=False, duplicates="drop"),
        axis=0,
    )

    # 1. Calculate turnover for each quantile
    turnover_by_quantile = []
    for q in range(quantiles):
        # Count changes in quantile assignment
        in_quantile = (quantile_assignments == q).astype(int)
        changes = in_quantile.diff().abs().sum(axis=1)
        total = in_quantile.sum(axis=1)
        turnover = (changes / (2 * total)).fillna(0).mean()
        turnover_by_quantile.append(turnover)

    # Add turnover bar chart
    fig.add_trace(
        go.Bar(
            x=list(range(1, quantiles + 1)),
            y=turnover_by_quantile,
            marker_color=COLORS["primary"],
            text=[f"{t:.1%}" for t in turnover_by_quantile],
            textposition="outside",
            hovertemplate="Quantile %{x}<br>Turnover: %{y:.1%}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Calculate autocorrelation decay
    autocorr_values = []
    for lag in lags:
        # Calculate autocorrelation for each asset
        autocorr = factor_values.apply(
            lambda x, current_lag=lag: x.autocorr(lag=current_lag),
        )
        autocorr_values.append(autocorr.mean())

    # Add autocorrelation decay plot
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=autocorr_values,
            mode="lines+markers",
            marker={"size": 10, "color": COLORS["secondary"]},
            line={"width": 3, "color": COLORS["secondary"]},
            hovertemplate="Lag %{x}<br>Autocorr: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Turnover heatmap (time vs quantile)
    # Sample time periods for visualization
    time_periods = min(20, len(factor_values) // 10)
    sample_indices = np.linspace(0, len(factor_values) - 2, time_periods, dtype=int)

    turnover_matrix = []
    for idx in sample_indices:
        period_turnover = []
        for q in range(quantiles):
            in_q_t0 = quantile_assignments.iloc[idx] == q
            in_q_t1 = quantile_assignments.iloc[idx + 1] == q
            stayed = (in_q_t0 & in_q_t1).sum()
            total = in_q_t0.sum()
            turnover = 1 - (stayed / total) if total > 0 else 0
            period_turnover.append(turnover)
        turnover_matrix.append(period_turnover)

    fig.add_trace(
        go.Heatmap(
            z=turnover_matrix,
            x=list(range(1, quantiles + 1)),
            y=sample_indices,
            colorscale="Reds",
            hovertemplate=("Time: %{y}<br>Quantile: %{x}<br>Turnover: %{z:.1%}<extra></extra>"),
            showscale=True,
            colorbar={"title": "Turnover", "x": 1.15},
        ),
        row=2,
        col=1,
    )

    # 4. Signal stability (rolling mean of factor values)
    rolling_mean = factor_values.mean(axis=1).rolling(window=20).mean()
    rolling_std = factor_values.mean(axis=1).rolling(window=20).std()

    fig.add_trace(
        go.Scatter(
            x=factor_values.index,
            y=rolling_mean,
            mode="lines",
            line={"color": COLORS["primary"], "width": 2},
            name="Rolling Mean",
            hovertemplate="Time: %{x}<br>Mean: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Add confidence bands
    fig.add_trace(
        go.Scatter(
            x=factor_values.index,
            y=rolling_mean + 2 * rolling_std,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=factor_values.index,
            y=rolling_mean - 2 * rolling_std,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(51, 102, 204, 0.2)",
            name="±2 STD",
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text="Quantile", row=1, col=1)
    fig.update_yaxes(title_text="Turnover Rate", tickformat=".0%", row=1, col=1)

    fig.update_xaxes(title_text="Lag (days)", row=1, col=2)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=2)

    fig.update_xaxes(title_text="Quantile", row=2, col=1)
    fig.update_yaxes(title_text="Time Period", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Factor Value", row=2, col=2)

    # Update layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=800,
        **DEFAULT_LAYOUT,
    )

    return fig


def plot_feature_distributions(
    features: pd.DataFrame,
    n_periods: int = 4,
    method: str = "box",
    title: str = "Feature Distribution Analysis",
) -> go.Figure:
    """Create small multiples showing feature distributions over time.

    Parameters
    ----------
    features : pd.DataFrame
        Feature values (index = time, columns = features)
    n_periods : int, default 4
        Number of time periods to show
    method : str, default "box"
        Plot type: "box", "violin", or "hist"
    title : str
        Plot title

    Returns:
    -------
    go.Figure
        Small multiples visualization
    """
    # Limit to first 9 features for readability
    n_features = min(9, features.shape[1])
    feature_cols = features.columns[:n_features]

    # Create time buckets
    period_size = len(features) // n_periods
    periods = []
    period_labels = []

    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else len(features)
        periods.append((start_idx, end_idx))

        if hasattr(features.index, "date"):
            start_date = features.index[start_idx].strftime("%Y-%m")
            end_date = features.index[end_idx - 1].strftime("%Y-%m")
            period_labels.append(f"{start_date} to {end_date}")
        else:
            period_labels.append(f"Period {i + 1}")

    # Create subplots
    n_rows = int(np.ceil(n_features / 3))
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=[str(col) for col in feature_cols],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Add plots for each feature
    for idx, feature in enumerate(feature_cols):
        row = idx // 3 + 1
        col = idx % 3 + 1

        for period_idx, (start, end) in enumerate(periods):
            period_data = features[feature].iloc[start:end]

            if method == "box":
                fig.add_trace(
                    go.Box(
                        y=period_data,
                        name=period_labels[period_idx],
                        marker_color=px.colors.qualitative.Set3[period_idx],
                        boxpoints="outliers",
                        showlegend=(idx == 0),
                        legendgroup=f"period{period_idx}",
                        hovertemplate="%{y:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            elif method == "violin":
                fig.add_trace(
                    go.Violin(
                        y=period_data,
                        name=period_labels[period_idx],
                        marker_color=px.colors.qualitative.Set3[period_idx],
                        box_visible=True,
                        meanline_visible=True,
                        showlegend=(idx == 0),
                        legendgroup=f"period{period_idx}",
                        hovertemplate="%{y:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            elif method == "hist":
                fig.add_trace(
                    go.Histogram(
                        x=period_data,
                        name=period_labels[period_idx],
                        marker_color=px.colors.qualitative.Set3[period_idx],
                        opacity=0.7,
                        showlegend=(idx == 0),
                        legendgroup=f"period{period_idx}",
                        hovertemplate="Value: %{x:.3f}<br>Count: %{y}<extra></extra>",
                        histnorm="probability",
                    ),
                    row=row,
                    col=col,
                )

    # Update layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=300 * n_rows,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        **DEFAULT_LAYOUT,
    )

    # Update axes
    if method == "hist":
        fig.update_xaxes(title_text="Value")
        fig.update_yaxes(title_text="Probability")
    else:
        fig.update_yaxes(title_text="Value")

    return fig


def plot_ic_decay(
    decay_results: dict[str, Any],
    show_half_life: bool = True,
    show_optimal: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Plot IC decay curve with half-life and optimal horizon annotations.

    Creates an interactive Plotly visualization showing how IC decays across
    prediction horizons, with optional markers for half-life and optimal horizon.

    Parameters
    ----------
    decay_results : dict
        Results from compute_ic_decay()
    show_half_life : bool, default True
        Show vertical line at estimated half-life
    show_optimal : bool, default True
        Show marker at optimal horizon
    title : str | None, default None
        Custom title for the plot. If None, uses "IC Decay Analysis"

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation.metrics import compute_ic_decay
    >>> from ml4t.diagnostic.evaluation.visualization import plot_ic_decay
    >>>
    >>> # Compute decay
    >>> decay = compute_ic_decay(pred_df, price_df, group_col="symbol")
    >>>
    >>> # Visualize
    >>> fig = plot_ic_decay(decay)
    >>> fig.show()
    """
    horizons = decay_results["horizons"]
    ic_by_horizon = decay_results["ic_by_horizon"]
    half_life = decay_results.get("half_life")
    optimal_horizon = decay_results.get("optimal_horizon")

    # Extract IC values in order
    ic_values = [ic_by_horizon[h] for h in horizons]

    # Create figure
    fig = go.Figure()

    # Add IC decay curve
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=ic_values,
            mode="lines+markers",
            name="IC",
            line={"color": COLORS["primary"], "width": 2},
            marker={"size": 8, "color": COLORS["primary"]},
            hovertemplate="Horizon: %{x} days<br>IC: %{y:.4f}<extra></extra>",
        )
    )

    # Add zero line for reference
    fig.add_hline(y=0, line={"color": COLORS["grid"], "width": 1, "dash": "dash"})

    # Add half-life marker
    if show_half_life and half_life is not None:
        # Calculate IC at half-life for the marker
        if horizons[0] in ic_by_horizon:
            initial_ic = ic_by_horizon[horizons[0]]
            half_life_ic = initial_ic * 0.5

            fig.add_vline(
                x=half_life,
                line={"color": COLORS["secondary"], "width": 2, "dash": "dash"},
                annotation_text=f"Half-life: {half_life:.1f}d",
                annotation_position="top right",
            )

            # Add marker at half-life point
            fig.add_trace(
                go.Scatter(
                    x=[half_life],
                    y=[half_life_ic],
                    mode="markers",
                    name="Half-life",
                    marker={"size": 12, "color": COLORS["secondary"], "symbol": "diamond"},
                    hovertemplate=f"Half-life: {half_life:.1f} days<br>IC: {half_life_ic:.4f}<extra></extra>",
                )
            )

    # Add optimal horizon marker
    if show_optimal and optimal_horizon is not None:
        optimal_ic = ic_by_horizon[optimal_horizon]

        fig.add_trace(
            go.Scatter(
                x=[optimal_horizon],
                y=[optimal_ic],
                mode="markers",
                name="Optimal",
                marker={
                    "size": 15,
                    "color": COLORS["positive"],
                    "symbol": "star",
                    "line": {"width": 2, "color": "white"},
                },
                hovertemplate=f"Optimal: {optimal_horizon} days<br>IC: {optimal_ic:.4f}<extra></extra>",
            )
        )

    # Update layout
    if title is None:
        title = "IC Decay Analysis"

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Horizon (days)",
        yaxis_title="Information Coefficient",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        **DEFAULT_LAYOUT,
    )

    return fig


def plot_monotonicity(
    monotonicity_results: dict[str, Any],
    title: str | None = None,
    show_correlation: bool = True,
) -> go.Figure:
    """Plot quantile analysis for monotonicity testing.

    Creates a bar chart showing mean outcomes across feature quantiles,
    with annotations for monotonicity metrics.

    Parameters
    ----------
    monotonicity_results : dict
        Results from compute_monotonicity()
    title : str | None, default None
        Custom title. If None, uses "Monotonicity Analysis"
    show_correlation : bool, default True
        Show correlation coefficient in subtitle

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation.metrics import compute_monotonicity
    >>> from ml4t.diagnostic.evaluation.visualization import plot_monotonicity
    >>>
    >>> # Compute monotonicity
    >>> result = compute_monotonicity(features, outcomes, n_quantiles=5)
    >>>
    >>> # Visualize
    >>> fig = plot_monotonicity(result)
    >>> fig.show()
    """
    quantile_labels = monotonicity_results["quantile_labels"]
    quantile_means = monotonicity_results["quantile_means"]
    correlation = monotonicity_results["correlation"]
    p_value = monotonicity_results["p_value"]
    is_monotonic = monotonicity_results["is_monotonic"]
    monotonicity_score = monotonicity_results["monotonicity_score"]
    direction = monotonicity_results["direction"]

    # Determine bar colors based on values
    colors = [COLORS["positive"] if x > 0 else COLORS["negative"] for x in quantile_means]

    # Create figure
    fig = go.Figure()

    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=quantile_labels,
            y=quantile_means,
            marker={"color": colors, "line": {"color": "white", "width": 1}},
            hovertemplate="<b>%{x}</b><br>Mean Outcome: %{y:.4f}<extra></extra>",
            name="Mean Outcome",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line={"color": COLORS["grid"], "width": 1, "dash": "dash"})

    # Build title and subtitle
    if title is None:
        title = "Monotonicity Analysis"

    subtitle_parts = []
    if show_correlation:
        subtitle_parts.append(f"Correlation: {correlation:.3f} (p={p_value:.4f})")

    subtitle_parts.append(f"Monotonicity: {monotonicity_score:.1%}")
    subtitle_parts.append(f"Direction: {direction.replace('_', ' ').title()}")

    if is_monotonic:
        subtitle_parts.append("✓ Monotonic")
    else:
        subtitle_parts.append("✗ Not Monotonic")

    subtitle = " | ".join(subtitle_parts)

    # Update layout
    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><sub>{subtitle}</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Feature Quantile",
        yaxis_title="Mean Outcome",
        showlegend=False,
        **DEFAULT_LAYOUT,
    )

    return fig
