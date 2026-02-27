"""Cross-validation fold structure visualization.

This module provides interactive Plotly-based visualizations for cross-validation
fold structures, showing train/validation/test periods as a timeline.

Supports both WalkForwardCV and CombinatorialCV splitters from ml4t.diagnostic.splitters.

Example workflow:
    >>> from ml4t.diagnostic.splitters import WalkForwardCV
    >>> from ml4t.diagnostic.visualization import plot_cv_folds
    >>>
    >>> # Configure CV
    >>> cv = WalkForwardCV(
    ...     n_splits=5,
    ...     test_period=126,
    ...     test_size=42,
    ...     train_size=252,
    ...     label_horizon=21,
    ... )
    >>>
    >>> # Visualize fold structure
    >>> fig = plot_cv_folds(cv, X)
    >>> fig.show()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

from ml4t.diagnostic.visualization.core import (
    apply_responsive_layout,
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ml4t.diagnostic.splitters.base import BaseSplitter

__all__ = ["plot_cv_folds"]


# Color definitions for fold visualization
COLORS = {
    "train": "#3498DB",  # Blue
    "val": "#2ECC71",  # Green
    "test": "#E74C3C",  # Red
    "gap": "#95A5A6",  # Gray
}

COLORS_DARK = {
    "train": "#5DADE2",  # Light blue
    "val": "#58D68D",  # Light green
    "test": "#EC7063",  # Light red
    "gap": "#AAB7B8",  # Light gray
}


@dataclass
class FoldInfo:
    """Information about a single CV fold."""

    fold_number: int
    train_start: int | pd.Timestamp
    train_end: int | pd.Timestamp
    val_start: int | pd.Timestamp
    val_end: int | pd.Timestamp
    train_samples: int
    val_samples: int
    # Purge gap info (between train_end and val_start)
    purge_start: int | pd.Timestamp | None = None
    purge_end: int | pd.Timestamp | None = None
    purge_samples: int = 0


def _extract_timestamps_for_viz(
    X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None,
    timestamp_col: str | None,
) -> pd.DatetimeIndex | None:
    """Extract timestamps from data for visualization.

    Parameters
    ----------
    X : DataFrame or ndarray, optional
        Input data that may contain timestamps.
    timestamp_col : str, optional
        Column name containing timestamps (for Polars DataFrames).

    Returns
    -------
    timestamps : pd.DatetimeIndex or None
        Timestamps if available, None otherwise.
    """
    if X is None:
        return None

    # Polars DataFrame: extract from column
    if isinstance(X, pl.DataFrame):
        if timestamp_col is not None and timestamp_col in X.columns:
            ts_series = X[timestamp_col].to_pandas()
            if pd.api.types.is_datetime64_any_dtype(ts_series):
                return pd.DatetimeIndex(ts_series)
        return None

    # pandas DataFrame: prefer index, fallback to column
    if isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.DatetimeIndex):
            return X.index
        if timestamp_col is not None and timestamp_col in X.columns:
            ts_series = X[timestamp_col]
            if pd.api.types.is_datetime64_any_dtype(ts_series):
                return pd.DatetimeIndex(ts_series)
        return None

    # numpy array: no timestamp information
    return None


def _get_n_samples(X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None) -> int | None:
    """Get number of samples from data.

    Parameters
    ----------
    X : DataFrame or ndarray, optional
        Input data.

    Returns
    -------
    n_samples : int or None
        Number of samples if determinable.
    """
    if X is None:
        return None
    if isinstance(X, pl.DataFrame):
        return X.height
    if isinstance(X, pd.DataFrame):
        return len(X)
    if isinstance(X, np.ndarray):
        return int(X.shape[0])
    return None


def _collect_fold_info(
    cv: BaseSplitter,
    X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None,
    y: pl.Series | pd.Series | NDArray[Any] | None,
    groups: pl.Series | pd.Series | NDArray[Any] | None,
    timestamps: pd.DatetimeIndex | None,
) -> list[FoldInfo]:
    """Collect information about each fold from the splitter.

    Parameters
    ----------
    cv : BaseSplitter
        Cross-validation splitter.
    X : DataFrame or ndarray, optional
        Input data.
    y : Series or ndarray, optional
        Target variable.
    groups : Series or ndarray, optional
        Group labels.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for the data.

    Returns
    -------
    folds : list of FoldInfo
        Information about each fold.
    """
    folds: list[FoldInfo] = []

    # If X is None, we can't iterate through folds
    # We need actual data to generate fold indices
    if X is None:
        return folds

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        # Get train boundaries
        train_start_idx = int(train_idx.min()) if len(train_idx) > 0 else 0
        train_end_idx = int(train_idx.max()) if len(train_idx) > 0 else 0

        # Get val boundaries
        val_start_idx = int(val_idx.min()) if len(val_idx) > 0 else 0
        val_end_idx = int(val_idx.max()) if len(val_idx) > 0 else 0

        # Convert to timestamps if available
        if timestamps is not None and len(timestamps) > 0:
            train_start = timestamps[train_start_idx]
            train_end = timestamps[train_end_idx]
            val_start = timestamps[val_start_idx]
            val_end = timestamps[val_end_idx]
        else:
            train_start = train_start_idx
            train_end = train_end_idx
            val_start = val_start_idx
            val_end = val_end_idx

        # Detect purge gap (difference between train_end and val_start)
        # The gap is the region where samples were purged
        purge_start = None
        purge_end = None
        purge_samples = 0

        # Calculate actual gap between train end and val start
        if len(train_idx) > 0 and len(val_idx) > 0:
            # Gap exists if there are indices between train_end and val_start
            gap_start_idx = train_end_idx + 1
            gap_end_idx = val_start_idx - 1

            if gap_end_idx >= gap_start_idx:
                purge_samples = gap_end_idx - gap_start_idx + 1
                if timestamps is not None and len(timestamps) > 0:
                    purge_start = timestamps[gap_start_idx]
                    purge_end = timestamps[gap_end_idx]
                else:
                    purge_start = gap_start_idx
                    purge_end = gap_end_idx

        fold_info = FoldInfo(
            fold_number=fold_num + 1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            train_samples=len(train_idx),
            val_samples=len(val_idx),
            purge_start=purge_start,
            purge_end=purge_end,
            purge_samples=purge_samples,
        )
        folds.append(fold_info)

    return folds


def _get_test_period_info(
    cv: BaseSplitter,
    timestamps: pd.DatetimeIndex | None,
    n_samples: int | None,
) -> tuple[Any, Any, int] | None:
    """Get held-out test period information if available.

    Parameters
    ----------
    cv : BaseSplitter
        Cross-validation splitter.
    timestamps : pd.DatetimeIndex, optional
        Timestamps for the data.
    n_samples : int, optional
        Total number of samples.

    Returns
    -------
    test_info : tuple or None
        (test_start, test_end, test_samples) if held-out test exists.
    """
    # Check if CV has held-out test indices (WalkForwardCV feature)
    test_indices = getattr(cv, "_test_indices", None)
    if test_indices is not None and len(test_indices) > 0:
        test_start_idx = int(test_indices.min())
        test_end_idx = int(test_indices.max())

        if timestamps is not None and len(timestamps) > 0:
            test_start = timestamps[test_start_idx]
            test_end = timestamps[test_end_idx]
        else:
            test_start = test_start_idx
            test_end = test_end_idx

        return (test_start, test_end, len(test_indices))

    return None


def _format_range_label(
    start: int | pd.Timestamp,
    end: int | pd.Timestamp,
    has_timestamps: bool,
) -> str:
    """Format a range label for display.

    Parameters
    ----------
    start : int or pd.Timestamp
        Start of range.
    end : int or pd.Timestamp
        End of range.
    has_timestamps : bool
        Whether values are timestamps.

    Returns
    -------
    label : str
        Formatted range label.
    """
    if has_timestamps and isinstance(start, pd.Timestamp):
        return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    return f"{start:,} to {end:,}"


def _get_x_value(
    value: int | pd.Timestamp,
    has_timestamps: bool,
) -> int | pd.Timestamp:
    """Get x-axis value for plotting.

    Parameters
    ----------
    value : int or pd.Timestamp
        The value to convert.
    has_timestamps : bool
        Whether we're using timestamps.

    Returns
    -------
    x_value : int or pd.Timestamp
        Value suitable for x-axis.
    """
    return value


def _compute_bar_width(
    start: int | pd.Timestamp,
    end: int | pd.Timestamp,
    has_timestamps: bool,
) -> pd.Timedelta | int:
    """Compute the width of a bar for plotting.

    Parameters
    ----------
    start : int or pd.Timestamp
        Start of range.
    end : int or pd.Timestamp
        End of range.
    has_timestamps : bool
        Whether values are timestamps.

    Returns
    -------
    width : pd.Timedelta or int
        Width of the bar.
    """
    if has_timestamps and isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        return end - start
    # For index-based, compute number of samples
    return int(end) - int(start) + 1  # type: ignore[arg-type]


def plot_cv_folds(
    cv: BaseSplitter,
    X: pl.DataFrame | pd.DataFrame | NDArray[Any] | None = None,
    y: pl.Series | pd.Series | NDArray[Any] | None = None,
    groups: pl.Series | pd.Series | NDArray[Any] | None = None,
    *,
    show_test_period: bool = True,
    show_purge_gaps: bool = True,
    show_sample_counts: bool = True,
    timestamp_col: str | None = None,
    theme: str | None = None,
    height: int | None = None,
    width: int | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot cross-validation fold structure as a timeline.

    Creates an interactive visualization showing the train/validation/test periods
    for each fold in the cross-validation. This helps verify that the CV structure
    is correct and understand how data is split.

    Parameters
    ----------
    cv : BaseSplitter
        Cross-validation splitter (WalkForwardCV or CombinatorialCV).
    X : DataFrame or ndarray, optional
        Input data. If None, shows a message that data is required.
    y : Series or ndarray, optional
        Target variable. Passed to cv.split() if provided.
    groups : Series or ndarray, optional
        Group labels. Passed to cv.split() if provided.
    show_test_period : bool, default=True
        Whether to show held-out test period (if available).
    show_purge_gaps : bool, default=True
        Whether to show purge gaps between train and validation.
    show_sample_counts : bool, default=True
        Whether to show sample counts in hover information.
    timestamp_col : str, optional
        Column name containing timestamps (for Polars DataFrames).
    theme : str, optional
        Theme name ("default", "dark", "print", "presentation").
        If None, uses current global theme.
    height : int, optional
        Figure height in pixels. If None, auto-sizes based on fold count.
    width : int, optional
        Figure width in pixels. If None, uses theme default.
    title : str, optional
        Figure title. If None, uses "Cross-Validation Fold Structure".

    Returns
    -------
    go.Figure
        Interactive Plotly figure with:
        - Horizontal bars showing train (blue), validation (green), test (red) periods
        - Gray indicators for purge gaps
        - Hover information with date ranges, sample counts
        - Legend for period types

    Examples
    --------
    >>> from ml4t.diagnostic.splitters import WalkForwardCV
    >>> from ml4t.diagnostic.visualization import plot_cv_folds
    >>>
    >>> # Basic usage
    >>> cv = WalkForwardCV(n_splits=5, test_size=42, train_size=252)
    >>> fig = plot_cv_folds(cv, X)
    >>> fig.show()
    >>>
    >>> # With held-out test period
    >>> cv = WalkForwardCV(
    ...     n_splits=5,
    ...     test_period=126,  # Reserve last 126 days for final test
    ...     test_size=42,
    ...     train_size=252,
    ...     label_horizon=21,
    ...     fold_direction="backward",
    ... )
    >>> fig = plot_cv_folds(cv, X, show_purge_gaps=True)
    >>> fig.show()
    >>>
    >>> # Customize appearance
    >>> fig = plot_cv_folds(
    ...     cv, X,
    ...     theme="dark",
    ...     height=400,
    ...     title="My CV Structure",
    ... )

    Notes
    -----
    - Timeline shows relative positions of train/validation/test periods
    - Purge gaps appear as thin gray bars between train and validation
    - Held-out test period (if configured) shown as separate row at bottom
    - X-axis shows dates if timestamps available, otherwise sample indices
    """
    # Validate theme for layout. Keep the fold palette deterministic unless the
    # caller explicitly asks for dark mode.
    resolved_theme = validate_theme(theme)
    theme_config = get_theme_config(resolved_theme)
    colors = COLORS_DARK if theme == "dark" else COLORS

    # Extract timestamps if available
    timestamps = _extract_timestamps_for_viz(X, timestamp_col)
    has_timestamps = timestamps is not None

    # Get number of samples
    n_samples = _get_n_samples(X)

    # Handle case where X is None
    if X is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No data provided. Pass X to visualize fold structure.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            title=title or "Cross-Validation Fold Structure",
            **theme_config["layout"],
            width=width or 800,
            height=height or 200,
        )
        return fig

    # Collect fold information
    folds = _collect_fold_info(cv, X, y, groups, timestamps)

    # Handle empty folds
    if not folds:
        fig = go.Figure()
        fig.add_annotation(
            text="No folds generated. Check CV configuration.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        fig.update_layout(
            title=title or "Cross-Validation Fold Structure",
            **theme_config["layout"],
            width=width or 800,
            height=height or 200,
        )
        return fig

    # Get held-out test info
    test_info = None
    if show_test_period:
        test_info = _get_test_period_info(cv, timestamps, n_samples)

    # Create figure
    fig = go.Figure()

    # Determine y-axis labels
    n_folds = len(folds)
    y_labels = [f"Fold {f.fold_number}" for f in folds]
    if test_info is not None:
        y_labels.append("Held-out Test")

    # Track whether we've added legend entries
    legend_added = {"train": False, "val": False, "test": False, "gap": False}

    # Add traces for each fold
    for i, fold in enumerate(folds):
        y_pos = n_folds - i - 1  # Reverse order so Fold 1 is at top

        # Training period (bar)
        train_hover = f"<b>Fold {fold.fold_number} - Training</b><br>"
        train_hover += (
            f"Range: {_format_range_label(fold.train_start, fold.train_end, has_timestamps)}<br>"
        )
        if show_sample_counts:
            train_hover += f"Samples: {fold.train_samples:,}"
        train_hover += "<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=[_compute_bar_width(fold.train_start, fold.train_end, has_timestamps)],
                y=[y_pos],
                base=[_get_x_value(fold.train_start, has_timestamps)],
                orientation="h",
                marker={"color": colors["train"]},
                name="Training" if not legend_added["train"] else "",
                showlegend=not legend_added["train"],
                legendgroup="train",
                hovertemplate=train_hover,
            )
        )
        legend_added["train"] = True

        # Purge gap (thin bar between train and val)
        if show_purge_gaps and fold.purge_samples > 0 and fold.purge_start is not None:
            gap_hover = f"<b>Fold {fold.fold_number} - Purge Gap</b><br>"
            gap_hover += f"Range: {_format_range_label(fold.purge_start, fold.purge_end, has_timestamps)}<br>"
            if show_sample_counts:
                gap_hover += f"Purged samples: {fold.purge_samples:,}"
            gap_hover += "<extra></extra>"

            # purge_end is guaranteed to be not None here due to outer if condition
            purge_end = fold.purge_end
            assert purge_end is not None  # Type narrowing for type checker

            fig.add_trace(
                go.Bar(
                    x=[_compute_bar_width(fold.purge_start, purge_end, has_timestamps)],
                    y=[y_pos],
                    base=[_get_x_value(fold.purge_start, has_timestamps)],
                    orientation="h",
                    marker={"color": colors["gap"], "opacity": 0.5},
                    name="Purge Gap" if not legend_added["gap"] else "",
                    showlegend=not legend_added["gap"],
                    legendgroup="gap",
                    hovertemplate=gap_hover,
                )
            )
            legend_added["gap"] = True

        # Validation period (bar)
        val_hover = f"<b>Fold {fold.fold_number} - Validation</b><br>"
        val_hover += (
            f"Range: {_format_range_label(fold.val_start, fold.val_end, has_timestamps)}<br>"
        )
        if show_sample_counts:
            val_hover += f"Samples: {fold.val_samples:,}"
        val_hover += "<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=[_compute_bar_width(fold.val_start, fold.val_end, has_timestamps)],
                y=[y_pos],
                base=[_get_x_value(fold.val_start, has_timestamps)],
                orientation="h",
                marker={"color": colors["val"]},
                name="Validation" if not legend_added["val"] else "",
                showlegend=not legend_added["val"],
                legendgroup="val",
                hovertemplate=val_hover,
            )
        )
        legend_added["val"] = True

    # Add held-out test period if available
    if test_info is not None:
        test_start, test_end, test_samples = test_info
        y_pos = -1  # Below all folds (will be at bottom after reversal)

        test_hover = "<b>Held-out Test Period</b><br>"
        test_hover += f"Range: {_format_range_label(test_start, test_end, has_timestamps)}<br>"
        if show_sample_counts:
            test_hover += f"Samples: {test_samples:,}"
        test_hover += "<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=[_compute_bar_width(test_start, test_end, has_timestamps)],
                y=[y_pos],
                base=[_get_x_value(test_start, has_timestamps)],
                orientation="h",
                marker={"color": colors["test"]},
                name="Held-out Test",
                showlegend=True,
                legendgroup="test",
                hovertemplate=test_hover,
            )
        )

    # Calculate y-axis tick positions and labels
    y_tick_vals = list(range(n_folds - 1, -1, -1))
    y_tick_text = [f"Fold {i + 1}" for i in range(n_folds)]

    if test_info is not None:
        y_tick_vals.append(-1)
        y_tick_text.append("Held-out Test")

    # Update layout
    fig.update_layout(
        title=title or "Cross-Validation Fold Structure",
        xaxis_title="Date" if has_timestamps else "Sample Index",
        yaxis={
            "title": "",
            "tickmode": "array",
            "tickvals": y_tick_vals,
            "ticktext": y_tick_text,
        },
        barmode="overlay",
        **theme_config["layout"],
        width=width or theme_config["defaults"]["width"],
        height=height or max(300, (n_folds + (1 if test_info else 0)) * 50 + 100),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    # Apply responsive layout
    apply_responsive_layout(fig)

    return fig
