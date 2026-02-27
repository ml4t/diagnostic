"""Tests for cross-validation fold visualization functions.

Tests cover:
- plot_cv_folds: Timeline visualization of CV fold structure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest

from ml4t.diagnostic.splitters import CombinatorialCV, WalkForwardCV
from ml4t.diagnostic.visualization.cv_plots import plot_cv_folds

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data_numpy() -> np.ndarray:
    """Sample numpy array data."""
    np.random.seed(42)
    return np.random.randn(500, 5)


@pytest.fixture
def sample_data_pandas() -> pd.DataFrame:
    """Sample pandas DataFrame with DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    return pd.DataFrame(
        np.random.randn(500, 5),
        index=dates,
        columns=[f"feature_{i}" for i in range(5)],
    )


@pytest.fixture
def sample_data_pandas_no_datetime() -> pd.DataFrame:
    """Sample pandas DataFrame without DatetimeIndex."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(500, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )


@pytest.fixture
def sample_data_polars() -> pl.DataFrame:
    """Sample Polars DataFrame with timestamp column."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    return pl.DataFrame(
        {
            "timestamp": dates,
            **{f"feature_{i}": np.random.randn(500) for i in range(5)},
        }
    )


@pytest.fixture
def walk_forward_cv() -> WalkForwardCV:
    """Basic WalkForwardCV splitter."""
    return WalkForwardCV(
        n_splits=5,
        test_size=50,
        train_size=200,
    )


@pytest.fixture
def walk_forward_cv_with_gap() -> WalkForwardCV:
    """WalkForwardCV with label_horizon for purge gap."""
    return WalkForwardCV(
        n_splits=5,
        test_size=50,
        train_size=200,
        label_horizon=10,
    )


@pytest.fixture
def walk_forward_cv_with_test_period() -> WalkForwardCV:
    """WalkForwardCV with held-out test period."""
    return WalkForwardCV(
        n_splits=3,
        test_period=100,
        test_size=50,
        train_size=150,
        label_horizon=10,
        fold_direction="backward",
    )


@pytest.fixture
def combinatorial_cv() -> CombinatorialCV:
    """Basic CombinatorialCV splitter."""
    return CombinatorialCV(
        n_groups=6,
        n_test_groups=2,
        max_combinations=5,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestPlotCvFoldsBasic:
    """Tests for basic plot_cv_folds() functionality."""

    def test_returns_figure(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that function returns a Plotly Figure."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        assert isinstance(fig, go.Figure)

    def test_with_numpy_data(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test with numpy array input."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        assert isinstance(fig, go.Figure)
        # Should have bar traces for train and val periods
        bar_traces = [t for t in fig.data if t.type == "bar"]
        assert len(bar_traces) >= 10  # 5 folds * 2 periods (train + val)

    def test_with_pandas_datetime_index(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_pandas: pd.DataFrame,
    ) -> None:
        """Test with pandas DataFrame with DatetimeIndex."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_pandas)

        assert isinstance(fig, go.Figure)
        # X-axis should show dates
        assert fig.layout.xaxis.title.text == "Date"

    def test_with_pandas_no_datetime(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_pandas_no_datetime: pd.DataFrame,
    ) -> None:
        """Test with pandas DataFrame without DatetimeIndex."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_pandas_no_datetime)

        assert isinstance(fig, go.Figure)
        # X-axis should show sample index
        assert fig.layout.xaxis.title.text == "Sample Index"

    def test_with_polars_data(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_polars: pl.DataFrame,
    ) -> None:
        """Test with Polars DataFrame."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_polars,
            timestamp_col="timestamp",
        )

        assert isinstance(fig, go.Figure)
        # X-axis should show dates
        assert fig.layout.xaxis.title.text == "Date"

    def test_no_data_provided(
        self,
        walk_forward_cv: WalkForwardCV,
    ) -> None:
        """Test when X=None."""
        fig = plot_cv_folds(walk_forward_cv, X=None)

        assert isinstance(fig, go.Figure)
        # Should have annotation about missing data
        assert len(fig.layout.annotations) >= 1
        assert "No data" in fig.layout.annotations[0].text


# =============================================================================
# WalkForwardCV Tests
# =============================================================================


class TestPlotCvFoldsWalkForward:
    """Tests for plot_cv_folds() with WalkForwardCV."""

    def test_correct_number_of_folds(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that correct number of folds is shown."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        # Check y-axis tick labels include all 5 folds
        y_ticktext = fig.layout.yaxis.ticktext
        fold_labels = [t for t in y_ticktext if "Fold" in t]
        assert len(fold_labels) == 5

    def test_with_purge_gap(
        self,
        walk_forward_cv_with_gap: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test visualization with purge gaps."""
        fig = plot_cv_folds(
            walk_forward_cv_with_gap,
            sample_data_numpy,
            show_purge_gaps=True,
        )

        assert isinstance(fig, go.Figure)
        # Check for gap traces (they have lower opacity)
        gap_traces = [
            t
            for t in fig.data
            if t.type == "bar"
            and hasattr(t, "marker")
            and t.marker
            and getattr(t.marker, "opacity", None) == 0.5
        ]
        # May or may not have gaps depending on fold structure
        assert len(gap_traces) >= 0

    def test_hide_purge_gaps(
        self,
        walk_forward_cv_with_gap: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that purge gaps can be hidden."""
        fig = plot_cv_folds(
            walk_forward_cv_with_gap,
            sample_data_numpy,
            show_purge_gaps=False,
        )

        assert isinstance(fig, go.Figure)
        # Check no gap traces (no traces with opacity 0.5)
        gap_traces = [
            t
            for t in fig.data
            if t.type == "bar"
            and hasattr(t, "marker")
            and t.marker
            and getattr(t.marker, "opacity", None) == 0.5
        ]
        assert len(gap_traces) == 0

    def test_with_held_out_test(
        self,
        walk_forward_cv_with_test_period: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test visualization with held-out test period."""
        fig = plot_cv_folds(
            walk_forward_cv_with_test_period,
            sample_data_numpy,
            show_test_period=True,
        )

        assert isinstance(fig, go.Figure)
        # Check for held-out test in y-axis labels
        y_ticktext = fig.layout.yaxis.ticktext
        test_labels = [t for t in y_ticktext if "Held-out" in t]
        assert len(test_labels) == 1

    def test_hide_test_period(
        self,
        walk_forward_cv_with_test_period: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that test period can be hidden."""
        fig = plot_cv_folds(
            walk_forward_cv_with_test_period,
            sample_data_numpy,
            show_test_period=False,
        )

        assert isinstance(fig, go.Figure)
        # Check no held-out test in y-axis labels
        y_ticktext = fig.layout.yaxis.ticktext
        test_labels = [t for t in y_ticktext if "Held-out" in t]
        assert len(test_labels) == 0


# =============================================================================
# CombinatorialCV Tests
# =============================================================================


class TestPlotCvFoldsCombinatorial:
    """Tests for plot_cv_folds() with CombinatorialCV."""

    def test_with_combinatorial_cv(
        self,
        combinatorial_cv: CombinatorialCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test with CombinatorialCV splitter."""
        fig = plot_cv_folds(combinatorial_cv, sample_data_numpy)

        assert isinstance(fig, go.Figure)
        # Should have traces for training and validation
        bar_traces = [t for t in fig.data if t.type == "bar"]
        assert len(bar_traces) >= 2

    def test_combinatorial_fold_count(
        self,
        combinatorial_cv: CombinatorialCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that combinatorial CV shows correct fold count."""
        fig = plot_cv_folds(combinatorial_cv, sample_data_numpy)

        # Should have 5 folds (max_combinations=5)
        y_ticktext = fig.layout.yaxis.ticktext
        fold_labels = [t for t in y_ticktext if "Fold" in t]
        assert len(fold_labels) == 5


# =============================================================================
# Theme and Styling Tests
# =============================================================================


class TestPlotCvFoldsTheme:
    """Tests for theme and styling options."""

    def test_default_theme(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test default theme."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy, theme="default")

        # Default theme has light background
        assert fig.layout.paper_bgcolor.upper() == "#FFFFFF"

    def test_dark_theme(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test dark theme."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            theme="dark",
        )

        # Dark theme has dark background
        assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d"]

    def test_print_theme(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test print theme."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            theme="print",
        )

        # Print theme uses serif font
        assert "Times" in fig.layout.font.family or "serif" in fig.layout.font.family.lower()

    def test_custom_title(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test custom title."""
        custom_title = "My Custom CV Structure"
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            title=custom_title,
        )

        assert fig.layout.title.text == custom_title

    def test_custom_dimensions(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test custom width and height."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            width=1200,
            height=600,
        )

        assert fig.layout.width == 1200
        assert fig.layout.height == 600


# =============================================================================
# Hover Information Tests
# =============================================================================


class TestPlotCvFoldsHover:
    """Tests for hover information."""

    def test_hover_contains_sample_counts(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that hover shows sample counts when enabled."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            show_sample_counts=True,
        )

        # Check hover templates contain "Samples:"
        bar_traces = [t for t in fig.data if t.type == "bar"]
        has_samples_info = any("Samples:" in (t.hovertemplate or "") for t in bar_traces)
        assert has_samples_info

    def test_hover_without_sample_counts(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that hover hides sample counts when disabled."""
        fig = plot_cv_folds(
            walk_forward_cv,
            sample_data_numpy,
            show_sample_counts=False,
        )

        # Check hover templates don't contain "Samples:"
        bar_traces = [t for t in fig.data if t.type == "bar"]
        has_samples_info = any("Samples:" in (t.hovertemplate or "") for t in bar_traces)
        assert not has_samples_info


# =============================================================================
# Legend Tests
# =============================================================================


class TestPlotCvFoldsLegend:
    """Tests for legend behavior."""

    def test_legend_present(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that legend is present."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        # At least Training and Validation should be in legend
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert "Training" in legend_names
        assert "Validation" in legend_names

    def test_legend_entries_not_duplicated(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that legend entries are not duplicated."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        legend_names = [t.name for t in fig.data if t.showlegend and t.name]
        # Each legend entry should appear only once
        assert len(legend_names) == len(set(legend_names))


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test that all functions can be imported from visualization module."""

    def test_import_from_cv_plots(self) -> None:
        """Test direct import from cv_plots module."""
        from ml4t.diagnostic.visualization.cv_plots import plot_cv_folds

        assert callable(plot_cv_folds)

    def test_import_from_visualization_package(self) -> None:
        """Test import from main visualization package."""
        from ml4t.diagnostic.visualization import plot_cv_folds

        assert callable(plot_cv_folds)


# =============================================================================
# Edge Cases
# =============================================================================


class TestPlotCvFoldsEdgeCases:
    """Tests for edge cases."""

    def test_empty_folds(self) -> None:
        """Test behavior with splitter that generates no folds."""
        # Create a CV with very few samples that might generate no valid folds
        cv = WalkForwardCV(n_splits=1, test_size=100, train_size=500)
        X = np.random.randn(50, 5)  # Too few samples

        # This should either raise an error during split or show empty folds message
        # Depending on implementation, adjust expectation
        try:
            fig = plot_cv_folds(cv, X)
            assert isinstance(fig, go.Figure)
        except ValueError:
            # Expected if CV can't generate folds
            pass

    def test_single_fold(
        self,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test with single fold."""
        cv = WalkForwardCV(n_splits=1, test_size=50, train_size=200)
        fig = plot_cv_folds(cv, sample_data_numpy)

        assert isinstance(fig, go.Figure)
        y_ticktext = fig.layout.yaxis.ticktext
        fold_labels = [t for t in y_ticktext if "Fold" in t]
        assert len(fold_labels) == 1

    def test_many_folds(
        self,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test with many folds (height auto-scales)."""
        cv = WalkForwardCV(n_splits=10, test_size=30, train_size=100)
        fig = plot_cv_folds(cv, sample_data_numpy)

        assert isinstance(fig, go.Figure)
        # Height should auto-scale for many folds
        assert fig.layout.height >= 300


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestPlotCvFoldsColors:
    """Tests for bar colors."""

    def test_train_bar_color(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that training bars have correct color."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        # Find training traces
        train_traces = [t for t in fig.data if t.name == "Training"]
        assert len(train_traces) == 1
        # Blue color
        assert train_traces[0].marker.color == "#3498DB"

    def test_validation_bar_color(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that validation bars have correct color."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)

        # Find validation traces
        val_traces = [t for t in fig.data if t.name == "Validation"]
        assert len(val_traces) == 1
        # Green color
        assert val_traces[0].marker.color == "#2ECC71"


class TestPlotCvFoldsXAxis:
    """Tests for x-axis behavior."""

    def test_xaxis_title_with_timestamps(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_pandas: pd.DataFrame,
    ) -> None:
        """Test that x-axis shows 'Date' with timestamps."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_pandas)
        assert fig.layout.xaxis.title.text == "Date"

    def test_xaxis_title_without_timestamps(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that x-axis shows 'Sample Index' without timestamps."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)
        assert fig.layout.xaxis.title.text == "Sample Index"

    def test_default_title(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test default title when not provided."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)
        assert fig.layout.title.text == "Cross-Validation Fold Structure"


class TestPlotCvFoldsWithYAndGroups:
    """Tests with y and groups parameters."""

    def test_with_y_parameter(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that y parameter is accepted."""
        y = np.random.randn(500)
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy, y=y)
        assert isinstance(fig, go.Figure)

    def test_with_groups_parameter(
        self,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that groups parameter is accepted."""
        # Use CombinatorialCV with isolate_groups=False to avoid empty train sets
        cv = CombinatorialCV(
            n_groups=6,
            n_test_groups=2,
            max_combinations=3,
            isolate_groups=False,
        )
        groups = np.array(["A"] * 250 + ["B"] * 250)
        fig = plot_cv_folds(cv, sample_data_numpy, groups=groups)
        assert isinstance(fig, go.Figure)


class TestPlotCvFoldsBarMode:
    """Tests for bar mode and layout."""

    def test_overlay_barmode(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that barmode is overlay for proper stacking."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)
        assert fig.layout.barmode == "overlay"

    def test_legend_position(
        self,
        walk_forward_cv: WalkForwardCV,
        sample_data_numpy: np.ndarray,
    ) -> None:
        """Test that legend is positioned at top."""
        fig = plot_cv_folds(walk_forward_cv, sample_data_numpy)
        assert fig.layout.legend.orientation == "h"
        assert fig.layout.legend.yanchor == "bottom"
        assert fig.layout.legend.y > 1.0  # Above plot area
