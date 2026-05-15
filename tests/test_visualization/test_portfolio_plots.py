"""Tests for portfolio visualization helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.portfolio_analysis import RollingMetricsResult
from ml4t.diagnostic.visualization.portfolio import plot_rolling_sharpe


def _make_rolling_result(*, windows: list[int]) -> RollingMetricsResult:
    dates = pl.Series(
        "date",
        pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), eager=True),
    )
    sharpe = {
        window: pl.Series(f"sharpe_{window}d", np.linspace(0.5, 1.5, len(dates)))
        for window in windows
    }
    return RollingMetricsResult(windows=windows, dates=dates, sharpe=sharpe)


class TestRollingSharpePlots:
    """Regression coverage for rolling Sharpe helpers."""

    def test_plot_rolling_sharpe_honors_custom_windows_from_result(self):
        """Externally computed windows should drive the rendered traces."""
        rolling = _make_rolling_result(windows=[365])

        fig = plot_rolling_sharpe(rolling_result=rolling)

        assert isinstance(fig, go.Figure)
        traces = [trace for trace in fig.data if trace.type == "scatter" and len(trace.x) > 0]
        assert len(traces) == 1
        assert traces[0].name == "365d"

    def test_plot_rolling_sharpe_raises_for_non_matching_explicit_windows(self):
        """Explicit window requests should fail loudly when no series match."""
        rolling = _make_rolling_result(windows=[365])

        with pytest.raises(ValueError, match="no rolling Sharpe series matched"):
            plot_rolling_sharpe(rolling_result=rolling, windows=[63, 126, 252])

    def test_plot_rolling_sharpe_reference_annotations_fit_narrow_width(self):
        """Reference-line labels should stay inside the plot area on narrow figures."""
        rolling = _make_rolling_result(windows=[63])

        fig = plot_rolling_sharpe(rolling_result=rolling, width=400)

        assert isinstance(fig, go.Figure)
        assert fig.layout.margin.r == 80
        annotation_texts = {annotation.text for annotation in fig.layout.annotations}
        assert "Good (1.0)" in annotation_texts
        assert "Excellent (2.0)" in annotation_texts
