"""Tests for portfolio visualization helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.portfolio_analysis import RollingMetricsResult
from ml4t.diagnostic.visualization.portfolio import (
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)


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


def test_monthly_heatmap_aligns_partial_year_by_calendar_month() -> None:
    """A series beginning in June leaves January through May empty."""

    class PartialYearAnalysis:
        @staticmethod
        def get_monthly_returns_matrix() -> pl.DataFrame:
            return pl.DataFrame({"year": [2024], "6": [0.06], "7": [0.07]})

    fig = plot_monthly_returns_heatmap(PartialYearAnalysis())  # type: ignore[arg-type]
    values = np.asarray(fig.data[0].z, dtype=float)[0]

    assert np.isnan(values[:5]).all()
    assert values[5] == pytest.approx(0.06)
    assert values[6] == pytest.approx(0.07)
