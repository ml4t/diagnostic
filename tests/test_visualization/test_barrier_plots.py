"""Tests for barrier analysis visualization functions.

Tests cover:
- plot_hit_rate_heatmap: Heatmap visualization of hit rates
- plot_profit_factor_bar: Bar chart of profit factor by quantile
- plot_precision_recall_curve: Precision/recall curves with F1
- plot_time_to_target_box: Box plots of time to exit
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.results.barrier_results import (
    HitRateResult,
    PrecisionRecallResult,
    ProfitFactorResult,
    TimeToTargetResult,
)
from ml4t.diagnostic.visualization.barrier_plots import (
    plot_hit_rate_heatmap,
    plot_precision_recall_curve,
    plot_profit_factor_bar,
    plot_time_to_target_box,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def quantile_labels() -> list[str]:
    """Standard 10 decile labels."""
    return [f"D{i}" for i in range(1, 11)]


@pytest.fixture
def mock_hit_rate_result(quantile_labels: list[str]) -> HitRateResult:
    """Create mock HitRateResult with realistic values."""
    np.random.seed(42)
    n_quantiles = 10

    # TP rate increases with signal strength (D1 is worst, D10 is best)
    tp_rates = [0.30 + 0.04 * i for i in range(n_quantiles)]
    sl_rates = [0.35 - 0.02 * i for i in range(n_quantiles)]
    timeout_rates = [1.0 - tp - sl for tp, sl in zip(tp_rates, sl_rates)]

    # Counts per quantile (roughly equal)
    counts = [100] * n_quantiles

    return HitRateResult(
        n_quantiles=n_quantiles,
        quantile_labels=quantile_labels,
        hit_rate_tp=dict(zip(quantile_labels, tp_rates)),
        hit_rate_sl=dict(zip(quantile_labels, sl_rates)),
        hit_rate_timeout=dict(zip(quantile_labels, timeout_rates)),
        count_tp={q: int(c * tp) for q, c, tp in zip(quantile_labels, counts, tp_rates)},
        count_sl={q: int(c * sl) for q, c, sl in zip(quantile_labels, counts, sl_rates)},
        count_timeout={q: int(c * to) for q, c, to in zip(quantile_labels, counts, timeout_rates)},
        count_total=dict(zip(quantile_labels, counts)),
        chi2_statistic=45.6,
        chi2_p_value=0.0001,
        chi2_dof=18,
        is_significant=True,
        significance_level=0.05,
        overall_hit_rate_tp=0.50,
        overall_hit_rate_sl=0.25,
        overall_hit_rate_timeout=0.25,
        n_observations=1000,
        tp_rate_monotonic=True,
        tp_rate_direction="increasing",
        tp_rate_spearman=0.95,
    )


@pytest.fixture
def mock_profit_factor_result(quantile_labels: list[str]) -> ProfitFactorResult:
    """Create mock ProfitFactorResult with realistic values."""
    np.random.seed(42)
    n_quantiles = 10

    # PF increases with signal strength
    pf_values = [0.7 + 0.15 * i for i in range(n_quantiles)]
    avg_returns = [0.001 * i - 0.002 for i in range(n_quantiles)]
    sum_returns = [r * 100 for r in avg_returns]

    counts = [100] * n_quantiles

    return ProfitFactorResult(
        n_quantiles=n_quantiles,
        quantile_labels=quantile_labels,
        profit_factor=dict(zip(quantile_labels, pf_values)),
        sum_tp_returns={q: max(0, sr * 0.6) for q, sr in zip(quantile_labels, sum_returns)},
        sum_sl_returns={q: min(0, sr * 0.4) - 0.1 for q, sr in zip(quantile_labels, sum_returns)},
        sum_timeout_returns=dict.fromkeys(quantile_labels, 0.0),
        sum_all_returns=dict(zip(quantile_labels, sum_returns)),
        avg_tp_return=dict.fromkeys(quantile_labels, 0.015),
        avg_sl_return=dict.fromkeys(quantile_labels, -0.01),
        avg_return=dict(zip(quantile_labels, avg_returns)),
        count_tp=dict.fromkeys(quantile_labels, 50),
        count_sl=dict.fromkeys(quantile_labels, 30),
        count_total=dict(zip(quantile_labels, counts)),
        overall_profit_factor=1.35,
        overall_sum_returns=0.25,
        overall_avg_return=0.0025,
        n_observations=1000,
        pf_monotonic=True,
        pf_direction="increasing",
        pf_spearman=0.92,
    )


@pytest.fixture
def mock_precision_recall_result(quantile_labels: list[str]) -> PrecisionRecallResult:
    """Create mock PrecisionRecallResult with realistic values."""
    np.random.seed(42)
    n_quantiles = 10

    # Per-quantile precision decreases from D10 to D1
    precision_tp = {q: 0.7 - 0.03 * i for i, q in enumerate(quantile_labels)}
    recall_tp = dict.fromkeys(quantile_labels, 0.1)  # Equal recall per quantile
    lift_tp = {q: precision_tp[q] / 0.5 for q in quantile_labels}  # Relative to baseline

    # Cumulative metrics (from D10 down)
    cum_precision = {}
    cum_recall = {}
    cum_f1 = {}
    cum_lift = {}

    total_tp = 500
    cumulative_tp = 0
    cumulative_total = 0

    for _i, q in enumerate(reversed(quantile_labels)):
        tp_in_q = int(precision_tp[q] * 100)
        cumulative_tp += tp_in_q
        cumulative_total += 100

        cum_prec = cumulative_tp / cumulative_total if cumulative_total > 0 else 0
        cum_rec = cumulative_tp / total_tp if total_tp > 0 else 0
        f1 = 2 * cum_prec * cum_rec / (cum_prec + cum_rec) if (cum_prec + cum_rec) > 0 else 0

        cum_precision[q] = cum_prec
        cum_recall[q] = cum_rec
        cum_f1[q] = f1
        cum_lift[q] = cum_prec / 0.5

    return PrecisionRecallResult(
        n_quantiles=n_quantiles,
        quantile_labels=quantile_labels,
        precision_tp=precision_tp,
        recall_tp=recall_tp,
        cumulative_precision_tp=cum_precision,
        cumulative_recall_tp=cum_recall,
        cumulative_f1_tp=cum_f1,
        lift_tp=lift_tp,
        cumulative_lift_tp=cum_lift,
        baseline_tp_rate=0.5,
        total_tp_count=total_tp,
        n_observations=1000,
        best_f1_quantile="D7",
        best_f1_score=0.65,
    )


@pytest.fixture
def mock_time_to_target_result(quantile_labels: list[str]) -> TimeToTargetResult:
    """Create mock TimeToTargetResult with realistic values."""
    np.random.seed(42)
    n_quantiles = 10

    # Mean bars decreases for TP as signal improves
    mean_tp = {q: 15 - 0.5 * i for i, q in enumerate(quantile_labels)}
    mean_sl = {q: 12 + 0.2 * i for i, q in enumerate(quantile_labels)}
    mean_timeout = dict.fromkeys(quantile_labels, 20.0)
    mean_all = {q: (mean_tp[q] + mean_sl[q] + mean_timeout[q]) / 3 for q in quantile_labels}

    # Median similar to mean
    median_tp = {q: mean_tp[q] - 1 for q in quantile_labels}
    median_sl = {q: mean_sl[q] - 1 for q in quantile_labels}
    median_all = {q: mean_all[q] - 0.5 for q in quantile_labels}

    # Std dev
    std_tp = dict.fromkeys(quantile_labels, 3.0)
    std_sl = dict.fromkeys(quantile_labels, 2.5)
    std_all = dict.fromkeys(quantile_labels, 4.0)

    # Counts
    count_tp = {q: 40 + i * 2 for i, q in enumerate(quantile_labels)}
    count_sl = {q: 35 - i for i, q in enumerate(quantile_labels)}
    count_timeout = dict.fromkeys(quantile_labels, 25)

    # Speed analysis
    tp_faster = {q: mean_tp[q] < mean_sl[q] for q in quantile_labels}
    speed_adv = {q: mean_sl[q] - mean_tp[q] for q in quantile_labels}

    return TimeToTargetResult(
        n_quantiles=n_quantiles,
        quantile_labels=quantile_labels,
        mean_bars_tp=mean_tp,
        mean_bars_sl=mean_sl,
        mean_bars_timeout=mean_timeout,
        mean_bars_all=mean_all,
        median_bars_tp=median_tp,
        median_bars_sl=median_sl,
        median_bars_all=median_all,
        std_bars_tp=std_tp,
        std_bars_sl=std_sl,
        std_bars_all=std_all,
        count_tp=count_tp,
        count_sl=count_sl,
        count_timeout=count_timeout,
        overall_mean_bars=15.0,
        overall_median_bars=14.0,
        overall_mean_bars_tp=12.0,
        overall_mean_bars_sl=13.0,
        n_observations=1000,
        tp_faster_than_sl=tp_faster,
        speed_advantage_tp=speed_adv,
    )


# =============================================================================
# plot_hit_rate_heatmap Tests
# =============================================================================


class TestPlotHitRateHeatmap:
    """Tests for plot_hit_rate_heatmap()."""

    def test_basic_usage(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test basic heatmap creation."""
        fig = plot_hit_rate_heatmap(mock_hit_rate_result)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_heatmap_dimensions(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test heatmap has correct dimensions."""
        fig = plot_hit_rate_heatmap(mock_hit_rate_result)

        # 3 outcomes x 10 quantiles
        z_data = fig.data[0].z
        assert len(z_data) == 3  # TP, SL, Timeout
        assert len(z_data[0]) == 10  # 10 quantiles

    def test_show_counts_parameter(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test show_counts parameter."""
        # With counts
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, show_counts=True)
        text = fig.data[0].text[0][0]
        assert "n=" in text

        # Without counts
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, show_counts=False)
        text = fig.data[0].text[0][0]
        assert "n=" not in text

    def test_show_chi2_parameter(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test chi-square annotation parameter."""
        # With chi2
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, show_chi2=True)
        annotations = fig.layout.annotations
        chi2_annotations = [a for a in annotations if "Chi-Square" in a.text]
        assert len(chi2_annotations) == 1

        # Without chi2
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, show_chi2=False)
        annotations = fig.layout.annotations
        chi2_annotations = [a for a in annotations if "Chi-Square" in a.text]
        assert len(chi2_annotations) == 0

    def test_theme_parameter(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test theme customization."""
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, theme="dark")
        assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d"]

    def test_custom_dimensions(self, mock_hit_rate_result: HitRateResult) -> None:
        """Test custom width and height."""
        fig = plot_hit_rate_heatmap(mock_hit_rate_result, width=1200, height=500)
        assert fig.layout.width == 1200
        assert fig.layout.height == 500


# =============================================================================
# plot_profit_factor_bar Tests
# =============================================================================


class TestPlotProfitFactorBar:
    """Tests for plot_profit_factor_bar()."""

    def test_basic_usage(self, mock_profit_factor_result: ProfitFactorResult) -> None:
        """Test basic bar chart creation."""
        fig = plot_profit_factor_bar(mock_profit_factor_result)

        assert isinstance(fig, go.Figure)
        # Should have bar trace
        bar_traces = [t for t in fig.data if t.type == "bar"]
        assert len(bar_traces) == 1

    def test_bar_values(self, mock_profit_factor_result: ProfitFactorResult) -> None:
        """Test bar chart has correct values."""
        fig = plot_profit_factor_bar(mock_profit_factor_result)

        bar_trace = [t for t in fig.data if t.type == "bar"][0]
        assert len(bar_trace.x) == 10  # 10 quantiles
        assert len(bar_trace.y) == 10

    def test_reference_line_parameter(self, mock_profit_factor_result: ProfitFactorResult) -> None:
        """Test reference line at PF=1.0."""
        # With reference line
        fig = plot_profit_factor_bar(mock_profit_factor_result, show_reference_line=True)
        shapes = fig.layout.shapes
        hlines = [s for s in shapes if s.type == "line" and s.y0 == 1.0]
        assert len(hlines) == 1

        # Without reference line
        fig = plot_profit_factor_bar(mock_profit_factor_result, show_reference_line=False)
        shapes = fig.layout.shapes
        hlines = [s for s in shapes if s.type == "line" and s.y0 == 1.0]
        assert len(hlines) == 0

    def test_show_average_return_parameter(
        self, mock_profit_factor_result: ProfitFactorResult
    ) -> None:
        """Test secondary axis for average return."""
        # With average return
        fig = plot_profit_factor_bar(mock_profit_factor_result, show_average_return=True)
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) == 1
        assert fig.layout.yaxis2 is not None

        # Without average return
        fig = plot_profit_factor_bar(mock_profit_factor_result, show_average_return=False)
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) == 0

    def test_theme_parameter(self, mock_profit_factor_result: ProfitFactorResult) -> None:
        """Test theme customization."""
        fig = plot_profit_factor_bar(mock_profit_factor_result, theme="presentation")
        # Presentation theme has larger fonts
        assert fig.layout.font.size >= 16

    def test_custom_dimensions(self, mock_profit_factor_result: ProfitFactorResult) -> None:
        """Test custom width and height."""
        fig = plot_profit_factor_bar(mock_profit_factor_result, width=1000, height=600)
        assert fig.layout.width == 1000
        assert fig.layout.height == 600


# =============================================================================
# plot_precision_recall_curve Tests
# =============================================================================


class TestPlotPrecisionRecallCurve:
    """Tests for plot_precision_recall_curve()."""

    def test_basic_usage(self, mock_precision_recall_result: PrecisionRecallResult) -> None:
        """Test basic curve creation."""
        fig = plot_precision_recall_curve(mock_precision_recall_result)

        assert isinstance(fig, go.Figure)
        # Should have multiple line traces
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) >= 3  # Precision, Recall, F1

    def test_trace_names(self, mock_precision_recall_result: PrecisionRecallResult) -> None:
        """Test that traces have correct names."""
        fig = plot_precision_recall_curve(mock_precision_recall_result)

        trace_names = [t.name for t in fig.data]
        assert "Cumulative Precision" in trace_names
        assert "Cumulative Recall" in trace_names
        assert "Cumulative F1" in trace_names

    def test_show_f1_peak_parameter(
        self, mock_precision_recall_result: PrecisionRecallResult
    ) -> None:
        """Test F1 peak marker."""
        # With F1 peak
        fig = plot_precision_recall_curve(mock_precision_recall_result, show_f1_peak=True)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert any("Best F1" in name for name in trace_names)

        # Without F1 peak
        fig = plot_precision_recall_curve(mock_precision_recall_result, show_f1_peak=False)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert not any("Best F1" in name for name in trace_names)

    def test_show_lift_parameter(self, mock_precision_recall_result: PrecisionRecallResult) -> None:
        """Test lift curve on secondary axis."""
        # With lift
        fig = plot_precision_recall_curve(mock_precision_recall_result, show_lift=True)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert any("Lift" in name for name in trace_names)
        assert fig.layout.yaxis2 is not None

        # Without lift
        fig = plot_precision_recall_curve(mock_precision_recall_result, show_lift=False)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert not any("Lift" in name for name in trace_names)

    def test_baseline_line(self, mock_precision_recall_result: PrecisionRecallResult) -> None:
        """Test baseline horizontal line is present."""
        fig = plot_precision_recall_curve(mock_precision_recall_result)
        shapes = fig.layout.shapes
        # Should have at least one horizontal line (baseline)
        assert len(shapes) >= 1

    def test_theme_parameter(self, mock_precision_recall_result: PrecisionRecallResult) -> None:
        """Test theme customization."""
        fig = plot_precision_recall_curve(mock_precision_recall_result, theme="dark")
        assert fig.layout.paper_bgcolor.lower() in ["#1e1e1e", "#2d2d2d"]


# =============================================================================
# plot_time_to_target_box Tests
# =============================================================================


class TestPlotTimeToTargetBox:
    """Tests for plot_time_to_target_box()."""

    def test_basic_usage(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test basic box plot creation."""
        fig = plot_time_to_target_box(mock_time_to_target_result)

        assert isinstance(fig, go.Figure)
        # Should have box traces
        box_traces = [t for t in fig.data if t.type == "box"]
        assert len(box_traces) == 10  # One per quantile

    def test_outcome_type_all(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test all outcomes mode."""
        fig = plot_time_to_target_box(mock_time_to_target_result, outcome_type="all")
        assert "All Outcomes" in fig.layout.title.text

    def test_outcome_type_tp(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test TP-only mode."""
        fig = plot_time_to_target_box(mock_time_to_target_result, outcome_type="tp")
        assert "Take-Profit" in fig.layout.title.text

    def test_outcome_type_sl(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test SL-only mode."""
        fig = plot_time_to_target_box(mock_time_to_target_result, outcome_type="sl")
        assert "Stop-Loss" in fig.layout.title.text

    def test_outcome_type_comparison(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test comparison mode (TP vs SL side by side)."""
        fig = plot_time_to_target_box(mock_time_to_target_result, outcome_type="comparison")

        assert "TP vs SL" in fig.layout.title.text
        # Should have grouped boxes
        assert fig.layout.boxmode == "group"

    def test_show_median_line_parameter(
        self, mock_time_to_target_result: TimeToTargetResult
    ) -> None:
        """Test median line parameter."""
        # With median line
        fig = plot_time_to_target_box(mock_time_to_target_result, show_median_line=True)
        shapes = fig.layout.shapes
        assert len(shapes) >= 1

        # Without median line
        fig = plot_time_to_target_box(mock_time_to_target_result, show_median_line=False)
        shapes = fig.layout.shapes
        # Should have no horizontal lines
        hlines = [s for s in shapes if s.type == "line"]
        assert len(hlines) == 0

    def test_theme_parameter(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test theme customization."""
        fig = plot_time_to_target_box(mock_time_to_target_result, theme="print")
        # Print theme uses serif font
        assert "Times" in fig.layout.font.family or "serif" in fig.layout.font.family.lower()

    def test_custom_dimensions(self, mock_time_to_target_result: TimeToTargetResult) -> None:
        """Test custom width and height."""
        fig = plot_time_to_target_box(mock_time_to_target_result, width=1100, height=550)
        assert fig.layout.width == 1100
        assert fig.layout.height == 550


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test that all functions can be imported from visualization module."""

    def test_import_from_barrier_plots(self) -> None:
        """Test direct import from barrier_plots module."""
        from ml4t.diagnostic.visualization.barrier_plots import (
            plot_hit_rate_heatmap,
            plot_precision_recall_curve,
            plot_profit_factor_bar,
            plot_time_to_target_box,
        )

        assert callable(plot_hit_rate_heatmap)
        assert callable(plot_profit_factor_bar)
        assert callable(plot_precision_recall_curve)
        assert callable(plot_time_to_target_box)

    def test_import_from_visualization_package(self) -> None:
        """Test import from main visualization package."""
        from ml4t.diagnostic.visualization import (
            plot_hit_rate_heatmap,
            plot_precision_recall_curve,
            plot_profit_factor_bar,
            plot_time_to_target_box,
        )

        assert callable(plot_hit_rate_heatmap)
        assert callable(plot_profit_factor_bar)
        assert callable(plot_precision_recall_curve)
        assert callable(plot_time_to_target_box)
