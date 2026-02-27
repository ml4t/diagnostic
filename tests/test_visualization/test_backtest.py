"""Tests for ml4t.diagnostic.visualization.backtest module.

Comprehensive tests for backtest tearsheet visualizations including:
- Executive summary with KPI cards and traffic lights
- Trade analysis plots (MFE/MAE, exit reasons, waterfall)
- Cost attribution (waterfall, sensitivity, by asset)
- Statistical validity (DSR gauge, confidence intervals, RAS)
- Unified tearsheet generation
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics() -> dict:
    """Create sample backtest metrics."""
    return {
        "n_trades": 100,
        "total_pnl": 15000.0,
        "win_rate": 0.52,
        "profit_factor": 1.85,
        "sharpe_ratio": 1.95,
        "max_drawdown": -5000.0,
        "avg_trade": 150.0,
        "total_commission": 500.0,
        "total_slippage": 250.0,
        "cagr": 0.18,
        "calmar_ratio": 2.5,
    }


@pytest.fixture
def sample_trades() -> pl.DataFrame:
    """Create sample trades DataFrame."""
    np.random.seed(42)
    n_trades = 50

    # Generate realistic trade data
    pnl = np.random.normal(50, 200, n_trades)
    entry_prices = 100 + np.random.normal(0, 5, n_trades)
    exit_prices = entry_prices + pnl / 100  # Simple price delta

    return pl.DataFrame(
        {
            "symbol": [f"ASSET_{i % 5}" for i in range(n_trades)],
            "entry_time": pl.datetime_range(
                start=pl.datetime(2023, 1, 1),
                end=pl.datetime(2023, 6, 30),
                interval="1d",
                eager=True,
            )[:n_trades],
            "exit_time": pl.datetime_range(
                start=pl.datetime(2023, 1, 5),
                end=pl.datetime(2023, 7, 4),
                interval="1d",
                eager=True,
            )[:n_trades],
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "quantity": np.random.randint(10, 100, n_trades),
            "direction": np.random.choice(["long", "short"], n_trades),
            "pnl": pnl,
            "pnl_percent": pnl / 1000,  # Simplified return
            "bars_held": np.random.randint(1, 30, n_trades),
            "commission": np.random.uniform(1, 10, n_trades),
            "slippage": np.random.uniform(0.5, 5, n_trades),
            "mfe": np.abs(np.random.normal(0.02, 0.01, n_trades)),  # Positive MFE
            "mae": -np.abs(np.random.normal(0.01, 0.005, n_trades)),  # Negative MAE
            "exit_reason": np.random.choice(
                ["take_profit", "stop_loss", "timeout", "signal_reversal"],
                n_trades,
            ),
        }
    )


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Create sample returns array."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)


# =============================================================================
# Executive Summary Tests
# =============================================================================


class TestExecutiveSummary:
    """Tests for executive_summary.py functions."""

    def test_create_executive_summary_basic(self, sample_metrics):
        """Test basic executive summary creation."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary

        fig = create_executive_summary(sample_metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6  # Default KPI selection
        assert all(trace.type == "indicator" for trace in fig.data)
        values = {trace.value for trace in fig.data}
        assert sample_metrics["sharpe_ratio"] in values
        assert sample_metrics["win_rate"] in values
        assert fig.layout.title is not None or fig.layout.annotations

    def test_create_executive_summary_themes(self, sample_metrics):
        """Test executive summary with different themes."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary

        for theme in ["default", "dark", "print", "presentation"]:
            fig = create_executive_summary(sample_metrics, theme=theme)
            assert isinstance(fig, go.Figure)

    def test_create_executive_summary_dimensions(self, sample_metrics):
        """Test executive summary respects height/width."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary

        fig = create_executive_summary(sample_metrics, height=600, width=1200)

        # Layout should have dimensions
        assert fig.layout.height == 600
        assert fig.layout.width == 1200

    def test_get_traffic_light_color(self):
        """Test traffic light color selection."""
        from ml4t.diagnostic.visualization.backtest import get_traffic_light_color

        # Test with standard metrics
        # Win rate - higher is better
        color = get_traffic_light_color(0.55, "win_rate")
        assert color in ["green", "yellow", "red", "neutral"]

        # Sharpe - higher is better
        color = get_traffic_light_color(1.5, "sharpe_ratio")
        assert color in ["green", "yellow", "red", "neutral"]

        # Max drawdown - lower magnitude is better
        color = get_traffic_light_color(-0.15, "max_drawdown")
        assert color in ["green", "yellow", "red", "neutral"]


# =============================================================================
# Trade Plots Tests
# =============================================================================


class TestTradePlots:
    """Tests for trade_plots.py functions."""

    def test_plot_mfe_mae_scatter_basic(self, sample_trades):
        """Test basic MFE/MAE scatter plot."""
        from ml4t.diagnostic.visualization.backtest import plot_mfe_mae_scatter

        fig = plot_mfe_mae_scatter(sample_trades)

        assert isinstance(fig, go.Figure)
        marker_trace = next(
            trace for trace in fig.data if trace.type == "scatter" and trace.mode == "markers"
        )
        assert len(marker_trace.x) == sample_trades.height
        assert len(marker_trace.y) == sample_trades.height
        assert np.all(np.array(marker_trace.x) >= 0.0)  # MAE is plotted as absolute values
        assert np.all(np.array(marker_trace.y) >= 0.0)  # Sample fixture uses positive MFE values
        assert any(
            getattr(trace, "name", "") == "Perfect Efficiency (Exit at MFE)" for trace in fig.data
        )

    def test_plot_mfe_mae_scatter_color_by(self, sample_trades):
        """Test MFE/MAE scatter with different color options."""
        from ml4t.diagnostic.visualization.backtest import plot_mfe_mae_scatter

        for color_by in ["pnl", "exit_reason", "symbol", None]:
            fig = plot_mfe_mae_scatter(sample_trades, color_by=color_by)
            assert isinstance(fig, go.Figure)

    def test_plot_exit_reason_breakdown_sunburst(self, sample_trades):
        """Test exit reason sunburst chart."""
        from ml4t.diagnostic.visualization.backtest import plot_exit_reason_breakdown

        fig = plot_exit_reason_breakdown(sample_trades, chart_type="sunburst")

        assert isinstance(fig, go.Figure)
        # Should have Sunburst trace
        assert any(isinstance(trace, go.Sunburst) for trace in fig.data)

    def test_plot_exit_reason_breakdown_treemap(self, sample_trades):
        """Test exit reason treemap chart."""
        from ml4t.diagnostic.visualization.backtest import plot_exit_reason_breakdown

        fig = plot_exit_reason_breakdown(sample_trades, chart_type="treemap")

        assert isinstance(fig, go.Figure)
        assert any(isinstance(trace, go.Treemap) for trace in fig.data)

    def test_plot_exit_reason_breakdown_bar(self, sample_trades):
        """Test exit reason bar chart."""
        from ml4t.diagnostic.visualization.backtest import plot_exit_reason_breakdown

        fig = plot_exit_reason_breakdown(sample_trades, chart_type="bar")

        assert isinstance(fig, go.Figure)
        assert any(isinstance(trace, go.Bar) for trace in fig.data)

    def test_plot_trade_waterfall(self, sample_trades):
        """Test trade waterfall plot."""
        from ml4t.diagnostic.visualization.backtest import plot_trade_waterfall

        fig = plot_trade_waterfall(sample_trades)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # PnL bars + cumulative line
        assert fig.data[0].type == "bar"
        assert fig.data[0].name == "Trade PnL"
        assert fig.data[1].type == "scatter"
        assert fig.data[1].name == "Cumulative Equity"
        expected_final_equity = 100000.0 + float(sample_trades["pnl"].sum())
        assert fig.data[1].y[-1] == pytest.approx(expected_final_equity)

    def test_plot_trade_waterfall_n_trades(self, sample_trades):
        """Test trade waterfall with limited trades."""
        from ml4t.diagnostic.visualization.backtest import plot_trade_waterfall

        fig = plot_trade_waterfall(sample_trades, n_trades=10)

        assert isinstance(fig, go.Figure)
        # Should have limited number of bars
        if fig.data:
            assert len(fig.data[0].x) <= 11  # 10 trades + total

    def test_plot_trade_duration_distribution(self, sample_trades):
        """Test trade duration histogram."""
        from ml4t.diagnostic.visualization.backtest import plot_trade_duration_distribution

        fig = plot_trade_duration_distribution(sample_trades)

        assert isinstance(fig, go.Figure)
        assert any(isinstance(trace, go.Histogram) for trace in fig.data)


# =============================================================================
# Cost Attribution Tests
# =============================================================================


class TestCostAttribution:
    """Tests for cost_attribution.py functions."""

    def test_plot_cost_waterfall(self):
        """Test cost waterfall chart."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_waterfall

        fig = plot_cost_waterfall(
            gross_pnl=50000.0,
            commission=1000.0,
            slippage=500.0,
        )

        assert isinstance(fig, go.Figure)
        # Should have Waterfall trace
        assert any(isinstance(trace, go.Waterfall) for trace in fig.data)
        trace = fig.data[0]
        assert list(trace.x) == ["Gross PnL", "Commission", "Slippage", "Net PnL"]
        assert list(trace.measure) == ["absolute", "relative", "relative", "total"]
        assert trace.y[0] == 50000.0
        assert trace.y[-1] == pytest.approx(48500.0)

    def test_plot_cost_waterfall_with_other_costs(self):
        """Test cost waterfall with additional cost categories."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_waterfall

        fig = plot_cost_waterfall(
            gross_pnl=50000.0,
            commission=1000.0,
            slippage=500.0,
            other_costs={"Financing": 200.0, "Exchange Fees": 100.0},
        )

        assert isinstance(fig, go.Figure)

    def test_plot_cost_sensitivity(self, sample_returns):
        """Test cost sensitivity analysis."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_sensitivity

        fig = plot_cost_sensitivity(
            returns=sample_returns,
            base_costs_bps=10.0,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_cost_sensitivity_custom_multipliers(self, sample_returns):
        """Test cost sensitivity with custom multipliers."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_sensitivity

        fig = plot_cost_sensitivity(
            returns=sample_returns,
            base_costs_bps=10.0,
            cost_multipliers=[0, 1, 2, 5, 10],
        )

        assert isinstance(fig, go.Figure)

    def test_plot_cost_by_asset(self, sample_trades):
        """Test cost by asset bar chart."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_by_asset

        # Add cost column
        trades_with_cost = sample_trades.with_columns(
            (pl.col("commission") + pl.col("slippage")).alias("cost")
        )

        fig = plot_cost_by_asset(trades_with_cost, cost_column="cost")

        assert isinstance(fig, go.Figure)


# =============================================================================
# Statistical Validity Tests
# =============================================================================


class TestStatisticalValidity:
    """Tests for statistical_validity.py functions."""

    def test_plot_dsr_gauge_basic(self):
        """Test DSR gauge chart."""
        from ml4t.diagnostic.visualization.backtest import plot_dsr_gauge

        fig = plot_dsr_gauge(
            dsr_probability=0.03,
            observed_sharpe=2.1,
        )

        assert isinstance(fig, go.Figure)
        # Should have Indicator trace
        assert any(isinstance(trace, go.Indicator) for trace in fig.data)

    def test_plot_dsr_gauge_with_extras(self):
        """Test DSR gauge with additional info."""
        from ml4t.diagnostic.visualization.backtest import plot_dsr_gauge

        fig = plot_dsr_gauge(
            dsr_probability=0.03,
            observed_sharpe=2.1,
            expected_max_sharpe=1.5,
            n_trials=100,
        )

        assert isinstance(fig, go.Figure)
        # Should have annotations for the extra info
        assert len(fig.layout.annotations) > 0

    def test_plot_confidence_intervals(self):
        """Test confidence interval forest plot."""
        from ml4t.diagnostic.visualization.backtest import plot_confidence_intervals

        metrics = {
            "Sharpe": {"point": 1.5, "lower_95": 0.8, "upper_95": 2.2},
            "CAGR": {"point": 0.15, "lower_95": 0.08, "upper_95": 0.22},
            "Max DD": {"point": -0.12, "lower_95": -0.18, "upper_95": -0.06},
        }

        fig = plot_confidence_intervals(metrics)

        assert isinstance(fig, go.Figure)
        # Provided data only includes 95% intervals + one point estimate per metric
        assert len(fig.data) == 6
        legend_names = {trace.name for trace in fig.data if trace.name}
        assert "95% CI" in legend_names
        assert "Point Estimate" in legend_names
        assert list(fig.layout.yaxis.ticktext) == ["Sharpe", "CAGR", "Max DD"]

    def test_plot_confidence_intervals_orientation(self):
        """Test CI plot with different orientations."""
        from ml4t.diagnostic.visualization.backtest import plot_confidence_intervals

        metrics = {
            "Sharpe": {"point": 1.5, "lower_95": 0.8, "upper_95": 2.2},
        }

        for orientation in ["h", "v"]:
            fig = plot_confidence_intervals(metrics, orientation=orientation)
            assert isinstance(fig, go.Figure)

    def test_plot_ras_analysis(self):
        """Test RAS analysis waterfall."""
        from ml4t.diagnostic.visualization.backtest import plot_ras_analysis

        fig = plot_ras_analysis(
            original_ic=0.05,
            adjusted_ic=0.03,
            rademacher_complexity=0.02,
        )

        assert isinstance(fig, go.Figure)

    def test_plot_minimum_track_record(self):
        """Test MinTRL visualization."""
        from ml4t.diagnostic.visualization.backtest import plot_minimum_track_record

        fig = plot_minimum_track_record(
            observed_sharpe=1.8,
            current_periods=500,  # ~2 years of daily data
            sr_benchmark=0.5,
        )

        assert isinstance(fig, go.Figure)


# =============================================================================
# Tearsheet Generation Tests
# =============================================================================


class TestTearsheetGeneration:
    """Tests for tearsheet.py functions."""

    def test_generate_backtest_tearsheet_full(self, sample_metrics, sample_trades, sample_returns):
        """Test full tearsheet generation."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            metrics=sample_metrics,
            trades=sample_trades,
            returns=sample_returns,
            template="full",
        )

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html>" in html.lower() or "<!doctype" in html.lower()
        # Should contain embedded Plotly charts
        assert "plotly" in html.lower()

    def test_generate_backtest_tearsheet_templates(
        self, sample_metrics, sample_trades, sample_returns
    ):
        """Test tearsheet with different templates."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        for template in ["quant_trader", "hedge_fund", "risk_manager", "full"]:
            html = generate_backtest_tearsheet(
                metrics=sample_metrics,
                trades=sample_trades,
                returns=sample_returns,
                template=template,
            )
            assert isinstance(html, str)
            assert len(html) > 0

    def test_generate_backtest_tearsheet_themes(
        self, sample_metrics, sample_trades, sample_returns
    ):
        """Test tearsheet with different themes."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        for theme in ["default", "dark"]:
            html = generate_backtest_tearsheet(
                metrics=sample_metrics,
                trades=sample_trades,
                returns=sample_returns,
                theme=theme,
            )
            assert isinstance(html, str)
            assert len(html) > 0

    def test_generate_backtest_tearsheet_minimal(self, sample_metrics):
        """Test tearsheet with minimal data."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        # Only metrics, no trades or returns
        html = generate_backtest_tearsheet(
            metrics=sample_metrics,
        )

        assert isinstance(html, str)
        assert len(html) > 0


# =============================================================================
# Template System Tests
# =============================================================================


class TestTemplateSystem:
    """Tests for template_system.py functions."""

    def test_get_template_valid(self):
        """Test getting valid templates."""
        from ml4t.diagnostic.visualization.backtest import get_template

        for template_name in ["quant_trader", "hedge_fund", "risk_manager", "full"]:
            template = get_template(template_name)

            # Template is a dataclass, not a dict
            assert hasattr(template, "name")
            assert hasattr(template, "sections")
            assert template.name == template_name

    def test_get_template_sections(self):
        """Test template section configuration."""
        from ml4t.diagnostic.visualization.backtest import get_template

        template = get_template("quant_trader")
        sections = template.sections

        # Should have section list
        assert isinstance(sections, list)
        assert len(sections) > 0

        # Check section has expected attributes
        first_section = sections[0]
        assert hasattr(first_section, "name")
        assert hasattr(first_section, "enabled")

    def test_template_priority_ordering(self):
        """Test templates have different section priorities."""
        from ml4t.diagnostic.visualization.backtest import get_template

        quant = get_template("quant_trader")
        risk = get_template("risk_manager")

        # Templates should have different section configurations
        quant_sections = {s.name: s.enabled for s in quant.sections}
        risk_sections = {s.name: s.enabled for s in risk.sections}

        # At least some sections should differ in enabled state
        assert quant_sections != risk_sections


# =============================================================================
# Interactive Controls Tests
# =============================================================================


class TestInteractiveControls:
    """Tests for interactive_controls.py functions."""

    def test_get_date_range_html(self):
        """Test date range picker HTML generation."""
        from ml4t.diagnostic.visualization.backtest import get_date_range_html

        html = get_date_range_html()

        assert isinstance(html, str)
        assert "date" in html.lower()

    def test_get_theme_switcher_html(self):
        """Test theme switcher HTML generation."""
        from ml4t.diagnostic.visualization.backtest import get_theme_switcher_html

        html = get_theme_switcher_html()

        assert isinstance(html, str)
        # Should include theme options
        assert "default" in html.lower() or "theme" in html.lower()

    def test_get_section_navigation_html(self):
        """Test section navigation HTML generation."""
        from ml4t.diagnostic.visualization.backtest import get_section_navigation_html

        # Function expects list of dicts with id and title
        sections = [
            {"id": "summary", "title": "Summary"},
            {"id": "trades", "title": "Trades"},
            {"id": "costs", "title": "Costs"},
            {"id": "statistics", "title": "Statistics"},
        ]
        html = get_section_navigation_html(sections)

        assert isinstance(html, str)
        for section in sections:
            assert section["title"].lower() in html.lower()


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_trades_dataframe(self):
        """Test handling of empty trades DataFrame."""
        from ml4t.diagnostic.visualization.backtest import plot_mfe_mae_scatter

        empty_trades = pl.DataFrame(
            {
                "symbol": [],
                "pnl": [],
                "mfe": [],
                "mae": [],
                "exit_reason": [],
            }
        ).cast(
            {
                "symbol": pl.Utf8,
                "pnl": pl.Float64,
                "mfe": pl.Float64,
                "mae": pl.Float64,
                "exit_reason": pl.Utf8,
            }
        )

        # Should handle gracefully (either return figure or raise informative error)
        try:
            fig = plot_mfe_mae_scatter(empty_trades)
            assert isinstance(fig, go.Figure)
        except ValueError as e:
            # Acceptable to raise ValueError for empty data
            assert "empty" in str(e).lower() or "no" in str(e).lower()

    def test_single_trade(self, sample_trades):
        """Test handling of single trade."""
        from ml4t.diagnostic.visualization.backtest import plot_trade_waterfall

        single_trade = sample_trades.head(1)
        fig = plot_trade_waterfall(single_trade)

        assert isinstance(fig, go.Figure)

    def test_negative_metrics(self):
        """Test handling of negative/losing strategy metrics."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary

        losing_metrics = {
            "n_trades": 50,
            "total_pnl": -10000.0,
            "win_rate": 0.35,
            "profit_factor": 0.65,
            "sharpe_ratio": -0.8,
            "max_drawdown": -25000.0,
            "avg_trade": -200.0,
        }

        fig = create_executive_summary(losing_metrics)
        assert isinstance(fig, go.Figure)

    def test_extreme_values(self):
        """Test handling of extreme metric values."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_waterfall

        # Very large values
        fig = plot_cost_waterfall(
            gross_pnl=1e9,
            commission=1e6,
            slippage=5e5,
        )
        assert isinstance(fig, go.Figure)

        # Very small values
        fig = plot_cost_waterfall(
            gross_pnl=0.01,
            commission=0.001,
            slippage=0.0005,
        )
        assert isinstance(fig, go.Figure)

    def test_zero_costs(self):
        """Test handling of zero transaction costs."""
        from ml4t.diagnostic.visualization.backtest import plot_cost_waterfall

        fig = plot_cost_waterfall(
            gross_pnl=10000.0,
            commission=0.0,
            slippage=0.0,
        )

        assert isinstance(fig, go.Figure)
