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
from ml4t.backtest import BacktestResult
from ml4t.backtest.types import Fill, OrderSide, Trade

from ml4t.diagnostic.integration import BacktestReportMetadata

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


@pytest.fixture
def sample_backtest_profile():
    """Create a sample BacktestProfile."""
    from datetime import datetime, timedelta

    from ml4t.diagnostic.integration import analyze_backtest_result

    base = datetime(2024, 1, 1, 9, 30)
    result = BacktestResult(
        trades=[
            Trade(
                symbol="AAPL",
                entry_time=base,
                exit_time=base + timedelta(days=1),
                entry_price=100.0,
                exit_price=102.0,
                quantity=10.0,
                pnl=20.0,
                pnl_percent=0.02,
                bars_held=1,
                fees=1.0,
                exit_slippage=0.1,
            ),
            Trade(
                symbol="MSFT",
                entry_time=base + timedelta(days=1),
                exit_time=base + timedelta(days=2),
                entry_price=200.0,
                exit_price=195.0,
                quantity=10.0,
                pnl=-50.0,
                pnl_percent=-0.025,
                bars_held=1,
                fees=2.0,
                exit_slippage=0.2,
            ),
        ],
        fills=[
            Fill(
                order_id="1",
                asset="AAPL",
                side=OrderSide.BUY,
                quantity=10.0,
                price=100.0,
                timestamp=base,
                rebalance_id="rebalance-1",
                commission=1.0,
                slippage=0.1,
                quote_mid_price=100.0,
            ),
            Fill(
                order_id="2",
                asset="MSFT",
                side=OrderSide.BUY,
                quantity=10.0,
                price=200.0,
                timestamp=base,
                rebalance_id="rebalance-1",
                commission=2.0,
                slippage=0.2,
            ),
        ],
        portfolio_state=[
            (base, 1000.0, 0.0, 1000.0, 1000.0, 2),
            (base + timedelta(days=1), 1020.0, 20.0, 1000.0, 1000.0, 2),
            (base + timedelta(days=2), 970.0, 970.0, 0.0, 0.0, 0),
        ],
        equity_curve=[
            (base, 1000.0),
            (base + timedelta(days=1), 1020.0),
            (base + timedelta(days=2), 970.0),
        ],
        metrics={},
    )
    return analyze_backtest_result(result, calendar="NYSE")


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

    def test_create_executive_summary_supports_profile_kpis(self):
        """Test executive summary can render profile-native KPI metrics."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary

        fig = create_executive_summary(
            {
                "sharpe_ratio": 1.2,
                "total_return": 0.12,
                "max_drawdown": 0.08,
                "avg_turnover": 0.45,
                "num_rebalance_events": 24,
                "avg_open_positions": 12.0,
            },
            selected_metrics=[
                "sharpe_ratio",
                "avg_turnover",
                "num_rebalance_events",
                "avg_open_positions",
            ],
        )

        assert isinstance(fig, go.Figure)
        values = {trace.value for trace in fig.data}
        assert 0.45 in values
        assert 24 in values
        assert 12.0 in values

    def test_create_executive_summary_html_supports_benchmark_context(self, sample_metrics):
        """Test HTML executive strip renders compact benchmark context."""
        from ml4t.diagnostic.visualization.backtest import create_executive_summary_html

        html = create_executive_summary_html(
            sample_metrics,
            selected_metrics=["sharpe_ratio", "cagr", "profit_factor"],
            benchmark_metrics={"sharpe_ratio": 1.25, "cagr": 0.11},
            benchmark_label="SPY",
        )

        assert "executive-strip" in html
        assert "Sharpe Ratio" in html
        assert "SPY" in html
        assert "Spread" in html

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

    def test_create_key_insights_uses_profile_native_signals(self, sample_backtest_profile):
        """Test profile-native burden and availability insights."""
        from ml4t.diagnostic.visualization.backtest import create_key_insights

        insights = create_key_insights(
            sample_backtest_profile.summary, profile=sample_backtest_profile
        )
        messages = [insight.message for insight in insights]

        assert any("implementation cost" in message for message in messages)
        assert any("Quote-aware execution audit" in message for message in messages)

    def test_create_key_metrics_table_html_with_benchmark(self, sample_metrics):
        """Test HTML metrics table renders dense grid with available metrics."""
        from ml4t.diagnostic.visualization.backtest import create_key_metrics_table_html

        html = create_key_metrics_table_html(
            sample_metrics,
            benchmark_metrics={"sharpe_ratio": 1.25, "cagr": 0.11, "max_drawdown": 0.09},
            benchmark_label="SPY",
        )

        # Dense grid layout — no table elements, uses flexbox
        assert "display:flex" in html
        assert "Sharpe Ratio" in html
        # No subjective labels
        assert "Better" not in html
        assert "Worse" not in html
        assert "Favorable" not in html

    def test_create_key_metrics_table_html_tolerates_non_numeric_benchmark_values(
        self,
        sample_metrics,
    ):
        """Test HTML metrics table does not fail on non-numeric benchmark values."""
        from ml4t.diagnostic.visualization.backtest import create_key_metrics_table_html

        html = create_key_metrics_table_html(
            sample_metrics,
            benchmark_metrics={"sharpe_ratio": "N/A", "cagr": 0.11},
            benchmark_label="SPY",
        )

        assert "display:flex" in html
        assert "Sharpe Ratio" in html


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
        assert "workspace-tabs" in html
        assert 'data-workspace="overview"' in html
        assert 'data-workspace="performance"' in html
        assert "report-section" in html
        assert "executive-strip" in html
        assert "Benchmark Context" not in html
        assert "Backtest Diagnostics" not in html
        assert "State-of-the-art" not in html
        # Trade waterfall and position size disabled per content architecture review
        assert "Trade-by-Trade PnL" not in html
        assert "Position Size Analysis" not in html

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

    def test_generate_backtest_tearsheet_from_profile(self, sample_backtest_profile):
        """Test tearsheet generation directly from BacktestProfile."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            profile=sample_backtest_profile,
            template="hedge_fund",
        )

        assert isinstance(html, str)
        assert "Activity" in html
        assert "Exposure" in html
        assert "Drawdown Anatomy" in html
        assert "Cost Attribution" in html
        assert "Rolling Performance" in html
        assert 'data-workspace="trading"' in html
        assert "report-section" in html

    def test_generate_backtest_tearsheet_renders_ml_workspace_when_prediction_surface_exists(
        self,
        sample_backtest_profile,
    ):
        """Test ML workspace becomes visible when prediction and signal surfaces exist."""
        from datetime import datetime

        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        sample_backtest_profile.result.to_predictions_df = lambda: pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30)],
                "asset": ["AAPL"],
                "prediction_value": [0.7],
            }
        )
        sample_backtest_profile.result.to_signals_df = lambda: pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30)],
                "asset": ["AAPL"],
                "signal_value": [1.0],
                "selected": [True],
            }
        )
        sample_backtest_profile.result.strategy_metadata = {
            "strategy_type": "ml",
            "mapping_name": "threshold",
        }

        html = generate_backtest_tearsheet(
            profile=sample_backtest_profile,
            template="full",
        )

        assert 'data-workspace="ml"' in html
        assert "Prediction Translation" in html
        assert "Prediction vs Trade Outcomes" in html
        assert "Predictions" in html
        assert "Signals" in html

    def test_generate_backtest_tearsheet_renders_dense_overview_without_details(
        self,
        sample_backtest_profile,
    ):
        """Test the integrated dashboard renders a visible overview table and no details blocks."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            profile=sample_backtest_profile,
            benchmark_returns=np.array([0.01, -0.02, 0.005]),
            benchmark_name="SPY",
            template="full",
        )

        assert "metrics-table" in html
        assert "Overview" in html
        assert 'data-workspace="performance"' in html

    def test_generate_backtest_tearsheet_with_benchmark_summary(
        self,
        sample_metrics,
        sample_trades,
        sample_returns,
    ):
        """Test benchmark-aware tearsheet shows benchmark table context."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        benchmark = sample_returns * 0.8
        html = generate_backtest_tearsheet(
            metrics=sample_metrics,
            trades=sample_trades,
            returns=sample_returns,
            benchmark_returns=benchmark,
            benchmark_name="SPY",
            template="full",
        )

        assert "SPY" in html
        # key_metrics_table disabled — benchmark context no longer in main table
        assert "Assessment" not in html
        assert "Better" not in html
        assert "Worse" not in html

    def test_generate_backtest_tearsheet_uses_structured_report_metadata(
        self,
        sample_metrics,
        sample_trades,
        sample_returns,
    ):
        """Test report metadata populates the visible shell and benchmark label."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            metrics=sample_metrics,
            trades=sample_trades,
            returns=sample_returns,
            benchmark_returns=sample_returns * 0.8,
            report_metadata=BacktestReportMetadata(
                strategy_name="Statistical Arbitrage",
                benchmark_name="SPY",
            ),
            template="full",
        )

        assert "Statistical Arbitrage" in html
        assert "SPY" in html

    def test_generate_backtest_tearsheet_allows_date_only_masthead(
        self,
        sample_metrics,
        sample_trades,
        sample_returns,
    ):
        """Test visible masthead can omit the title when metadata is absent."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            metrics=sample_metrics,
            trades=sample_trades,
            returns=sample_returns,
            title=None,
            subtitle=None,
            template="full",
        )

        assert '<h1 class="report-title"></h1>' in html
        assert 'report-meta-label">Report Date' in html

    def test_portfolio_sections_use_profile_daily_dates_for_intraday_equity(self):
        """Test portfolio plots inherit real daily dates from an intraday profile."""
        from datetime import UTC, datetime, timedelta

        from ml4t.diagnostic.integration import analyze_backtest_result
        from ml4t.diagnostic.visualization.backtest.tearsheet import _create_section_figure

        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        result = BacktestResult(
            trades=[],
            fills=[],
            portfolio_state=[],
            equity_curve=[
                (base, 100_000.0),
                (base + timedelta(hours=8), 101_000.0),
                (base + timedelta(hours=16), 102_500.0),
                (base + timedelta(days=1), 101_500.0),
                (base + timedelta(days=1, hours=8), 103_000.0),
                (base + timedelta(days=1, hours=16), 104_500.0),
            ],
            metrics={},
        )
        profile = analyze_backtest_result(result, calendar="crypto")
        returns = result.to_daily_returns(calendar="crypto")

        equity_fig = _create_section_figure(
            "equity_curve",
            profile=profile,
            returns=returns,
        )
        drawdown_fig = _create_section_figure(
            "drawdowns",
            profile=profile,
            returns=returns,
        )

        assert equity_fig is not None
        assert drawdown_fig is not None
        assert str(equity_fig.data[0].x[0]).startswith("2024-01-01")
        assert str(drawdown_fig.data[0].x[0]).startswith("2024-01-01")


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
        assert hasattr(first_section, "band")

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


# =============================================================================
# Tail Risk Tests
# =============================================================================


class TestTailRisk:
    """Tests for tail_risk.py functions."""

    def test_plot_tail_risk_basic(self, sample_returns):
        """Test basic tail risk analysis."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        fig = plot_tail_risk_analysis(sample_returns)

        assert isinstance(fig, go.Figure)
        # Should have histogram + table traces
        assert len(fig.data) >= 2
        # First trace should be histogram
        assert fig.data[0].type == "histogram"
        # Last trace should be table
        assert fig.data[-1].type == "table"

    def test_plot_tail_risk_themes(self, sample_returns):
        """Test tail risk with all themes."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        for theme in ["default", "dark", "print", "presentation"]:
            fig = plot_tail_risk_analysis(sample_returns, theme=theme)
            assert isinstance(fig, go.Figure)

    def test_plot_tail_risk_dimensions(self, sample_returns):
        """Test tail risk respects height/width."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        fig = plot_tail_risk_analysis(sample_returns, height=700, width=1200)

        assert fig.layout.height == 700
        assert fig.layout.width == 1200

    def test_plot_tail_risk_custom_confidence(self, sample_returns):
        """Test tail risk with custom confidence levels."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        fig = plot_tail_risk_analysis(sample_returns, confidence_levels=(0.90, 0.95, 0.99))

        assert isinstance(fig, go.Figure)
        # Table should have rows for all 3 levels
        table_trace = fig.data[-1]
        assert table_trace.type == "table"

    def test_plot_tail_risk_negative_skew(self):
        """Test with negatively skewed returns (crash-like)."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        np.random.seed(42)
        # Negative skew distribution
        returns = np.concatenate(
            [
                np.random.normal(0.001, 0.01, 200),  # Normal period
                np.random.normal(-0.05, 0.03, 20),  # Crash period
            ]
        )

        fig = plot_tail_risk_analysis(returns)
        assert isinstance(fig, go.Figure)

    def test_plot_tail_risk_insufficient_data(self):
        """Test with insufficient data returns placeholder."""
        from ml4t.diagnostic.visualization.backtest import plot_tail_risk_analysis

        fig = plot_tail_risk_analysis(np.array([0.01]))
        assert isinstance(fig, go.Figure)
        # Should have annotation about insufficient data
        assert len(fig.layout.annotations) > 0

    def test_tail_risk_in_risk_manager_template(self, sample_returns):
        """Test tail_risk section works via risk_manager template."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        html = generate_backtest_tearsheet(
            returns=sample_returns,
            metrics={"sharpe": 1.5, "n_periods": 252},
            template="risk_manager",
        )

        assert isinstance(html, str)
        assert "Tail Risk" in html


# =============================================================================
# SHAP Error Patterns Tests
# =============================================================================


@pytest.fixture
def sample_shap_result():
    """Create a sample TradeShapResult with explanations and error patterns."""
    from datetime import datetime

    from ml4t.diagnostic.evaluation.trade_shap.models import (
        ErrorPattern,
        TradeShapExplanation,
        TradeShapResult,
    )

    n_features = 5
    feature_names = ["momentum_20d", "volatility_10d", "rsi_14", "volume_ratio", "trend_60d"]

    # Create 20 explanations
    explanations = []
    for i in range(20):
        sv = np.random.randn(n_features) * 0.1
        sorted_feats = sorted(
            [(fname, sv[j]) for j, fname in enumerate(feature_names)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        explanations.append(
            TradeShapExplanation(
                trade_id=f"AAPL_{datetime(2023, 1, i + 1).isoformat()}",
                timestamp=datetime(2023, 1, i + 1),
                top_features=sorted_feats,
                feature_values={fname: np.random.randn() for fname in feature_names},
                shap_vector=sv,
            )
        )

    # Create 2 error patterns
    error_patterns = [
        ErrorPattern(
            cluster_id=0,
            n_trades=12,
            description="High momentum + Low volatility -> Losses",
            top_features=[
                ("momentum_20d", 0.45, 0.001, 0.002, True),
                ("volatility_10d", -0.32, 0.003, 0.004, True),
                ("rsi_14", 0.15, 0.08, 0.09, False),
            ],
            separation_score=1.2,
            distinctiveness=1.8,
            hypothesis="Trend-following entries during low-vol regimes reverse quickly",
            confidence=0.85,
            actions=["Add vol filter", "Reduce position size in low-vol"],
        ),
        ErrorPattern(
            cluster_id=1,
            n_trades=8,
            description="High RSI + High volume -> Losses",
            top_features=[
                ("rsi_14", 0.38, 0.001, 0.001, True),
                ("volume_ratio", 0.29, 0.01, 0.02, True),
            ],
            separation_score=0.9,
            distinctiveness=1.5,
            hypothesis="Overbought entries with high volume signal reversals",
            confidence=0.72,
        ),
    ]

    return TradeShapResult(
        n_trades_analyzed=20,
        n_trades_explained=20,
        n_trades_failed=0,
        explanations=explanations,
        error_patterns=error_patterns,
    )


class TestShapPatterns:
    """Tests for shap_patterns.py functions."""

    def test_plot_shap_error_patterns_basic(self, sample_shap_result):
        """Test basic error pattern visualization."""
        from ml4t.diagnostic.visualization.backtest import plot_shap_error_patterns

        fig = plot_shap_error_patterns(sample_shap_result)

        assert isinstance(fig, go.Figure)
        # Should have bar traces (one per cluster) + table trace
        bar_traces = [t for t in fig.data if t.type == "bar"]
        table_traces = [t for t in fig.data if t.type == "table"]
        assert len(bar_traces) == 2  # 2 clusters
        assert len(table_traces) == 1

    def test_plot_shap_error_patterns_themes(self, sample_shap_result):
        """Test error patterns with all themes."""
        from ml4t.diagnostic.visualization.backtest import plot_shap_error_patterns

        for theme in ["default", "dark", "print", "presentation"]:
            fig = plot_shap_error_patterns(sample_shap_result, theme=theme)
            assert isinstance(fig, go.Figure)

    def test_plot_shap_error_patterns_no_patterns(self):
        """Test graceful handling when no patterns exist."""
        from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult
        from ml4t.diagnostic.visualization.backtest import plot_shap_error_patterns

        empty_result = TradeShapResult(
            n_trades_analyzed=10,
            n_trades_explained=10,
            n_trades_failed=0,
            explanations=[],
            error_patterns=[],
        )

        fig = plot_shap_error_patterns(empty_result)
        assert isinstance(fig, go.Figure)
        # Should be a placeholder with annotation
        assert len(fig.layout.annotations) > 0

    def test_plot_shap_worst_trades_basic(self, sample_shap_result):
        """Test worst trades SHAP contribution chart."""
        from ml4t.diagnostic.visualization.backtest import plot_shap_worst_trades

        fig = plot_shap_worst_trades(sample_shap_result)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # All traces should be horizontal bars
        for trace in fig.data:
            assert trace.type == "bar"

    def test_plot_shap_worst_trades_dimensions(self, sample_shap_result):
        """Test worst trades respects dimensions."""
        from ml4t.diagnostic.visualization.backtest import plot_shap_worst_trades

        fig = plot_shap_worst_trades(sample_shap_result, height=800, width=1200)
        assert fig.layout.width == 1200

    def test_plot_shap_worst_trades_n_trades(self, sample_shap_result):
        """Test worst trades with limited trade count."""
        from ml4t.diagnostic.visualization.backtest import plot_shap_worst_trades

        fig = plot_shap_worst_trades(sample_shap_result, n_trades=5)
        assert isinstance(fig, go.Figure)
        # Each bar trace should have at most 5 y values
        for trace in fig.data:
            assert len(trace.y) <= 5

    def test_shap_errors_tearsheet_integration(self, sample_shap_result, sample_returns):
        """Test shap_errors section via BacktestTearsheet."""
        from ml4t.diagnostic.visualization.backtest import BacktestTearsheet

        ts = BacktestTearsheet(template="quant_trader")
        ts.add_returns(sample_returns)
        ts.add_shap_result(sample_shap_result)
        ts.enable_section("shap_errors")

        html = ts.generate()
        assert isinstance(html, str)
        assert "SHAP Error Patterns" in html
        assert "report-section" in html

    def test_backtest_tearsheet_builder_accepts_profile(self, sample_backtest_profile):
        """Test BacktestTearsheet can be hydrated from BacktestProfile."""
        from ml4t.diagnostic.visualization.backtest import BacktestTearsheet

        html = BacktestTearsheet(template="full").add_profile(sample_backtest_profile).generate()

        assert isinstance(html, str)
        assert "Attribution" in html

    def test_backtest_tearsheet_builder_adds_benchmark_context(
        self,
        sample_metrics,
        sample_trades,
        sample_returns,
    ):
        """Test builder benchmark input reaches the report shell."""
        from ml4t.diagnostic.visualization.backtest import BacktestTearsheet

        html = (
            BacktestTearsheet(template="full")
            .add_metrics(sample_metrics)
            .add_trades(sample_trades)
            .add_returns(sample_returns)
            .add_benchmark(sample_returns * 0.7, name="SPY")
            .generate()
        )

        assert "SPY" in html
        assert "Assessment" not in html

    def test_shap_errors_none_graceful(self, sample_returns):
        """Test that shap_errors section returns None when no shap_result."""
        from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

        # quant_trader has shap_errors disabled by default, so enable it
        html = generate_backtest_tearsheet(
            returns=sample_returns,
            template="full",
        )

        # Should not crash, shap_errors section should be absent
        assert isinstance(html, str)
