from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from ml4t.backtest import BacktestResult
from ml4t.backtest.types import Fill, OrderSide, Trade

from ml4t.diagnostic.integration import analyze_backtest_result


@pytest.fixture
def sample_result() -> BacktestResult:
    base_time = datetime(2024, 1, 1, 10, 0)
    trades = [
        Trade(
            symbol="AAPL",
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=103.0,
            quantity=10.0,
            pnl=30.0,
            pnl_percent=0.03,
            bars_held=1,
            fees=1.0,
            exit_slippage=0.1,
        ),
        Trade(
            symbol="MSFT",
            entry_time=base_time + timedelta(hours=1),
            exit_time=base_time + timedelta(hours=3),
            entry_price=200.0,
            exit_price=194.0,
            quantity=10.0,
            pnl=-60.0,
            pnl_percent=-0.03,
            bars_held=2,
            fees=2.0,
            exit_slippage=0.2,
        ),
    ]
    fills = [
        Fill(
            order_id="fill-1",
            asset="AAPL",
            side=OrderSide.BUY,
            quantity=10.0,
            price=100.0,
            timestamp=base_time,
            rebalance_id="rebalance-1",
            commission=1.0,
            slippage=0.1,
            quote_mid_price=100.0,
        ),
        Fill(
            order_id="fill-2",
            asset="MSFT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=200.0,
            timestamp=base_time,
            rebalance_id="rebalance-1",
            commission=2.0,
            slippage=0.2,
        ),
        Fill(
            order_id="fill-3",
            asset="MSFT",
            side=OrderSide.SELL,
            quantity=10.0,
            price=194.0,
            timestamp=base_time + timedelta(hours=3),
            commission=2.0,
            slippage=0.2,
        ),
    ]
    equity_curve = [
        (base_time, 1000.0),
        (base_time + timedelta(hours=1), 1030.0),
        (base_time + timedelta(hours=2), 980.0),
        (base_time + timedelta(hours=3), 970.0),
        (base_time + timedelta(hours=4), 990.0),
    ]
    portfolio_state = [
        (base_time, 1000.0, 0.0, 1000.0, 1000.0, 2),
        (base_time + timedelta(hours=1), 1030.0, 30.0, 1000.0, 1000.0, 2),
        (base_time + timedelta(hours=2), 980.0, 20.0, 960.0, 960.0, 1),
        (base_time + timedelta(hours=3), 970.0, 970.0, 0.0, 0.0, 0),
        (base_time + timedelta(hours=4), 990.0, 990.0, 0.0, 0.0, 0),
    ]
    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        fills=fills,
        portfolio_state=portfolio_state,
        metrics={},
    )


def test_analyze_backtest_result_builds_lazy_profile(sample_result: BacktestResult):
    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    assert profile.summary["num_trades"] == 2
    assert profile.activity["metrics"]["num_rebalance_events"] == 2
    assert profile.occupancy["metrics"]["max_open_positions"] == 2
    assert profile.attribution["by_symbol"]["symbol"].to_list() == ["AAPL", "MSFT"]


def test_analyze_backtest_result_prefers_explicit_rebalance_ids(sample_result: BacktestResult):
    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    rebalance_summary = profile.activity["rebalance_summary"]
    assert rebalance_summary.height == 2
    assert rebalance_summary["symbols_touched"].to_list()[0] == 2


def test_analyze_backtest_result_reports_availability_states(sample_result: BacktestResult):
    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    assert profile.availability.families["activity"].status.value == "available"
    assert profile.availability.families["execution"].status.value == "partial"
    assert profile.availability.families["ml"].status.value == "unavailable"
    assert profile.availability.metrics["execution_audit"].coverage == pytest.approx(1 / 3)
    assert profile.availability.metrics["prediction_translation"].status.value == "unavailable"


def test_analyze_backtest_result_detects_prediction_surface(sample_result: BacktestResult):
    sample_result.predictions = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 10, 30),
            ],
            "asset": ["AAPL", "MSFT"],
            "score": [0.8, -0.4],
            "selected": [True, False],
            "trade_id": ["AAPL-1", None],
        }
    )

    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    assert profile.has_predictions is True
    assert profile.predictions_df.height == 2
    assert profile.ml["available"] is True
    assert profile.ml["metrics"]["n_prediction_assets"] == 2
    assert profile.availability.surfaces["predictions"].status.value == "available"
    assert profile.availability.families["ml"].status.value == "partial"
    assert profile.availability.metrics["prediction_translation"].status.value == "degraded"


def test_analyze_backtest_result_supports_new_prediction_method_names(sample_result: BacktestResult):
    sample_result.to_predictions_df = lambda: pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 10, 30),
            ],
            "asset": ["AAPL", "MSFT"],
            "prediction_value": [0.8, -0.4],
        }
    )
    sample_result.to_signals_df = lambda: pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 10, 30),
            ],
            "asset": ["AAPL", "MSFT"],
            "signal_value": [1.0, 0.0],
            "selected": [True, False],
        }
    )
    sample_result.strategy_metadata = {
        "strategy_type": "ml",
        "mapping_name": "threshold",
    }

    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    assert profile.has_predictions is True
    assert profile.has_signals is True
    assert profile.predictions_df.height == 2
    assert profile.signals_df.height == 2
    assert profile.ml["metrics"]["translation_ready"] is True
    assert profile.ml["metrics"]["trade_prediction_coverage"] == pytest.approx(1.0)
    assert "entry_prediction_value" in profile.ml["metrics"]["entry_prediction_columns"]
    assert profile.ml["strategy_metadata"]["mapping_name"] == "threshold"
    assert profile.availability.surfaces["signals"].status.value == "available"
    assert profile.availability.metrics["prediction_translation"].status.value == "available"


def test_analyze_backtest_result_normalizes_real_prediction_surface_schema(
    sample_result: BacktestResult,
):
    sample_result.to_predictions_dataframe = lambda: pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAPL", "MSFT"],
            "y_true": [0.02, -0.01],
            "y_score": [0.8, -0.4],
        }
    )

    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    assert profile.has_predictions is True
    assert "asset" in profile.predictions_df.columns
    assert profile.predictions_df.schema["timestamp"] == pl.Datetime
    assert profile.ml["metrics"]["trade_prediction_coverage"] == pytest.approx(1.0)
    assert "entry_y_score" in profile.ml["metrics"]["entry_prediction_columns"]


def test_analyze_backtest_result_drawdown_contributors_use_peak_to_trough_window(
    sample_result: BacktestResult,
):
    profile = analyze_backtest_result(sample_result, calendar="NYSE")

    episodes = profile.drawdown["episodes"]
    assert episodes.height == 1
    assert episodes["top_contributors"][0] == "MSFT"


def test_analyze_backtest_result_handles_tz_aware_equity_with_naive_trade_timestamps():
    base_time = datetime(2024, 1, 1, 10, 0)
    result = BacktestResult(
        trades=[
            Trade(
                symbol="ETHUSDT",
                entry_time=base_time,
                exit_time=base_time + timedelta(hours=2),
                entry_price=100.0,
                exit_price=94.0,
                quantity=10.0,
                pnl=-60.0,
                pnl_percent=-0.06,
                bars_held=2,
                fees=1.0,
            )
        ],
        fills=[],
        equity_curve=[
            (base_time.replace(tzinfo=UTC), 1000.0),
            ((base_time + timedelta(hours=1)).replace(tzinfo=UTC), 980.0),
            ((base_time + timedelta(hours=2)).replace(tzinfo=UTC), 940.0),
            ((base_time + timedelta(hours=3)).replace(tzinfo=UTC), 970.0),
        ],
        portfolio_state=[],
        metrics={},
    )

    profile = analyze_backtest_result(result, calendar="crypto")

    assert profile.drawdown["episodes"].height == 1
    assert profile.drawdown["episodes"]["top_contributors"][0] == "ETHUSDT"
