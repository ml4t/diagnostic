from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, BacktestResult
from ml4t.backtest.types import Trade
from ml4t.specs import FeedSpec

from ml4t.diagnostic.integration import (
    BacktestReportMetadata,
    compute_metrics_from_result,
    generate_tearsheet_from_result,
    generate_tearsheet_from_run_artifacts,
    portfolio_analysis_from_result,
    profile_from_run_artifacts,
)


def create_sample_result(n_trades: int = 20) -> BacktestResult:
    np.random.seed(42)
    trades = []
    base_time = datetime(2023, 1, 1)

    for i in range(n_trades):
        entry_time = base_time + timedelta(days=i * 2)
        exit_time = entry_time + timedelta(days=1)
        pnl = np.random.normal(50, 200)
        trades.append(
            Trade(
                symbol=f"ASSET_{i % 3}",
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=100.0,
                exit_price=100.0 + pnl / 100,
                quantity=100.0,
                pnl=pnl,
                pnl_percent=pnl / 10000,
                bars_held=2,
                fees=5.0,
                exit_slippage=2.0,
                exit_reason="signal",
                mfe=0.02,
                mae=-0.01,
            )
        )

    equity = [
        (base_time + timedelta(days=i), 100000 + i * 100 + np.random.normal(0, 500))
        for i in range(252)
    ]

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        fills=[],
        metrics={
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.12,
            "total_return_pct": 25.5,
            "final_value": 125500,
        },
    )


def test_integration_import_does_not_eagerly_load_backtest_bridge():
    sys.modules.pop("ml4t.diagnostic.integration.backtest", None)

    import ml4t.diagnostic.integration as integration

    importlib.reload(integration)

    assert "ml4t.diagnostic.integration.backtest" not in sys.modules


def test_backtest_bridge_raises_helpful_error_when_optional_dep_missing(monkeypatch):
    import ml4t.diagnostic.integration as integration

    integration._load_backtest_bridge.cache_clear()

    def _raise_missing(*args, **kwargs):
        err = ModuleNotFoundError("No module named 'ml4t.backtest'")
        err.name = "ml4t.backtest"
        raise err

    monkeypatch.setattr(integration.importlib, "import_module", _raise_missing)

    with pytest.raises(ImportError, match=r"ml4t-diagnostic\[backtest\]"):
        integration.compute_metrics_from_result(result=None)

    integration._load_backtest_bridge.cache_clear()


def test_portfolio_analysis_from_result_uses_session_aligned_returns():
    result = BacktestResult(
        trades=[],
        equity_curve=[
            (datetime(2025, 1, 6, 22, 30, tzinfo=UTC), 100.0),
            (datetime(2025, 1, 6, 23, 30, tzinfo=UTC), 110.0),
            (datetime(2025, 1, 7, 21, 0, tzinfo=UTC), 120.0),
            (datetime(2025, 1, 7, 23, 30, tzinfo=UTC), 130.0),
        ],
        fills=[],
        metrics={},
        config=BacktestConfig(calendar="CME_Equity", timezone="America/Chicago"),
    )

    expected = result.to_daily_returns(calendar="CME_Equity").to_list()
    analysis = portfolio_analysis_from_result(result, calendar="CME_Equity")

    assert list(analysis.returns) == pytest.approx(expected)


def test_portfolio_analysis_from_result_uses_feed_semantics_for_auto_alignment():
    result = BacktestResult(
        trades=[],
        equity_curve=[
            (datetime(2024, 1, 1, 18, 0), 100000.0),
            (datetime(2024, 1, 2, 10, 0), 101000.0),
        ],
        fills=[],
        metrics={},
        config=BacktestConfig(
            calendar="NYSE",
            timezone="America/New_York",
            feed_spec=FeedSpec(
                calendar="NYSE",
                session_start_time="17:00",
                timestamp_semantics="event_time",
            ),
        ),
    )

    analysis = portfolio_analysis_from_result(result)
    assert len(analysis.returns) == 1


def test_compute_metrics_from_result_handles_empty_inputs():
    result = BacktestResult(trades=[], equity_curve=[], fills=[], metrics={})
    metrics = compute_metrics_from_result(result, calendar="NYSE")

    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["total_return"] == 0.0
    assert metrics["cagr"] == 0.0
    assert metrics["calmar_ratio"] == 0.0
    assert metrics["num_trades"] == 0


def test_compute_metrics_from_result_uses_trade_analyzer():
    result = create_sample_result()
    result.trade_analyzer = SimpleNamespace(
        num_trades=7,
        win_rate=0.57,
        profit_factor=1.8,
        expectancy=0.012,
        avg_trade=0.009,
        avg_win=0.021,
        avg_loss=-0.008,
        total_fees=34.0,
    )

    metrics = compute_metrics_from_result(result, calendar="NYSE", confidence_intervals=True)

    assert metrics["num_trades"] == 7
    assert metrics["total_fees"] == 34.0
    assert "sharpe_ratio_ci_lower" in metrics
    assert "sharpe_ratio_ci_upper" in metrics


def test_compute_metrics_from_result_uses_dollar_trade_metrics_consistently():
    result = create_sample_result()
    expected_avg_trade = sum(trade.pnl for trade in result.trades) / len(result.trades)
    expected_avg_winner = sum(trade.pnl for trade in result.trades if trade.pnl > 0) / sum(
        1 for trade in result.trades if trade.pnl > 0
    )
    expected_avg_loser = sum(trade.pnl for trade in result.trades if trade.pnl < 0) / sum(
        1 for trade in result.trades if trade.pnl < 0
    )
    result.trade_analyzer = SimpleNamespace(
        num_trades=7,
        win_rate=0.57,
        profit_factor=1.8,
        expectancy=0.012,
        avg_trade=0.009,
        avg_win=0.021,
        avg_loss=-0.008,
        total_fees=34.0,
    )

    metrics = compute_metrics_from_result(result, calendar="NYSE")

    assert metrics["avg_trade"] == pytest.approx(expected_avg_trade)
    assert metrics["expectancy"] == pytest.approx(expected_avg_trade)
    assert metrics["avg_winner"] == pytest.approx(expected_avg_winner)
    assert metrics["avg_loser"] == pytest.approx(expected_avg_loser)


def test_compute_metrics_from_result_profit_factor_is_infinite_for_all_winners():
    trade = Trade(
        symbol="AAPL",
        entry_time=datetime(2024, 1, 1, 9, 30),
        exit_time=datetime(2024, 1, 2, 9, 30),
        entry_price=100.0,
        exit_price=101.0,
        quantity=10.0,
        pnl=10.0,
        pnl_percent=0.01,
        bars_held=1,
    )
    result = BacktestResult(trades=[trade], equity_curve=[], fills=[], metrics={})

    metrics = compute_metrics_from_result(result, calendar="NYSE")

    assert metrics["profit_factor"] == float("inf")


def test_generate_tearsheet_from_result_extracts_fallback_metrics(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def _generate_backtest_tearsheet(**kwargs):
        captured.update(kwargs)
        return "<html></html>"

    monkeypatch.setattr(
        "ml4t.diagnostic.visualization.backtest.generate_backtest_tearsheet",
        _generate_backtest_tearsheet,
    )

    trade = Trade(
        symbol="AAPL",
        entry_time=datetime(2024, 1, 1, 9, 30),
        exit_time=datetime(2024, 1, 2, 9, 30),
        entry_price=100.0,
        exit_price=101.0,
        quantity=10.0,
        pnl=10.0,
        pnl_percent=0.01,
        bars_held=1,
        exit_slippage=0.20,
        entry_slippage=0.10,
    )
    result = BacktestResult(
        trades=[trade],
        equity_curve=[(datetime(2024, 1, 1, 9, 30), 100000.0)],
        fills=[],
        metrics={},
    )

    output_path = tmp_path / "tearsheet.html"
    html = generate_tearsheet_from_result(result, output_path=output_path)

    assert html == "<html></html>"
    assert output_path.read_text() == html
    metrics = captured["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["total_slippage"] == pytest.approx(trade.total_slippage_cost)


def test_profile_from_run_artifacts_resolves_predictions_and_weights(tmp_path):
    run_log = tmp_path / "run_log"
    backtest_hash = "abc123def456"
    prediction_hash = "pred987654321"
    backtest_dir = run_log / "backtest" / backtest_hash
    prediction_dir = run_log / "predictions" / prediction_hash
    backtest_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_time": [datetime(2024, 1, 1, 0, 0)],
            "exit_time": [datetime(2024, 1, 2, 0, 0)],
            "entry_price": [100.0],
            "exit_price": [102.0],
            "quantity": [10.0],
            "direction": ["long"],
            "pnl": [20.0],
            "pnl_percent": [0.02],
            "bars_held": [1],
            "fees": [1.0],
            "exit_slippage": [0.1],
            "mfe": [0.03],
            "mae": [-0.01],
            "entry_slippage": [0.1],
            "multiplier": [1.0],
            "gross_pnl": [22.0],
            "net_return": [0.019],
            "total_slippage_cost": [2.0],
            "cost_drag": [0.001],
            "exit_reason": ["signal"],
            "status": ["closed"],
        }
    ).write_parquet(backtest_dir / "trades.parquet")
    pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "daily_return": [0.01, -0.005],
        }
    ).write_parquet(backtest_dir / "daily_returns.parquet")
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0, 0)],
            "symbol": ["AAPL"],
            "weight": [0.2],
        }
    ).write_parquet(backtest_dir / "weights.parquet")
    (backtest_dir / "spec.json").write_text(
        json.dumps(
            {
                "preset_id": "etfs:base",
                "strategy": {
                    "signal": {"method": "equal_weight_top_k", "top_k": 2},
                    "allocation": {"method": "hrp"},
                },
                "backtest_config": {
                    "cash": {"initial": 100000.0},
                    "calendar": {"calendar": "NYSE"},
                    "metadata": {"prediction_hash": prediction_hash},
                },
            }
        )
    )
    pl.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "symbol": ["AAPL"],
            "y_true": [0.02],
            "y_score": [0.8],
            "config_name": ["cae"],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")

    profile = profile_from_run_artifacts(backtest_dir)

    assert profile.has_predictions is True
    assert profile.has_signals is True
    assert profile.predictions_df["asset"].to_list() == ["AAPL"]
    assert profile.signals_df["signal_value"].to_list() == [0.2]
    assert profile.strategy_metadata["mapping_name"] == "equal_weight_top_k"
    assert profile.ml["metrics"]["translation_ready"] is True
    assert profile.occupancy["metrics"]["time_in_market"] > 0
    assert profile.activity["metrics"]["num_rebalance_events"] > 0


def test_profile_from_run_artifacts_respects_feed_column_contract(tmp_path):
    run_log = tmp_path / "run_log"
    backtest_hash = "abc123def456"
    prediction_hash = "pred987654321"
    backtest_dir = run_log / "backtest" / backtest_hash
    prediction_dir = run_log / "predictions" / prediction_hash
    backtest_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_time": [datetime(2024, 1, 1, 0, 0)],
            "exit_time": [datetime(2024, 1, 2, 0, 0)],
            "entry_price": [100.0],
            "exit_price": [102.0],
            "quantity": [10.0],
            "direction": ["long"],
            "pnl": [20.0],
            "pnl_percent": [0.02],
            "bars_held": [1],
            "fees": [1.0],
            "exit_slippage": [0.1],
            "mfe": [0.03],
            "mae": [-0.01],
            "entry_slippage": [0.1],
            "multiplier": [1.0],
            "gross_pnl": [22.0],
            "net_return": [0.019],
            "total_slippage_cost": [2.0],
            "cost_drag": [0.001],
            "exit_reason": ["signal"],
            "status": ["closed"],
        }
    ).write_parquet(backtest_dir / "trades.parquet")
    pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "daily_return": [0.01, -0.005],
        }
    ).write_parquet(backtest_dir / "daily_returns.parquet")
    pl.DataFrame(
        {
            "ts_event": [datetime(2024, 1, 1, 0, 0)],
            "ticker": ["AAPL"],
            "weight": [0.2],
        }
    ).write_parquet(backtest_dir / "weights.parquet")
    (backtest_dir / "spec.json").write_text(
        json.dumps(
            {
                "backtest_config": {
                    "cash": {"initial": 100000.0},
                    "calendar": {"calendar": "NYSE"},
                    "feed": {
                        "timestamp_col": "ts_event",
                        "entity_col": "ticker",
                    },
                    "metadata": {"prediction_hash": prediction_hash},
                }
            }
        )
    )
    pl.DataFrame(
        {
            "ts_event": ["2024-01-01"],
            "ticker": ["AAPL"],
            "y_true": [0.02],
            "y_score": [0.8],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")

    profile = profile_from_run_artifacts(backtest_dir)

    assert profile.predictions_df["asset"].to_list() == ["AAPL"]
    assert profile.signals_df["asset"].to_list() == ["AAPL"]
    assert profile.signals_df["signal_value"].to_list() == [0.2]
    assert "timestamp" in profile.predictions_df.columns
    assert "timestamp" in profile.signals_df.columns


def test_profile_from_run_artifacts_trims_pre_live_warmup_window(tmp_path):
    run_log = tmp_path / "run_log"
    backtest_hash = "abc123def456"
    backtest_dir = run_log / "backtest" / backtest_hash
    backtest_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_time": [datetime(2024, 1, 5, 0, 0)],
            "exit_time": [datetime(2024, 1, 8, 0, 0)],
            "entry_price": [100.0],
            "exit_price": [102.0],
            "quantity": [10.0],
            "direction": ["long"],
            "pnl": [20.0],
            "pnl_percent": [0.02],
            "bars_held": [1],
            "fees": [1.0],
            "exit_slippage": [0.1],
            "mfe": [0.03],
            "mae": [-0.01],
            "entry_slippage": [0.1],
            "multiplier": [1.0],
            "gross_pnl": [22.0],
            "net_return": [0.019],
            "total_slippage_cost": [2.0],
            "cost_drag": [0.001],
            "exit_reason": ["signal"],
            "status": ["closed"],
        }
    ).write_parquet(backtest_dir / "trades.parquet")
    pl.DataFrame(
        {
            "timestamp": [
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2024, 1, 5),
                date(2024, 1, 8),
            ],
            "daily_return": [0.0, 0.0, 0.01, -0.005],
        }
    ).write_parquet(backtest_dir / "daily_returns.parquet")
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 5, 0, 0)],
            "symbol": ["AAPL"],
            "weight": [0.2],
        }
    ).write_parquet(backtest_dir / "weights.parquet")
    (backtest_dir / "spec.json").write_text(
        json.dumps(
            {
                "backtest_config": {
                    "cash": {"initial": 100000.0},
                    "calendar": {"calendar": "NYSE"},
                }
            }
        )
    )

    profile = profile_from_run_artifacts(backtest_dir)

    assert profile.equity_df["timestamp"][0] == datetime(2024, 1, 5, 0, 0)
    assert len(profile.daily_returns) == 2
    assert profile.daily_returns[1] == pytest.approx(-0.005)


def test_generate_tearsheet_from_run_artifacts_renders_ml_workspace(tmp_path):
    run_log = tmp_path / "run_log"
    backtest_hash = "abc123def456"
    prediction_hash = "pred987654321"
    backtest_dir = run_log / "backtest" / backtest_hash
    prediction_dir = run_log / "predictions" / prediction_hash
    backtest_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_time": [datetime(2024, 1, 1, 0, 0)],
            "exit_time": [datetime(2024, 1, 2, 0, 0)],
            "entry_price": [100.0],
            "exit_price": [102.0],
            "quantity": [10.0],
            "direction": ["long"],
            "pnl": [20.0],
            "pnl_percent": [0.02],
            "bars_held": [1],
            "fees": [1.0],
            "exit_slippage": [0.1],
            "mfe": [0.03],
            "mae": [-0.01],
            "entry_slippage": [0.1],
            "multiplier": [1.0],
            "gross_pnl": [22.0],
            "net_return": [0.019],
            "total_slippage_cost": [2.0],
            "cost_drag": [0.001],
            "exit_reason": ["signal"],
            "status": ["closed"],
        }
    ).write_parquet(backtest_dir / "trades.parquet")
    pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
            "daily_return": [0.01, -0.005],
        }
    ).write_parquet(backtest_dir / "daily_returns.parquet")
    pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0, 0)],
            "symbol": ["AAPL"],
            "weight": [0.2],
        }
    ).write_parquet(backtest_dir / "weights.parquet")
    (backtest_dir / "spec.json").write_text(
        json.dumps(
            {
                "preset_id": "etfs:base",
                "strategy": {
                    "signal": {"method": "equal_weight_top_k", "top_k": 2},
                    "allocation": {"method": "hrp"},
                },
                "backtest_config": {
                    "cash": {"initial": 100000.0},
                    "calendar": {"calendar": "NYSE"},
                    "metadata": {"prediction_hash": prediction_hash},
                },
            }
        )
    )
    pl.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "symbol": ["AAPL"],
            "y_true": [0.02],
            "y_score": [0.8],
            "config_name": ["cae"],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")

    html = generate_tearsheet_from_run_artifacts(backtest_dir, template="full")

    assert 'data-workspace="ml"' in html
    assert "Prediction" in html


def test_generate_tearsheet_from_result_normalizes_backtest_metric_keys(monkeypatch):
    captured: dict[str, object] = {}

    def _generate_backtest_tearsheet(**kwargs):
        captured.update(kwargs)
        return "<html></html>"

    monkeypatch.setattr(
        "ml4t.diagnostic.visualization.backtest.generate_backtest_tearsheet",
        _generate_backtest_tearsheet,
    )

    result = BacktestResult(
        trades=[],
        equity_curve=[
            (datetime(2024, 1, 1, 9, 30), 100000.0),
            (datetime(2024, 1, 2, 9, 30), 101000.0),
        ],
        fills=[],
        metrics={
            "sharpe": 1.5,
            "sortino": 2.0,
            "calmar": 1.2,
            "total_return_pct": 12.5,
            "max_drawdown": -0.1,
            "num_trades": 4,
        },
    )

    generate_tearsheet_from_result(result)

    metrics = captured["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["sharpe"] == 1.5
    assert metrics["sharpe_ratio"] == 1.5
    assert metrics["sortino_ratio"] == 2.0
    assert metrics["calmar_ratio"] == 1.2
    assert metrics["total_return"] == pytest.approx(0.125)
    assert metrics["n_trades"] == 4
    assert metrics["max_drawdown"] == pytest.approx(0.1)


def test_generate_tearsheet_from_result_forwards_report_metadata(monkeypatch):
    captured: dict[str, object] = {}

    def _generate_backtest_tearsheet(**kwargs):
        captured.update(kwargs)
        return "<html></html>"

    monkeypatch.setattr(
        "ml4t.diagnostic.visualization.backtest.generate_backtest_tearsheet",
        _generate_backtest_tearsheet,
    )

    result = BacktestResult(
        trades=[],
        equity_curve=[
            (datetime(2024, 1, 1, 9, 30), 100000.0),
            (datetime(2024, 1, 2, 9, 30), 101000.0),
        ],
        fills=[],
        metrics={},
    )
    metadata = BacktestReportMetadata(
        strategy_name="Momentum Rotation",
        benchmark_name="SPY",
    )

    generate_tearsheet_from_result(result, report_metadata=metadata)

    assert captured["report_metadata"] == metadata
