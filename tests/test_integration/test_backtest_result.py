from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from ml4t.backtest import BacktestConfig, BacktestResult
from ml4t.backtest.types import Trade
from ml4t.data.artifacts.market_data import FeedSpec
from ml4t.diagnostic.integration import (
    compute_metrics_from_result,
    generate_tearsheet_from_result,
    portfolio_analysis_from_result,
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
                slippage=2.0,
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
        slippage=0.20,
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
