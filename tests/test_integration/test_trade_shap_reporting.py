"""A Trade-SHAP result from a real backtest reaches the tearsheet and reads as a static figure.

The fixture runs a multi-asset ml4t-backtest strategy, converts its trades to
``TradeRecord`` objects and explains the worst of them with ``TradeShapAnalyzer``. Its SHAP
rows follow two regimes, so the pipeline finds error clusters as it would on a model.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import TradeShapAnalyzer
from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult
from ml4t.diagnostic.integration.backtest_contract import TradeRecord

backtest = pytest.importorskip("ml4t.backtest")

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
FEATURES = ["momentum", "volatility", "value", "carry"]
DATES = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(150)]
SHAP_SECTIONS = ("shap_errors", "shap_worst_trades")


class _Rotate(backtest.Strategy):
    """Buy each symbol in turn, hold it for two bars, then close."""

    def __init__(self) -> None:
        self.step = 0

    def on_data(self, timestamp, data, context, broker):
        if self.step % 3 == 0:
            broker.submit_order(SYMBOLS[(self.step // 3) % len(SYMBOLS)], 10.0)
        elif self.step % 3 == 2:
            for asset, pos in broker.get_positions().items():
                if pos.quantity != 0:
                    broker.close_position(asset)
        self.step += 1


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(3)
    bars = pl.DataFrame(
        [
            {
                "timestamp": ts,
                "asset": symbol,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1e6,
            }
            for j, symbol in enumerate(SYMBOLS)
            for ts, price in zip(
                DATES, 100 + 5 * j + np.cumsum(rng.normal(0, 1, len(DATES))), strict=True
            )
        ]
    )
    return backtest.Engine(
        backtest.DataFeed(prices_df=bars), _Rotate(), backtest.BacktestConfig(initial_cash=1e5)
    ).run()


@pytest.fixture(scope="module")
def shap_result(run) -> TradeShapResult:
    trades = [
        TradeRecord(
            timestamp=t.exit_time.replace(tzinfo=None),
            entry_timestamp=t.entry_time.replace(tzinfo=None),
            symbol=t.symbol,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl=t.pnl,
            duration=t.exit_time - t.entry_time,
            direction="long",
            quantity=abs(t.quantity),
        )
        for t in run.trades
        if t.exit_time is not None
    ]
    rng = np.random.default_rng(5)
    panel = pl.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": symbol,
                **dict(zip(FEATURES, rng.normal(size=4), strict=True)),
            }
            for ts in DATES
            for symbol in SYMBOLS
        ]
    )
    # Two regimes of SHAP vectors, alternating every six days.
    regime = np.array([(i // 6) % 2 for i in range(len(DATES)) for _ in SYMBOLS])
    patterns = np.array([[0.9, 0.3, -0.1, 0.0], [-0.2, 0.0, 0.8, -0.6]])
    shap_values = patterns[regime] + rng.normal(0, 0.05, (panel.height, len(FEATURES)))

    analyzer = TradeShapAnalyzer(model=None, features_df=panel, shap_values=shap_values)
    result = analyzer.explain_worst_trades(trades, n=30)
    assert result.n_trades_explained == 30
    assert result.error_patterns
    return result


@pytest.mark.parametrize("template", ["quant_trader", "hedge_fund", "risk_manager", "full"])
def test_backtest_tearsheet_renders_supplied_shap_result_in_every_template(
    run, shap_result, template
):
    from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

    html = generate_backtest_tearsheet(
        trades=run.to_trades_dataframe(),
        returns=run.to_daily_returns().to_numpy(),
        template=template,
        shap_result=shap_result,
    )

    for section in SHAP_SECTIONS:
        assert f'data-section="{section}"' in html


def test_tearsheet_from_result_accepts_and_renders_shap_result(run, shap_result):
    from ml4t.diagnostic.integration import generate_tearsheet_from_result

    html = generate_tearsheet_from_result(
        run, calendar=None, periods_per_year=365, shap_result=shap_result
    )

    for section in SHAP_SECTIONS:
        assert f'data-section="{section}"' in html


def test_tearsheet_builder_add_shap_result_enables_its_sections(run, shap_result):
    from ml4t.diagnostic.visualization.backtest import BacktestTearsheet

    html = (
        BacktestTearsheet(template="hedge_fund")
        .add_returns(run.to_daily_returns().to_numpy())
        .add_shap_result(shap_result)
        .generate()
    )

    for section in SHAP_SECTIONS:
        assert f'data-section="{section}"' in html


def test_ml_workspace_export_renders_supplied_shap_result(run, shap_result):
    from ml4t.diagnostic.visualization.backtest import export_workspaces

    html = export_workspaces(
        ("ml",), returns=run.to_daily_returns().to_numpy(), shap_result=shap_result
    )

    for section in SHAP_SECTIONS:
        assert f'data-section="{section}"' in html
