"""Undefined lifecycle-PnL ratios must remain unavailable through the default report."""

import html
import math
import re
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import (
    analyze_backtest_result,
    compute_metrics_from_result,
    generate_tearsheet_from_result,
)


class _PlannedOrders(Strategy):
    def __init__(self, orders):
        self.orders = orders
        self.index = 0

    def on_data(self, timestamp, data, context, broker):
        if self.index in self.orders:
            broker.submit_order("A", self.orders[self.index])
        self.index += 1


@pytest.mark.parametrize(
    ("marks", "orders", "pnls", "expected"),
    [
        ([100.0] * 4, {}, [], math.nan),
        ([100.0] * 4, {0: 4}, [0.0], math.nan),
        ([100.0] * 4, {0: 4, 2: -4}, [0.0], math.nan),
        ([100.0, 100.0, 110.0, 120.0], {0: 4, 2: -4}, [80.0], math.inf),
        ([100.0, 100.0, 90.0, 80.0], {0: 4, 2: -4}, [-80.0], 0.0),
        (
            [100.0, 100.0, 110.0, 110.0, 100.0, 100.0, 80.0, 80.0],
            {0: 4, 2: -4, 4: 4, 6: -4},
            [40.0, -80.0],
            0.5,
        ),
    ],
    ids=["no-trades", "marked-flat", "closed-flat", "gain-only", "loss-only", "mixed-cash-pnl"],
)
def test_actual_undefined_trade_profit_factor_survives_profile_bridge_and_default_html(
    marks, orders, pnls, expected
):
    dates = [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7, 8, 9, 10, 13]]
    bars = pl.DataFrame(
        {
            "timestamp": dates[: len(marks)],
            "asset": ["A"] * len(marks),
            **dict.fromkeys(["open", "high", "low", "close"], marks),
            "volume": [10000.0] * len(marks),
        }
    )
    result = Engine(
        DataFeed(prices_df=bars), _PlannedOrders(orders), BacktestConfig(initial_cash=1000)
    ).run()
    assert [trade.pnl for trade in result.trades] == pnls
    trades, fills = result.to_trades_dataframe().clone(), result.to_fills_dataframe().clone()
    producer_profit_factor = result.metrics.get("profit_factor")
    profile = analyze_backtest_result(result)
    assert profile.edge["metrics"]["num_trades"] == len(pnls)
    values = [
        profile.edge["metrics"]["profit_factor"],
        profile.summary["profit_factor"],
        compute_metrics_from_result(result)["profit_factor"],
    ]
    for value in values:
        if math.isnan(expected):
            assert math.isnan(value), "Zero gains and losses do not establish a trade profit factor"
        elif math.isinf(expected):
            assert value == math.inf
        else:
            assert value == pytest.approx(expected, rel=1e-12, abs=0)
    rendered = generate_tearsheet_from_result(result, template="full", include_statistical=False)
    labels = re.findall(
        r'<div class="executive-kpi-label">PROFIT FACTOR</div>\s*'
        r'<div class="executive-kpi-value"[^>]*>\s*([^<]+)',
        rendered,
        re.IGNORECASE,
    )
    expected_label = (
        "N/A" if math.isnan(expected) else "∞" if math.isinf(expected) else f"{expected:.2f}"
    )
    assert labels and {html.unescape(label) for label in labels} == {expected_label}
    assert result.to_trades_dataframe().equals(trades)
    assert result.to_fills_dataframe().equals(fills)
    if isinstance(producer_profit_factor, float) and math.isnan(producer_profit_factor):
        assert math.isnan(result.metrics["profit_factor"])
    else:
        assert result.metrics.get("profit_factor") == producer_profit_factor
