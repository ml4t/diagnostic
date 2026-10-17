"""Actual contract fills must retain their dollar units in reports and analytics."""

import json
import re
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.backtest import AssetClass, BacktestConfig, ContractSpec, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result, generate_tearsheet_from_result
from ml4t.diagnostic.visualization.backtest import plot_activity_overview


class _BuyOrSellOnce(Strategy):
    def __init__(self, quantity):
        self.quantity = quantity
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("ES", self.quantity)
            self.submitted = True


def _executed_contract(multiplier=50.0, quantity=1, costs=False):
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7]],
            "asset": ["ES"] * 4,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0] * 4),
            "volume": [10000.0] * 4,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _BuyOrSellOnce(quantity),
        BacktestConfig(
            initial_cash=10000,
            allow_short_selling=True,
            commission_type="per_trade",
            commission_per_trade=2.0 if costs else 0.0,
            slippage_type="fixed",
            slippage_fixed=0.5 if costs else 0.0,
        ),
        contract_specs={
            "ES": ContractSpec(symbol="ES", asset_class=AssetClass.FUTURE, multiplier=multiplier)
        },
    ).run()
    expected_price = 100.0 + (0.5 if quantity > 0 else -0.5) if costs else 100.0
    assert len(result.fills) == 1
    assert result.fills[0].quantity == 1 and result.fills[0].price == expected_price
    assert result.trades[0].multiplier == multiplier
    expected_cost = 2.0 + 0.5 * multiplier if costs else 0.0
    assert result.to_equity_dataframe()["equity"][-1] == 10000 - expected_cost
    return result, expected_price * multiplier, expected_cost


@pytest.mark.parametrize("multiplier", [1.0, 50.0])
@pytest.mark.parametrize("quantity", [1, -1], ids=["buy", "sell"])
@pytest.mark.parametrize("costs", [False, True], ids=["no-costs", "cash-fee-point-slippage"])
def test_actual_contract_activity_scales_notional_and_slippage_but_not_cash_commission(
    multiplier, quantity, costs
):
    result, notional, cash_cost = _executed_contract(multiplier, quantity, costs)
    fills, trades = result.to_fills_dataframe().clone(), result.to_trades_dataframe().clone()
    state = result.to_portfolio_state_dataframe().clone()
    profile = analyze_backtest_result(result)
    activity = profile.activity
    attribution = profile.attribution["by_symbol"].row(0, named=True)
    assert {
        "activity_notional": activity["metrics"]["total_filled_notional"],
        "attribution_notional": attribution["turnover_notional"],
        "activity_cash_cost": activity["metrics"]["total_implementation_cost"],
        "attribution_cash_cost": attribution["implementation_cost"],
    } == {
        "activity_notional": notional,
        "attribution_notional": notional,
        "activity_cash_cost": cash_cost,
        "attribution_cash_cost": cash_cost,
    }
    active = activity["turnover_timeline"].filter(pl.col("filled_notional") > 0)
    assert active.height == 1
    assert active["turnover"][0] == pytest.approx(notional / (10000 - cash_cost))
    assert active["cost_drag"][0] == pytest.approx(cash_cost / (10000 - cash_cost))
    assert activity["rebalance_summary"]["filled_notional"].to_list() == [notional]
    assert result.to_fills_dataframe().equals(fills)
    assert result.to_trades_dataframe().equals(trades)
    assert result.to_portfolio_state_dataframe().equals(state)


def test_actual_contract_turnover_reaches_public_plot_and_html_report():
    result, _, _ = _executed_contract()
    chart = plot_activity_overview(analyze_backtest_result(result))
    assert isinstance(chart, str)
    match = re.search(r'Plotly.newPlot\(\s*"[^"]+",\s*', chart)
    assert match is not None
    traces, _ = json.JSONDecoder().raw_decode(chart[match.end() :])
    assert traces[0]["name"] == "Turnover"
    assert traces[0]["y"] == [0.0, 0.5, 0.0, 0.0]
    assert traces[0]["customdata"][1] == [10000.0, 5000.0, 0.0, "defined"]
    html = generate_tearsheet_from_result(result, template="full", include_statistical=False)
    values = re.findall(
        r'<div class="executive-kpi-label">AVG TURNOVER</div>\s*'
        r'<div class="executive-kpi-value"[^>]*>\s*([^<]+)',
        html,
    )
    assert values and set(values) == {"0.50"}
