"""A weekend-spanning observed bar must not be reported as one elapsed day."""

from datetime import UTC, datetime
from html.parser import HTMLParser

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import generate_tearsheet_from_result


class _ShortThenCover(Strategy):
    def __init__(self):
        self.observations = 0

    def on_data(self, timestamp, data, context, broker):
        self.observations += 1
        if self.observations == 1:
            broker.submit_order("A", -10)
        elif self.observations == 2:
            broker.submit_order("A", 10)


def _actual_result():
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6]],
            "asset": ["A"] * 3,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, 150.0]),
            "volume": [10000.0] * 3,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _ShortThenCover(),
        BacktestConfig(initial_cash=1000, allow_short_selling=True),
    ).run()
    assert len(result.fills) == 2 and len(result.trades) == 1
    assert [fill.timestamp for fill in result.fills] == prices["timestamp"].to_list()[1:]
    assert [fill.price for fill in result.fills] == [100, 150]
    assert result.trades[0].bars_held == 1
    assert (result.fills[1].timestamp - result.fills[0].timestamp).days == 3
    return result


class _Cards(HTMLParser):
    def __init__(self):
        super().__init__()
        self.part = None
        self.label = None
        self.values = {}
        self.text = []

    def handle_starttag(self, tag, attrs):
        css = dict(attrs).get("class")
        if css in ("executive-kpi-label", "metrics-sidebar-label"):
            self.part = "label"
        elif css in ("executive-kpi-value", "metrics-sidebar-value"):
            self.part = "value"

    def handle_endtag(self, tag):
        if tag in ("div", "span"):
            self.part = None

    def handle_data(self, text):
        if text.strip():
            self.text.append(text.strip())
        if self.part == "label":
            self.label = text.strip()
        elif self.part == "value":
            self.values[self.label] = text.strip()


@pytest.mark.parametrize("template", ["full", "quant_trader", "hedge_fund"])
def test_actual_one_bar_three_day_trade_is_reported_in_bars_without_changing_trade_data(template):
    result = _actual_result()
    fills = result.to_fills_dataframe().clone()
    trades = result.to_trades_dataframe().clone()
    cards = _Cards()
    cards.feed(generate_tearsheet_from_result(result, template=template, include_statistical=False))
    assert cards.values["AVG HOLDING"] == "1.0 bars"
    assert result.to_fills_dataframe().equals(fills)
    assert result.to_trades_dataframe().equals(trades)


def test_actual_overview_sidebar_preserves_the_producer_holding_bar_unit():
    result = _actual_result()
    cards = _Cards()
    cards.feed(generate_tearsheet_from_result(result, include_statistical=False))
    assert cards.values["Avg Holding"] == "1.0 bars"


@pytest.mark.parametrize("renderer", ["figure", "html"])
def test_actual_trade_executive_summary_declares_bars_without_changing_the_count(renderer):
    from ml4t.diagnostic.integration import compute_metrics_from_result
    from ml4t.diagnostic.visualization.backtest import (
        create_executive_summary,
        create_executive_summary_html,
    )

    result = _actual_result()
    metrics = compute_metrics_from_result(result)
    assert metrics["avg_bars_held"] == 1
    if renderer == "figure":
        figure = create_executive_summary(metrics, selected_metrics=["avg_bars_held"], cols=1)
        assert figure.data[0].value == 1
        cards = _Cards()
        cards.feed(figure.data[0].title.text)
        assert cards.text == ["Avg Holding Period (bars)"]
    else:
        cards = _Cards()
        cards.feed(create_executive_summary_html(metrics, selected_metrics=["avg_bars_held"]))
        assert "Avg Holding Period (bars)" in cards.text
        assert "1" in cards.text
    assert result.trades[0].bars_held == 1


def test_actual_holding_count_without_declared_grading_ranges_has_neutral_color():
    from ml4t.diagnostic.integration import compute_metrics_from_result
    from ml4t.diagnostic.visualization.backtest import get_traffic_light_color

    metrics = compute_metrics_from_result(_actual_result())
    assert get_traffic_light_color(metrics["avg_bars_held"], "avg_bars_held") == "neutral"
