"""A real flat executed account must not crash reports on undefined PnL shares."""

import html
import re
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result, generate_tearsheet_from_result
from ml4t.diagnostic.visualization.backtest import plot_attribution_overview
from ml4t.diagnostic.visualization.backtest.executive_summary import TRAFFIC_LIGHT_COLORS
from ml4t.diagnostic.visualization.backtest.html_tables import create_top_contributors_html


class _BuyOnce(Strategy):
    def __init__(self, assets=("A",)):
        self.assets = assets
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            for asset in self.assets:
                broker.submit_order(asset, 4)
            self.submitted = True


def _executed_account(mark):
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7]],
            "asset": ["A"] * 4,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, mark, mark]),
            "volume": [10000.0] * 4,
        }
    )
    result = Engine(DataFeed(prices_df=prices), _BuyOnce(), BacktestConfig(initial_cash=1000)).run()
    assert len(result.fills) == 1 and result.fills[0].quantity == 4
    assert result.fills[0].price == 100 and result.trades[0].pnl == 4 * (mark - 100)
    return result


@pytest.mark.parametrize("mark", [100.0, 110.0, 90.0], ids=["flat", "gain", "loss"])
def test_actual_contribution_strip_uses_signed_cash_pnl_when_net_share_is_undefined(mark):
    profile = analyze_backtest_result(_executed_account(mark))
    attribution = profile.attribution
    before = attribution["by_symbol"].clone()
    if mark == 100:
        assert before["pnl_contribution_share"].to_list() == [None]
    rendered = create_top_contributors_html(attribution)
    assert rendered is not None
    text = html.unescape(re.sub(r"<[^>]+>", " ", rendered))
    assert f"${4 * (mark - 100):+,.0f}" in text
    assert "%" not in text and not re.search(r"\b(?:nan|inf)\b", text, re.I)
    styles = re.findall(r'<span style="([^"]+)">A [^<]+</span>', rendered)
    color = (
        TRAFFIC_LIGHT_COLORS["green"]
        if mark > 100
        else TRAFFIC_LIGHT_COLORS["red"]
        if mark < 100
        else "var(--c-text-muted)"
    )
    assert styles and all(f"color:{color};" in style for style in styles)
    assert attribution["by_symbol"].equals(before)


@pytest.mark.parametrize("mark", [100.0, 110.0, 90.0], ids=["flat", "gain", "loss"])
def test_actual_share_only_export_does_not_imply_profit_direction_from_net_pnl_share(mark):
    profile = analyze_backtest_result(_executed_account(mark))
    exported = profile.attribution["by_symbol"].select("symbol", "pnl_contribution_share")
    rendered = create_top_contributors_html({"by_symbol": exported})
    assert rendered is not None
    text = html.unescape(re.sub(r"<[^>]+>", " ", rendered))
    if mark == 100:
        assert "P&L contribution shares unavailable" in text
        assert "%" not in text
    else:
        assert "+100.0%" in text
        styles = re.findall(r'<span style="([^"]+)">A [^<]+</span>', rendered)
        assert styles and all("var(--c-text-muted)" in style for style in styles)


@pytest.mark.parametrize("mark", [100.0, 110.0, 90.0], ids=["flat", "gain", "loss"])
def test_actual_attribution_plot_uses_signed_cash_pnl_and_keeps_undefined_shares_visible(mark):
    profile = analyze_backtest_result(_executed_account(mark))
    before = profile.attribution["by_symbol"].clone()
    figure = plot_attribution_overview(profile)
    assert list(figure.data[0].y) == [4 * (mark - 100)]
    assert figure.layout.yaxis.title.text == "Net P&L ($)"
    assert figure.data[1].header.values[1] == "Share of net P&L"
    assert list(figure.data[1].cells.values[1]) == (["N/A"] if mark == 100 else [1.0])
    assert profile.attribution["by_symbol"].equals(before)


@pytest.mark.parametrize("template", ["full", "quant_trader", "hedge_fund"])
@pytest.mark.parametrize("mark", [100.0, 110.0, 90.0], ids=["flat", "gain", "loss"])
def test_actual_flat_and_nonflat_accounts_finish_public_reports_without_mutating_inputs(
    template, mark
):
    result = _executed_account(mark)
    fills = result.to_fills_dataframe().clone()
    trades = result.to_trades_dataframe().clone()
    rendered = generate_tearsheet_from_result(result, template=template, include_statistical=False)
    assert "<!DOCTYPE html>" in rendered
    if template == "full":
        match = re.search(
            r'<div class="cost-summary-line"[^>]*>\s*<span[^>]*>'
            r"Top / Bottom</span>(.*?)</div>",
            rendered,
            re.S,
        )
        assert match is not None
        text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        assert f"${4 * (mark - 100):+,.0f}" in text
        assert "%" not in text
    assert result.to_fills_dataframe().equals(fills)
    assert result.to_trades_dataframe().equals(trades)


def _executed_assets(marks_by_asset, initial_cash):
    dates = [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7]]
    rows = []
    for asset, marks in marks_by_asset.items():
        for timestamp, price in zip(dates, marks, strict=True):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 10000.0,
                }
            )
    return Engine(
        DataFeed(prices_df=pl.DataFrame(rows).sort("timestamp", "asset")),
        _BuyOnce(tuple(marks_by_asset)),
        BacktestConfig(initial_cash=initial_cash),
    ).run()


def test_actual_losing_total_does_not_rank_the_losing_asset_above_the_winner():
    result = _executed_assets(
        {"A": [100.0, 100.0, 110.0, 110.0], "B": [100.0, 100.0, 80.0, 80.0]}, 2000
    )
    assert {trade.symbol: trade.pnl for trade in result.trades} == {"A": 40.0, "B": -80.0}
    attribution = analyze_backtest_result(result).attribution
    before = attribution["by_symbol"].clone()
    assert dict(before.select("symbol", "pnl_contribution_share").rows()) == {"A": -1.0, "B": 2.0}
    rendered = create_top_contributors_html(attribution, top_n=1)
    assert rendered is not None
    text = html.unescape(re.sub(r"<[^>]+>", " ", rendered))
    assert re.search(r"A \$\+40\s*\|\s*B \$-80", text)
    assert attribution["by_symbol"].equals(before)


def test_actual_share_table_keeps_existing_halfway_decimal_rounding():
    result = _executed_assets(
        {"A": [100.0, 100.0, 12445.0, 12445.0], "B": [100.0, 100.0, 87755.0, 87755.0]},
        1000000,
    )
    assert {trade.symbol: trade.pnl for trade in result.trades} == {"A": 49380.0, "B": 350620.0}
    figure = plot_attribution_overview(analyze_backtest_result(result))
    table = figure.data[1]
    row = list(table.cells.values[0]).index("A")
    assert table.cells.values[1][row] == 0.1234
