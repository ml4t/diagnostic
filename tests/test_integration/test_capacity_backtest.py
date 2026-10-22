"""Actual captured broker accounts must reach supplied-market capacity unchanged."""

import inspect
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.integration import portfolio_analysis_from_result


def test_actual_bar_account_capacity_preserves_cash_quantities_and_prior_market_rows(tmp_path):
    backtest = pytest.importorskip("ml4t.backtest")
    if "holdings_sampling" not in inspect.signature(backtest.Engine).parameters:
        pytest.skip("Installed Backtest lacks separately qualified opt-in holdings capture")
    from ml4t.backtest.config import CommissionType, SlippageType

    class BuyThenSell(backtest.Strategy):
        def __init__(self):
            self.step = 0

        def on_data(self, timestamp, data, context, broker):
            if self.step == 0:
                broker.submit_order("A", 2)
            elif self.step == 2:
                broker.submit_order("A", -2)
            self.step += 1

    dates = [datetime(2025, 7, 7, 14, minute, tzinfo=UTC) for minute in range(5)]
    market = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["A"] * 5,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, 101.0, 102.0, 102.0]),
            "volume": [10000.0] * 5,
        }
    )
    produced = backtest.Engine(
        backtest.DataFeed(prices_df=market),
        BuyThenSell(),
        backtest.BacktestConfig(
            initial_cash=1000,
            calendar="NYSE",
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        ),
        holdings_sampling="bar",
    ).run()
    analysis = portfolio_analysis_from_result(produced, calendar=None, periods_per_year=252 * 390)
    owned = analysis.input_data
    assert owned is not None
    np.testing.assert_allclose(owned.valuations["equity"], [1000, 1000, 1002, 1004, 1004])
    np.testing.assert_allclose(owned.valuations["cash"], [1000, 800, 800, 1004, 1004])
    assert owned.transactions["quantity"].to_list() == [2.0, -2.0]
    before = owned.to_json()
    feed = FeedSpec(
        timestamp_col="timestamp",
        entity_col="asset",
        price_col="close",
        volume_col="volume",
        calendar=None,
        time_contract_version=2,
    )
    result = analysis.compute_capacity(
        market,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        feed_spec=feed,
        lookback=1,
        base_capital=1000,
        scenario_capitals=[1000],
    )
    assert result.timeline["date"].equals(analysis.dates)
    np.testing.assert_allclose(
        result.timeline["book_liquidation_duration"], [0, 0.002, 0.00202, 0, 0]
    )
    np.testing.assert_allclose(result.participation["participation"], [0.0002, 0.0002])
    np.testing.assert_allclose(
        result.scenarios["extra_impact_return"], [0, 8e-10, 0, 8.16e-7 / 1002, 0]
    )
    assert owned.to_json() == before
    report = analysis.create_tear_sheet(capacity=result)
    output = tmp_path / "captured-capacity.html"
    report.save_html(output, include_plotlyjs=True)
    html = output.read_text(encoding="utf-8")
    assert "Capacity assumptions and coverage" in html and "0.00202" in html
    assert report.capacity is result
