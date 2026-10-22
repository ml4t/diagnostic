"""Actual producer accounts must retain initial capital, external flows and net fees."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_backtest_flows_and_first_fill_fee_reach_the_final_report(tmp_path):
    pytest.importorskip("ml4t.backtest")
    from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
    from ml4t.backtest.config import CommissionType, ExecutionMode

    from ml4t.diagnostic.integration import portfolio_analysis_from_result

    class BuyOnce(Strategy):
        def __init__(self):
            self.ordered = False

        def on_data(self, timestamp, data, context, broker):
            if not self.ordered:
                broker.submit_order("A", 2)
                self.ordered = True

    dates = [datetime(2025, 7, day, 16, tzinfo=UTC) for day in (7, 8, 9)]
    prices = pl.DataFrame({"timestamp": dates, "asset": ["A"] * 3, "close": [100.0, 110.0, 90.0]})
    flows = pl.DataFrame(
        {
            "timestamp": dates[1:],
            "event_id": ["contribution", "withdrawal"],
            "kind": ["delta", "delta"],
            "value": [100.0, -50.0],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        BuyOnce(),
        BacktestConfig(
            initial_cash=1000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=1.0,
        ),
        holdings_sampling="bar",
    ).run()
    np.testing.assert_allclose(result.to_equity_dataframe()["equity"], [999, 1119, 1029])
    expected_returns = [-1 / 1000, 20 / 999, -40 / 1119]
    analysis = portfolio_analysis_from_result(result, calendar=None, periods_per_year=252)
    np.testing.assert_allclose(analysis.returns, expected_returns)
    activity = analysis.compute_transaction_analysis(incremental_bps=[0, 10])
    assert activity.timeline["beginning_nav"][0] == 1000
    assert activity.timeline["observed_cost_return"][0] == pytest.approx(0.001)
    assert analysis.input_data.metadata["initial_nav"] == 1000
    zero_cost = activity.scenarios.filter(pl.col("incremental_bps") == 0)
    np.testing.assert_allclose(zero_cost["net_return"], expected_returns)
    report = analysis.create_tear_sheet(transactions=activity)
    capital = report.availability["external_capital_flows"]
    assert capital["net_cash_flow"] == 50
    assert [event["cash_delta"] for event in capital["events"]] == [100, -50]
    wealth = next(
        trace for trace in report.figures["Cumulative Returns"].data if trace.name == "Strategy"
    )
    assert wealth.y[-1] == pytest.approx(np.prod(1 + np.array(expected_returns)) - 1)
    output = tmp_path / "producer-account.html"
    report.save_html(output, include_plotlyjs=True)
    html = output.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html and "2 supplied external capital events" in html
    from ml4t.backtest.result import BacktestResult

    result.to_parquet(tmp_path / "account")
    restored = portfolio_analysis_from_result(
        BacktestResult.from_parquet(tmp_path / "account"), calendar=None, periods_per_year=252
    )
    np.testing.assert_allclose(restored.returns, expected_returns)
    assert restored.input_data.metadata["initial_nav"] == 1000
    assert restored.input_data.metadata["source"]["capital_flows"] == capital["events"]
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        browser = runtime.chromium.launch(
            executable_path=chrome or runtime.chromium.executable_path, args=["--no-sandbox"]
        )
        try:
            page = browser.new_page()
            page.context.set_offline(True)
            page.goto(output.as_uri(), wait_until="load")
            chart = (
                page.locator(".plot-container")
                .filter(has=page.get_by_role("heading", name="Cumulative Returns", exact=True))
                .locator(".js-plotly-plot")
            )
            plotted = chart.evaluate(
                "node => Array.from(node._fullData.find(t => t.name === 'Strategy').y).at(-1)"
            )
            assert plotted == pytest.approx(np.prod(1 + np.array(expected_returns)) - 1)
        finally:
            browser.close()


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("exit_bar", [False, True])
def test_published_engineer_outcomes_reach_complete_reports_with_owned_endpoints(
    missing, exit_bar, tmp_path
):
    labeling = pytest.importorskip("ml4t.engineer.labeling")
    from ml4t.specs import FeedSpec

    from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
    from ml4t.diagnostic.signal import (
        analyze_signal_information,
        analyze_signal_returns,
        prepare_signal_panel,
    )
    from ml4t.diagnostic.visualization.signal import generate_signal_report

    fixtures = ROOT / "tests/test_signal/data/outcomes"
    factor = pl.read_ipc(fixtures / "factor.arrow")
    prices = pl.read_ipc(fixtures / ("prices-missing.arrow" if missing else "prices.arrow"))
    provenance = json.loads((fixtures / "provenance.json").read_text())
    feed = FeedSpec.from_mapping(provenance["feed_spec"])
    outcomes = labeling.forward_returns(
        prices.lazy(),
        horizons={"1ns": "1ns", "2ns": "2ns"},
        price_basis="raw",
        entries=factor.select("timestamp", "asset"),
        feed_spec=feed,
        window_unit="elapsed",
        available_at_col="known_at",
        missing_price_policy="null",
        tolerance="0ns",
        endpoint_method="exit_bar" if exit_bar else "cumulative",
    ).collect()
    original = outcomes.clone()
    panel = prepare_signal_panel(
        factor,
        outcomes,
        horizon_col="horizon",
        feed_spec=feed,
        quantiles=2,
        max_loss=1 / 6,
    )
    assert panel.metadata["loss_by_stage"]["returns"] == (2 if missing else 0)
    assert panel.metadata["calendar"] is None
    assert panel.outcome_data.select(outcomes.columns).equals(outcomes)
    for row in panel.data.with_columns(pl.col("timestamp").cast(pl.Int64)).iter_rows(named=True):
        for horizon, steps in (("1ns", 1), ("2ns", 2)):
            start_time = row["timestamp"] + (steps - 1 if exit_bar else 0)
            start = prices.filter(
                (pl.col("asset") == row["asset"])
                & (pl.col("timestamp").cast(pl.Int64) == start_time)
            ).item(0, "close")
            finish = prices.filter(
                (pl.col("asset") == row["asset"])
                & (pl.col("timestamp").cast(pl.Int64) == row["timestamp"] + steps)
            ).item(0, "close")
            assert row[horizon] == pytest.approx((finish - start) / start)
    returns = analyze_signal_returns(panel, annualization_factors={"1ns": 1, "2ns": 1})
    information = analyze_signal_information(panel, min_obs=2)
    turnover = TurnoverAnalysisResult.from_observations(
        panel.quantile_turnover(lags=[1]), panel.rank_autocorrelation(lags=[1])
    )
    output = tmp_path / "engineer-report.html"
    generate_signal_report(returns, information, turnover, output_file=output, rolling_window=2)
    html = output.read_text(encoding="utf-8")
    assert 'id="return-regression"' in html and 'id="information-summary"' in html
    assert returns.get_table("prepared_data").equals(panel.data)
    assert outcomes.equals(original)
