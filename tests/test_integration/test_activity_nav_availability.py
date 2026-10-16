"""An actually executed closing fill must not become zero turnover at nonpositive NAV."""

from datetime import UTC, datetime
from html.parser import HTMLParser

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result
from ml4t.diagnostic.integration.backtest_analytics import compute_activity_metrics


class _ShortThenCover(Strategy):
    def __init__(self, close_observation=3):
        self.observations = 0
        self.close_observation = close_observation

    def on_data(self, timestamp, data, context, broker):
        self.observations += 1
        if self.observations == 1:
            broker.submit_order("A", -10)
        elif self.observations == self.close_observation:
            broker.submit_order("A", 10)


def _actual_result(mark=200.0, close_observation=3):
    n = close_observation + 1
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7][:n]],
            "asset": ["A"] * n,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, mark, mark][:n]),
            "volume": [10000.0] * n,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _ShortThenCover(close_observation),
        BacktestConfig(initial_cash=1000, allow_short_selling=True),
    ).run()
    assert len(result.fills) == 2
    assert [fill.quantity for fill in result.fills] == [10, 10]
    assert [fill.price for fill in result.fills] == [100, mark]
    assert [fill.timestamp for fill in result.fills] == [
        prices["timestamp"][1],
        prices["timestamp"][-1],
    ]
    return result


@pytest.mark.parametrize("mark", [200.0, 250.0])
def test_actual_closing_fill_at_nonpositive_nav_retains_notional_but_no_turnover_ratio(mark):
    result = _actual_result(mark)
    raw = result.to_portfolio_state_dataframe().clone()
    fills = result.to_fills_dataframe().clone()
    activity = analyze_backtest_result(result).activity
    last = activity["turnover_timeline"].row(-1, named=True)
    assert last["equity"] == 2000 - 10 * mark
    assert last["filled_notional"] == 10 * mark and last["implementation_cost"] == 0
    assert last["turnover"] is None and last["cost_drag"] is None
    assert last["normalization_status"] == "nonpositive_equity"
    assert activity["metrics"]["num_fills"] == 2
    assert activity["metrics"]["total_filled_notional"] == 1000 + 10 * mark
    assert result.to_portfolio_state_dataframe().equals(raw)
    assert result.to_fills_dataframe().equals(fills)


@pytest.mark.parametrize("mark", [200.0, 250.0])
def test_actual_active_observation_mean_excludes_undefined_closing_ratio_and_reports_coverage(mark):
    profile = analyze_backtest_result(_actual_result(mark))
    activity = profile.activity
    assert activity["metrics"]["avg_turnover"] == 1
    assert activity["metrics"]["max_turnover"] == 1
    assert activity["coverage"] == {
        "n_observations": 4,
        "n_normalized_observations": 2,
        "normalization_coverage": 0.5,
        "status": "partial",
        "n_active_observations": 2,
        "n_normalized_active_observations": 1,
        "active_normalization_coverage": 0.5,
    }
    for metric in ["turnover", "cost_drag"]:
        availability = profile.availability.metrics[metric]
        assert availability.status.value == "partial" and availability.coverage == 0.5
        assert "2 of 4" in availability.reason
    family = profile.availability.families["activity"]
    assert family.status.value == "partial" and family.coverage == 0.5
    assert "2 of 4" in family.reason


@pytest.mark.parametrize("mark", [200.0, 250.0])
def test_real_insolvent_window_retains_executions_without_fabricating_aggregate_ratios(mark):
    result = _actual_result(mark)
    state = result.to_portfolio_state_dataframe().filter(pl.col("equity") <= 0)
    fills = result.to_fills_dataframe().filter(
        pl.col("timestamp").is_in(state["timestamp"].implode())
    )
    assert state.height == 2 and fills.height == 1
    activity = compute_activity_metrics(fills, state)
    assert activity["metrics"]["avg_turnover"] is None
    assert activity["metrics"]["max_turnover"] is None
    assert activity["metrics"]["total_filled_notional"] == 10 * mark
    assert activity["coverage"]["n_normalized_observations"] == 0
    assert activity["coverage"]["status"] == "unavailable"


def test_real_positive_equity_no_trade_observations_keep_known_zero_activity():
    activity = analyze_backtest_result(_actual_result(150.0)).activity
    timeline = activity["turnover_timeline"]
    assert timeline["turnover"].to_list() == [0, 1, 0, 3]
    assert timeline["cost_drag"].to_list() == [0, 0, 0, 0]
    assert activity["metrics"]["avg_turnover"] == 2
    assert activity["metrics"]["max_turnover"] == 3


class _ActivityCaption(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.text = []
        self.has_turnover_section = False
        self.kpis = {}
        self.kpi_part = None
        self.kpi_label = None

    def handle_starttag(self, tag, attrs):
        css = dict(attrs).get("class", "")
        if css == "executive-kpi-label":
            self.kpi_part = "label"
        elif css == "executive-kpi-value":
            self.kpi_part = "value"
        if dict(attrs).get("data-section") == "activity_overview":
            self.has_turnover_section = True
        if tag == "p" and dict(attrs).get("data-activity-normalization") == "partial":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "div":
            self.kpi_part = None
        if tag == "p":
            self.active = False

    def handle_data(self, text):
        if self.kpi_part == "label":
            self.kpi_label = text.strip()
        elif self.kpi_part == "value":
            self.kpis[self.kpi_label] = text.strip()
        if self.active:
            self.text.append(text)


def test_actual_zero_nav_closing_fill_full_report_states_its_active_normalization_sample():
    from ml4t.diagnostic.integration import generate_tearsheet_from_result

    result = _actual_result(200.0, close_observation=2)
    state = result.to_portfolio_state_dataframe()
    assert state["equity"].to_list() == [1000, 1000, 0]
    parser = _ActivityCaption()
    parser.feed(generate_tearsheet_from_result(result, include_statistical=False))
    text = " ".join(parser.text)
    assert "2 of 3 observations" in text
    assert "1 of 2 active observations" in text
    assert "defined active observations only" in text
    assert parser.has_turnover_section
    assert parser.kpis["AVG TURNOVER"] == "1.00"


@pytest.mark.parametrize("mark,expected", [(150.0, "2.00"), (200.0, "1.00")])
@pytest.mark.parametrize("template", ["full", "quant_trader", "hedge_fund"])
@pytest.mark.parametrize("periods_per_year", [None, 12])
def test_actual_report_kpi_uses_diagnostic_active_mean_without_mutating_engine_metrics(
    mark, expected, template, periods_per_year
):
    from ml4t.diagnostic.integration import generate_tearsheet_from_result

    result = _actual_result(mark, close_observation=2)
    before = dict(result.metrics)
    parser = _ActivityCaption()
    parser.feed(
        generate_tearsheet_from_result(
            result,
            template=template,
            include_statistical=False,
            periods_per_year=periods_per_year,
        )
    )
    assert parser.kpis["AVG TURNOVER"] == expected
    assert result.metrics == before
