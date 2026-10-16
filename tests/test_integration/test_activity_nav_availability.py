"""An actually executed closing fill must not become zero turnover at nonpositive NAV."""

from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.integration import analyze_backtest_result
from ml4t.diagnostic.integration.backtest_analytics import compute_activity_metrics


class _ShortThenCover(Strategy):
    def __init__(self):
        self.observations = 0

    def on_data(self, timestamp, data, context, broker):
        self.observations += 1
        if self.observations == 1:
            broker.submit_order("A", -10)
        elif self.observations == 3:
            broker.submit_order("A", 10)


def _actual_result(mark=200.0):
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7]],
            "asset": ["A"] * 4,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0, 100.0, mark, mark]),
            "volume": [10000.0] * 4,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _ShortThenCover(),
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
    }
    for metric in ["turnover", "cost_drag"]:
        availability = profile.availability.metrics[metric]
        assert availability.status.value == "partial" and availability.coverage == 0.5


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
