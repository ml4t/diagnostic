"""A flat executed account cannot establish gains without losses."""

import math
from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.evaluation import PortfolioAnalysis


class _BuyOnce(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("A", 4)
            self.submitted = True


@pytest.mark.parametrize("periods_per_year", [12, 252])
@pytest.mark.parametrize(
    ("marks", "expected"),
    [
        ([100.0, 100.0, 100.0, 100.0], math.nan),
        ([100.0, 100.0, 110.0, 120.0], math.inf),
        ([100.0, 100.0, 90.0, 80.0], 0.0),
        ([100.0, 100.0, 110.0, 90.0], 0.52),
    ],
    ids=["flat-undefined", "gains-no-losses", "losses-no-gains", "mixed"],
)
def test_actual_flat_account_profit_factor_is_unavailable_without_changing_defined_controls(
    marks, expected, periods_per_year
):
    bars = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6, 7]],
            "asset": ["A"] * 4,
            **dict.fromkeys(["open", "high", "low", "close"], marks),
            "volume": [10000.0] * 4,
        }
    )
    result = Engine(DataFeed(prices_df=bars), _BuyOnce(), BacktestConfig(initial_cash=1000)).run()
    account, fills = result.to_equity_dataframe(), result.to_fills_dataframe()
    assert fills.height == 1 and fills["quantity"][0] == 4 and fills["price"][0] == 100
    assert account["equity"].to_list() == [
        1000,
        1000,
        1000 + 4 * (marks[2] - 100),
        1000 + 4 * (marks[3] - 100),
    ]
    returns = result.to_returns_series().clone()
    analysis = PortfolioAnalysis(
        returns, dates=account["timestamp"], periods_per_year=periods_per_year
    )
    for force in [False, False, True]:
        metrics = analysis.compute_summary_stats(force_recompute=force)
        values = [
            metrics.profit_factor,
            metrics.to_dict()["profit_factor"],
            metrics.to_dataframe()["profit_factor"][0],
        ]
        for value in values:
            if math.isnan(expected):
                assert math.isnan(value), "No positive or negative returns means unavailable0/0"
            elif math.isinf(expected):
                assert value == math.inf
            else:
                assert value == pytest.approx(expected)
    assert result.to_returns_series().equals(returns)
