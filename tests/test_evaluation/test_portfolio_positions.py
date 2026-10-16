"""Prepared holdings must retain actual signed values and observed account keys."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy

from ml4t.diagnostic.evaluation import PortfolioAnalysis


class _LongShort(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("A", 4)
            broker.submit_order("B", -3)
            self.submitted = True


def _actual_prepared_inputs():
    dates = [datetime(2025, 1, day, 15, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame(
        {
            "timestamp": [date for date in dates for _ in range(2)],
            "asset": ["A", "B"] * 3,
            **dict.fromkeys(["open", "high", "low", "close"], [100.0] * 6),
            "volume": [10000.0] * 6,
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        _LongShort(),
        BacktestConfig(initial_cash=1000, allow_short_selling=True),
    ).run()
    fills = result.to_fills_dataframe()
    assert {(r["asset"], r["side"], r["quantity"], r["price"]) for r in fills.to_dicts()} == {
        ("A", "buy", 4, 100),
        ("B", "sell", 3, 100),
    }
    account = result.to_portfolio_state_dataframe()
    positions = pd.DataFrame(
        {"A": [0.0, 400.0, 400.0], "B": [0.0, -300.0, -300.0]},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    np.testing.assert_array_equal(positions.sum(axis=1), account["net_exposure"])
    np.testing.assert_array_equal(positions.abs().sum(axis=1), account["gross_exposure"])
    assert account["cash"].to_list() == [1000, 900, 900]
    assert account["equity"].to_list() == [1000, 1000, 1000]
    valuations = account.select(pl.col("timestamp").alias("date"), "equity", "cash")
    returns = pd.Series([0.0] * 3, index=positions.index)
    return returns, positions, valuations, account


@pytest.mark.parametrize("shape", ["wide", "long"])
def test_actual_prepared_long_short_holdings_match_account_and_independent_concentration(shape):
    returns, wide, valuations, account = _actual_prepared_inputs()
    positions = (
        wide
        if shape == "wide"
        else pl.from_pandas(wide.reset_index()).unpivot(index="date", variable_name="asset")
    )
    analysis = PortfolioAnalysis(returns, positions=positions)
    result = analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    timeline = result.timeline
    assert timeline["date"].to_list() == wide.index.to_list()
    for column in ["gross_exposure", "net_exposure", "equity", "cash", "open_positions"]:
        assert timeline[column].to_list() == account[column].to_list()
    row = timeline.row(-1, named=True)
    assert row["long_exposure"] == 400 and row["short_exposure"] == 300
    assert row["long_positions"] == row["short_positions"] == 1
    assert row["long_exposure_fraction"] == pytest.approx(0.4)
    assert row["short_exposure_fraction"] == pytest.approx(0.3)
    assert row["gross_exposure_fraction"] == pytest.approx(0.7)
    assert row["net_exposure_fraction"] == pytest.approx(0.1)
    assert row["cash_fraction"] == pytest.approx(0.9)
    assert row["largest_gross_share"] == pytest.approx(4 / 7)
    assert row["gross_hhi"] == pytest.approx(25 / 49)
    active = result.positions.filter(pl.col("date") == wide.index[-1]).sort("asset")
    assert active["value"].to_list() == [400, -300]
    assert active["gross_rank"].to_list() == [1, 2]
    np.testing.assert_allclose(active["gross_share"], [4 / 7, 3 / 7])
    np.testing.assert_allclose(active["nav_weight"], [0.4, -0.3])
    cash_only = timeline.row(0, named=True)
    assert cash_only["gross_exposure_fraction"] == cash_only["net_exposure_fraction"] == 0
    assert cash_only["gross_hhi"] is None and cash_only["largest_gross_share"] is None
    assert result.snapshot_kind == "observed"
    assert result.coverage["normalization_coverage"] == 1


@pytest.mark.parametrize("equity", [None, 0.0, -1000.0])
def test_missing_or_nonpositive_reported_nav_keeps_raw_holdings_without_zero_risk(equity):
    returns, positions, valuations, _ = _actual_prepared_inputs()
    valuations = valuations.with_columns(pl.lit(equity, dtype=pl.Float64).alias("equity"))
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True, snapshot_kind="intraday"
    )
    assert result.snapshot_kind == "intraday"
    assert result.timeline["gross_exposure"].to_list() == [0, 700, 700]
    assert result.timeline["gross_exposure_fraction"].null_count() == 3
    assert result.timeline["long_exposure_fraction"].null_count() == 3
    assert result.positions["nav_weight"].null_count() == 6
    assert result.timeline["gross_hhi"][-1] == pytest.approx(25 / 49)
    assert result.metrics["avg_gross_exposure"] is None
    assert result.coverage["status"] == "unavailable"


def test_reported_nav_is_not_reconstructed_from_cash_and_holdings_for_leveraged_accounts():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    valuations = valuations.with_columns(pl.lit(200.0).alias("equity"))
    result = PortfolioAnalysis(returns, positions=positions).compute_holdings_analysis(
        valuations, complete_snapshots=True
    )
    assert result.timeline["gross_exposure_fraction"][-1] == 3.5
    assert result.timeline["cash"][-1] == 900
    assert result.timeline["net_exposure"][-1] == 100


def test_prepared_position_index_and_owned_inputs_survive_caller_and_result_mutation():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    analysis = PortfolioAnalysis(returns, positions=positions)
    positions.iloc[-1, 0] = 99999
    before = valuations.clone()
    first = analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    first.positions.replace_column(2, pl.Series("value", [123.0] * 6))
    first.metrics["avg_gross_exposure"] = 123
    second = analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    assert second.timeline["gross_exposure"][-1] == 700
    assert second.positions["value"].max() == 400
    assert second.metrics["avg_gross_exposure"] == pytest.approx(1.4 / 3)
    assert valuations.equals(before)


@pytest.mark.parametrize(
    "defect", ["missing_date", "duplicate_asset", "missing_value", "wrong_time"]
)
def test_incomplete_or_ambiguous_prepared_holdings_do_not_silently_change_the_sample(defect):
    returns, positions, valuations, _ = _actual_prepared_inputs()
    long = pl.from_pandas(positions.reset_index()).unpivot(index="date", variable_name="asset")
    if defect == "missing_date":
        long = long.filter(pl.col("date") != positions.index[-1])
    elif defect == "duplicate_asset":
        long = pl.concat([long, long.head(1)])
    elif defect == "missing_value":
        long = long.with_columns(
            pl.when(pl.col("asset") == "A").then(None).otherwise("value").alias("value")
        )
    else:
        valuations = valuations.with_columns(pl.col("date") + pl.duration(hours=1))
    before = long.clone()
    analysis = PortfolioAnalysis(returns, positions=long)
    with pytest.raises((ValueError, TypeError), match="date|unique|finite|numeric|observation"):
        analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
    assert long.equals(before)


def test_incomplete_snapshot_declaration_and_undated_positions_cannot_imply_zero_assets():
    returns, positions, valuations, _ = _actual_prepared_inputs()
    analysis = PortfolioAnalysis(returns, positions=positions)
    with pytest.raises(ValueError, match="complete"):
        analysis.compute_holdings_analysis(valuations, complete_snapshots=False)
    analysis = PortfolioAnalysis(returns, positions=positions.reset_index(drop=True))
    with pytest.raises(ValueError, match="date"):
        analysis.compute_holdings_analysis(valuations, complete_snapshots=True)
