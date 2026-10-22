"""Actual bar account snapshots must retain beginning NAV and availability in attribution."""

import inspect
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.factor import compute_holdings_attribution
from ml4t.diagnostic.visualization.backtest import export_workspaces


def captured_inputs():
    backtest = pytest.importorskip("ml4t.backtest")
    if "holdings_sampling" not in inspect.signature(backtest.Engine).parameters:
        pytest.skip("Installed Backtest lacks the separately qualified opt-in holdings capture")
    from ml4t.backtest.config import CommissionType, SlippageType

    class LongShort(backtest.Strategy):
        def __init__(self):
            self.ordered = False

        def on_data(self, timestamp, data, context, broker):
            if not self.ordered:
                broker.submit_order("A", 4)
                broker.submit_order("B", -3)
                self.ordered = True

    dates = [datetime(2025, 7, 7, 14, minute, tzinfo=UTC) for minute in range(4)]
    prices = pl.DataFrame(
        {
            "timestamp": [stamp for stamp in dates for _ in range(2)],
            "asset": ["A", "B"] * 4,
            **dict.fromkeys(
                ["open", "high", "low", "close"],
                [100.0, 100.0, 100.0, 100.0, 110.0, 90.0, 121.0, 81.0],
            ),
            "volume": [10000.0] * 8,
        }
    )
    produced = backtest.Engine(
        backtest.DataFeed(prices_df=prices),
        LongShort(),
        backtest.BacktestConfig(
            initial_cash=1000,
            calendar="NYSE",
            allow_short_selling=True,
            commission_type=CommissionType.NONE,
            slippage_type=SlippageType.NONE,
        ),
        holdings_sampling="bar",
    ).run()
    account = produced.to_portfolio_state_dataframe()
    holdings = produced.to_holdings_dataframe()
    np.testing.assert_allclose(account["equity"], [1000, 1000, 1070, 1141])
    np.testing.assert_allclose(account["cash"], [1000, 900, 900, 900])
    assert account["holdings_complete"].to_list() == [True] * 4
    assert holdings.filter(pl.col("timestamp") == dates[0]).is_empty()
    intervals = pl.DataFrame(
        {
            "timestamp": account["timestamp"].slice(1),
            "period_start": account["timestamp"].head(3),
            "beginning_nav": account["equity"].head(3),
            "return": account["equity"].pct_change().slice(1),
        }
    )
    weights = holdings.join(intervals, left_on="timestamp", right_on="period_start").select(
        pl.col("timestamp_right").alias("timestamp"),
        "asset",
        pl.col("timestamp").alias("known_at"),
        (pl.col("signed_notional") / pl.col("beginning_nav")).alias("weight"),
    )
    loadings = weights.select("timestamp", "asset", "known_at").with_columns(
        pl.when(pl.col("asset") == "A").then(1.0).otherwise(-1.0).alias("market")
    )
    factors = intervals.select("timestamp").with_columns(pl.Series("market", [0.0, 0.1, 0.1]))
    return intervals, weights, factors, loadings, account, holdings


def test_captured_long_short_and_all_cash_observations_reconcile_actual_nav_and_report():
    intervals, weights, factors, loadings, _, _ = captured_inputs()
    result = compute_holdings_attribution(
        intervals.select("timestamp", "period_start", "return"),
        weights,
        factors,
        loadings,
        complete_snapshots=True,
        observed_snapshot_dates=intervals["timestamp"],
    )
    np.testing.assert_allclose(result.exposures["market"], [0.0, 0.7, 71 / 107])
    np.testing.assert_allclose(result.contributions["common_returns"], [0.0, 0.07, 71 / 1070])
    np.testing.assert_allclose(
        result.contributions["specific_returns"], [0.0, 0.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(result.cumulative["total_returns"], [0.0, 0.07, 0.141])
    html = export_workspaces(
        ("factors",),
        returns=intervals["return"].to_numpy(),
        metrics={"holdings_attribution": result},
        periods_per_year=252 * 390,
    )
    assert "14.1000%" in html and "not annualized" in html


def test_actual_bar_end_weights_cannot_be_reinterpreted_as_beginning_period_information():
    intervals, _, factors, loadings, account, holdings = captured_inputs()
    weights = holdings.join(account.select("timestamp", "equity"), on="timestamp").select(
        "timestamp",
        "asset",
        pl.col("timestamp").alias("known_at"),
        (pl.col("signed_notional") / pl.col("equity")).alias("weight"),
    )
    with pytest.raises(ValueError, match="known_at|beginning"):
        compute_holdings_attribution(
            intervals.select("timestamp", "period_start", "return"),
            weights,
            factors,
            loadings,
            complete_snapshots=True,
            observed_snapshot_dates=intervals["timestamp"],
        )
