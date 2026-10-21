"""Capacity uses declared units and past available bars, not future liquidity."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis import analyze_capacity


def _inputs():
    dates = [date(2025, 1, 6) + timedelta(days=i) for i in range(4)]
    positions = pl.DataFrame(
        {
            "date": [day for day in dates for _ in range(2)],
            "asset": ["A", "B"] * 4,
            "value": [10000.0, -10000.0] * 4,
        }
    )
    market = pl.DataFrame(
        {
            "date": [day for day in dates for _ in range(2)],
            "asset": ["A", "B"] * 4,
            "price": [10.0] * 8,
            "volume": [10000.0] * 8,
        }
    )
    return positions, market


def _analyze(positions, market, **kwargs):
    return analyze_capacity(
        positions,
        market,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        complete_snapshots=True,
        calendar=None,
        lookback=1,
        **kwargs,
    )


def test_long_and_short_liquidation_use_dollars_not_shares_or_signed_exposure():
    positions, market = _inputs()
    result = _analyze(positions, market)
    assert result.liquidation["liquidation_duration"].to_list() == [
        None,
        None,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert result.timeline["book_liquidation_duration"].to_list() == [None, 1.0, 1.0, 1.0]
    assert result.metadata["window_unit"] == "observation"
    exported = result.liquidation
    exported.replace_column(2, pl.Series("value", [0.0] * 8))
    assert result.liquidation["value"].to_list() == [10000.0, -10000.0] * 4


def test_zero_volume_is_unbounded_while_missing_coverage_stays_unavailable():
    positions, market = _inputs()
    market = market.with_columns(
        pl.when(pl.col("asset") == "A").then(0.0).otherwise(None).alias("volume")
    )
    result = _analyze(positions, market)
    assert np.isposinf(result.liquidation["liquidation_duration"][2])
    assert result.liquidation["liquidation_duration"][3] is None
    assert result.timeline["book_liquidation_duration"].to_list() == [None] * 4
    assert result.metadata["unbounded_liquidation_rows"] == 3


def test_current_and_future_volume_cannot_change_past_liquidation_denominators():
    positions, market = _inputs()
    baseline = _analyze(positions, market).liquidation
    future = market.with_columns(
        pl.when(pl.col("date") >= date(2025, 1, 8))
        .then(1e9)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    actual = _analyze(positions, future).liquidation
    assert baseline.head(6).equals(actual.head(6))
    assert actual["liquidation_duration"][6] == pytest.approx(0.00001)


def test_missing_asset_row_is_not_compressed_out_of_the_common_market_clock():
    positions, market = _inputs()
    sparse = market.filter(~((pl.col("date") == date(2025, 1, 7)) & (pl.col("asset") == "A")))
    result = _analyze(positions, sparse).liquidation
    assert result["liquidation_duration"][4] is None
    assert result["liquidation_duration"][5] == 1.0


def test_late_market_publication_cannot_be_used_before_its_declared_known_at():
    positions, market = _inputs()
    market = market.with_columns(
        pl.when(pl.col("date") == date(2025, 1, 6))
        .then(date(2025, 1, 8))
        .otherwise(pl.col("date"))
        .alias("available")
    )
    result = _analyze(positions, market, known_at_col="available").liquidation
    assert result["liquidation_duration"].to_list()[:6] == [None] * 4 + [1.0, 1.0]


def test_shared_feed_mapping_is_applied_once_without_mutating_caller_frames():
    positions, market = _inputs()
    renamed = market.rename(
        {"date": "time", "asset": "instrument", "price": "mark", "volume": "shares"}
    )
    before = renamed.clone()
    spec = FeedSpec(
        timestamp_col="time",
        entity_col="instrument",
        price_col="mark",
        volume_col="shares",
        time_contract_version=2,
        calendar=None,
    )
    result = analyze_capacity(
        positions,
        renamed,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        complete_snapshots=True,
        feed_spec=spec,
        lookback=1,
    )
    assert result.liquidation["liquidation_duration"].to_list()[2:] == [1.0] * 6
    assert renamed.equals(before)
    with pytest.raises(ValueError, match="conflicts"):
        _analyze(positions, renamed, feed_spec=spec, entity_col="asset")


def test_explicit_calendar_none_overrides_declared_calendar_and_preserves_ns_rows():
    times = pd.to_datetime(["2025-01-11T12:00:00.000000001Z", "2025-01-11T12:00:00.000000002Z"])
    positions = pl.from_pandas(
        pd.DataFrame({"date": times, "asset": ["A"] * 2, "value": [10000.0] * 2})
    )
    market = positions.drop("value").with_columns(
        pl.lit(10.0).alias("price"), pl.lit(10000.0).alias("volume")
    )
    spec = FeedSpec(
        timestamp_col="date",
        entity_col="asset",
        price_col="price",
        calendar="NYSE",
        time_contract_version=2,
    )
    result = _analyze(positions, market, feed_spec=spec)
    assert result.liquidation["date"].dtype == pl.Datetime("ns", "UTC")
    assert result.liquidation["liquidation_duration"].to_list() == [None, 1.0]


def test_capital_scenario_uses_quadratic_volume_share_and_beginning_nav_once():
    positions, market = _inputs()
    transactions = pl.DataFrame(
        {"timestamp": [date(2025, 1, 7)] * 2, "asset": ["A", "B"], "quantity": [100.0, -100.0]}
    )
    returns = (
        positions.select("date")
        .unique()
        .sort("date")
        .with_columns(pl.Series("return", [0.1, -0.1, 0.0, 0.1]))
    )
    result = _analyze(
        positions,
        market,
        transactions=transactions,
        returns=returns,
        base_capital=100000.0,
        scenario_capitals=[100000.0, 200000.0],
    )
    assert result.participation["participation"].to_list() == [0.01, 0.01]
    # Each asset: (.01**2)*.1*(100*10) = .01 cash. NAV before day2=110000.
    assert result.scenarios["extra_impact_cash"].to_list() == [
        0.0,
        0.02,
        0.0,
        0.0,
        0.0,
        0.16,
        0.0,
        0.0,
    ]
    assert result.scenarios["adjusted_return"][1] == pytest.approx(-0.1 - 0.02 / 110000)
    assert result.scenarios["adjusted_return"][5] == pytest.approx(-0.1 - 0.16 / 220000)


def test_missing_active_volume_cannot_turn_a_trade_into_zero_cost():
    positions, market = _inputs()
    market = market.filter(~((pl.col("date") == date(2025, 1, 7)) & (pl.col("asset") == "A")))
    transactions = pl.DataFrame(
        {"timestamp": [date(2025, 1, 7)], "asset": ["A"], "quantity": [100.0]}
    )
    returns = (
        positions.select("date").unique().sort("date").with_columns(pl.lit(0.01).alias("return"))
    )
    result = _analyze(
        positions,
        market,
        transactions=transactions,
        returns=returns,
        base_capital=100000.0,
        scenario_capitals=[100000.0],
    )
    assert result.scenarios["extra_impact_cash"].to_list() == [0.0, None, 0.0, 0.0]
    assert result.scenarios["adjusted_return"][1] is None


def test_incompatible_units_and_duplicate_market_keys_fail_before_analysis():
    positions, market = _inputs()
    with pytest.raises(ValueError, match="same adjustment basis"):
        analyze_capacity(
            positions,
            market,
            currency="USD",
            price_basis="split_adjusted",
            volume_basis="raw",
            complete_snapshots=True,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _analyze(positions, pl.concat([market, market.head(1)]))


def test_trailing_mean_is_mean_of_dollar_volume_not_product_of_separate_means():
    positions, market = _inputs()
    market = market.with_columns(
        pl.when(pl.col("date") == date(2025, 1, 6)).then(5.0).otherwise(20.0).alias("price"),
        pl.when(pl.col("date") == date(2025, 1, 6)).then(20000.0).otherwise(5000.0).alias("volume"),
    )
    result = analyze_capacity(
        positions,
        market,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        complete_snapshots=True,
        calendar=None,
        lookback=2,
    )
    assert result.liquidation["lagged_dollar_volume"].to_list() == [None] * 4 + [100000.0] * 4
    assert result.liquidation["liquidation_duration"].to_list() == [None] * 4 + [1.0] * 4


def test_session_window_keeps_a_whole_missing_scheduled_market_row():
    positions, market = _inputs()
    positions = positions.filter(pl.col("date") != date(2025, 1, 7))
    market = market.filter(pl.col("date") != date(2025, 1, 7))
    positions = positions.with_columns((pl.col("date") + pl.duration(days=7)).alias("date"))
    market = market.with_columns((pl.col("date") + pl.duration(days=7)).alias("date"))
    result = analyze_capacity(
        positions,
        market,
        currency="USD",
        price_basis="raw",
        volume_basis="raw",
        complete_snapshots=True,
        calendar="NYSE",
        lookback=1,
        window_unit="session",
    )
    assert result.liquidation["liquidation_duration"].to_list() == [None] * 4 + [1.0] * 2


def test_market_and_execution_point_multipliers_cannot_silently_disagree():
    positions, market = _inputs()
    market = market.with_columns(pl.lit(100.0).alias("multiplier"))
    result = _analyze(positions, market)
    assert result.liquidation["liquidation_duration"].to_list()[2:] == [0.01] * 6
    fills = pl.DataFrame(
        {"timestamp": [date(2025, 1, 7)], "asset": ["A"], "quantity": [1.0], "multiplier": [1.0]}
    )
    with pytest.raises(ValueError, match="multipliers"):
        _analyze(positions, market, transactions=fills)


def test_owned_account_capacity_preserves_observed_all_cash_dates_without_fills():
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis

    positions, market = _inputs()
    returns = (
        positions.select("date").unique().sort("date").with_columns(pl.lit(0.01).alias("return"))
    )
    positions = positions.with_columns(
        pl.when(pl.col("date") == date(2025, 1, 7))
        .then(0.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, calendar=None, complete_snapshots=True
    )
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    assert result.timeline["book_liquidation_duration"].to_list() == [None, 0.0, 1.0, 1.0]
    assert result.timeline["open_positions"].to_list() == [2, 0, 2, 2]
    assert result.participation.is_empty()
    assert result.metadata["transactions_supplied"] is False


def test_actual_owned_cash_only_boundary_retains_clock_without_fake_cash_asset():
    from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis

    positions, market = _inputs()
    returns = (
        positions.select("date").unique().sort("date").with_columns(pl.lit(0.01).alias("return"))
    )
    cash = returns.select("date").with_columns(pl.lit(10000.0).alias("cash"))
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=cash, complete_snapshots=True, calendar=None
    )
    result = analysis.compute_capacity(
        market, currency="USD", price_basis="raw", volume_basis="raw", lookback=1
    )
    assert result.timeline["book_liquidation_duration"].to_list() == [0.0] * 4
    assert result.liquidation.is_empty()
    assert result.worst_positions.is_empty()
