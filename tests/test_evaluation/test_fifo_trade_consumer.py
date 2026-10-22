"""Existing trade analytics must consume real FIFO rows without changing accounting facts."""

from datetime import timedelta

import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation import PortfolioAnalysis, TradeAnalysis
from ml4t.diagnostic.evaluation.trade_analysis import TradeMetrics
from ml4t.diagnostic.integration import TradeRecord


def _produced_trade(
    *,
    prices=(100.0, 110.0),
    multiplier=1.0,
    same_instant=False,
    nanosecond_duration=False,
    fees=(0.0, 0.0),
):
    dates = pd.date_range("2025-07-07T15:00:00.000000001Z", periods=2)
    timestamps = [dates[0], dates[0]] if same_instant else dates
    if nanosecond_duration:
        timestamps = [dates[0], dates[0] + pd.Timedelta(1, unit="ns")]
    analysis = PortfolioAnalysis.from_frames(
        pd.Series([0.0, 0.0], index=dates),
        transactions=pd.DataFrame(
            {
                "timestamp": timestamps,
                "asset": [7, 7],
                "quantity": [2.0, -2.0],
                "price": prices,
                "multiplier": [multiplier] * 2,
                "commission": fees,
            }
        ),
        calendar=None,
        transaction_timestamp_col="timestamp",
    )
    segment = analysis.compute_round_trips(initial_state="flat").segments.to_pandas().iloc[0]
    return TradeRecord(
        timestamp=segment.exit_timestamp,
        entry_timestamp=segment.entry_timestamp,
        symbol="asset7",
        entry_price=segment.entry_price,
        exit_price=segment.exit_price,
        pnl=segment.net_pnl,
        duration=segment.exit_timestamp - segment.entry_timestamp,
        direction=segment.direction,
        quantity=segment.quantity,
        multiplier=segment.multiplier,
        fees=segment.commission,
        metadata={"stable_asset_id": 7},
    )


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, 20.0),
        ({"same_instant": True}, 20.0),
        ({"multiplier": 50.0}, 1000.0),
        ({"prices": (-100.0, -90.0), "multiplier": 50.0}, 1000.0),
        ({"fees": (-1.0, 0.0)}, 21.0),
    ],
)
def test_actual_fifo_trade_reaches_existing_statistics_without_changing_units(options, expected):
    trade = _produced_trade(**options)
    metrics = TradeMetrics.from_trade_record(trade)
    statistics = TradeAnalysis([trade]).compute_statistics()
    assert statistics.total_pnl == expected
    assert statistics.n_trades == 1
    assert trade.quantity == 2.0
    assert metrics.quantity == 2.0
    assert trade.multiplier == options.get("multiplier", 1.0)
    assert metrics.multiplier == trade.multiplier
    assert metrics.metadata["stable_asset_id"] == 7
    assert TradeMetrics.to_dataframe([metrics])["multiplier"].to_list() == [trade.multiplier]


def test_same_instant_trade_keeps_cash_profit_but_has_undefined_pnl_per_day():
    trade = _produced_trade(same_instant=True)
    metrics = TradeMetrics.from_trade_record(trade)
    assert metrics.pnl == 20.0
    assert metrics.duration_days == 0.0
    assert metrics.pnl_per_day is None
    assert metrics.to_dict()["pnl_per_day"] is None
    assert TradeMetrics.to_dataframe([metrics])["pnl_per_day"].to_list() == [None]
    assert "asset7" in TradeAnalysis([trade]).analyze().summary()


def test_zero_entry_price_keeps_cash_pnl_and_explicitly_undefined_percentage_return():
    trade = _produced_trade(prices=(0.0, 10.0))
    metrics = TradeMetrics.from_trade_record(trade)
    assert metrics.pnl == 20.0
    assert metrics.return_pct is None
    assert metrics.to_dict()["return_pct"] is None
    assert TradeMetrics.to_dataframe([metrics])["return_pct"].to_list() == [None]
    assert "undefined" in TradeAnalysis([trade]).analyze().summary()


def test_consumer_frames_and_json_retain_producer_nanosecond_instants():
    trade = _produced_trade()
    instant = pd.Timestamp("2025-07-07T15:00:00.000000001Z")
    metrics = TradeMetrics.from_trade_record(trade)
    frame = TradeMetrics.to_dataframe([metrics])
    assert frame["entry_timestamp"].cast(pl.Int64).to_list() == [instant.value]
    assert "000000001" in trade.model_dump_json()
    assert "000000001" in metrics.to_dict()["entry_timestamp"]
    restored = TradeRecord.model_validate_json(trade.model_dump_json())
    assert pd.Timestamp(restored.entry_timestamp).value == instant.value


def test_one_nanosecond_trade_duration_is_not_rounded_to_zero_in_frames_or_json():
    trade = _produced_trade(nanosecond_duration=True)
    metrics = TradeMetrics.from_trade_record(trade)
    assert metrics.duration_seconds == pytest.approx(1.0e-9, rel=1.0e-14, abs=0.0)
    assert metrics.pnl_per_day == pytest.approx(20.0 / (1.0e-9 / 86400.0))
    assert TradeMetrics.to_dataframe([metrics])["duration_seconds"].to_list() == [1.0e-9]
    for model, value in ((TradeRecord, trade), (TradeMetrics, metrics)):
        restored = model.model_validate_json(value.model_dump_json())
        assert restored.duration == pd.Timedelta(1, unit="ns")


@pytest.mark.parametrize(
    "field", ["entry_price", "exit_price", "pnl", "quantity", "multiplier", "fees", "slippage"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_cash_price_or_units_cannot_bypass_trade_model_validation(field, value):
    values = _produced_trade().model_dump()
    values[field] = value
    for model in (TradeRecord, TradeMetrics):
        with pytest.raises(ValueError, match=field):
            model(**values)


def test_negative_durations_remain_invalid_after_accepting_real_zero_duration_trades():
    values = _produced_trade().model_dump()
    values["duration"] = timedelta(seconds=-1)
    for model in (TradeRecord, TradeMetrics):
        with pytest.raises(ValueError, match="Duration"):
            model(**values)


def test_integer_execution_prices_are_not_rounded_by_the_completed_trade_consumer():
    price = 2**53 + 1
    trade = _produced_trade(prices=(price, price + 1))
    assert trade.entry_price == price
    assert trade.exit_price == price + 1
    metrics = TradeMetrics.from_trade_record(trade)
    assert metrics.entry_price == price
    assert metrics.exit_price == price + 1
    assert TradeAnalysis([trade]).compute_statistics().total_pnl == 2.0
