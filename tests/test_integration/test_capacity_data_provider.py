"""Actual optional Data output reaches the independently usable capacity route."""

from decimal import Decimal

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis


def test_actual_offline_data_provider_preserves_market_keys_lag_and_future_invariance():
    provider_module = pytest.importorskip("ml4t.data.providers")
    provider = provider_module.SyntheticProvider(seed=42)
    requested_symbol = "SYNTH"
    try:
        market = provider.fetch_ohlcv(requested_symbol, "2025-01-13", "2025-01-24", "daily")
    finally:
        provider.close()
    if "symbol" not in market.columns:
        market = market.with_columns(pl.lit(requested_symbol).alias("symbol"))
    else:
        assert market["symbol"].null_count() == 0
        assert set(market["symbol"].unique()) == {requested_symbol}
    before = market.clone()
    positions = market.select(
        pl.col("timestamp").alias("date"),
        pl.col("symbol").alias("asset"),
        (10000.0 * pl.col("close") / market["close"][0]).alias("value"),
    )
    returns = market.select(
        pl.col("timestamp").alias("date"),
        (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0.0).alias("return"),
    )
    analysis = PortfolioAnalysis.from_frames(
        returns, positions=positions, complete_snapshots=True, calendar=None
    )
    feed = FeedSpec(
        timestamp_col="timestamp",
        entity_col="symbol",
        price_col="close",
        volume_col="volume",
        time_contract_version=2,
        calendar=None,
    )

    def consume(frame):
        return analysis.compute_capacity(
            frame, currency="USD", price_basis="raw", volume_basis="raw", feed_spec=feed, lookback=1
        )

    result = consume(market)
    assert result.timeline["date"].equals(market["timestamp"].rename("date"))
    assert result.liquidation["asset"].to_list() == market["symbol"].to_list()
    expected_volume = Decimal(str(market["close"][0])) * Decimal(str(market["volume"][0]))
    expected_duration = Decimal(str(positions["value"][1])) / (Decimal("0.1") * expected_volume)
    assert result.liquidation["lagged_dollar_volume"][1] == pytest.approx(
        float(expected_volume), rel=1e-12
    )
    assert result.liquidation["liquidation_duration"][1] == pytest.approx(
        float(expected_duration), rel=1e-12
    )
    changed = market.with_columns(
        pl.when(pl.col("timestamp") >= market["timestamp"][5])
        .then(pl.col("volume") * 1000)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    assert consume(changed).liquidation.head(6).equals(result.liquidation.head(6))
    assert market.equals(before)
