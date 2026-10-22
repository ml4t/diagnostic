"""Supplied-account capacity, familiar adapters and optional offline Data output.

These APIs are unreleased candidate functionality, not published Diagnostic0.1.9.
Run with --output capacity.html; --compare requires Pyfolio0.10.0. The optional
--data-provider uses Data's actual SyntheticProvider, not network market data.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ml4t.diagnostic.compat.pyfolio import capacity
from ml4t.diagnostic.evaluation.portfolio_analysis import PortfolioAnalysis


def supplied_inputs():
    dates = pd.date_range("2025-01-13", periods=8, freq="B", tz="UTC")
    price = np.array([10., 10.5, 10.3, 10.4, 10.7, 10.2, 10.9, 11.])
    quantity_a = np.array([1000., 1100., 1100., 1050., 1050., 1050., 1050., 1050.])
    quantity_b = np.array([-1000., -1100., -1100., -1100., -1100., -1100., -1100., -1100.])
    # Independent toy account: buy100A/sell100B on day2; sell50A on day4.
    cash = np.array([100000., 99950., 99950., 100470., 100470., 100470., 100470., 100470.])
    positions = pd.DataFrame({"A": quantity_a * price, "B": quantity_b * 10., "cash": cash}, index=dates)
    nav = positions.sum(axis=1)
    returns = nav.pct_change().fillna(0.).rename("returns")
    fills = pd.DataFrame({"symbol": ["A", "B", "A"], "asset": ["A", "B", "A"],
        "quantity": [100., -100., -50.], "amount": [100., -100., -50.],
        "price": [10.5, 10., 10.4]}, index=dates[[1, 1, 3]])
    market = pl.from_pandas(pd.DataFrame({"date": dates.repeat(2), "asset": ["A", "B"] * 8,
        "price": np.column_stack([price, np.repeat(10., 8)]).reshape(-1),
        "volume": [10000.] * 16}))
    legacy_market = pd.DataFrame({asset: np.column_stack([
        price if asset == "A" else np.repeat(10., 8), np.repeat(10000., 8)]).reshape(-1)
        for asset in ("A", "B")}, index=pd.MultiIndex.from_product([dates, ["price", "volume"]]))
    analysis = PortfolioAnalysis.from_frames(returns, positions=positions,
        transactions=fills.rename_axis("timestamp"), valuations=pd.DataFrame({"equity": nav, "cash": cash}, index=dates),
        complete_snapshots=True, calendar=None)
    return analysis, market, returns, positions, fills, legacy_market


def provider_inputs():
    from ml4t.data.providers import SyntheticProvider

    provider = SyntheticProvider(seed=42)
    try:
        market = provider.fetch_ohlcv("SYNTH", "2025-01-13", "2025-01-24", "daily")
    finally:
        provider.close()
    positions = market.select(pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"),
        (10000.*pl.col("close")/market["close"][0]).alias("value"))
    returns = market.select(pl.col("timestamp").alias("date"),
        (pl.col("close")/pl.col("close").shift(1)-1).fill_null(0.).alias("return"))
    analysis = PortfolioAnalysis.from_frames(returns, positions=positions, complete_snapshots=True, calendar=None)
    feed = FeedSpec(timestamp_col="timestamp", entity_col="symbol", price_col="close",
        volume_col="volume", time_contract_version=2, calendar=None)
    print("Actual offline SyntheticProvider output; no empirical liquidity calibration", version("ml4t-data"))
    return analysis, market, feed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--data-provider", action="store_true")
    args = parser.parse_args()
    print({name: version(name) for name in ("ml4t-diagnostic", "pandas", "polars", "numpy", "ml4t-specs")})
    if args.data_provider:
        if args.compare:
            parser.error("--compare uses the shared supplied-account fixture, not the separate provider demonstration")
        analysis, market, feed = provider_inputs()
        result = analysis.compute_capacity(market, currency="USD", price_basis="raw", volume_basis="raw", feed_spec=feed, lookback=1)
    else:
        analysis, market, returns, positions, fills, legacy_market = supplied_inputs()
        result = analysis.compute_capacity(market, currency="USD", price_basis="raw", volume_basis="raw",
            lookback=1, base_capital=100000., scenario_capitals=[100000., 200000., 300000.])
        assert result.liquidation["liquidation_duration"][2] == 1.155
        daily = capacity.daily_txns_with_bar_data(fills, legacy_market)
        assert daily["symbol"].dropna().tolist() == ["A", "B", "A"]
        if args.compare:
            from pyfolio import capacity as reference

            print("Actual comparison reference", version("pyfolio-reloaded"))
            pd.testing.assert_frame_equal(capacity.days_to_liquidate_positions(positions, legacy_market,
                .1, 100000., 1), reference.days_to_liquidate_positions(positions, legacy_market, .1, 100000., 1))
            single = fills[fills["symbol"] == "A"].copy()
            expected_daily = reference.daily_txns_with_bar_data(single.copy(), legacy_market)
            actual_daily = capacity.daily_txns_with_bar_data(single, legacy_market)
            pd.testing.assert_frame_equal(actual_daily, expected_daily, check_dtype=False, check_freq=False)
            pd.testing.assert_series_equal(capacity.apply_slippage_penalty(returns, actual_daily, 200000., 100000.),
                reference.apply_slippage_penalty(returns, expected_daily, 200000., 100000.))
            print("Defined reference workflows match; multi-asset asfreq failure and zero/missing-volume corrections are documented separately")
    print(result.timeline)
    print(result.metadata)
    if args.output:
        analysis.create_tear_sheet(capacity=result).save_html(args.output, include_plotlyjs=True)
        args.output.with_suffix(".json").write_text(result.to_json(), encoding="utf-8")
        print("Wrote HTML and typed JSON", args.output)


if __name__ == "__main__":
    main()
