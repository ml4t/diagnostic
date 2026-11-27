"""Trade SHAP explains each backtest trade from the feature row its order was decided on.

A ``TradeRecord`` is stamped at its exit, and ml4t-backtest fills an order on the bar after
the one it was submitted on. The row that explains a trade is therefore the trade's own
symbol's row at the submission bar: not the exit row, not the entry bar, and never another
asset's row stamped at the same time. Each SHAP row below carries its own row number, so an
explanation names the row it came from.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config import TradeAlignmentSettings, TradeConfig
from ml4t.diagnostic.evaluation import (
    TradeExplainFailure,
    TradeShapAnalyzer,
    TradeShapExplanation,
)
from ml4t.diagnostic.integration.backtest_contract import TradeRecord

backtest = pytest.importorskip("ml4t.backtest")

SYMBOLS = ["AAA", "BBB", "CCC"]
DATES = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
FEATURES = ["momentum", "volatility"]


def _bars(symbols: list[str]) -> pl.DataFrame:
    rows = []
    for j, symbol in enumerate(symbols):
        for i, ts in enumerate(DATES):
            price = 100.0 + 10 * j + (-1) ** (i + j) * 0.5 * i
            rows.append(
                {
                    "timestamp": ts,
                    "asset": symbol,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": 1e6,
                }
            )
    return pl.DataFrame(rows)


class _Rotate(backtest.Strategy):
    """Hold each symbol in turn for two bars, recording the bar each order was decided on."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbols = symbols
        self.step = 0
        self.decided: list[tuple[str, datetime]] = []

    def on_data(self, timestamp, data, context, broker):
        if self.step % 3 == 0 and self.step < 9:
            symbol = self.symbols[(self.step // 3) % len(self.symbols)]
            broker.submit_order(symbol, 10.0)
            self.decided.append((symbol, timestamp.replace(tzinfo=None)))
        elif self.step % 3 == 2:
            for asset, pos in broker.get_positions().items():
                if pos.quantity != 0:
                    broker.close_position(asset)
        self.step += 1


def _run(symbols: list[str]) -> tuple[list[TradeRecord], list[tuple[str, datetime]]]:
    strategy = _Rotate(symbols)
    result = backtest.Engine(
        backtest.DataFeed(prices_df=_bars(symbols)),
        strategy,
        backtest.BacktestConfig(initial_cash=100_000),
    ).run()
    trades = [
        TradeRecord(
            timestamp=t.exit_time.replace(tzinfo=None),
            entry_timestamp=t.entry_time.replace(tzinfo=None),
            symbol=t.symbol,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl=t.pnl,
            duration=t.exit_time - t.entry_time,
            direction="long",
            quantity=abs(t.quantity),
        )
        for t in result.trades
        if t.exit_time is not None
    ]
    trades.sort(key=lambda t: t.entry_timestamp)
    assert len(trades) == len(strategy.decided) == 3
    # The engine fills on the bar after the decision; the exit is later still.
    for trade, (symbol, decided_at) in zip(trades, strategy.decided, strict=True):
        assert trade.symbol == symbol
        assert trade.entry_timestamp > decided_at
        assert trade.timestamp > trade.entry_timestamp
    return trades, strategy.decided


def _panel(symbols: list[str], *, with_symbol: bool) -> tuple[pl.DataFrame, np.ndarray]:
    panel = pl.DataFrame(
        [
            {"timestamp": ts, "symbol": symbol, "momentum": 0.1 * i, "volatility": 0.2 * j}
            for i, ts in enumerate(DATES)
            for j, symbol in enumerate(symbols)
        ]
    )
    # Column 0 of each SHAP row is the row's own number.
    shap_values = np.column_stack([np.arange(panel.height, dtype=float), np.ones(panel.height)])
    return (panel if with_symbol else panel.drop("symbol")), shap_values


def _source_row(panel: pl.DataFrame, explanation: TradeShapExplanation) -> dict:
    return panel.row(int(explanation.shap_vector[0]), named=True)


def test_multi_asset_panel_explains_each_trade_from_its_symbols_decision_row():
    trades, decided = _run(SYMBOLS)
    panel, shap_values = _panel(SYMBOLS, with_symbol=True)
    analyzer = TradeShapAnalyzer(model=None, features_df=panel, shap_values=shap_values)

    result = analyzer.explain_worst_trades(trades, n=len(trades))

    assert result.n_trades_explained == len(trades)
    assert analyzer.feature_names == FEATURES
    for explanation in result.explanations:
        trade = next(t for t in trades if t.timestamp == explanation.timestamp)
        row = _source_row(panel, explanation)
        assert (row["symbol"], row["timestamp"]) == next(d for d in decided if d[0] == trade.symbol)
        assert explanation.symbol == trade.symbol
        assert explanation.entry_timestamp == trade.entry_timestamp
        assert explanation.feature_timestamp == row["timestamp"]
        assert explanation.feature_values["volatility"] == pytest.approx(row["volatility"])


def test_single_asset_frame_without_symbol_uses_decision_row_not_exit_row():
    trades, decided = _run(["AAA"])
    panel, shap_values = _panel(["AAA"], with_symbol=False)
    analyzer = TradeShapAnalyzer(model=None, features_df=panel, shap_values=shap_values)

    result = analyzer.explain_worst_trades(trades, n=len(trades))

    assert result.n_trades_explained == len(trades)
    decided_at = {t.timestamp: d for t, (_, d) in zip(trades, decided, strict=True)}
    for explanation in result.explanations:
        used = _source_row(panel, explanation)["timestamp"]
        assert used == decided_at[explanation.timestamp]
        assert used < explanation.entry_timestamp < explanation.timestamp


def test_at_or_before_entry_uses_row_stamped_at_entry_for_shifted_features():
    """Features stamped when they become usable put the decision on the entry bar."""
    trades, _ = _run(SYMBOLS)
    panel, shap_values = _panel(SYMBOLS, with_symbol=True)
    config = TradeConfig(alignment=TradeAlignmentSettings(decision_time="at_or_before_entry"))
    analyzer = TradeShapAnalyzer(
        model=None, features_df=panel, shap_values=shap_values, config=config
    )

    result = analyzer.explain_worst_trades(trades, n=len(trades))

    assert result.n_trades_explained == len(trades)
    for explanation in result.explanations:
        row = _source_row(panel, explanation)
        assert row["symbol"] == explanation.symbol
        assert row["timestamp"] == explanation.entry_timestamp


def test_trade_without_entry_time_is_reported_not_explained_from_its_exit():
    trades, _ = _run(["AAA"])
    panel, shap_values = _panel(["AAA"], with_symbol=False)
    analyzer = TradeShapAnalyzer(model=None, features_df=panel, shap_values=shap_values)
    exit_only = trades[0].model_copy(update={"entry_timestamp": None, "duration": None})

    outcome = analyzer.explain_trade(exit_only)

    assert isinstance(outcome, TradeExplainFailure)
    assert outcome.reason == "entry_time_missing"


def test_explanation_identifies_the_exact_nanosecond_feature_row():
    import pandas as pd

    base = pd.Timestamp("2024-01-01").value
    ticks = [base + 101, base + 899, base + 1100]
    frame = pl.DataFrame(
        {"timestamp": pl.Series(ticks).cast(pl.Datetime("ns")), "value": [1.0, 2.0, 3.0]}
    )
    trade = TradeRecord(
        timestamp=pd.Timestamp(base + 2000),
        entry_timestamp=pd.Timestamp(base + 1000),
        duration=timedelta(microseconds=1),
        symbol="A",
        entry_price=1.0,
        exit_price=2.0,
        pnl=1.0,
        direction="long",
        quantity=1,
    )
    analyzer = TradeShapAnalyzer(
        model=None, features_df=frame, shap_values=np.array([[1.0], [2.0], [3.0]])
    )
    result = analyzer.explain_trade(trade)
    assert isinstance(result, TradeShapExplanation)
    assert result.feature_values["value"] == 2.0
    assert result.feature_timestamp == pd.Timestamp(ticks[1])
