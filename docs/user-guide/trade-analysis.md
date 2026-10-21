# Trade Analysis

`TradeAnalysis` ranks individual trades and computes aggregate statistics from
normalized `TradeRecord` objects.
Each record needs an exit timestamp, symbol, entry and exit prices, PnL,
and duration. Direction and quantity are optional. Compare trades under the same accounting convention;
inconsistent currencies, fees, or partial-fill treatment distort rankings
and aggregate statistics.

## Analyze synthetic trades

```python
from datetime import datetime, timedelta

from ml4t.diagnostic.evaluation import TradeAnalysis
from ml4t.diagnostic.integration.backtest_contract import TradeRecord

trades = [
    TradeRecord(
        timestamp=datetime(2025, 1, day),
        symbol=f"asset_{day % 3}",
        entry_price=100.0,
        exit_price=100.0 + pnl / 100.0,
        pnl=pnl,
        duration=timedelta(days=day % 4 + 1),
        direction="long",
        quantity=100.0,
    )
    for day, pnl in enumerate(
        [-240.0, 180.0, -75.0, 320.0, -410.0, 95.0, 210.0, -130.0],
        start=1,
    )
]

analysis = TradeAnalysis(trades)
worst = analysis.worst_trades(n=3)
statistics = analysis.compute_statistics()

assert [trade.pnl for trade in worst] == [-410.0, -240.0, -130.0]
print([trade.pnl for trade in worst])
print(statistics.summary())
```

`worst_trades` sorts by PnL ascending. `best_trades` sorts by PnL descending.
Use `TradeFilters` to restrict symbols, dates, duration, or PnL before analysis.

SHAP-based trade diagnostics require model-aligned feature rows and SHAP values.
Use the executable dashboard script in `examples/trade_shap_dashboard_demo.py`
as the supported starting point for that workflow.
The dashboard is optional; the core ranking above runs without SHAP or a
browser.

## Preserve execution units and timing

Unreleased after 0.1.9, `TradeRecord` and `TradeMetrics` retain an explicit
`multiplier`, defaulting to one cash unit per price point per execution unit.
For example, these two synthetic contracts realize 1000 cash units without
changing the actual quantity to 100:

```python
point_trade = TradeRecord(
    timestamp=datetime(2025, 1, 2, 16),
    entry_timestamp=datetime(2025, 1, 2, 15),
    symbol="synthetic_contract", entry_price=100., exit_price=110.,
    quantity=2., multiplier=50., direction="long", fees=0.,
    pnl=1000., duration=timedelta(hours=1),
)
assert TradeAnalysis([point_trade]).compute_statistics().total_pnl == 1000.
assert point_trade.quantity == 2.
```

Prices must be finite; zero and negative actual execution prices are valid.
Integer prices remain integers through PnL validation, avoiding cancellation
after premature Float64 conversion. Cash fees are signed: a reported negative
fee is a rebate. Existing positive-price, unit-multiplier calculations retain
their definitions. A trade's supplied PnL includes its cash costs; prices already
reflect execution-price slippage, which must not be charged a second time.

Same-instant entry/exit is valid with zero duration. Negative duration and
reversed entry/exit chronology remain errors. `TradeMetrics.return_pct` uses
absolute entry-price magnitude and is undefined (`None`) for zero entry price;
`pnl_per_day` is undefined for zero duration. Defined cash PnL is retained in
statistics and summaries even when a ratio is undefined. These are price and
duration ratios, not NAV-normalized portfolio contributions.

Pandas nanosecond timestamps and durations retain their precision in JSON and
metric tables. Plain Python datetime inputs cannot recover nanoseconds already
lost upstream. Actual fractional-fill reconstruction is described in the
[FIFO migration workflow](migration.md#match-actual-fractional-fifo-lots-from-an-explicitly-flat-book);
non-flat opening inventory and complete round-trip reports remain separate work.

See the [evaluation API surface](../api/index.md#evaluation-workflows)
for `TradeAnalysis` and the book's
[trade-SHAP notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/19_risk_management/05_trade_shap_diagnostics.ipynb),
which directly calls Diagnostic in a model-aligned workflow. The
[migration guide](migration.md#pyfolio-portfolio-and-performance-analysis)
explains why Pyfolio fills need conversion to completed trades first.
