# Trade Analysis

`TradeAnalysis` ranks individual trades and computes aggregate statistics from
normalized `TradeRecord` objects.

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

print([trade.pnl for trade in worst])
print(statistics.summary())
```

`worst_trades` sorts by PnL ascending. `best_trades` sorts by PnL descending.
Use `TradeFilters` to restrict symbols, dates, duration, or PnL before analysis.

SHAP-based trade diagnostics require model-aligned feature rows and SHAP values.
Use the executable dashboard script in `examples/trade_shap_dashboard_demo.py`
as the supported starting point for that workflow.
