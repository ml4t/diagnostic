# Backtest Tearsheets

Generate a standalone HTML report from normalized metrics, trades, and returns.
The minimal supported input is a metrics dictionary or a return series.
The example uses synthetic daily returns and does not require a backtest engine.
Install the `viz` extra for Plotly rendering. Use real returns at the frequency
assumed by the metrics and statistical corrections.

## Generate an HTML report

```python
from pathlib import Path

import numpy as np

from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

rng = np.random.default_rng(42)
daily_returns = rng.normal(loc=0.0005, scale=0.01, size=252)
metrics = {
    "n_trades": 80,
    "total_pnl": 12_500.0,
    "win_rate": 0.54,
    "profit_factor": 1.6,
    "sharpe_ratio": 1.3,
    "max_drawdown": -0.12,
}

output = Path("backtest_report.html")
html = generate_backtest_tearsheet(
    metrics=metrics,
    returns=daily_returns,
    template="quant_trader",
    theme="default",
    output_path=output,
    n_trials=25,
)

assert output.exists()
assert "plotly" in html.lower()
print(f"Wrote {output}")
```

The file-existence and Plotly assertions check that rendering completed.
Inspect reported Sharpe, drawdown, and trial count against your source
backtest; HTML generation does not validate supplied metrics.

## Choose a template

| Template | Primary content |
|----------|-----------------|
| `quant_trader` | Trades, performance, and model diagnostics |
| `hedge_fund` | Performance, costs, and reporting context |
| `risk_manager` | Risk and statistical validation |
| `full` | Every available section |

Use `BacktestProfile` when you have normalized trades, returns, positions,
costs, predictions, and factor results. Use
`generate_tearsheet_from_result` for an `ml4t-backtest` result. The integration
normalizes the backtest object before rendering.

PDF export also depends on the browser and rendering packages documented in the
[installation guide](../getting-started/installation.md).
A headless environment can verify HTML generation without claiming that
browser-dependent PDF export was exercised.

See the [visualization API](../api/index.md#visualization) and the book's
[performance reporting notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/09_performance_reporting.ipynb),
which directly calls Diagnostic in a longer backtest workflow.
