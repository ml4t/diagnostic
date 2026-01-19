# signal/ - Factor Signal Analysis

Alphalens-style signal quality analysis.

## Modules

| File | Purpose |
|------|---------|
| core.py | `analyze_signal()` entry point |
| result.py | `SignalResult` dataclass |
| ic.py | IC computation |
| quantile.py | Quantile returns, spread |
| turnover.py | Turnover, autocorrelation |

## Key Functions

`analyze_signal()`, `compute_ic_series()`, `compute_quantile_returns()`, `compute_turnover()`
