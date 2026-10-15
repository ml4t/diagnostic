# signal/ - Signal Validation Facade

Factor-style signal analysis with a clean public surface.

## Main Files

- `core.py` - `analyze_signal`
- `_prepared.py` and `core.py` - strict `analyze_prepared_signal` input and shared calculations
- `result.py` - `SignalResult`
- `signal_ic.py` - signal-specific IC extraction helpers
- `quantile.py` - quantile-return and spread analysis
- `turnover.py` - turnover and signal autocorrelation
- `_report.py` - text/report formatting helpers

## Common Usage

```python
from ml4t.diagnostic import analyze_signal

result = analyze_signal(factor=factor_df, prices=prices_df, periods=(1, 5, 21))
print(result.ic["1D"], result.ic_t_stat["1D"])
```

`analyze_prepared_signal` accepts already-complete panels without prices or new bucket assignments.
Its result retains an owned `prepared_data` snapshot. Supplied horizon labels do not establish a
calendar or session/bar validation; native pooled quantile/spread definitions still apply.
