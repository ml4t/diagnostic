# signal/ - Signal Validation Facade

Factor-style signal analysis with a clean public surface.

## Main Files

- `core.py` - `analyze_signal`
- `_prepared.py` and `core.py` - strict `analyze_prepared_signal` input and shared calculations
- `panel.py` - `prepare_signal_panel`, owned common supplied-return samples and staged losses
- `result.py` - `SignalResult`
- `signal_ic.py` - per-date/group IC observations and legacy finite-series extraction
- `returns.py` - inspectable finite-pair date/group return demeaning shared with IC
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

`prepare_signal_panel` cleans one common factor/return/group sample before shared qcut/cut
bucketing. Its owned panel exposes independent data/audit copies; `analyze()` delegates to the
existing prepared analysis engine for integer-D selectors and contiguous buckets. Do not infer
group-neutral portfolios, calendar validation or full report coverage from that preparation route.
Use `panel.ic_by_date(group_adjust=..., by_group=...)` for explicit adjusted/grouped IC observations,
counts and undefined statuses. `panel.demean_returns(by_group=...)` exposes the shared adjusted
outcomes without changing keys or buckets. The direct `compute_signal_ic` and `demean_signal_returns`
routes accept explicit supplied return-column selectors, without interpreting them as durations.
