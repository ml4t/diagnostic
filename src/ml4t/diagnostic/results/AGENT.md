# results/ - Result Dataclasses

Immutable results with `.to_dict()`, `.summary()`, `.get_dataframe()`.

## Modules

| File | Purpose |
|------|---------|
| base.py | `BaseResult` abstract |
| feature_results.py | Feature diagnostics |
| sharpe_results.py | DSR, PSR results |
| event_results.py | Event study results |
| portfolio_results.py | Portfolio analysis |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| signal_results/ | Signal analysis results (7 modules) |
| barrier_results/ | Barrier analysis results (6 modules) |

## Key Classes

`SignalResult`, `BarrierTearSheet`, `DSRResult`, `FeatureDiagnosticsResult`, `EventStudyResult`
