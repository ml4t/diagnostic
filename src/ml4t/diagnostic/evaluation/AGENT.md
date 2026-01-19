# evaluation/ - Analysis Framework

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [stats/](stats/AGENT.md) | DSR, RAS, FDR, HAC |
| [metrics/](metrics/AGENT.md) | IC, importance, interactions |
| distribution/ | Moments, tails, tests |
| drift/ | PSI, Wasserstein |
| stationarity/ | ADF, KPSS, PP |

## Key Modules

| File | Lines | Purpose |
|------|-------|---------|
| framework.py | 935 | `Evaluator` class |
| validated_cv.py | ~200 | `ValidatedCrossValidation` |
| barrier_analysis.py | 1050 | `BarrierAnalysis` |
| binary_metrics.py | 910 | Classification metrics |
| trade_analysis.py | 1078 | Trade-level analysis |
| autocorrelation.py | 531 | ACF/PACF |

## Key Classes

`Evaluator`, `ValidatedCrossValidation`, `BarrierAnalysis`, `FeatureDiagnostics`
