# ML4T Diagnostic - Agent Navigation

Statistical validation for ML-based quantitative trading.

## Navigation

| Need | Location |
|------|----------|
| Signal analysis | `src/ml4t/diagnostic/signal/` |
| Cross-validation | `src/ml4t/diagnostic/splitters/` |
| Statistical tests | `src/ml4t/diagnostic/evaluation/stats/` |
| Feature metrics | `src/ml4t/diagnostic/evaluation/metrics/` |
| Visualization | `src/ml4t/diagnostic/visualization/` |
| Configuration | `src/ml4t/diagnostic/config/` |
| Results | `src/ml4t/diagnostic/results/` |

## Key Exports

```python
from ml4t.diagnostic import analyze_signal, ValidatedCrossValidation, BarrierAnalysis
from ml4t.diagnostic.splitters import CombinatorialPurgedCV
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio, ras_sharpe_adjustment
```

## Package Details

See [src/ml4t/diagnostic/AGENT.md](src/ml4t/diagnostic/AGENT.md)
