# ML4T Diagnostic - Agent Guide

Statistical validation and diagnostics for ML-based quantitative trading workflows.

## Start Here

- Package index: [src/ml4t/diagnostic/AGENTS.md](src/ml4t/diagnostic/AGENTS.md)
- Public docs landing page: [docs/index.md](docs/index.md)
- Book mapping: [docs/book-guide/index.md](docs/book-guide/index.md)

## Where To Work

| Need | Location |
|------|----------|
| Signal analysis and IC workflows | `src/ml4t/diagnostic/signal/` |
| Purged CV, held-out tests, fold persistence | `src/ml4t/diagnostic/splitters/` |
| Statistical tests and inference | `src/ml4t/diagnostic/evaluation/stats/` |
| Feature metrics and importance | `src/ml4t/diagnostic/evaluation/metrics/` |
| Trade SHAP diagnostics | `src/ml4t/diagnostic/evaluation/trade_shap/` |
| Factor attribution and exposures | `src/ml4t/diagnostic/evaluation/factor/` |
| Backtest bridges and tearsheet entry points | `src/ml4t/diagnostic/integration/` |
| Feature triage pipeline | `src/ml4t/diagnostic/selection/` |
| Plotly reporting and tearsheets | `src/ml4t/diagnostic/visualization/` |
| Result schemas and summaries | `src/ml4t/diagnostic/results/` |

## High-Value Entry Points

```python
from ml4t.diagnostic import ValidatedCrossValidation, analyze_signal
from ml4t.diagnostic.evaluation import FeatureDiagnostics, TradeShapAnalyzer
from ml4t.diagnostic.integration import generate_tearsheet_from_result
from ml4t.diagnostic.selection import FeatureSelector
from ml4t.diagnostic.splitters import CombinatorialCV, WalkForwardCV
```

## Notes

- `ml4t.diagnostic` sits between `ml4t-engineer` and `ml4t-backtest` in the ML4T stack.
- The docs and book guide are current and should be treated as part of the package surface.
- Module-level agent guides now use the canonical `AGENTS.md` filename throughout the repo.
