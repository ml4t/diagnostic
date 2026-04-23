# evaluation/ - Core Analysis Workflows

This package holds the main research and validation workflows.

## Main Entry Points

- `ValidatedCrossValidation` for CPCV plus DSR
- `FeatureDiagnostics` for feature-quality checks
- `PortfolioAnalysis` for return and risk analytics
- `TradeAnalysis` and `TradeShapAnalyzer` for trade-level diagnostics
- `FactorAnalysis` for factor exposure and attribution
- `Evaluator` for the broader evaluation framework

## Deeper Guides

| Directory | Purpose |
|-----------|---------|
| [stats/](stats/AGENTS.md) | DSR, RAS, FDR, HAC, PBO, MinTRL |
| [../metrics/](../metrics/AGENTS.md) | IC, importance, interactions, feature-outcome analysis |
| [factor/](factor/AGENTS.md) | Returns-based factor modeling and attribution |
| [trade_shap/](trade_shap/AGENTS.md) | Trade-level SHAP pipeline components |

## Notes

- User-facing wrappers for Trade SHAP live in `trade_shap_diagnostics.py`.
- Signal analysis has its own facade in `src/ml4t/diagnostic/signal/`.
- The public docs for these workflows live under `docs/user-guide/` and `docs/methods/`.
