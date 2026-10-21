# evaluation/ - Core Analysis Workflows

This package holds the main research and validation workflows.

## Main Entry Points

- `ValidatedCrossValidation` for CPCV plus DSR
- `audit_lookahead` and `assert_causal` for model-agnostic feature causality checks
- `FeatureDiagnostics` for feature-quality checks
- `PortfolioAnalysis` for return and risk analytics
- `PortfolioAnalysis.from_frames` for owned supplied-account inputs, without reconstructing returns or fills
- `PortfolioAnalysis.compute_round_trips` for actual fractional FIFO segments, explicitly complete opening inventory and known-at open-lot marks; optional synthetic closes remain separate from actual realization
- `RoundTripResult.completed_trades`, `.to_trade_analysis` and `.win_probability` for closing-fill samples, existing trade summaries and explicit Beta(1,1) uncertainty with gross/net/missingness coverage
- `RoundTripResult.statistics_tables` for cash/count/duration/return/asset tables, explicit lot/NAV denominators and per-metric coverage; completed-cash summaries reuse `TradeStatistics.from_frame` without invented execution prices
- `TradeAnalysis` and `TradeShapAnalyzer` for trade-level diagnostics
- `FactorAnalysis` for factor exposure and attribution
- `Evaluator` for the broader evaluation framework

## Deeper Guides

| Directory | Purpose |
|-----------|---------|
| [stats/](stats/AGENTS.md) | DSR/PSR, correlation-adjusted K_eff, RAS, FDR, HAC, PBO, MinTRL |
| [../metrics/](../metrics/AGENTS.md) | IC, importance, interactions, feature-outcome analysis |
| [factor/](factor/AGENTS.md) | Returns-based factor modeling and attribution |
| [trade_shap/](trade_shap/AGENTS.md) | Trade-level SHAP pipeline components |

## Notes

- User-facing wrappers for Trade SHAP live in `trade_shap_diagnostics.py`.
- Signal analysis has its own facade in `src/ml4t/diagnostic/signal/`.
- The public docs for these workflows live under `docs/user-guide/` and `docs/methods/`.
- `evaluation.stats` exposes `effective_number_of_trials()` and
  `deflated_sharpe_ratio(..., correlation_method=..., min_k_eff=...)` for
  correlated strategy cohorts.
