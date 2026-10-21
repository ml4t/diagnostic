# evaluation/ - Core Analysis Workflows

This package holds the main research and validation workflows.

## Main Entry Points

- `ValidatedCrossValidation` for CPCV plus DSR
- `audit_lookahead` and `assert_causal` for model-agnostic feature causality checks
- `FeatureDiagnostics` for feature-quality checks
- `PortfolioAnalysis` for return and risk analytics
- `PortfolioAnalysis.compute_holdings_attribution` for supplied beginning-period NAV
  weights/loadings; explicit intervals and known-at validation, not regression-derived betas
- `portfolio_analysis.benchmark_batting_average` for finite paired benchmark-relative wins/counts and an explicit neutral-market convention; not factor-monotonicity or a strategy's positive-return fraction
- `PortfolioAnalysis.from_frames` for owned supplied-account inputs, without reconstructing returns or fills
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
