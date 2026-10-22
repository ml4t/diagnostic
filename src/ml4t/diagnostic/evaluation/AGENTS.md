# evaluation/ - Core Analysis Workflows

This package holds the main research and validation workflows.

## Main Entry Points

- `ValidatedCrossValidation` for CPCV plus DSR
- `audit_lookahead` and `assert_causal` for model-agnostic feature causality checks
- `FeatureDiagnostics` for feature-quality checks
- `PortfolioAnalysis` for return and risk analytics
- `PortfolioAnalysis.compute_stress` and `compute_live_split` for owned dated regime samples, unchanged native summary definitions and exact return/account boundary membership
- `PortfolioAnalysis.compute_wealth_cones` and independent `portfolio_analysis.analyze_wealth_cones` for training-only IID/stationary forecasts, paired benchmark draws and actual live overlays; horizon steps count observations, not inferred future dates
- `PortfolioAnalysis.from_frames` for owned supplied-account inputs, without reconstructing returns or fills
- `portfolio_analysis.analyze_intraday_exposure` for complete observed marked holdings on their own clock, reusing shared holdings/occupancy calculations without return reconstruction or continuous-peak claims
- `PortfolioAnalysis.compute_intraday_estimate` and independent `portfolio_analysis.estimate_intraday_positions` for explicit daily transaction-notional estimates; `estimated_intraday` is not observed holdings or cash-price account replay, and unknown first-period scalars require no fabricated return series
- `RoundTripResult.reconstruct_intraday_exposure` for explicit cash-price replay from complete opening quantities/cash, original ordered fills and known-at supplied marks; same-instant states retain observation IDs, not inferred timestamps or a futures margin model
- `PortfolioAnalysis.compute_round_trips` for actual fractional FIFO segments, explicitly complete opening inventory and known-at open-lot marks; optional synthetic closes remain separate from actual realization
- `RoundTripResult.completed_trades`, `.to_trade_analysis` and `.win_probability` for closing-fill samples, existing trade summaries and explicit Beta(1,1) uncertainty with gross/net/missingness coverage
- `RoundTripResult.statistics_tables` for cash/count/duration/return/asset tables, explicit lot/NAV denominators and per-metric coverage; completed-cash summaries reuse `TradeStatistics.from_frame` without invented execution prices
- `RoundTripResult.profit_attribution` for signed cash-PnL shares by original asset and optional supplied sector, without inferred NAV or complete-case row dropping
- `PortfolioAnalysis.compute_holdings_attribution` for supplied beginning-period NAV
  weights/loadings; explicit intervals and known-at validation, not regression-derived betas
- `portfolio_analysis.benchmark_batting_average` for finite paired benchmark-relative wins/counts and an explicit neutral-market convention; not factor-monotonicity or a strategy's positive-return fraction
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
