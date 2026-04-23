# Architecture

This page describes the current `ml4t.diagnostic` package structure and the
intended reader journey through the library.

## Position In The ML4T Stack

`ml4t.diagnostic` sits between feature engineering and execution:

```text
ml4t.data -> ml4t.engineer -> ml4t.diagnostic -> ml4t.backtest -> ml4t.live
```

Its role is to answer four questions:

1. Are the features or signals statistically credible?
2. Is the validation scheme leak-free?
3. Is the backtest performance robust after multiple-testing correction?
4. Can the results be turned into reports, dashboards, and auditable analysis?

## Public Surfaces

The package exposes six main surfaces:

| Surface | Purpose | Typical entry points |
|---|---|---|
| `ml4t.diagnostic` | Convenience imports for common workflows | `ValidatedCrossValidation`, `analyze_signal`, `FeatureSelector` |
| `ml4t.diagnostic.api` | Stable integration-safe API | `ValidatedCrossValidation`, `FeatureDiagnostics`, `PortfolioAnalysis` |
| `ml4t.diagnostic.metrics` | Reusable metric and feature-statistic functions | `cross_sectional_ic_series`, `pooled_ic`, `sharpe_ratio`, `analyze_ml_importance` |
| `ml4t.diagnostic.evaluation` | Research and analysis workflows | `FeatureDiagnostics`, `TradeAnalysis`, `PortfolioAnalysis`, factor attribution |
| `ml4t.diagnostic.splitters` | Time-series CV and fold persistence | `WalkForwardCV`, `CombinatorialCV`, `save_folds()` |
| `ml4t.diagnostic.visualization` | Plotly figures and report assembly | `plot_cv_folds()`, signal/portfolio/factor plots |

For production code that values import stability over convenience, prefer
`ml4t.diagnostic.api`.

## Package Layout

| Package | Responsibility |
|---|---|
| `config/` | Pydantic configuration schemas for diagnostics, signals, trade analysis, reporting, and validated CV |
| `signal/` | Factor and alpha evaluation: IC, quantiles, spreads, turnover, half-life |
| `splitters/` | Walk-forward CV, CPCV, calendar-aware splitting, fold persistence |
| `evaluation/` | Core analytical workflows: diagnostics, portfolio analysis, trade analysis, stats, factor attribution |
| `visualization/` | Plotly figures, dashboards, HTML/PDF report assembly |
| `integration/` | Contracts and bridges for `ml4t.data`, `ml4t.engineer`, and `ml4t.backtest` |
| `results/` | Typed result containers and comparison schemas |
| `selection/` | Systematic feature selection workflows |
| `validation/` | Input validation helpers and guardrails |
| `caching/`, `backends/`, `utils/` | Performance, adapters, and shared utilities |

## Workflow Architecture

The library is organized around reusable building blocks rather than one giant
orchestrator.

### 1. Signal And Feature Diagnostics

These workflows help decide whether a signal or feature is worth modeling:

- `analyze_signal()` computes forward returns, IC summaries, quantile behavior,
  and turnover from factor and price data.
- `FeatureDiagnostics.run_diagnostics()` runs stationarity, autocorrelation,
  volatility, and distribution checks on a single feature series.
- `analyze_ml_importance()` and `FeatureSelector` support multivariate triage
  once a model is available.

### 2. Validation And Statistical Inference

These workflows help avoid leakage and false discoveries:

- `WalkForwardCV` and `CombinatorialCV` provide time-aware fold generation.
- `ValidatedCrossValidation` combines CPCV with DSR in one workflow.
- `evaluation.stats` provides DSR, MinTRL, PBO, White's Reality Check, RAS,
  and FDR tools.

### 3. Backtest And Portfolio Analysis

These workflows turn return streams and backtest artifacts into risk-aware
diagnostics:

- `PortfolioAnalysis` computes performance, drawdown, rolling, and distribution
  metrics.
- `integration` bridges `ml4t.backtest.BacktestResult` into diagnostics and
  tearsheets.
- Factor attribution tools in `evaluation.factor` estimate exposures, rolling
  betas, return attribution, and risk decomposition.

### 4. Visualization And Reporting

The visualization layer is Plotly-first:

- `visualization.signal` provides signal and multi-signal dashboards.
- `visualization.portfolio` provides portfolio and tear-sheet components.
- `visualization.backtest` assembles HTML backtest tearsheets.
- `visualization.factor` renders attribution, beta, and diagnostics plots.
- `visualization.trade_shap` and `evaluation.trade_shap_dashboard` provide the
  optional Streamlit trade-debugging flow.

## Recommended Import Strategy

Choose imports based on how stable and low-level you need them to be:

| Use case | Recommended import path |
|---|---|
| Application code | `ml4t.diagnostic.api` |
| Interactive analysis | `ml4t.diagnostic` or `ml4t.diagnostic.evaluation` |
| Standalone metrics and feature statistics | `ml4t.diagnostic.metrics` |
| Low-level statistical functions | `ml4t.diagnostic.evaluation.stats` |
| Cross-validation infrastructure | `ml4t.diagnostic.splitters` |
| Plotly figures and dashboards | `ml4t.diagnostic.visualization` |
| Backtest bridge code | `ml4t.diagnostic.integration` |

## Backtest Reporting Pipeline

The backtest reporting path is intentionally layered:

1. A backtest engine or case study produces trade, return, and metadata artifacts.
2. `ml4t.diagnostic.integration` normalizes those artifacts into a
   `BacktestProfile` or `PortfolioAnalysis`.
3. `visualization.backtest` turns those normalized objects into reusable sections
   and full tearsheets.
4. The book and case studies reuse the same bridge helpers rather than forking
   reporting logic in each notebook.

This is the basis for:

- `generate_tearsheet_from_result()`
- `generate_tearsheet_from_run_artifacts()`
- `portfolio_analysis_from_result()`

## Optional Dependencies

The package keeps optional dependencies separated by workflow:

| Extra | Adds |
|---|---|
| `viz` | Plotly, matplotlib, kaleido, PDF helpers |
| `ml` | LightGBM and XGBoost backends |
| `dashboard` | Streamlit-based trade dashboard support |
| `backtest` | `ml4t-backtest` bridge |
| `all` | All non-GPU optional extras |

Core analytical workflows remain importable without the visualization stack.

## Relationship To The Book

The book uses the same building blocks in two directions:

- chapter notebooks teach the method and then point to the production API
- case studies reuse the production API through shared helpers

Use the [Book Guide](../book-guide/index.md) for the chapter and case-study map.

## Design Principles

- Prefer typed result objects and schemas over loose dict protocols.
- Keep visualization optional so analytical imports stay lightweight.
- Preserve low-level building blocks for research notebooks.
- Provide higher-level convenience functions for case studies and application code.
- Favor calendar-aware and leakage-aware defaults for financial time series.
