# ml4t.diagnostic - Package Index

## Stable Top-Level Workflows

- `analyze_signal` for factor-style signal validation
- `analyze_prepared_signal` for owned precomputed signal panels with supplied buckets
- `ValidatedCrossValidation` for CPCV plus DSR in one workflow
- `FeatureSelector` for IC, correlation, importance, and drift-based triage
- `FeatureDiagnostics`, `TradeAnalysis`, and `TradeShapAnalyzer` from `evaluation`
- `generate_tearsheet_from_result` and `generate_tearsheet_from_run_artifacts` from `integration`

## Subpackages

| Package | Purpose |
|---------|---------|
| [signal/](signal/AGENTS.md) | Factor signal analysis and IC/quantile workflows |
| [metrics/](metrics/AGENTS.md) | Feature, signal, and risk metric functions |
| [splitters/](splitters/AGENTS.md) | Walk-forward, CPCV, calendars, and fold persistence |
| [evaluation/](evaluation/AGENTS.md) | Core analysis workflows and statistical validation |
| `compat.pyfolio.txn`, `compat.pyfolio.utils`, `compat.pyfolio.round_trips`, `compat.pyfolio.plotting` | Thin pandas transaction/Zipline-tuple, round-trip extraction and five-table statistics adapters, plus optional Plotly transaction charts over native calculations/inputs; no legacy runtime dependency |
| `compat.pyfolio.perf_attrib` | Supplied dollar/fractional holdings attribution and component statistics over shared native kernels; uncovered assets retain the full account NAV denominator |
| `compat.pyfolio.capacity`, `compat.pyfolio.tears.create_capacity_tear_sheet` | Familiar fixed-capital liquidation, actual participation and hypothetical impact/report workflow; explicit intraday estimates reuse native kernels, not observed account facts |
| [selection/](selection/AGENTS.md) | Systematic feature selection pipeline |
| [integration/](integration/AGENTS.md) | `ml4t-backtest`, `ml4t-data`, and `ml4t-engineer` bridges |
| [visualization/](visualization/AGENTS.md) | Plotly charts, dashboards, and tearsheets |
| [results/](results/AGENTS.md) | Result schemas, summaries, and serialization helpers |
| [config/](config/AGENTS.md) | Pydantic configuration surface and presets |
| [reporting/](reporting/AGENTS.md) | Renderer abstraction for HTML, JSON, and Markdown reports |

## Notes

- Public API is defined by `__all__` exports in package modules.
- `compat.empyrical` and `compat.pyfolio` adapt legacy analytical signatures over
  native calculations. Pyfolio attribution figures and `tears.create_perf_attrib_tear_sheet`
  use optional Plotly/HTML presentation, not a mandatory Reloaded or Matplotlib runtime.
- Reporting and tearsheet entry points are in `integration/` and `visualization/backtest/`.
- Correlation-adjusted DSR now lives in `evaluation.stats` via
  `effective_number_of_trials()` plus `deflated_sharpe_ratio(..., correlation_method=...)`.
- For book and case-study workflows, set `periods_per_year` from the return
  series cadence, not from rebalance cadence.

## Migration Notes For Book Agents

- Import metric functions from `ml4t.diagnostic.metrics`, not `ml4t.diagnostic.evaluation.metrics`.
- Use `cross_sectional_ic_series()` for per-date IC and `cross_sectional_ic()` for its summary.
- Use `pooled_ic()` for a single pooled IC across all observations.
- Use `extract_signal_ic_series()` only inside signal-analysis workflows that need `(dates, values)`.
- Use `plot_ic_term_structure()` for horizon heatmaps and `plot_monthly_ic_heatmap()` for monthly signal IC heatmaps.
- Pydantic result schemas use explicit names such as `PortfolioMetricsResultSchema`,
  `PSRResultSchema`, `MinTRLResultSchema`, and `DSRResultSchema`.
- For correlated backtest cohorts, prefer `correlation_method="effective_rank"`
  first, persist `n_trials_raw` and `n_trials_effective`, and only raise
  `min_k_eff` above `1.0` when you want an explicit conservative floor.
