# ml4t.diagnostic - Package Index

## Stable Top-Level Workflows

- `analyze_signal` for factor-style signal validation
- `ValidatedCrossValidation` for CPCV plus DSR in one workflow
- `FeatureSelector` for IC, correlation, importance, and drift-based triage
- `FeatureDiagnostics`, `TradeAnalysis`, and `TradeShapAnalyzer` from `evaluation`
- `generate_tearsheet_from_result` and `generate_tearsheet_from_run_artifacts` from `integration`

## Subpackages

| Package | Purpose |
|---------|---------|
| [signal/](signal/AGENTS.md) | Factor signal analysis and IC/quantile workflows |
| [splitters/](splitters/AGENTS.md) | Walk-forward, CPCV, calendars, and fold persistence |
| [evaluation/](evaluation/AGENTS.md) | Core analysis workflows and statistical validation |
| [selection/](selection/AGENTS.md) | Systematic feature selection pipeline |
| [integration/](integration/AGENTS.md) | `ml4t-backtest`, `ml4t-data`, and `ml4t-engineer` bridges |
| [visualization/](visualization/AGENTS.md) | Plotly charts, dashboards, and tearsheets |
| [results/](results/AGENTS.md) | Result schemas, summaries, and serialization helpers |
| [config/](config/AGENTS.md) | Pydantic configuration surface and presets |
| [reporting/](reporting/AGENTS.md) | Renderer abstraction for HTML, JSON, and Markdown reports |

## Notes

- Public API is defined by `__all__` exports in package modules.
- `get_agent_docs()` returns packaged `AGENTS.md` files for tool-assisted navigation.
- Recent reporting work is centered in `integration/` and `visualization/backtest/`.
