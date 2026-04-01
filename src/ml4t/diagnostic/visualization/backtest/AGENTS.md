# visualization/backtest/ - Tearsheet And Reporting Surface

HTML tearsheet generation, statistical-validity visuals, trade plots, and workspace export for backtest analysis.

## Main Entry Points

- `generate_backtest_tearsheet`
- `BacktestTearsheet`
- `get_template`
- `export_workspaces`

## Core Files

- `tearsheet.py` - main orchestrator and object-oriented API
- `presets.py` - workspace order and template presets
- `template_system.py` - sections and template definitions
- `profile_sections.py` - profile-driven overview sections
- `dashboard_model.py` - normalized dashboard context
- `html_tables.py` - HTML snippets and summary tables
- `trade_plots.py`, `cost_attribution.py`, `statistical_validity.py`, `tail_risk.py`, `shap_patterns.py`, `ml_plots.py` - section visuals
- `export.py` and `interactive_controls.py` - bundled exports and navigation controls

## Notes

- `integration/backtest.py` is the bridge from `ml4t-backtest` and artifact directories into this package.
- Recent tearsheet work is concentrated here.
