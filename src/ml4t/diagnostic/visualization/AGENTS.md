# visualization/ - Plotly And Report Rendering

Visualization primitives and report assembly for signals, features, portfolio analytics, factor attribution, and backtest tearsheets.

## Main Areas

| Path | Purpose |
|------|---------|
| [backtest/](backtest/AGENTS.md) | HTML tearsheets, templates, statistical-validity and ML sections |
| [factor/](factor/AGENTS.md) | Factor exposure, attribution, and diagnostics plots |
| `cv_plots.py` | Fold and timeline visualization |
| `feature_plots.py`, `interaction_plots.py` | Feature importance and interaction charts |
| `signal/` and `portfolio/` | Signal and portfolio plotting packages |
| `report_generation.py` | Combined HTML and PDF export helpers |

## Notes

- `visualization/__init__.py` re-exports the public plotting surface.
- The newer reporting bridge lands in `visualization/backtest/` plus `integration/`.
