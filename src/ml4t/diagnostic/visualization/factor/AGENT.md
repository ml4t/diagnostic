# Factor Visualization

9 Plotly plot functions for factor exposure, attribution, risk, and diagnostics.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| exposure_plots.py | ~160 | plot_factor_betas_bar, plot_rolling_betas |
| attribution_plots.py | ~140 | plot_return_attribution_waterfall, plot_return_attribution_area |
| risk_plots.py | ~100 | plot_risk_attribution_pie, plot_risk_attribution_bar |
| diagnostic_plots.py | ~200 | plot_residual_diagnostics, plot_factor_correlation_heatmap, plot_vif_bar |

## All Plot Functions

All follow pattern: `(result, *, theme=None, height=500, width=None) -> go.Figure`

- `plot_factor_betas_bar` — Horizontal bars + CI error bars
- `plot_rolling_betas` — Multi-line time series with optional R² subplot
- `plot_return_attribution_waterfall` — Waterfall chart (go.Waterfall)
- `plot_return_attribution_area` — Stacked area with total return line
- `plot_risk_attribution_pie` — Donut chart (hole=0.4) + idiosyncratic
- `plot_risk_attribution_bar` — MCTR bar chart
- `plot_residual_diagnostics` — 2x2: residual, QQ, ACF, histogram
- `plot_factor_correlation_heatmap` — Diverging RdBu heatmap
- `plot_vif_bar` — VIF bars with threshold lines at 5 and 10
