# visualization/factor/ - Factor Plots

Plotly figures for factor exposure, attribution, risk, and model diagnostics.

## Plot Families

- Exposure: `plot_factor_betas_bar`, `plot_rolling_betas`
- Attribution: `plot_return_attribution_waterfall`, `plot_return_attribution_area`
- Risk: `plot_risk_attribution_pie`, `plot_risk_attribution_bar`
- Diagnostics: `plot_residual_diagnostics`, `plot_factor_correlation_heatmap`, `plot_vif_bar`

## Usage Pattern

All functions take a factor-analysis result object plus optional layout arguments and return `go.Figure`.
