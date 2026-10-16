# API Reference

This reference is organized by import surface rather than by source tree alone.

## Recommended Imports

| Use case | Import surface |
|---|---|
| Stable application code | `ml4t.diagnostic.api` |
| Notebook and exploratory work | `ml4t.diagnostic` |
| Metrics and feature statistics | `ml4t.diagnostic.metrics` |
| Statistical primitives | `ml4t.diagnostic.evaluation.stats` |
| Splitters and fold persistence | `ml4t.diagnostic.splitters` |
| Signal analysis | `ml4t.diagnostic.signal` |
| Backtest bridges | `ml4t.diagnostic.integration` |
| Plotly figures and dashboards | `ml4t.diagnostic.visualization` |

## Stable API (`ml4t.diagnostic.api`)

Use this module when you want imports that are less sensitive to future re-export
cleanup at the package root.

| Category | Objects |
|---|---|
| Validation workflows | `ValidatedCrossValidation`, `ValidatedCrossValidationConfig`, `validated_cross_val_score`, `ValidationResult`, `ValidationFoldResult` |
| Diagnostics | `FeatureDiagnostics`, `FeatureDiagnosticsResult`, `TradeAnalysis`, `PortfolioAnalysis`, `BarrierAnalysis` |
| Signal analysis | `analyze_signal`, `SignalResult` |
| Splitters | `CombinatorialCV`, `WalkForwardCV` |
| Metrics | `quantile_profile`, `QuantileProfile`, `cross_sectional_ic_series`, `cross_sectional_ic`, `pooled_ic`, `compute_ic_hac_stats`, `compute_mdi_importance`, `compute_permutation_importance`, `compute_shap_importance`, `compute_h_statistic`, `compute_shap_interactions`, `analyze_ml_importance`, `analyze_interactions` |

## Package-Level Convenience API (`ml4t.diagnostic`)

The package root re-exports the most common classes and configs for interactive use:

| Category | Objects |
|---|---|
| Core workflows | `ValidatedCrossValidation`, `FeatureSelector`, `analyze_signal`, `BarrierAnalysis` |
| Metrics | `quantile_profile`, `QuantileProfile` |
| Result types | `SignalResult`, data-quality schemas |
| Configuration | `DiagnosticConfig`, `StatisticalConfig`, `PortfolioConfig`, `TradeConfig`, `SignalConfig`, `EventConfig`, `BarrierConfig`, `ReportConfig`, `RuntimeConfig` |
| Optional visuals | selected barrier-analysis plot functions when viz dependencies are installed |

## Configuration

::: ml4t.diagnostic.config
    options:
      show_root_heading: true
      members:
        - DiagnosticConfig
        - StatisticalConfig
        - PortfolioConfig
        - TradeConfig
        - SignalConfig
        - EventConfig
        - BarrierConfig
        - ReportConfig
        - RuntimeConfig
        - ValidatedCrossValidationConfig

## Signal Analysis

::: ml4t.diagnostic.signal
    options:
      show_root_heading: true
      members:
        - analyze_signal
        - analyze_prepared_signal
        - prepare_signal_panel
        - PreparedSignalPanel
        - SignalResult
        - prepare_data
        - extract_signal_ic_series
        - compute_signal_ic
        - demean_signal_returns
        - compute_factor_weights
        - compute_factor_returns
        - compute_factor_cumulative_returns
        - compute_ic_summary
        - compute_quantile_returns
        - compute_quantile_statistics
        - compute_quantile_spread
        - compute_quantile_turnover
        - compute_rank_autocorrelation
        - compute_spread
        - compute_turnover
        - estimate_half_life

### IC report sample selection

`SignalResult.to_ic_result()` retains the common finite IC dates across its
requested horizons. Its statistics describe that retained sample; original
per-horizon `SignalResult.ic` and related maps stay unchanged. Use
`to_ic_result(period=1)` to retain one horizon's full finite IC sample. Empty
intersections have unavailable statistics; a singleton has a defined mean but
no sample inference. The IC time-series plot displays `CI unavailable` rather
than a confidence band when variance is undefined. Required numeric summary fields round-trip JSON nulls as
NaN without changing optional HAC/RAS fields. See the
[IC report how-to](../user-guide/migration.md#match-ic-report-statistics-to-selected-dates).

`plot_monthly_ic_heatmap` groups that component's selected observations by each
timestamp's supplied civil year/month, preserving dimensionless IC means and
missing cells. It labels observed months, keeps defined zero means visible and
marks an empty sample unavailable. It does not infer exchange sessions. See the
[monthly IC how-to](../user-guide/migration.md#inspect-monthly-ic-on-the-selected-sample).

::: ml4t.diagnostic.visualization.signal.plot_monthly_ic_heatmap

### Dated persistence result views

Persistence-enabled signal analyses retain independent dated observation tables
through `SignalResult.to_turnover_result()` and the tear-sheet bridge. The
structured result accepts paired dated observations separately from its legacy
summary-map representation. Views label observation lags explicitly; they do
not validate a calendar or infer uncertainty or half-life.

::: ml4t.diagnostic.results.signal_results.TurnoverAnalysisResult
    options:
      show_root_heading: true
      members:
        - from_observations
        - get_dataframe
        - list_available_dataframes
        - summary

For a dated component, the existing membership and rank plot functions select
an observation `lag` and render actual dated estimates, gaps, counts and
statuses. The turnover heatmap reduces its recorded bucket/lag observations;
historical horizon-summary components retain their existing paths. See the
[dated plotting how-to](../user-guide/migration.md#plot-dated-rank-and-membership-persistence).

::: ml4t.diagnostic.visualization.signal.plot_top_bottom_turnover

::: ml4t.diagnostic.visualization.signal.plot_autocorrelation

::: ml4t.diagnostic.visualization.signal.turnover_plots.plot_turnover_heatmap

### Empirical quantile report observations

Signal analyses retain actual dated bucket means in
`SignalResult.quantile_observations`. `to_quantile_result()` and the tear-sheet
bridge preserve an owned observation table through JSON. Historical summary-only
results cannot reconstruct that sample. Pooled bar summaries remain unchanged.
See the [empirical quantile how-to](../user-guide/migration.md#plot-empirical-dated-quantile-observations)
for adjustment, units, missing-data and report limits.

Use `get_monotonicity(period)` for an availability-aware decision and direction.
An undefined rank correlation yields `(None, "unavailable")`; the `spread`
DataFrame uses a nullable Boolean, and summaries/charts show `N/A`. Historical
raw boolean/direction maps keep their serialized shape. The existing native
minimum of three defined bucket means and finite decision threshold are
unchanged. See [unavailable monotonicity](../user-guide/migration.md#distinguish-unavailable-monotonicity-from-a-negative-decision).

::: ml4t.diagnostic.visualization.signal.plot_quantile_returns_violin
    options:
      show_root_heading: true

`plot_spread_timeseries` consumes those dated means through the shared quantile
spread/error calculation. Its bands describe independent propagated standard
error, not confidence coverage; smoothing windows count observations. Historical
cumulative-only inputs cannot establish that empirical sample. See the
[dated spread how-to](../user-guide/migration.md#plot-dated-spreads-and-propagated-errors).

::: ml4t.diagnostic.visualization.signal.plot_spread_timeseries
    options:
      show_root_heading: true

::: ml4t.diagnostic.results.signal_results.QuantileAnalysisResult
    options:
      show_root_heading: true
      members:
        - get_monotonicity
        - get_dataframe
        - list_available_dataframes

## Metrics

Use `ml4t.diagnostic.metrics` for reusable metric and feature-statistic helpers.

::: ml4t.diagnostic.metrics
    options:
      show_root_heading: true
      members:
        - quantile_profile
        - QuantileProfile
        - pooled_ic
        - cross_sectional_ic_series
        - cross_sectional_ic
        - compute_ic_summary_stats
        - compute_ic_hac_stats
        - compute_ic_decay
        - compute_conditional_ic
        - compute_monotonicity
        - compute_mdi_importance
        - compute_permutation_importance
        - compute_shap_importance
        - compute_mda_importance
        - analyze_ml_importance
        - compute_h_statistic
        - compute_shap_interactions
        - analyze_interactions
        - sharpe_ratio
        - sortino_ratio
        - maximum_drawdown
        - hit_rate
        - compute_forward_returns
        - brier_score
        - brier_decomposition
        - BrierDecomposition
        - log_loss
        - reliability_table
        - expected_calibration_error
        - max_calibration_error
        - calibration_slope
        - CalibrationSlope

## Cross-Validation

::: ml4t.diagnostic.splitters
    options:
      show_root_heading: true
      members:
        - BaseSplitter
        - CombinatorialCV
        - CombinatorialConfig
        - WalkForwardCV
        - WalkForwardConfig
        - SplitterConfig
        - save_config
        - load_config
        - save_folds
        - load_folds
        - verify_folds

## Evaluation Workflows

These workflows live under `ml4t.diagnostic.evaluation`:

| Area | Objects |
|---|---|
| Generic orchestration | `Evaluator`, `EvaluationResult`, `ValidatedCrossValidation` |
| Feature and signal diagnostics | `FeatureDiagnostics`, `MultiSignalAnalysis`, `analyze_ml_importance`, `compute_ic_hac_stats` |
| Portfolio and backtest evaluation | `PortfolioAnalysis`, factor attribution helpers |
| Trade diagnostics | `TradeAnalysis`, `TradeShapAnalyzer`, `TradeShapResult` |
| Event and barrier workflows | `EventStudyAnalysis`, `BarrierAnalysis` |

## Statistical Tests

::: ml4t.diagnostic.evaluation.stats
    options:
      show_root_heading: true
      members:
        - deflated_sharpe_ratio
        - deflated_sharpe_ratio_from_statistics
        - compute_min_trl
        - min_trl_fwer
        - compute_pbo
        - ras_ic_adjustment
        - ras_sharpe_adjustment
        - benjamini_hochberg_fdr
        - holm_bonferroni
        - multiple_testing_summary
        - hac_adjusted_ic
        - robust_ic
        - whites_reality_check

## Integration

The integration surface focuses on contracts and the `ml4t-backtest` bridge:

| Category | Objects |
|---|---|
| Contracts | `TradeRecord`, `DataQualityReport`, `DataQualityMetrics`, `DataAnomaly`, `BacktestReportMetadata` |
| Backtest bridge | `compute_metrics_from_result`, `analyze_backtest_result`, `portfolio_analysis_from_result` |
| Tearsheet generation | `generate_tearsheet_from_result`, `profile_from_run_artifacts`, `generate_tearsheet_from_run_artifacts` |

`analyze_backtest_result(result).occupancy` provides `timeline`, `metrics` and
`coverage`. Exposure fractions require finite positive equity and finite
exposures. Undefined ratios remain null; normalized aggregates use defined
snapshots only. Raw exposure amounts and position counts remain available.
Coverage names the total and normalized sample sizes; an empty observed window
has no estimates. See the [executed exposure workflow](../user-guide/migration.md#retain-raw-exposure-when-equity-cannot-normalize-it)
for the zero/negative-equity behavior and full performance-report restriction.

The profile's `activity` provides `metrics`, `turnover_timeline`,
`rebalance_summary` and `coverage`. Turnover/cost-drag ratios require finite
positive equity; raw execution totals remain available otherwise. The turnover
mean uses defined active observations only. Coverage discloses both all-observation
and active-observation sample sizes. The [executed transaction workflow](../user-guide/migration.md#retain-executed-turnover-when-equity-cannot-normalize-it)
describes the current timestamp/NAV denominator and remaining session/cost gaps.

## Visualization

The visualization namespace is Plotly-first and grouped by workflow:

| Area | Representative functions |
|---|---|
| Cross-validation | `plot_cv_folds` |
| Signal analysis | `plot_ic_ts`, `plot_quantile_returns_bar`, `SignalDashboard`, `MultiSignalDashboard` |
| Portfolio analysis | `create_portfolio_dashboard`, `plot_portfolio_cumulative_returns`, `plot_monthly_returns_heatmap`, `plot_drawdown_underwater`, `plot_rolling_sharpe` |
| Factor analysis | `plot_factor_betas_bar`, `plot_rolling_betas`, `plot_return_attribution_waterfall` |
| Reporting | `combine_figures_to_html`, `generate_combined_report`, `export_figures_to_pdf` |

For task-by-task input conversion from Alphalens and Pyfolio, see the
[migration guide](../user-guide/migration.md). For a package-layout overview,
see the [Architecture](../reference/architecture.md) page.
