# API Reference

This reference is organized by import surface rather than by source tree alone.

## Recommended Imports

| Use case | Import surface |
|---|---|
| Stable application code | `ml4t.diagnostic.api` |
| Notebook and exploratory work | `ml4t.diagnostic` |
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
| Metrics | `compute_ic_series`, `compute_ic_hac_stats`, `compute_mdi_importance`, `compute_permutation_importance`, `compute_shap_importance`, `compute_h_statistic`, `compute_shap_interactions`, `analyze_ml_importance`, `analyze_interactions` |

## Package-Level Convenience API (`ml4t.diagnostic`)

The package root re-exports the most common classes and configs for interactive use:

| Category | Objects |
|---|---|
| Core workflows | `ValidatedCrossValidation`, `FeatureSelector`, `analyze_signal`, `BarrierAnalysis` |
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
        - SignalResult
        - prepare_data
        - compute_ic_series
        - compute_ic_summary
        - compute_quantile_returns
        - compute_spread
        - compute_turnover
        - estimate_half_life

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

## Visualization

The visualization namespace is Plotly-first and grouped by workflow:

| Area | Representative functions |
|---|---|
| Cross-validation | `plot_cv_folds` |
| Signal analysis | `plot_ic_ts`, `plot_quantile_returns_bar`, `SignalDashboard`, `MultiSignalDashboard` |
| Portfolio analysis | `create_portfolio_dashboard`, `plot_portfolio_cumulative_returns`, `plot_monthly_returns_heatmap`, `plot_drawdown_underwater`, `plot_rolling_sharpe` |
| Factor analysis | `plot_factor_betas_bar`, `plot_rolling_betas`, `plot_return_attribution_waterfall` |
| Reporting | `combine_figures_to_html`, `generate_combined_report`, `export_figures_to_pdf` |

For a package-layout overview, see the [Architecture](../reference/architecture.md) page.
