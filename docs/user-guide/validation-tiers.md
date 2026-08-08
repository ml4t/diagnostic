# Validation Stages

Apply diagnostics at four points in a research and production process. Each
stage answers a different question and produces inputs for the next stage.

## Feature stage

Test whether each candidate feature has suitable time-series and distribution
properties. Measure predictive value separately with information coefficient
analysis. Run feature selection inside each training fold to prevent leakage.

Relevant APIs:

- `FeatureDiagnostics`
- `compute_ic_hac_stats`
- `FeatureSelector`

Start with [feature diagnostics](feature-diagnostics.md) and
[feature selection](feature-selection.md).

## Signal and model stage

Measure cross-sectional IC, quantile spreads, monotonicity, and turnover on
out-of-sample predictions. Use purged validation when forward labels overlap.

Relevant APIs:

- `analyze_signal`
- `WalkForwardCV`
- `CombinatorialCV`

Start with the [quickstart](../getting-started/quickstart.md) and
[cross-validation guide](cross-validation.md).

## Backtest stage

Correct performance claims for the number and dependence of tested variants.
Inspect trade distributions and recurring losses. Preserve the complete trial
set so Deflated Sharpe Ratio and PBO receive the intended inputs.

Relevant APIs:

- `deflated_sharpe_ratio`
- `compute_pbo`
- `benjamini_hochberg_fdr`
- `TradeAnalysis`

Start with [statistical tests](statistical-tests.md) and
[trade analysis](trade-analysis.md).

## Portfolio stage

Monitor realized performance, drawdowns, tail risk, and factor exposure. Keep
the research correction inputs, data window, benchmark, and configuration with
each report.

Relevant APIs:

- `PortfolioAnalysis`
- `FactorAnalysis`
- `generate_backtest_tearsheet`

Start with [backtest tearsheets](backtest-tearsheets.md). The executable
[workflow](workflows.md) connects signal, cross-validation, strategy, and
portfolio checks.
