# Backtest Tearsheets

Template-based backtest tearsheets are the main reporting surface for completed
strategy runs. The current implementation supports direct chart assembly, a
`BacktestResult` bridge, and a case-study artifact bridge.

!!! info "See it in the book"
    Ch16 `code/16_strategy_simulation/09_performance_reporting.py` introduces the
    reporting workflow, and the case studies reuse it through
    `code/case_studies/utils/backtest_tearsheets.py` and
    `code/case_studies/utils/backtest_loaders.py`.

## Choose The Entry Point

| Starting point | Recommended API |
|---|---|
| You already have normalized trades, returns, equity, and metrics | `generate_backtest_tearsheet()` |
| You have an `ml4t.backtest.BacktestResult` | `generate_tearsheet_from_result()` |
| You have a case-study artifact directory on disk | `generate_tearsheet_from_run_artifacts()` |
| You want to inspect analytics families before rendering | `analyze_backtest_result()` / `BacktestProfile` |

## 1. Direct Tearsheet Generation

Use the visualization entry point when your reporting surfaces are already in memory:

```python
from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

html = generate_backtest_tearsheet(
    trades=trades_df,
    returns=daily_returns,
    equity_curve=equity_df,
    metrics=metrics,
    template="hedge_fund",
    theme="default",
    output_path="report.html",
    n_trials=100,
)
```

You can also pass a precomputed `BacktestProfile`, SHAP results, factor data, or
structured report metadata to enrich the report.

## 2. Bridge From `BacktestResult`

This is the simplest path when you use `ml4t-backtest` directly:

```python
from ml4t.diagnostic.integration import (
    BacktestReportMetadata,
    generate_tearsheet_from_result,
)

html = generate_tearsheet_from_result(
    result=backtest_result,
    template="risk_manager",
    theme="print",
    report_metadata=BacktestReportMetadata(
        strategy_name="ETF Momentum",
        universe="US sector ETFs",
        evaluation_window="2018-01-01 to 2025-12-31",
        benchmark_name="SPY",
    ),
    output_path="etf_momentum_tearsheet.html",
)
```

The bridge normalizes the backtest result into a `BacktestProfile`, infers
default tearsheet metrics, and forwards the data into the visualization layer.

## 3. Bridge From Run Artifacts

Case studies and batch pipelines can render directly from saved run artifacts:

```python
from ml4t.diagnostic.integration import generate_tearsheet_from_run_artifacts

html = generate_tearsheet_from_run_artifacts(
    backtest_dir="artifacts/backtests/etf_momentum",
    predictions_path="artifacts/backtests/etf_momentum/predictions.parquet",
    signals_path="artifacts/backtests/etf_momentum/weights.parquet",
    template="full",
    theme="default",
    output_path="artifacts/backtests/etf_momentum/report.html",
)
```

This path is the best fit for the third-edition case studies because it works
with persisted backtest, prediction, and signal outputs instead of requiring the
live Python objects to still be in memory.

## `BacktestProfile`

`BacktestProfile` is the compositional analytics object behind the new tearsheet
pipeline. It exposes lazy properties so the report can degrade gracefully when
some raw surfaces are missing.

### What It Exposes

| Property | What it summarizes |
|---|---|
| `performance` | Returns, Sharpe-family metrics, stability, confidence intervals |
| `edge` | Trade lifecycle and trade-quality metrics |
| `activity` | Fill and rebalance activity |
| `occupancy` | Portfolio-state exposure and occupancy |
| `attribution` | Symbol contribution and burden metrics |
| `drawdown` | Drawdown episodes and peak-to-trough anatomy |
| `ml` | Prediction/signal surface availability and translation metrics |
| `availability` | Explicit availability and fallback metadata |
| `summary` | Flat metric summary for compatibility layers |

### Inspect Before Rendering

```python
from ml4t.diagnostic.integration import analyze_backtest_result

profile = analyze_backtest_result(
    result=backtest_result,
    confidence_intervals=True,
)

print(profile.performance["metrics"]["sharpe_ratio"])
print(profile.ml["available"])
print(profile.availability.families["ml"].status)
```

## Structured Report Metadata

`BacktestReportMetadata` lets you control visible titles and contextual labels
without manually stitching strings into every template call:

```python
from ml4t.diagnostic.integration import BacktestReportMetadata

metadata = BacktestReportMetadata(
    title="ETF Momentum Backtest",
    subtitle="Monthly rebalance, vol-targeted",
    strategy_name="ETF Momentum",
    strategy_id="etf-mom-v3",
    universe="US sector ETFs",
    benchmark_name="SPY",
    evaluation_window="2018-01-01 to 2025-12-31",
    run_id="run-20260326-001",
)
```

## Presets

The current tearsheet architecture uses shared dashboard workspaces with preset
ordering and hero metrics instead of maintaining separate rendering stacks.

| Preset | Best for | Workspace order |
|---|---|---|
| `quant_trader` | Trade-level diagnosis and execution detail | overview -> trading -> performance -> validation -> ML -> factors |
| `hedge_fund` | Performance, costs, drawdowns, and attribution | overview -> performance -> trading -> validation -> factors -> ML |
| `risk_manager` | Credibility and risk review | overview -> validation -> performance -> trading -> factors -> ML |
| `full` | Comprehensive presentation | overview -> performance -> trading -> validation -> factors -> ML |

## What The Current Tearsheet Adds

The reporting work committed in the latest tearsheet pass added or strengthened:

- an explicit `BacktestProfile` bridge for lazy analytics families
- structured `BacktestReportMetadata`
- six-tab layout: Overview, Performance, Trading, Validation, ML (conditional), Factors (conditional)
- credibility-oriented reporting with DSR, PBO, MinTRL, and confidence intervals
- ML tab with IC time series, decile returns, prediction-trade alignment, and signal utilization
- Trading tab with rebalance event timeline, execution quality (implementation shortfall), and cost sensitivity
- factor tab with exposure bars, regression statistics, attribution waterfall, and risk decomposition
- `load_fama_french_5factor()` for one-line factor data loading (`pip install ml4t-diagnostic[factors]`)
- artifact-driven tearsheet generation for case-study workflows

## Related APIs

- `ml4t.diagnostic.integration.generate_tearsheet_from_result`
- `ml4t.diagnostic.integration.generate_tearsheet_from_run_artifacts`
- `ml4t.diagnostic.integration.analyze_backtest_result`
- `ml4t.diagnostic.visualization.backtest.generate_backtest_tearsheet`
- `ml4t.diagnostic.visualization.backtest.BacktestTearsheet`
