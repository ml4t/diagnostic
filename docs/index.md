# ML4T Diagnostic

Validate signals, models, and backtest results so you can tell whether strong
performance is robust or just an artifact of leakage, overfitting, or multiple testing.

`ml4t.diagnostic` is the validation layer in the ML4T stack. Use it after
feature engineering and before or alongside backtesting to answer practical
questions such as "is this Sharpe real?", "are these features actually
predictive?", and "what is driving my worst trades?" If you are new to the
library, start with the [Quickstart](getting-started/quickstart.md). If you are
coming from *Machine Learning for Trading, Third Edition*, use the
[Book Guide](book-guide/index.md) to jump from notebooks to the production API.

Chapters 6-9 develop validation techniques manually. This library implements the
same CPCV, DSR, HAC-adjusted IC, and feature triage workflows as tested,
reusable functions. Chapters 16-19 add reporting, attribution, and trade-level
SHAP. See the [Book Guide](book-guide/index.md) for the exact notebook-to-API map.

<div class="grid cards" markdown>

-   :material-shield-search:{ .lg .middle } __Is Your Sharpe Real?__

    ---

    Deflated Sharpe Ratio corrects for multiple testing.
    Check whether your best backtest survived selection bias.
    [:octicons-arrow-right-24: Statistical Tests](user-guide/statistical-tests.md)

-   :material-chart-sankey:{ .lg .middle } __Purged Cross-Validation__

    ---

    CPCV and purged walk-forward with embargo and label-horizon handling.
    Validate without leakage between train and test sets.
    [:octicons-arrow-right-24: Cross-Validation](user-guide/cross-validation.md)

-   :material-magnify-scan:{ .lg .middle } __Feature And Trade Diagnostics__

    ---

    HAC-adjusted IC, importance analysis, drift checks, and SHAP-based trade diagnostics.
    Find out what is actually predictive and what is failing.
    [:octicons-arrow-right-24: Feature Diagnostics](user-guide/feature-diagnostics.md)

-   :material-book-open-variant:{ .lg .middle } __From Book To API__

    ---

    The book develops these methods manually. The library packages them into
    reusable workflows for research and production reporting.
    [:octicons-arrow-right-24: Book Guide](book-guide/index.md)

</div>

## Quick Example

If you already have a model that looks good in backtest, the fastest way to
check whether it still looks credible after leakage-safe cross-validation and
multiple-testing correction is `ValidatedCrossValidation`.

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.config import ValidatedCrossValidationConfig

config = ValidatedCrossValidationConfig(n_groups=10, n_test_groups=2, label_horizon=5)
vcv = ValidatedCrossValidation(config=config)
result = vcv.fit_evaluate(X, y, model, times=times)

print(f"Mean Sharpe: {result.mean_sharpe:.2f}")
print(f"DSR probability: {result.dsr:.4f}")
print(f"Significant: {result.is_significant}")
```

## What You Can Validate Right Now

### Your Model Looks Good. Is It Overfit?

Use Deflated Sharpe Ratio when you tested many variants and need to know whether
the best result still looks real after selection bias.

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

result = deflated_sharpe_ratio([strategy_a, strategy_b, strategy_c], frequency="daily")
print(f"Probability of skill: {result.probability:.3f}")
print(f"Expected max from noise: {result.expected_max_sharpe:.3f}")
```

### Are Your Features Actually Predictive?

Use HAC-adjusted Information Coefficient statistics when naive IC t-stats are
too optimistic because the signal is autocorrelated across time.

```python
from ml4t.diagnostic.metrics import compute_ic_hac_stats

stats = compute_ic_hac_stats(ic_series, ic_col="ic")
print(f"Mean IC: {stats['mean_ic']:.4f}")
print(f"HAC t-stat: {stats['t_stat']:.2f}")
```

### Is Your Cross-Validation Leaking?

Use purged walk-forward or CPCV when forward labels and temporal dependence make
standard `KFold` results unreliable.

```python
from ml4t.diagnostic.splitters import WalkForwardCV

cv = WalkForwardCV(n_splits=5, train_size=252, test_size=63, label_horizon=5)
for train_idx, test_idx in cv.split(X):
    pass
```

### What Is Driving Your Worst Trades?

Use trade-level SHAP diagnostics when summary metrics are not enough and you
need to understand recurring failure modes in losing trades.

```python
from ml4t.diagnostic.evaluation import TradeAnalysis, TradeShapAnalyzer

worst_trades = TradeAnalysis(trade_records).worst_trades(n=20)
result = TradeShapAnalyzer(model, features_df, shap_values).explain_worst_trades(worst_trades)
print(result.error_patterns[0].hypothesis)
```

For full HTML reporting from normalized surfaces, `BacktestResult`, or saved run
artifacts, see [Backtest Tearsheets](user-guide/backtest-tearsheets.md).

## Four-Tier Validation Framework

This is the organizing structure behind the library. It keeps feature triage,
signal validation, backtest credibility, and portfolio analysis in one coherent path.

| Tier | Stage | Focus | Example Problem Caught |
|------|-------|-------|------------------------|
| **1** | Pre-modeling | Feature importance, interactions, drift | A feature looks predictive in-sample but is unstable across regimes |
| **2** | During modeling | Predictions, calibration, stability | A model ranks signals inconsistently or loses IC after HAC adjustment |
| **3** | Post-modeling | Performance metrics, statistical validity | A strong Sharpe disappears after CPCV or DSR multiple-testing correction |
| **4** | Production | Portfolio composition, risk, attribution | Returns are concentrated in one exposure bucket or one recurring trade error mode |

## Statistical Methods

These are the core methods the library uses to turn "looks good" into "survives scrutiny."

| Test | Purpose |
|------|---------|
| **DSR** | Deflated Sharpe Ratio for multiple-testing correction |
| **RAS** | Rademacher Anti-Serum for backtest overfitting detection |
| **FDR** | Benjamini-Hochberg adjustment for many simultaneous tests |
| **HAC** | Autocorrelation-robust IC significance testing |

## Installation

```bash
pip install ml4t-diagnostic
```

For SHAP workflows, Plotly reporting, and the `ml4t-backtest` bridge, see the
[Installation Guide](getting-started/installation.md) for optional extras.

## Where To Start

- [Quickstart](getting-started/quickstart.md) - first end-to-end validation workflow
- [Cross-Validation](user-guide/cross-validation.md) - leakage-safe splitter selection
- [Statistical Tests](user-guide/statistical-tests.md) - DSR, RAS, FDR, and robust significance
- [Backtest Tearsheets](user-guide/backtest-tearsheets.md) - reporting from results and artifacts
- [API Reference](api/index.md) - exact public import surfaces
- [Book Guide](book-guide/index.md) - chapter and case-study mapping

## See It In The Book

`ml4t.diagnostic` is used throughout *Machine Learning for Trading, Third Edition*:

- Ch06 for purged walk-forward CV and CPCV
- Ch07 for HAC-adjusted IC, FDR, DSR, and PBO
- Ch08-Ch09 for feature triage, robustness checks, and diagnostics
- Ch16-Ch19 for performance reporting, allocator analysis, factor attribution, and trade-SHAP
- Nine case studies under `third_edition/code/case_studies/`

Use the [Book Guide](book-guide/index.md) when you want the exact notebook and
case-study entry points.

## Part of the ML4T Library Suite

```text
ml4t-data -> ml4t-engineer -> ml4t-diagnostic -> ml4t-backtest -> ml4t-live
```

`ml4t.diagnostic` is the point in that workflow where you decide whether a
signal, model, or backtest result is credible enough to carry forward.
