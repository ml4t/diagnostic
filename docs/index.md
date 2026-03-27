# ML4T Diagnostic

Statistical validation and diagnostics for ML-based quantitative trading strategies.

## Overview

ML4T Diagnostic is the **modern Alphalens + Pyfolio replacement** for the machine learning era. It provides rigorous validation tools implementing a Four-Tier Validation Framework to combat data leakage, backtest overfitting, and statistical fallacies.

## Where It Fits in the ML4T Workflow

`ml4t.diagnostic` sits between feature engineering and backtesting. It is the
part of the stack where you check whether a signal, model, or backtest result is
actually robust before promoting it to portfolio construction or live trading.

```text
ml4t-data -> ml4t-engineer -> ml4t-diagnostic -> ml4t-backtest -> ml4t-live
```

## Who It's For

- Researchers validating alpha signals, feature sets, and model-selection decisions
- Strategy developers who need defensible backtest statistics, tearsheets, and attribution
- Readers of *Machine Learning for Trading, Third Edition* moving from notebooks to the production API

## Key Features

- **Cross-Validation**: CPCV, Purged Walk-Forward with proper embargo/purging
- **Statistical Validity**: DSR, RAS, FDR corrections for multiple testing
- **Feature Analysis**: IC, importance (MDI/PFI/MDA/SHAP), interactions
- **Trade Diagnostics**: SHAP-based error pattern analysis
- **10-100x Faster**: Polars-first implementation

## Quick Example

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.config import ValidatedCrossValidationConfig

# One-step validated cross-validation with DSR
config = ValidatedCrossValidationConfig(n_groups=10, n_test_groups=2)
vcv = ValidatedCrossValidation(config=config)
result = vcv.fit_evaluate(X, y, model, times=times)

if result.is_significant:
    print(f"Mean Sharpe: {result.mean_sharpe:.2f}, DSR probability: {result.dsr:.4f}")
```

## Four-Tier Validation Framework

| Tier | Stage | Focus |
|------|-------|-------|
| **1** | Pre-modeling | Feature importance, interactions, drift |
| **2** | During modeling | Predictions, calibration, stability |
| **3** | Post-modeling | Performance metrics, statistical validity |
| **4** | Production | Portfolio composition, risk, attribution |

## Statistical Tests

| Test | Purpose |
|------|---------|
| **DSR** | Deflated Sharpe Ratio - Multiple testing correction |
| **RAS** | Rademacher Anti-Serum - Backtest overfitting detection |
| **FDR** | Benjamini-Hochberg - p-value adjustment |
| **HAC** | Heteroskedastic & autocorrelation-consistent IC |

## Installation

```bash
pip install ml4t-diagnostic
```

## Next Steps

- [Getting Started](getting-started/quickstart.md) - Install the library and run your first validation workflow
- [User Guide](user-guide/cross-validation.md) - Learn the main research and reporting workflows
- [API Reference](api/index.md) - Browse the stable public import surface
- [Book Guide](book-guide/index.md) - Map chapters and case studies to the production API
- [Backtest Tearsheets](user-guide/backtest-tearsheets.md) - Start with the reporting bridge for `BacktestResult` and run artifacts

## See It In The Book

`ml4t.diagnostic` is used throughout *Machine Learning for Trading, Third Edition*:

- Ch06 for purged walk-forward CV and CPCV
- Ch07 for HAC-adjusted IC, FDR, DSR, and PBO
- Ch08-Ch09 for feature triage, robustness checks, and diagnostics
- Ch16-Ch19 for performance reporting, allocator analysis, factor attribution, and trade-SHAP
- Nine case studies under `third_edition/code/case_studies/`

Use the [Book Guide](book-guide/index.md) when you want the exact notebook and
case-study entry points.
