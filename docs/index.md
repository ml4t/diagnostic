# ML4T Diagnostic

Statistical validation and diagnostics for ML-based quantitative trading strategies.

## Overview

ML4T Diagnostic is the **modern Alphalens + Pyfolio replacement** for the machine learning era. It provides rigorous validation tools implementing a Four-Tier Validation Framework to combat data leakage, backtest overfitting, and statistical fallacies.

## Key Features

- **Cross-Validation**: CPCV, Purged Walk-Forward with proper embargo/purging
- **Statistical Validity**: DSR, RAS, FDR corrections for multiple testing
- **Feature Analysis**: IC, importance (MDI/PFI/MDA/SHAP), interactions
- **Trade Diagnostics**: SHAP-based error pattern analysis
- **10-100x Faster**: Polars-first implementation

## Quick Example

```python
from ml4t.diagnostic import ValidatedCrossValidation

# One-step validated cross-validation with DSR
vcv = ValidatedCrossValidation(n_splits=10)
result = vcv.fit_validate(model, X, y, times)

if result.is_significant:
    print(f"Sharpe: {result.sharpe:.2f}, DSR p-value: {result.dsr_pvalue:.4f}")
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

- [Installation Guide](getting-started/installation.md) - Detailed setup instructions
- [Quickstart](getting-started/quickstart.md) - Get running in 5 minutes
- [API Reference](api/index.md) - Complete API documentation

## Part of the ML4T Library Suite

ML4T Diagnostic integrates seamlessly with other ML4T libraries:

```
ml4t-data → ml4t-engineer → ml4t-diagnostic → ml4t-backtest → ml4t-live
```
