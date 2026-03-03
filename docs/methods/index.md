# Statistical Methods

ml4t-diagnostic implements rigorous statistical methods from the academic literature
to address the specific challenges of evaluating ML-based trading strategies.
Each method page explains the problem it solves, the mathematics behind it,
and how to use it in practice.

## Why These Methods Matter

Standard evaluation metrics (Sharpe ratio, accuracy, R-squared) are misleading
when applied naively to trading strategies. The core problems:

- **Multiple testing**: Testing many strategies inflates the best result
- **Temporal dependence**: Financial time series are autocorrelated
- **Information leakage**: Forward-looking labels contaminate train/test splits

The methods below address these problems with mathematical rigor.

## Method Overview

| Method | Problem Solved | Key Function | Reference |
|--------|---------------|--------------|-----------|
| [Deflated Sharpe Ratio](deflated-sharpe-ratio.md) | Selection bias from testing many strategies | `deflated_sharpe_ratio()` | Lopez de Prado et al. (2025) |
| [CPCV](cpcv.md) | Backtest overfitting detection | `CombinatorialCV` | Lopez de Prado (2018) |
| [HAC-adjusted IC](hac-ic.md) | Autocorrelation in IC significance testing | `compute_ic_hac_stats()` | Newey & West (1987) |

## Methods by Category

### Multiple Testing Corrections

| Method | When to Use | Computational Cost |
|--------|------------|-------------------|
| **DSR** | Quick assessment of best strategy | O(1) |
| **RAS** | Correlated strategies, rigorous bounds | O(n_sim x T x N) |
| **FDR** | Screening many p-values | O(N log N) |
| **Holm-Bonferroni** | Confirmatory analysis, no false positives | O(N log N) |

### Cross-Validation

| Method | When to Use | # Paths |
|--------|------------|---------|
| **WalkForwardCV** | Standard time-series validation | N folds |
| **CombinatorialCV** | Backtest overfitting detection | C(N,k) paths |

### Information Coefficient Analysis

| Method | When to Use | Handles Autocorrelation? |
|--------|------------|------------------------|
| **Naive IC** | Quick signal assessment | No |
| **HAC-adjusted IC** | Publication-grade significance | Yes (Newey-West) |
| **Bootstrap IC** | Non-parametric inference | Yes (stationary bootstrap) |

## Decision Flowchart

```
Is your Sharpe ratio "too good to be true"?
├── Yes → How many strategies did you test?
│   ├── Known → Deflated Sharpe Ratio (DSR)
│   ├── Many & correlated → Rademacher Anti-Serum (RAS)
│   └── Many & independent → False Discovery Rate (FDR)
└── No → Is your IC significant?
    ├── Check with HAC-adjusted IC
    └── Validate with CPCV backtest paths
```

## Further Reading

- [Academic References](../reference/references.md) -- full citation list with 10+ papers
- [Statistical Tests Guide](../user-guide/statistical-tests.md) -- stationarity, normality, ARCH tests
- [Four-Tier Validation](../user-guide/validation-tiers.md) -- how methods compose into validation tiers
