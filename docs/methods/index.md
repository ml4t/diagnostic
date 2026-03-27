# Statistical Methods

ml4t-diagnostic implements rigorous statistical methods from the academic literature
to address the specific challenges of evaluating ML-based trading strategies.
Each method page explains the problem it solves, the mathematics behind it,
and how to use it in practice.

Use this section when you want the statistical rationale and assumptions behind a
workflow. If you are looking for the fastest path to a working implementation,
start with the [User Guide](../user-guide/cross-validation.md) and return here
when you need method-level justification.

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

## See It In The Book

The book introduces these methods in the validation chapters and then reuses them
throughout the case studies:

- Ch06 for walk-forward validation and CPCV
- Ch07 for HAC-adjusted IC, DSR, and related significance testing
- Ch08-Ch09 for feature triage and robustness workflows

Use the [Book Guide](../book-guide/index.md) for the notebook and case-study map.

## Next Steps

- [Cross-Validation](../user-guide/cross-validation.md) -- apply CPCV and walk-forward validation in practice
- [Statistical Tests Guide](../user-guide/statistical-tests.md) -- see how these methods fit a broader testing workflow
- [Four-Tier Validation](../user-guide/validation-tiers.md) -- place each method inside the full validation stack
- [Academic References](../reference/references.md) -- review the underlying papers and citations
