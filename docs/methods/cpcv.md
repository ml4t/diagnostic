# Combinatorial Purged Cross-Validation (CPCV)

## The Problem

You backtested a strategy on 5 years of daily data and got a Sharpe ratio of 1.8.
Is this a robust result, or did you overfit to a single historical path?

Standard backtesting gives you **one number** from **one path** through history.
You have no way to assess the variability of that result. Standard k-fold
cross-validation doesn't help either -- it assumes observations are independent,
but financial time series have:

- **Serial correlation**: adjacent returns are dependent
- **Overlapping labels**: forward-looking targets create information leakage

If your label is "5-day forward return," then a training sample at day 95 has a
label computed from prices on days 95-100. If day 98 is in the test set, training
on sample 95 leaks test information.

## The Solution

CPCV generates a **distribution** of backtest results instead of a single path.
It partitions the time series into N groups, then evaluates the strategy on all
$\binom{N}{k}$ ways to choose k groups as test sets. Each combination produces
an independent backtest path with proper train/test separation.

The key innovations over standard cross-validation:

1. **Purging**: removes training samples whose labels overlap with test data
2. **Embargo**: adds buffer zones after test periods to handle autocorrelation
3. **Combinatorial paths**: generates dozens to hundreds of evaluation paths

With a distribution of results, you can ask: *"What fraction of backtest paths
are profitable?"* If less than 50%, the strategy is likely overfit.

## Mathematical Foundation

### Partition and Combination

Given T observations, divide into N contiguous groups of approximately T/N samples
each. Choose k groups for testing, giving $\binom{N}{k}$ total combinations:

| Configuration | Combinations | Test Fraction |
|--------------|-------------|---------------|
| N=6, k=2 | 15 | 33% |
| N=8, k=2 | 28 | 25% |
| N=10, k=3 | 120 | 30% |
| N=12, k=4 | 495 | 33% |

### Purging

For test group spanning indices $[t_s, t_e]$ and label horizon $h$:

$$
\text{Purge: remove training samples where } t_{\text{train}} \in [t_s - h, t_s)
$$

This eliminates training samples whose forward-looking labels extend into the test period.

### Embargo

After each test group, exclude an additional buffer of $e$ samples from training:

$$
\text{Embargo: remove training samples where } t_{\text{train}} \in (t_e, t_e + e]
$$

This handles autocorrelation -- samples immediately after a test period may carry
correlated information from within the test window.

### Backtest Overfitting Probability

The probability of backtest overfitting (PBO) is estimated as:

$$
PBO = \frac{\text{\\# paths with negative OOS performance}}{\text{total \\# paths}}
$$

A PBO > 0.50 indicates the strategy is more likely overfit than genuine.

## ml4t-diagnostic API

```python
from ml4t.diagnostic.splitters import CombinatorialCV
import numpy as np

# Your time-series data
X = np.random.randn(2000, 10)  # 2000 samples, 10 features
y = np.random.randn(2000)       # Target (e.g., forward returns)

# Configure CPCV
cv = CombinatorialCV(
    n_groups=8,           # Split into 8 time groups
    n_test_groups=2,      # 2 groups for testing per combination → C(8,2) = 28 paths
    label_horizon=5,      # Labels look 5 samples forward (purging)
    embargo_size=2,       # 2-sample buffer after test groups
    max_combinations=20,  # Cap at 20 paths for efficiency
)

# Evaluate your strategy across all paths
scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train and evaluate your model
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    # scores.append(score)
    pass

# Analyze distribution of results
# pbo = np.mean(np.array(scores) < 0)
# print(f"PBO: {pbo:.1%}")  # > 50% → likely overfit
```

### Multi-Asset Support

For multi-asset strategies, CPCV handles each asset independently to prevent
cross-asset information leakage:

```python
import polars as pl

# Panel data with asset identifiers
df = pl.DataFrame({
    "date": dates,
    "symbol": symbols,
    "features": feature_values,
    "target": targets,
})

cv = CombinatorialCV(
    n_groups=8,
    n_test_groups=2,
    label_horizon=5,
    embargo_size=2,
)

# groups parameter enables per-asset purging
for train_idx, test_idx in cv.split(X, groups=df["symbol"]):
    # Each split purges correctly within each asset
    pass
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `n_groups` | Number of time partitions | 6-12 typical; more = more paths but smaller test sets |
| `n_test_groups` | Groups held out for testing per split | 2-4 typical; higher = larger test sets but fewer paths |
| `label_horizon` | Forward-looking label window size | Must match your target definition (e.g., 5 for 5-day returns) |
| `embargo_size` | Buffer after test groups | 1-5 typical; higher for strongly autocorrelated data |
| `max_combinations` | Cap on number of splits | Use when C(N,k) is very large (e.g., C(12,4) = 495) |

## Interpreting Results

### Probability of Backtest Overfitting (PBO)

| PBO Range | Interpretation | Action |
|-----------|---------------|--------|
| < 0.25 | Strong evidence of genuine strategy | Proceed to live testing |
| 0.25 - 0.50 | Some evidence, but uncertain | Increase data or simplify strategy |
| > 0.50 | More likely overfit than genuine | Reject -- do not deploy |

### Distribution Analysis

Beyond PBO, examine the full distribution of backtest scores:

- **Median performance**: more robust than mean (outlier-resistant)
- **Score variance**: high variance suggests fragile strategy
- **Worst path**: if worst path is catastrophic, strategy has hidden risks
- **Skewness**: negative skew means occasional large losses

## Common Pitfalls

1. **Ignoring label horizon**

    Setting `label_horizon=0` when your target is 5-day forward returns
    creates severe data leakage. The purging mechanism only works if you
    accurately specify how far forward your labels look.

2. **Too few groups**

    With `n_groups=4, n_test_groups=2`, you get only C(4,2) = 6 paths --
    far too few for reliable PBO estimation. Use at least N=8 for
    meaningful distributions.

3. **No embargo with intraday data**

    Intraday data has strong autocorrelation over short horizons. Even with
    purging, adjacent samples carry correlated microstructure information.
    Always use embargo_size >= 1 for intraday strategies.

4. **Confusing CPCV with standard k-fold**

    Standard k-fold doesn't purge or embargo. Using `sklearn.KFold` on
    financial time series produces inflated performance estimates.
    Always use CPCV or WalkForwardCV for temporal data.

5. **Treating PBO as a p-value**

    PBO = 0.30 does not mean "30% probability of overfitting."
    It means "30% of backtest paths showed negative performance."
    The interpretation depends on the strategy and market conditions.

## References

- **Lopez de Prado, M. (2018)**.
  "Advances in Financial Machine Learning." Wiley.
  Chapter 7: Cross-Validation in Finance.
  Chapter 12: Backtesting through Cross-Validation.

- **Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2017)**.
  "The Probability of Backtest Overfitting."
  *Journal of Computational Finance*, 20(4), 39-69.

### Comparison with WalkForwardCV

| Property | WalkForwardCV | CombinatorialCV |
|----------|--------------|-----------------|
| # of paths | N (sequential) | C(N,k) (combinatorial) |
| Uses all data for testing | No (expanding window) | Yes (every sample appears in test) |
| Detects overfitting | Limited | Yes (PBO) |
| Calendar-aware | Yes (trading sessions) | Yes (with calendar config) |
| Computational cost | Low | Higher (more paths) |
