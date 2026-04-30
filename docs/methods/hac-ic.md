# HAC-adjusted Information Coefficient

Use this method when your IC series is measured across time and you need a
significance test that respects autocorrelation instead of assuming each period
is independent.

## The Problem

You computed the Information Coefficient (IC) of your alpha signal and got a mean
IC of 0.03 with a t-statistic of 2.5. Should you trust this result?

The standard t-test assumes IC observations are independent across time periods.
In practice, IC time series exhibit **autocorrelation**: a signal that works well
this week tends to work well next week too (and vice versa). This violates the
independence assumption, causing:

- **Underestimated standard errors**: the true uncertainty is higher than the
  naive estimate
- **Inflated t-statistics**: significance appears stronger than it really is
- **False positives**: signals that appear significant may not be

A naive IC t-stat of 2.5 might drop to 1.3 after proper HAC adjustment --
the difference between "statistically significant" and "noise."

## The Solution

HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors correct
for temporal dependence in IC time series. The Newey-West estimator computes
standard errors that account for both:

1. **Heteroskedasticity**: IC variance changes over time (volatile vs. calm markets)
2. **Autocorrelation**: IC values are correlated with recent IC values

The key insight: instead of dividing by $\sigma / \sqrt{T}$ (which assumes
independence), we compute a "long-run variance" that incorporates autocovariances.

## Mathematical Foundation

### Naive IC Standard Error

The standard (naive) approach treats IC observations as i.i.d.:

$$
SE_{\text{naive}} = \frac{s}{\sqrt{T}}, \quad t_{\text{naive}} = \frac{\bar{IC}}{SE_{\text{naive}}}
$$

where $s$ is the sample standard deviation and $T$ is the number of periods.

### Newey-West HAC Estimator

The HAC standard error incorporates autocovariances up to lag $L$:

$$
\hat{\Omega} = \hat{\Gamma}_0 + \sum_{j=1}^{L} w_j (\hat{\Gamma}_j + \hat{\Gamma}_j^{\top})
$$

where $\hat{\Gamma}_j = \frac{1}{T} \sum_{t=j+1}^{T} \hat{u}_t \hat{u}_{t-j}$ is the
sample autocovariance at lag $j$, and $w_j$ are kernel weights.

### Bartlett Kernel (Default)

The Newey-West estimator uses the Bartlett (triangular) kernel:

$$
w_j = 1 - \frac{|j|}{L + 1}
$$

This gives declining weight to higher-order autocovariances, with the property
that the resulting covariance matrix is always positive semi-definite.

### Automatic Lag Selection

The optimal lag length follows the Newey-West formula:

$$
L = \lfloor 4 \cdot (T / 100)^{2/9} \rfloor
$$

| Periods (T) | Automatic Lags (L) |
|-------------|-------------------|
| 52 (1 year weekly) | 3 |
| 252 (1 year daily) | 5 |
| 504 (2 years daily) | 6 |
| 1260 (5 years daily) | 8 |

### HAC-adjusted t-statistic

$$
SE_{\text{HAC}} = \sqrt{\hat{\Omega} / T}, \quad t_{\text{HAC}} = \frac{\bar{IC}}{SE_{\text{HAC}}}
$$

Typically $SE_{\text{HAC}} > SE_{\text{naive}}$, so $|t_{\text{HAC}}| < |t_{\text{naive}}|$.

## Minimal Working Example

### Primary Function: `compute_ic_hac_stats()`

```python
from ml4t.diagnostic.metrics import compute_ic_hac_stats

# ic_series: DataFrame with IC values per time period
# (e.g., from cross_sectional_ic_series())
stats = compute_ic_hac_stats(
    ic_series,           # Polars/Pandas DataFrame or numpy array
    ic_col="ic",         # Column name for IC values
    maxlags=None,        # None = automatic Newey-West formula
    kernel="bartlett",   # Kernel: "bartlett", "uniform", or "parzen"
    use_correction=True, # Small-sample correction
)

print(f"Mean IC:       {stats['mean_ic']:.4f}")
print(f"HAC SE:        {stats['hac_se']:.4f}")
print(f"HAC t-stat:    {stats['t_stat']:.2f}")
print(f"HAC p-value:   {stats['p_value']:.4f}")
print(f"Naive SE:      {stats['naive_se']:.4f}")
print(f"Naive t-stat:  {stats['naive_t_stat']:.2f}")
print(f"Effective lags: {stats['effective_lags']}")
```

### Bootstrap Alternative: `robust_ic()`

For non-parametric inference that makes no distributional assumptions:

```python
from ml4t.diagnostic.evaluation.stats import robust_ic

result = robust_ic(
    predictions,          # Model predictions
    returns,              # Forward returns
    n_samples=1000,       # Bootstrap iterations
    return_details=True,
)

print(f"IC:          {result['ic']:.4f}")
print(f"Bootstrap SE: {result['bootstrap_std']:.4f}")
print(f"t-stat:       {result['t_stat']:.2f}")
print(f"95% CI:      [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
```

### Return Fields

| Field | Description |
|-------|-------------|
| `mean_ic` | Mean IC across time series |
| `hac_se` | HAC-adjusted standard error |
| `t_stat` | t-statistic (mean_ic / hac_se) |
| `p_value` | Two-tailed p-value for H0: IC = 0 |
| `n_periods` | Number of time periods |
| `effective_lags` | Lag window used in HAC adjustment |
| `naive_se` | Standard (naive) standard error |
| `naive_t_stat` | Naive t-stat (for comparison) |

## Interpreting Results

### Significance Thresholds

| HAC t-stat | Interpretation | Confidence |
|-----------|---------------|------------|
| > 3.0 | Strongly significant | Very high |
| 2.0 - 3.0 | Significant at 5% level | High |
| 1.5 - 2.0 | Marginal -- treat with caution | Moderate |
| < 1.5 | Not significant after HAC adjustment | Low |

### Comparing Naive vs. HAC

The ratio of HAC SE to naive SE reveals the degree of autocorrelation:

| Ratio (HAC/Naive) | Autocorrelation | Impact |
|-------------------|----------------|--------|
| ~1.0 | Negligible | Naive and HAC agree |
| 1.5 - 2.0 | Moderate | Naive overstates significance |
| > 2.0 | Strong | Naive severely misleading |

**Rule of thumb**: if the HAC SE is more than 1.5x the naive SE, your signal
has meaningful autocorrelation and naive significance tests cannot be trusted.

### When to Use HAC vs. Bootstrap

| Method | Strengths | Weaknesses |
|--------|----------|------------|
| **HAC** (Newey-West) | Fast, parametric, well-understood | Assumes linear dependence structure |
| **Bootstrap** (stationary) | Non-parametric, handles nonlinear dependence | Slower (1000+ resamples), random |

**Recommendation**: Use HAC as the default. Switch to bootstrap when:

- IC distribution is highly non-normal (heavy tails, skewness)
- You need confidence intervals (bootstrap provides these directly)
- You suspect nonlinear temporal dependence

## Common Pitfalls

1. **Reporting only naive t-statistics**

    Many signal analysis reports show IC with naive standard errors, leading to
    overstated significance. Always report HAC-adjusted statistics for any
    signal with potential autocorrelation (which is nearly all financial signals).

2. **Using too few lags**

    Setting `maxlags=1` when your signal has weekly or monthly persistence
    underestimates the true standard error. Let the automatic formula choose,
    or err on the side of more lags (the Bartlett kernel downweights distant lags).

3. **Ignoring heteroskedasticity**

    IC tends to be higher during trending markets and lower during choppy
    markets. Even without autocorrelation, non-constant IC variance inflates
    naive t-statistics. HAC handles both problems simultaneously.

4. **Confusing statistical and economic significance**

    A HAC t-stat of 3.0 means the IC is statistically distinguishable from zero.
    It does not mean the signal is profitable after transaction costs. An IC of
    0.02 (t=3.0, T=2000) is statistically significant but may not cover costs.

5. **Computing IC on overlapping periods**

    If your forward return horizon is 5 days and you compute IC daily, the IC
    series has built-in autocorrelation from overlapping return windows. HAC
    partially addresses this, but it is better to compute IC on non-overlapping
    periods when possible.

## References

- **Newey, W. K., & West, K. D. (1987)**.
  "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703-708.

- **Politis, D. N., & Romano, J. P. (1994)**.
  "The Stationary Bootstrap."
  *Journal of the American Statistical Association*, 89(428), 1303-1313.

- **Patton, A., Politis, D. N., & White, H. (2009)**.
  "Correction to Automatic Block-Length Selection for the Dependent Bootstrap."
  *Econometric Reviews*, 28, 372-375.

### Related Functions in ml4t-diagnostic

| Function | Purpose |
|----------|---------|
| `cross_sectional_ic_series()` | Compute per-date cross-sectional IC from predictions and returns |
| `cross_sectional_ic()` | Reduce cross-sectional IC to summary statistics |
| `pooled_ic()` | Compute a pooled correlation across all supplied observations |
| `compute_ic_summary_stats()` | Naive IC statistics (no HAC) |
| `compute_ic_hac_stats()` | HAC-adjusted IC statistics |
| `compute_ic_uncertainty()` | Bundle of naive, HAC, and block-bootstrap CIs for a daily-IC series (recommended one-call entry point) |
| `cross_sectional_auc_series()` | Per-date cross-sectional AUC for binary labels (Polars-vectorized Mann-Whitney U) |
| `compute_auc_uncertainty()` | AUC analogue of `compute_ic_uncertainty()`, centred on `null_value=0.5` |
| `robust_ic()` | Stationary bootstrap IC with confidence intervals |
| `compute_ic_by_horizon()` | IC analysis across multiple forward return horizons |

### Recommended one-call entry point

For a daily-IC series pooled across CV folds, `compute_ic_uncertainty(daily_ic, horizon=H)`
returns mean IC together with three confidence intervals side-by-side: naive
(independence assumption), HAC (Newey-West, lag = `max(H-1, NW_auto)`), and
stationary block bootstrap (block length defaults to `max(H, n^{1/3})`).
Reporting all three lets the reader see how much of the SE inflation comes
from autocorrelation versus distributional shape.

## See It In The Book

HAC-adjusted IC is used in the signal-validation material and then reused in the
feature-triage case studies:

- Ch07 for robust IC significance testing
- `code/08_feature_engineering/*` and case-study evaluation notebooks for practical usage

Use the [Book Guide](../book-guide/index.md) for the chapter and notebook map.

## Next Steps

- [Feature Diagnostics](../user-guide/feature-diagnostics.md) - Use robust IC inside a wider diagnostics workflow
- [Feature Selection](../user-guide/feature-selection.md) - Turn IC evidence into a repeatable triage pipeline
- [Statistical Methods](index.md) - Compare HAC-adjusted IC to the other core validation methods
- [Book Guide](../book-guide/index.md) - Find the matching notebook and case-study implementations
