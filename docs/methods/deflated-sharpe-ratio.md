# Deflated Sharpe Ratio (DSR)

Use DSR when you evaluated multiple strategy variants and need to know whether
the best Sharpe ratio still looks credible after accounting for selection bias.

## The Problem

You tested 50 parameter combinations for a momentum strategy and selected the one
with the highest Sharpe ratio of 1.5. Is this a genuine edge, or is it the
inevitable result of picking the maximum from 50 random draws?

Even if every strategy has **zero** expected return, the best one will show a
positive Sharpe ratio simply due to chance. The expected "spurious" Sharpe ratio
grows logarithmically with the number of strategies tested:

$$
E[\max\{SR\}] \approx \sqrt{\text{Var}[\{SR_k\}]} \cdot \sqrt{2 \log K}
$$

For K=50 strategies with typical variance, this spurious maximum is around
0.4-0.8 -- enough to look like a tradeable signal. This is **selection bias**,
and it is the most common source of backtest overfitting.

When the trial cohort is highly correlated, raw K can be too conservative.
`ml4t-diagnostic` therefore also supports correlation-adjusted
`K_eff` via `effective_rank`, `marchenko_pastur`, and `clustering`.

## The Solution

The Deflated Sharpe Ratio accounts for the number of strategies tested when
assessing whether the selected Sharpe ratio exceeds a search-adjusted null.

Instead of testing $H_0: SR = 0$, DSR tests $H_0: SR = E[\max\{SR\}]$ -- a
much harder threshold that accounts for the selection process.

## Mathematical Foundation

DSR extends the Probabilistic Sharpe Ratio (PSR) to multiple testing using
extreme value theory.

### Step 1: Expected Maximum Under Null

For K independent strategies, the expected maximum of K standard normals:

$$
E[\max\{Z_1, \ldots, Z_K\}] \approx (1-\gamma)\Phi^{-1}(1-1/K) + \gamma\Phi^{-1}(1-1/(Ke))
$$

where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.

This is an approximation, not the exact Normal-maximum mean. For fractional
`K_eff` between 1 and 2 it is unreliable and can become negative near one.
The library emits a `RuntimeWarning` in that range and constrains the estimate
to be nonnegative: the maximum of centered Normal variables cannot have a
negative mean. This constraint does not make the approximation exact. An
explicitly justified `min_k_eff=2.0` avoids that small-count range; the default
remains 1.0.

### Step 2: Expected Maximum Sharpe Ratio

Scale by the empirical standard deviation of Sharpe ratios across strategies:

$$
SR_0 = \sqrt{\text{Var}[\{SR_1, \ldots, SR_K\}]} \cdot E[\max\{Z\}]
$$

### Step 3: Standard Deviation of the Maximum Sharpe Ratio

Rescale the same cross-sectional standard deviation by the standard deviation
of the maximum of K independent standard normals, which the paper tabulates
for K = 1 to 100:

$$
\sigma[SR_0] = \sqrt{\text{Var}[\{SR_1, \ldots, SR_K\}]} \cdot \sqrt{V[\max\{Z_1, \ldots, Z_K\}]}
$$

The sample length and the skewness, kurtosis and autocorrelation of the
selected strategy do not appear in this expression. They affect DSR only
through the dispersion of the trial Sharpe ratios.

### Step 4: Deflated Test Statistic

$$
DSR = \Phi\left(\frac{\hat{SR} - SR_0}{\sigma[SR_0]}\right)
$$

The output is a Normal-CDF confidence score in [0, 1]. The reported upper-tail
`p_value` is one minus this score. Values above 0.95 reject the adjusted null
at the nominal 5% level under the model's assumptions.

This is a Normal approximation using the location and scale of the selected
maximum, not its exact distribution or a Bayesian probability of skill. The
scaling formula assumes independent, identically distributed Normal trial
Sharpe ratios. Estimated effective trial counts are a heuristic for a
correlated cohort; they do not establish the same null distribution. Severe non-Normality and small samples can distort the
false-positive rate; the paper's Exhibit 5 reports those limitations.

### Numerical Compatibility

The Equation 30 correction changes multi-trial probabilities and z-scores from
the earlier implementation, which combined the selected strategy's estimator
variance with the maximum's scaling factor. Single-trial probabilities retain
their previous definition. Public signatures and result fields are unchanged.
The scaling factor is now integrated rather than interpolated, and continues
to decrease beyond 100 trials instead of being capped at the value for 100.
The cross-sectional null and single-strategy PSR use different scales, so DSR
is not universally less than or equal to PSR. For example, a 60-observation
cohort with trial Sharpes 0.05, 0.15 and 0.25, Normal moments and no serial
correlation gives DSR 0.986 versus PSR 0.974 for the selected strategy. These
are model confidence scores, not measured false-positive coverage.

The paper's formula and its worked example are authoritative for this method.
The authors' linked code at revision
[`6199ecb`](https://github.com/zoonek/2025-sharpe-ratio/blob/6199ecb1c84da7eee7bd81d9a017cf12b8ff0e0d/functions.py)
uses a different estimator-based denominator; numerical parity with that code
is not claimed. The two definitions must not be mixed when comparing results.

### Single Strategy (PSR)

With one strategy there is no maximum over trials. The threshold is the
benchmark Sharpe ratio $SR_0$ (zero by default) and the denominator is the
standard deviation of the Sharpe ratio estimator at that benchmark, which
accounts for skewness $\gamma_3$, Pearson kurtosis $\gamma_4$ and first-order
autocorrelation $\rho$:

$$
V[\hat{SR}] = \frac{1}{T}\left(\frac{1+\rho}{1-\rho}
- \left(1 + \frac{\rho}{1-\rho} + \frac{\rho^2}{1-\rho^2}\right)\gamma_3 \cdot SR_0
+ \frac{1+\rho^2}{1-\rho^2} \cdot \frac{\gamma_4 - 1}{4} \cdot SR_0^2\right)
$$

The library also uses this denominator when every trial has the same Sharpe
ratio, because the trials then do not differ.

### Worked Example From the Paper

A two-year track record of monthly returns has a Sharpe ratio of 0.456
(0.036 / 0.079), skewness -2.448, Pearson kurtosis 10.164 and autocorrelation
0.2. As a single strategy its PSR against a zero benchmark is 0.966. If it is
the best of K = 10 trials whose Sharpe ratios have variance 0.1, the expected
maximum is 0.498, the standard deviation of the maximum is
$\sqrt{0.1} \cdot 0.587 = 0.186$, and the DSR is 0.410.
`deflated_sharpe_ratio_from_statistics()` returns these values.

## Minimal Working Example

```python
from ml4t.diagnostic.evaluation.stats import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics,
    effective_number_of_trials,
)
import numpy as np

# Recommended path: pass raw returns for each trial
np.random.seed(42)
trial_returns = [
    np.random.normal(0.0005, 0.01, 252),
    np.random.normal(0.0008, 0.01, 252),
    np.random.normal(0.0002, 0.012, 252),
]

k_eff = effective_number_of_trials(
    np.column_stack(trial_returns),
    method="effective_rank",
)

result = deflated_sharpe_ratio(
    trial_returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)

print(f"Adjusted confidence: {result.probability:.3f}")
print(f"P-value: {result.p_value:.3f}")
print(f"Expected max from noise: {result.expected_max_sharpe:.3f}")
print(f"Deflated Sharpe: {result.deflated_sharpe:.3f}")
print(f"K_eff: {result.n_trials_effective:.2f} (raw K={result.n_trials_raw})")

# Secondary path: use pre-computed statistics if your pipeline already has them
stats_result = deflated_sharpe_ratio_from_statistics(
    observed_sharpe=0.12,
    n_samples=252,
    n_trials=50,
    variance_trials=0.03,
    frequency="daily",
    effective_trials=8.5,
    correlation_method="effective_rank",
    min_k_eff=2.0,
)
```

`deflated_sharpe_ratio()` is the recommended entry point for most users because
it derives the required moments directly from raw returns. Use
`deflated_sharpe_ratio_from_statistics()` when your pipeline already computes the
Sharpe moments and trial variance upstream.

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `returns` | Single return series or sequence of trial return series | Pass multiple trials for DSR, one series for PSR |
| `frequency` | Return frequency | `"daily"` by default; affects annualized display values |
| `benchmark_sharpe` | Null-hypothesis Sharpe threshold | Leave at `0.0` unless you need a stricter hurdle |
| `n_trials` | Total strategies tested | Relevant for the statistics-based helper; include all trials |
| `correlation_method` | Correlation-aware `K_eff` estimator | Use `"effective_rank"` first; compare against `"marchenko_pastur"` or `"clustering"` when needed |
| `min_k_eff` | Conservative floor for correlation-adjusted `K_eff` | Leave at `1.0` by default; raise only when you want an explicit residual multiplicity penalty |
| `variance_trials` | Var[{SR_1, ..., SR_K}] across all strategies | Must be computed, not assumed, when using the statistics-based helper |
| `n_samples` | Number of return observations | T >= 50 minimum, >= 252 recommended |

### Correlation-Adjusted Trial Counts

When your candidate strategies are closely related, raw `K` can overstate the
true breadth of the search. The library supports three estimators for
correlation-adjusted `K_eff`:

| Estimator | What it measures | Notes |
|-----------|------------------|-------|
| `effective_rank` | entropy-weighted eigenvalue breadth | Recommended default; smooth and cheap |
| `marchenko_pastur` | non-noise eigenvalue count | Uses the iterative denoising variant |
| `clustering` | number of independent strategy families | Also swaps in cluster-level Sharpe variance |

The `deflated_sharpe_ratio()` result reports both `n_trials_raw` and
`n_trials_effective`. If `min_k_eff > 1`, `n_trials_effective` reflects the
post-floor value actually used in the DSR adjustment.

## Interpreting Results

| DSR Value | Interpretation | Action |
|-----------|---------------|--------|
| >= 0.95 | Strategy survives multiple testing at 5% level | Proceed to out-of-sample validation |
| 0.50 - 0.95 | Does not reject the adjusted null at 5% | Gather more data or reduce strategy space |
| < 0.50 | Observed Sharpe is below the adjusted null location | Investigate selection bias before deployment |

**Critical**: DSR is a necessary but not sufficient condition. A high DSR means the
strategy *might* be real. You still need out-of-sample testing, walk-forward validation,
and realistic transaction cost modeling.

## Common Pitfalls

1. **Fabricating `variance_trials`**

    Using `variance_trials=1.0` as a "reasonable default" defeats the purpose.
    You **must** compute the actual variance from all K strategies tested.
    If you don't have access to all K Sharpe ratios, DSR cannot be meaningfully calculated.

2. **Undercounting trials**

    Every parameter variation, feature combination, and lookback period counts as a trial.
    If you tested 10 signals x 5 lookbacks x 3 thresholds = 150 trials, not 10.

3. **Mixing return cadence and annualization**

    `periods_per_year` should reflect the cadence of the return series passed to
    DSR, not the rebalance cadence that generated those returns. Daily crypto
    return series usually need `periods_per_year=365`, not `252`.

4. **Misinterpreting the output**

    DSR = 0.42 does **not** mean "the strategy has 42% of its claimed Sharpe."
    It corresponds to an upper-tail p-value of 0.58 against the adjusted null.
    It does not give the posterior probability that the strategy has skill.

5. **Ignoring non-normality**

    Strategies with negative skewness and fat tails (common in trend-following)
    have higher Sharpe ratio estimation variance. `skewness`, `excess_kurtosis`
    and `autocorrelation` enter PSR and the minimum track record length
    directly. With more than one trial they do not change the DSR probability
    for a given `variance_trials`, so compute that variance from the actual
    trial returns.

6. **Using DSR without enough data**

    With T < 50 observations, the standard error of the Sharpe ratio estimator
    is so large that DSR becomes unreliable regardless of the observed value.

## See It In The Book

DSR appears in the validation chapters and in reporting workflows that compare
many model or parameter variants:

- Ch07 for the statistical foundations
- Ch16-Ch19 for backtest evaluation and reporting pipelines

Use the [Book Guide](../book-guide/index.md) for exact notebook and case-study paths.

## References

- **Lopez de Prado, M., Lipton, A., & Zoonekynd, V. (2025)**.
  "How to Use the Sharpe Ratio."
  *ADIA Lab Research Paper Series*, No. 19. The formulas and the worked example
  on this page follow the version dated 7 March 2026.
  Reference implementation: [github.com/zoonek/2025-sharpe-ratio](https://github.com/zoonek/2025-sharpe-ratio)

- **Bailey, D. H., & Lopez de Prado, M. (2014)**.
  "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting, and Non-Normality."
  *Journal of Portfolio Management*, 40(5), 94-107.

### Relationship to Other Methods

| Method | Advantages over DSR | Disadvantages |
|--------|-------------------|---------------|
| **RAS** | Handles correlated strategies, non-asymptotic bounds | Computationally expensive |
| **FDR** | Controls false discovery proportion, not just best strategy | No correlation handling |
| **PSR** | Simpler (single strategy) | No multiple testing correction |

## Next Steps

- [Statistical Tests](../user-guide/statistical-tests.md) - Place DSR alongside FDR, RAS, and related checks
- [Cross-Validation](../user-guide/cross-validation.md) - Pair DSR with leakage-safe model evaluation
- [Validation Tiers](../user-guide/validation-tiers.md) - See where DSR fits in the full validation framework
- [Book Guide](../book-guide/index.md) - Jump to the notebook and case-study usage
