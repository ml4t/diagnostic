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

For K=50 strategies with typical variance, this spurious maximum is around 0.4-0.8 --
enough to look like a tradeable signal. This is **selection bias**, and it is the
most common source of backtest overfitting.

## The Solution

The Deflated Sharpe Ratio adjusts an observed Sharpe ratio downward to account for
the number of strategies tested. It answers: *"What is the probability that the
true Sharpe ratio exceeds zero, given that we selected the best of K strategies?"*

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

### Step 2: Expected Maximum Sharpe Ratio

Scale by the empirical standard deviation of Sharpe ratios across strategies:

$$
SR_0 = \sqrt{\text{Var}[\{SR_1, \ldots, SR_K\}]} \cdot E[\max\{Z\}]
$$

### Step 3: Variance of Sharpe Ratio Estimator

Accounting for non-normality (skewness $\gamma_3$, Pearson kurtosis $\gamma_4$):

$$
V[\hat{SR}] = \frac{1}{T}\left(1 - \gamma_3 \cdot SR_0 + \frac{\gamma_4 - 1}{4} \cdot SR_0^2\right)
$$

### Step 4: Deflated Test Statistic

$$
DSR = \Phi\left(\frac{\hat{SR} - SR_0}{\sqrt{V[\hat{SR}]}}\right)
$$

The output is a probability in [0, 1]. Values above 0.95 indicate the strategy
survives multiple testing correction at the 5% level.

## Minimal Working Example

```python
from ml4t.diagnostic.evaluation.stats import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics,
)
import numpy as np

# Recommended path: pass raw returns for each trial
np.random.seed(42)
trial_returns = [
    np.random.normal(0.0005, 0.01, 252),
    np.random.normal(0.0008, 0.01, 252),
    np.random.normal(0.0002, 0.012, 252),
]

result = deflated_sharpe_ratio(trial_returns, frequency="daily")

print(f"Probability of skill: {result.probability:.3f}")
print(f"P-value: {result.p_value:.3f}")
print(f"Expected max from noise: {result.expected_max_sharpe:.3f}")
print(f"Deflated Sharpe: {result.deflated_sharpe:.3f}")

# Secondary path: use pre-computed statistics if your pipeline already has them
stats_result = deflated_sharpe_ratio_from_statistics(
    observed_sharpe=0.12,
    n_samples=252,
    n_trials=50,
    variance_trials=0.03,
    frequency="daily",
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
| `variance_trials` | Var[{SR_1, ..., SR_K}] across all strategies | Must be computed, not assumed, when using the statistics-based helper |
| `n_samples` | Number of return observations | T >= 50 minimum, >= 252 recommended |

## Interpreting Results

| DSR Value | Interpretation | Action |
|-----------|---------------|--------|
| >= 0.95 | Strategy survives multiple testing at 5% level | Proceed to out-of-sample validation |
| 0.50 - 0.95 | Inconclusive -- may be overfit | Gather more data or reduce strategy space |
| < 0.50 | Strategy is likely explained by selection bias | Reject -- do not deploy |

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

3. **Misinterpreting the output**

    DSR = 0.42 does **not** mean "the strategy has 42% of its claimed Sharpe."
    It means "there is 42% probability the true Sharpe exceeds zero after
    accounting for multiple testing."

4. **Ignoring non-normality**

    Strategies with negative skewness and fat tails (common in trend-following)
    have higher Sharpe ratio estimation variance. Always provide `skewness` and
    `excess_kurtosis` from your actual returns.

5. **Using DSR without enough data**

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
  "How to use the Sharpe Ratio: A multivariate case study."
  *ADIA Lab Research Paper Series*, No. 19.
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
