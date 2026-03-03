# Deflated Sharpe Ratio (DSR)

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

## ml4t-diagnostic API

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio
import numpy as np

# You tested 50 strategies and selected the best one
# First, compute the variance across ALL strategies tested
np.random.seed(42)
all_sharpe_ratios = np.random.normal(0.5, 0.8, 50)
best_sharpe = np.max(all_sharpe_ratios)
variance_trials = np.var(all_sharpe_ratios, ddof=1)

result = deflated_sharpe_ratio(
    observed_sharpe=best_sharpe,    # Best strategy's SR
    n_trials=50,                    # Total strategies tested
    variance_trials=variance_trials, # Var[{SR_1, ..., SR_50}]
    n_samples=252,                  # Trading days of data
    skewness=0.0,                   # Return skewness (Fisher)
    excess_kurtosis=0.0,            # Excess kurtosis (Fisher, normal=0)
    return_components=True
)

print(f"Observed Sharpe: {best_sharpe:.3f}")
print(f"DSR probability: {result['dsr']:.3f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Expected max under null: {result['expected_max_sharpe']:.3f}")
```

### Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `observed_sharpe` | Best strategy's annualized Sharpe ratio | Must be the selected maximum |
| `n_trials` | Total number of strategies tested | Include ALL, even failures |
| `variance_trials` | Var[{SR_1, ..., SR_K}] across all strategies | **Must be computed, not assumed** |
| `n_samples` | Number of return observations | T >= 50 minimum, >= 252 recommended |
| `skewness` | Fisher skewness of returns | 0.0 for normal assumption |
| `excess_kurtosis` | Fisher excess kurtosis (normal = 0) | Matches scipy/pandas convention |

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
