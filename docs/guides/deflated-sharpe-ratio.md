# Deflated Sharpe Ratio (DSR) Guide

The Deflated Sharpe Ratio corrects for multiple testing bias that arises when selecting the best strategy from many backtests.

## Formulations

### 2025 Formulation (Default)
López de Prado et al. (2025) returns Φ(z) where Φ is the standard normal CDF.
Result is a probability in [0, 1] representing confidence that the true Sharpe ratio is positive after accounting for multiple testing.

### 2014 Formulation (Legacy)
Bailey & López de Prado (2014) returns the z-statistic directly.
Result is in (-∞, +∞) representing standard deviations from expected max.

**Relationship**: DSR_2025 = Φ(DSR_2014)

## References

- López de Prado, M., Lipton, A., & Zoonekynd, V. (2025).
  "How to use the Sharpe Ratio: A multivariate case study."
  ADIA Lab Research Paper Series, No. 19.
  (Primary reference - enhanced formulation with probability output)

- Bailey, D. H., & López de Prado, M. (2014).
  "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting, and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.
  (Original formulation with z-score output)

## Implementation Notes

Implementation follows reference code: https://github.com/zoonek/2025-sharpe-ratio

The formula uses a quantile-based approximation for E[max]:
```
E[max{SR}] = sqrt(Var[{SR_k}]) * E[max{Z}]
where E[max{Z}] = (1-γ)*Φ⁻¹(1-1/K) + γ*Φ⁻¹(1-1/(K*e))
and γ ≈ 0.5772 is Euler-Mascheroni constant
```

### Why variance_trials is Required

DSR corrects for selection bias from picking the best of K strategies.
This requires knowing how variable those K strategies actually were.
If you only have one Sharpe ratio and no knowledge of the others,
you cannot calculate DSR meaningfully - any result would be based on
arbitrary assumptions about variance, making it useless.

**Data requirements**:
1. Test K independent strategies
2. Compute Sharpe ratio for each
3. Select the maximum
4. Compute the empirical variance across all K Sharpe ratios

## Parameter Guidance

### n_trials (Number of strategies tested)
- Effect: ↑ trials → ↑ E[max] → ↓ DSR (more deflation)
- Recommended: 1-1000 strategies
- Growth pattern: Logarithmic (E[max] ≈ √(2 log K))
- Critical: Include ALL strategies tested, not just "winners"
- Underestimating n_trials leads to inflated DSR (false confidence)

### variance_trials (Empirical variance across strategies)
- **Must be actual variance**: Var[{SR_1, SR_2, ..., SR_K}]
- Cannot be assumed or estimated without access to all K strategies
- Typical range: 0.1-5.0 (depends on strategy diversity)
- Higher variance = more diverse strategies = more deflation needed
- If variance_trials=0 (all strategies identical), E[max]=0 (no deflation)

### n_samples (Observations per strategy)
- Effect: ↑ samples → ↓ uncertainty → more reliable DSR
- Minimum: T ≥ 50 (below this, estimates unreliable)
- Recommended: T ≥ 252 (one year of daily data)
- Higher frequency: Minute/hourly data needs larger T to compensate for noise

### skewness and excess_kurtosis (Return distribution moments)
- Default: skewness=0.0, excess_kurtosis=0.0 (normal distribution)
- Uses Fisher convention: excess_kurtosis = kurtosis - 3 (matches scipy/pandas)
- Non-zero skewness/kurtosis → ↑ uncertainty → lower DSR
- Negative skewness slightly helps (crash risk already priced in)
- High kurtosis (fat tails) increases variance of Sharpe estimate
- Extreme values (|skew| > 5, excess_kurt > 22) still handled correctly

## Examples

### Basic Usage - Parameter Sweep Scenario

You tested 50 parameter combinations and selected the best:

```python
import numpy as np
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

# Simulate 50 strategies with Sharpe ratios under null (all ≈ 0)
np.random.seed(123)
sharpe_ratios = np.random.normal(0, 1/np.sqrt(252), 50)
best_sharpe = np.max(sharpe_ratios)  # 0.127 (selected by luck!)
variance_trials = np.var(sharpe_ratios, ddof=1)  # 0.0039

# Apply DSR correction
dsr = deflated_sharpe_ratio(
    observed_sharpe=best_sharpe,
    n_trials=50,
    variance_trials=variance_trials,
    n_samples=252,
    return_components=True
)
print(f"Observed Sharpe: {best_sharpe:.3f}")
print(f"DSR: {dsr['dsr']:.3f} (p={dsr['p_value']:.3f})")
print(f"Expected max under null: {dsr['expected_max_sharpe']:.3f}")
# Output:
# Observed Sharpe: 0.127
# DSR: 0.234 (p=0.766)
# Expected max under null: 0.125
# Conclusion: Not significant! The "best" strategy is just luck.
```

### Real-World Workflow - Trade Analysis Integration

```python
from ml4t.diagnostic.evaluation.trade_analysis import TradeAnalysis

# Analyze backtest trades
analyzer = TradeAnalysis(trades)
stats = analyzer.compute_statistics()

# Calculate Sharpe from returns
returns = np.array(daily_returns)
observed_sharpe = np.mean(returns) / np.std(returns, ddof=1)

# Apply DSR (assuming parameter sweep of 20 strategies)
skewness = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3))
# 4th moment gives Pearson kurtosis (normal=3), convert to Fisher (normal=0)
kurtosis_pearson = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4))
excess_kurtosis = kurtosis_pearson - 3.0

dsr_result = deflated_sharpe_ratio(
    observed_sharpe=observed_sharpe,
    n_trials=20,
    variance_trials=1.0,  # From empirical sweep
    n_samples=len(returns),
    skewness=skewness,
    excess_kurtosis=excess_kurtosis,
    return_components=True
)

# Decision threshold: Reject if DSR < 0.05 (p-value > 0.95)
is_significant = dsr_result['p_value'] < 0.05
if is_significant:
    print("Strategy survives multiple testing correction!")
else:
    print("Strategy likely overfit - reject")
```

### Multiple Timeframes Analysis

```python
timeframes = {
    "1day": 252,      # 1 year daily
    "1hour": 6 * 252, # 1 year hourly
    "1min": 390 * 252 # 1 year minute
}

for timeframe, n_samples in timeframes.items():
    # Higher frequency = lower Sharpe (noise, costs)
    observed_sharpe = 1.0 - np.log10(n_samples / 252) * 0.3

    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=10,
        variance_trials=1.0,
        n_samples=n_samples,
        return_components=True
    )

    print(f"{timeframe}: SR={observed_sharpe:.2f}, DSR={dsr_result['dsr']:.3f}")
# Output:
# 1day: SR=1.00, DSR=0.723
# 1hour: SR=0.69, DSR=0.612
# 1min: SR=0.37, DSR=0.489
# More data doesn't always help - noise can dominate
```

### Regime-Specific Analysis

```python
regimes = {
    "bull": (2.0, 126),     # High Sharpe, half year
    "sideways": (0.5, 126),
    "bear": (-1.0, 126)
}

for regime, (sharpe, n_samples) in regimes.items():
    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=5,  # 5 strategies tested per regime
        variance_trials=1.0,
        n_samples=n_samples,
        return_components=True
    )
    print(f"{regime}: DSR z-score = {dsr_result['dsr_zscore']:.2f}")
# Output:
# bull: DSR z-score = 1.85
# sideways: DSR z-score = -0.52
# bear: DSR z-score = -2.87
# Only bull regime survives deflation
```

## Common Pitfalls

1. **Confusing variance_trials with something else**
   - ❌ WRONG: Using 1.0 as default without actual variance
   - ✅ RIGHT: Computing Var[{SR_1, ..., SR_K}] from all K strategies

2. **Not accounting for all strategies tested**
   - ❌ WRONG: n_trials=10 when you tested 100 strategies (publication bias)
   - ✅ RIGHT: Count every variation, even those that "didn't work"

3. **Expecting DSR = PSR when n_trials=1**
   - ⚠️ PARTIAL: DSR reduces to PSR formula, but E[max]=0 (no selection bias)
   - ✅ RIGHT: DSR(K=1) ≠ PSR in value, but same statistical test

4. **Misinterpreting the output**
   - ❌ WRONG: "DSR=0.416 means strategy has 41.6% Sharpe"
   - ✅ RIGHT: "41.6% confidence that true SR > 0 after multiple testing"

5. **Using DSR without enough data**
   - ❌ WRONG: T=20 observations (standard error too high)
   - ✅ RIGHT: T ≥ 50 minimum, T ≥ 252 recommended

6. **Thinking DSR can "fix" a bad strategy**
   - ❌ WRONG: "Low DSR means I need to adjust my strategy"
   - ✅ RIGHT: "Low DSR means strategy is likely overfit - start over"

## Mathematical Background

DSR corrects the Probabilistic Sharpe Ratio (PSR) for selection bias using extreme value theory.

### Key Steps

1. **Expected Maximum under Null**: E[max{Z}] for K i.i.d. standard normals
   Uses quantile approximation (equation 26, López de Prado et al. 2025):
   ```
   E[max{Z}] ≈ (1-γ)Φ⁻¹(1-1/K) + γΦ⁻¹(1-1/(K·e))
   ```
   where γ ≈ 0.5772 is the Euler-Mascheroni constant.

2. **Expected Maximum Sharpe**: SR₀ = √Var[{SR_k}] · E[max{Z}]
   Scales the standard normal result by empirical strategy variance.

3. **PSR with Adjusted Target**: DSR = PSR(SR_hat | target=SR₀)
   Standard PSR formula but testing against SR₀ instead of 0.

4. **Variance Adjustment**: Accounts for skewness and kurtosis
   ```
   V[SR] = (1/T)(1 - γ₃·SR₀ + (γ₄-1)/4·SR₀²)
   ```
   where γ₄ is Pearson kurtosis = excess_kurtosis + 3

5. **Deflation**: Z = (SR_hat - SR₀) / √V[SR]
   Measures how many standard deviations observed SR exceeds expected maximum.

### Relationship to Other Methods

- DSR is conservative by design (0.5% actual vs 5% nominal Type I error)
- For exact Type I error control, use RAS or FDR methods instead
- DSR is fast (no simulation), RAS is accurate (with simulation)

### Variance Rescaling Analysis

The variance rescaling factors (Std[max{Z_k}]) from López de Prado et al. (2025)
Exhibit 3 are intermediate values used IN the DSR calculation, not outputs.

## See Also

- `probabilistic_sharpe_ratio()` - PSR without multiple testing correction
- `benjamini_hochberg_fdr()` - FDR correction for multiple comparisons
- `ras_ic_adjustment()` - Rademacher Anti-Serum for backtest overfitting
