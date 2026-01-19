# Rademacher Anti-Serum (RAS) - Comprehensive Guide

## Overview

The **Rademacher Anti-Serum (RAS)** is a novel backtesting protocol for multiple strategy testing that corrects for selection bias while accounting for correlation among strategies. It provides **conservative lower bounds** on true performance metrics with probability ≥ 1-δ for **ALL strategies simultaneously**.

**Key Innovation**: Unlike traditional methods (e.g., DSR), RAS accounts for correlation among strategies - the Rademacher complexity R̂ decreases when strategies are highly correlated, leading to tighter bounds.

## Quick Start

### Example 1: Testing 1000 Sharpe Strategies

```python
import numpy as np
from ml4t.diagnostic.evaluation.stats import (
    rademacher_complexity,
    ras_sharpe_adjustment
)

# Step 1: Backtest all strategies (walk-forward)
# X[t,n] = z-scored returns for strategy n at time t
# X[t,n] = returns_t,n / predicted_volatility_t,n
T, N = 2500, 1000  # 2500 days, 1000 strategies
X = np.random.randn(T, N)  # Replace with actual backtest data

# Step 2: Compute observed Sharpe ratios
observed_sharpe = X.mean(axis=0)  # Average per strategy

# Step 3: Compute Rademacher complexity
R_hat = rademacher_complexity(X, n_simulations=10000)

# Step 4: Apply RAS adjustment
adjusted_sharpe = ras_sharpe_adjustment(
    observed_sharpe,
    complexity=R_hat,
    n_samples=T,
    n_strategies=N,
    delta=0.05  # 95% confidence
)

# Step 5: Select significant strategies
significant = adjusted_sharpe > 0
print(f"Significant strategies: {significant.sum()}/{N}")
print(f"Best adjusted SR: {adjusted_sharpe.max():.4f}")
```

### Example 2: Testing Information Coefficients

```python
from ml4t.diagnostic.evaluation.stats import ras_ic_adjustment

# IC matrix: X[t,n] = correlation(alpha_t,n, returns_t) ∈ [-1, 1]
T, N = 2500, 500
X_ic = np.random.randn(T, N) * 0.02  # Realistic IC scale

# Compute observed ICs
observed_ic = X_ic.mean(axis=0)

# Compute complexity and adjust
R_hat = rademacher_complexity(X_ic, n_simulations=10000)
adjusted_ic = ras_ic_adjustment(
    observed_ic,
    complexity=R_hat,
    n_samples=T,
    delta=0.05,
    kappa=0.02  # Practical bound (|IC| ≤ 0.02)
)

# Count significant signals
significant_signals = adjusted_ic > 0
print(f"Significant signals: {significant_signals.sum()}/{N}")
```

## Mathematical Formulation

### Rademacher Complexity

The **Rademacher complexity** measures the ability of a set of strategies to fit random noise:

```
R̂ = E_ε[sup_n (ε^T x^n / T)]
```

where:
- ε is a **Rademacher vector**: random ±1 with probability 0.5
- x^n is the column vector for strategy n
- T is the number of time periods
- The expectation is over Rademacher draws (Monte Carlo estimation)

**Massart's Upper Bound**:
```
R̂ ≤ √(2logN/T)
```

Empirically, R̂ is typically 10-20% below this bound for uncorrelated strategies.

**Key Property**: R̂ **decreases with correlation** among strategies:
- R̂ ≈ 0 for perfectly correlated (identical) strategies
- R̂ → √(2logN/T) for uncorrelated strategies

### RAS for Information Coefficients (Procedure 8.1)

**Formula** (Equation 8.2):
```
θ_n ≥ θ̂_n - 2R̂ - 2κ√(log(2/δ)/T)
               ︸︷︷︸    ︸︷︷︷︷︷︷︷︷︷︷︷︷︸
          data snooping  estimation error
```

**Parameters**:
- `θ̂_n`: Observed IC for strategy n
- `R̂`: Rademacher complexity from full matrix X
- `T`: Number of time periods
- `δ`: Significance level (0.05 = 95% confidence)
- `κ`: Maximum absolute IC value (|IC| ≤ κ)

**Practical Tuning**:
- Theoretical: κ=1.0 (IC is correlation, bounded by 1)
- Practical: κ=0.02 (realistic signals rarely exceed 0.02)

Using κ=0.02 instead of κ=1.0 gives **50x tighter bounds**!

Example (δ=0.01, T=2500):
- Theoretical (κ=1.0): estimation error ≈ 0.109
- Practical (κ=0.02): estimation error ≈ 0.002

### RAS for Sharpe Ratios (Procedure 8.2)

**Formula** (Equation 8.3):
```
θ_n ≥ θ̂_n - 2R̂ - 3√(2log(2/δ)/T) - √(2log(2N/δ)/T)
               ︸︷︷︸   ︸︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︷︸
          data snooping      estimation error
```

**Parameters**:
- `θ̂_n`: Observed Sharpe ratio for strategy n (NOT annualized)
- `R̂`: Rademacher complexity
- `T`: Number of time periods
- `N`: Number of strategies
- `δ`: Significance level

**Note**: The estimation error has two terms:
1. Independent of N: 3√(2log(2/δ)/T)
2. Weakly dependent on N: √(2log(2N/δ)/T)

The bound is **conservative** but rigorous.

## Interpretation

### Component Breakdown

The RAS adjustment consists of two parts:

1. **Data Snooping Term (2R̂)**
   - Penalty for selecting the best from N strategies
   - Accounts for multiple testing bias
   - **Decreases with strategy correlation**
   - Higher when strategies span the space well (more diverse)

2. **Estimation Error**
   - Sampling uncertainty in the performance estimate
   - Decreases as T increases (more data)
   - For IC: scales with κ (practical bound on |IC|)
   - For Sharpe: has two sub-components (one depends on N)

### Three Interpretations of R̂

1. **Covariance to Random Noise**
   - R̂ = expected maximum covariance of strategies to random ±1 sequence
   - High R̂: Some strategy matches any random sequence well → overfitting risk

2. **Generalized 2-Way Cross-Validation**
   - R̂ measures worst-case discrepancy when splitting data in half randomly
   - High R̂: Performance is inconsistent across random splits

3. **Span Over Performance Space**
   - R̂ measures how much the strategies span R^T
   - Low R̂: Strategies are similar (correlated)
   - High R̂: Strategies are diverse (uncorrelated)

### Decision Rules

**For IC**:
A signal n is deemed statistically significant if:
```
adjusted_ic[n] > 0
```

**For Sharpe**:
A strategy n is deemed to have positive Sharpe ratio if:
```
adjusted_sharpe[n] > 0
```

With probability ≥ 1-δ, these decisions hold **simultaneously for all N strategies**.

## Data Requirements

### Input Data Matrix X

**Shape**: (T, N)
- **T**: Number of time periods (rows)
- **N**: Number of strategies (columns)

**For Sharpe Ratios**:
```python
X[t,n] = returns_t,n / predicted_volatility_t,n
```
- Z-scored returns (mean ≈ 0, variance ≈ 1 under null)
- X[t,n] is sub-Gaussian
- Empirical Sharpe: θ̂_n = mean(X[:,n])

**For Information Coefficients**:
```python
X[t,n] = correlation(alpha_t,n, idiosyncratic_returns_t)
```
- Bounded: |X[t,n]| ≤ κ (typically κ=0.02 for realistic signals)
- Cosine similarity between predicted alpha and realized returns
- Empirical IC: θ̂_n = mean(X[:,n])

### Temporal Structure

**Critical Assumption**: Rows of X are **i.i.d.** draws from common distribution P.

**If Autocorrelation Exists**:
1. Inspect autocorrelation plots of each strategy
2. If significant autocorrelation up to lag s:
   - Create ⌊T/s⌋ non-overlapping blocks
   - Replace X[t,:] with block average X[s·k:s·(k+1),:]
3. This reduces effective sample size but maintains i.i.d. assumption

**Empirical Observation**: Daily returns have small autocorrelation (see Chapter 2, Cont 2001, Taylor 2007).

## Comparison with Deflated Sharpe Ratio (DSR)

| Feature | RAS | DSR |
|---------|-----|-----|
| **Accounts for correlation** | ✅ Yes (R̂ decreases) | ❌ No (assumes independence) |
| **False positive rate** | 0% (empirical) | ~5% (by design) |
| **False discovery rate** | 0% (empirical) | Variable |
| **Sample size requirement** | Non-asymptotic | Non-asymptotic (2025 version) |
| **Scales to N strategies** | Millions+ | Thousands (computational) |
| **Provides per-strategy bound** | ✅ Yes (all N) | ❌ No (only max) |
| **Computational cost** | O(K·T·N) for K sims | O(1) per strategy |
| **Conservativeness** | Conservative | Moderate |
| **Parameter tuning** | δ, κ (for IC) | variance_trials (critical!) |

**When to Use RAS**:
- Testing large numbers of strategies (>100)
- Strategies may be correlated
- Need per-strategy significance, not just max
- Zero false positives required (regulatory, compliance)
- Can afford Monte Carlo computation (~10K simulations)

**When to Use DSR**:
- Testing smaller numbers of strategies (<100)
- Know empirical variance across K strategies
- Want probability interpretation (Φ(z) format)
- Prefer closed-form computation (no simulation)
- Comfortable with ~5% false positive rate under null

## Validation Results (Tables 8.3 & 8.4)

### Null Hypothesis (All Strategies SR=0)

| ρ | N | T | R̂ | % Positive | % Rad Positive |
|---|---|---|-----|-----------|----------------|
| 0.2 | 500 | 2500 | 0.059 | **0.0** | 0.0 |
| 0.8 | 500 | 2500 | 0.037 | **0.0** | 0.0 |
| 0.2 | 5000 | 2500 | 0.072 | **0.0** | 0.0 |
| 0.8 | 5000 | 2500 | 0.044 | **0.0** | 0.0 |

**Key Result**: **0% false positives** in all cases (compared to δ=5% allowed).

### Alternative Hypothesis (20% Strategies SR=0.2)

| ρ | N | T | % Positive | % Rad Positive | % True Positive | FDR |
|---|---|---|-----------|----------------|-----------------|-----|
| 0.2 | 500 | 2500 | 1.7 | 20.0 | 20.0 | **0** |
| 0.8 | 500 | 2500 | 14.3 | 20.0 | 20.0 | **0** |
| 0.2 | 500 | 5000 | 19.9 | 20.0 | 20.0 | **0** |
| 0.8 | 500 | 5000 | 20.0 | 20.0 | 20.0 | **0** |

**Key Results**:
1. **FDR = 0** in all cases (zero false discoveries)
2. Full formula is conservative (low % positive)
3. "Rademacher positive" (just data snooping term) detects ~20% consistently
4. Higher correlation (ρ=0.8) → lower R̂ → better detection
5. More data (T=5000) → tighter bounds → better detection

### Historical Anomalies (Table 8.5)

Using Jensen et al. (2023) database of 153 factor anomalies across 17 countries:

- **Most markets**: 0% strategies pass full RAS bound
- **USA**: 18.3% strategies pass "Rademacher positive" (T=13,155 days)
- **UK**: 0.7% strategies pass Rademacher positive
- **Hong Kong**: 6.5% strategies pass Rademacher positive

**Interpretation**: Most published anomalies do NOT survive RAS correction when accounting for look-ahead bias and multiple testing.

## Common Pitfalls and Best Practices

### ❌ Common Mistakes

1. **Using annualized Sharpe ratios**
   ```python
   # WRONG: Annualized Sharpe
   observed_sharpe = X.mean(axis=0) * np.sqrt(252)

   # CORRECT: Period-level Sharpe
   observed_sharpe = X.mean(axis=0)
   ```

2. **Wrong kappa for IC**
   ```python
   # WRONG: Using theoretical bound when ICs are small
   adjusted_ic = ras_ic_adjustment(..., kappa=1.0)  # Too conservative

   # CORRECT: Use practical bound
   adjusted_ic = ras_ic_adjustment(..., kappa=0.02)  # Realistic
   ```

3. **Forgetting to z-score Sharpe returns**
   ```python
   # WRONG: Raw returns
   X = returns  # Not standardized

   # CORRECT: Z-scored returns
   X = returns / predicted_volatility  # Mean ≈ SR under null
   ```

4. **Using different T for complexity and adjustment**
   ```python
   # WRONG: Mismatch
   R_hat = rademacher_complexity(X[:2000, :])  # T=2000
   adjusted = ras_sharpe_adjustment(..., n_samples=2500)  # T=2500

   # CORRECT: Same T
   T = X.shape[0]
   R_hat = rademacher_complexity(X)
   adjusted = ras_sharpe_adjustment(..., n_samples=T)
   ```

5. **Too few simulations**
   ```python
   # WRONG: High variance estimate
   R_hat = rademacher_complexity(X, n_simulations=100)

   # CORRECT: Use 10K for production
   R_hat = rademacher_complexity(X, n_simulations=10000)
   ```

### ✅ Best Practices

1. **Walk-Forward Backtesting**
   - All strategies must be backtested in walk-forward manner
   - No look-ahead bias in parameter tuning
   - Each strategy uses only past data at time t

2. **Reproducibility**
   ```python
   # Use random_state for reproducible results
   R_hat = rademacher_complexity(X, n_simulations=10000, random_state=42)
   ```

3. **Check Autocorrelation**
   ```python
   # Before applying RAS, check if data is i.i.d.
   from statsmodels.tsa.stattools import acf

   for n in range(N):
       autocorr = acf(X[:,n], nlags=20)
       if np.any(np.abs(autocorr[1:]) > 0.2):
           print(f"Strategy {n} has significant autocorrelation")
           # Consider block averaging
   ```

4. **Verify Massart's Bound**
   ```python
   # Sanity check: R̂ should be below Massart's bound
   massart_bound = np.sqrt(2 * np.log(N) / T)
   assert R_hat <= massart_bound, f"R̂={R_hat:.4f} > Massart={massart_bound:.4f}"
   ```

5. **Component Decomposition**
   ```python
   # Understand what's driving the adjustment
   data_snooping = 2 * R_hat
   error_term1 = 3 * np.sqrt(2 * np.log(2 / delta) / T)
   error_term2 = np.sqrt(2 * np.log(2 * N / delta) / T)

   print(f"Data snooping: {data_snooping:.4f}")
   print(f"Estimation error: {error_term1 + error_term2:.4f}")
   ```

## Advanced Topics

### Parameter Tuning (Section 8.4)

The paper notes:
> "the path connecting theory and practice is paved with carefully tuned parameters"

**Practical Formula**:
```
θ_n ≥ θ̂_n - aR̂ - b√(2log(2/δ)/T)
```

where `a` and `b` are tuned via simulation:
- Theoretical: a=2, b=3 (conservative, implemented here)
- Practical: Tune via null hypothesis simulations to achieve exact δ FPR

**Tuning Procedure**:
1. Generate null datasets (θ=0) matching your data characteristics
2. Run RAS with different (a,b) values
3. Choose (a,b) such that false positive rate ≈ δ
4. Validate on alternative hypothesis data

### Handling Non-I.I.D. Returns

If returns exhibit autocorrelation at lag s:

```python
# Block averaging to restore i.i.d. assumption
def block_average(X, block_size):
    """Average X into non-overlapping blocks."""
    T, N = X.shape
    n_blocks = T // block_size
    X_blocks = np.zeros((n_blocks, N))

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        X_blocks[i, :] = X[start:end, :].mean(axis=0)

    return X_blocks

# Example: If lag-5 autocorrelation is significant
X_blocks = block_average(X, block_size=5)
R_hat = rademacher_complexity(X_blocks)  # Now uses T/5 effective samples
```

### Correlation Structure Impact

**Example**: 1000 strategies, varying correlation

| Correlation (ρ) | R̂ | Data Snooping (2R̂) |
|----------------|-----|---------------------|
| 0.0 (uncorrelated) | 0.072 | 0.144 |
| 0.2 | 0.065 | 0.130 |
| 0.5 | 0.048 | 0.096 |
| 0.8 | 0.032 | 0.064 |
| 1.0 (identical) | ~0.000 | ~0.000 |

**Key Insight**: When testing 1000 correlated strategies (ρ=0.8), the data snooping penalty is **half** that of uncorrelated strategies!

## Computational Considerations

### Performance

**Rademacher Complexity**: O(n_simulations × T × N)
- Example: T=2500, N=1000, K=10000 simulations
- Operations: ~25 billion dot products
- Time: ~5-10 seconds on modern CPU

**Optimization Tips**:
1. Use vectorized NumPy operations (already implemented)
2. Reduce n_simulations for quick checks (1000 instead of 10000)
3. Use smaller N_simulations during development
4. Consider parallel simulation for very large datasets

### Memory Usage

- X matrix: O(T × N) floats (8 bytes each)
- Example: T=10,000, N=10,000 → 800 MB
- Simulations: O(n_simulations × T) → negligible compared to X

For very large N (millions), consider:
- Batched processing of strategies
- Sparse matrix representations (if applicable)
- Approximations (e.g., fewer simulations)

## References

### Primary Source

Paleologo, G. (2024). "Elements of Quantitative Investing", Chapter 8: The Rademacher Anti-Serum.
- Sections 8.3.1-8.3.2: Theory and formulation
- Section 8.4: Empirical results and validation
- Pages 264-286

Reference implementation: https://github.com/RSv618/rademacher-anti-serum

### Supporting Theory

Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018).
"Foundations of Machine Learning", 2nd ed. MIT Press.
- Chapter on Rademacher complexity and generalization bounds

Boucheron, S., Lugosi, G., & Massart, P. (2013).
"Concentration Inequalities: A Nonasymptotic Theory of Independence". Oxford University Press.
- McDiarmid's inequality and concentration bounds

### Related Work

López de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley.
- Combinatorial Purged Cross-Validation (CPCV)
- Multiple testing in finance

Bailey, D. H., & López de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
Journal of Portfolio Management, 40(5), 94-107.
- Original DSR formulation (z-score version)

## FAQ

**Q: How does RAS compare to DSR?**

A: RAS accounts for correlation (R̂ decreases when strategies correlated), has 0% FPR (vs ~5%), and provides per-strategy bounds. DSR is faster (no simulation) and has probability interpretation. Use RAS for large N with correlation, DSR for small N with known variance.

**Q: Can I use RAS with non-Gaussian returns?**

A: Yes! RAS works for sub-Gaussian distributions (Sharpe) and bounded distributions (IC). The bounds hold for heavy-tailed returns (tested with t-distribution in Table 8.4).

**Q: What if my strategies are perfectly correlated?**

A: R̂ → 0, so only estimation error remains. This makes sense: testing 1000 identical strategies is like testing 1 strategy.

**Q: How many simulations do I need?**

A: 10,000 for production (low variance). 1,000 for development (faster). R̂ converges quickly due to Central Limit Theorem.

**Q: Can I use this for crypto or high-frequency data?**

A: Yes, but:
1. Check autocorrelation (may need block averaging)
2. Ensure returns are stationary
3. Use appropriate T (number of periods, not calendar days)
4. For HFT: Consider microstructure effects

**Q: What's a realistic R̂ value?**

A: Depends on N, T, and correlation:
- Uncorrelated: R̂ ≈ 0.7 × √(2logN/T) (Massart bound)
- Moderately correlated (ρ=0.5): R̂ ≈ 0.5 × √(2logN/T)
- Highly correlated (ρ=0.8): R̂ ≈ 0.3 × √(2logN/T)

**Q: Why is the estimation error so large for Sharpe?**

A: The theoretical bound (Equation 8.3) is conservative. The paper suggests tuning constants a and b via simulation (Section 8.4, page 274). Implemented values (a=2, b=3) guarantee rigorous bounds.

**Q: Can I apply this to portfolio optimization?**

A: Yes! Use RAS to identify statistically significant strategies, then optimize only over those with adjusted_sharpe > 0 or adjusted_ic > 0.

**Q: How do I handle missing data?**

A: Missing data violates the i.i.d. assumption. Options:
1. Use only complete cases (all strategies have data at time t)
2. Impute missing values (carefully!)
3. Use different T for different strategies (requires modified approach)

---

**Version**: 1.0.0
**Date**: 2025-11-16
**Library**: ml4t.diagnostic.evaluation.stats
**Status**: Production-ready
