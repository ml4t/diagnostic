# ML4T Diagnostic: Academic Foundations & Implementation Guide

This reference links implemented methods to their academic foundations and
describes their purpose, implementation, and source material.

---

## Table of Contents

1. [Multiple Testing & False Discoveries](#1-multiple-testing-false-discoveries)
   - [The Problem: Selection Bias](#the-problem-selection-bias)
   - [Deflated Sharpe Ratio (DSR)](#deflated-sharpe-ratio-dsr)
   - [Rademacher Anti-Serum (RAS)](#rademacher-anti-serum-ras)
   - [False Discovery Rate (FDR)](#false-discovery-rate-fdr)
   - [Choosing Between DSR, RAS, and FDR](#choosing-between-dsr-ras-and-fdr)
2. [Cross-Validation for Financial Data](#2-cross-validation-for-financial-data)
   - [Why Standard K-Fold Fails](#why-standard-k-fold-fails)
   - [Combinatorial Purged Cross-Validation (CPCV)](#combinatorial-purged-cross-validation-cpcv)
   - [Purging & Embargo](#purging-embargo)
3. [Information Coefficient Analysis](#3-information-coefficient-analysis)
4. [Statistical Tests](#4-statistical-tests)
5. [Feature Importance](#5-feature-importance)
6. [Risk Model Evaluation](#6-risk-model-evaluation)
7. [Complete Reference List](#7-complete-reference-list)

---

## 1. Multiple Testing & False Discoveries

### The Problem: Selection Bias

When you test N trading strategies and select the best one, you're not evaluating that strategy in isolation—you're evaluating the **maximum of N random variables**. Even if all strategies have zero expected return, you'll observe a positive Sharpe ratio for the best one simply due to chance.

**Example**: Test 100 random strategies (all with true SR = 0). The expected observed Sharpe for the best one is approximately:

```
E[max{SR}] ≈ √(2 × log(100)) / √T ≈ 3.0 / √T
```

For T = 252 daily observations, this is about **0.19** — a spurious "edge" from pure noise.

This is the **multiple testing problem**, and it's pervasive in quantitative finance where researchers routinely test hundreds or thousands of parameter combinations.

---

### Deflated Sharpe Ratio (DSR)

**Implementation**: `ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio()`

**What it does**: Adjusts an observed Sharpe ratio downward to account for the number of strategies tested, providing a probability that the true Sharpe ratio exceeds zero after accounting for selection bias.

**Why it matters**: A strategy with SR = 2.0 sounds impressive, but if you tested 1000 parameter combinations to find it, that SR may be entirely explained by selection bias.

#### Mathematical Foundation

The DSR extends the Probabilistic Sharpe Ratio (PSR) to account for multiple testing. The key insight is that the maximum of K independent Sharpe ratios follows an extreme value distribution.

**Formula** (López de Prado et al., 2025):

```
DSR = Φ((SR̂ - E[max{SR}]) / σ[SR̂])
```

Where:
- `SR̂` = observed (best) Sharpe ratio
- `E[max{SR}] = √Var[{SR_k}] × E[max{Z}]` = expected maximum under null
- `E[max{Z}] ≈ (1-γ)Φ⁻¹(1-1/K) + γΦ⁻¹(1-1/(Ke))` (Euler-Mascheroni approximation)
- `σ[SR̂] = √((1 - γ₃SR₀ + (γ₄-1)/4 × SR₀²) / T)` = standard error accounting for skewness/kurtosis
- `γ ≈ 0.5772` = Euler-Mascheroni constant

#### Usage Example

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics

# You tested 50 strategies, selected the best one, and already computed the cohort statistics
result = deflated_sharpe_ratio_from_statistics(
    observed_sharpe=1.5,      # Best strategy's SR
    n_trials=50,              # Number tested
    variance_trials=1.0,      # Var[{SR_1, ..., SR_50}]
    n_samples=252,            # Days of data
    skewness=0.0,             # Return skewness
    excess_kurtosis=0.0,      # Fisher convention (normal=0)
)

print(f"DSR probability: {result.probability:.3f}")
print(f"Expected max under null: {result.expected_max_sharpe:.3f}")
print(f"p-value: {result.p_value:.3f}")
```

**Critical Requirement**: You must provide the **actual variance** of Sharpe ratios across all K strategies tested. DSR cannot be meaningfully calculated without access to all strategies.

**Interpretation**:
- DSR = 0.95: 95% confidence that true SR > 0 after multiple testing correction
- DSR = 0.50: Coin flip—strategy is likely explained by selection bias
- DSR < 0.05: Strategy almost certainly overfit

**Primary References**:

1. **López de Prado, M., Lipton, A., & Zoonekynd, V. (2025)**
   "How to use the Sharpe Ratio: A multivariate case study."
   *ADIA Lab Research Paper Series*, No. 19.
   - Reference implementation: https://github.com/zoonek/2025-sharpe-ratio

2. **Bailey, D. H., & López de Prado, M. (2014)**
   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
   *Journal of Portfolio Management*, 40(5), 94-107.

---

### Rademacher Anti-Serum (RAS)

**Implementation**: `ml4t.diagnostic.evaluation.stats.rademacher_complexity()`,
`ras_ic_adjustment()`, `ras_sharpe_adjustment()`.

RAS uses the joint period-by-strategy matrix to account for selection among correlated
strategies. Repeating an identical strategy does not add complexity. It does not guarantee
zero false discoveries.

#### IC population bound and sampling assumptions

For `ras_ic_adjustment`, `observed_ic[n]` is the mean of period ICs `x[t, n]`.
`kappa` bounds **every possible period IC in the population**, almost surely, rather than
the mean IC. Pearson and rank correlations admit the safe default `kappa=1.0`.
A smaller bound needs an external justification. A typical alpha mean, a standard deviation,
and `max(abs(X))` from a finite sample cannot certify that unseen outcomes respect the bound.

The optional keyword-only `ic_samples=X` checks period-level violations, dimensions,
finite values and agreement with the supplied means. It rejects observed violations.
Passing these checks still does not prove a population bound. Without the matrix, an
obviously out-of-bound mean emits the existing warning; absence of that warning proves nothing.
`kappa=None` does not select an automatic sample-based bound.

For a fixed strategy class and IID period vectors in `[-kappa, kappa]`, the empirical
Rademacher theorem gives simultaneous one-sided lower bounds:

```text
R_hat = E_sigma[max_n sum_t sigma[t] * X[t,n] / T]
lower[n] = mean_IC[n] - 2*R_hat - 3*kappa*sqrt(2*log(2/delta)/T)
```

The expectation is over independent random signs. The estimation term includes the
concentration needed to replace population complexity with empirical complexity.
It follows by rescaling the empirical bound in Theorem 3.3 of
[Mohri, Rostamizadeh and Talwalkar (2018)](https://cs.nyu.edu/~mohri/mlbook/)
from `[0,1]` to `[-kappa,kappa]`; the
[author's lecture, slides 6-9](https://cs.nyu.edu/~mohri/mls/ml_learning_with_infinite_hypothesis_sets.pdf)
gives the bound and derivation. [Hoeffding (1963), Theorem 1](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)
requires population ranges for the independent summands. Small observed means do not supply those ranges.
Serial dependence, overlapping return horizons, and an unaccounted adaptive strategy class
invalidate the IID/fixed-class argument. This function does not correct those violations.

`complexity` must be the exact empirical expectation or a conservative upper bound for
the same matrix. `rademacher_complexity()` returns a Monte Carlo point estimate, which can
underestimate that expectation. Its simulation error is not automatically included in `delta`.
A certified Monte Carlo upper bound needs its own failure budget; a deterministic Massart
upper bound avoids that extra uncertainty at the cost of a larger penalty.

#### Runnable IC example with a deterministic complexity bound

```python
import numpy as np
from ml4t.diagnostic.evaluation.stats import ras_ic_adjustment

# Synthetic IID period IC vectors, T periods by N fixed strategies.
X = np.random.default_rng(42).uniform(-0.1, 0.1, (252, 5))
T, N = X.shape
# Massart's finite-class bound on the exact empirical complexity of this matrix.
complexity_upper = np.max(np.linalg.norm(X, axis=0)) * np.sqrt(2*np.log(N)) / T
result = ras_ic_adjustment(
    X.mean(axis=0), complexity_upper, T,
    ic_samples=X, return_result=True,
)
print(result.adjusted_values)
```

Negative lower bounds are retained. The default is conservative even when observed ICs
are small. Set `kappa=0.1` in this synthetic example only because its generator has known
support `[-0.1,0.1]`, not because the observed sample happens to fit that interval.
`plot_ras_analysis` displays the actual estimation penalty from the supplied original/adjusted
values and complexity; it does not subtract `2*kappa` as a separate penalty.

**Compatibility correction**: the previous default `kappa=0.02` described typical means,
not period bounds. Both the default and the previous smaller estimation coefficient have
been corrected, so results are more conservative, including calls with explicit `kappa`.
Existing positional arguments, array returns and `return_result=True` remain supported.

The separate Sharpe adjustment uses sub-Gaussian assumptions on standardized period
performance, not the bounded-IC contract. Raw heavy-tailed returns are not automatically
covered. Its implementation is unchanged by the IC correction.

**References**:

- Paleologo, G. (2025), *The Elements of Quantitative Investing*, Wiley,
  Rademacher Anti-Serum discussion. The IC coefficient above is justified by the
  explicit empirical-complexity theorem cited above, not an unverified book procedure number.
- [Bartlett and Mendelson (2002), *Rademacher and Gaussian Complexities*](https://jmlr.org/papers/v3/bartlett02a.html).
  Their normalization uses an absolute supremum and a factor of two; the displayed
  `R_hat` definition here follows the non-absolute, `1/T` convention.

---

### False Discovery Rate (FDR)

**Implementation**: `ml4t.diagnostic.evaluation.stats.benjamini_hochberg_fdr()`, `holm_bonferroni()`

**What it does**: Controls the expected proportion of false discoveries among rejected hypotheses.

**Why it matters**: FWER methods (Bonferroni, Holm) control the probability of *any* false positive, which becomes extremely conservative when testing hundreds of hypotheses. FDR instead controls the *proportion* of false positives, allowing more discoveries.

#### Benjamini-Hochberg Procedure (1995)

1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Find largest k such that p_(k) ≤ k × q / m
3. Reject hypotheses 1, 2, ..., k

This controls FDR at level q:
```
E[V/R] ≤ q
```
where V = false discoveries, R = total rejections.

#### Local FDR vs Tail-Area FDR (Efron & Hastie, 2016)

The **local** false discovery rate at z:
```
fdr(z) = π₀f₀(z) / f(z)
```

The **tail-area** FDR:
```
Fdr(z) = π₀F₀(z) / F(z)
```

- **fdr(z)**: "Given this specific z-value, what's the probability it's null?"
- **Fdr(z)**: "Among all z-values at least this extreme, what proportion are null?"

#### Usage Example

```python
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr, holm_bonferroni

p_values = [0.001, 0.01, 0.03, 0.08, 0.12, 0.25, 0.45]

# FDR control (less conservative, more discoveries)
fdr_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)
print(f"BH rejections: {fdr_result['n_rejected']}")

# FWER control (more conservative, fewer false positives)
fwer_result = holm_bonferroni(p_values, alpha=0.05)
print(f"Holm rejections: {fwer_result['n_rejected']}")
```

**When to use FDR vs FWER**:
- **FDR (Benjamini-Hochberg)**: Exploratory analysis, acceptable to have some false discoveries
- **FWER (Holm-Bonferroni)**: Confirmatory analysis, must avoid any false positive

**References**:

- **Benjamini, Y., & Hochberg, Y. (1995)**
  "Controlling the False Discovery Rate."
  *Journal of the Royal Statistical Society B*, 57(1), 289-300.

- **Efron, B., & Hastie, T. (2016)**
  "Computer Age Statistical Inference." Cambridge University Press.
  Chapter 15: Large-Scale Hypothesis Testing and False-Discovery Rates.

---

### Choosing Between DSR, RAS, and FDR

| Method | Best For | Accounts for Correlation? | Computational Cost |
|--------|----------|---------------------------|-------------------|
| **DSR** | Quick assessment of single best strategy | No | O(1) |
| **RAS** | Rigorous bounds on all strategies simultaneously | Yes | O(n_sim × T × N) |
| **FDR** | Screening many p-values, exploratory | No | O(N log N) |

**Decision Guide**:

1. **I tested many parameter combinations and selected the best one**:
   - Use **DSR** for quick assessment
   - Use **RAS** if strategies are correlated (parameter variations usually are)

2. **I have p-values from multiple independent tests**:
   - Use **FDR (BH)** for exploratory analysis
   - Use **FWER (Holm)** for confirmatory analysis

3. **I need simultaneous bounds on all strategies**:
   - Use **RAS** — bounds hold for all N strategies at once

---

## 2. Cross-Validation for Financial Data

### Why Standard K-Fold Fails

Standard K-fold cross-validation assumes observations are independent and identically distributed (i.i.d.). Financial time series violate this:

1. **Serial Correlation**: Adjacent returns are not independent
2. **Overlapping Labels**: Forward-looking labels create information leakage

**Example of Label Leakage**:
```
Sample 95's label = returns from days 95-100
Sample 100 is in test set
→ Training on sample 95 reveals information about test sample 100
```

**Consequence**: Inflated performance estimates, models that don't generalize.

---

### Combinatorial Purged Cross-Validation (CPCV)

**Implementation**: `ml4t.diagnostic.splitters.CombinatorialCV`

**What it does**: Generates multiple train/test splits by combining groups, with proper purging and embargo to prevent information leakage.

**Why it matters**: Instead of a single backtest path, CPCV provides a *distribution* of performance metrics, enabling detection of backtest overfitting.

#### How It Works

1. **Partitioning**: Divide time-series into N contiguous groups
2. **Combination Generation**: Generate all C(N,k) combinations of k test groups
3. **Purging**: Remove training samples that overlap with test labels
4. **Embargo**: Add buffer periods after test groups

Number of backtest paths = C(N,k):
- C(8,2) = 28 paths
- C(10,3) = 120 paths
- C(12,4) = 495 paths

#### Usage Example

```python
from ml4t.diagnostic.splitters import CombinatorialCV
import numpy as np

X = np.random.randn(1000, 10)
y = np.random.randn(1000)

cv = CombinatorialCV(
    n_groups=8,           # Split into 8 time groups
    n_test_groups=2,      # 2 groups for testing → C(8,2)=28 combinations
    label_horizon=5,      # Labels look forward 5 samples
    embargo_size=2,       # 2-sample buffer after test
    max_combinations=20   # Limit for efficiency
)

scores = []
for train_idx, test_idx in cv.split(X):
    # Train and evaluate your model
    pass
```

**Reference**:

- **López de Prado, M. (2018)**
  "Advances in Financial Machine Learning." Wiley.
  Chapter 7: Cross-Validation in Finance.

---

### Purging & Embargo

**Implementation**: `ml4t.diagnostic.core.purging.apply_purging_and_embargo()`

**Purging** removes training samples whose labels temporally overlap with test samples.

**Formula**: For test range [t_start, t_end] and label_horizon h:
```
Remove training samples where: t_train > t_start - h
```

**Embargo** adds a buffer after test groups to account for serial correlation.

**Formula**: After test range with embargo e:
```
Additionally remove: samples t_end+1 to t_end+e
```

**Reference**: López de Prado (2018), Chapter 7, Section 7.4.

---

## 3. Information Coefficient Analysis

### HAC-Adjusted Standard Errors

**Implementation**: `ml4t.diagnostic.evaluation.stats.hac_adjusted_ic()`

Computes standard errors for the Information Coefficient that account for heteroskedasticity and autocorrelation.

**The Newey-West estimator** (1987):

```
V̂[β] = (X'X)⁻¹ (Σ_j=-L^L w_j Γ̂_j) (X'X)⁻¹
```

Where w_j = 1 - |j|/(L+1) (Bartlett kernel).

### Stationary Bootstrap

**Implementation**: `ml4t.diagnostic.evaluation.stats.stationary_bootstrap_ic()`

The **stationary bootstrap** (Politis & Romano, 1994) preserves temporal
dependence through random block lengths and provides a resampling alternative
to HAC inference.

**References**:

- **Newey, W. K., & West, K. D. (1987)**
  "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703-708.

- **Politis, D. N., & Romano, J. P. (1994)**
  "The Stationary Bootstrap."
  *Journal of the American Statistical Association*, 89(428), 1303-1313.

---

## 4. Statistical Tests

### Stationarity Tests

| Test | Null Hypothesis | Implementation |
|------|----------------|----------------|
| **ADF** | Unit root (non-stationary) | `adf_test()` |
| **KPSS** | Stationary | `kpss_test()` |
| **Phillips-Perron** | Unit root | `pp_test()` |

**Best Practice**: Use ADF and KPSS together. If ADF rejects and KPSS doesn't reject, series is stationary.

### Distribution Tests

| Test | Best For | Implementation |
|------|----------|----------------|
| **Jarque-Bera** | Large samples (n > 2000) | `jarque_bera_test()` |
| **Shapiro-Wilk** | Small samples (n < 2000) | `shapiro_wilk_test()` |

### Volatility Tests

| Test | Purpose | Implementation |
|------|---------|----------------|
| **ARCH-LM** | Test for volatility clustering | `arch_lm_test()` |
| **GARCH** | Model conditional volatility | `fit_garch()` |

---

## 5. Feature Importance

**Implementation**: `ml4t.diagnostic.metrics`

| Method | Type | Strengths | Weaknesses |
|--------|------|-----------|------------|
| **MDI** | In-sample, tree-based | Fast | Biased toward high-cardinality |
| **MDA/PFI** | Out-of-sample, permutation | Unbiased | Computationally expensive |
| **SHAP** | Model-agnostic | Theoretically grounded | Most expensive |

**Best Practice** (López de Prado, 2018):
1. Use **clustered MDI** to group correlated features
2. Validate with **MDA** on out-of-sample data
3. Never trust a single method — use consensus

**Key Insight**: "Backtesting is not a research tool. Feature importance is."

**References**:

- **López de Prado, M. (2018)**. Chapter 8: Feature Importance.

- **Lundberg, S. M., & Lee, S. I. (2017)**
  "A Unified Approach to Interpreting Model Predictions." NeurIPS.

---

## 6. Risk Model Evaluation

### Why This Matters

Risk models (covariance matrices) are fundamental to portfolio construction. A flawed risk model leads to suboptimal portfolios, excessive concentration, and unexpected drawdowns. Yet validation is often neglected because "testing is hard, mundane, and humbling."

> "Everybody wants to save the Earth; nobody wants to help Mom do the dishes. This is a chapter about doing dishes."
>
> — Paleologo (2024), Chapter 5

### The R-Squared Trap

**Do NOT use R-squared to evaluate factor risk models.** R-squared is **rotationally invariant**: applying random rotation matrices to factors changes the covariance structure but leaves R-squared identical.

**Mathematical Proof** (Paleologo, 2024): Let B be factor loadings. Applying random rotation R:
- New covariance: (BR)(BR)ᵀ = BRRᵀBᵀ ≠ BBᵀ (structure changed)
- R-squared: identical (depends only on explained variance ratio)

### Robust Volatility Loss Functions

**QLIKE Loss** (Patton, 2009):
```
QLIKE(σ̂, r) := (1/T) Σ [r²/σ̂² - log(r²/σ̂²) - 1]
```

**Why QLIKE > MSE**: QLIKE is proportional to negative log-likelihood of normal distribution. It penalizes underestimation of risk more appropriately in certain regimes.

**Implementation**: `ml4t.diagnostic.evaluation.risk.qlike_loss()`

### Precision Matrix Testing

Portfolio weights in mean-variance optimization depend on Σ⁻¹ (precision matrix). Errors in Σ⁻¹ amplify realized risk.

**Mahalanobis Variance Test (MALV)**:
```
νₜ := (1/n) rₜᵀ Σ̂⁻¹ rₜ
MALV := var(ν₁, ..., νT)
```

If the risk model is perfect and returns are Gaussian, νₜ follows χ²(n)/n distribution with low variance. High MALV indicates the precision matrix fails to properly whiten residuals.

**Reference**: Paleologo (2024), Chapter 5, Section 5.2.2.

### Minimum Variance Portfolio Test

**Theorem** (Engle & Colacito, 2006; Paleologo, 2024):
> The realized volatility of portfolio w(Σ̂) is greater than w(Σ), and equal if and only if Σ̂ ∝ Σ.

**Implication**: Build MVP using your predicted Σ̂, measure realized variance. Lower is better.

### Factor Model Turnover

```
turnover_F := (1/T) Σ ||Pₜ - Pₜ₋₁||_F
```

Where Pₜ = factor mimicking portfolio weights.

High turnover means factor definitions change frequently, inducing trading costs *independent of alpha*. This is often overlooked.

---

## 7. Complete Reference List

### Primary Books

1. **López de Prado, M. (2018)**
   "Advances in Financial Machine Learning." Wiley.
   - Chapter 7: Cross-Validation in Finance
   - Chapter 8: Feature Importance
   - Chapter 11: The Dangers of Backtesting
   - Chapter 14: Backtest Statistics

2. **Paleologo, G. (2024)**
   "Elements of Quantitative Investing."
   - Chapter 5: Evaluating Risk (QLIKE, MALV, R² trap)
   - Chapter 8: Evaluating Excess Returns (RAS)

3. **Efron, B., & Hastie, T. (2016)**
   "Computer Age Statistical Inference." Cambridge University Press.
   - Chapter 15: Large-Scale Hypothesis Testing and FDR

### Key Papers

4. **Bailey, D. H., & López de Prado, M. (2014)**
   "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94-107.

5. **López de Prado, M., Lipton, A., & Zoonekynd, V. (2025)**
   "How to use the Sharpe Ratio." *ADIA Lab Research Paper Series*, No. 19.

6. **Benjamini, Y., & Hochberg, Y. (1995)**
   "Controlling the False Discovery Rate." *JRSS-B*, 57(1), 289-300.

7. **Newey, W. K., & West, K. D. (1987)**
   "HAC Covariance Matrix." *Econometrica*, 55(3), 703-708.

8. **Politis, D. N., & Romano, J. P. (1994)**
   "The Stationary Bootstrap." *JASA*, 89(428), 1303-1313.

9. **Patton, A. J., & Sheppard, K. (2009)**
   "Evaluating Volatility and Correlation Forecasts."
   *Handbook of Financial Time Series*.

10. **Engle, R. F., & Colacito, R. (2006)**
    "Testing and Valuing Dynamic Correlations for Asset Allocation."
    *Journal of Business & Economic Statistics*, 24(2), 238-253.

11. **Murphy, A. H. (1973)**
    "A New Vector Partition of the Probability Score."
    *Journal of Applied Meteorology*, 12(4), 595-600.

12. **Stephenson, D. B., Coelho, C. A. S., & Jolliffe, I. T. (2008)**
    "Two Extra Components in the Brier Score Decomposition."
    *Weather and Forecasting*, 23(4), 752-757.

13. **Cameron, A. C., & Miller, D. L. (2015)**
    "A Practitioner's Guide to Cluster-Robust Inference."
    *Journal of Human Resources*, 50(2), 317-372.

14. **Snowberg, E., & Wolfers, J. (2010)**
    "Explaining the Favorite-Long Shot Bias: Is it Risk-Love or Misperceptions?"
    *Journal of Political Economy*, 118(4), 723-746.

### Reference Implementations

- DSR: https://github.com/zoonek/2025-sharpe-ratio
- RAS: https://github.com/RSv618/rademacher-anti-serum

---

## Implementation Notes

All implementations in ML4T Diagnostic:

1. **Follow original methodology**: Formulas match published papers
2. **Include numerical safeguards**: Handle edge cases, NaN values, small samples
3. **Support multiple data formats**: Polars, Pandas, NumPy
4. **Provide confidence measures**: p-values, confidence intervals
5. **Are extensively tested**: Property-based tests validate mathematical properties

For implementation details, see source code docstrings which include formula derivations.

---

*Document Version: 2.1*
*Last Updated: 2025-12-16*
*Library Version: ml4t-diagnostic 1.2.0*
