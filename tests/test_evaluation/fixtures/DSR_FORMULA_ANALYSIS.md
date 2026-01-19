# DSR Formula Analysis: 2014 vs 2025 Formulations

## Executive Summary

**CRITICAL FINDING**: The 2014 and 2025 formulations of DSR return **mathematically equivalent but differently formatted results**:

- **2014 (Bailey & López de Prado)**: Returns DSR as a **z-score** (range: -∞ to +∞)
- **2025 (López de Prado et al.)**: Returns DSR as a **probability** (range: 0 to 1)

**Relationship**: `DSR_2025 = Φ(DSR_2014)` where Φ is the standard normal CDF.

**Our Current Implementation**: Uses 2014 z-score formulation.

**Required Action**: Add `return_format` parameter to support both formulations.

---

## Detailed Analysis

### Test Case: Page 13 Example (K=10 trials)

**Input Parameters**:
- Observed Sharpe: 0.456
- N trials: 10
- Skewness: -2.448
- Kurtosis: 10.164
- N samples: 24

**Paper Expectations (2025)**:
- DSR: **0.416** (probability)
- E[max SR]: 0.498

**Our Implementation (2014 style)**:
- DSR: **-0.153269** (z-score)
- E[max SR]: 0.547602
- P-value: 0.560907

**Conversion Test**:
```python
from scipy.stats import norm

dsr_z_score = -0.153269
dsr_probability = norm.cdf(dsr_z_score)
# Result: 0.439093

paper_dsr = 0.416
difference = abs(0.439093 - 0.416)  # 0.023093 = 2.3% error
```

**Conclusion**: When we convert our z-score to a probability using `norm.cdf()`, we get **0.439** which is within **2.3%** of the paper's **0.416**. The remaining discrepancy is due to differences in the E[max] calculation (see below).

---

## Mathematical Formulation Differences

### Common Ground (Both Papers)

**Step 1**: Calculate E[max{SR_k}] under null hypothesis
```
E[max] = √(2 log K) - adjustments
```

**Step 2**: Calculate Std[max{SR_k}]
```
Std[max] = π / (√(2 log K) * √6)
```

**Step 3**: Compute test statistic
```
z = (observed_SR - E[max]) / Std[max]
```

### Divergence Point

**2014 Formulation** (Bailey & López de Prado):
```python
# Return the z-score directly
DSR_2014 = z  # Range: (-∞, +∞)
p_value = 1 - Φ(z)  # Right-tail probability
```

**2025 Formulation** (López de Prado et al.):
```python
# Return the CDF (left-tail probability)
DSR_2025 = Φ(z)  # Range: (0, 1)
# Interpretation: Probability that true SR > 0 given K trials
```

**Key Insight**: Both formulations are **mathematically equivalent**:
- `DSR_2025 = Φ(DSR_2014)`
- `DSR_2014 = Φ⁻¹(DSR_2025)`

---

## E[max] Calculation Differences

Our implementation uses:
```python
log_n = np.log(n_trials)
sqrt_2log_n = np.sqrt(2 * log_n)
expected_max = sqrt_2log_n - (np.log(log_n) + np.log(4 * np.pi)) / (2 * sqrt_2log_n)
```

**For K=10**:
- Our E[max]: **0.5476**
- Paper E[max]: **0.498**
- Difference: **0.0496** (10% higher)

**Likely Cause**: Different terms in the extreme value approximation. The 2025 paper may use:
```python
# Simpler approximation (conjecture)
expected_max = sqrt_2log_n - log(log_n)/(2*sqrt_2log_n) + γ/sqrt_2log_n
# where γ ≈ 0.5772 (Euler-Mascheroni constant)
```

**Impact**: This E[max] difference propagates through to the final DSR value, explaining why our converted probability (0.439) still differs from the paper's (0.416) by 2.3%.

---

## Semantic Interpretation Differences

### 2014 DSR (Z-Score)

**Returns**: Standard normal z-statistic
**Interpretation**: Number of standard deviations the observed SR is above/below the expected maximum under null
**Usage**:
- DSR < -1.96 → significantly worse than expected by chance (α=0.05, two-tailed)
- DSR > 0 → better than expected maximum under null

**Example**: DSR = -0.15 means "observed SR is 0.15 standard deviations below the expected maximum"

### 2025 DSR (Probability)

**Returns**: Probability (via CDF)
**Interpretation**: Probability that the true Sharpe ratio is positive, after adjusting for multiple testing
**Usage**:
- DSR < 0.5 → more likely to be a false discovery
- DSR > 0.95 → high confidence of true positive

**Example**: DSR = 0.42 means "42% probability the strategy has true positive SR after accounting for K trials"

---

## User Impact Analysis

### Who Needs Which Format?

**Z-Score Format (2014)**:
- Statistical practitioners familiar with hypothesis testing
- Users who want to set custom significance thresholds
- Those comparing DSR to other z-statistics
- Academic researchers citing the 2014 paper

**Probability Format (2025)**:
- Practitioners wanting intuitive "chance of success" metric
- Those implementing Bayesian workflows
- Users following the 2025 paper methodology
- Business stakeholders needing interpretable metrics

### Current User Confusion

Without both formats, users face:
1. **Mismatch with 2025 paper**: Cannot replicate published results
2. **Unclear interpretation**: Z-scores are less intuitive than probabilities
3. **Version ambiguity**: Unclear which paper formulation we implement

---

## Recommended Solution

### Implementation Plan

Add `return_format` parameter to `deflated_sharpe_ratio()`:

```python
def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    return_components: bool = False,
    return_format: Literal["zscore", "probability"] = "zscore",  # NEW
) -> float | dict[str, float]:
    """Calculate Deflated Sharpe Ratio.

    Parameters
    ----------
    return_format : {"zscore", "probability"}, default "zscore"
        Output format for DSR:
        - "zscore": Returns z-statistic (2014 formulation, range: -∞ to +∞)
        - "probability": Returns CDF probability (2025 formulation, range: 0 to 1)
    """
    # ... existing calculation ...

    # Calculate z-score (2014 formulation)
    dsr_zscore = (observed_sharpe - expected_max) / std_max

    # Convert to probability if requested (2025 formulation)
    if return_format == "probability":
        dsr = norm.cdf(dsr_zscore)
    else:
        dsr = dsr_zscore

    # ... return logic ...
```

### Backwards Compatibility

- **Default**: `return_format="zscore"` preserves existing behavior
- **No breaking changes**: All existing code continues to work
- **Clear migration path**: Users can opt into 2025 format

### Documentation Updates

1. **Docstring**: Explain both formulations and when to use each
2. **Examples**: Show both formats with interpretation
3. **References**: Cite both 2014 and 2025 papers
4. **Migration guide**: How to convert between formats

---

## Test Strategy

### Validation Tests

1. **2014 Format (Z-Score)**:
   ```python
   result = deflated_sharpe_ratio(..., return_format="zscore")
   # Validate: -∞ < result < +∞
   # Validate: p_value = 1 - norm.cdf(result)
   ```

2. **2025 Format (Probability)**:
   ```python
   result = deflated_sharpe_ratio(..., return_format="probability")
   # Validate: 0 <= result <= 1
   # Validate: matches paper values within tolerance
   ```

3. **Format Consistency**:
   ```python
   z = deflated_sharpe_ratio(..., return_format="zscore")
   p = deflated_sharpe_ratio(..., return_format="probability")
   assert abs(p - norm.cdf(z)) < 1e-10
   ```

### Test Updates

- **Existing tests**: Continue passing (default="zscore")
- **New tests**: Add `return_format="probability"` validation against 2025 paper
- **Remove XFAIL**: Tests should pass with correct format

---

## Open Questions

### 1. Should we fix the E[max] calculation?

**Current**: E[max] is 10% higher than 2025 paper
**Options**:
- A) Keep current (conservative estimate)
- B) Match 2025 paper exactly (requires reverse-engineering their approximation)
- C) Make E[max] formula configurable

**Recommendation**: **Option B** - Match 2025 paper for consistency

### 2. Should we deprecate z-score format?

**Arguments for**: 2025 paper is newer, probability is more intuitive
**Arguments against**: Z-scores are standard in statistics, breaking change for existing users

**Recommendation**: **Keep both** - z-score as default for backwards compatibility

### 3. Should p-value calculation change?

**Current**: `p_value = 1 - norm.cdf(z)` (right-tail)
**2025 Paper**: Uses left-tail probability (DSR itself)

**Recommendation**: When `return_format="probability"`, set `p_value = dsr` for consistency

---

## References

1. **Bailey, D. H., & López de Prado, M. (2014)**. "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.
   - Original DSR formulation (z-score)

2. **López de Prado, M., Lipton, A., & Zoonekynd, V. (2025)**. "How to use the Sharpe Ratio: A multivariate case study." *ADIA Lab Research Paper Series*, No. 19.
   - Enhanced DSR formulation (probability)
   - Page 13: DSR = 0.416 for K=10 example

---

## Implementation Checklist

- [ ] Add `return_format` parameter to `deflated_sharpe_ratio()`
- [ ] Implement CDF conversion for `return_format="probability"`
- [ ] Update docstring with both formulations
- [ ] Add examples for both formats
- [ ] Update tests to validate both formats
- [ ] Remove XFAIL markers from DSR tests
- [ ] Add migration guide to documentation
- [ ] (Optional) Fix E[max] calculation to match 2025 paper

---

**Status**: Analysis complete - ready for implementation
**Priority**: HIGH - blocking TASK-012 completion
**Estimated**: 2 hours implementation + 1 hour testing
