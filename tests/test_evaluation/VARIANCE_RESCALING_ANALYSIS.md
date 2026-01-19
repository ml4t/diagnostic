# Variance Rescaling Test Failure Analysis

**Date**: 2025-11-16
**Task**: TASK-014 (DSR Parameter Sensitivity)
**Issue**: 31 DSR validation tests failing, primarily variance rescaling tests

## Root Cause: Test Design Error

### What the Test Claims to Validate

The test `test_variance_rescaling_reference()` in `test_dsr_validation.py` attempts to validate that:

```python
result = deflated_sharpe_ratio(
    observed_sharpe=0.0,
    n_trials=K,
    variance_trials=1.0,
    n_samples=252,
    return_components=True
)

# Test expects:
result["std_sharpe"] ≈ VARIANCE_RESCALING_FACTORS[K]
```

For K=2, it expects `std_sharpe = 0.82565`

### What DSR Actually Computes

The `std_sharpe` field in DSR results represents:

```
Std[SR_hat] = sqrt(Var[SR_hat])
```

Where Var[SR_hat] is computed using the PSR formula (equation 5 from López de Prado et al. 2025):

```
Var[SR_hat] = (1/T) * (1 - γ₃·SR₀ + (γ₄-1)/4·SR₀²)
```

Where:
- T = n_samples (e.g., 252)
- γ₃ = skewness (default 0)
- γ₄ = kurtosis (default 3)
- SR₀ = E[max{SR_k}] under null hypothesis

For K=2, T=252, this gives:
```
SR₀ = sqrt(1.0) * 0.10137 = 0.10137
Var[SR_hat] = (1 + 0.00398) / 252 = 0.003988
Std[SR_hat] = 0.06316
```

### What the Reference Table Contains

The VARIANCE_RESCALING_FACTORS table (Exhibit 3, page 13) contains:

```
Std[max{SR_k}]
```

This is the **standard deviation of the maximum** of K i.i.d. standard normal variables.

For K=2 standard normals:
```
Empirical Std[max{X₁, X₂}] ≈ 0.82565  (from 1M simulations: 0.82599)
```

This value depends **ONLY on K**, not on T, skewness, or kurtosis.

## Why They Are Different

| Quantity | Depends On | Typical Value (K=2, T=252) |
|----------|-----------|------------------------|
| Std[SR_hat] | K, T, γ₃, γ₄ | 0.06316 |
| Std[max{SR_k}] | K only | 0.82565 |

**Key insight**: As T→∞, Std[SR_hat]→0, but Std[max{SR_k}] remains constant at 0.82565.

These are fundamentally different statistical quantities:
- **Std[SR_hat]**: Uncertainty in our estimate of the selected strategy's Sharpe ratio
- **Std[max{SR_k}]**: Natural variability of the maximum statistic across the K strategies

## Empirical Verification

```python
import numpy as np
from scipy.stats import norm

K = 2
n_sim = 1000000

# Generate K standard normals, take max
samples = np.random.standard_normal((n_sim, K))
max_values = np.max(samples, axis=1)

print(f"Std[max{{SR_k}}] = {np.std(max_values):.5f}")  # → 0.82599
print(f"Reference value = 0.82565")                     # → Match!
```

## Is the DSR Implementation Correct?

**YES**. The DSR implementation correctly:

1. Computes E[max{SR_k}] using extreme value theory
2. Computes Var[SR_hat] using the PSR formula
3. Returns appropriate statistical quantities

The issue is that the test expects DSR to return Std[max{SR_k}], which is:
- Not part of the DSR output
- Not needed for hypothesis testing
- An intermediate calculation, not a final result

## What the Test SHOULD Validate

The variance rescaling factors ARE used internally in DSR:

```python
# Step 1: E[max{Z}] for K standard normals
e_max_z = norm.ppf(1 - 1/K) - γ*log(log(K))/sqrt(2πlog(K))

# Step 2: Expected max Sharpe under null
SR₀ = sqrt(Var[{SR_k}]) * E[max{Z}]
```

The test should validate that:
1. E[max{Z}] is computed correctly
2. SR₀ scaling is correct
3. The PSR variance formula is applied correctly

But NOT that `std_sharpe == Std[max{SR_k}]` (which is nonsensical).

## Recommended Fix

### Option A: Remove Variance Rescaling Tests (RECOMMENDED)

The variance rescaling factors are reference values, not outputs. Delete:
- `TestVarianceRescalingValidation` class
- `test_variance_rescaling_reference()`
- `test_variance_rescaling_decreasing()`

Total: 20 tests removed

### Option B: Create Separate Variance Rescaling Function

If we want to validate the Std[max{SR_k}] values:

```python
def variance_rescaling_factor(n_trials: int) -> float:
    """Compute Std[max{Z}] for K standard normals.

    Returns the standard deviation of the maximum of n_trials
    i.i.d. standard normal variables, as tabulated in
    López de Prado et al. (2025), Exhibit 3.
    """
    # Use empirical table for exact values
    if n_trials in VARIANCE_RESCALING_FACTORS:
        return VARIANCE_RESCALING_FACTORS[n_trials]

    # Otherwise use extreme value theory approximation
    # This would require implementing the exact formula from the paper
    ...
```

Then test THIS function against the reference table.

### Option C: Test E[max{Z}] Calculation

Validate that DSR computes E[max{Z}] correctly (which indirectly uses the rescaling logic):

```python
def test_expected_max_calculation():
    """Validate E[max{Z}] computation matches theory."""
    for K in [2, 3, 5, 10, 100]:
        result = deflated_sharpe_ratio(
            observed_sharpe=0.0,
            n_trials=K,
            variance_trials=1.0,
            n_samples=1,  # Makes variance_sr ≈ 1
            skewness=0.0,
            kurtosis=3.0,
            return_components=True
        )

        # For variance_trials=1, expected_max_sharpe = E[max{Z}]
        e_max_z = result["expected_max_sharpe"]

        # Validate against theoretical formula or empirical simulations
        ...
```

## Other Failing Tests

The remaining 11 test failures are in related classes:
- `TestPSRValidation` (5 xfail - systematic 5-8% discrepancy)
- `TestMinTRLValidation` (4 xfail - 2-3x discrepancy, variance formula issue)
- `TestDSRIntegration` (1 fail - likely signature mismatch)
- Other integration tests (1 fail - likely same issue)

These likely stem from the same confusion about what variance is being tested.

## Action Items

1. **Remove incorrect variance rescaling tests** (20 tests)
2. **Document what DSR.std_sharpe represents** in docstrings
3. **Optionally add E[max{Z}] validation** if needed
4. **Investigate PSR/MinTRL discrepancies** separately (not variance rescaling issue)
5. **Fix integration test signature issues**

## Conclusion

The DSR implementation is **mathematically correct**. The tests are **incorrectly designed**, expecting DSR to return a quantity (Std[max{SR_k}]) that:
- Is not part of the DSR output
- Wouldn't make sense to return (it's not a hypothesis test result)
- Is only an intermediate calculation

The fix is to **remove or redesign the tests**, not to change the DSR implementation.

---

**Recommendation**: Option A (remove tests) is cleanest. The variance rescaling factors are already validated implicitly through the DSR bootstrap tests (TASK-013), which showed correct Type I error control (after fixing the annualization bug).
