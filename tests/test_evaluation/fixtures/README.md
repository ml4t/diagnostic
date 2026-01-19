# DSR Reference Test Fixtures

## Overview

This directory contains reference test fixtures extracted from the authoritative 2025 paper by López de Prado, Lipton, and Zoonekynd. These fixtures provide ground truth for validating ml4t-diagnostic's implementation of:

- **PSR**: Probabilistic Sharpe Ratio
- **MinTRL**: Minimum Track Record Length
- **DSR**: Deflated Sharpe Ratio
- **Power**: Statistical power (true positive rate)

## Reference Paper

**Title**: How to use the Sharpe Ratio: A multivariate case study
**Authors**: Marcos López de Prado, Alexander Lipton, Vincent Zoonekynd
**Publication**: ADIA Lab Research Paper Series, No. 19
**Date**: September 23, 2025
**Location**: `/home/stefan/ml4t/software/evaluation/references/`

## Test Case Summary

### PSR Test Cases (6 cases)

1. **Portfolio manager 2-year monthly track** (Page 6)
   - SR = 0.456, skew = -2.448, kurt = 10.164, T = 24
   - Expected PSR = 0.987 (SR_0 = 0)
   - **Key insight**: Main worked example from paper

2. **Portfolio manager with SR_0=0.1** (Page 7)
   - Same parameters as #1, but SR_0 = 0.1
   - Expected PSR = 0.939
   - **Key insight**: Shows impact of non-zero benchmark

3. **Normal returns** (Page 7)
   - SR = 1.0, skew = 0, kurt = 3.0, T = 100
   - Expected PSR ≈ 1.0
   - **Key insight**: Validates normal returns edge case

4. **Small sample** (T=12)
   - SR = 0.5, skew = -0.5, kurt = 4.0, T = 12
   - Expected PSR = 0.937
   - **Key insight**: Tests small sample behavior

5. **High kurtosis** (Exhibit 1, Page 8)
   - Severe non-normality (skew = -2.4, kurt = 13.8)
   - Expected PSR = 0.987
   - **Key insight**: Validates extreme non-normality correction

6. **Negative skewness** (Exhibit 1, Page 8)
   - Moderate non-normality (skew = -1.7, kurt = 7.5)
   - Expected PSR = 0.999
   - **Key insight**: Tests skewness adjustment

### MinTRL Test Cases (4 cases)

1. **MinTRL with SR_0=0** (Page 8)
   - SR = 0.456, SR_0 = 0, α = 0.05
   - Expected MinTRL = 13.029 months
   - **Key insight**: Main worked example

2. **MinTRL with SR_0=0.1** (Page 8)
   - SR = 0.456, SR_0 = 0.1, α = 0.05
   - Expected MinTRL = 27.109 months
   - **Key insight**: Shows doubling of required samples with higher benchmark

3. **Normal returns**
   - SR = 1.0, SR_0 = 0.5, normal distribution
   - Expected MinTRL = 15.4
   - **Key insight**: Validates normal returns simplification

4. **High target SR**
   - SR = 1.5, SR_0 = 1.0
   - Expected MinTRL = 14.0
   - **Key insight**: Tests high SR scenarios

### DSR Test Cases (4 cases)

1. **DSR with K=10 trials** (Page 13)
   - SR = 0.456, K = 10, V[SR] = 0.1, T = 24
   - Expected DSR = 0.416 (vs PSR = 0.987!)
   - Expected E[max{SR}] = 0.498
   - **Key insight**: Main DSR example showing dramatic deflation

2. **Few trials** (K=3)
   - SR = 0.5, K = 3, V[SR] = 0.05
   - Expected DSR = 0.65, E[max] = 0.26
   - **Key insight**: Tests small K behavior

3. **Many trials** (K=100)
   - SR = 1.0, K = 100, V[SR] = 0.2
   - Expected DSR = 0.30 (heavily deflated)
   - Expected E[max] = 1.05
   - **Key insight**: Tests large K deflation

4. **Normal returns with multiple trials**
   - SR = 0.8, K = 10, normal distribution
   - Expected DSR = 0.60, E[max] = 0.498
   - **Key insight**: Validates normal returns case

### Power Test Cases (3 cases)

1. **Power calculation** (Page 10)
   - SR_0 = 0, SR_1 = 0.5, T = 24
   - Expected power = 0.685, β = 0.315
   - **Key insight**: Main power example

2. **Normal returns power**
   - SR_0 = 0, SR_1 = 1.0, T = 100, normal
   - Expected power = 1.0
   - **Key insight**: High power with large effect size

3. **Low power**
   - SR_0 = 0, SR_1 = 0.2, T = 30
   - Expected power = 0.30, β = 0.70
   - **Key insight**: Small effect size detection

### Variance Re-scaling Factors (19 values)

Extracted from **Exhibit 3, Page 13**: Standard deviation re-scaling factors for maximum of K standard normal variables.

Used in DSR calculation to adjust variance for multiple testing:
- K=1: 1.00000
- K=2: 0.82565
- K=10: 0.58681
- K=100: 0.42942

**Reference**: Appendix 2 (Pages 21-22) for derivation

## Important Notes

### Non-Annualized Sharpe Ratios

**All Sharpe ratios in the paper and these fixtures are NON-ANNUALIZED** (computed in the frequency of observations).

From Page 6:
> "Throughout this paper, we compute Sharpe ratios in the frequency of the observations, without annualizing, because annualization is unnecessary for inference."

### Numerical Precision

- Most cases use tolerance = 0.001 (0.1% error)
- MinTRL uses tolerance = 0.01 observations
- Some edge cases use looser tolerances (0.01-0.1) for extreme probabilities

### Test Case Validation

Each test case includes:
1. **Reference**: Exact citation to paper location (page, equation, exhibit)
2. **Tolerance**: Expected numerical error bounds
3. **Key insight**: What the test validates

## Usage in Tests

```python
from tests.test_evaluation.fixtures.dsr_reference import (
    PSR_CASES,
    MINTRL_CASES,
    DSR_CASES,
    POWER_CASES,
    VARIANCE_RESCALING_FACTORS,
)

# Example: Test PSR against all reference cases
@pytest.mark.parametrize("case", PSR_CASES, ids=lambda c: c.name)
def test_psr_reference(case):
    psr = calculate_psr(
        sharpe_ratio=case.sharpe_ratio,
        skewness=case.skewness,
        kurtosis=case.kurtosis,
        n_samples=case.n_samples,
        sharpe_star=case.sharpe_star,
    )
    assert abs(psr - case.expected_psr) < case.tolerance
```

## Paper Locations

### Key Equations
- **PSR**: Equation (9), Page 7
- **MinTRL**: Equation (11), Page 8
- **DSR (E[max])**: Equation (26), Page 12
- **Power**: Equations (15-17), Pages 9-10

### Key Exhibits
- **Exhibit 1** (Page 8): PSR vs t-test validation
- **Exhibit 2** (Page 10): Precision and recall
- **Exhibit 3** (Page 13): Variance re-scaling factors
- **Exhibit 4** (Page 15): FDR thresholds

### Key Examples
- **Main example**: Portfolio manager with 2-year monthly track (Pages 6-7, 8, 10, 13)
- **DSR deflation**: K=10 trials showing PSR=0.987 → DSR=0.416 (Page 13)

## Interpretation Decisions

Some test cases required interpretation or calculation from paper formulas:

1. **Small sample PSR**: Calculated using equations (4-5) with given parameters
2. **Normal returns cases**: Simplified using kurt=3, skew=0 assumptions
3. **Power cases**: Derived from equations (15-17) with specified parameters
4. **E[max{SR}]**: Calculated using False Strategy Theorem (Equation 26)

All interpretations are documented in the `reference` field of each test case.

## Future Enhancements

Potential additions from paper:

1. **oFDR cases** (Page 11): Observed Bayesian False Discovery Rate
2. **pFDR cases** (Page 10-11): Planned Bayesian False Discovery Rate
3. **FDR threshold cases** (Pages 14-15): Rejection thresholds for FDR control
4. **Hybrid FWER-FDR cases** (Pages 15-16): Combined corrections
5. **Monte Carlo validation** (Exhibit 1, 2, 5, 6): Statistical test validation

## References

Bailey, D. and M. López de Prado (2012): "The Sharpe Ratio Efficient Frontier." Journal of Risk, Vol. 15, No. 2, pp. 3-44.

Bailey, D. and M. López de Prado (2014): "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." The Journal of Portfolio Management, Vol. 40, No. 5, pp. 94-107.

López de Prado, M., Lipton, A., & Zoonekynd, V. (2025). How to use the Sharpe Ratio: A multivariate case study. ADIA Lab Research Paper Series, No. 19.
