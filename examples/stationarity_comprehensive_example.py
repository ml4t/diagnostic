"""Comprehensive stationarity analysis example.

Demonstrates the analyze_stationarity() function which combines ADF, KPSS,
and PP tests to provide robust stationarity assessment with consensus interpretation.
"""

import numpy as np

from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity

# =============================================================================
# Example 1: White Noise (Stationary)
# =============================================================================

print("=" * 80)
print("Example 1: White Noise (Stationary)")
print("=" * 80)

np.random.seed(42)
white_noise = np.random.randn(1000)

result = analyze_stationarity(white_noise)

print("\n" + result.summary())
print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 2: Random Walk (Non-Stationary)
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 2: Random Walk (Non-Stationary)")
print("=" * 80)

np.random.seed(42)
random_walk = np.cumsum(np.random.randn(1000))

result = analyze_stationarity(random_walk)

print("\n" + result.summary())
print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 3: Quasi-Stationary (Tests May Disagree)
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 3: Quasi-Stationary Series (Tests May Disagree)")
print("=" * 80)

np.random.seed(42)
t = np.arange(1000)
quasi_stationary = np.random.randn(1000) + 0.005 * t  # Weak trend

result = analyze_stationarity(quasi_stationary)

print("\n" + result.summary())
print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 4: Selective Test Execution
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 4: Run Only ADF and KPSS Tests")
print("=" * 80)

np.random.seed(42)
data = np.random.randn(1000)

# Run only ADF and KPSS (skip PP if not needed)
result = analyze_stationarity(data, include_tests=["adf", "kpss"])

print(f"\nNumber of tests run: {result.n_tests_run}")
print(f"Consensus: {result.consensus}")
print(f"Agreement: {result.agreement_score:.1%}")

print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 5: Custom Significance Level
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 5: Custom Significance Level (alpha=0.01)")
print("=" * 80)

np.random.seed(42)
data = np.random.randn(1000)

result = analyze_stationarity(data, alpha=0.01, include_tests=["adf", "kpss"])

print(f"\nConsensus at 1% level: {result.consensus}")
print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 6: Custom Test Parameters
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 6: Test for Trend Stationarity")
print("=" * 80)

np.random.seed(42)
# Series with linear trend
t = np.arange(1000)
trending = np.random.randn(1000) + 0.01 * t

result = analyze_stationarity(
    trending,
    include_tests=["adf", "kpss"],
    regression="ct",  # Test for trend stationarity
    maxlag=20,  # ADF parameter
)

print(f"\nConsensus with trend regression: {result.consensus}")
print("\nSummary DataFrame:")
print(result.summary_df)

# =============================================================================
# Example 7: Accessing Individual Test Results
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 7: Accessing Individual Test Results")
print("=" * 80)

np.random.seed(42)
data = np.random.randn(1000)

result = analyze_stationarity(data, include_tests=["adf", "kpss"])

print("\nADF Test:")
print(f"  Statistic: {result.adf_result.test_statistic:.4f}")
print(f"  P-value: {result.adf_result.p_value:.4f}")
print(f"  Stationary: {result.adf_result.is_stationary}")
print(f"  Lags used: {result.adf_result.lags_used}")

print("\nKPSS Test:")
print(f"  Statistic: {result.kpss_result.test_statistic:.4f}")
print(f"  P-value: {result.kpss_result.p_value:.4f}")
print(f"  Stationary: {result.kpss_result.is_stationary}")
print(f"  Lags used: {result.kpss_result.lags_used}")

# =============================================================================
# Example 8: Interpretation Guide
# =============================================================================

print("\n\n" + "=" * 80)
print("Example 8: Interpretation Guide")
print("=" * 80)

print("""
Consensus Interpretations:

1. strong_stationary (agreement = 1.0):
   - All tests agree the series is stationary
   - Safe to use in models requiring stationarity
   - White noise typically shows this

2. likely_stationary (agreement >= 0.67):
   - Majority of tests agree on stationarity
   - Generally safe for stationary models
   - Some uncertainty remains

3. inconclusive (agreement = 0.5 or 1-1-1 split):
   - Tests provide conflicting evidence
   - May be quasi-stationary or borderline case
   - Consider differencing or detrending

4. likely_nonstationary (agreement >= 0.67):
   - Majority of tests agree on non-stationarity
   - Apply differencing before modeling
   - Some uncertainty remains

5. strong_nonstationary (agreement = 1.0):
   - All tests agree the series has unit root
   - Requires differencing or cointegration approach
   - Random walk typically shows this

Key Differences Between Tests:
- ADF/PP: H0 = unit root (non-stationary), reject => stationary
- KPSS: H0 = stationary, reject => non-stationary
- Strong evidence requires agreement between tests with opposite hypotheses
""")
