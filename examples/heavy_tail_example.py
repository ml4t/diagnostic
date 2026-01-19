"""Heavy Tail Analysis Example.

This example demonstrates heavy tail detection using:
- Hill estimator for tail index estimation
- QQ plots for distribution comparison
- Comprehensive tail analysis combining multiple methods

Heavy tail analysis is critical for financial risk management as it identifies
distributions with higher probability of extreme events than normal distributions.
"""

import numpy as np

from ml4t.diagnostic.evaluation.distribution import (
    analyze_tails,
    generate_qq_data,
    hill_estimator,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("HEAVY TAIL ANALYSIS EXAMPLE")
print("=" * 80)
print()

# =============================================================================
# Example 1: Normal Distribution (Thin Tails)
# =============================================================================
print("Example 1: Normal Distribution (Thin Tails)")
print("-" * 80)

# Generate normal data
normal_data = np.random.normal(0, 1, 1000)

# Hill estimator
hill_normal = hill_estimator(normal_data, tail="both")
print(hill_normal.summary())
print()

# =============================================================================
# Example 2: Student's t Distribution (df=3) - Heavy Tails
# =============================================================================
print("\n" + "=" * 80)
print("Example 2: Student's t (df=3) - Heavy Tails")
print("-" * 80)

# Generate t-distributed data (heavy tails)
t_data = np.random.standard_t(df=3, size=1000)

# Hill estimator
hill_t = hill_estimator(t_data, tail="both")
print(hill_t.summary())
print()

# =============================================================================
# Example 3: QQ Plot Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Example 3: QQ Plot Comparison")
print("-" * 80)

# Generate QQ plots for t-distributed data
qq_normal = generate_qq_data(t_data, distribution="normal")
qq_t = generate_qq_data(t_data, distribution="t", df=3)

print("Comparing t-distributed data against different distributions:\n")
print(qq_normal.summary())
print()
print(qq_t.summary())
print()

print(f"Conclusion: t-distribution fits better (R² = {qq_t.r_squared:.4f}) than")
print(f"normal distribution (R² = {qq_normal.r_squared:.4f})")
print()

# =============================================================================
# Example 4: Comprehensive Tail Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Example 4: Comprehensive Tail Analysis")
print("-" * 80)

# Analyze heavy-tailed data
result = analyze_tails(t_data)
print(result.summary())
print()

# =============================================================================
# Example 5: Comparing Different Distributions
# =============================================================================
print("\n" + "=" * 80)
print("Example 5: Comparing Different Distributions")
print("-" * 80)

# Generate different distributions
distributions = {
    "Normal (df=∞)": np.random.normal(0, 1, 1000),
    "Student's t (df=10)": np.random.standard_t(df=10, size=1000),
    "Student's t (df=5)": np.random.standard_t(df=5, size=1000),
    "Student's t (df=3)": np.random.standard_t(df=3, size=1000),
}

print("Tail Index Comparison:\n")
print(f"{'Distribution':<25} {'Tail Index (α)':<18} {'Classification':<15} {'Best Fit':<15}")
print("-" * 80)

for name, data in distributions.items():
    result = analyze_tails(data)
    alpha = result.hill_result.tail_index
    classification = result.hill_result.classification
    best_fit = result.best_fit
    print(
        f"{name:<25} {alpha:>6.2f} ± {result.hill_result.tail_index_se:>4.2f}      "
        f"{classification:<15} {best_fit:<15}"
    )

print()
print("Observations:")
print("  - Lower tail index (α) indicates heavier tails")
print("  - α ≤ 2: Heavy tails (infinite variance regime)")
print("  - 2 < α ≤ 4: Medium tails (finite variance, typical for finance)")
print("  - α > 4: Thin tails (approaching normal)")
print()

# =============================================================================
# Example 6: Financial Returns Simulation
# =============================================================================
print("\n" + "=" * 80)
print("Example 6: Simulated Financial Returns")
print("-" * 80)

# Simulate daily returns with heavy tails (realistic for financial data)
# Using t(4) which is typical for equity returns
returns = np.random.standard_t(df=4, size=252)  # One year of daily returns
returns = returns * 0.015  # Scale to realistic volatility (~1.5% daily std)

# Analyze tail behavior
returns_analysis = analyze_tails(returns)

print("Analysis of simulated financial returns (1 year daily):\n")
print(f"Tail Index: {returns_analysis.hill_result.tail_index:.2f}")
print(f"Classification: {returns_analysis.hill_result.classification}")
print(f"Best Fit Distribution: {returns_analysis.best_fit}")
print()

print("Risk Management Implications:")
if returns_analysis.hill_result.classification == "heavy":
    print("  ⚠️  Heavy tails detected:")
    print("  - Standard VaR may underestimate tail risk")
    print("  - Use CVaR (Expected Shortfall) instead")
    print("  - Consider extreme value theory for tail modeling")
    print("  - Sharpe ratio may be unreliable")
elif returns_analysis.hill_result.classification == "medium":
    print("  ⚠️  Medium tails detected (typical for finance):")
    print("  - Higher extreme event probability than normal")
    print("  - Use robust risk measures (CVaR, Sortino ratio)")
    print("  - Apply robust portfolio optimization")
    print("  - Monitor tail risk metrics")
else:
    print("  ✓ Thin tails:")
    print("  - Standard risk measures appropriate")
    print("  - Normal distribution assumptions reasonable")

print()

# =============================================================================
# Example 7: Upper vs Lower Tail Analysis
# =============================================================================
print("\n" + "=" * 80)
print("Example 7: Upper vs Lower Tail Analysis")
print("-" * 80)

# Generate asymmetric data (negative skew, common in equity returns)
skewed_returns = np.concatenate(
    [
        np.random.normal(0.001, 0.01, 900),  # Normal small movements
        np.random.normal(-0.03, 0.01, 100),  # Crash events (left tail)
    ]
)
np.random.shuffle(skewed_returns)

# Analyze each tail separately
upper_tail = hill_estimator(skewed_returns, tail="upper")
lower_tail = hill_estimator(skewed_returns, tail="lower")
both_tails = hill_estimator(skewed_returns, tail="both")

print("Analyzing asymmetric returns (negative skew):\n")
print(f"Upper Tail (gains):  α = {upper_tail.tail_index:.2f} ({upper_tail.classification})")
print(f"Lower Tail (losses): α = {lower_tail.tail_index:.2f} ({lower_tail.classification})")
print(f"Both Tails (min):    α = {both_tails.tail_index:.2f} ({both_tails.classification})")
print()

if lower_tail.tail_index < upper_tail.tail_index:
    print("Interpretation: Lower tail is heavier (more extreme downside risk)")
    print("  - Typical pattern for equity returns")
    print("  - Crashes are more extreme than rallies")
    print("  - Use downside risk measures (downside deviation, CVaR)")
else:
    print("Interpretation: Upper tail is heavier (more extreme upside)")
    print("  - Less common pattern")

print()

# =============================================================================
# Example 8: Real-World Application
# =============================================================================
print("\n" + "=" * 80)
print("Example 8: Decision Framework")
print("-" * 80)

print("Use tail analysis to inform:")
print()
print("1. Risk Model Selection:")
print("   - Heavy tails (α ≤ 2) → Extreme value theory, power law models")
print("   - Medium tails (2 < α ≤ 4) → Student's t, robust statistics")
print("   - Thin tails (α > 4) → Normal distribution, standard methods")
print()
print("2. Risk Measure Choice:")
print("   - Heavy tails → CVaR, Expected Shortfall, Maximum Drawdown")
print("   - Medium tails → CVaR, Robust Sharpe, Sortino Ratio")
print("   - Thin tails → VaR, Sharpe Ratio, Standard Deviation")
print()
print("3. Portfolio Construction:")
print("   - Heavy tails → Robust optimization, tail risk parity")
print("   - Medium tails → Modified mean-variance with robust covariance")
print("   - Thin tails → Standard mean-variance optimization")
print()
print("4. Position Sizing:")
print("   - Heavy tails → Conservative sizing, wider stops")
print("   - Medium tails → Moderate sizing based on CVaR")
print("   - Thin tails → Standard Kelly/Sharpe-based sizing")

print()
print("=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
