"""Example: Comprehensive distribution analysis for financial returns.

This example demonstrates how to use analyze_distribution() to perform complete
statistical characterization of return distributions, including:
- Distribution moments (skewness, kurtosis)
- Normality tests (Jarque-Bera, Shapiro-Wilk)
- Tail analysis (Hill estimator, QQ plots)
- Distribution recommendation for risk modeling
"""

import numpy as np

from ml4t.diagnostic.evaluation.distribution import analyze_distribution

# Set random seed for reproducibility
np.random.seed(42)


def example_normal_returns():
    """Example 1: Normal returns (ideal case)."""
    print("=" * 80)
    print("EXAMPLE 1: Normal Returns (Ideal Case)")
    print("=" * 80)

    # Simulate normal returns
    returns = np.random.normal(loc=0.001, scale=0.02, size=1000)

    # Comprehensive analysis
    result = analyze_distribution(returns, alpha=0.05, compute_tails=True)

    # Print full summary
    print(result.summary())

    # Extract key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print(f"  Is Normal: {result.is_normal}")
    print(f"  Recommended Distribution: {result.recommended_distribution}")
    if result.recommended_df:
        print(f"  Degrees of Freedom: {result.recommended_df}")
    print("=" * 80)


def example_heavy_tailed_returns():
    """Example 2: Heavy-tailed returns (realistic)."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Heavy-Tailed Returns (Realistic)")
    print("=" * 80)

    # Simulate Student's t returns (df=5) - heavier tails than normal
    returns = np.random.standard_t(df=5, size=1000) * 0.02 + 0.001

    # Comprehensive analysis
    result = analyze_distribution(returns, alpha=0.05, compute_tails=True)

    # Print full summary
    print(result.summary())

    # Risk management implications
    print("\n" + "=" * 80)
    print("RISK MANAGEMENT DECISIONS:")
    if result.recommended_distribution == "t":
        print(f"  Use Student's t (df={result.recommended_df}) for:")
        print("    - VaR estimation")
        print("    - Monte Carlo simulations")
        print("    - Risk scenario analysis")
    elif result.recommended_distribution in ["stable", "heavy-tailed"]:
        print("  Use extreme value theory:")
        print("    - CVaR instead of VaR")
        print("    - Tail risk hedging")
        print("    - Robust portfolio optimization")
    print("=" * 80)


if __name__ == "__main__":
    # Run examples
    example_normal_returns()
    example_heavy_tailed_returns()
