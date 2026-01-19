"""Example usage of FeatureDiagnostics for comprehensive feature analysis.

This example demonstrates how to use the FeatureDiagnostics class to perform
comprehensive statistical analysis on trading features/signals.
"""

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation import FeatureDiagnostics, FeatureDiagnosticsConfig

# Set random seed for reproducibility
np.random.seed(42)


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("=" * 80)
    print("Example 1: Basic Feature Diagnostics (White Noise)")
    print("=" * 80)

    # Generate white noise (ideal feature)
    feature = np.random.randn(1000)

    # Create diagnostics with default config
    diagnostics = FeatureDiagnostics()

    # Run diagnostics
    result = diagnostics.run_diagnostics(feature, name="white_noise_signal")

    # Display results
    print(result.summary())
    print("\n" + "-" * 80)
    print("Summary DataFrame:")
    print(result.summary_df)
    print()


def example_2_nonstationary_feature():
    """Example 2: Diagnosing non-stationary feature (random walk)."""
    print("=" * 80)
    print("Example 2: Non-Stationary Feature (Random Walk)")
    print("=" * 80)

    # Generate random walk (non-stationary)
    feature = np.cumsum(np.random.randn(1000))

    diagnostics = FeatureDiagnostics()
    result = diagnostics.run_diagnostics(feature, name="random_walk")

    # Check specific properties
    print(result.summary())
    print(f"\nStationarity Consensus: {result.stationarity.consensus}")
    print(f"Flags Raised: {result.flags}")
    print()


def example_3_autocorrelated_feature():
    """Example 3: Feature with autocorrelation (AR(1) process)."""
    print("=" * 80)
    print("Example 3: Autocorrelated Feature (AR(1))")
    print("=" * 80)

    # Generate AR(1) process
    n = 1000
    phi = 0.7
    feature = np.zeros(n)
    noise = np.random.randn(n)

    for i in range(1, n):
        feature[i] = phi * feature[i - 1] + noise[i]

    diagnostics = FeatureDiagnostics()
    result = diagnostics.run_diagnostics(feature, name="ar1_signal")

    print(result.summary())
    print(f"\nSuggested ARIMA order: {result.autocorrelation.suggested_arima_order}")
    print(f"Significant ACF lags: {result.autocorrelation.significant_acf_lags[:10]}")  # First 10
    print()


def example_4_volatility_clustering():
    """Example 4: Feature with volatility clustering (GARCH process)."""
    print("=" * 80)
    print("Example 4: Volatility Clustering (GARCH)")
    print("=" * 80)

    # Generate GARCH(1,1) process
    n = 1000
    omega = 0.1
    alpha = 0.3
    beta = 0.6

    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for i in range(1, n):
        sigma2[i] = omega + alpha * returns[i - 1] ** 2 + beta * sigma2[i - 1]
        returns[i] = np.sqrt(sigma2[i]) * np.random.randn()

    diagnostics = FeatureDiagnostics()
    result = diagnostics.run_diagnostics(returns, name="garch_returns")

    print(result.summary())
    if result.volatility.garch_result:
        print("\nGARCH Model: (1,1)")  # Currently always GARCH(1,1)
        print(f"Converged: {result.volatility.garch_result.converged}")
        print(f"Persistence: {result.volatility.persistence:.3f}")
    print()


def example_5_custom_configuration():
    """Example 5: Custom configuration (selective diagnostics)."""
    print("=" * 80)
    print("Example 5: Custom Configuration (Stationarity & Distribution Only)")
    print("=" * 80)

    # Generate Student-t distributed data (heavy tails)
    feature = np.random.standard_t(df=3, size=1000)

    # Custom config: only run stationarity and distribution
    config = FeatureDiagnosticsConfig(
        run_stationarity=True,
        run_autocorrelation=False,
        run_volatility=False,
        run_distribution=True,
        compute_tails=True,  # Enable tail analysis
        verbose=False,
    )

    diagnostics = FeatureDiagnostics(config)
    result = diagnostics.run_diagnostics(feature, name="heavy_tail_signal")

    print(result.summary())
    print(f"\nRecommended Distribution: {result.distribution.recommended_distribution}")
    if result.distribution.tail_analysis_result:
        hill = result.distribution.tail_analysis_result.hill_result
        if hill:
            print(f"Hill Tail Index: {hill.tail_index:.2f}")
    print()


def example_6_batch_processing():
    """Example 6: Batch processing multiple features."""
    print("=" * 80)
    print("Example 6: Batch Processing Multiple Features")
    print("=" * 80)

    # Create multi-feature DataFrame
    n = 500
    df = pd.DataFrame(
        {
            "momentum": np.random.randn(n),  # White noise
            "mean_reversion": np.cumsum(np.random.randn(n)),  # Random walk
            "volatility": np.abs(np.random.randn(n)),  # Non-negative
        }
    )

    diagnostics = FeatureDiagnostics()
    results = diagnostics.run_batch_diagnostics(df)

    # Display health scores
    print("Feature Health Scores:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:20s}: {result.health_score:.2f} ({len(result.flags)} flags)")

    print("\nTop Recommendations by Feature:")
    print("-" * 40)
    for name, result in results.items():
        print(f"\n{name}:")
        for i, rec in enumerate(result.recommendations[:2], 1):  # Top 2
            print(f"  {i}. {rec}")

    print()


def example_7_comparing_transformations():
    """Example 7: Comparing feature transformations."""
    print("=" * 80)
    print("Example 7: Comparing Original vs Transformed Feature")
    print("=" * 80)

    # Original: Random walk (non-stationary)
    raw_feature = np.cumsum(np.random.randn(1000))

    # Transformed: First difference (stationary)
    diff_feature = np.diff(raw_feature)

    diagnostics = FeatureDiagnostics()

    # Diagnose both
    result_raw = diagnostics.run_diagnostics(raw_feature, name="raw_feature")
    result_diff = diagnostics.run_diagnostics(diff_feature, name="differenced_feature")

    print("Original Feature:")
    print(f"  Stationarity: {result_raw.stationarity.consensus}")
    print(f"  Health Score: {result_raw.health_score:.2f}")
    print(f"  Flags: {result_raw.flags}")

    print("\nDifferenced Feature:")
    print(f"  Stationarity: {result_diff.stationarity.consensus}")
    print(f"  Health Score: {result_diff.health_score:.2f}")
    print(f"  Flags: {result_diff.flags}")

    print(
        "\n✅ Transformation improved feature quality!"
        if result_diff.health_score > result_raw.health_score
        else "\n❌ Transformation did not improve feature quality"
    )
    print()


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_nonstationary_feature()
    example_3_autocorrelated_feature()
    example_4_volatility_clustering()
    example_5_custom_configuration()
    example_6_batch_processing()
    example_7_comparing_transformations()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
