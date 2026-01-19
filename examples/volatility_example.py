#!/usr/bin/env python3
"""Examples demonstrating ARCH-LM test for volatility clustering detection.

This script shows how to use the ARCH Lagrange Multiplier test to detect
conditional heteroscedasticity (volatility clustering) in time series data.

Key Concepts:
    - ARCH effects: Time-varying volatility (heteroscedasticity)
    - Volatility clustering: Large changes follow large changes
    - H0: No ARCH effects (constant variance)
    - H1: ARCH effects present (time-varying variance)

Usage:
    python examples/volatility_example.py
"""

import numpy as np

try:
    from arch import arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not installed. Some examples will be skipped.")
    print("Install with: pip install arch")

from ml4t.diagnostic.evaluation.volatility import analyze_volatility, arch_lm_test, fit_garch


def example_white_noise():
    """Example 1: White noise (no volatility clustering)."""
    print("\n" + "=" * 70)
    print("Example 1: White Noise (No ARCH Effects)")
    print("=" * 70)

    np.random.seed(42)
    # Generate white noise (constant variance)
    white_noise = np.random.randn(1000)

    print(f"\nGenerated {len(white_noise)} observations of white noise")
    print(f"Mean: {np.mean(white_noise):.4f}")
    print(f"Std Dev: {np.std(white_noise):.4f}")

    # Run ARCH-LM test
    result = arch_lm_test(white_noise, lags=12, demean=True)

    print("\n" + result.summary())

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if not result.has_arch_effects:
        print("✓ As expected, white noise shows NO volatility clustering")
        print("✓ Constant variance assumption is valid")
        print("✓ Standard statistical methods are appropriate")
    else:
        print("⚠ Unexpected: white noise showing ARCH effects (may be chance)")


def example_garch_process():
    """Example 2: GARCH(1,1) process (strong volatility clustering)."""
    print("\n" + "=" * 70)
    print("Example 2: GARCH(1,1) Process (Strong ARCH Effects)")
    print("=" * 70)

    np.random.seed(42)

    # Manually simulate GARCH(1,1) process
    # sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    print("\nSimulating GARCH(1,1) process:")
    print("  sigma_t^2 = 0.01 + 0.1 * eps_{t-1}^2 + 0.85 * sigma_{t-1}^2")
    print("  (omega=0.01, alpha=0.1, beta=0.85)")

    n = 1000
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance

    for t in range(1, n):
        sigma2[t] = (
            omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
        )

    garch_data = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(garch_data)} observations")
    print(f"Mean: {np.mean(garch_data):.4f}")
    print(f"Std Dev: {np.std(garch_data):.4f}")

    # Run ARCH-LM test
    result = arch_lm_test(garch_data, lags=12, demean=True)

    print("\n" + result.summary())

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if result.has_arch_effects:
        print("✓ As expected, GARCH process shows ARCH effects")
        print("✓ Volatility clustering detected")
        print("✓ GARCH/EGARCH models appropriate for forecasting")
        print("✓ Standard errors need HAC correction")
    else:
        print("⚠ Unexpected: GARCH not showing ARCH effects")


def example_arch_process():
    """Example 3: ARCH(1) process (moderate volatility clustering)."""
    print("\n" + "=" * 70)
    print("Example 3: ARCH(1) Process (Moderate ARCH Effects)")
    print("=" * 70)

    np.random.seed(42)

    # Manually simulate ARCH(1) process
    # sigma_t^2 = omega + alpha * eps_{t-1}^2
    print("\nSimulating ARCH(1) process:")
    print("  sigma_t^2 = 0.01 + 0.5 * eps_{t-1}^2")
    print("  (omega=0.01, alpha=0.5)")

    n = 1000
    omega = 0.01
    alpha = 0.5

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha)  # Unconditional variance

    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2

    arch_data = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(arch_data)} observations")
    print(f"Mean: {np.mean(arch_data):.4f}")
    print(f"Std Dev: {np.std(arch_data):.4f}")

    # Run ARCH-LM test with different lags
    for lags in [6, 12, 24]:
        print(f"\n--- Testing with {lags} lags ---")
        result = arch_lm_test(arch_data, lags=lags, demean=True)
        print(f"Test Statistic: {result.test_statistic:.4f}")
        print(f"P-value: {result.p_value:.4f}")
        print(
            f"Conclusion: {'ARCH effects detected' if result.has_arch_effects else 'No ARCH effects'}"
        )


def example_simulated_returns():
    """Example 4: Simulated returns with volatility clustering."""
    print("\n" + "=" * 70)
    print("Example 4: Simulated Returns with Volatility Clustering")
    print("=" * 70)

    np.random.seed(42)

    # Simulate returns with time-varying volatility
    n = 1000
    eps = np.random.randn(n)
    sigma = np.ones(n)

    print("\nSimulating returns with volatility clustering:")
    print("  sigma_t^2 = 0.01 + 0.05 * eps_{t-1}^2 + 0.9 * sigma_{t-1}^2")

    # Generate volatility process
    for i in range(1, n):
        sigma[i] = np.sqrt(0.01 + 0.05 * eps[i - 1] ** 2 + 0.9 * sigma[i - 1] ** 2)

    returns = sigma * eps

    print(f"\nGenerated {len(returns)} daily returns")
    print(f"Mean: {np.mean(returns):.4f}")
    print(f"Std Dev: {np.std(returns):.4f}")
    print(f"Skewness: {_compute_skewness(returns):.4f}")
    print(f"Kurtosis: {_compute_kurtosis(returns):.4f}")

    # Run ARCH-LM test
    result = arch_lm_test(returns, lags=12, demean=True)

    print("\n" + result.summary())

    # Additional analysis with different lags
    print("\n" + "-" * 70)
    print("Robustness Check: Testing with different lag specifications")
    print("-" * 70)

    lag_specs = [6, 12, 24, 36]
    for lags in lag_specs:
        result = arch_lm_test(returns, lags=lags, demean=True)
        status = "✓ ARCH" if result.has_arch_effects else "✗ No ARCH"
        print(f"Lags={lags:2d}: {status}  (p={result.p_value:.4f})")


def example_demean_comparison():
    """Example 5: Compare demean=True vs demean=False."""
    print("\n" + "=" * 70)
    print("Example 5: Demeaning Comparison")
    print("=" * 70)

    np.random.seed(42)

    # Generate data with non-zero mean
    data = np.random.randn(1000) + 2.0

    print("\nGenerated data with non-zero mean")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std Dev: {np.std(data):.4f}")

    # Test with demeaning
    print("\n--- With demeaning (demean=True) ---")
    result_demean = arch_lm_test(data, lags=12, demean=True)
    print(f"Test Statistic: {result_demean.test_statistic:.4f}")
    print(f"P-value: {result_demean.p_value:.4f}")
    print(f"ARCH effects: {'Yes' if result_demean.has_arch_effects else 'No'}")

    # Test without demeaning
    print("\n--- Without demeaning (demean=False) ---")
    result_no_demean = arch_lm_test(data, lags=12, demean=False)
    print(f"Test Statistic: {result_no_demean.test_statistic:.4f}")
    print(f"P-value: {result_no_demean.p_value:.4f}")
    print(f"ARCH effects: {'Yes' if result_no_demean.has_arch_effects else 'No'}")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("For returns (approximately zero mean), demean=True is standard")
    print("For other series, choice depends on whether mean is structural")


def example_practical_workflow():
    """Example 6: Practical workflow for volatility analysis."""
    print("\n" + "=" * 70)
    print("Example 6: Practical Workflow for Volatility Analysis")
    print("=" * 70)

    np.random.seed(42)

    # Simulate realistic daily returns with GARCH
    print("\nStep 1: Generate sample returns data")
    n = 1000
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = (
            omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
        )

    returns = eps * np.sqrt(sigma2)

    print(f"  Generated {len(returns)} daily returns")

    # Step 1: Test for ARCH effects
    print("\nStep 2: Test for ARCH effects")
    result = arch_lm_test(returns, lags=12, demean=True)
    print(f"  Test Statistic: {result.test_statistic:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(
        f"  Conclusion: {'ARCH effects detected' if result.has_arch_effects else 'No ARCH effects'}"
    )

    # Step 2: Decision tree
    print("\nStep 3: Recommended next steps")
    if result.has_arch_effects:
        print("  ✓ ARCH effects detected - Recommendations:")
        print("    1. Fit GARCH/EGARCH model for volatility forecasting")
        print("    2. Use HAC standard errors for inference")
        print("    3. Consider volatility in risk models (VaR, CVaR)")
        print("    4. Account for clustering in trading strategies")
    else:
        print("  ✓ No ARCH effects - Recommendations:")
        print("    1. Constant variance assumption is valid")
        print("    2. Standard OLS methods appropriate")
        print("    3. Classical risk models acceptable")

    # Step 3: Lag sensitivity
    print("\nStep 4: Lag sensitivity analysis")
    print("  Testing robustness across lag specifications:")
    for lags in [6, 12, 18, 24]:
        r = arch_lm_test(returns, lags=lags, demean=True)
        print(
            f"    Lags={lags:2d}: p={r.p_value:.4f} {'(ARCH)' if r.has_arch_effects else '(No ARCH)'}"
        )


def _compute_skewness(data: np.ndarray) -> float:
    """Compute sample skewness."""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 3)


def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute sample excess kurtosis."""
    mean = np.mean(data)
    std = np.std(data)
    return np.mean(((data - mean) / std) ** 4) - 3


def example_garch_fitting():
    """Example 7: GARCH model fitting and parameter recovery."""
    if not HAS_ARCH:
        print("\n" + "=" * 70)
        print("Example 7: GARCH Model Fitting [SKIPPED - arch not installed]")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 7: GARCH(1,1) Model Fitting")
    print("=" * 70)

    np.random.seed(42)

    # Simulate GARCH(1,1) with known parameters
    print("\nSimulating GARCH(1,1) process:")
    print("  sigma_t^2 = 0.01 + 0.1 * eps_{t-1}^2 + 0.85 * sigma_{t-1}^2")
    print("  True parameters: omega=0.01, alpha=0.1, beta=0.85")
    print("  True persistence: 0.95")

    n = 1000
    omega_true = 0.01
    alpha_true = 0.1
    beta_true = 0.85

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega_true / (1 - alpha_true - beta_true)

    for t in range(1, n):
        sigma2[t] = (
            omega_true
            + alpha_true * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2
            + beta_true * sigma2[t - 1]
        )

    returns = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(returns)} observations")
    print(f"Mean: {np.mean(returns):.6f}")
    print(f"Std Dev: {np.std(returns):.6f}")

    # Step 1: Test for ARCH effects
    print("\n" + "-" * 70)
    print("Step 1: Test for ARCH effects")
    print("-" * 70)

    arch_result = arch_lm_test(returns, lags=12)
    print(f"ARCH-LM Test Statistic: {arch_result.test_statistic:.4f}")
    print(f"P-value: {arch_result.p_value:.4f}")
    print(f"ARCH effects detected: {'Yes' if arch_result.has_arch_effects else 'No'}")

    # Step 2: Fit GARCH model
    print("\n" + "-" * 70)
    print("Step 2: Fit GARCH(1,1) model")
    print("-" * 70)

    garch_result = fit_garch(returns, p=1, q=1)

    print("\n" + garch_result.summary())

    # Step 3: Compare estimated vs true parameters
    print("\n" + "-" * 70)
    print("Step 3: Parameter Recovery Analysis")
    print("-" * 70)

    print(f"\n{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Error':<12}")
    print("-" * 55)
    print(
        f"{'omega':<15} {omega_true:<12.6f} {garch_result.omega:<12.6f} {abs(garch_result.omega - omega_true):<12.6f}"
    )
    print(
        f"{'alpha':<15} {alpha_true:<12.6f} {garch_result.alpha:<12.6f} {abs(garch_result.alpha - alpha_true):<12.6f}"
    )
    print(
        f"{'beta':<15} {beta_true:<12.6f} {garch_result.beta:<12.6f} {abs(garch_result.beta - beta_true):<12.6f}"
    )
    persistence_true = alpha_true + beta_true
    print(
        f"{'persistence':<15} {persistence_true:<12.6f} {garch_result.persistence:<12.6f} {abs(garch_result.persistence - persistence_true):<12.6f}"
    )

    print("\n✓ Parameters recovered within reasonable tolerance")
    print("  (Some error is expected due to finite sample)")


def example_garch_conditional_volatility():
    """Example 8: Extracting and analyzing conditional volatility."""
    if not HAS_ARCH:
        print("\n" + "=" * 70)
        print("Example 8: Conditional Volatility [SKIPPED - arch not installed]")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 8: Conditional Volatility Analysis")
    print("=" * 70)

    np.random.seed(42)

    # Simulate GARCH with volatility clustering
    n = 1000
    omega = 0.01
    alpha = 0.15
    beta = 0.80

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = (
            omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
        )

    returns = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(returns)} observations with volatility clustering")

    # Fit GARCH
    garch_result = fit_garch(returns, p=1, q=1)

    print("\nFitted parameters:")
    print(f"  omega: {garch_result.omega:.6f}")
    print(f"  alpha: {garch_result.alpha:.6f}")
    print(f"  beta: {garch_result.beta:.6f}")
    print(f"  persistence: {garch_result.persistence:.4f}")

    # Analyze conditional volatility
    print("\n" + "-" * 70)
    print("Conditional Volatility Statistics")
    print("-" * 70)

    cond_vol = garch_result.conditional_volatility.values
    print(f"\nMean volatility:       {np.mean(cond_vol):.6f}")
    print(f"Std dev of volatility: {np.std(cond_vol):.6f}")
    print(f"Min volatility:        {np.min(cond_vol):.6f}")
    print(f"Max volatility:        {np.max(cond_vol):.6f}")
    print(f"Volatility range:      {np.max(cond_vol) - np.min(cond_vol):.6f}")

    # Compute volatility ratios
    ratio = np.max(cond_vol) / np.min(cond_vol)
    print(f"\nMax/Min ratio:         {ratio:.2f}x")
    print(f"  → Volatility varies by factor of {ratio:.1f} over sample period")

    # Analyze standardized residuals
    print("\n" + "-" * 70)
    print("Standardized Residuals Diagnostics")
    print("-" * 70)

    std_resid = garch_result.standardized_residuals.values
    print(f"\nMean:     {np.mean(std_resid):.6f}  (should be ≈ 0)")
    print(f"Std Dev:  {np.std(std_resid):.6f}  (should be ≈ 1)")
    print(f"Skewness: {_compute_skewness(std_resid):.4f}  (normal = 0)")
    print(f"Kurtosis: {_compute_kurtosis(std_resid):.4f}  (normal = 0)")

    if abs(np.mean(std_resid)) < 0.1 and 0.9 < np.std(std_resid) < 1.1:
        print("\n✓ Standardized residuals are well-behaved")
        print("  Model successfully captures volatility dynamics")


def example_garch_persistence():
    """Example 9: Understanding persistence and stationarity."""
    if not HAS_ARCH:
        print("\n" + "=" * 70)
        print("Example 9: Persistence Analysis [SKIPPED - arch not installed]")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 9: Persistence and Stationarity")
    print("=" * 70)

    np.random.seed(42)

    print("\nTesting three GARCH processes with different persistence:")

    configs = [
        {"name": "Low Persistence", "omega": 0.01, "alpha": 0.05, "beta": 0.70},
        {"name": "High Persistence", "omega": 0.01, "alpha": 0.10, "beta": 0.88},
        {
            "name": "Very High Persistence",
            "omega": 0.01,
            "alpha": 0.12,
            "beta": 0.87,
        },
    ]

    for config in configs:
        print("\n" + "-" * 70)
        print(f"{config['name']}")
        print("-" * 70)

        omega = config["omega"]
        alpha = config["alpha"]
        beta = config["beta"]
        persistence_true = alpha + beta  # type: ignore[operator]

        print("\nTrue parameters:")
        print(f"  omega: {omega:.4f}, alpha: {alpha:.4f}, beta: {beta:.4f}")
        print(f"  Persistence: {persistence_true:.4f}")

        # Simulate
        n = 1000
        eps = np.random.randn(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)  # type: ignore[operator,operator]

        for t in range(1, n):
            sigma2[t] = (
                omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
            )

        returns = eps * np.sqrt(sigma2)

        # Fit GARCH
        result = fit_garch(returns, p=1, q=1)

        print(f"\nEstimated persistence: {result.persistence:.4f}")

        # Compute unconditional variance
        uncond_var = result.omega / (1 - result.persistence)
        print(f"Unconditional variance: {uncond_var:.6f}")
        print(f"Unconditional volatility: {np.sqrt(uncond_var):.6f}")

        # Half-life of shocks
        if result.persistence < 1.0:
            half_life = np.log(0.5) / np.log(result.persistence)
            print(f"\nShock half-life: {half_life:.1f} periods")
            print(f"  → Volatility shocks decay 50% in {half_life:.1f} periods")
        else:
            print("\n⚠ Non-stationary: shocks do not decay")

        # Interpretation
        if result.persistence < 0.90:
            print("✓ Low persistence: Fast mean reversion")
        elif result.persistence < 0.95:
            print("✓ Moderate persistence: Typical for daily returns")
        elif result.persistence < 1.0:
            print("⚠ High persistence: Slow mean reversion")
        else:
            print("⚠ Non-stationary: Integrated GARCH (IGARCH)")


def example_garch_forecasting():
    """Example 10: Using GARCH for volatility forecasting."""
    if not HAS_ARCH:
        print("\n" + "=" * 70)
        print("Example 10: Volatility Forecasting [SKIPPED - arch not installed]")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 10: GARCH Volatility Forecasting")
    print("=" * 70)

    np.random.seed(42)

    # Simulate GARCH data
    n = 1000
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = (
            omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
        )

    returns = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(returns)} observations")

    # Fit GARCH
    result = fit_garch(returns, p=1, q=1)

    print("\nFitted GARCH(1,1) parameters:")
    print(f"  omega: {result.omega:.6f}")
    print(f"  alpha: {result.alpha:.6f}")
    print(f"  beta: {result.beta:.6f}")

    # Get current conditional volatility
    current_vol = result.conditional_volatility.iloc[-1]
    current_return = returns[-1]
    current_error = current_return / current_vol

    print("\nCurrent state (period T):")
    print(f"  Return: {current_return:.6f}")
    print(f"  Conditional volatility: {current_vol:.6f}")
    print(f"  Squared error: {current_error**2:.6f}")

    # Forecast volatility for next periods
    print("\n" + "-" * 70)
    print("Volatility Forecasts")
    print("-" * 70)

    # One-step ahead forecast
    # sigma_{T+1}^2 = omega + alpha * eps_T^2 + beta * sigma_T^2
    vol_forecast_1 = np.sqrt(
        result.omega + result.alpha * current_return**2 + result.beta * current_vol**2
    )

    print("\n1-step ahead (T+1):")
    print(f"  Forecast volatility: {vol_forecast_1:.6f}")

    # Multi-step ahead forecasts
    # sigma_{T+h}^2 = omega * (1 - (alpha+beta)^(h-1)) / (1 - alpha - beta)
    #                 + (alpha+beta)^(h-1) * sigma_{T+1}^2
    print("\nMulti-step forecasts:")
    horizons = [1, 2, 5, 10, 20]
    persistence = result.persistence
    uncond_vol = np.sqrt(result.omega / (1 - persistence))

    for h in horizons:
        if h == 1:
            vol_h = vol_forecast_1
        else:
            # Recursive formula
            vol_h_sq = uncond_vol**2 + (persistence ** (h - 1)) * (
                vol_forecast_1**2 - uncond_vol**2
            )
            vol_h = np.sqrt(vol_h_sq)

        print(f"  {h:2d}-step ahead: {vol_h:.6f}")

    print(f"\nUnconditional volatility: {uncond_vol:.6f}")
    print(f"  → Long-run forecast converges to {uncond_vol:.6f} as h → ∞")


def example_comprehensive_analysis():
    """Example 11: Comprehensive volatility analysis (unified function)."""
    if not HAS_ARCH:
        print("\n" + "=" * 70)
        print("Example 11: Comprehensive Analysis [SKIPPED - arch not installed]")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Example 11: Comprehensive Volatility Analysis")
    print("=" * 70)

    np.random.seed(42)

    # Simulate GARCH(1,1) process
    print("\nSimulating GARCH(1,1) process with known parameters:")
    print("  omega=0.01, alpha=0.1, beta=0.85")

    n = 1000
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    eps = np.random.randn(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = (
            omega + alpha * (eps[t - 1] * np.sqrt(sigma2[t - 1])) ** 2 + beta * sigma2[t - 1]
        )

    returns = eps * np.sqrt(sigma2)

    print(f"\nGenerated {len(returns)} observations")
    print(f"Mean: {np.mean(returns):.6f}")
    print(f"Std Dev: {np.std(returns):.6f}")

    # Run comprehensive analysis
    print("\n" + "-" * 70)
    print("Running comprehensive volatility analysis...")
    print("-" * 70)

    result = analyze_volatility(
        returns,
        arch_lags=12,
        fit_garch_model=True,
        alpha=0.05,
    )

    # Display full summary
    print("\n" + result.summary())

    # Additional checks
    print("\n" + "-" * 70)
    print("Additional Diagnostics")
    print("-" * 70)

    if result.has_volatility_clustering:
        print("\n✓ Volatility clustering confirmed")
        if result.persistence is not None:
            print(f"  Persistence: {result.persistence:.4f}")

            if result.persistence >= 0.99:
                print("  ⚠ WARNING: Very high persistence (near unit root)")
                print("  → Volatility shocks persist for very long periods")
            elif result.persistence >= 0.95:
                print("  → High persistence (typical for financial returns)")
                print("  → Volatility forecasting is important for risk management")

            # Compare estimated vs true parameters
            print("\n" + "-" * 70)
            print("Parameter Recovery (vs True Values)")
            print("-" * 70)
            print(f"\n{'Parameter':<15} {'True':<12} {'Estimated':<12} {'Error':<12}")
            print("-" * 55)
            if isinstance(result.garch_result.alpha, float):
                print(
                    f"{'alpha':<15} {alpha:<12.6f} {result.garch_result.alpha:<12.6f} "
                    f"{abs(result.garch_result.alpha - alpha):<12.6f}"
                )
            if isinstance(result.garch_result.beta, float):
                print(
                    f"{'beta':<15} {beta:<12.6f} {result.garch_result.beta:<12.6f} "
                    f"{abs(result.garch_result.beta - beta):<12.6f}"
                )
            persistence_true = alpha + beta
            print(
                f"{'persistence':<15} {persistence_true:<12.6f} {result.persistence:<12.6f} "
                f"{abs(result.persistence - persistence_true):<12.6f}"
            )
    else:
        print("\n✗ No volatility clustering detected")

    # Example with white noise for comparison
    print("\n" + "=" * 70)
    print("Comparison: White Noise (No Clustering Expected)")
    print("=" * 70)

    white_noise = np.random.randn(1000)
    wn_result = analyze_volatility(white_noise, arch_lags=12, fit_garch_model=True)

    print("\nARCH-LM Test:")
    print(f"  Test Statistic: {wn_result.arch_lm_result.test_statistic:.4f}")
    print(f"  P-value: {wn_result.arch_lm_result.p_value:.4f}")
    print(f"  ARCH effects: {'Yes' if wn_result.has_volatility_clustering else 'No'}")

    if wn_result.has_volatility_clustering:
        print("  ⚠ Unexpected ARCH effects (may be Type I error)")
    else:
        print("  ✓ As expected, no ARCH effects in white noise")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Volatility Analysis Examples")
    print("ARCH-LM Test & GARCH Model Fitting")
    print("=" * 70)

    # ARCH-LM test examples
    example_white_noise()
    example_garch_process()
    example_arch_process()
    example_simulated_returns()
    example_demean_comparison()
    example_practical_workflow()

    # GARCH fitting examples
    example_garch_fitting()
    example_garch_conditional_volatility()
    example_garch_persistence()
    example_garch_forecasting()

    # Comprehensive analysis example
    example_comprehensive_analysis()

    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. ARCH-LM test detects volatility clustering")
    print("  2. White noise → No ARCH effects")
    print("  3. GARCH/ARCH processes → ARCH effects detected")
    print("  4. Financial returns often show ARCH effects")
    print("  5. Use demean=True for return series")
    print("  6. Test robustness across different lag specifications")
    print("\nGARCH Modeling:")
    print("  7. GARCH(1,1) models time-varying volatility")
    print("  8. Persistence (α+β) measures shock decay rate")
    print("  9. Conditional volatility useful for risk management")
    print("  10. Multi-step forecasts converge to unconditional volatility")
    print("\nComprehensive Analysis:")
    print("  11. Use analyze_volatility() for unified workflow")
    print("  12. Combines ARCH-LM test and GARCH fitting")
    print("  13. Automatic interpretation and recommendations")
    print("  14. Handles edge cases gracefully")
    print("\nNext Steps:")
    print("  - If ARCH detected → Fit GARCH models")
    print("  - Use conditional volatility for VaR/CVaR")
    print("  - Account for volatility clustering in risk models")
    print("  - Consider Student's t for fat-tailed distributions")


if __name__ == "__main__":
    main()
