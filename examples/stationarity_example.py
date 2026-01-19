"""Example usage of stationarity testing with ADF, KPSS, and PP tests.

This script demonstrates how to use stationarity tests to assess time series data:

1. Augmented Dickey-Fuller (ADF) - parametric unit root test
2. KPSS - tests null of stationarity
3. Phillips-Perron (PP) - robust non-parametric alternative to ADF

The script shows:
- Testing white noise (stationary)
- Testing random walk (non-stationary)
- Testing trending series
- Different regression specifications
- Comparing ADF vs PP on heteroscedastic data
- Complementary testing strategies

Run this script:
    python examples/stationarity_example.py
"""

import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation.stationarity import adf_test, kpss_test

# Try to import pp_test (requires arch package)
try:
    from ml4t.diagnostic.evaluation.stationarity import pp_test

    HAS_PP = True
except ImportError:
    HAS_PP = False
    print("Warning: arch package not installed - PP test examples will be skipped")
    print("Install with: pip install arch or pip install ml4t-evaluation[advanced]\n")

# Set random seed for reproducibility
np.random.seed(42)


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def example_white_noise() -> None:
    """Example 1: Test white noise (stationary process)."""
    print_section("Example 1: White Noise (Stationary)")

    # Generate white noise
    white_noise = np.random.randn(1000)

    print("Testing white noise process: X_t ~ N(0, 1)")
    print(f"Sample size: {len(white_noise)}")
    print(f"Mean: {np.mean(white_noise):.4f}")
    print(f"Std Dev: {np.std(white_noise):.4f}")

    # Run ADF test with default parameters
    result = adf_test(white_noise)

    print("\n" + result.summary())


def example_random_walk() -> None:
    """Example 2: Test random walk (non-stationary process)."""
    print_section("Example 2: Random Walk (Non-Stationary)")

    # Generate random walk
    innovations = np.random.randn(1000)
    random_walk = np.cumsum(innovations)

    print("Testing random walk process: X_t = X_{t-1} + ε_t")
    print(f"Sample size: {len(random_walk)}")
    print(f"Starting value: {random_walk[0]:.4f}")
    print(f"Ending value: {random_walk[-1]:.4f}")

    # Run ADF test
    result = adf_test(random_walk)

    print("\n" + result.summary())


def example_mean_reverting() -> None:
    """Example 3: Test mean-reverting AR(1) process."""
    print_section("Example 3: Mean-Reverting AR(1) Process")

    # Generate AR(1) with phi < 1 (stationary)
    n = 1000
    phi = 0.8  # Autoregressive coefficient
    data = np.zeros(n)
    data[0] = np.random.randn()

    for t in range(1, n):
        data[t] = phi * data[t - 1] + np.random.randn()

    print(f"Testing AR(1) process: X_t = {phi} * X_{{t-1}} + ε_t")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Autocorr(lag=1): {np.corrcoef(data[:-1], data[1:])[0, 1]:.4f}")

    result = adf_test(data)

    print("\n" + result.summary())


def example_trending_series() -> None:
    """Example 4: Test series with linear trend."""
    print_section("Example 4: Series with Linear Trend")

    # Generate series with trend
    n = 1000
    t = np.arange(n)
    trend = 0.05 * t
    noise = np.random.randn(n)
    data = trend + noise

    print("Testing series: X_t = 0.05*t + ε_t")
    print(f"Sample size: {len(data)}")
    print(f"Starting value: {data[0]:.4f}")
    print(f"Ending value: {data[-1]:.4f}")

    # Test with constant only (may fail to detect stationarity)
    print("\n--- Test with Constant Only ('c') ---")
    result_c = adf_test(data, regression="c")
    print(f"Test Statistic: {result_c.test_statistic:.4f}")
    print(f"P-value: {result_c.p_value:.4f}")
    print(f"Conclusion: {'Stationary' if result_c.is_stationary else 'Non-stationary'}")

    # Test with constant and trend (should detect trend-stationarity)
    print("\n--- Test with Constant and Trend ('ct') ---")
    result_ct = adf_test(data, regression="ct")
    print(f"Test Statistic: {result_ct.test_statistic:.4f}")
    print(f"P-value: {result_ct.p_value:.4f}")
    print(f"Conclusion: {'Stationary' if result_ct.is_stationary else 'Non-stationary'}")

    print(
        "\nNote: Series is trend-stationary. The 'ct' specification correctly "
        "identifies this, while 'c' may not."
    )


def example_regression_types() -> None:
    """Example 5: Compare different regression specifications."""
    print_section("Example 5: Different Regression Specifications")

    # Generate stationary data
    data = np.random.randn(1000)

    print("Testing the same stationary series with different specifications:\n")

    # Test all regression types
    for regression in ["c", "ct", "ctt", "n"]:
        result = adf_test(data, regression=regression)

        print(f"Regression: {regression:>3s}")
        print(f"  Test Statistic: {result.test_statistic:>8.4f}")
        print(f"  P-value:        {result.p_value:>8.4f}")
        print(f"  Lags Used:      {result.lags_used:>8d}")
        print(f"  Stationary:     {str(result.is_stationary):>8s}")
        print()

    print("Regression types:")
    print("  'c':   Constant only (default)")
    print("  'ct':  Constant and linear trend")
    print("  'ctt': Constant, linear and quadratic trend")
    print("  'n':   No constant or trend")


def example_lag_selection() -> None:
    """Example 6: Compare lag selection methods."""
    print_section("Example 6: Lag Selection Methods")

    # Generate stationary data
    data = np.random.randn(1000)

    print("Testing the same series with different lag selection methods:\n")

    # Test different autolag methods
    for autolag in ["AIC", "BIC", "t-stat"]:
        result = adf_test(data, autolag=autolag)

        print(f"Autolag: {autolag}")
        print(f"  Lags Selected:  {result.lags_used}")
        print(f"  Test Statistic: {result.test_statistic:.4f}")
        print(f"  P-value:        {result.p_value:.4f}")
        print()

    # Manual lag specification
    result = adf_test(data, maxlag=15, autolag=None)
    print("Manual Lag Specification (maxlag=15):")
    print(f"  Lags Used:      {result.lags_used}")
    print(f"  Test Statistic: {result.test_statistic:.4f}")
    print(f"  P-value:        {result.p_value:.4f}")

    print("\nNote: BIC typically selects fewer lags than AIC (penalizes complexity more)")


def example_pandas_series() -> None:
    """Example 7: Using pandas Series with datetime index."""
    print_section("Example 7: Pandas Series with Datetime Index")

    # Create time series with datetime index
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    values = np.random.randn(1000)
    series = pd.Series(values, index=dates)

    print("Testing pandas Series with DatetimeIndex:")
    print(f"Start date: {series.index[0]}")
    print(f"End date:   {series.index[-1]}")
    print("Frequency:  Daily")
    print(f"Sample size: {len(series)}")

    result = adf_test(series)

    print("\n" + result.summary())


def example_financial_returns() -> None:
    """Example 8: Realistic financial returns example."""
    print_section("Example 8: Simulated Financial Returns")

    # Simulate daily returns (stationary)
    n_days = 1000
    daily_returns = np.random.randn(n_days) * 0.01  # 1% daily volatility

    # Create price series (non-stationary)
    prices = 100 * np.exp(np.cumsum(daily_returns))

    print("Financial Time Series Example:")
    print(f"Trading days: {n_days}")
    print(f"Initial price: ${prices[0]:.2f}")
    print(f"Final price:   ${prices[-1]:.2f}")
    print(f"Mean return:   {np.mean(daily_returns) * 100:.4f}%")
    print(f"Return vol:    {np.std(daily_returns) * 100:.4f}%")

    # Test prices (should be non-stationary)
    print("\n--- Testing Price Series ---")
    result_prices = adf_test(prices, regression="c")
    print(f"Test Statistic: {result_prices.test_statistic:.4f}")
    print(f"P-value: {result_prices.p_value:.4f}")
    print(f"Conclusion: {'Stationary' if result_prices.is_stationary else 'Non-stationary'}")

    # Test returns (should be stationary)
    print("\n--- Testing Return Series ---")
    result_returns = adf_test(daily_returns, regression="c")
    print(f"Test Statistic: {result_returns.test_statistic:.4f}")
    print(f"P-value: {result_returns.p_value:.4f}")
    print(f"Conclusion: {'Stationary' if result_returns.is_stationary else 'Non-stationary'}")

    print(
        "\nKey insight: Prices are typically non-stationary (random walk), "
        "while returns are stationary."
    )


def example_critical_values() -> None:
    """Example 9: Understanding critical values."""
    print_section("Example 9: Understanding Critical Values")

    # Generate borderline stationary series
    data = np.random.randn(1000)
    result = adf_test(data)

    print("Critical Values Interpretation:")
    print(f"\nTest Statistic: {result.test_statistic:.4f}")
    print("\nCritical Values:")

    for level, value in sorted(result.critical_values.items()):
        reject = result.test_statistic < value
        symbol = "✓ REJECT" if reject else "✗ FAIL TO REJECT"
        print(f"  {level:>4s}: {value:>8.4f}  {symbol}")

    print(f"\nP-value: {result.p_value:.4f}")
    print(f"\nConclusion at 5% level: {'Stationary' if result.is_stationary else 'Non-stationary'}")

    print("\nInterpretation:")
    print("- More negative test statistic = stronger evidence against unit root")
    print("- Test statistic < critical value => reject null hypothesis (stationary)")
    print("- P-value < 0.05 => reject null hypothesis at 5% level (stationary)")


def example_kpss_white_noise() -> None:
    """Example 10: KPSS test on white noise."""
    print_section("Example 10: KPSS Test on White Noise")

    # Generate white noise
    white_noise = np.random.randn(1000)

    print("Testing white noise with KPSS test")
    print(f"Sample size: {len(white_noise)}")
    print(f"Mean: {np.mean(white_noise):.4f}")
    print(f"Std Dev: {np.std(white_noise):.4f}")

    # Run KPSS test
    result = kpss_test(white_noise)

    print("\n" + result.summary())


def example_kpss_random_walk() -> None:
    """Example 11: KPSS test on random walk."""
    print_section("Example 11: KPSS Test on Random Walk")

    # Generate random walk
    random_walk = np.cumsum(np.random.randn(1000))

    print("Testing random walk with KPSS test")
    print(f"Sample size: {len(random_walk)}")

    result = kpss_test(random_walk)

    print("\n" + result.summary())


def example_complementary_testing() -> None:
    """Example 12: Using ADF and KPSS together."""
    print_section("Example 12: Complementary ADF + KPSS Testing")

    print("Demonstrating complementary use of ADF and KPSS tests.")
    print("Both tests should be used together for robust stationarity assessment.\n")

    # Test 1: White noise (stationary)
    print("--- Test 1: White Noise (Stationary) ---")
    white_noise = np.random.randn(1000)

    adf_wn = adf_test(white_noise)
    kpss_wn = kpss_test(white_noise)

    print(f"ADF:  p-value={adf_wn.p_value:.4f}, stationary={adf_wn.is_stationary}")
    print(f"KPSS: p-value={kpss_wn.p_value:.4f}, stationary={kpss_wn.is_stationary}")
    print(f"Agreement: {'YES' if adf_wn.is_stationary == kpss_wn.is_stationary else 'NO'}")
    print("Expected: Both agree it's stationary (ADF rejects, KPSS fails to reject)\n")

    # Test 2: Random walk (non-stationary)
    print("--- Test 2: Random Walk (Non-Stationary) ---")
    random_walk = np.cumsum(np.random.randn(1000))

    adf_rw = adf_test(random_walk)
    kpss_rw = kpss_test(random_walk)

    print(f"ADF:  p-value={adf_rw.p_value:.4f}, stationary={adf_rw.is_stationary}")
    print(f"KPSS: p-value={kpss_rw.p_value:.4f}, stationary={kpss_rw.is_stationary}")
    print(f"Agreement: {'YES' if adf_rw.is_stationary == kpss_rw.is_stationary else 'NO'}")
    print("Expected: Both agree it's non-stationary (ADF fails to reject, KPSS rejects)\n")

    # Test 3: Trending series
    print("--- Test 3: Trending Series ---")
    t = np.arange(1000)
    trending = 0.05 * t + np.random.randn(1000)

    print("With constant only ('c'):")
    adf_c = adf_test(trending, regression="c")
    kpss_c = kpss_test(trending, regression="c")
    print(f"  ADF:  p-value={adf_c.p_value:.4f}, stationary={adf_c.is_stationary}")
    print(f"  KPSS: p-value={kpss_c.p_value:.4f}, stationary={kpss_c.is_stationary}")

    print("\nWith trend ('ct'):")
    adf_ct = adf_test(trending, regression="ct")
    kpss_ct = kpss_test(trending, regression="ct")
    print(f"  ADF:  p-value={adf_ct.p_value:.4f}, stationary={adf_ct.is_stationary}")
    print(f"  KPSS: p-value={kpss_ct.p_value:.4f}, stationary={kpss_ct.is_stationary}")
    print("Expected: With 'ct', both should detect trend-stationarity\n")


def example_kpss_lag_selection() -> None:
    """Example 13: KPSS lag selection methods."""
    print_section("Example 13: KPSS Lag Selection")

    data = np.random.randn(1000)

    print("Testing the same series with different lag selection methods:\n")

    # Test different nlags options
    for nlags in ["auto", "legacy", 10]:
        result = kpss_test(data, nlags=nlags)

        print(f"nlags={nlags!r}")
        print(f"  Lags Used:      {result.lags_used}")
        print(f"  Test Statistic: {result.test_statistic:.4f}")
        print(f"  P-value:        {result.p_value:.4f}")
        print()

    print("Note: 'auto' uses 12*(n/100)^0.25, 'legacy' uses 4*(n/100)^0.25")


def example_interpretation_guide() -> None:
    """Example 14: Guide to interpreting ADF and KPSS results."""
    print_section("Example 14: Interpretation Guide")

    print("Interpreting ADF and KPSS Results Together:\n")

    print("Case 1: ADF rejects + KPSS fails to reject")
    print("  => Strong evidence for STATIONARITY")
    print("  Example: ADF p=0.001, KPSS p=0.10\n")

    print("Case 2: ADF fails to reject + KPSS rejects")
    print("  => Strong evidence for NON-STATIONARITY")
    print("  Example: ADF p=0.80, KPSS p=0.01\n")

    print("Case 3: Both reject")
    print("  => Inconclusive (quasi-stationary, structural breaks?)")
    print("  Example: ADF p=0.02, KPSS p=0.02")
    print("  Action: Investigate further, check for regime changes\n")

    print("Case 4: Both fail to reject")
    print("  => Inconclusive (insufficient power, near unit root?)")
    print("  Example: ADF p=0.08, KPSS p=0.08")
    print("  Action: Use longer time series or more powerful tests\n")

    print("Key Differences:")
    print("  ADF:  H0 = non-stationary (unit root)")
    print("        Low p-value (<0.05) => reject H0 => STATIONARY")
    print("  KPSS: H0 = stationary")
    print("        High p-value (≥0.05) => fail to reject H0 => STATIONARY\n")

    print("Remember: Tests have opposite null hypotheses!")


def example_pp_white_noise() -> None:
    """Example 14: PP test on white noise."""
    if not HAS_PP:
        return

    print_section("Example 14: Phillips-Perron Test - White Noise")

    white_noise = np.random.randn(1000)

    print("Comparing ADF and PP tests on white noise:")
    print(f"Sample size: {len(white_noise)}")
    print()

    # Run both tests
    adf_result = adf_test(white_noise)
    pp_result = pp_test(white_noise)

    print("ADF Test:")
    print(f"  Test statistic: {adf_result.test_statistic:.4f}")
    print(f"  P-value: {adf_result.p_value:.4f}")
    print(f"  Stationary: {adf_result.is_stationary}")
    print()

    print("PP Test:")
    print(f"  Test statistic: {pp_result.test_statistic:.4f}")
    print(f"  P-value: {pp_result.p_value:.4f}")
    print(f"  Stationary: {pp_result.is_stationary}")
    print()

    print("Both tests should agree that white noise is stationary.")


def example_pp_heteroscedastic() -> None:
    """Example 15: PP vs ADF on heteroscedastic data."""
    if not HAS_PP:
        return

    print_section("Example 15: PP vs ADF - Heteroscedastic Data")

    # Generate heteroscedastic white noise (GARCH-like)
    n = 1000
    volatility = 1 + 0.5 * np.abs(np.random.randn(n))
    data = np.random.randn(n) * volatility

    print("Testing heteroscedastic white noise:")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Volatility range: [{volatility.min():.2f}, {volatility.max():.2f}]")
    print()

    # Run both tests
    adf_result = adf_test(data)
    pp_result = pp_test(data)

    print("ADF Test (parametric, assumes homoscedasticity):")
    print(f"  Test statistic: {adf_result.test_statistic:.4f}")
    print(f"  P-value: {adf_result.p_value:.4f}")
    print(f"  Stationary: {adf_result.is_stationary}")
    print()

    print("PP Test (non-parametric, robust to heteroscedasticity):")
    print(f"  Test statistic: {pp_result.test_statistic:.4f}")
    print(f"  P-value: {pp_result.p_value:.4f}")
    print(f"  Stationary: {pp_result.is_stationary}")
    print()

    print("PP is more reliable when data exhibits volatility clustering")
    print("(common in financial returns). Both should still agree on")
    print("stationarity, but PP handles the heteroscedasticity better.")


def example_pp_test_types() -> None:
    """Example 16: Different PP test types (tau vs rho)."""
    if not HAS_PP:
        return

    print_section("Example 16: PP Test Types - Tau vs Rho")

    data = np.random.randn(1000)

    print("PP offers two test types:")
    print()

    # Tau test (default, recommended)
    tau_result = pp_test(data, test_type="tau")
    print("Tau test (t-statistic based, default):")
    print(f"  Test statistic: {tau_result.test_statistic:.4f}")
    print(f"  P-value: {tau_result.p_value:.4f}")
    print(f"  Stationary: {tau_result.is_stationary}")
    print()

    # Rho test
    rho_result = pp_test(data, test_type="rho")
    print("Rho test (regression coefficient based):")
    print(f"  Test statistic: {rho_result.test_statistic:.4f}")
    print(f"  P-value: {rho_result.p_value:.4f}")
    print(f"  Stationary: {rho_result.is_stationary}")
    print()

    print("Tau test is generally recommended and more commonly used.")
    print("Both should give similar conclusions on stationarity.")


def example_pp_vs_adf_comparison() -> None:
    """Example 17: Detailed ADF vs PP comparison."""
    if not HAS_PP:
        return

    print_section("Example 17: ADF vs PP - Detailed Comparison")

    print("Key Differences:")
    print()
    print("ADF (Augmented Dickey-Fuller):")
    print("  - Parametric test with multiple lagged differences")
    print("  - Assumes homoscedastic errors")
    print("  - Lag selection affects results")
    print()
    print("PP (Phillips-Perron):")
    print("  - Non-parametric correction using Newey-West estimator")
    print("  - Robust to heteroscedasticity and serial correlation")
    print("  - Uses only 1 lag in regression")
    print("  - More appropriate for financial data with volatility clustering")
    print()

    # Test on stationary AR(1)
    n = 1000
    phi = 0.7
    ar_data = np.zeros(n)
    ar_data[0] = np.random.randn()
    for t in range(1, n):
        ar_data[t] = phi * ar_data[t - 1] + np.random.randn()

    print(f"Testing AR(1) with φ = {phi}:")
    print()

    adf_result = adf_test(ar_data)
    pp_result = pp_test(ar_data)

    print(
        f"ADF: stat={adf_result.test_statistic:.4f}, p={adf_result.p_value:.4f}, "
        f"stationary={adf_result.is_stationary}"
    )
    print(
        f"PP:  stat={pp_result.test_statistic:.4f}, p={pp_result.p_value:.4f}, "
        f"stationary={pp_result.is_stationary}"
    )
    print()
    print("Both tests should agree on stationarity for clean data.")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" ML4T Evaluation Stationarity Testing Examples")
    print(" ADF, KPSS, and Phillips-Perron Tests")
    print("=" * 80)

    # ADF examples
    example_white_noise()
    example_random_walk()
    example_mean_reverting()
    example_trending_series()
    example_regression_types()
    example_lag_selection()
    example_pandas_series()
    example_financial_returns()
    example_critical_values()

    # KPSS examples
    example_kpss_white_noise()
    example_kpss_random_walk()
    example_complementary_testing()
    example_kpss_lag_selection()
    example_interpretation_guide()

    # PP examples (if arch package available)
    if HAS_PP:
        example_pp_white_noise()
        example_pp_heteroscedastic()
        example_pp_test_types()
        example_pp_vs_adf_comparison()
    else:
        print_section("Phillips-Perron Examples Skipped")
        print("The arch package is not installed. PP test examples are skipped.")
        print("Install with: pip install arch or pip install ml4t-evaluation[advanced]")

    # Summary
    print_section("Summary")
    print("Key Takeaways:")
    print()
    print("1. ADF tests H0: unit root (non-stationary)")
    print("   - Reject H0 (p < 0.05) => series is stationary")
    print("   - Fail to reject H0 => series is non-stationary")
    print()
    print("2. KPSS tests H0: stationarity")
    print("   - Fail to reject H0 (p >= 0.05) => series is stationary")
    print("   - Reject H0 (p < 0.05) => series is non-stationary")
    print()
    print("3. PP tests H0: unit root (same as ADF)")
    print("   - More robust to heteroscedasticity than ADF")
    print("   - Useful for financial data with volatility clustering")
    print("   - Requires arch package")
    print()
    print("4. Use multiple tests together for robust assessment:")
    print("   - Stationary: ADF/PP rejects + KPSS fails to reject")
    print("   - Non-stationary: ADF/PP fails to reject + KPSS rejects")
    print("   - Quasi-stationary: Both reject or both fail to reject")
    print()
    print("5. Use appropriate regression specification:")
    print("   - 'c' (constant/level) for most financial returns")
    print("   - 'ct' (constant + trend) for trending series")
    print()
    print("6. Financial applications:")
    print("   - Prices: Typically non-stationary (random walk)")
    print("   - Returns: Typically stationary")
    print("   - Use PP for volatility clustering (GARCH effects)")
    print("   - Always test with complementary methods")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
