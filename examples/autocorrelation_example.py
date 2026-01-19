"""Example: Autocorrelation Function (ACF) Analysis.

This example demonstrates how to use compute_acf() to analyze different
time series patterns:
    1. White noise (no autocorrelation)
    2. AR(1) process (exponential decay)
    3. MA(1) process (single spike)
    4. Financial returns (weak autocorrelation)
    5. Absolute returns (volatility clustering)

The ACF measures correlation between a time series and its lagged values,
helping identify:
    - Market efficiency (returns should be uncorrelated)
    - Predictability and memory effects
    - Appropriate model orders (AR, MA)
    - Volatility clustering
"""

import matplotlib.pyplot as plt
import numpy as np

from ml4t.diagnostic.evaluation.autocorrelation import (
    analyze_autocorrelation,
    compute_acf,
    compute_pacf,
)


def plot_acf(result, title: str, ax=None):
    """Plot ACF with confidence intervals.

    Args:
        result: ACFResult object
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Plot ACF values
    ax.stem(result.lags, result.acf_values, linefmt="C0-", markerfmt="C0o", basefmt=" ")

    # Plot confidence intervals
    ci_lower = result.conf_int[:, 0]
    ci_upper = result.conf_int[:, 1]
    ax.fill_between(result.lags, ci_lower, ci_upper, alpha=0.2, color="gray")

    # Add horizontal line at 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add text with significant lags
    n_sig = len(result.significant_lags)
    ax.text(
        0.98,
        0.98,
        f"Significant lags: {n_sig}/{len(result.lags) - 1}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )


def example_white_noise():
    """Example 1: White noise (no autocorrelation)."""
    print("=" * 60)
    print("Example 1: White Noise (No Autocorrelation)")
    print("=" * 60)

    np.random.seed(42)
    white_noise = np.random.randn(1000)

    result = compute_acf(white_noise, nlags=40)

    print("\nWhite Noise ACF:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Lags analyzed: {len(result.lags) - 1}")
    print(f"  Significant lags: {len(result.significant_lags)}")
    print("  First 10 ACF values:")
    for i in range(min(10, len(result.acf_values))):
        sig = "*" if i in result.significant_lags else " "
        print(f"    Lag {i:2d}: {result.acf_values[i]:7.4f} {sig}")

    print("\nInterpretation:")
    print("  - ACF at lag 0 is always 1.0 (perfect correlation with itself)")
    print("  - All other lags should be near 0 (no correlation)")
    print("  - Expect ~5% false positives (2 lags out of 40)")
    print(f"  - Observed: {len(result.significant_lags)} significant lags")

    return result


def example_ar1_process():
    """Example 2: AR(1) process (exponential decay)."""
    print("\n" + "=" * 60)
    print("Example 2: AR(1) Process (Exponential Decay)")
    print("=" * 60)

    np.random.seed(42)
    phi = 0.7
    n = 1000

    # Generate AR(1): X_t = phi * X_{t-1} + epsilon_t
    noise = np.random.randn(n)
    ar1 = [noise[0]]
    for t in range(1, n):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    result = compute_acf(ar1, nlags=20)

    print(f"\nAR(1) with φ = {phi}:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {len(result.significant_lags)}")
    print("  ACF values (actual vs theoretical φ^k):")
    for k in range(min(10, len(result.acf_values))):
        theoretical = phi**k
        sig = "*" if k in result.significant_lags else " "
        print(f"    Lag {k:2d}: {result.acf_values[k]:7.4f} (theory: {theoretical:7.4f}) {sig}")

    print("\nInterpretation:")
    print("  - AR(1) ACF decays exponentially: ACF[k] ≈ φ^k")
    print("  - Persistent autocorrelation (many significant lags)")
    print("  - Memory effects: past values influence future")

    return result


def example_ma1_process():
    """Example 3: MA(1) process (single spike)."""
    print("\n" + "=" * 60)
    print("Example 3: MA(1) Process (Single Spike)")
    print("=" * 60)

    np.random.seed(42)
    theta = 0.5
    n = 1000

    # Generate MA(1): X_t = epsilon_t + theta * epsilon_{t-1}
    noise = np.random.randn(n + 1)
    ma1 = noise[1:] + theta * noise[:-1]

    result = compute_acf(ma1, nlags=20)

    print(f"\nMA(1) with θ = {theta}:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {result.significant_lags[:10]}")
    print("  ACF values:")

    theoretical_acf1 = theta / (1 + theta**2)
    for k in range(min(10, len(result.acf_values))):
        if k == 0:
            theoretical = 1.0
        elif k == 1:
            theoretical = theoretical_acf1
        else:
            theoretical = 0.0
        sig = "*" if k in result.significant_lags else " "
        print(f"    Lag {k:2d}: {result.acf_values[k]:7.4f} (theory: {theoretical:7.4f}) {sig}")

    print("\nInterpretation:")
    print("  - MA(1) ACF: single spike at lag 1, then zero")
    print(f"  - Theory: ACF[1] = θ/(1+θ²) = {theoretical_acf1:.4f}")
    print("  - Only immediate past matters (short memory)")

    return result


def example_financial_returns():
    """Example 4: Financial returns (weak autocorrelation)."""
    print("\n" + "=" * 60)
    print("Example 4: Financial Returns (Weak Autocorrelation)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    # Simulate daily returns with tiny autocorrelation (microstructure effects)
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    for t in range(1, n):
        returns[t] += 0.05 * returns[t - 1]  # Weak persistence

    result = compute_acf(returns, nlags=20)

    print("\nDaily Returns ACF:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {len(result.significant_lags)}")
    print(f"  Max absolute ACF (excl. lag 0): {np.abs(result.acf_values[1:]).max():.4f}")
    print("  First 10 ACF values:")
    for i in range(min(10, len(result.acf_values))):
        sig = "*" if i in result.significant_lags else " "
        print(f"    Lag {i:2d}: {result.acf_values[i]:7.4f} {sig}")

    print("\nInterpretation:")
    print("  - Market efficiency: returns should be nearly uncorrelated")
    print("  - Weak autocorrelation suggests limited predictability")
    print("  - Strong autocorrelation would indicate inefficiency or model misspecification")

    return result


def example_volatility_clustering():
    """Example 5: Absolute returns (volatility clustering)."""
    print("\n" + "=" * 60)
    print("Example 5: Absolute Returns (Volatility Clustering)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    # Simulate returns with GARCH-like volatility clustering
    returns = np.random.randn(n)
    volatility = [0.02]

    for t in range(1, n):
        volatility.append(0.01 + 0.1 * returns[t - 1] ** 2 + 0.8 * volatility[-1])
        returns[t] = returns[t] * np.sqrt(volatility[-1])

    # Analyze absolute returns (volatility proxy)
    abs_returns = np.abs(returns)
    result = compute_acf(abs_returns, nlags=20)

    print("\nAbsolute Returns ACF:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {len(result.significant_lags)}")
    print("  First 10 ACF values:")
    for i in range(min(10, len(result.acf_values))):
        sig = "*" if i in result.significant_lags else " "
        print(f"    Lag {i:2d}: {result.acf_values[i]:7.4f} {sig}")

    print("\nInterpretation:")
    print("  - While returns are uncorrelated, volatility is highly correlated")
    print("  - High ACF at many lags indicates volatility clustering")
    print("  - 'Volatility begets volatility' - GARCH effects")
    print("  - Important for risk management and option pricing")

    return result


def plot_pacf(result, title: str, ax=None):
    """Plot PACF with confidence intervals.

    Args:
        result: PACFResult object
        title: Plot title
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Plot PACF values
    ax.stem(result.lags, result.pacf_values, linefmt="C1-", markerfmt="C1o", basefmt=" ")

    # Plot confidence intervals
    ci_lower = result.conf_int[:, 0]
    ci_upper = result.conf_int[:, 1]
    ax.fill_between(result.lags, ci_lower, ci_upper, alpha=0.2, color="gray")

    # Add horizontal line at 0
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Partial Autocorrelation")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add text with significant lags
    n_sig = len(result.significant_lags)
    ax.text(
        0.98,
        0.98,
        f"Significant lags: {n_sig}/{len(result.lags) - 1}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )


def example_ar1_pacf():
    """Example 6: AR(1) PACF (single spike at lag 1)."""
    print("\n" + "=" * 60)
    print("Example 6: AR(1) PACF (Single Spike Pattern)")
    print("=" * 60)

    np.random.seed(42)
    phi = 0.7
    n = 1000

    # Generate AR(1): X_t = phi * X_{t-1} + epsilon_t
    noise = np.random.randn(n)
    ar1 = [noise[0]]
    for t in range(1, n):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    result = compute_pacf(ar1, nlags=20)

    print(f"\nAR(1) PACF with φ = {phi}:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {result.significant_lags}")
    print("  PACF values (cutoff pattern expected):")
    for k in range(min(10, len(result.pacf_values))):
        sig = "*" if k in result.significant_lags else " "
        print(f"    Lag {k:2d}: {result.pacf_values[k]:7.4f} {sig}")

    print("\nInterpretation:")
    print("  - AR(1) PACF cuts off after lag 1 (sharp drop to ~0)")
    print(f"  - PACF[1] ≈ φ = {phi}")
    print("  - PACF[k] ≈ 0 for k > 1 (key AR(1) identifier)")
    print("  - Compare to ACF which decays exponentially")

    return result


def example_ar2_pacf():
    """Example 7: AR(2) PACF (two spikes at lags 1-2)."""
    print("\n" + "=" * 60)
    print("Example 7: AR(2) PACF (Two Spike Pattern)")
    print("=" * 60)

    np.random.seed(42)
    phi1 = 0.5
    phi2 = 0.3
    n = 1000

    # Generate AR(2): X_t = phi1 * X_{t-1} + phi2 * X_{t-2} + epsilon_t
    noise = np.random.randn(n)
    ar2 = [noise[0], noise[1]]
    for t in range(2, n):
        ar2.append(phi1 * ar2[-1] + phi2 * ar2[-2] + noise[t])
    ar2 = np.array(ar2)  # type: ignore[assignment]

    result = compute_pacf(ar2, nlags=20)

    print(f"\nAR(2) PACF with φ1 = {phi1}, φ2 = {phi2}:")
    print(f"  Observations: {result.n_obs}")
    print(f"  Significant lags: {result.significant_lags}")
    print("  PACF values (cutoff after lag 2 expected):")
    for k in range(min(10, len(result.pacf_values))):
        sig = "*" if k in result.significant_lags else " "
        print(f"    Lag {k:2d}: {result.pacf_values[k]:7.4f} {sig}")

    print("\nInterpretation:")
    print("  - AR(2) PACF cuts off after lag 2")
    print("  - PACF[1] and PACF[2] are significant")
    print("  - PACF[k] ≈ 0 for k > 2 (identifies AR(2) order)")
    print("  - Number of significant lags = AR order (p=2)")

    return result


def example_acf_vs_pacf_comparison():
    """Example 8: ACF vs PACF for Model Identification."""
    print("\n" + "=" * 60)
    print("Example 8: ACF vs PACF for Model Identification")
    print("=" * 60)

    np.random.seed(42)
    phi = 0.7
    n = 5000  # Larger sample for clearer patterns

    # Generate AR(1): X_t = phi * X_{t-1} + epsilon_t
    noise = np.random.randn(n)
    ar1 = [noise[0]]
    for t in range(1, n):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    acf_result = compute_acf(ar1, nlags=20)
    pacf_result = compute_pacf(ar1, nlags=20)

    print(f"\nAR(1) Model Identification with φ = {phi}:")
    print("\nACF Pattern (Exponential Decay):")
    print(f"  Significant lags: {len(acf_result.significant_lags)}")
    print("  First 5 ACF values:")
    for k in range(1, 6):
        theoretical = phi**k
        print(f"    ACF[{k}] = {acf_result.acf_values[k]:.4f} (theory: {theoretical:.4f})")

    print("\nPACF Pattern (Cutoff after lag 1):")
    print(f"  Significant lags: {pacf_result.significant_lags}")
    print("  First 5 PACF values:")
    for k in range(1, 6):
        sig = "*" if k in pacf_result.significant_lags else " "
        print(f"    PACF[{k}] = {pacf_result.pacf_values[k]:.4f} {sig}")

    print("\nBox-Jenkins Identification Guide:")
    print("  ACF: Slow decay → Not MA process")
    print("  PACF: Cuts off at lag 1 → AR(1) process")
    print("  Conclusion: This is an AR(1) model")
    print("\nModel Identification Rules:")
    print("  AR(p):    ACF decays,    PACF cuts off at lag p")
    print("  MA(q):    ACF cuts off at lag q,    PACF decays")
    print("  ARMA(p,q): Both ACF and PACF decay (no clear cutoff)")

    return acf_result, pacf_result


def create_acf_pacf_comparison_plot():
    """Create side-by-side ACF vs PACF comparison for different processes."""
    print("\n" + "=" * 60)
    print("Creating ACF vs PACF Comparison Plot")
    print("=" * 60)

    np.random.seed(42)

    # Generate AR(1) and MA(1) processes
    phi = 0.7
    theta = 0.5
    noise = np.random.randn(5000)

    # AR(1)
    ar1 = [noise[0]]
    for t in range(1, len(noise)):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    # MA(1)
    noise_ma = np.random.randn(5001)
    ma1 = noise_ma[1:] + theta * noise_ma[:-1]

    # Compute ACF and PACF
    ar1_acf = compute_acf(ar1, nlags=20)
    ar1_pacf = compute_pacf(ar1, nlags=20)
    ma1_acf = compute_acf(ma1, nlags=20)
    ma1_pacf = compute_pacf(ma1, nlags=20)

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # AR(1) ACF and PACF
    plot_acf(ar1_acf, "AR(1) ACF: Exponential Decay", ax=axes[0, 0])
    plot_pacf(ar1_pacf, "AR(1) PACF: Cutoff at Lag 1", ax=axes[0, 1])

    # MA(1) ACF and PACF
    plot_acf(ma1_acf, "MA(1) ACF: Cutoff at Lag 1", ax=axes[1, 0])
    plot_pacf(ma1_pacf, "MA(1) PACF: Exponential Decay", ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("acf_pacf_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: acf_pacf_comparison.png")
    print("\nKey Observations:")
    print("  AR(1): ACF decays (many lags), PACF cuts off (lag 1 only)")
    print("  MA(1): ACF cuts off (lag 1 only), PACF decays (many lags)")
    print("  Use this complementary pattern for model identification!")


def create_comparison_plot():
    """Create comparison plot of all examples."""
    print("\n" + "=" * 60)
    print("Creating Comparison Plot")
    print("=" * 60)

    np.random.seed(42)

    # Generate all processes
    white_noise = np.random.randn(1000)

    phi = 0.7
    noise = np.random.randn(1000)
    ar1 = [noise[0]]
    for t in range(1, len(noise)):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    theta = 0.5
    noise_ma = np.random.randn(1001)
    ma1 = noise_ma[1:] + theta * noise_ma[:-1]

    returns = np.random.randn(1000) * 0.02
    for t in range(1, len(returns)):
        returns[t] += 0.05 * returns[t - 1]

    # Compute ACF for all
    results = [
        (compute_acf(white_noise, nlags=20), "White Noise"),
        (compute_acf(ar1, nlags=20), "AR(1) φ=0.7"),
        (compute_acf(ma1, nlags=20), "MA(1) θ=0.5"),
        (compute_acf(returns, nlags=20), "Financial Returns"),
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (result, title) in enumerate(results):
        plot_acf(result, title, ax=axes[i])

    plt.tight_layout()
    plt.savefig("acf_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: acf_comparison.png")


def example_comprehensive_analysis():
    """Example 9: Comprehensive Analysis with ARIMA Order Suggestion."""
    print("\n" + "=" * 60)
    print("Example 9: Comprehensive Autocorrelation Analysis")
    print("=" * 60)
    print("\nThis function combines ACF + PACF analysis and suggests ARIMA orders")

    # Example 1: White Noise
    print("\n" + "-" * 60)
    print("Case 1: White Noise")
    print("-" * 60)
    np.random.seed(42)
    white_noise = np.random.randn(1000)
    result = analyze_autocorrelation(white_noise, max_lags=20)
    print(result)
    print("\nSummary DataFrame (first 5 lags):")
    print(result.summary_df.head())

    # Example 2: AR(1)
    print("\n" + "-" * 60)
    print("Case 2: AR(1) Process")
    print("-" * 60)
    np.random.seed(42)
    phi = 0.7
    n = 5000
    noise = np.random.randn(n)
    ar1 = [noise[0]]
    for t in range(1, n):
        ar1.append(phi * ar1[-1] + noise[t])
    ar1 = np.array(ar1)  # type: ignore[assignment]

    result = analyze_autocorrelation(ar1, max_lags=20)
    print(result)
    print("\nSummary DataFrame (first 5 lags):")
    print(result.summary_df.head())

    # Example 3: AR(2)
    print("\n" + "-" * 60)
    print("Case 3: AR(2) Process")
    print("-" * 60)
    np.random.seed(42)
    phi1, phi2 = 0.5, 0.3
    n = 5000
    noise = np.random.randn(n)
    ar2 = [noise[0], noise[1]]
    for t in range(2, n):
        ar2.append(phi1 * ar2[-1] + phi2 * ar2[-2] + noise[t])
    ar2 = np.array(ar2)  # type: ignore[assignment]

    result = analyze_autocorrelation(ar2, max_lags=20)
    print(result)
    print("\nSummary DataFrame (first 5 lags):")
    print(result.summary_df.head())

    # Example 4: MA(1)
    print("\n" + "-" * 60)
    print("Case 4: MA(1) Process")
    print("-" * 60)
    np.random.seed(42)
    theta = 0.5
    n = 5000
    noise = np.random.randn(n + 1)
    ma1 = noise[1:] + theta * noise[:-1]

    result = analyze_autocorrelation(ma1, max_lags=20)
    print(result)
    print("\nSummary DataFrame (first 5 lags):")
    print(result.summary_df.head())

    print("\n" + "=" * 60)
    print("Comprehensive Analysis Usage Tips:")
    print("=" * 60)
    print("\n1. Use analyze_autocorrelation() for quick ARIMA order suggestion")
    print("2. Access individual ACF/PACF results: result.acf_result, result.pacf_result")
    print("3. Get summary DataFrame: result.summary_df (side-by-side ACF + PACF)")
    print("4. ARIMA order: result.suggested_arima_order → (p, d, q)")
    print("   Note: d=0 always (use analyze_stationarity() to determine d)")
    print("\n5. Interpretation:")
    print("   - p (AR order) from PACF cutoff")
    print("   - q (MA order) from ACF cutoff")
    print("   - For pure AR processes, ACF may not cutoff (expect high q)")
    print("   - For pure MA processes, PACF may not cutoff (expect high p)")
    print("   - Use PACF for AR order, ACF for MA order")


def main():
    """Run all ACF and PACF examples."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  ACF & PACF Analysis Examples".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)

    # Run ACF examples
    print("\n" + "=" * 60)
    print("PART 1: ACF (Autocorrelation Function) Examples")
    print("=" * 60)
    example_white_noise()
    example_ar1_process()
    example_ma1_process()
    example_financial_returns()
    example_volatility_clustering()

    # Run PACF examples
    print("\n" + "=" * 60)
    print("PART 2: PACF (Partial Autocorrelation Function) Examples")
    print("=" * 60)
    example_ar1_pacf()
    example_ar2_pacf()
    example_acf_vs_pacf_comparison()

    # Run comprehensive analysis
    print("\n" + "=" * 60)
    print("PART 3: Comprehensive Analysis with ARIMA Suggestion")
    print("=" * 60)
    example_comprehensive_analysis()

    # Create visualizations
    try:
        create_comparison_plot()
        create_acf_pacf_comparison_plot()
    except ImportError:
        print("\nNote: Install matplotlib to generate plots (pip install matplotlib)")

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("\nACF Patterns:")
    print("  1. White noise: No autocorrelation (all lags ≈ 0)")
    print("  2. AR(1): Exponential decay (persistent memory)")
    print("  3. MA(1): Single spike (short memory)")
    print("  4. Financial returns: Weak autocorrelation (market efficiency)")
    print("  5. Volatility: Strong autocorrelation (clustering effects)")
    print("\nPACF Patterns:")
    print("  6. AR(1): Single spike at lag 1, then cutoff")
    print("  7. AR(2): Two spikes at lags 1-2, then cutoff")
    print("  8. MA processes: Exponential decay (no sharp cutoff)")
    print("\nBox-Jenkins Model Identification:")
    print("  - AR(p): ACF decays, PACF cuts off at lag p")
    print("  - MA(q): ACF cuts off at lag q, PACF decays")
    print("  - ARMA(p,q): Both ACF and PACF decay")
    print("  - Use ACF + PACF together for accurate identification!")
    print("\nComprehensive Analysis:")
    print("  - analyze_autocorrelation() combines ACF + PACF")
    print("  - Automatically suggests ARIMA(p,0,q) orders")
    print("  - Provides summary DataFrame for easy comparison")
    print("  - Use with analyze_stationarity() to determine d order")


if __name__ == "__main__":
    main()
