"""Demonstration of ML4T Evaluation 2.0 Result Schemas.

This example shows how to create, serialize, and use result schemas
from all evaluation modules.
"""

import json

from ml4t.diagnostic.results import (
    ACFResult,
    BayesianComparisonResult,
    DSRResult,
    FeatureDiagnosticsResult,
    FeatureOutcomeResult,
    ICAnalysisResult,
    MinTRLResult,
    PortfolioEvaluationResult,
    PortfolioMetrics,
    PSRResult,
    SharpeFrameworkResult,
    StationarityTestResult,
    ThresholdAnalysisResult,
)


def demo_feature_diagnostics():
    """Demonstrate Module A: Feature Diagnostics results."""
    print("=" * 60)
    print("Module A: Feature Diagnostics")
    print("=" * 60)

    # Create stationarity test result
    stat_result = StationarityTestResult(
        feature_name="momentum",
        adf_statistic=-4.2,
        adf_pvalue=0.001,
        adf_is_stationary=True,
        kpss_statistic=0.15,
        kpss_pvalue=0.10,
        kpss_is_stationary=True,
    )

    print("\nStationarity Test Result:")
    print(stat_result.summary())

    # Create ACF result
    acf_result = ACFResult(
        feature_name="momentum",
        acf_values=[1.0, 0.65, 0.42, 0.28, 0.15],
        pacf_values=[1.0, 0.65, 0.08, 0.03, 0.01],
        significant_lags_acf=[1, 2, 3],
        significant_lags_pacf=[1],
        ljung_box_statistic=45.3,
        ljung_box_pvalue=0.0001,
    )

    print("\n" + acf_result.summary())

    # Combined diagnostics
    diagnostics = FeatureDiagnosticsResult(
        stationarity_tests=[stat_result],
        acf_results=[acf_result],
        volatility_clustering={"garch_detected": True, "p": 1, "q": 1},
        distribution_stats={
            "skewness": -0.3,
            "kurtosis": 4.2,
            "jarque_bera_pvalue": 0.001,
        },
    )

    print("\n" + diagnostics.summary())

    # Export to JSON
    json_str = diagnostics.to_json_string(indent=2)
    print(f"\nJSON export (first 300 chars):\n{json_str[:300]}...")

    return diagnostics


def demo_feature_outcome():
    """Demonstrate Module C: Feature-Outcome results."""
    print("\n" + "=" * 60)
    print("Module C: Feature-Outcome Relationships")
    print("=" * 60)

    # IC analysis result
    ic_result = ICAnalysisResult(
        feature_name="rsi",
        ic_values=[0.045, 0.042, 0.038, 0.035],
        mean_ic=0.040,
        ic_std=0.004,
        ic_ir=10.0,
        pvalue=0.0001,
        hac_adjusted_pvalue=0.001,
    )

    print("\n" + ic_result.summary())

    # Threshold analysis
    threshold_result = ThresholdAnalysisResult(
        feature_name="rsi",
        optimal_threshold=70.0,
        precision=0.68,
        recall=0.52,
        f1_score=0.59,
        lift=2.1,
        coverage=0.15,
    )

    print("\n" + threshold_result.summary())

    # Combined feature-outcome
    feature_outcome = FeatureOutcomeResult(
        ic_results=[ic_result],
        threshold_results=[threshold_result],
        ml_importance={"rsi": 0.75, "volatility": 0.18, "volume": 0.07},
    )

    print("\n" + feature_outcome.summary())

    # DataFrame access
    ic_df = feature_outcome.get_ic_dataframe()
    print(f"\nIC DataFrame shape: {ic_df.shape}")
    print(ic_df)

    return feature_outcome


def demo_portfolio_evaluation():
    """Demonstrate Module D: Portfolio Evaluation results."""
    print("\n" + "=" * 60)
    print("Module D: Portfolio Evaluation")
    print("=" * 60)

    # Portfolio metrics
    metrics = PortfolioMetrics(
        total_return=0.65,
        annualized_return=0.18,
        annualized_volatility=0.22,
        sharpe_ratio=0.82,
        sortino_ratio=1.15,
        max_drawdown=-0.28,
        calmar_ratio=0.64,
        omega_ratio=1.45,
        win_rate=0.58,
        avg_win=0.025,
        avg_loss=-0.018,
        profit_factor=1.72,
        skewness=-0.15,
        kurtosis=2.8,
    )

    print("\n" + metrics.summary())

    # Bayesian comparison
    comparison = BayesianComparisonResult(
        strategy_a_name="ML Strategy",
        strategy_b_name="Benchmark",
        prior_sharpe_mean=0.5,
        prior_sharpe_std=0.3,
        posterior_sharpe_mean=0.82,
        posterior_sharpe_std=0.12,
        probability_a_better=0.92,
        credible_interval_95=(0.58, 1.06),
    )

    print("\n" + comparison.summary())

    # Complete portfolio evaluation
    portfolio_eval = PortfolioEvaluationResult(
        metrics=metrics,
        bayesian_comparison=comparison,
        drawdown_analysis={
            "max_duration_days": 89,
            "avg_drawdown": -0.12,
            "num_drawdowns": 7,
        },
    )

    print("\n" + portfolio_eval.summary())

    return portfolio_eval


def demo_sharpe_framework():
    """Demonstrate Enhanced Sharpe Framework results."""
    print("\n" + "=" * 60)
    print("Enhanced Sharpe Framework")
    print("=" * 60)

    # PSR
    psr = PSRResult(
        observed_sharpe=0.82,
        target_sharpe=0.50,
        psr_value=0.96,
        confidence_level=0.95,
        skewness=-0.15,
        kurtosis=2.8,
        n_observations=1250,
    )

    print("\n" + psr.summary())

    # MinTRL
    min_trl = MinTRLResult(
        observed_sharpe=0.82,
        target_sharpe=0.50,
        min_trl_days=450,
        actual_days=1250,
        is_sufficient=True,
        confidence_level=0.95,
        skewness=-0.15,
        kurtosis=2.8,
    )

    print("\n" + min_trl.summary())

    # DSR
    dsr = DSRResult(
        observed_sharpe=0.82,
        dsr_value=0.71,
        adjusted_pvalue=0.015,
        is_significant=True,
        n_trials=8,
        variance_trials=0.08,
        alpha=0.05,
    )

    print("\n" + dsr.summary())

    # Complete framework
    framework = SharpeFrameworkResult(
        psr=psr,
        min_trl=min_trl,
        dsr=dsr,
    )

    print("\n" + framework.summary())

    # DataFrame access
    df = framework.get_dataframe()
    print(f"\nFramework DataFrame:\n{df}")

    return framework


def demo_json_round_trip():
    """Demonstrate JSON serialization and deserialization."""
    print("\n" + "=" * 60)
    print("JSON Serialization Round-Trip")
    print("=" * 60)

    # Create a result
    original = PortfolioMetrics(
        total_return=0.65,
        annualized_return=0.18,
        annualized_volatility=0.22,
        sharpe_ratio=0.82,
        sortino_ratio=1.15,
        max_drawdown=-0.28,
        calmar_ratio=0.64,
        win_rate=0.58,
        avg_win=0.025,
        avg_loss=-0.018,
        profit_factor=1.72,
        skewness=-0.15,
        kurtosis=2.8,
    )

    # Serialize to JSON
    json_str = original.to_json_string(indent=2)
    print(f"\nSerialized to JSON ({len(json_str)} bytes)")

    # Deserialize
    data = json.loads(json_str)
    reconstructed = PortfolioMetrics(**data)

    # Verify round-trip
    print(f"Sharpe ratio matches: {reconstructed.sharpe_ratio == original.sharpe_ratio}")
    print(f"Max drawdown matches: {reconstructed.max_drawdown == original.max_drawdown}")
    print("JSON round-trip successful!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("ML4T Evaluation 2.0 Result Schemas Demo")
    print("=" * 60)

    demo_feature_diagnostics()
    demo_feature_outcome()
    demo_portfolio_evaluation()
    demo_sharpe_framework()
    demo_json_round_trip()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ Type-safe result schemas with Pydantic validation")
    print("✓ JSON serialization and deserialization")
    print("✓ DataFrame conversion for programmatic access")
    print("✓ Human-readable summaries")
    print("✓ IDE autocomplete support via type hints")


if __name__ == "__main__":
    main()
