"""Example: ML4T Backtest Integration - Complete Workflow

This example demonstrates the complete strategy lifecycle from backtest
to paper to live, using ML4T Evaluation's ML4T Backtest integration contract.

Workflow:
1. Evaluate backtest results
2. Export to ML4T Backtest for storage
3. Compare paper trading vs backtest
4. Evaluate promotion criteria
5. Monitor live performance

Author: ML4T Evaluation Team
Date: 2025-11-03
"""

from datetime import datetime

from ml4t.diagnostic.integration import (
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    EnvironmentType,
    EvaluationExport,
    PromotionWorkflow,
    StrategyMetadata,
)


def example_1_backtest_evaluation():
    """Example 1: Evaluate and export backtest results."""
    print("=" * 70)
    print("Example 1: Backtest Evaluation & Export")
    print("=" * 70)

    # Simulate backtest metrics (normally from PortfolioEvaluator)
    backtest_metrics = {
        "sharpe_ratio": 1.85,
        "cagr": 0.24,
        "max_drawdown": -0.18,
        "volatility": 0.15,
        "calmar_ratio": 1.33,
        "sortino_ratio": 2.15,
        "total_trades": 1250,
        "win_rate": 0.58,
    }

    # Create metadata
    metadata = StrategyMetadata(
        strategy_id="momentum_rsi_v1",
        version="1.0.0",
        environment=EnvironmentType.BACKTEST,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        config_hash="abc123def456",
        description="Momentum strategy with RSI filter on crypto futures",
        tags={"asset_class": "crypto", "timeframe": "1h", "universe": "top_20"},
    )

    # Create evaluation export
    export = EvaluationExport(
        metadata=metadata,
        metrics=backtest_metrics,
        sharpe_framework={
            "psr": 0.92,  # Probabilistic Sharpe Ratio
            "dsr": 1.45,  # Deflated Sharpe Ratio
            "min_trl": 250,  # Minimum Track Record Length
        },
        diagnostics={
            "stationarity": {"adf_pvalue": 0.02, "stationary": True},
            "autocorrelation": {"ljung_box_pvalue": 0.15, "independent": True},
        },
        qeval_version="2.0.0",
    )

    # Export to dictionary (for ML4T Backtest storage)
    export.to_dict()
    print("\n📊 Backtest Evaluation Summary:")
    print(f"   Strategy: {metadata.strategy_id} v{metadata.version}")
    print(f"   Period: {metadata.start_date.date()} to {metadata.end_date.date()}")
    print(f"   Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")
    print(f"   CAGR: {backtest_metrics['cagr']:.1%}")
    print(f"   Max Drawdown: {backtest_metrics['max_drawdown']:.1%}")
    print(f"   PSR: {export.sharpe_framework['psr']:.2f}")
    print(f"   DSR: {export.sharpe_framework['dsr']:.2f}")

    # Save to file (simulate ML4T Backtest storage)
    json_output = export.to_json()
    print(f"\n💾 Export size: {len(json_output):,} bytes")
    print("   Ready for ML4T Backtest storage: ml4t.backtest.store_evaluation(export.to_dict())")

    return export


def example_2_paper_vs_backtest_comparison():
    """Example 2: Compare paper trading results to backtest."""
    print("\n" + "=" * 70)
    print("Example 2: Paper vs Backtest Comparison")
    print("=" * 70)

    # Backtest results (from Example 1)
    backtest_export = EvaluationExport(
        metadata=StrategyMetadata(
            strategy_id="momentum_rsi_v1",
            version="1.0.0",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
        ),
        metrics={
            "sharpe_ratio": 1.85,
            "cagr": 0.24,
            "max_drawdown": -0.18,
            "volatility": 0.15,
        },
    )

    # Paper trading results (30 days)
    paper_export = EvaluationExport(
        metadata=StrategyMetadata(
            strategy_id="momentum_rsi_v1",
            version="1.0.0",
            environment=EnvironmentType.PAPER,
            start_date=datetime(2024, 10, 1),
            end_date=datetime(2024, 10, 31),
        ),
        metrics={
            "sharpe_ratio": 1.72,  # Slightly lower (expected)
            "cagr": 0.22,  # Slightly lower
            "max_drawdown": -0.16,  # Slightly better
            "volatility": 0.14,  # Similar
        },
    )

    # Create comparison request
    request = ComparisonRequest(
        strategy_id="momentum_rsi_v1",
        backtest_export=backtest_export,
        live_export=paper_export,
        comparison_type=ComparisonType.BAYESIAN,
        confidence_level=0.95,
        hypothesis="paper >= 0.9 * backtest",  # Allow 10% degradation
    )

    print("\n📋 Comparison Request:")
    print(f"   Strategy: {request.strategy_id}")
    print(f"   Method: {request.comparison_type.value}")
    print(f"   Hypothesis: {request.hypothesis}")
    print(f"   Confidence Level: {request.confidence_level:.0%}")

    # Simulate comparison result (normally from BayesianComparison)
    result = ComparisonResult(
        strategy_id="momentum_rsi_v1",
        comparison_type=ComparisonType.BAYESIAN,
        decision="PROMOTE",
        confidence=0.92,
        metrics_comparison={
            "sharpe_ratio": {
                "backtest": 1.85,
                "live": 1.72,
                "diff": -0.13,
                "pct_change": -7.0,
            },
            "cagr": {
                "backtest": 0.24,
                "live": 0.22,
                "diff": -0.02,
                "pct_change": -8.3,
            },
            "max_drawdown": {
                "backtest": -0.18,
                "live": -0.16,
                "diff": 0.02,
                "pct_change": 11.1,
            },
        },
        statistical_tests={
            "bayesian": {
                "bayes_factor": 3.2,
                "posterior_prob": 0.92,
                "prior_prob": 0.5,
                "interpretation": "Moderate evidence for H1",
            },
            "t_test": {"t_statistic": 1.85, "p_value": 0.067},
        },
        bayesian_evidence={
            "bayes_factor": 3.2,
            "posterior_prob": 0.92,
        },
        recommendation=(
            "Paper trading performance is within acceptable range of backtest. "
            "Sharpe ratio declined by 7%, which is within tolerance. "
            "Ready for live promotion pending final approval."
        ),
        warnings=[
            "Sample size limited (30 days) - monitor closely in live",
            "Market conditions differ from backtest period",
        ],
    )

    # Display comparison summary
    print("\n" + result.summary())
    print(f"\n✅ Decision: {result.decision} (confidence: {result.confidence:.0%})")

    return result


def example_3_promotion_workflow():
    """Example 3: Evaluate promotion from paper to live."""
    print("\n" + "=" * 70)
    print("Example 3: Paper-to-Live Promotion Workflow")
    print("=" * 70)

    # Define promotion criteria
    workflow = PromotionWorkflow(
        strategy_id="momentum_rsi_v1",
        paper_duration_days=30,
        promotion_criteria={
            "min_sharpe": 1.5,  # Minimum Sharpe ratio
            "max_drawdown": -0.20,  # Maximum drawdown tolerance
            "min_trades": 100,  # Minimum number of trades
            "bayesian_confidence": 0.90,  # Minimum Bayesian confidence
        },
        approval_required=True,
        risk_limits={
            "max_position_size": 0.05,  # 5% max position
            "max_leverage": 2.0,  # 2x max leverage
            "max_daily_loss": -0.02,  # 2% max daily loss
        },
    )

    print("\n📋 Promotion Criteria:")
    print(f"   Strategy: {workflow.strategy_id}")
    print(f"   Paper Duration: {workflow.paper_duration_days} days")
    print(f"   Approval Required: {workflow.approval_required}")
    print("\n   Performance Requirements:")
    for key, value in workflow.promotion_criteria.items():
        print(f"      {key}: {value}")
    print("\n   Risk Limits:")
    for key, value in workflow.risk_limits.items():
        print(f"      {key}: {value}")

    # Get comparison result from Example 2
    comparison_result = ComparisonResult(
        strategy_id="momentum_rsi_v1",
        comparison_type=ComparisonType.BAYESIAN,
        decision="PROMOTE",
        confidence=0.92,
        metrics_comparison={
            "sharpe_ratio": {"backtest": 1.85, "live": 1.72},
            "max_drawdown": {"backtest": -0.18, "live": -0.16},
        },
        statistical_tests={},
        recommendation="Ready for promotion",
    )

    # Evaluate promotion
    is_eligible = workflow.evaluate_promotion(comparison_result)

    print("\n🔍 Promotion Evaluation:")
    print(f"   Decision: {comparison_result.decision}")
    print(f"   Confidence: {comparison_result.confidence:.0%}")
    print(f"   Meets Criteria: {'✅ YES' if is_eligible else '❌ NO'}")

    if is_eligible:
        if workflow.approval_required:
            print("\n⚠️  NEXT STEP: Manual approval required before live deployment")
            print("   Review:")
            print("      - Comparison results")
            print(f"      - Risk limits: {workflow.risk_limits}")
            print("      - Market conditions")
        else:
            print("\n✅ READY FOR LIVE: All criteria met, auto-promotion enabled")
            print(f"   ml4t.backtest.promote_to_live('{workflow.strategy_id}', risk_limits)")
    else:
        print("\n❌ NOT READY: Promotion criteria not satisfied")

    return is_eligible


def example_4_live_monitoring():
    """Example 4: Monitor live performance and detect drift."""
    print("\n" + "=" * 70)
    print("Example 4: Live Performance Monitoring")
    print("=" * 70)

    # Backtest baseline
    backtest_export = EvaluationExport(
        metadata=StrategyMetadata(
            strategy_id="momentum_rsi_v1",
            version="1.0.0",
            environment=EnvironmentType.BACKTEST,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
        ),
        metrics={"sharpe_ratio": 1.85, "max_drawdown": -0.18},
    )

    # Live performance (first 30 days)
    live_export = EvaluationExport(
        metadata=StrategyMetadata(
            strategy_id="momentum_rsi_v1",
            version="1.0.0",
            environment=EnvironmentType.LIVE,
            start_date=datetime(2024, 11, 1),
            end_date=datetime(2024, 11, 30),
        ),
        metrics={"sharpe_ratio": 1.25, "max_drawdown": -0.25},  # Performance degraded
    )

    # Create comparison request with CUSUM drift detection
    ComparisonRequest(
        strategy_id="momentum_rsi_v1",
        backtest_export=backtest_export,
        live_export=live_export,
        comparison_type=ComparisonType.CUSUM,
    )

    # Simulate drift detection result
    result = ComparisonResult(
        strategy_id="momentum_rsi_v1",
        comparison_type=ComparisonType.CUSUM,
        decision="REJECT",
        confidence=0.88,
        metrics_comparison={
            "sharpe_ratio": {
                "backtest": 1.85,
                "live": 1.25,
                "diff": -0.60,
                "pct_change": -32.4,
            },
            "max_drawdown": {
                "backtest": -0.18,
                "live": -0.25,
                "diff": -0.07,
                "pct_change": -38.9,
            },
        },
        statistical_tests={
            "cusum": {
                "drift_detected": True,
                "drift_time": "2024-11-15",
                "significance": 0.01,
            }
        },
        recommendation=(
            "Significant performance degradation detected. "
            "Sharpe ratio declined by 32%, max drawdown increased by 39%. "
            "Recommend pausing strategy and investigating root cause."
        ),
        warnings=[
            "⚠️  CRITICAL: Sharpe ratio below acceptable threshold",
            "⚠️  CRITICAL: Drawdown exceeds risk limits",
            "CUSUM drift detected on 2024-11-15",
            "Possible market regime change or strategy decay",
        ],
    )

    print("\n📊 Live Performance vs Backtest:")
    print(result.summary())

    if result.decision == "REJECT":
        print("\n🚨 ALERT: Performance degradation detected!")
        print("   Action: Pause strategy immediately")
        print("   Next Steps:")
        print("      1. Investigate root cause")
        print("      2. Review recent market conditions")
        print("      3. Check for execution issues")
        print("      4. Re-evaluate strategy parameters")
        print(f"\n   ml4t.backtest.pause_strategy('{result.strategy_id}')")

    return result


def example_5_complete_lifecycle():
    """Example 5: Complete strategy lifecycle from backtest to live."""
    print("\n" + "=" * 70)
    print("Example 5: Complete Strategy Lifecycle")
    print("=" * 70)

    stages = [
        ("BACKTEST", EnvironmentType.BACKTEST, {"sharpe_ratio": 1.85}),
        ("PAPER", EnvironmentType.PAPER, {"sharpe_ratio": 1.72}),
        ("LIVE", EnvironmentType.LIVE, {"sharpe_ratio": 1.68}),
    ]

    print("\n📈 Strategy Progression:")
    for i, (stage_name, env, metrics) in enumerate(stages, 1):
        export = EvaluationExport(
            metadata=StrategyMetadata(
                strategy_id="momentum_rsi_v1",
                version="1.0.0",
                environment=env,
                start_date=datetime(2024, i, 1),
                end_date=datetime(2024, i, 28),
            ),
            metrics=metrics,
        )

        print(f"\n   Stage {i}: {stage_name}")
        print(f"      Environment: {export.metadata.environment.value}")
        print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"      Status: {'✅ Passed' if i < len(stages) else '🔄 Active'}")

    print("\n✅ Strategy successfully deployed to LIVE trading!")


if __name__ == "__main__":
    print("ML4T Backtest Integration - Complete Workflow Examples\n")

    # Run all examples
    backtest_export = example_1_backtest_evaluation()
    comparison_result = example_2_paper_vs_backtest_comparison()
    is_promoted = example_3_promotion_workflow()
    live_monitoring = example_4_live_monitoring()
    example_5_complete_lifecycle()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
