"""Example: ML4T Engineer Integration - Generating Preprocessing Recommendations.

This example demonstrates how to use ML4T Evaluation's feature diagnostics to generate
preprocessing recommendations for the ML4T Engineer library.

Workflow:
1. Create synthetic feature data
2. Run feature diagnostics
3. Generate ML4T Engineer recommendations
4. Export to ML4T Engineer format
"""

from ml4t.diagnostic.integration import TransformType
from ml4t.diagnostic.results.feature_results import (
    FeatureDiagnosticsResultSchema,
    StationarityTestResult,
)


def main():
    """Run ML4T Engineer integration example."""
    print("=" * 70)
    print("ML4T Engineer Integration Example")
    print("=" * 70)
    print()

    # Simulate feature diagnostics results
    # In real usage, these would come from FeatureEvaluator.evaluate_diagnostics()
    diagnostics = create_example_diagnostics()

    # Generate ML4T Engineer recommendations
    print("Generating preprocessing recommendations...")
    qf_config = diagnostics.to_qfeatures_config()
    print()

    # Display summary
    print(qf_config.summary())
    print()

    # Show individual recommendations
    print("Individual Recommendations:")
    print("-" * 70)
    for rec in qf_config.recommendations:
        print(f"\n{rec.feature_name}:")
        print(f"  Transform: {rec.transform.value.upper()}")
        print(f"  Reason: {rec.reason}")
        print(f"  Confidence: {rec.confidence:.2%}")
        if rec.diagnostics:
            print(f"  Diagnostics: {rec.diagnostics}")

    # Export to ML4T Engineer format
    print("\n" + "=" * 70)
    print("ML4T Engineer-Compatible Export:")
    print("=" * 70)
    preprocessing_dict = qf_config.to_qfeatures_dict()

    import json

    print(json.dumps(preprocessing_dict, indent=2))

    # Filter examples
    print("\n" + "=" * 70)
    print("Filtering Examples:")
    print("=" * 70)

    # Features needing differencing
    diff_recs = qf_config.get_recommendations_by_transform(TransformType.DIFF)
    print(f"\nFeatures needing DIFF: {[r.feature_name for r in diff_recs]}")

    # Features that are good as-is
    none_recs = qf_config.get_recommendations_by_transform(TransformType.NONE)
    print(f"Features needing NONE: {[r.feature_name for r in none_recs]}")

    # High confidence recommendations
    high_conf = [r for r in qf_config.recommendations if r.confidence >= 0.9]
    print(f"\nHigh-confidence (≥0.9): {[r.feature_name for r in high_conf]}")

    # Integration with ML4T Engineer (pseudo-code)
    print("\n" + "=" * 70)
    print("Integration with ML4T Engineer (pseudo-code):")
    print("=" * 70)
    print("""
# When ML4T Engineer is available:
from ml4t.engineer import PreprocessingPipeline

# Apply recommendations
pipeline = PreprocessingPipeline(preprocessing_dict)
transformed_features = pipeline.transform(features_df)

# Use transformed features for modeling
model.fit(transformed_features, targets)
    """)


def create_example_diagnostics() -> FeatureDiagnosticsResultSchema:
    """Create example diagnostics with various scenarios.

    Returns:
        FeatureDiagnosticsResultSchema with stationarity tests
    """
    # Feature 1: Clearly non-stationary (all tests agree)
    price = StationarityTestResult(
        feature_name="price",
        adf_statistic=-1.2,
        adf_pvalue=0.85,
        adf_is_stationary=False,
        kpss_statistic=2.8,
        kpss_pvalue=0.005,
        kpss_is_stationary=False,
        pp_statistic=-1.5,
        pp_pvalue=0.78,
        pp_is_stationary=False,
    )

    # Feature 2: Clearly stationary (all tests agree)
    log_returns = StationarityTestResult(
        feature_name="log_returns",
        adf_statistic=-8.5,
        adf_pvalue=0.0001,
        adf_is_stationary=True,
        kpss_statistic=0.15,
        kpss_pvalue=0.1,
        kpss_is_stationary=True,
        pp_statistic=-9.2,
        pp_pvalue=0.0001,
        pp_is_stationary=True,
    )

    # Feature 3: Mixed signals (moderate confidence)
    volume = StationarityTestResult(
        feature_name="volume",
        adf_statistic=-2.5,
        adf_pvalue=0.12,
        adf_is_stationary=False,
        kpss_statistic=0.3,
        kpss_pvalue=0.15,
        kpss_is_stationary=True,
        pp_statistic=-3.2,
        pp_pvalue=0.08,
        pp_is_stationary=True,
    )

    # Feature 4: Another non-stationary
    market_cap = StationarityTestResult(
        feature_name="market_cap",
        adf_statistic=-1.8,
        adf_pvalue=0.65,
        adf_is_stationary=False,
        kpss_statistic=2.2,
        kpss_pvalue=0.01,
        kpss_is_stationary=False,
        pp_statistic=-2.0,
        pp_pvalue=0.55,
        pp_is_stationary=False,
    )

    # Feature 5: Stationary indicator
    rsi_14 = StationarityTestResult(
        feature_name="rsi_14",
        adf_statistic=-6.2,
        adf_pvalue=0.001,
        adf_is_stationary=True,
        kpss_statistic=0.2,
        kpss_pvalue=0.12,
        kpss_is_stationary=True,
        pp_statistic=-6.8,
        pp_pvalue=0.0005,
        pp_is_stationary=True,
    )

    return FeatureDiagnosticsResultSchema(
        stationarity_tests=[price, log_returns, volume, market_cap, rsi_14]
    )


if __name__ == "__main__":
    main()
