"""Demo script for analyze_ml_importance function.

This demonstrates the comprehensive ML feature importance analysis that compares
multiple methods (MDI, PFI, MDA, SHAP) and generates consensus rankings.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ml4t.diagnostic.evaluation import analyze_ml_importance


def main():
    """Run demo of ML importance analysis."""
    print("=" * 80)
    print("ML Feature Importance Analysis Demo")
    print("=" * 80)
    print()

    # Create synthetic dataset with clear informative features
    print("Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        random_state=42,
    )
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print()

    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)
    train_score = model.score(X, y)
    print(f"Training accuracy: {train_score:.3f}")
    print()

    # Run comprehensive importance analysis
    print("Running comprehensive importance analysis (MDI + PFI)...")
    print("This compares multiple methods and identifies consensus features...")
    print()

    result = analyze_ml_importance(
        model, X, y, methods=["mdi", "pfi"], n_repeats=10, random_state=42
    )

    # Print interpretation
    print("=" * 80)
    print("AUTO-GENERATED INTERPRETATION")
    print("=" * 80)
    print(result["interpretation"])
    print()

    # Print consensus ranking
    print("=" * 80)
    print("CONSENSUS FEATURE RANKING")
    print("=" * 80)
    print("(Average rank across all methods)")
    print()
    for i, feature in enumerate(result["consensus_ranking"][:5], 1):
        print(f"{i}. {feature}")
    print()

    # Print method agreement
    print("=" * 80)
    print("METHOD AGREEMENT")
    print("=" * 80)
    for pair, corr in result["method_agreement"].items():
        print(f"{pair}: {corr:.3f}")
    print()

    # Print consensus top features
    print("=" * 80)
    print("CONSENSUS TOP FEATURES")
    print("=" * 80)
    print("(Features in top 10 for ALL methods)")
    print()
    if result["top_features_consensus"]:
        for feature in sorted(result["top_features_consensus"]):
            print(f"  - {feature}")
    else:
        print("  (No features in top 10 for all methods)")
    print()

    # Print detailed results by method
    print("=" * 80)
    print("DETAILED RESULTS BY METHOD")
    print("=" * 80)
    print()

    for method_name in result["methods_run"]:
        print(f"{method_name.upper()} Top 5:")
        method_result = result["method_results"][method_name]
        for i in range(min(5, len(method_result["feature_names"]))):
            fname = method_result["feature_names"][i]
            if method_name == "pfi":
                importance = method_result["importances_mean"][i]
            else:
                importance = method_result["importances"][i]
            print(f"  {i + 1}. {fname}: {importance:.4f}")
        print()

    # Print warnings
    if result["warnings"]:
        print("=" * 80)
        print("WARNINGS")
        print("=" * 80)
        for warning in result["warnings"]:
            print(f"⚠  {warning}")
        print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
