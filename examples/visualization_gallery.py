"""Visualization Gallery for ml4t.diagnostic.
================================

This example demonstrates all visualization capabilities in ml4t-diagnostic,
showcasing the interactive Plotly visualizations for financial ML evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml4t.diagnostic.evaluation import create_evaluation_dashboard, viz


def generate_example_data(n_samples=1000, n_features=10, n_assets=50):
    """Generate synthetic financial data for examples."""
    np.random.seed(42)

    # Time index
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")

    # Features with realistic properties
    features = pd.DataFrame(index=dates)

    # Momentum feature (trending)
    features["momentum"] = np.cumsum(np.random.randn(n_samples) * 0.01)

    # Mean reversion feature
    features["mean_reversion"] = -features["momentum"] * 0.5 + np.random.randn(n_samples) * 0.5

    # Volatility feature (always positive)
    features["volatility"] = np.abs(np.random.randn(n_samples)) * 0.02

    # Technical indicators
    for i in range(n_features - 3):
        features[f"technical_{i}"] = np.random.randn(n_samples)

    # Forward returns at different horizons
    base_returns = 0.001 + 0.02 * np.random.randn(n_samples)
    returns = pd.DataFrame(index=dates)
    returns["1d"] = base_returns
    returns["5d"] = pd.Series(base_returns).rolling(5).mean().fillna(0).values
    returns["10d"] = pd.Series(base_returns).rolling(10).mean().fillna(0).values
    returns["20d"] = pd.Series(base_returns).rolling(20).mean().fillna(0).values

    # Predictions (correlated with returns)
    predictions = (
        0.3 * features["momentum"]
        + 0.2 * features["mean_reversion"]
        - 0.1 * features["volatility"]
        + 0.1 * np.random.randn(n_samples)
    )

    # Multi-asset factor values
    asset_names = [f"asset_{i}" for i in range(n_assets)]
    factor_values = pd.DataFrame(
        np.random.randn(n_samples, n_assets),
        index=dates,
        columns=asset_names,
    )
    # Add some persistence
    for col in factor_values.columns:
        factor_values[col] = factor_values[col].rolling(10).mean().fillna(0)

    return features, returns, predictions, factor_values


def example_1_ic_heatmap():
    """Example 1: Information Coefficient Term Structure Heatmap."""
    print("\n" + "=" * 60)
    print("Example 1: IC Term Structure Heatmap")
    print("=" * 60)

    # Generate data
    features, returns, predictions, _ = generate_example_data()

    # Create IC heatmap
    fig = viz.plot_ic_heatmap(
        predictions=predictions,
        returns=returns,
        horizons=[1, 5, 10, 20],
        time_index=returns.index,
        title="Information Coefficient Across Prediction Horizons",
    )

    # Show the plot (in Jupyter this would display inline)
    # fig.show()

    # Save as HTML
    fig.write_html("ic_heatmap_example.html")
    print("✓ IC heatmap saved to ic_heatmap_example.html")

    return fig


def example_2_quantile_analysis():
    """Example 2: Quantile Analysis with Drill-Down."""
    print("\n" + "=" * 60)
    print("Example 2: Quantile Analysis")
    print("=" * 60)

    # Generate data
    features, returns, predictions, _ = generate_example_data()

    # Create quantile analysis plot
    fig = viz.plot_quantile_returns(
        predictions=predictions,
        returns=returns["10d"],  # Use 10-day returns
        n_quantiles=5,
        show_cumulative=True,
        title="10-Day Returns by Prediction Quantile",
    )

    # fig.show()
    fig.write_html("quantile_analysis_example.html")
    print("✓ Quantile analysis saved to quantile_analysis_example.html")

    return fig


def example_3_turnover_decay():
    """Example 3: Factor Turnover and Decay Analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Turnover and Decay Analysis")
    print("=" * 60)

    # Generate data
    _, _, _, factor_values = generate_example_data()

    # Create turnover decay plot
    fig = viz.plot_turnover_decay(
        factor_values=factor_values,
        quantiles=5,
        lags=[1, 5, 10, 20, 30],
        title="Factor Stability Analysis",
    )

    # fig.show()
    fig.write_html("turnover_decay_example.html")
    print("✓ Turnover analysis saved to turnover_decay_example.html")

    return fig


def example_4_feature_distributions():
    """Example 4: Feature Distribution Evolution."""
    print("\n" + "=" * 60)
    print("Example 4: Feature Distribution Analysis")
    print("=" * 60)

    # Generate data
    features, _, _, _ = generate_example_data()

    # Select subset of features for visualization
    selected_features = features[
        [
            "momentum",
            "mean_reversion",
            "volatility",
            "technical_0",
            "technical_1",
            "technical_2",
        ]
    ]

    # Create distribution plot
    fig = viz.plot_feature_distributions(
        features=selected_features,
        n_periods=4,
        method="box",
        title="Feature Distribution Evolution Over Time",
    )

    # fig.show()
    fig.write_html("feature_distributions_example.html")
    print("✓ Feature distributions saved to feature_distributions_example.html")

    return fig


def example_5_complete_dashboard():
    """Example 5: Complete Evaluation Dashboard."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Evaluation Dashboard")
    print("=" * 60)

    # Generate data
    features, returns, predictions, factor_values = generate_example_data(n_samples=500)

    # Split features and target
    x = features
    y = returns["10d"]

    # Train a simple model
    model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)

    # Create evaluator
    evaluator = ml4t.diagnostic.Evaluator(tier=2, random_state=42)  # type: ignore[name-defined]

    # Evaluate model
    print("Evaluating model...")
    result = evaluator.evaluate(x, y, model)

    # Print summary
    print("\nEvaluation Summary:")
    print(
        f"IC: {result.metrics_results['ic']['mean']:.4f} ± {result.metrics_results['ic']['std']:.4f}",
    )
    print(f"Sharpe: {result.metrics_results['sharpe']['mean']:.4f}")
    print(f"Hit Rate: {result.metrics_results['hit_rate']['mean']:.1f}%")

    # Generate full dashboard
    create_evaluation_dashboard(
        result=result,
        output_file="full_evaluation_dashboard.html",
        predictions=predictions.to_frame("predictions"),
        returns=returns,
        features=features,
        title="Complete Model Evaluation Dashboard",
    )

    print("\n✓ Full dashboard saved to full_evaluation_dashboard.html")

    return result


def example_6_theme_comparison():
    """Example 6: Different Themes for Visualizations."""
    print("\n" + "=" * 60)
    print("Example 6: Theme Comparison")
    print("=" * 60)

    from ml4t.diagnostic.evaluation.themes import apply_theme

    # Generate data
    features, returns, predictions, _ = generate_example_data(n_samples=200)

    # Create base plot
    base_fig = viz.plot_quantile_returns(
        predictions=predictions,
        returns=returns["5d"],
        n_quantiles=5,
        show_cumulative=False,
    )

    # Apply different themes
    themes = ["default", "dark", "print"]

    for theme in themes:
        fig = apply_theme(base_fig, theme)
        fig.update_layout(title=f"Quantile Returns - {theme.title()} Theme")
        fig.write_html(f"theme_{theme}_example.html")
        print(f"✓ {theme.title()} theme saved to theme_{theme}_example.html")


def example_7_tier_specific_visualizations():
    """Example 7: Tier-Specific Visualization Patterns."""
    print("\n" + "=" * 60)
    print("Example 7: Tier-Specific Visualizations")
    print("=" * 60)

    # Generate data
    features, returns, _, _ = generate_example_data(n_samples=300)
    x = features.iloc[:, :3]  # Use only 3 features for speed
    y = returns["5d"]

    model = LinearRegression()

    # Tier 3: Fast screening visualization
    print("\nTier 3 - Fast Screening:")
    evaluator_t3 = ml4t.diagnostic.Evaluator(tier=3)  # type: ignore[name-defined]
    result_t3 = evaluator_t3.evaluate(x, y, model)

    # Simple IC plot for screening
    fig_t3 = result_t3.plot(x.mean(axis=1), y)
    if fig_t3:
        fig_t3.update_layout(title="Tier 3: Quick IC Screening")
        fig_t3.write_html("tier3_visualization.html")
        print("✓ Tier 3 visualization saved")

    # Tier 2: Statistical significance
    print("\nTier 2 - Statistical Significance:")
    evaluator_t2 = ml4t.diagnostic.Evaluator(tier=2)  # type: ignore[name-defined]
    result_t2 = evaluator_t2.evaluate(x, y, model)

    # More detailed analysis
    if result_t2.statistical_tests:
        print(
            f"HAC-adjusted IC p-value: {result_t2.statistical_tests.get('hac_ic', {}).get('p_value', 'N/A')}",
        )

    # Tier 1: Full rigorous analysis
    print("\nTier 1 - Rigorous Backtesting:")
    evaluator_t1 = ml4t.diagnostic.Evaluator(tier=1)  # type: ignore[name-defined]
    result_t1 = evaluator_t1.evaluate(x, y, model)

    if result_t1.statistical_tests.get("dsr"):
        print(
            f"Deflated Sharpe Ratio: {result_t1.statistical_tests['dsr'].get('dsr', 'N/A')}",
        )


def main():
    """Run all visualization examples."""
    print("\nqeval Visualization Gallery")
    print("=" * 60)
    print("This script demonstrates all visualization capabilities in ml4t.diagnostic.")
    print("Each example creates an interactive HTML file you can open in a browser.")

    # Run all examples
    example_1_ic_heatmap()
    example_2_quantile_analysis()
    example_3_turnover_decay()
    example_4_feature_distributions()
    example_5_complete_dashboard()
    example_6_theme_comparison()
    example_7_tier_specific_visualizations()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Open the generated HTML files in your browser to view interactive plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
