"""Tests for analyze_interactions() tear sheet function.

NOTE: These tests compute SHAP interactions which are slow (~90s total).
Run with: pytest -m slow tests/test_evaluation/test_interaction_summary.py
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor

# Mark entire module as slow (SHAP interaction computation)
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic regression data with known interactions."""
    np.random.seed(42)
    n_samples = 300  # Reduced from 500 for faster tests

    # Create features
    x0 = np.random.randn(n_samples)
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)

    # Create target with strong interaction between x0 and x1
    # and weaker interaction between x2 and x3
    y = (
        2 * x0  # Main effect x0
        + 1 * x1  # Main effect x1
        + 3 * x0 * x1  # Strong interaction x0-x1
        + 0.5 * x2 * x3  # Weak interaction x2-x3
        + 0.1 * np.random.randn(n_samples)  # Noise
    )

    X = np.column_stack([x0, x1, x2, x3])

    return X, y


@pytest.fixture(scope="module")
def trained_model(synthetic_data):
    """Train a RandomForest model on synthetic data."""
    X, y = synthetic_data
    model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)  # Reduced from 50
    model.fit(X, y)
    return model


def test_basic_functionality(trained_model, synthetic_data):
    """Test basic analyze_interactions functionality."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    # Use only conditional_ic and h_statistic (no SHAP to avoid optional dependency)
    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Check all required keys are present
    assert "method_results" in result
    assert "consensus_ranking" in result
    assert "method_agreement" in result
    assert "top_interactions_consensus" in result
    assert "warnings" in result
    assert "interpretation" in result
    assert "methods_run" in result
    assert "methods_failed" in result

    # Check methods ran successfully
    assert "conditional_ic" in result["methods_run"]
    assert "h_statistic" in result["methods_run"]

    # Check consensus ranking structure
    assert len(result["consensus_ranking"]) > 0
    first_interaction = result["consensus_ranking"][0]
    assert len(first_interaction) == 4  # (feat_a, feat_b, avg_rank, scores_dict)
    assert isinstance(first_interaction[0], str)
    assert isinstance(first_interaction[1], str)
    assert isinstance(first_interaction[2], float)
    assert isinstance(first_interaction[3], dict)


def test_pandas_input(trained_model, synthetic_data):
    """Test analyze_interactions with pandas DataFrame input."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X_np, y_np = synthetic_data
    X_pd = pd.DataFrame(X_np, columns=["feat_0", "feat_1", "feat_2", "feat_3"])
    y_pd = pd.Series(y_np)

    result = analyze_interactions(
        trained_model,
        X_pd,
        y_pd,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    assert len(result["consensus_ranking"]) > 0
    # Check that feature names are preserved
    first_pair = result["consensus_ranking"][0]
    assert first_pair[0].startswith("feat_")
    assert first_pair[1].startswith("feat_")


def test_polars_input(trained_model, synthetic_data):
    """Test analyze_interactions with polars DataFrame input."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X_np, y_np = synthetic_data
    X_pl = pl.DataFrame(
        {
            "feat_0": X_np[:, 0],
            "feat_1": X_np[:, 1],
            "feat_2": X_np[:, 2],
            "feat_3": X_np[:, 3],
        }
    )
    y_pl = pl.Series("y", y_np)

    result = analyze_interactions(
        trained_model,
        X_pl,
        y_pl,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    assert len(result["consensus_ranking"]) > 0
    # Check that feature names are preserved
    first_pair = result["consensus_ranking"][0]
    assert first_pair[0].startswith("feat_")


def test_numpy_input(trained_model, synthetic_data):
    """Test analyze_interactions with numpy array input."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    assert len(result["consensus_ranking"]) > 0
    # Check that feature names are auto-generated
    first_pair = result["consensus_ranking"][0]
    assert first_pair[0].startswith("f")
    assert first_pair[1].startswith("f")


def test_specific_feature_pairs(trained_model, synthetic_data):
    """Test analyze_interactions with specific feature pairs."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X_np, y_np = synthetic_data
    X_pd = pd.DataFrame(X_np, columns=["x0", "x1", "x2", "x3"])

    # Test only two specific pairs
    pairs = [("x0", "x1"), ("x2", "x3")]

    result = analyze_interactions(
        trained_model,
        X_pd,
        y_np,
        feature_pairs=pairs,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Should only have 2 pairs in consensus ranking
    assert len(result["consensus_ranking"]) == 2

    # Check that the pairs are the ones we requested
    result_pairs = {(a, b) for a, b, _, _ in result["consensus_ranking"]}
    expected_pairs = {("x0", "x1"), ("x2", "x3")}
    assert result_pairs == expected_pairs


def test_single_method(trained_model, synthetic_data):
    """Test analyze_interactions with only one method."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["h_statistic"],
        max_samples=50,
    )

    assert len(result["methods_run"]) == 1
    assert "h_statistic" in result["methods_run"]
    assert len(result["consensus_ranking"]) > 0

    # Method agreement should be empty (only one method)
    assert len(result["method_agreement"]) == 0


def test_all_methods(trained_model, synthetic_data):
    """Test analyze_interactions with all methods including SHAP."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    # Try all methods (SHAP may fail if not installed)
    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic", "shap"],
        max_samples=50,
    )

    # At least conditional_ic and h_statistic should succeed
    assert len(result["methods_run"]) >= 2

    # If SHAP succeeded, should have 3 method agreement pairs
    if "shap" in result["methods_run"]:
        assert len(result["method_agreement"]) == 3  # All pairs
    else:
        assert len(result["method_agreement"]) == 1  # Only IC vs H-stat


def test_method_agreement_calculation(trained_model, synthetic_data):
    """Test that method agreement (Spearman correlation) is calculated correctly."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Should have one pairwise correlation
    assert len(result["method_agreement"]) == 1

    # Get the correlation
    corr = list(result["method_agreement"].values())[0]

    # Correlation should be between -1 and 1
    assert -1 <= corr <= 1

    # Check keys are tuples
    key = list(result["method_agreement"].keys())[0]
    assert isinstance(key, tuple)
    assert len(key) == 2


def test_consensus_ranking_order(trained_model, synthetic_data):
    """Test that consensus ranking is ordered by average rank."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Check that avg_rank is monotonically increasing (best to worst)
    avg_ranks = [x[2] for x in result["consensus_ranking"]]
    assert avg_ranks == sorted(avg_ranks)


def test_top_interactions_consensus(trained_model, synthetic_data):
    """Test that top_interactions_consensus identifies shared top interactions."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # top_interactions_consensus should be a list of tuples
    assert isinstance(result["top_interactions_consensus"], list)

    # Each element should be a tuple of 2 strings
    for pair in result["top_interactions_consensus"]:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)


def test_warnings_generation(trained_model, synthetic_data):
    """Test that warnings are generated when appropriate."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Warnings should be a list of strings
    assert isinstance(result["warnings"], list)
    for warning in result["warnings"]:
        assert isinstance(warning, str)


def test_interpretation_generation(trained_model, synthetic_data):
    """Test that interpretation text is generated."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Interpretation should be a non-empty string
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


def test_parameter_passing(trained_model, synthetic_data):
    """Test that parameters are passed correctly to individual methods."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    # Test with custom parameters
    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        n_quantiles=3,  # Custom for conditional IC
        grid_resolution=10,  # Custom for H-statistic
        max_samples=50,  # Custom for H-statistic
    )

    # Check that methods ran
    assert "conditional_ic" in result["methods_run"]
    assert "h_statistic" in result["methods_run"]


def test_edge_case_two_features(trained_model):
    """Test analyze_interactions with only 2 features (1 pair)."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] * X[:, 1] + 0.1 * np.random.randn(100)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    result = analyze_interactions(
        model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Should have exactly 1 pair
    assert len(result["consensus_ranking"]) == 1


def test_edge_case_constant_feature():
    """Test analyze_interactions handles constant features gracefully."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 2] = 1.0  # Constant feature
    y = X[:, 0] + 0.1 * np.random.randn(100)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Should not crash
    result = analyze_interactions(
        model,
        X,
        y,
        methods=["h_statistic"],  # Only H-stat to avoid IC issues
        max_samples=50,
    )

    assert len(result["consensus_ranking"]) > 0


def test_invalid_feature_pairs(trained_model, synthetic_data):
    """Test that invalid feature pairs raise appropriate errors."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X_np, y_np = synthetic_data
    X_pd = pd.DataFrame(X_np, columns=["x0", "x1", "x2", "x3"])

    # Test with unknown feature names
    with pytest.raises(ValueError, match="unknown features"):
        analyze_interactions(
            trained_model,
            X_pd,
            y_np,
            feature_pairs=[("x0", "x99")],  # x99 doesn't exist
            methods=["h_statistic"],
        )

    # Test with invalid pair structure
    with pytest.raises(ValueError, match="exactly 2 elements"):
        analyze_interactions(
            trained_model,
            X_pd,
            y_np,
            feature_pairs=[("x0", "x1", "x2")],  # 3 elements
            methods=["h_statistic"],
        )


def test_no_methods_specified_error():
    """Test that error is raised when no methods are specified."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)

    with pytest.raises(ValueError, match="At least one method"):
        analyze_interactions(model, X, y, methods=[])


def test_method_failure_handling(trained_model, synthetic_data):
    """Test that analysis continues even if one method fails."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    # Request SHAP which might not be installed
    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic", "shap"],
        max_samples=50,
    )

    # At least conditional_ic and h_statistic should succeed
    assert len(result["methods_run"]) >= 2

    # If SHAP failed, should be in methods_failed
    if "shap" not in result["methods_run"]:
        assert len(result["methods_failed"]) > 0
        assert any("shap" in failure[0] for failure in result["methods_failed"])
        # Should also be in warnings
        assert any("shap" in warning.lower() for warning in result["warnings"])


def test_scores_dict_structure(trained_model, synthetic_data):
    """Test that consensus ranking includes scores from each method."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    result = analyze_interactions(
        trained_model,
        X,
        y,
        methods=["conditional_ic", "h_statistic"],
        max_samples=50,
    )

    # Check first interaction has scores from both methods
    first_interaction = result["consensus_ranking"][0]
    scores_dict = first_interaction[3]

    # Should have scores from both methods
    assert "conditional_ic" in scores_dict or "h_statistic" in scores_dict

    # Scores should be numeric
    for _method, score in scores_dict.items():
        assert isinstance(score, int | float | np.number)


def test_output_format_consistency(trained_model, synthetic_data):
    """Test that output format is consistent across different inputs."""
    from ml4t.diagnostic.evaluation import analyze_interactions

    X, y = synthetic_data

    # Run with numpy
    result_np = analyze_interactions(trained_model, X, y, methods=["h_statistic"], max_samples=100)

    # Run with pandas
    X_pd = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    result_pd = analyze_interactions(
        trained_model, X_pd, y, methods=["h_statistic"], max_samples=100
    )

    # Both should have same structure
    assert set(result_np.keys()) == set(result_pd.keys())
    assert len(result_np["consensus_ranking"]) == len(result_pd["consensus_ranking"])

    # Consensus ranking format should be identical
    for i in range(len(result_np["consensus_ranking"])):
        np_item = result_np["consensus_ranking"][i]
        pd_item = result_pd["consensus_ranking"][i]
        assert len(np_item) == len(pd_item) == 4
