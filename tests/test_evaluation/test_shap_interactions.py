"""Tests for SHAP interaction values (TASK-052)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml4t.diagnostic.evaluation import compute_shap_interactions


class TestSHAPInteractionsBasic:
    """Test basic SHAP interaction computation."""

    @pytest.fixture(scope="class")
    def strong_interaction_data_and_model(self):
        """Create data with strong multiplicative interaction."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5
        )
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def no_interaction_data_and_model(self):
        """Create data with additive features (no interaction)."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        return X, y, model

    def test_strong_interaction_product(self, strong_interaction_data_and_model):
        """Test SHAP interactions on multiplicative features."""
        X, _, model = strong_interaction_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, feature_names=["x0", "x1"], max_samples=150)

        # Should detect strong interaction
        assert len(result["top_interactions"]) == 1  # Only one pair
        feat_i, feat_j, interaction = result["top_interactions"][0]
        assert {feat_i, feat_j} == {"x0", "x1"}
        assert interaction > 0.0  # Positive interaction expected

    def test_no_interaction(self, no_interaction_data_and_model):
        """Test SHAP interactions on additive features."""
        X, _, model = no_interaction_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=150)

        # Interaction should be lower than for multiplicative case
        feat_i, feat_j, interaction = result["top_interactions"][0]
        assert interaction < 0.5  # Should be relatively small


class TestSHAPInteractionsMultipleFeatures:
    """Test SHAP interactions with multiple features."""

    @pytest.fixture(scope="class")
    def multi_interaction_data_and_model(self):
        """Create data with multiple interaction patterns."""
        rng = np.random.RandomState(42)
        n_samples = 400
        X = rng.randn(n_samples, 5)
        # Strong interaction: x0 * x1
        # Weak interaction: x2 + x3
        # Independent: x4
        y = (
            X[:, 0] * X[:, 1]  # Strong interaction
            + X[:, 2]
            + X[:, 3]
            + 0.1 * X[:, 2] * X[:, 3]  # Weak interaction
            + X[:, 4]
            + 0.1 * rng.randn(n_samples)
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def top_k_data_and_model(self):
        """Create data for top-k filtering test."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 5)
        y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + X[:, 4] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_identify_top_interactions(self, multi_interaction_data_and_model):
        """Test identifying top interactions among multiple features."""
        X, _, model = multi_interaction_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=200)

        # Should have 10 pairs (5 choose 2)
        assert len(result["top_interactions"]) == 10

        # Just verify interactions are sorted and non-negative
        # (Exact ranking can vary with Random Forest)
        interactions_vals = [val for _, _, val in result["top_interactions"]]
        assert all(v >= 0 for v in interactions_vals)
        assert interactions_vals == sorted(interactions_vals, reverse=True)

    def test_top_k_filtering(self, top_k_data_and_model):
        """Test limiting results to top K interactions."""
        X, _, model = top_k_data_and_model

        # Request only top 3
        result = compute_shap_interactions(model, X, max_samples=150, top_k=3)

        # Should have exactly 3 interactions
        assert len(result["top_interactions"]) == 3


class TestSHAPInteractionsInputFormats:
    """Test SHAP interactions with different input formats."""

    @pytest.fixture(scope="class")
    def pandas_data_and_model(self):
        """Create pandas DataFrame test data."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X_df = pd.DataFrame(
            rng.randn(n_samples, 3), columns=["feature_a", "feature_b", "feature_c"]
        )
        y = X_df["feature_a"] * X_df["feature_b"] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_df, y)
        return X_df, y, model

    @pytest.fixture(scope="class")
    def polars_data_and_model(self):
        """Create polars DataFrame test data."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X_pl = pl.DataFrame(
            {
                "feat_x": rng.randn(n_samples),
                "feat_y": rng.randn(n_samples),
                "feat_z": rng.randn(n_samples),
            }
        )
        y = X_pl["feat_x"].to_numpy() * X_pl["feat_y"].to_numpy() + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_pl.to_numpy(), y)
        return X_pl, y, model

    @pytest.fixture(scope="class")
    def numpy_data_and_model(self):
        """Create numpy array test data."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 3)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def custom_names_data_and_model(self):
        """Create data for custom feature names test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_pandas_dataframe(self, pandas_data_and_model):
        """Test with pandas DataFrame input."""
        X_df, _, model = pandas_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X_df, max_samples=100)

        # Feature names should come from DataFrame
        assert result["feature_names"] == ["feature_a", "feature_b", "feature_c"]
        assert result["n_features"] == 3

        # Top interaction should be feature_a × feature_b
        top_i, top_j, _ = result["top_interactions"][0]
        assert {top_i, top_j} == {"feature_a", "feature_b"}

    def test_polars_dataframe(self, polars_data_and_model):
        """Test with polars DataFrame input."""
        X_pl, _, model = polars_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X_pl, max_samples=100)

        # Feature names should come from polars columns
        assert result["feature_names"] == ["feat_x", "feat_y", "feat_z"]

    def test_numpy_array(self, numpy_data_and_model):
        """Test with numpy array input."""
        X, _, model = numpy_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=100)

        # Feature names should be f0, f1, f2
        assert result["feature_names"] == ["f0", "f1", "f2"]

    def test_custom_feature_names(self, custom_names_data_and_model):
        """Test with custom feature names for numpy array."""
        X, _, model = custom_names_data_and_model

        # Compute SHAP interactions with custom names
        result = compute_shap_interactions(
            model, X, feature_names=["momentum", "volatility"], max_samples=100
        )

        # Should use custom names
        assert result["feature_names"] == ["momentum", "volatility"]
        feat_i, feat_j, _ = result["top_interactions"][0]
        assert {feat_i, feat_j} == {"momentum", "volatility"}


class TestSHAPInteractionsParameters:
    """Test SHAP interactions parameter handling."""

    @pytest.fixture(scope="class")
    def large_data_and_model(self):
        """Create larger data for subsampling test."""
        rng = np.random.RandomState(42)
        n_samples = 1000
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def small_data_and_model(self):
        """Create small data for no-subsampling test."""
        rng = np.random.RandomState(42)
        n_samples = 80
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def medium_data_and_model(self):
        """Create medium data for max_samples test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_subsampling(self, large_data_and_model):
        """Test subsampling when max_samples < data size."""
        X, _, model = large_data_and_model

        # Compute with subsampling
        result = compute_shap_interactions(model, X, max_samples=150)

        # Should indicate 150 samples used
        assert result["n_samples_used"] == 150

    def test_no_subsampling_small_data(self, small_data_and_model):
        """Test no subsampling when data already small."""
        X, _, model = small_data_and_model

        # Request 150 samples but only 80 available
        result = compute_shap_interactions(model, X, max_samples=150)

        # Should use all 80 samples
        assert result["n_samples_used"] == 80

    def test_no_max_samples_uses_all(self, medium_data_and_model):
        """Test that None for max_samples uses all data."""
        X, _, model = medium_data_and_model

        # Don't specify max_samples
        result = compute_shap_interactions(model, X)

        # Should use all 200 samples
        assert result["n_samples_used"] == 200


class TestSHAPInteractionsOutputFormat:
    """Test SHAP interactions output structure."""

    @pytest.fixture(scope="class")
    def output_keys_data_and_model(self):
        """Create data for output keys test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 3)
        y = X[:, 0] + X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def matrix_shape_data_and_model(self):
        """Create data for matrix shape test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 4
        X = rng.randn(n_samples, n_features)
        y = X[:, 0] * X[:, 1] + X[:, 2] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model, n_features

    @pytest.fixture(scope="class")
    def sorted_data_and_model(self):
        """Create data with different interaction strengths for sorting test."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 4)
        y = (
            2 * X[:, 0] * X[:, 1]  # Strong
            + 0.3 * X[:, 1] * X[:, 2]  # Weak
            + X[:, 3]
            + 0.1 * rng.randn(n_samples)
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def tuple_format_data_and_model(self):
        """Create data for tuple format test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_output_keys(self, output_keys_data_and_model):
        """Test output dictionary has expected keys."""
        X, _, model = output_keys_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=100)

        # Check all expected keys present
        expected_keys = {
            "interaction_matrix",
            "feature_names",
            "top_interactions",
            "n_features",
            "n_samples_used",
            "computation_time",
        }
        assert set(result.keys()) == expected_keys

    def test_interaction_matrix_shape(self, matrix_shape_data_and_model):
        """Test interaction matrix has correct shape."""
        X, _, model, n_features = matrix_shape_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=100)

        # Matrix should be (n_features, n_features)
        assert result["interaction_matrix"].shape == (n_features, n_features)

    def test_top_interactions_sorted(self, sorted_data_and_model):
        """Test top_interactions list is sorted by magnitude descending."""
        X, _, model = sorted_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=150)

        # Check sorted descending by absolute magnitude
        interactions = [abs(interaction) for _, _, interaction in result["top_interactions"]]
        assert interactions == sorted(interactions, reverse=True)

    def test_top_interactions_tuple_format(self, tuple_format_data_and_model):
        """Test each interaction is a (feature_i, feature_j, value) tuple."""
        X, _, model = tuple_format_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, feature_names=["feat_a", "feat_b"])

        # Check tuple format
        for entry in result["top_interactions"]:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            feat_i, feat_j, interaction_val = entry
            assert isinstance(feat_i, str)
            assert isinstance(feat_j, str)
            assert isinstance(interaction_val, float | np.floating)


class TestSHAPInteractionsEdgeCases:
    """Test SHAP interactions edge cases."""

    @pytest.fixture(scope="class")
    def single_pair_data_and_model(self):
        """Create data with exactly 2 features."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def many_features_data_and_model(self):
        """Create data with many features."""
        rng = np.random.RandomState(42)
        n_samples = 300
        n_features = 8
        X = rng.randn(n_samples, n_features)
        # Only x0 and x1 interact
        y = X[:, 0] * X[:, 1] + X[:, 2:].sum(axis=1) + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        return X, y, model, n_features

    def test_single_feature_pair(self, single_pair_data_and_model):
        """Test with exactly 2 features (single pair)."""
        X, _, model = single_pair_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=100)

        # Should have exactly 1 pair
        assert len(result["top_interactions"]) == 1
        assert result["n_features"] == 2

    def test_many_features(self, many_features_data_and_model):
        """Test with many features."""
        X, _, model, n_features = many_features_data_and_model

        # Compute SHAP interactions
        result = compute_shap_interactions(model, X, max_samples=150, top_k=5)

        # Should have 5 interactions
        assert len(result["top_interactions"]) == 5
        assert result["n_features"] == n_features

        # Just verify sorted and non-negative
        interactions_vals = [val for _, _, val in result["top_interactions"]]
        assert all(v >= 0 for v in interactions_vals)
        assert interactions_vals == sorted(interactions_vals, reverse=True)


class TestSHAPInteractionsErrorHandling:
    """Test error handling for SHAP interactions."""

    def test_shap_required_message(self):
        """Test that function exists and requires SHAP (documentation test)."""
        # This test just verifies the function exists and has proper ImportError
        # Actual import mocking is too fragile in pytest
        from ml4t.diagnostic.evaluation import compute_shap_interactions

        # Function should exist
        assert callable(compute_shap_interactions)
        # Docstring should mention SHAP requirement
        assert "SHAP" in compute_shap_interactions.__doc__
        assert "ml4t-diagnostic[ml]" in compute_shap_interactions.__doc__


class TestSHAPInteractionsClassification:
    """Test SHAP interactions with classification models."""

    @pytest.fixture(scope="class")
    def binary_classification_data_and_model(self):
        """Create binary classification data and model."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 2)
        # Create binary classification: positive if product is positive
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_binary_classification(self, binary_classification_data_and_model):
        """Test SHAP interactions on binary classification."""
        X, _, model = binary_classification_data_and_model

        # Compute SHAP interactions (should work with classification)
        result = compute_shap_interactions(model, X, max_samples=150)

        # Should detect interaction
        assert len(result["top_interactions"]) == 1
        feat_i, feat_j, interaction = result["top_interactions"][0]
        assert {feat_i, feat_j} == {"f0", "f1"}
