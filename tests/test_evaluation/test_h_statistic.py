"""Tests for H-statistic feature interaction detection (TASK-051).

Tests train RandomForest models and compute H-statistics.
Optimized with reduced n_estimators (10) and grid_resolution (10) for fast execution (~2.5s).
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml4t.diagnostic.evaluation import compute_h_statistic


class TestHStatisticBasic:
    """Test basic H-statistic computation."""

    @pytest.fixture(scope="class")
    def additive_data_and_model(self):
        """Create additive data (no interaction) and train model."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=10)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def xor_data_and_model(self):
        """Create XOR interaction data and train model."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 2)
        y = np.where(X[:, 0] * X[:, 1] > 0, 1.0, -1.0) + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(
            n_estimators=10, random_state=42, max_depth=10, min_samples_leaf=5
        )
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def product_data_and_model(self):
        """Create multiplicative interaction data and train model."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(
            n_estimators=10, random_state=42, max_depth=10, min_samples_leaf=5
        )
        model.fit(X, y)
        return X, y, model

    def test_no_interaction(self, additive_data_and_model):
        """Test H-statistic on additive features (no interaction)."""
        X, _, model = additive_data_and_model

        result = compute_h_statistic(
            model, X, feature_names=["x0", "x1"], n_samples=100, grid_resolution=10
        )

        # Should have low H-statistic (no interaction)
        assert len(result["h_statistics"]) == 1  # Only one pair
        feat_i, feat_j, h_val = result["h_statistics"][0]
        assert h_val < 0.7  # Should be lower than true interaction cases

    def test_strong_interaction_xor(self, xor_data_and_model):
        """Test H-statistic on XOR interaction."""
        X, _, model = xor_data_and_model

        result = compute_h_statistic(
            model, X, feature_names=["x0", "x1"], n_samples=150, grid_resolution=10
        )

        # Should have high H-statistic (strong interaction)
        feat_i, feat_j, h_val = result["h_statistics"][0]
        assert h_val > 0.4  # Strong interaction expected

    def test_strong_interaction_product(self, product_data_and_model):
        """Test H-statistic on multiplicative interaction."""
        X, _, model = product_data_and_model

        result = compute_h_statistic(
            model, X, feature_names=["x0", "x1"], n_samples=150, grid_resolution=10
        )

        # Should have high H-statistic (strong interaction)
        feat_i, feat_j, h_val = result["h_statistics"][0]
        assert h_val > 0.5  # Very strong interaction expected


class TestHStatisticMultipleFeatures:
    """Test H-statistic with multiple features."""

    @pytest.fixture(scope="class")
    def multi_interaction_data_and_model(self):
        """Create data with multiple interactions and train model."""
        rng = np.random.RandomState(42)
        n_samples = 400
        X = rng.randn(n_samples, 5)
        y = (
            X[:, 0] * X[:, 1]  # Strong interaction
            + X[:, 2]
            + X[:, 3]
            + 0.2 * X[:, 2] * X[:, 3]  # Weak interaction
            + X[:, 4]  # Independent
            + 0.1 * rng.randn(n_samples)
        )
        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=12)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def specific_pairs_data_and_model(self):
        """Create data for specific pairs test."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 5)
        y = X[:, 0] * X[:, 2] + X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_identify_top_interactions(self, multi_interaction_data_and_model):
        """Test identifying top interactions among multiple features."""
        X, _, model = multi_interaction_data_and_model

        result = compute_h_statistic(model, X, n_samples=200, grid_resolution=10)

        # Should detect (x0, x1) as top interaction
        h_stats = result["h_statistics"]
        assert len(h_stats) == 10  # 5 features → 10 pairs

        # Top interaction should be x0-x1
        top_feat_i, top_feat_j, top_h = h_stats[0]
        assert {top_feat_i, top_feat_j} == {"f0", "f1"}
        assert top_h > 0.4

        # x2-x3 should have moderate H
        x2_x3_h = next(h for fi, fj, h in h_stats if {fi, fj} == {"f2", "f3"})
        assert 0.1 < x2_x3_h < 0.5

    def test_specific_feature_pairs(self, specific_pairs_data_and_model):
        """Test computing H-statistic for specific pairs only."""
        X, _, model = specific_pairs_data_and_model

        result = compute_h_statistic(
            model, X, feature_pairs=[(0, 2), (1, 3)], n_samples=150, grid_resolution=10
        )

        # Should only have 2 results
        assert len(result["h_statistics"]) == 2
        assert result["n_pairs_tested"] == 2

        # (0, 2) should have higher H than (1, 3)
        h_02 = next(h for fi, fj, h in result["h_statistics"] if {fi, fj} == {"f0", "f2"})
        h_13 = next(h for fi, fj, h in result["h_statistics"] if {fi, fj} == {"f1", "f3"})
        assert h_02 > h_13


class TestHStatisticInputFormats:
    """Test H-statistic with different input formats."""

    @pytest.fixture(scope="class")
    def pandas_data_and_model(self):
        """Create pandas DataFrame and train model."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X_df = pd.DataFrame(
            rng.randn(n_samples, 3), columns=["feature_a", "feature_b", "feature_c"]
        )
        y = X_df["feature_a"] * X_df["feature_b"] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        return X_df, y, model

    @pytest.fixture(scope="class")
    def polars_data_and_model(self):
        """Create polars DataFrame and train model."""
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
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_pl.to_numpy(), y)
        return X_pl, y, model

    @pytest.fixture(scope="class")
    def numpy_data_and_model(self):
        """Create numpy data and train model."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 3)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def custom_names_data_and_model(self):
        """Create data for custom names test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_pandas_dataframe(self, pandas_data_and_model):
        """Test with pandas DataFrame input."""
        X_df, _, model = pandas_data_and_model

        result = compute_h_statistic(model, X_df, n_samples=100, grid_resolution=10)

        # Feature names should come from DataFrame
        assert result["feature_names"] == ["feature_a", "feature_b", "feature_c"]
        assert result["n_features"] == 3
        assert result["n_pairs_tested"] == 3  # 3 choose 2

        # Top pair should be feature_a × feature_b
        top_i, top_j, top_h = result["h_statistics"][0]
        assert {top_i, top_j} == {"feature_a", "feature_b"}

    def test_polars_dataframe(self, polars_data_and_model):
        """Test with polars DataFrame input."""
        X_pl, _, model = polars_data_and_model

        result = compute_h_statistic(model, X_pl, n_samples=100, grid_resolution=10)

        # Feature names should come from polars columns
        assert result["feature_names"] == ["feat_x", "feat_y", "feat_z"]
        assert result["n_features"] == 3

    def test_numpy_array(self, numpy_data_and_model):
        """Test with numpy array input."""
        X, _, model = numpy_data_and_model

        result = compute_h_statistic(model, X, n_samples=100, grid_resolution=10)

        # Feature names should be f0, f1, f2
        assert result["feature_names"] == ["f0", "f1", "f2"]
        assert result["n_features"] == 3

    def test_custom_feature_names(self, custom_names_data_and_model):
        """Test with custom feature names for numpy array."""
        X, _, model = custom_names_data_and_model

        result = compute_h_statistic(
            model, X, feature_names=["momentum", "volatility"], n_samples=100
        )

        # Should use custom names
        assert result["feature_names"] == ["momentum", "volatility"]
        feat_i, feat_j, h_val = result["h_statistics"][0]
        assert {feat_i, feat_j} == {"momentum", "volatility"}


class TestHStatisticParameters:
    """Test H-statistic parameter handling."""

    @pytest.fixture(scope="class")
    def large_data_and_model(self):
        """Create large dataset for subsampling test."""
        rng = np.random.RandomState(42)
        n_samples = 1000
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def small_data_and_model(self):
        """Create small dataset for no-subsampling test."""
        rng = np.random.RandomState(42)
        n_samples = 80
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def grid_resolution_data_and_model(self):
        """Create data for grid resolution test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def pairs_name_data_and_model(self):
        """Create pandas DataFrame for pairs-by-name test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X_df = pd.DataFrame(rng.randn(n_samples, 4), columns=["a", "b", "c", "d"])
        y = X_df["a"] * X_df["c"] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        return X_df, y, model

    def test_subsampling(self, large_data_and_model):
        """Test subsampling when n_samples < data size."""
        X, _, model = large_data_and_model

        result = compute_h_statistic(model, X, n_samples=150, grid_resolution=10)

        # Should indicate 150 samples used
        assert result["n_samples_used"] == 150

    def test_no_subsampling_small_data(self, small_data_and_model):
        """Test no subsampling when data already small."""
        X, _, model = small_data_and_model

        result = compute_h_statistic(model, X, n_samples=150, grid_resolution=10)

        # Should use all 80 samples
        assert result["n_samples_used"] == 80

    @pytest.mark.xfail(
        strict=False,
        reason="Timing comparison can fail under parallel load - computation times vary",
    )
    def test_grid_resolution(self, grid_resolution_data_and_model):
        """Test grid resolution parameter."""
        X, _, model = grid_resolution_data_and_model

        # Test different grid resolutions (8 vs 10 for faster tests)
        result_coarse = compute_h_statistic(model, X, n_samples=100, grid_resolution=8)
        result_fine = compute_h_statistic(model, X, n_samples=100, grid_resolution=10)

        assert result_coarse["grid_resolution"] == 8
        assert result_fine["grid_resolution"] == 10

        # Fine grid should take longer
        assert result_fine["computation_time"] > result_coarse["computation_time"]

    def test_feature_pairs_with_names(self, pairs_name_data_and_model):
        """Test specifying feature pairs by name."""
        X_df, _, model = pairs_name_data_and_model

        result = compute_h_statistic(
            model, X_df, feature_pairs=[("a", "c"), ("b", "d")], n_samples=100
        )

        assert result["n_pairs_tested"] == 2
        pairs = {(fi, fj) for fi, fj, _ in result["h_statistics"]}
        assert pairs == {("a", "c"), ("c", "a"), ("b", "d"), ("d", "b")}.intersection(pairs)


class TestHStatisticOutputFormat:
    """Test H-statistic output structure."""

    @pytest.fixture(scope="class")
    def output_keys_data_and_model(self):
        """Create data for output keys test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 3)
        y = X[:, 0] + X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def sorted_data_and_model(self):
        """Create data with different interaction strengths."""
        rng = np.random.RandomState(42)
        n_samples = 300
        X = rng.randn(n_samples, 4)
        y = (
            2 * X[:, 0] * X[:, 1]  # Strong
            + 0.5 * X[:, 1] * X[:, 2]  # Weak
            + X[:, 3]  # Independent
            + 0.1 * rng.randn(n_samples)
        )
        model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=10)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def tuple_format_data_and_model(self):
        """Create data for tuple format test."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_output_keys(self, output_keys_data_and_model):
        """Test output dictionary has expected keys."""
        X, _, model = output_keys_data_and_model

        result = compute_h_statistic(model, X, n_samples=100, grid_resolution=10)

        # Check all expected keys present
        expected_keys = {
            "h_statistics",
            "feature_names",
            "n_features",
            "n_pairs_tested",
            "n_samples_used",
            "grid_resolution",
            "computation_time",
        }
        assert set(result.keys()) == expected_keys

    def test_h_statistics_sorted(self, sorted_data_and_model):
        """Test h_statistics list is sorted by H value descending."""
        X, _, model = sorted_data_and_model

        result = compute_h_statistic(model, X, n_samples=150, grid_resolution=10)

        # Check sorted descending
        h_values = [h for _, _, h in result["h_statistics"]]
        assert h_values == sorted(h_values, reverse=True)

    def test_h_statistics_tuple_format(self, tuple_format_data_and_model):
        """Test each H-statistic is a (feature_i, feature_j, H_value) tuple."""
        X, _, model = tuple_format_data_and_model

        result = compute_h_statistic(model, X, feature_names=["feat_a", "feat_b"])

        # Check tuple format
        for entry in result["h_statistics"]:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            feat_i, feat_j, h_val = entry
            assert isinstance(feat_i, str)
            assert isinstance(feat_j, str)
            assert isinstance(h_val, float)
            assert 0 <= h_val <= 1  # H should be in [0, 1]


class TestHStatisticEdgeCases:
    """Test H-statistic edge cases."""

    @pytest.fixture(scope="class")
    def single_pair_data_and_model(self):
        """Create data with exactly 2 features (single pair)."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def constant_feature_data_and_model(self):
        """Create data with constant feature (no variation)."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = np.column_stack(
            [
                np.ones(n_samples),  # Constant
                rng.randn(n_samples),
            ]
        )
        y = X[:, 1] + 0.1 * rng.randn(n_samples)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def perfect_prediction_data_and_model(self):
        """Create data with no noise for perfect prediction."""
        rng = np.random.RandomState(42)
        n_samples = 200
        X = rng.randn(n_samples, 2)
        y = X[:, 0] * X[:, 1]  # No noise
        # Train overfitting model
        model = RandomForestRegressor(
            n_estimators=10, random_state=42, max_depth=None, min_samples_leaf=1
        )
        model.fit(X, y)
        return X, y, model

    def test_single_feature_pair(self, single_pair_data_and_model):
        """Test with exactly 2 features (single pair)."""
        X, _, model = single_pair_data_and_model

        # Compute H-statistic
        result = compute_h_statistic(model, X, n_samples=100)

        # Should have exactly 1 pair
        assert len(result["h_statistics"]) == 1
        assert result["n_pairs_tested"] == 1
        assert result["n_features"] == 2

    def test_constant_features(self, constant_feature_data_and_model):
        """Test with constant features (no variation)."""
        X, _, model = constant_feature_data_and_model

        # Compute H-statistic (should not crash)
        result = compute_h_statistic(model, X, n_samples=100, grid_resolution=10)

        # Should complete without error
        assert result["n_pairs_tested"] == 1

    def test_perfect_prediction(self, perfect_prediction_data_and_model):
        """Test when model perfectly predicts (no noise)."""
        X, _, model = perfect_prediction_data_and_model

        # Compute H-statistic
        result = compute_h_statistic(model, X, n_samples=100, grid_resolution=10)

        # Should still compute H-statistic
        feat_i, feat_j, h_val = result["h_statistics"][0]
        assert 0 <= h_val <= 1


# ============================================================================
# Fast tests (no slow marker) for interpretation and mocked SHAP
# ============================================================================


class TestGenerateInteractionInterpretation:
    """Test interpretation generation (no model training required)."""

    def test_strong_consensus(self):
        """Test interpretation with strong consensus."""
        from ml4t.diagnostic.evaluation.metrics.interactions import (
            _generate_interaction_interpretation,
        )

        top_interactions = [("a", "b"), ("c", "d"), ("e", "f")]
        method_agreement = {("h_stat", "shap"): 0.85, ("h_stat", "ic"): 0.78}
        warnings = []
        n_consensus = 3

        result = _generate_interaction_interpretation(
            top_interactions, method_agreement, warnings, n_consensus
        )

        assert "Strong consensus" in result
        assert "3 interactions" in result
        assert "(a, b)" in result
        assert "High agreement" in result

    def test_weak_consensus(self):
        """Test interpretation with weak consensus."""
        from ml4t.diagnostic.evaluation.metrics.interactions import (
            _generate_interaction_interpretation,
        )

        top_interactions = [("a", "b")]
        method_agreement = {("h_stat", "shap"): 0.3}
        warnings = []
        n_consensus = 0

        result = _generate_interaction_interpretation(
            top_interactions, method_agreement, warnings, n_consensus
        )

        assert "Weak consensus" in result
        assert "Low agreement" in result

    def test_moderate_agreement(self):
        """Test interpretation with moderate agreement."""
        from ml4t.diagnostic.evaluation.metrics.interactions import (
            _generate_interaction_interpretation,
        )

        top_interactions = [("a", "b"), ("c", "d")]
        method_agreement = {("h_stat", "shap"): 0.6, ("h_stat", "ic"): 0.55}
        warnings = []
        n_consensus = 1

        result = _generate_interaction_interpretation(
            top_interactions, method_agreement, warnings, n_consensus
        )

        assert "Moderate agreement" in result

    def test_with_warnings(self):
        """Test interpretation includes warnings."""
        from ml4t.diagnostic.evaluation.metrics.interactions import (
            _generate_interaction_interpretation,
        )

        top_interactions = [("a", "b")]
        method_agreement = {}
        warnings = ["Method X failed", "Low data quality"]
        n_consensus = 0

        result = _generate_interaction_interpretation(
            top_interactions, method_agreement, warnings, n_consensus
        )

        assert "Potential Issues" in result
        assert "Method X failed" in result
        assert "Low data quality" in result

    def test_empty_method_agreement(self):
        """Test interpretation with no method agreement data."""
        from ml4t.diagnostic.evaluation.metrics.interactions import (
            _generate_interaction_interpretation,
        )

        top_interactions = [("a", "b")]
        method_agreement = {}  # Empty
        warnings = []
        n_consensus = 1

        result = _generate_interaction_interpretation(
            top_interactions, method_agreement, warnings, n_consensus
        )

        # Should not crash and should produce some output
        assert isinstance(result, str)
        assert len(result) > 0


class TestComputeShapInteractions:
    """Test SHAP interaction computation with mocking."""

    def test_shap_not_installed(self, monkeypatch):
        """Test error when SHAP not installed."""

        from ml4t.diagnostic.evaluation.metrics.interactions import compute_shap_interactions

        # Mock shap to raise ImportError
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("No module named 'shap'")
            return original_import(name, *args, **kwargs)

        # Create simple model and data
        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()
        X = np.random.randn(50, 3)

        # Patch the import in the module
        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="SHAP is required"):
            compute_shap_interactions(model, X)

    def test_shap_with_pandas_input(self, monkeypatch):
        """Test SHAP interactions with pandas input (mocked)."""
        from unittest.mock import MagicMock, patch

        from ml4t.diagnostic.evaluation.metrics.interactions import compute_shap_interactions

        # Create mock SHAP module
        mock_shap = MagicMock()
        mock_explainer = MagicMock()

        # Create fake interaction values: (n_samples, n_features, n_features)
        n_samples, n_features = 50, 3
        fake_interactions = np.random.randn(n_samples, n_features, n_features)
        mock_explainer.shap_interaction_values.return_value = fake_interactions
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Patch shap import
        with patch.dict("sys.modules", {"shap": mock_shap}):
            # Create model and data
            class MockModel:
                def predict(self, X):
                    return X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0]

            model = MockModel()
            X_df = pd.DataFrame(np.random.randn(n_samples, n_features), columns=["a", "b", "c"])

            result = compute_shap_interactions(model, X_df, max_samples=30)

            assert result["feature_names"] == ["a", "b", "c"]
            assert result["n_features"] == 3
            assert result["n_samples_used"] == 30  # Subsampled
            assert "interaction_matrix" in result
            assert "top_interactions" in result
            assert "computation_time" in result

    def test_shap_with_polars_input(self, monkeypatch):
        """Test SHAP interactions with polars input (mocked)."""
        from unittest.mock import MagicMock, patch

        from ml4t.diagnostic.evaluation.metrics.interactions import compute_shap_interactions

        # Create mock SHAP module
        mock_shap = MagicMock()
        mock_explainer = MagicMock()

        n_samples, n_features = 40, 2
        fake_interactions = np.random.randn(n_samples, n_features, n_features)
        mock_explainer.shap_interaction_values.return_value = fake_interactions
        mock_shap.TreeExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):

            class MockModel:
                def predict(self, X):
                    return X[:, 0]

            model = MockModel()
            X_pl = pl.DataFrame(
                {"feat_x": np.random.randn(n_samples), "feat_y": np.random.randn(n_samples)}
            )

            result = compute_shap_interactions(model, X_pl)

            assert result["feature_names"] == ["feat_x", "feat_y"]
            assert result["n_features"] == 2

    def test_shap_top_k_limit(self, monkeypatch):
        """Test SHAP interactions with top_k limit."""
        from unittest.mock import MagicMock, patch

        from ml4t.diagnostic.evaluation.metrics.interactions import compute_shap_interactions

        mock_shap = MagicMock()
        mock_explainer = MagicMock()

        n_samples, n_features = 50, 5
        fake_interactions = np.random.randn(n_samples, n_features, n_features)
        mock_explainer.shap_interaction_values.return_value = fake_interactions
        mock_shap.TreeExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):

            class MockModel:
                def predict(self, X):
                    return X[:, 0]

            model = MockModel()
            X = np.random.randn(n_samples, n_features)

            result = compute_shap_interactions(model, X, top_k=3)

            # Should limit to top 3 interactions
            assert len(result["top_interactions"]) == 3

    def test_shap_binary_classification_format(self, monkeypatch):
        """Test SHAP with binary classification list format."""
        from unittest.mock import MagicMock, patch

        from ml4t.diagnostic.evaluation.metrics.interactions import compute_shap_interactions

        mock_shap = MagicMock()
        mock_explainer = MagicMock()

        n_samples, n_features = 50, 3
        # Binary classification returns list of 2 arrays
        fake_interactions_class0 = np.random.randn(n_samples, n_features, n_features)
        fake_interactions_class1 = np.random.randn(n_samples, n_features, n_features)
        mock_explainer.shap_interaction_values.return_value = [
            fake_interactions_class0,
            fake_interactions_class1,
        ]
        mock_shap.TreeExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):

            class MockModel:
                def predict(self, X):
                    return X[:, 0]

            model = MockModel()
            X = np.random.randn(n_samples, n_features)

            result = compute_shap_interactions(model, X)

            # Should use positive class (index 1)
            assert result["interaction_matrix"].shape == (n_features, n_features)


class TestAnalyzeInteractions:
    """Test comprehensive interaction analysis."""

    def test_no_methods_specified(self):
        """Test error when no methods specified."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="At least one method must be specified"):
            analyze_interactions(model, X, y, methods=[])

    def test_conditional_ic_only(self):
        """Test with conditional IC method only (no model dependency)."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        rng = np.random.RandomState(42)
        n_samples = 100
        X = rng.randn(n_samples, 3)
        # Create target with some correlation to features
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * rng.randn(n_samples)

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()

        result = analyze_interactions(model, X, y, methods=["conditional_ic"])

        assert "method_results" in result
        assert "conditional_ic" in result["method_results"]
        assert "consensus_ranking" in result
        assert "methods_run" in result
        assert "conditional_ic" in result["methods_run"]

    def test_invalid_feature_pairs(self):
        """Test error with invalid feature pairs."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()
        X_df = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randn(50)

        # Unknown feature name
        with pytest.raises(ValueError, match="unknown features"):
            analyze_interactions(model, X_df, y, feature_pairs=[("a", "unknown")])

    def test_feature_pairs_wrong_length(self):
        """Test error with wrong pair length."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()
        X_df = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="must have exactly 2 elements"):
            analyze_interactions(model, X_df, y, feature_pairs=[("a", "b", "c")])

    def test_all_methods_fail(self, monkeypatch):
        """Test error when all methods fail."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        class BrokenModel:
            def predict(self, X):
                raise RuntimeError("Model broken")

        model = BrokenModel()
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Monkeypatch conditional IC to also fail
        def failing_ic(*args, **kwargs):
            raise RuntimeError("IC broken")

        monkeypatch.setattr(
            "ml4t.diagnostic.evaluation.metrics.interactions.compute_conditional_ic",
            failing_ic,
        )

        with pytest.raises(ValueError, match="All methods failed"):
            analyze_interactions(model, X, y, methods=["conditional_ic", "h_statistic"])

    def test_method_failure_captured(self):
        """Test that method failures are captured, not raised."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        class BrokenModel:
            def predict(self, X):
                raise RuntimeError("Model broken")

        model = BrokenModel()
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = 0.5 * X[:, 0] + rng.randn(100) * 0.1

        # Run with conditional_ic (works) and h_statistic (fails)
        result = analyze_interactions(model, X, y, methods=["conditional_ic", "h_statistic"])

        # Should complete with partial results
        assert "conditional_ic" in result["methods_run"]
        assert len(result["methods_failed"]) > 0
        assert any("h_statistic" in f[0] for f in result["methods_failed"])

    def test_pandas_input(self):
        """Test with pandas DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        rng = np.random.RandomState(42)
        n_samples = 100
        X_df = pd.DataFrame(rng.randn(n_samples, 3), columns=["feat_a", "feat_b", "feat_c"])
        y = pd.Series(0.5 * X_df["feat_a"] + 0.3 * X_df["feat_b"] + 0.1 * rng.randn(n_samples))

        class MockModel:
            def predict(self, X):
                return X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0].values

        model = MockModel()

        result = analyze_interactions(model, X_df, y, methods=["conditional_ic"])

        # Should use DataFrame column names
        assert any("feat_a" in str(r) for r in result["consensus_ranking"])

    def test_polars_input(self):
        """Test with polars DataFrame input."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        rng = np.random.RandomState(42)
        n_samples = 100
        X_pl = pl.DataFrame(
            {
                "col_x": rng.randn(n_samples),
                "col_y": rng.randn(n_samples),
            }
        )
        y = pl.Series(0.5 * X_pl["col_x"].to_numpy() + 0.1 * rng.randn(n_samples))

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()

        result = analyze_interactions(model, X_pl, y, methods=["conditional_ic"])

        assert any("col_x" in str(r) for r in result["consensus_ranking"])

    def test_specific_feature_pairs(self):
        """Test with specific feature pairs specified."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        rng = np.random.RandomState(42)
        n_samples = 100
        X_df = pd.DataFrame(rng.randn(n_samples, 4), columns=["a", "b", "c", "d"])
        y = rng.randn(n_samples)

        class MockModel:
            def predict(self, X):
                return X[:, 0] if isinstance(X, np.ndarray) else X.iloc[:, 0].values

        model = MockModel()

        result = analyze_interactions(
            model, X_df, y, feature_pairs=[("a", "b"), ("c", "d")], methods=["conditional_ic"]
        )

        # Should only test 2 pairs
        assert len(result["consensus_ranking"]) == 2

    def test_output_structure(self):
        """Test output contains all expected keys."""
        from ml4t.diagnostic.evaluation.metrics.interactions import analyze_interactions

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = X[:, 0] + 0.1 * rng.randn(100)

        class MockModel:
            def predict(self, X):
                return X[:, 0]

        model = MockModel()

        result = analyze_interactions(model, X, y, methods=["conditional_ic"])

        expected_keys = {
            "method_results",
            "consensus_ranking",
            "method_agreement",
            "top_interactions_consensus",
            "warnings",
            "interpretation",
            "methods_run",
            "methods_failed",
        }
        assert set(result.keys()) == expected_keys
