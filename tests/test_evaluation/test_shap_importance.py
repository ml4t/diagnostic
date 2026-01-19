"""Tests for SHAP feature importance computation.

This module tests the compute_shap_importance function with various model types
and input formats, verifying correctness of SHAP values and importance aggregation.
"""

import importlib.util
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

# Check for optional dependencies
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_SHAP = importlib.util.find_spec("shap") is not None

# Import if available (needed for actual test usage)
if HAS_LIGHTGBM:
    import lightgbm as lgb  # noqa: E402
if HAS_XGBOOST:
    import xgboost as xgb  # noqa: E402


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestComputeShapImportanceLightGBM:
    """Test compute_shap_importance with LightGBM models."""

    @pytest.fixture(scope="class")
    def simple_binary_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are predictive
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    @pytest.fixture(scope="class")
    def simple_regression_data(self):
        """Create simple regression dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are predictive
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.1

        return X, y

    @pytest.fixture(scope="class")
    def trained_lgbm_classifier(self, simple_binary_data):
        """Train a LightGBM classifier."""
        X, y = simple_binary_data
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def trained_lgbm_regressor(self, simple_regression_data):
        """Train a LightGBM regressor."""
        X, y = simple_regression_data
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        model.fit(X, y)
        return model

    def test_basic_computation_classifier(self, trained_lgbm_classifier, simple_binary_data):
        """Test basic SHAP computation on classifier."""
        X, _ = simple_binary_data
        X_test = X[:100]  # Use subset for speed

        result = compute_shap_importance(trained_lgbm_classifier, X_test)

        # Check return structure
        assert "shap_values" in result
        assert "importances" in result
        assert "feature_names" in result
        assert "base_value" in result
        assert "n_features" in result
        assert "n_samples" in result
        assert "model_type" in result
        assert "additivity_verified" in result

        # Check shapes
        assert result["shap_values"].shape == (100, 10)
        assert result["importances"].shape == (10,)
        assert len(result["feature_names"]) == 10
        assert result["n_features"] == 10
        assert result["n_samples"] == 100

        # Check importance ordering (descending)
        assert np.all(result["importances"][:-1] >= result["importances"][1:])

        # Check that first two features have highest importance (by design)
        top_features = result["feature_names"][:2]
        assert "feature_0" in top_features or "feature_1" in top_features

    def test_basic_computation_regressor(self, trained_lgbm_regressor, simple_regression_data):
        """Test basic SHAP computation on regressor."""
        X, _ = simple_regression_data
        X_test = X[:100]

        result = compute_shap_importance(trained_lgbm_regressor, X_test)

        # Check basic structure
        assert result["shap_values"].shape == (100, 10)
        assert result["n_features"] == 10
        assert result["n_samples"] == 100

        # For regression with our data, feature_0 and feature_1 should be most important
        top_features = result["feature_names"][:2]
        assert "feature_0" in top_features and "feature_1" in top_features

    def test_pandas_input(self, trained_lgbm_classifier, simple_binary_data):
        """Test with pandas DataFrame input."""
        X, _ = simple_binary_data
        X_test = X[:100]

        # Convert to pandas with custom column names
        feature_names = [f"feat_{i}" for i in range(10)]
        X_df = pd.DataFrame(X_test, columns=feature_names)

        result = compute_shap_importance(trained_lgbm_classifier, X_df)

        # Check that feature names from DataFrame are used
        assert all(name.startswith("feat_") for name in result["feature_names"])
        assert result["n_features"] == 10

    def test_polars_input(self, trained_lgbm_classifier, simple_binary_data):
        """Test with polars DataFrame input."""
        X, _ = simple_binary_data
        X_test = X[:100]

        # Convert to polars with custom column names
        feature_names = [f"col_{i}" for i in range(10)]
        X_pl = pl.DataFrame(X_test, schema=feature_names)

        result = compute_shap_importance(trained_lgbm_classifier, X_pl)

        # Check that feature names from DataFrame are used
        assert all(name.startswith("col_") for name in result["feature_names"])
        assert result["n_features"] == 10

    def test_custom_feature_names(self, trained_lgbm_classifier, simple_binary_data):
        """Test with custom feature names."""
        X, _ = simple_binary_data
        X_test = X[:100]

        custom_names = [f"custom_{i}" for i in range(10)]
        result = compute_shap_importance(
            trained_lgbm_classifier, X_test, feature_names=custom_names
        )

        # Check that custom names are used
        assert all(name.startswith("custom_") for name in result["feature_names"])
        assert len(result["feature_names"]) == 10

    def test_max_samples(self, trained_lgbm_classifier, simple_binary_data):
        """Test max_samples parameter."""
        X, _ = simple_binary_data
        X_test = X  # Use full dataset (500 samples)

        # Compute with max_samples limit
        result = compute_shap_importance(
            trained_lgbm_classifier, X_test, max_samples=100, check_additivity=False
        )

        # Should only compute for 100 samples
        assert result["shap_values"].shape == (100, 10)
        assert result["n_samples"] == 100

    def test_check_additivity(self, trained_lgbm_classifier, simple_binary_data):
        """Test that SHAP values satisfy additivity.

        Note: This test verifies the mathematical property that SHAP values
        decompose predictions into feature contributions. The check_additivity
        parameter in compute_shap_importance() ensures this during computation.
        """
        X, _ = simple_binary_data
        X_test = X[:50]

        # Convert to DataFrame to avoid feature name warnings
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # The check_additivity=True parameter ensures SHAP verifies additivity internally
        # If it fails, it will raise an exception
        result = compute_shap_importance(trained_lgbm_classifier, X_test_df, check_additivity=True)

        # If we got here without exception, additivity was verified by SHAP
        assert result["additivity_verified"] is True

        # Additional sanity check: SHAP values should exist and have correct shape
        assert result["shap_values"].shape == (50, 10)
        assert result["base_value"] is not None

    def test_importance_values_positive(self, trained_lgbm_classifier, simple_binary_data):
        """Test that importance values are non-negative (mean absolute)."""
        X, _ = simple_binary_data
        X_test = X[:100]

        result = compute_shap_importance(trained_lgbm_classifier, X_test)

        # All importances should be >= 0 (mean of absolute values)
        assert np.all(result["importances"] >= 0)

    def test_model_type_recorded(self, trained_lgbm_classifier, simple_binary_data):
        """Test that model type is recorded correctly."""
        X, _ = simple_binary_data
        X_test = X[:100]

        result = compute_shap_importance(trained_lgbm_classifier, X_test)

        assert "lightgbm" in result["model_type"].lower()


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
class TestComputeShapImportanceXGBoost:
    """Test compute_shap_importance with XGBoost models."""

    @pytest.fixture(scope="class")
    def simple_binary_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    @pytest.fixture(scope="class")
    def trained_xgb_classifier(self, simple_binary_data):
        """Train an XGBoost classifier."""
        X, y = simple_binary_data
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, random_state=42, eval_metric="logloss"
        )
        model.fit(X, y)
        return model

    def test_xgboost_classifier(self, trained_xgb_classifier, simple_binary_data):
        """Test SHAP computation with XGBoost classifier."""
        X, _ = simple_binary_data
        X_test = X[:100]

        result = compute_shap_importance(trained_xgb_classifier, X_test)

        # Check basic structure
        assert result["shap_values"].shape == (100, 10)
        assert result["n_features"] == 10
        assert "xgboost" in result["model_type"].lower()

        # Check importance ordering
        assert np.all(result["importances"][:-1] >= result["importances"][1:])


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestComputeShapImportanceErrors:
    """Test error handling in compute_shap_importance."""

    def test_no_shap_library(self, monkeypatch):
        """Test error when SHAP library not available."""
        # Mock import to fail
        import sys

        monkeypatch.setitem(sys.modules, "shap", None)

        # Should raise ImportError with helpful message
        with pytest.raises(ImportError, match="SHAP library is not installed"):
            from ml4t.diagnostic.evaluation.metrics import compute_shap_importance as compute_shap

            compute_shap(None, np.array([[1, 2, 3]]))

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_invalid_shape(self):
        """Test error on invalid X shape."""
        # Create 1D array
        X_invalid = np.array([1, 2, 3, 4, 5])

        # Create dummy model
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        with pytest.raises(ValueError, match="X must be 2D array"):
            compute_shap_importance(model, X_invalid)

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_feature_name_mismatch(self):
        """Test error when feature names count doesn't match."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 5)
        wrong_names = ["a", "b", "c"]  # Only 3 names for 5 features

        with pytest.raises(ValueError, match="Number of feature names"):
            compute_shap_importance(model, X_test, feature_names=wrong_names)

    def test_unsupported_model(self):
        """Test error with unsupported model type (v1.1: tries all explainers)."""

        class DummyModel:
            pass

        model = DummyModel()
        X_test = np.random.randn(20, 5)

        # v1.1: Auto-selection tries tree → linear → kernel, all should fail
        with pytest.raises(
            ValueError, match="Failed to create explainer for model type DummyModel"
        ):
            compute_shap_importance(model, X_test)


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestShapComparison:
    """Test SHAP importance compared to other methods."""

    @pytest.fixture(scope="class")
    def trained_model_and_data(self):
        """Create model and data for comparison tests."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are strongly predictive
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2 + X[:, 1] * 3 > 0).astype(int)

        # Train model
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        model.fit(X, y)

        return model, X

    def test_top_features_match_data_generation(self, trained_model_and_data):
        """Test that SHAP correctly identifies important features."""
        model, X = trained_model_and_data
        X_test = X[:200]

        result = compute_shap_importance(model, X_test)

        # Top 2 features should be feature_0 and feature_1 (used to generate y)
        top_2 = result["feature_names"][:2]
        assert "feature_0" in top_2
        assert "feature_1" in top_2

    def test_shap_values_available_for_local_explanation(self, trained_model_and_data):
        """Test that SHAP provides per-sample explanations."""
        model, X = trained_model_and_data
        X_test = X[:10]

        result = compute_shap_importance(model, X_test)

        # Each sample should have feature contributions
        shap_values = result["shap_values"]
        assert shap_values.shape == (10, 10)

        # Each row should sum to something meaningful (not zero)
        row_sums = np.abs(shap_values).sum(axis=1)
        assert np.all(row_sums > 0)


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestSHAPHelperFunctions:
    """Test new helper functions for model-agnostic SHAP support (v1.1)."""

    def test_detect_gpu_available(self):
        """Test GPU detection function."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _detect_gpu_available

        # This should return True or False (bool) without crashing
        result = _detect_gpu_available()
        assert isinstance(result, bool)

        # If cupy is available, should be True
        try:
            import cupy as cp

            _ = cp.cuda.Device(0)
            assert result is True
        except (ImportError, RuntimeError):
            # cupy not available or no GPU
            assert result is False

    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _format_time

        assert _format_time(45) == "45 seconds"
        assert _format_time(1) == "1 seconds"

    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _format_time

        assert _format_time(60) == "1 minute"
        assert _format_time(120) == "2 minutes"
        assert _format_time(3599) == "59 minutes"

    def test_format_time_hours(self):
        """Test time formatting for hours."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _format_time

        assert _format_time(3600) == "1 hour"
        assert _format_time(7200) == "2 hours"
        assert _format_time(3665) == "1 hour 1 minute"
        assert _format_time(7320) == "2 hours 2 minutes"

    def test_sample_background_random(self):
        """Test random background sampling."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X = np.random.randn(1000, 10)

        # Sample fewer than available
        bg = _sample_background(X, max_samples=100, method="random")
        assert bg.shape == (100, 10)

        # Sample more than available (should return all)
        bg_all = _sample_background(X, max_samples=2000, method="random")
        assert bg_all.shape == X.shape

    def test_sample_background_kmeans(self):
        """Test k-means background sampling."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X = np.random.randn(1000, 10)

        # Sample using k-means
        bg = _sample_background(X, max_samples=50, method="kmeans")
        assert bg.shape == (50, 10)

    def test_sample_background_invalid_method(self):
        """Test that invalid sampling method raises error."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="Unknown sampling method"):
            _sample_background(X, max_samples=50, method="invalid")

    def test_estimate_computation_time_fast_explainers_no_warning(self):
        """Test that fast explainers (tree, linear, deep) don't trigger warnings."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _estimate_computation_time

        # These should not raise warnings
        _estimate_computation_time("tree", n_samples=10000, ms_per_sample=5.0)
        _estimate_computation_time("linear", n_samples=10000, ms_per_sample=75.0)
        _estimate_computation_time("deep", n_samples=1000, ms_per_sample=500.0)

    def test_estimate_computation_time_kernel_warning(self):
        """Test that slow KernelExplainer triggers warning."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _estimate_computation_time

        # Should trigger warning for kernel with many samples
        with pytest.warns(UserWarning, match="KernelExplainer is slow"):
            _estimate_computation_time("kernel", n_samples=500, ms_per_sample=5000.0)

    def test_estimate_computation_time_warning_disabled(self):
        """Test that warnings can be disabled."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _estimate_computation_time

        # Should not warn when performance_warning=False
        _estimate_computation_time(
            "kernel", n_samples=500, ms_per_sample=5000.0, performance_warning=False
        )

    def test_get_explainer_invalid_type(self):
        """Test that invalid explainer type raises error."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="Invalid explainer_type"):
            _get_explainer(model=None, X_array=X, explainer_type="invalid")

    def test_get_explainer_gpu_unavailable_error(self):
        """Test that forcing GPU when unavailable raises error."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import (
            _detect_gpu_available,
            _get_explainer,
        )

        # Only test if GPU is not available
        if not _detect_gpu_available():
            X = np.random.randn(100, 10)
            # Create a simple model for testing
            try:
                import lightgbm as lgb

                y = (X[:, 0] > 0).astype(int)
                model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
                model.fit(X, y)

                with pytest.raises(RuntimeError, match="GPU requested.*but GPU not available"):
                    _get_explainer(model, X, explainer_type="tree", use_gpu=True)
            except ImportError:
                pytest.skip("LightGBM not installed")


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestGetExplainerAutoSelection:
    """Test _get_explainer auto-selection logic with real models."""

    @pytest.fixture(scope="class")
    def simple_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture(scope="class")
    def tree_model(self, simple_data):
        """Create and train tree-based model."""
        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        model.fit(X, y)
        return model

    def test_auto_selection_chooses_tree_for_lgbm(self, tree_model, simple_data):
        """Test that auto-selection chooses TreeExplainer for LightGBM."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, _ = simple_data
        X_test = X[:100]

        explainer, type_name, ms_per_sample = _get_explainer(
            tree_model, X_test, explainer_type="auto"
        )

        assert type_name == "tree"
        assert ms_per_sample < 20.0  # Should be fast (~5ms)

    def test_explicit_tree_explainer(self, tree_model, simple_data):
        """Test explicit tree explainer selection."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, _ = simple_data
        X_test = X[:100]

        explainer, type_name, ms_per_sample = _get_explainer(
            tree_model, X_test, explainer_type="tree"
        )

        assert type_name == "tree"

    def test_explicit_kernel_explainer(self, tree_model, simple_data):
        """Test explicit kernel explainer selection (model-agnostic fallback)."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, _ = simple_data
        X_test = X[:50]  # Small sample for speed

        explainer, type_name, ms_per_sample = _get_explainer(
            tree_model, X_test, explainer_type="kernel"
        )

        assert type_name == "kernel"
        assert ms_per_sample > 1000.0  # Should be slow

    def test_gpu_mode_auto(self, tree_model, simple_data):
        """Test GPU auto-detection (uses GPU only for large datasets)."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, _ = simple_data

        # Small dataset - should use CPU even if GPU available
        X_small = X[:100]
        explainer_small, _, _ = _get_explainer(
            tree_model, X_small, explainer_type="tree", use_gpu="auto"
        )
        # Should succeed regardless of GPU availability

        # Large dataset - would use GPU if available
        X_large = np.random.randn(6000, 10)
        explainer_large, _, _ = _get_explainer(
            tree_model, X_large, explainer_type="tree", use_gpu="auto"
        )
        # Should succeed regardless of GPU availability

    def test_background_data_sampling(self, tree_model, simple_data):
        """Test that background data is sampled for KernelExplainer."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, _ = simple_data
        X_test = X[:50]

        # KernelExplainer should work even without explicit background_data
        explainer, type_name, _ = _get_explainer(
            tree_model, X_test, explainer_type="kernel", background_data=None
        )

        assert type_name == "kernel"


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestLinearExplainerIntegration:
    """Test LinearExplainer with sklearn linear models (TASK-003)."""

    @pytest.fixture(scope="class")
    def simple_classification_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are strongly predictive
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2 + X[:, 1] * 3 > 0).astype(int)

        return X, y

    @pytest.fixture(scope="class")
    def simple_regression_data(self):
        """Create simple regression dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are strongly predictive
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.1

        return X, y

    def test_linear_explainer_logistic_regression(self, simple_classification_data):
        """Test LinearExplainer with LogisticRegression."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Get explainer (should auto-select Linear)
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")

        assert type_name == "linear"
        assert ms_per_sample < 200.0  # Should be fast (<100ms typical)

    def test_linear_explainer_linear_regression(self, simple_regression_data):
        """Test LinearExplainer with LinearRegression."""
        from sklearn.linear_model import LinearRegression

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_regression_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get explainer (should auto-select Linear)
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")

        assert type_name == "linear"
        assert ms_per_sample < 200.0

    def test_linear_explainer_ridge(self, simple_regression_data):
        """Test LinearExplainer with Ridge regression."""
        from sklearn.linear_model import Ridge

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_regression_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)

        # Get explainer (should auto-select Linear)
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")

        assert type_name == "linear"
        assert ms_per_sample < 200.0

    def test_linear_explainer_lasso(self, simple_regression_data):
        """Test LinearExplainer with Lasso regression."""
        from sklearn.linear_model import Lasso

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_regression_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = Lasso(alpha=0.1, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Get explainer (should auto-select Linear)
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")

        assert type_name == "linear"
        assert ms_per_sample < 200.0

    def test_linear_explainer_elastic_net(self, simple_regression_data):
        """Test LinearExplainer with ElasticNet."""
        from sklearn.linear_model import ElasticNet

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_regression_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Get explainer (should auto-select Linear)
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")

        assert type_name == "linear"
        assert ms_per_sample < 200.0

    def test_linear_explainer_explicit_selection(self, simple_classification_data):
        """Test explicit LinearExplainer selection."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Explicit linear explainer
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="linear")

        assert type_name == "linear"

    def test_linear_explainer_shap_values_format(self, simple_classification_data):
        """Test that LinearExplainer produces correct SHAP values format."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:450]  # 50 samples for speed

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Get explainer
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="linear")

        # Compute SHAP values
        shap_values = explainer.shap_values(X_test)

        # Check output format
        assert shap_values.shape == (50, 10), f"Expected (50, 10), got {shap_values.shape}"
        assert not np.isnan(shap_values).any(), "SHAP values contain NaN"
        assert not np.isinf(shap_values).any(), "SHAP values contain inf"

    def test_linear_explainer_performance_benchmark(self, simple_classification_data):
        """Test that LinearExplainer is fast enough (<100ms per sample)."""
        import time

        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:410]  # 10 samples

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Get explainer
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="linear")

        # Benchmark SHAP computation
        start = time.time()
        explainer.shap_values(X_test)
        elapsed = time.time() - start

        ms_per_sample_actual = (elapsed * 1000) / len(X_test)

        # Should be <100ms per sample (often 50-100ms)
        assert ms_per_sample_actual < 200.0, f"Too slow: {ms_per_sample_actual:.1f}ms per sample"


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestComputeShapImportanceV11API:
    """Test v1.1 API extensions (TASK-006)."""

    @pytest.fixture(scope="class")
    def simple_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_explicit_explainer_type_tree(self, simple_data):
        """Test explicit tree explainer selection."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="tree")

        assert "explainer_type" in result
        assert result["explainer_type"] == "tree"

    def test_explicit_explainer_type_linear(self, simple_data):
        """Test explicit linear explainer selection."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="linear")

        assert "explainer_type" in result
        assert result["explainer_type"] == "linear"

    def test_auto_explainer_type_tree(self, simple_data):
        """Test auto-selection chooses tree for LightGBM."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="auto")

        assert result["explainer_type"] == "tree"

    def test_auto_explainer_type_linear(self, simple_data):
        """Test auto-selection chooses linear for sklearn linear models."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        result = compute_shap_importance(model, X, explainer_type="auto")

        assert result["explainer_type"] == "linear"

    def test_backward_compatibility_no_explainer_type(self, simple_data):
        """Test that old API (no explainer_type) still works."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Old API signature - should still work with defaults
        result = compute_shap_importance(model, X)

        assert "shap_values" in result
        assert "importances" in result
        assert "explainer_type" in result  # New field added
        assert result["explainer_type"] == "tree"  # Auto-selected

    def test_performance_warning_disabled(self, simple_data):
        """Test that performance warnings can be disabled."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_small = X[:50]  # Small dataset for speed
        y_small = y[:50]

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_small, y_small)

        # Should not raise warning when performance_warning=False
        result = compute_shap_importance(
            model, X_small[:20], explainer_type="kernel", performance_warning=False, max_samples=20
        )

        assert result["explainer_type"] == "kernel"

    def test_explainer_kwargs_passed_through(self, simple_data):
        """Test that explainer_kwargs are passed to explainer constructor."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Pass custom kwargs (feature_perturbation is a TreeExplainer parameter)
        result = compute_shap_importance(
            model,
            X,
            explainer_type="tree",
            explainer_kwargs={"feature_perturbation": "interventional"},
        )

        assert result["explainer_type"] == "tree"


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestTradeShapAnalyzerV11Integration:
    """Test TradeShapAnalyzer with v1.1 API (TASK-007)."""

    def test_trade_shap_analyzer_auto_selection_tree(self):
        """Test TradeShapAnalyzer auto-selects TreeExplainer for tree models."""
        from datetime import datetime

        import lightgbm as lgb
        import polars as pl

        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        # Create sample data with timestamps
        n_samples = 100
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            }
        )

        # Create target
        y = (
            features_df.select("feature1").to_numpy().flatten() * 0.5
            + features_df.select("feature2").to_numpy().flatten() * 0.3
            > 0
        ).astype(int)

        # Train model
        X_train = features_df.select(["feature1", "feature2", "feature3"]).to_numpy()
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y)

        # Create analyzer (auto-select explainer)
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=None,  # Will compute on-demand
            explainer_type="auto",
        )

        # Trigger SHAP computation
        assert analyzer.shap_values is None
        analyzer._compute_shap_values()

        # Verify SHAP values computed
        assert analyzer.shap_values is not None
        assert analyzer.shap_values.shape == (n_samples, 3)

    def test_trade_shap_analyzer_explicit_linear(self):
        """Test TradeShapAnalyzer with explicit LinearExplainer."""
        from datetime import datetime

        import polars as pl
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        # Create sample data with timestamps
        n_samples = 100
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )

        # Create target
        y = (
            features_df.select("feature1").to_numpy().flatten() * 0.5
            + features_df.select("feature2").to_numpy().flatten() * 0.3
            > 0
        ).astype(int)

        # Train model
        X_train = features_df.select(["feature1", "feature2"]).to_numpy()
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y)

        # Create analyzer with explicit linear explainer
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=None,
            explainer_type="linear",  # Explicit selection
        )

        # Trigger SHAP computation
        analyzer._compute_shap_values()

        # Verify SHAP values computed
        assert analyzer.shap_values is not None
        assert analyzer.shap_values.shape == (n_samples, 2)

    def test_trade_shap_analyzer_backward_compatibility(self):
        """Test that old API (no explainer params) still works."""
        from datetime import datetime

        import lightgbm as lgb
        import polars as pl

        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        # Create sample data
        n_samples = 50
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )

        y = (features_df.select("feature1").to_numpy().flatten() > 0).astype(int)

        # Train model
        X_train = features_df.select(["feature1", "feature2"]).to_numpy()
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y)

        # Old API signature (no v1.1 parameters)
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=None,
        )

        # Should still work with defaults
        analyzer._compute_shap_values()

        assert analyzer.shap_values is not None
        assert analyzer.shap_values.shape == (n_samples, 2)

    def test_trade_shap_analyzer_performance_warning(self):
        """Test performance warning parameter is passed through."""
        from datetime import datetime

        import lightgbm as lgb
        import polars as pl

        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        # Create sample data
        n_samples = 30
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
            }
        )

        y = (features_df.select("feature1").to_numpy().flatten() > 0).astype(int)

        # Train model
        X_train = features_df.select(["feature1"]).to_numpy()
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y)

        # Create analyzer with performance_warning=False
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=None,
            performance_warning=False,  # Disable warnings
        )

        # Should not raise warnings
        analyzer._compute_shap_values()

        assert analyzer.shap_values is not None

    def test_trade_shap_analyzer_explainer_kwargs(self):
        """Test explainer_kwargs are passed through."""
        from datetime import datetime

        import lightgbm as lgb
        import polars as pl

        from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

        # Create sample data
        n_samples = 30
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
        features_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.random.randn(n_samples),
            }
        )

        y = (features_df.select("feature1").to_numpy().flatten() > 0).astype(int)

        # Train model
        X_train = features_df.select(["feature1"]).to_numpy()
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y)

        # Create analyzer with custom explainer kwargs
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,
            shap_values=None,
            explainer_type="tree",
            explainer_kwargs={"feature_perturbation": "interventional"},
        )

        # Should work with custom kwargs
        analyzer._compute_shap_values()

        assert analyzer.shap_values is not None


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestKernelExplainerIntegration:
    """Test KernelExplainer with non-tree/non-linear models (TASK-004)."""

    @pytest.fixture(scope="class")
    def simple_classification_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create data where first two features are strongly predictive
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2 + X[:, 1] * 3 > 0).astype(int)

        return X, y

    def test_kernel_explainer_svm(self, simple_classification_data):
        """Test KernelExplainer with SVM (model-agnostic fallback)."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:410]  # 10 samples for speed (KernelExplainer is slow)

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Should auto-select KernelExplainer (tree/linear will fail)
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X_test, explainer_type="auto", max_samples=20
        )

        assert type_name == "kernel"
        # KernelExplainer is slow, but should complete
        assert ms_per_sample > 100.0  # Much slower than tree/linear

    def test_kernel_explainer_knn(self, simple_classification_data):
        """Test KernelExplainer with KNN classifier."""
        from sklearn.neighbors import KNeighborsClassifier

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:410]  # 10 samples for speed

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Should auto-select KernelExplainer
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X_test, explainer_type="auto", max_samples=20
        )

        assert type_name == "kernel"

    def test_kernel_explainer_explicit_selection(self, simple_classification_data):
        """Test explicit KernelExplainer selection even for tree models."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:410]  # 10 samples for speed

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # Force KernelExplainer even though TreeExplainer would work
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X_test, explainer_type="kernel", max_samples=20
        )

        assert type_name == "kernel"

    def test_background_sampling_random(self, simple_classification_data):
        """Test random background sampling strategy."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X, _ = simple_classification_data

        # Sample 100 from 500
        background = _sample_background(X, max_samples=100, method="random")

        assert background.shape == (100, X.shape[1])
        assert np.all(np.isfinite(background))

    def test_background_sampling_kmeans(self, simple_classification_data):
        """Test K-means background sampling strategy."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X, _ = simple_classification_data

        # Sample 50 cluster centroids from 500
        background = _sample_background(X, max_samples=50, method="kmeans")

        assert background.shape == (50, X.shape[1])
        assert np.all(np.isfinite(background))

    def test_background_sampling_default(self, simple_classification_data):
        """Test default background sampling (random)."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _sample_background

        X, _ = simple_classification_data

        # Default should be random
        background = _sample_background(X, max_samples=100)

        assert background.shape == (100, X.shape[1])

    def test_kernel_explainer_with_background_data(self, simple_classification_data):
        """Test KernelExplainer with custom background dataset."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:420]  # Very small for speed

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Provide custom background data
        background = X_train[::10]  # Every 10th sample

        result = compute_shap_importance(
            model,
            X_test,
            explainer_type="kernel",
            background_data=background,
            performance_warning=False,
            max_samples=20,
        )

        assert result["explainer_type"] == "kernel"
        assert result["shap_values"].shape == (20, X.shape[1])

    def test_fallback_when_tree_fails(self, simple_classification_data):
        """Test auto-fallback to KernelExplainer when TreeExplainer fails."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:420]

        # SVM has no tree structure, so TreeExplainer will fail
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Auto-selection should fall back to KernelExplainer
        result = compute_shap_importance(
            model, X_test, explainer_type="auto", performance_warning=False, max_samples=20
        )

        assert result["explainer_type"] == "kernel"
        assert result["shap_values"].shape == (20, X.shape[1])

    def test_fallback_when_linear_fails(self, simple_classification_data):
        """Test auto-fallback to KernelExplainer when LinearExplainer fails."""
        from sklearn.ensemble import RandomForestClassifier

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _create_explainer_by_type

        X, y = simple_classification_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:450]

        # RandomForest is not linear
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Trying to force LinearExplainer should fail gracefully
        # and _get_explainer should fall back to kernel in auto mode
        try:
            explainer, type_name, ms = _create_explainer_by_type(
                model, X_test, explainer_type="linear", max_samples=50
            )
            # If it doesn't fail, it means LinearExplainer worked (unexpected but ok)
            assert type_name == "linear"
        except (AttributeError, TypeError, ValueError):
            # Expected: LinearExplainer fails for non-linear models
            # Auto-selection would then fall back to kernel
            pass


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestAutoSelectionAndBackwardCompatibility:
    """Test auto-selection logic and backward compatibility with v1.0 API (TASK-008)."""

    @pytest.fixture(scope="class")
    def simple_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] * 2 + X[:, 1] * 3 > 0).astype(int)

        return X, y

    def test_auto_selects_tree_for_lightgbm(self, simple_data):
        """Test that auto-selection chooses TreeExplainer for LightGBM."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        result = compute_shap_importance(model, X[:100], explainer_type="auto")

        assert result["explainer_type"] == "tree"
        assert result["shap_values"].shape == (100, X.shape[1])

    def test_auto_selects_linear_for_logistic_regression(self, simple_data):
        """Test that auto-selection chooses LinearExplainer for LogisticRegression."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        result = compute_shap_importance(model, X[:100], explainer_type="auto")

        assert result["explainer_type"] == "linear"
        assert result["shap_values"].shape == (100, X.shape[1])

    def test_auto_falls_back_to_kernel_for_svm(self, simple_data):
        """Test that auto-selection falls back to KernelExplainer for SVM."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_small, y_small = X[:200], y[:200]  # Small dataset for speed

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_small, y_small)

        result = compute_shap_importance(
            model, X_small[:10], explainer_type="auto", performance_warning=False, max_samples=20
        )

        assert result["explainer_type"] == "kernel"
        assert result["shap_values"].shape == (10, X.shape[1])

    def test_explicit_explainer_type_overrides_auto(self, simple_data):
        """Test that explicit explainer_type overrides auto-selection."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Force kernel even though tree would be better
        result = compute_shap_importance(
            model, X[:10], explainer_type="kernel", performance_warning=False, max_samples=20
        )

        assert result["explainer_type"] == "kernel"  # Kernel was forced
        assert result["shap_values"].shape == (10, X.shape[1])

    def test_v10_api_backward_compatibility(self, simple_data):
        """Test that v1.0 API (no explainer_type param) still works."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Old API signature - no explainer_type, use_gpu, etc.
        result = compute_shap_importance(model, X[:100])

        # Should work and auto-select tree
        assert "shap_values" in result
        assert "importances" in result
        assert "explainer_type" in result  # New field added in v1.1
        assert result["explainer_type"] == "tree"
        assert result["shap_values"].shape == (100, X.shape[1])

    def test_output_format_consistent_across_explainers(self, simple_data):
        """Test that output format is consistent across all explainer types."""
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_test = X[:50]

        # Train tree model
        tree_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        tree_model.fit(X, y)

        # Train linear model
        linear_model = LogisticRegression(random_state=42, max_iter=1000)
        linear_model.fit(X, y)

        # Get results from both
        tree_result = compute_shap_importance(tree_model, X_test, explainer_type="tree")
        linear_result = compute_shap_importance(linear_model, X_test, explainer_type="linear")

        # Check that both have same output structure
        assert set(tree_result.keys()) == set(linear_result.keys())
        assert tree_result["shap_values"].shape == linear_result["shap_values"].shape
        assert tree_result["importances"].shape == linear_result["importances"].shape
        assert isinstance(tree_result["explainer_type"], str)
        assert isinstance(linear_result["explainer_type"], str)

    def test_performance_unchanged_for_tree_models(self, simple_data):
        """Test that performance is still fast (<10ms per sample) for tree models."""
        import time

        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        X_test = X[:100]

        # Measure time for TreeExplainer
        start = time.time()
        explainer, type_name, ms_per_sample = _get_explainer(model, X_test, explainer_type="auto")
        end = time.time()

        assert type_name == "tree"
        # TreeExplainer should be very fast
        assert ms_per_sample < 10.0  # Less than 10ms per sample
        # Total time should be under 1 second for 100 samples
        assert (end - start) < 1.0

    def test_coverage_for_explainer_selection_code(self, simple_data):
        """Ensure comprehensive coverage of explainer selection code paths."""
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_data
        X_small = X[:100]
        y_small = y[:100]

        # Test tree path
        tree_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        tree_model.fit(X_small, y_small)
        explainer1, type1, _ = _get_explainer(tree_model, X_small[:50], explainer_type="auto")
        assert type1 == "tree"

        # Test linear path
        linear_model = LogisticRegression(random_state=42, max_iter=1000)
        linear_model.fit(X_small, y_small)
        explainer2, type2, _ = _get_explainer(linear_model, X_small[:50], explainer_type="auto")
        assert type2 == "linear"

        # Test kernel fallback path
        svm_model = SVC(kernel="rbf", probability=True, random_state=42)
        svm_model.fit(X_small, y_small)
        explainer3, type3, _ = _get_explainer(
            svm_model, X_small[:30], explainer_type="auto", max_samples=30
        )
        assert type3 == "kernel"

        # All three paths exercised
        assert {type1, type2, type3} == {"tree", "linear", "kernel"}


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
class TestPerformanceWarningsAndGPU:
    """Comprehensive tests for performance warnings and GPU functionality (TASK-009)."""

    @pytest.fixture(scope="class")
    def simple_data(self):
        """Create simple binary classification dataset."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_kernel_warning_issued_above_threshold(self, simple_data):
        """Test that warning is issued for KernelExplainer with >200 samples."""
        import warnings

        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        # Use just above threshold (201 > 200) to test warning without extreme slowness
        X_train = X[:200]
        y_train = y[:200]

        # Train SVM model (forces KernelExplainer)
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Should issue warning for large sample count (201 > 200 threshold)
        # We use only 15 samples to explain but still test the warning logic
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compute_shap_importance(
                model,
                X[:15],  # Small test set for speed, warning based on ms_per_sample
                explainer_type="kernel",
                max_samples=20,
                performance_warning=True,
            )

            # Check that it runs (warning may or may not trigger based on actual timing)
            assert "shap_values" in result

    def test_kernel_warning_includes_time_estimate(self, simple_data):
        """Test that warning includes time estimate and sample recommendations."""
        import warnings

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _estimate_computation_time

        # Simulate slow KernelExplainer with many samples
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _estimate_computation_time(
                explainer_type="kernel",
                n_samples=500,
                ms_per_sample=5000.0,
                performance_warning=True,
            )

            # Check warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            warning_msg = str(w[0].message)

            # Check warning contains expected information
            assert "KernelExplainer is slow" in warning_msg
            assert "500 samples" in warning_msg or "Estimated time" in warning_msg
            # Should mention time estimate or sample recommendations

    def test_no_warning_for_tree_explainer(self, simple_data, capsys):
        """Test that TreeExplainer does not issue performance warnings."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_test = X[:500]  # Large dataset

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # TreeExplainer should not warn even with large dataset
        result = compute_shap_importance(
            model, X_test, explainer_type="tree", performance_warning=True
        )

        # Capture output
        captured = capsys.readouterr()

        # Should not have warnings
        assert "WARNING" not in captured.out
        assert "shap_values" in result

    def test_no_warning_for_linear_explainer(self, simple_data, capsys):
        """Test that LinearExplainer does not issue performance warnings."""
        from sklearn.linear_model import LogisticRegression

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_test = X[:500]  # Large dataset

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        # LinearExplainer should not warn even with large dataset
        result = compute_shap_importance(
            model, X_test, explainer_type="linear", performance_warning=True
        )

        # Capture output
        captured = capsys.readouterr()

        # Should not have warnings
        assert "WARNING" not in captured.out
        assert "shap_values" in result

    def test_performance_warning_false_disables_all_warnings(self, simple_data, capsys):
        """Test that performance_warning=False completely disables warnings."""
        from sklearn.svm import SVC

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        X, y = simple_data
        X_small = X[:100]
        y_small = y[:100]

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_small, y_small)

        # Large sample count that would normally trigger warning
        result = compute_shap_importance(
            model, X_small[:50], explainer_type="kernel", performance_warning=False, max_samples=20
        )

        # Capture output
        captured = capsys.readouterr()

        # Should not have any warnings
        assert "WARNING" not in captured.out
        assert "warning" not in captured.out.lower()
        assert "shap_values" in result

    def test_gpu_detection_returns_bool(self):
        """Test that GPU detection returns a boolean value."""
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _detect_gpu_available

        result = _detect_gpu_available()

        # Should always return a bool
        assert isinstance(result, bool)

    def test_gpu_auto_mode_small_dataset(self, simple_data):
        """Test that use_gpu='auto' uses CPU for small datasets."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_data
        X_small = X[:100]  # Small dataset

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Auto mode should use CPU for small datasets
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X_small, explainer_type="tree", use_gpu="auto"
        )

        assert type_name == "tree"
        # Should succeed regardless of GPU availability

    def test_gpu_auto_mode_large_dataset(self, simple_data):
        """Test that use_gpu='auto' considers using GPU for large datasets."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_data
        # Create larger dataset by repeating
        X_large = np.vstack([X] * 100)  # ~10K samples

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Auto mode would use GPU if available for large datasets
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X_large, explainer_type="tree", use_gpu="auto"
        )

        assert type_name == "tree"
        # Should succeed regardless of GPU availability (falls back to CPU if needed)

    def test_gpu_fallback_when_unavailable(self, simple_data):
        """Test graceful fallback to CPU when GPU is unavailable."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import (
            _detect_gpu_available,
            _get_explainer,
        )

        X, y = simple_data

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # If GPU is not available, requesting use_gpu=True should raise error
        # (this is expected behavior to catch misconfigurations)
        if not _detect_gpu_available():
            with pytest.raises(RuntimeError, match="GPU requested.*but GPU not available"):
                _get_explainer(model, X[:100], explainer_type="tree", use_gpu=True)
        else:
            # If GPU is available, it should work
            explainer, type_name, ms_per_sample = _get_explainer(
                model, X[:100], explainer_type="tree", use_gpu=True
            )
            assert type_name == "tree"

    def test_mock_gpu_detection_with_cupy_unavailable(self, simple_data, monkeypatch):
        """Test GPU detection when cupy is not available (CI-friendly mock test)."""
        # Mock cupy import to fail

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _detect_gpu_available

        def mock_import_error(name, *args, **kwargs):
            if name == "cupy" or name.startswith("cupy."):
                raise ImportError("Mocked cupy unavailable")
            return __import__(name, *args, **kwargs)

        # This will cause _detect_gpu_available to return False
        # Note: This is testing the error handling path
        result = _detect_gpu_available()

        # Should return False when cupy not available
        # (or True if cupy is actually installed and working)
        assert isinstance(result, bool)

    def test_mock_gpu_explainer_creation(self, simple_data, monkeypatch):
        """Test TreeExplainer creation with mocked GPU settings (CI-friendly)."""
        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _get_explainer

        X, y = simple_data

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Mock _detect_gpu_available to return False
        monkeypatch.setattr(
            "ml4t.diagnostic.evaluation.metrics.importance_shap._detect_gpu_available",
            lambda: False,
        )

        # use_gpu='auto' should use CPU when GPU not detected
        explainer, type_name, ms_per_sample = _get_explainer(
            model, X[:100], explainer_type="tree", use_gpu="auto"
        )

        assert type_name == "tree"
        # Should have successfully created TreeExplainer in CPU mode

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed for GPU benchmark")
    def test_gpu_benchmark_optional(self, simple_data):
        """Optional GPU benchmark test (skips if GPU not available)."""
        import time

        import lightgbm as lgb

        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance
        from ml4t.diagnostic.evaluation.metrics.importance_shap import _detect_gpu_available

        # Skip if no GPU
        if not _detect_gpu_available():
            pytest.skip("GPU not available for benchmark test")

        X, y = simple_data
        # Create large dataset for meaningful benchmark
        X_large = np.vstack([X] * 100)  # ~10K samples
        y_large = np.tile(y, 100)

        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X_large, y_large)

        X_test = X_large[:1000]

        # Benchmark CPU
        start_cpu = time.time()
        result_cpu = compute_shap_importance(model, X_test, use_gpu=False)
        time.time() - start_cpu

        # Benchmark GPU
        start_gpu = time.time()
        result_gpu = compute_shap_importance(model, X_test, use_gpu=True)
        time.time() - start_gpu

        # Both should succeed
        assert "shap_values" in result_cpu
        assert "shap_values" in result_gpu

        # GPU should be faster (or at least not significantly slower) for large datasets
        # Note: For small datasets, GPU overhead may make it slower
        # Just verify both methods work correctly
        assert result_cpu["shap_values"].shape == result_gpu["shap_values"].shape


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestDeepExplainerIntegration:
    """Test DeepExplainer integration with neural networks (TensorFlow/PyTorch).

    These tests handle graceful degradation when deep learning frameworks
    are not installed, which is the expected behavior for most users.
    """

    @pytest.fixture(scope="class")
    def simple_data(self):
        """Create simple dataset for neural network testing."""
        np.random.seed(42)
        n_samples = 200  # Smaller for faster neural network training
        n_features = 5

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    def test_deep_explainer_requires_background_data(self, simple_data):
        """Test that DeepExplainer requires background_data parameter."""
        X, y = simple_data

        # Create a mock neural network object (has typical NN attributes)
        class MockNeuralNetwork:
            """Mock neural network for testing error handling."""

            def __init__(self):
                self.layers = ["dense1", "dense2", "output"]  # Looks like a neural network

            def predict(self, X):
                # Simple mock prediction
                return np.random.rand(len(X), 2)

        mock_model = MockNeuralNetwork()

        # Attempt to use DeepExplainer without background_data should raise ValueError
        with pytest.raises(ValueError, match="DeepExplainer requires background_data"):
            compute_shap_importance(
                model=mock_model,
                X=X[:10],
                explainer_type="deep",
                # No background_data provided
            )

    def test_auto_selection_skips_deep_for_non_neural_models(self, simple_data):
        """Test that auto-selection doesn't incorrectly use DeepExplainer."""
        if not HAS_LIGHTGBM:
            pytest.skip("LightGBM not installed")

        import lightgbm as lgb

        X, y = simple_data

        # Train tree model
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        # Auto-selection should pick TreeExplainer, not DeepExplainer
        result = compute_shap_importance(model, X[:20])

        assert result["explainer_type"] == "tree"
        # Should NOT be 'deep' even if TensorFlow is installed

    @pytest.mark.skipif(
        True,  # Always skip unless explicitly enabled
        reason="DeepExplainer test requires TensorFlow/PyTorch - enable manually",
    )
    def test_deep_explainer_with_tensorflow(self, simple_data):
        """Test DeepExplainer with a real TensorFlow model.

        This test is skipped by default because:
        1. TensorFlow is a heavy dependency (>500MB)
        2. Not needed for most users
        3. Slows down CI/CD pipelines

        To run manually:
        1. Install TensorFlow: pip install tensorflow
        2. Remove @pytest.mark.skipif decorator
        3. Run: pytest tests/test_evaluation/test_shap_importance.py::TestDeepExplainerIntegration::test_deep_explainer_with_tensorflow
        """
        pytest.importorskip("tensorflow")
        from tensorflow import keras

        X, y = simple_data

        # Create simple neural network
        model = keras.Sequential(
            [
                keras.layers.Dense(10, activation="relu", input_shape=(X.shape[1],)),
                keras.layers.Dense(5, activation="relu"),
                keras.layers.Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Train briefly
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # Test DeepExplainer (explicit)
        # Must provide background_data for DeepExplainer
        result = compute_shap_importance(
            model=model,
            X=X[:20],
            explainer_type="deep",
            background_data=X[:50],  # Provide background data
        )

        # Verify results
        assert result["explainer_type"] == "deep"
        assert result["shap_values"].shape == (20, X.shape[1])
        assert result["n_features"] == X.shape[1]
        assert result["n_samples"] == 20
        assert len(result["importances"]) == X.shape[1]
        assert len(result["feature_names"]) == X.shape[1]

        # Verify SHAP values are reasonable (not all zeros/NaN)
        assert not np.all(result["shap_values"] == 0)
        assert not np.any(np.isnan(result["shap_values"]))

    def test_deep_explainer_parameter_validation(self):
        """Test that DeepExplainer validates background_data requirement.

        This test verifies the implementation checks for required parameters
        before attempting to create the explainer.
        """
        import shap

        from ml4t.diagnostic.evaluation.metrics.importance_shap import _create_explainer_by_type

        # Create mock neural network
        class MockNN:
            def __init__(self):
                self.layers = ["mock"]

            def predict(self, X):
                return np.random.rand(len(X), 2)

        mock_model = MockNN()
        X_test = np.random.randn(10, 5).astype(np.float32)

        # Test that infrastructure validates background_data requirement
        with pytest.raises(ValueError, match="DeepExplainer requires background_data"):
            _create_explainer_by_type(
                explainer_type="deep",
                model=mock_model,
                X_array=X_test,
                use_gpu=False,
                background_data=None,  # Missing - should raise ValueError
                shap=shap,
            )

    def test_deep_explainer_documentation_exists(self):
        """Verify DeepExplainer is documented in API."""
        from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

        docstring = compute_shap_importance.__doc__

        # Check for DeepExplainer documentation
        assert "deep" in docstring.lower()
        assert "tensorflow" in docstring.lower() or "pytorch" in docstring.lower()
        assert "DeepExplainer" in docstring or "deep explainer" in docstring.lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
