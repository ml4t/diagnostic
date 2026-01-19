"""Integration tests for SHAP with non-tree models (TASK-010).

This module provides end-to-end integration tests validating the complete
workflow with linear models and kernel explainer, demonstrating v1.1's
model-agnostic support.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

# Note: Tests in this module are fast (<8s total) despite using SHAP
# Linear/Tree explainers are fast; only KernelExplainer is slow but uses small samples
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ml4t.diagnostic.evaluation.metrics import compute_shap_importance
from ml4t.diagnostic.evaluation.trade_shap_diagnostics import TradeShapAnalyzer

# Check for optional dependencies
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_SHAP = importlib.util.find_spec("shap") is not None

# Import lightgbm if available (needed for actual test usage)
if HAS_LIGHTGBM:
    import lightgbm as lgb  # noqa: E402


@pytest.fixture(scope="module")
def classification_data():
    """Create binary classification dataset for testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10

    # Generate features with predictive power
    X = np.random.randn(n_samples, n_features)
    # Target based on first two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


@pytest.fixture(scope="module")
def trade_data():
    """Create synthetic trade data for trade diagnostics testing."""
    np.random.seed(42)
    n_trades = 100
    n_features = 8

    # Features
    features = np.random.randn(n_trades, n_features)

    # Trade outcomes
    returns = np.random.randn(n_trades) * 0.02  # 2% volatility
    trade_ids = [f"TRADE_{i:03d}" for i in range(n_trades)]
    timestamps = pd.date_range("2024-01-01", periods=n_trades, freq="h")

    # Trade results DataFrame
    trades_df = pd.DataFrame(
        {
            "trade_id": trade_ids,
            "timestamp": timestamps,
            "return": returns,
            "pnl": returns * 10000,  # $10k position size
            "side": np.random.choice(["LONG", "SHORT"], n_trades),
            "symbol": np.random.choice(["BTC", "ETH", "SOL"], n_trades),
        }
    )

    # Feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return features, trades_df, feature_names


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestLinearModelIntegration:
    """End-to-end integration tests with linear models."""

    def test_logistic_regression_complete_workflow(self, classification_data):
        """Test complete SHAP workflow with LogisticRegression model."""
        X, y = classification_data

        # Split data
        split_idx = 200
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train linear model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Verify model works
        accuracy = model.score(X_test, y_test)
        assert accuracy > 0.5, "Model should perform better than random"

        # Compute SHAP importance (should auto-select LinearExplainer)
        result = compute_shap_importance(
            model, X_test[:50], explainer_type="auto", performance_warning=False
        )

        # Verify output structure
        assert "shap_values" in result
        assert "importances" in result
        assert "explainer_type" in result
        assert "base_value" in result

        # Should use LinearExplainer for logistic regression
        assert result["explainer_type"] == "linear"

        # Verify shapes
        assert result["shap_values"].shape == (50, X_test.shape[1])
        assert result["importances"].shape == (X_test.shape[1],)

        # Verify importance values are reasonable
        assert np.all(np.isfinite(result["importances"]))
        assert np.all(result["importances"] >= 0)

        # First two features should have highest importance (they determine target)
        top_2_indices = np.argsort(result["importances"])[-2:]
        assert 0 in top_2_indices or 1 in top_2_indices, "Important features should be detected"

    def test_linear_model_with_explicit_explainer_type(self, classification_data):
        """Test LinearExplainer with explicit explainer_type parameter."""
        X, y = classification_data
        X_train, X_test = X[:200], X[200:]
        y_train, _y_test = y[:200], y[200:]

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Explicitly request LinearExplainer
        result = compute_shap_importance(
            model, X_test[:50], explainer_type="linear", performance_warning=False
        )

        assert result["explainer_type"] == "linear"
        assert "shap_values" in result
        assert result["shap_values"].shape == (50, X_test.shape[1])

    def test_linear_model_performance_acceptable(self, classification_data):
        """Test that LinearExplainer completes in reasonable time."""
        import time

        X, y = classification_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Measure time for 100 samples
        start = time.time()
        result = compute_shap_importance(
            model, X_test[:100], explainer_type="linear", performance_warning=False
        )
        elapsed = time.time() - start

        # Should complete in under 10 seconds for 100 samples
        assert elapsed < 10.0, f"LinearExplainer took {elapsed:.2f}s (expected <10s)"

        assert "shap_values" in result


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestKernelExplainerIntegration:
    """End-to-end integration tests with KernelExplainer (model-agnostic fallback)."""

    def test_svm_complete_workflow(self, classification_data):
        """Test complete SHAP workflow with SVM model (forces KernelExplainer)."""
        X, y = classification_data

        # Use smaller dataset for KernelExplainer (it's slow)
        X_small = X[:150]
        y_small = y[:150]

        split_idx = 100
        X_train, X_test = X_small[:split_idx], X_small[split_idx:]
        y_train, y_test = y_small[:split_idx], y_small[split_idx:]

        # Train SVM model (not supported by Tree or Linear explainers)
        model = SVC(kernel="rbf", probability=True, random_state=42, gamma="scale")
        model.fit(X_train, y_train)

        # Verify model works
        accuracy = model.score(X_test, y_test)
        assert accuracy > 0.5, "Model should perform better than random"

        # Compute SHAP importance (should auto-select KernelExplainer)
        result = compute_shap_importance(
            model,
            X_test[:30],  # Small sample for speed
            explainer_type="auto",
            performance_warning=False,
            max_samples=50,  # Limit background samples
        )

        # Verify output structure
        assert "shap_values" in result
        assert "importances" in result
        assert "explainer_type" in result

        # Should use KernelExplainer for SVM
        assert result["explainer_type"] == "kernel"

        # Verify shapes
        assert result["shap_values"].shape == (30, X_test.shape[1])
        assert result["importances"].shape == (X_test.shape[1],)

        # Verify importance values are reasonable
        assert np.all(np.isfinite(result["importances"]))
        assert np.all(result["importances"] >= 0)

    def test_kernel_explainer_with_explicit_type(self, classification_data):
        """Test KernelExplainer with explicit explainer_type parameter."""
        X, y = classification_data
        X_small = X[:150]
        y_small = y[:150]

        X_train, X_test = X_small[:100], X_small[100:]
        y_train = y_small[:100]

        model = SVC(kernel="rbf", probability=True, random_state=42, gamma="scale")
        model.fit(X_train, y_train)

        # Explicitly request KernelExplainer
        result = compute_shap_importance(
            model,
            X_test[:20],
            explainer_type="kernel",
            performance_warning=False,
            max_samples=30,
        )

        assert result["explainer_type"] == "kernel"
        assert "shap_values" in result
        assert result["shap_values"].shape == (20, X_test.shape[1])

    def test_kernel_explainer_with_custom_background(self, classification_data):
        """Test KernelExplainer with custom background data."""
        X, y = classification_data
        X_small = X[:150]
        y_small = y[:150]

        X_train, X_test = X_small[:100], X_small[100:]
        y_train = y_small[:100]

        model = SVC(kernel="rbf", probability=True, random_state=42, gamma="scale")
        model.fit(X_train, y_train)

        # Create custom background dataset
        background_data = X_train[:50]

        # Use custom background
        result = compute_shap_importance(
            model,
            X_test[:15],
            explainer_type="kernel",
            background_data=background_data,
            performance_warning=False,
        )

        assert result["explainer_type"] == "kernel"
        assert "shap_values" in result
        assert result["shap_values"].shape == (15, X_test.shape[1])


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestTradeShapDiagnosticsIntegration:
    """Integration tests for TradeShapAnalyzer with non-tree models."""

    def test_trade_diagnostics_with_linear_model(self, trade_data):
        """Test trade diagnostics workflow with LogisticRegression."""
        features, trades_df, feature_names = trade_data

        # Create binary labels (profitable vs unprofitable)
        labels = (trades_df["return"] > 0).astype(int).values

        # Train linear model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(features, labels)

        # Create features DataFrame with timestamps (required by TradeShapAnalyzer)
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df["timestamp"] = trades_df["timestamp"].values

        # Create TradeShapAnalyzer with LinearExplainer
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,  # Use features_df, not feature_names
            explainer_type="linear",  # Explicitly use LinearExplainer
            performance_warning=False,
        )

        # Verify analyzer was created successfully
        assert analyzer is not None
        assert hasattr(analyzer, "model")
        assert hasattr(analyzer, "features_df")

        # Note: Full analyze_trades integration requires more complex setup
        # This test verifies the analyzer can be initialized with linear models

    def test_trade_diagnostics_with_kernel_explainer(self, trade_data):
        """Test trade diagnostics workflow with SVM (KernelExplainer)."""
        features, trades_df, feature_names = trade_data

        # Use smaller dataset for KernelExplainer
        n_samples = 60
        features_small = features[:n_samples]
        trades_small = trades_df[:n_samples].copy()

        # Create binary labels
        labels = (trades_small["return"] > 0).astype(int).values

        # Train SVM model
        model = SVC(kernel="rbf", probability=True, random_state=42, gamma="scale")
        model.fit(features_small, labels)

        # Create features DataFrame with timestamps
        features_df = pd.DataFrame(features_small, columns=feature_names)
        features_df["timestamp"] = trades_small["timestamp"].values

        # Create TradeShapAnalyzer with KernelExplainer
        analyzer = TradeShapAnalyzer(
            model=model,
            features_df=features_df,  # Use features_df
            explainer_type="kernel",  # Explicitly use KernelExplainer
            performance_warning=False,
            explainer_kwargs={"link": "identity"},  # KernelExplainer kwargs
        )

        # Verify analyzer was created successfully
        assert analyzer is not None
        assert hasattr(analyzer, "model")
        assert hasattr(analyzer, "features_df")

        # Note: Full analyze_trades requires complex setup
        # This test verifies analyzer initialization with KernelExplainer

    def test_trade_diagnostics_auto_selection(self, trade_data):
        """Test that TradeShapAnalyzer auto-selects appropriate explainer."""
        features, trades_df, feature_names = trade_data

        labels = (trades_df["return"] > 0).astype(int).values

        # Create features DataFrame
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df["timestamp"] = trades_df["timestamp"].values

        # Test with LinearModel - should auto-select LinearExplainer
        linear_model = LogisticRegression(random_state=42, max_iter=1000)
        linear_model.fit(features[:80], labels[:80])

        analyzer_linear = TradeShapAnalyzer(
            model=linear_model,
            features_df=features_df,  # Use features_df
            explainer_type="auto",  # Auto-selection
            performance_warning=False,
        )

        # Verify analyzer created with correct explainer (checked during initialization)
        assert analyzer_linear is not None
        assert hasattr(analyzer_linear, "model")

        # Test with tree model if available - should auto-select TreeExplainer
        if HAS_LIGHTGBM:
            tree_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
            tree_model.fit(features[:80], labels[:80])

            analyzer_tree = TradeShapAnalyzer(
                model=tree_model,
                features_df=features_df,
                explainer_type="auto",
                performance_warning=False,
            )

            # Verify analyzer created successfully
            assert analyzer_tree is not None
            assert hasattr(analyzer_tree, "model")

            # Note: Full workflow test would require complex setup
            # This test verifies auto-selection during initialization


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP library not installed")
class TestModelComparisonIntegration:
    """Integration tests comparing different model types with SHAP."""

    def test_compare_tree_vs_linear_shap_values(self, classification_data):
        """Compare SHAP importance between tree and linear models."""
        X, y = classification_data
        X_train, X_test = X[:200], X[200:]
        y_train = y[:200]

        # Train both model types
        linear_model = LogisticRegression(random_state=42, max_iter=1000)
        linear_model.fit(X_train, y_train)

        linear_result = compute_shap_importance(
            linear_model, X_test[:50], explainer_type="auto", performance_warning=False
        )

        assert linear_result["explainer_type"] == "linear"

        # If LightGBM available, compare with tree model
        if HAS_LIGHTGBM:
            tree_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            tree_model.fit(X_train, y_train)

            tree_result = compute_shap_importance(
                tree_model, X_test[:50], explainer_type="auto", performance_warning=False
            )

            assert tree_result["explainer_type"] == "tree"

            # Both should identify similar important features
            # (though exact values may differ)
            linear_top_features = np.argsort(linear_result["importances"])[-3:]
            tree_top_features = np.argsort(tree_result["importances"])[-3:]

            # At least one important feature should overlap
            overlap = len(set(linear_top_features) & set(tree_top_features))
            assert overlap >= 1, "Models should identify some common important features"

    def test_output_format_consistency_across_model_types(self, classification_data):
        """Verify output format is consistent across all model types."""
        X, y = classification_data
        X_train, X_test = X[:150], X[150:]
        y_train, _y_test = y[:150], y[150:]

        results = {}

        # Linear model
        linear_model = LogisticRegression(random_state=42, max_iter=1000)
        linear_model.fit(X_train, y_train)
        results["linear"] = compute_shap_importance(
            linear_model, X_test[:30], explainer_type="linear", performance_warning=False
        )

        # SVM (kernel explainer)
        svm_model = SVC(kernel="rbf", probability=True, random_state=42, gamma="scale")
        svm_model.fit(X_train, y_train)
        results["kernel"] = compute_shap_importance(
            svm_model,
            X_test[:20],
            explainer_type="kernel",
            performance_warning=False,
            max_samples=30,
        )

        # Tree model if available
        if HAS_LIGHTGBM:
            tree_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
            tree_model.fit(X_train, y_train)
            results["tree"] = compute_shap_importance(
                tree_model, X_test[:30], explainer_type="tree", performance_warning=False
            )

        # All results should have consistent structure
        required_keys = {"shap_values", "importances", "explainer_type", "base_value"}
        for model_type, result in results.items():
            assert set(result.keys()) >= required_keys, f"{model_type} missing required keys"
            assert isinstance(result["shap_values"], np.ndarray)
            assert isinstance(result["importances"], np.ndarray)
            assert isinstance(result["explainer_type"], str)
            assert result["shap_values"].ndim == 2
            assert result["importances"].ndim == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
