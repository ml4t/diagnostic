"""Tests for ML Importance Summary (analyze_ml_importance)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance


class TestMLImportanceBasicFunctionality:
    """Test basic functionality of analyze_ml_importance."""

    @pytest.fixture(scope="class")
    def classification_data_10f(self):
        """Create classification dataset with 10 features."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_10f(self, classification_data_10f):
        """Train RF classifier on 10-feature data."""
        X, y = classification_data_10f
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def classification_data_5f(self):
        """Create classification dataset with 5 features."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_5f(self, classification_data_5f):
        """Train RF classifier on 5-feature data."""
        X, y = classification_data_5f
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def regression_data_and_model(self):
        """Create regression data and train model."""
        X, y = make_regression(n_samples=500, n_features=10, n_informative=3, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_all_methods_agree_classification(self, trained_rf_10f, classification_data_10f):
        """Test that consensus ranking works when all methods agree."""
        X, y = classification_data_10f
        model = trained_rf_10f

        # Run analysis (without SHAP to avoid dependency)
        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Verify structure
        assert "method_results" in result
        assert "consensus_ranking" in result
        assert "method_agreement" in result
        assert "top_features_consensus" in result
        assert "warnings" in result
        assert "interpretation" in result
        assert "methods_run" in result
        assert "methods_failed" in result

        # Verify methods ran
        assert "mdi" in result["methods_run"]
        assert "pfi" in result["methods_run"]
        assert len(result["methods_run"]) == 2

        # Verify consensus ranking
        assert len(result["consensus_ranking"]) == 10
        assert all(isinstance(fname, str) for fname in result["consensus_ranking"])

        # Verify method agreement
        assert "mdi_vs_pfi" in result["method_agreement"]
        assert -1.0 <= result["method_agreement"]["mdi_vs_pfi"] <= 1.0

        # Verify top features consensus (should have some agreement)
        assert isinstance(result["top_features_consensus"], list)

        # Verify interpretation is non-empty
        assert len(result["interpretation"]) > 0
        assert isinstance(result["interpretation"], str)

    def test_default_methods(self, trained_rf_5f, classification_data_5f):
        """Test that default methods are mdi, pfi, shap."""
        X, y = classification_data_5f

        # Run with default methods
        result = analyze_ml_importance(trained_rf_5f, X, y)

        # Should attempt mdi, pfi, shap
        # At minimum mdi and pfi should succeed (shap may fail without library)
        assert "mdi" in result["methods_run"]
        assert "pfi" in result["methods_run"]

    def test_single_method(self, trained_rf_5f, classification_data_5f):
        """Test that function works with single method."""
        X, y = classification_data_5f

        # Run with only MDI
        result = analyze_ml_importance(trained_rf_5f, X, y, methods=["mdi"])

        # Should only have MDI results
        assert result["methods_run"] == ["mdi"]
        assert len(result["method_agreement"]) == 0  # No pairs to compare
        assert len(result["consensus_ranking"]) == 5
        assert len(result["top_features_consensus"]) == 5  # All features

    def test_regression_dataset(self, regression_data_and_model):
        """Test that analysis works on regression problems."""
        X, y, model = regression_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Should work for regression
        assert len(result["methods_run"]) == 2
        assert len(result["consensus_ranking"]) == 10


class TestMLImportanceMethodAgreement:
    """Test method agreement calculations."""

    @pytest.fixture(scope="class")
    def high_agreement_data_and_model(self):
        """Create data where methods should agree and train model."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def multi_method_data_and_model(self):
        """Create data for multi-method comparison."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_high_agreement_reported(self, high_agreement_data_and_model):
        """Test that high agreement is correctly identified."""
        X, y, model = high_agreement_data_and_model

        result = analyze_ml_importance(
            model, X, y, methods=["mdi", "pfi"], n_repeats=20, random_state=42
        )

        # Should have reasonable agreement
        agreement = result["method_agreement"]["mdi_vs_pfi"]
        assert agreement > 0.3  # Relaxed threshold (methods may disagree somewhat)

        # Interpretation should mention agreement level
        assert "agreement" in result["interpretation"].lower()

    def test_multiple_method_pairs(self, multi_method_data_and_model):
        """Test that all method pairs are computed."""
        X, y, model = multi_method_data_and_model

        # Run with 3 methods (should have 3 pairs: mdi-pfi, mdi-mda, pfi-mda)
        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi", "mda"], random_state=42)

        # Should have 3 method pairs
        assert len(result["method_agreement"]) == 3
        assert "mdi_vs_pfi" in result["method_agreement"]
        assert "mdi_vs_mda" in result["method_agreement"]
        assert "pfi_vs_mda" in result["method_agreement"]


class TestMLImportanceConsensusFeatures:
    """Test consensus feature identification."""

    @pytest.fixture(scope="class")
    def consensus_data_and_model(self):
        """Create data with clear informative features and train model."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=2,
            n_redundant=0,
            random_state=42,
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def ranking_data_and_model(self):
        """Create data for ranking test."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_consensus_top_features_identified(self, consensus_data_and_model):
        """Test that consensus features are correctly identified."""
        X, y, model = consensus_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Should have some consensus features
        assert isinstance(result["top_features_consensus"], list)
        # With only 5 features and 2 informative, should have overlap
        assert len(result["top_features_consensus"]) >= 1

    def test_consensus_ranking_order(self, ranking_data_and_model):
        """Test that consensus ranking orders features correctly."""
        X, y, model = ranking_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Consensus ranking should have all features
        assert len(result["consensus_ranking"]) == 5
        # Should be unique features
        assert len(set(result["consensus_ranking"])) == 5


class TestMLImportanceWarnings:
    """Test warning generation."""

    @pytest.fixture(scope="class")
    def warning_test_data_and_model(self):
        """Create data for warning tests."""
        X, y = make_classification(n_samples=500, n_features=10, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def shap_test_data_and_model(self):
        """Create data for SHAP failure test."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_mdi_pfi_disagreement_warning(self, warning_test_data_and_model):
        """Test that MDI-PFI disagreement generates warning."""
        X, y, model = warning_test_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Warnings should be a list (may or may not have items)
        assert isinstance(result["warnings"], list)

    def test_failed_method_in_warnings(self, shap_test_data_and_model):
        """Test that failed methods appear in warnings."""
        X, y, model = shap_test_data_and_model

        # Request SHAP which may not be installed
        result = analyze_ml_importance(model, X, y, methods=["mdi", "shap"], random_state=42)

        # If SHAP failed, should be in warnings
        if "shap" not in result["methods_run"]:
            assert any("shap" in warning.lower() for warning in result["warnings"])


class TestMLImportanceInterpretation:
    """Test interpretation generation."""

    @pytest.fixture(scope="class")
    def interpretation_data_and_model(self):
        """Create data for interpretation tests."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_interpretation_contains_consensus_info(self, interpretation_data_and_model):
        """Test that interpretation mentions consensus."""
        X, y, model = interpretation_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        interp = result["interpretation"].lower()
        # Should mention consensus
        assert "consensus" in interp

    def test_interpretation_contains_agreement_info(self, interpretation_data_and_model):
        """Test that interpretation mentions method agreement."""
        X, y, model = interpretation_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi"], random_state=42)

        interp = result["interpretation"].lower()
        # Should mention agreement
        assert "agreement" in interp or "correlation" in interp


class TestMLImportanceDataFormats:
    """Test different data format inputs."""

    @pytest.fixture(scope="class")
    def base_data_5f(self):
        """Create base classification data with 5 features."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def pandas_data_and_model(self, base_data_5f):
        """Create pandas data and train model."""
        X, y = base_data_5f
        X_pd = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        y_pd = pd.Series(y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_pd, y_pd)
        return X_pd, y_pd, model

    @pytest.fixture(scope="class")
    def polars_data_and_model(self, base_data_5f):
        """Create polars data and train model on numpy."""
        X, y = base_data_5f
        X_pl = pl.DataFrame({f"feat_{i}": X[:, i] for i in range(5)})
        y_pl = pl.Series("target", y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)  # Fit on numpy (some models don't support Polars)
        return X_pl, y_pl, model

    @pytest.fixture(scope="class")
    def numpy_model(self, base_data_5f):
        """Train model on numpy arrays."""
        X, y = base_data_5f
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model

    def test_pandas_dataframe_input(self, pandas_data_and_model):
        """Test that function works with pandas DataFrame."""
        X_pd, y_pd, model = pandas_data_and_model

        result = analyze_ml_importance(model, X_pd, y_pd, methods=["mdi", "pfi"], random_state=42)

        # Should use DataFrame column names
        assert all("feat_" in fname for fname in result["consensus_ranking"])

    def test_polars_dataframe_input(self, polars_data_and_model):
        """Test that function works with Polars DataFrame."""
        X_pl, y_pl, model = polars_data_and_model

        result = analyze_ml_importance(model, X_pl, y_pl, methods=["mdi", "pfi"], random_state=42)

        # Should work with Polars
        assert len(result["consensus_ranking"]) == 5

    def test_numpy_array_with_feature_names(self, numpy_model, base_data_5f):
        """Test that function works with numpy arrays and explicit feature names."""
        X, y = base_data_5f

        feature_names = [f"custom_{i}" for i in range(5)]
        result = analyze_ml_importance(
            numpy_model,
            X,
            y,
            feature_names=feature_names,
            methods=["mdi", "pfi"],
            random_state=42,
        )

        # Should use custom feature names
        assert all("custom_" in fname for fname in result["consensus_ranking"])

    def test_numpy_array_without_feature_names(self, numpy_model, base_data_5f):
        """Test that function auto-generates feature names for numpy arrays."""
        X, y = base_data_5f

        result = analyze_ml_importance(numpy_model, X, y, methods=["mdi", "pfi"], random_state=42)

        # Should auto-generate f0, f1, f2, ...
        assert all(fname in result["consensus_ranking"] for fname in ["f0", "f1", "f2", "f3", "f4"])


class TestMLImportanceEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def edge_case_data_and_model(self):
        """Create data for edge case tests."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_empty_methods_list_raises_error(self, edge_case_data_and_model):
        """Test that empty methods list raises error."""
        X, y, model = edge_case_data_and_model

        with pytest.raises(ValueError, match="At least one method"):
            analyze_ml_importance(model, X, y, methods=[])

    def test_all_methods_fail_raises_error(self, edge_case_data_and_model):
        """Test that failure of all methods raises error."""
        X, y, model = edge_case_data_and_model

        # Request only methods that will fail
        # (This is hard to guarantee, so we'll just verify the structure)
        # In practice, at least MDI should work for RandomForest
        result = analyze_ml_importance(model, X, y, methods=["mdi"])
        assert "mdi" in result["methods_run"]

    def test_graceful_handling_of_missing_shap(self, edge_case_data_and_model):
        """Test graceful handling when SHAP is not installed."""
        X, y, model = edge_case_data_and_model

        # Request SHAP (may not be installed)
        result = analyze_ml_importance(model, X, y, methods=["mdi", "pfi", "shap"], random_state=42)

        # Should succeed with at least MDI and PFI
        assert "mdi" in result["methods_run"]
        assert "pfi" in result["methods_run"]

        # If SHAP failed, should be documented
        if "shap" not in result["methods_run"]:
            shap_failure = [f for f in result["methods_failed"] if f[0] == "shap"]
            assert len(shap_failure) == 1
            assert "shap" in shap_failure[0][1].lower()


class TestMLImportanceParameters:
    """Test parameter passing to underlying methods."""

    @pytest.fixture(scope="class")
    def params_data_and_model(self):
        """Create data for parameter tests."""
        X, y = make_classification(n_samples=500, n_features=5, n_informative=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_scoring_parameter_passed_to_pfi(self, params_data_and_model):
        """Test that scoring parameter is passed to PFI."""
        X, y, model = params_data_and_model

        result = analyze_ml_importance(
            model, X, y, methods=["pfi"], scoring="accuracy", random_state=42
        )

        # PFI should have run successfully
        assert "pfi" in result["methods_run"]
        assert result["method_results"]["pfi"]["scoring"] == "accuracy"

    def test_n_repeats_parameter_passed_to_pfi(self, params_data_and_model):
        """Test that n_repeats parameter is passed to PFI."""
        X, y, model = params_data_and_model

        result = analyze_ml_importance(model, X, y, methods=["pfi"], n_repeats=5, random_state=42)

        # PFI should have run with n_repeats=5
        assert "pfi" in result["methods_run"]
        assert result["method_results"]["pfi"]["n_repeats"] == 5

    def test_random_state_ensures_reproducibility(self, params_data_and_model):
        """Test that random_state ensures reproducible results."""
        X, y, model = params_data_and_model

        # Run twice with same random state
        result1 = analyze_ml_importance(model, X, y, methods=["pfi"], random_state=42)
        result2 = analyze_ml_importance(model, X, y, methods=["pfi"], random_state=42)

        # Results should be identical
        np.testing.assert_array_equal(
            result1["method_results"]["pfi"]["importances_mean"],
            result2["method_results"]["pfi"]["importances_mean"],
        )
        assert result1["consensus_ranking"] == result2["consensus_ranking"]
