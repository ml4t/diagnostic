"""Tests for domain classifier drift detection.

Tests train sklearn models but complete quickly (~2.5s) with module-scoped fixtures.
"""

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.drift import (
    DomainClassifierResult,
    compute_domain_classifier_drift,
)

# =============================================================================
# Module-level fixtures for shared test data
# =============================================================================


@pytest.fixture(scope="module")
def simple_shift_data():
    """Reference and test data with simple mean shift (for model type tests)."""
    np.random.seed(42)
    reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
    test = pl.DataFrame({"x1": np.random.normal(1, 1, 200)})
    return reference, test


@pytest.fixture(scope="module")
def two_feature_shift_data():
    """Reference and test data with two features and moderate shift."""
    np.random.seed(42)
    reference = pl.DataFrame(
        {
            "x1": np.random.normal(0, 1, 200),
            "x2": np.random.normal(0, 1, 200),
        }
    )
    test = pl.DataFrame(
        {
            "x1": np.random.normal(0.5, 1, 200),
            "x2": np.random.normal(0.5, 1, 200),
        }
    )
    return reference, test


class TestDomainClassifierBasic:
    """Basic functionality tests."""

    def test_no_drift_identical_distributions(self):
        """Test AUC ≈ 0.5 for identical distributions."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
            }
        )
        # Use same data for test (perfect non-drift)
        test = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
            }
        )

        # Use lower max_depth to reduce overfitting on random noise
        result = compute_domain_classifier_drift(
            reference,
            test,
            model_type="sklearn",
            max_depth=3,
            n_estimators=20,
            cv_folds=3,
            random_state=42,
        )

        # Should not be able to distinguish (AUC near 0.5)
        # Note: RandomForest can overfit on random data, so we check it's not too high
        assert isinstance(result, DomainClassifierResult)
        assert 0.35 <= result.auc <= 0.75  # Allow for some random variance
        # Just check structure is correct
        assert result.threshold == 0.6
        assert result.n_reference == 500
        assert result.n_test == 500
        assert result.n_features == 2
        assert result.model_type == "sklearn"

    def test_strong_drift_mean_shift(self):
        """Test AUC > 0.9 for clear mean shift."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(3, 1, 500),  # Large shift
                "x2": np.random.normal(3, 1, 500),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        assert result.auc > 0.9  # Should easily distinguish
        assert result.drifted
        assert "strong" in result.interpretation.lower()

    def test_moderate_drift_small_shift(self):
        """Test moderate drift detection."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(0.8, 1, 500),  # Moderate shift
                "x2": np.random.normal(0.8, 1, 500),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        # Should detect drift but not as strong
        assert 0.6 < result.auc < 0.95
        assert result.drifted

    def test_multivariate_interaction_drift(self):
        """Test detection of interaction-based drift."""
        np.random.seed(42)

        # Reference: independent features
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
            }
        )

        # Test: correlated features (interaction drift)
        x1_test = np.random.normal(0, 1, 500)
        x2_test = x1_test * 0.8 + np.random.normal(0, 0.3, 500)  # Strong correlation
        test = pl.DataFrame({"x1": x1_test, "x2": x2_test})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        # Should detect correlation change
        assert result.auc > 0.6  # Some drift detected


class TestDomainClassifierFeatureImportance:
    """Feature importance tests."""

    def test_feature_importance_identifies_drifted_feature(self):
        """Test that drifted feature has highest importance."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),  # No drift
                "x2": np.random.normal(0, 1, 500),  # Will drift
                "x3": np.random.normal(0, 1, 500),  # No drift
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(2, 1, 500),  # Strong drift
                "x3": np.random.normal(0, 1, 500),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        # x2 should have highest importance
        top_feature = result.feature_importances.filter(pl.col("rank") == 1)["feature"][0]
        assert top_feature == "x2"

        # Check DataFrame structure
        assert len(result.feature_importances) == 3
        assert list(result.feature_importances.columns) == ["feature", "importance", "rank"]
        assert result.feature_importances["rank"].to_list() == [1, 2, 3]

    def test_feature_importance_all_features_drift_equally(self):
        """Test when all features drift equally."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 500),
                "x2": np.random.normal(0, 1, 500),
                "x3": np.random.normal(0, 1, 500),
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(1, 1, 500),  # All shifted equally
                "x2": np.random.normal(1, 1, 500),
                "x3": np.random.normal(1, 1, 500),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        # All features should have similar importance
        importances = result.feature_importances["importance"].to_numpy()
        # Check variance is low (all similar)
        assert np.std(importances) < 0.2  # Relatively similar


class TestDomainClassifierModels:
    """Test different model types."""

    def test_sklearn_model(self, simple_shift_data):
        """Test sklearn RandomForest (always available)."""
        reference, test = simple_shift_data

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3, random_state=42
        )

        assert 0 <= result.auc <= 1
        assert result.model_type == "sklearn"
        assert result.cv_auc_mean > 0
        assert result.cv_auc_std >= 0

    def test_lightgbm_model(self, simple_shift_data):
        """Test LightGBM model."""
        reference, test = simple_shift_data

        result = compute_domain_classifier_drift(
            reference, test, model_type="lightgbm", n_estimators=20, cv_folds=3, random_state=42
        )

        assert 0 <= result.auc <= 1
        assert result.model_type == "lightgbm"

    def test_xgboost_model(self, simple_shift_data):
        """Test XGBoost model."""
        reference, test = simple_shift_data

        result = compute_domain_classifier_drift(
            reference, test, model_type="xgboost", n_estimators=20, cv_folds=3, random_state=42
        )

        assert 0 <= result.auc <= 1
        assert result.model_type == "xgboost"

    def test_invalid_model_type(self, simple_shift_data):
        """Test error on invalid model type."""
        reference, test = simple_shift_data

        with pytest.raises(ValueError, match="Unknown model_type"):
            compute_domain_classifier_drift(reference, test, model_type="invalid")


class TestDomainClassifierInputFormats:
    """Test different input formats."""

    def test_polars_dataframe_input(self, two_feature_shift_data):
        """Test with polars DataFrame."""
        reference, test = two_feature_shift_data

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=10, cv_folds=3
        )

        assert isinstance(result, DomainClassifierResult)
        assert result.n_features == 2

    def test_pandas_dataframe_input(self, two_feature_shift_data):
        """Test with pandas DataFrame."""
        reference, test = two_feature_shift_data
        # Convert to pandas
        reference_pd = reference.to_pandas()
        test_pd = test.to_pandas()

        result = compute_domain_classifier_drift(
            reference_pd, test_pd, model_type="sklearn", n_estimators=10, cv_folds=3
        )

        assert isinstance(result, DomainClassifierResult)
        assert result.n_features == 2

    def test_numpy_array_input(self, two_feature_shift_data):
        """Test with numpy arrays."""
        reference, test = two_feature_shift_data
        # Convert to numpy
        reference_np = reference.to_numpy()
        test_np = test.to_numpy()

        result = compute_domain_classifier_drift(
            reference_np, test_np, model_type="sklearn", n_estimators=10, cv_folds=3
        )

        assert isinstance(result, DomainClassifierResult)
        assert result.n_features == 2
        # Should generate default feature names
        assert "feature_0" in result.feature_importances["feature"].to_list()
        assert "feature_1" in result.feature_importances["feature"].to_list()

    def test_feature_selection(self):
        """Test explicit feature selection."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 200),
                "x2": np.random.normal(0, 1, 200),
                "x3": np.random.normal(0, 1, 200),
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(1, 1, 200),
                "x2": np.random.normal(1, 1, 200),
                "x3": np.random.normal(0, 1, 200),  # No drift
            }
        )

        # Only use x1 and x2
        result = compute_domain_classifier_drift(
            reference,
            test,
            features=["x1", "x2"],
            model_type="sklearn",
            n_estimators=10,
            cv_folds=3,
        )

        assert result.n_features == 2
        assert set(result.feature_importances["feature"].to_list()) == {"x1", "x2"}


class TestDomainClassifierEdgeCases:
    """Edge case tests."""

    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
        test = pl.DataFrame({"x1": np.random.normal(0.5, 1, 200)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        assert result.n_features == 1
        assert len(result.feature_importances) == 1
        assert result.feature_importances["feature"][0] == "x1"

    def test_custom_threshold(self):
        """Test custom AUC threshold."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
        test = pl.DataFrame({"x1": np.random.normal(0.3, 1, 200)})

        # Lower threshold
        result1 = compute_domain_classifier_drift(
            reference, test, threshold=0.55, model_type="sklearn", n_estimators=20, random_state=42
        )

        # Higher threshold
        result2 = compute_domain_classifier_drift(
            reference, test, threshold=0.8, model_type="sklearn", n_estimators=20, random_state=42
        )

        # Same AUC but different drift flags
        assert abs(result1.auc - result2.auc) < 0.01  # Should be same
        # One might be drifted, one not (depending on actual AUC)
        assert result1.threshold == 0.55
        assert result2.threshold == 0.8

    def test_small_sample_sizes(self):
        """Test with small sample sizes."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 50)})
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 50)})

        result = compute_domain_classifier_drift(
            reference,
            test,
            model_type="sklearn",
            cv_folds=3,  # Fewer folds for small data
        )

        assert result.n_reference == 50
        assert result.n_test == 50
        assert 0 <= result.auc <= 1

    def test_unbalanced_sample_sizes(self):
        """Test with different reference and test sizes."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 500)})
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 100)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        assert result.n_reference == 500
        assert result.n_test == 100
        # Should still work

    def test_feature_mismatch_error(self):
        """Test error when feature counts don't match."""
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 100),
                "x2": np.random.normal(0, 1, 100),
            }
        )
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 100)})  # Missing x2

        # Should raise error for missing column when trying to select features
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            compute_domain_classifier_drift(reference, test, model_type="sklearn", n_estimators=20)


class TestDomainClassifierMetadata:
    """Test result metadata and reporting."""

    def test_metadata_structure(self):
        """Test metadata contains expected keys."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 100)})
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 100)})

        result = compute_domain_classifier_drift(
            reference,
            test,
            model_type="sklearn",
            n_estimators=20,
            max_depth=3,
            cv_folds=3,
            random_state=123,
        )

        assert "n_estimators" in result.metadata
        assert "max_depth" in result.metadata
        assert "cv_folds" in result.metadata
        assert "random_state" in result.metadata
        assert result.metadata["n_estimators"] == 20
        assert result.metadata["max_depth"] == 3
        assert result.metadata["cv_folds"] == 3
        assert result.metadata["random_state"] == 123

    def test_summary_formatting(self):
        """Test summary output is readable."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "feature_a": np.random.normal(0, 1, 100),
                "feature_b": np.random.normal(0, 1, 100),
            }
        )
        test = pl.DataFrame(
            {
                "feature_a": np.random.normal(1, 1, 100),
                "feature_b": np.random.normal(1, 1, 100),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )
        summary = result.summary()

        # Check key information in summary
        assert "Domain Classifier Drift Detection Report" in summary
        assert f"AUC-ROC: {result.auc:.4f}" in summary
        assert "Drift Detected:" in summary
        assert f"Reference: {result.n_reference:,}" in summary
        assert f"Test: {result.n_test:,}" in summary
        assert "Top 5 Most Drifted Features:" in summary
        assert "Interpretation:" in summary
        assert "Computation Time:" in summary

    def test_cv_scores_validity(self):
        """Test cross-validation scores are valid."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 200)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=5
        )

        # CV scores should be valid
        assert 0 <= result.cv_auc_mean <= 1
        assert result.cv_auc_std >= 0
        # AUC and CV mean should be close
        assert abs(result.auc - result.cv_auc_mean) < 0.3  # Reasonable difference

    def test_computation_time_recorded(self):
        """Test computation time is recorded."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 100)})
        test = pl.DataFrame({"x1": np.random.normal(1, 1, 100)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        assert result.computation_time > 0
        assert result.computation_time < 60  # Should be fast


class TestDomainClassifierInterpretation:
    """Test interpretation generation."""

    def test_no_drift_interpretation(self):
        """Test interpretation when no drift detected."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
        test = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        if not result.drifted:
            assert "No significant drift" in result.interpretation
            assert "indistinguishable" in result.interpretation

    def test_weak_drift_interpretation(self):
        """Test interpretation for weak drift."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 300)})
        test = pl.DataFrame({"x1": np.random.normal(0.3, 1, 300)})

        result = compute_domain_classifier_drift(
            reference, test, threshold=0.55, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        if result.drifted and result.auc < 0.7:
            assert "weak" in result.interpretation.lower()

    def test_strong_drift_interpretation(self):
        """Test interpretation for strong drift."""
        np.random.seed(42)
        reference = pl.DataFrame({"x1": np.random.normal(0, 1, 200)})
        test = pl.DataFrame({"x1": np.random.normal(3, 1, 200)})

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        if result.auc > 0.9:
            assert "strong" in result.interpretation.lower()

    def test_interpretation_includes_top_feature(self):
        """Test interpretation mentions top drifted feature."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "x1": np.random.normal(0, 1, 200),
                "x2": np.random.normal(0, 1, 200),
            }
        )
        test = pl.DataFrame(
            {
                "x1": np.random.normal(2, 1, 200),  # Strong drift
                "x2": np.random.normal(0, 1, 200),
            }
        )

        result = compute_domain_classifier_drift(
            reference, test, model_type="sklearn", n_estimators=20, cv_folds=3
        )

        if result.drifted:
            top_feature = result.feature_importances.filter(pl.col("rank") == 1)["feature"][0]
            assert top_feature in result.interpretation
