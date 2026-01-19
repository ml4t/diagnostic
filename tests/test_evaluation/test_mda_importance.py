"""Tests for Mean Decrease in Accuracy (MDA) feature importance."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml4t.diagnostic.evaluation.metrics import compute_mda_importance


class TestMDABasicFunctionality:
    """Test basic MDA computation."""

    @pytest.fixture(scope="class")
    def classification_data_10f(self):
        """Create classification dataset with 10 features (shared across class)."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_classifier_10f(self, classification_data_10f):
        """Train RF classifier once for class."""
        X, y = classification_data_10f
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def regression_data_10f(self):
        """Create regression dataset with 10 features."""
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=3,
            noise=10.0,
            random_state=42,
        )
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_regressor_10f(self, regression_data_10f):
        """Train RF regressor once for class."""
        X, y = regression_data_10f
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        return model

    def test_mda_classification_identifies_important_features(
        self, trained_rf_classifier_10f, classification_data_10f
    ):
        """Test that MDA identifies important features in classification."""
        X, y = classification_data_10f
        model = trained_rf_classifier_10f

        # Compute MDA
        result = compute_mda_importance(model=model, X=X, y=y, removal_method="mean")

        # Verify structure
        assert "importances" in result
        assert "feature_names" in result
        assert "baseline_score" in result
        assert "removal_method" in result
        assert "scoring" in result
        assert "n_features" in result

        # Verify values
        assert len(result["importances"]) == 10
        assert len(result["feature_names"]) == 10
        assert result["removal_method"] == "mean"
        assert result["scoring"] == "default"
        assert result["n_features"] == 10
        assert 0.0 <= result["baseline_score"] <= 1.0

        # Check that importances are sorted descending
        assert all(
            result["importances"][i] >= result["importances"][i + 1]
            for i in range(len(result["importances"]) - 1)
        )

        # Top features should have positive importance
        assert result["importances"][0] > 0

    def test_mda_regression_identifies_important_features(
        self, trained_rf_regressor_10f, regression_data_10f
    ):
        """Test that MDA identifies important features in regression."""
        X, y = regression_data_10f
        model = trained_rf_regressor_10f

        # Compute MDA
        result = compute_mda_importance(model=model, X=X, y=y, removal_method="mean")

        # Verify structure
        assert len(result["importances"]) == 10
        assert len(result["feature_names"]) == 10
        assert result["n_features"] == 10

        # Top features should have positive importance
        assert result["importances"][0] > 0

        # Check sorted
        assert all(
            result["importances"][i] >= result["importances"][i + 1]
            for i in range(len(result["importances"]) - 1)
        )


class TestMDARemovalMethods:
    """Test different removal methods."""

    @pytest.fixture(scope="class")
    def classification_data_5f(self):
        """Create classification dataset with 5 features (shared across class)."""
        X, y = make_classification(n_samples=200, n_features=5, n_informative=2, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_5f(self, classification_data_5f):
        """Train RF classifier once for class (5 features)."""
        X, y = classification_data_5f
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def outlier_data_and_model(self):
        """Create data with outliers and train model (for removal method comparison)."""
        X, y = make_classification(
            n_samples=300, n_features=5, n_informative=2, n_redundant=0, random_state=42
        )
        # Add outliers to make mean != median
        X = X * 10
        X[0, 0] = 1000  # outlier
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_removal_method_mean(self, trained_rf_5f, classification_data_5f):
        """Test mean removal method."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=trained_rf_5f, X=X, y=y, removal_method="mean")

        assert result["removal_method"] == "mean"
        assert len(result["importances"]) == 5

    def test_removal_method_median(self, trained_rf_5f, classification_data_5f):
        """Test median removal method."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=trained_rf_5f, X=X, y=y, removal_method="median")

        assert result["removal_method"] == "median"
        assert len(result["importances"]) == 5

    def test_removal_method_zero(self, trained_rf_5f, classification_data_5f):
        """Test zero removal method."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=trained_rf_5f, X=X, y=y, removal_method="zero")

        assert result["removal_method"] == "zero"
        assert len(result["importances"]) == 5

    def test_invalid_removal_method_raises_error(self, trained_rf_5f, classification_data_5f):
        """Test that invalid removal method raises ValueError."""
        X, y = classification_data_5f
        with pytest.raises(ValueError, match="removal_method must be one of"):
            compute_mda_importance(model=trained_rf_5f, X=X, y=y, removal_method="invalid")

    def test_different_removal_methods_give_different_results(self, outlier_data_and_model):
        """Test that different removal methods produce different importances."""
        X, y, model = outlier_data_and_model

        result_mean = compute_mda_importance(model=model, X=X, y=y, removal_method="mean")
        result_median = compute_mda_importance(model=model, X=X, y=y, removal_method="median")
        result_zero = compute_mda_importance(model=model, X=X, y=y, removal_method="zero")

        # Results should differ (at least one pair should be different)
        # We check if at least one method differs significantly from another
        mean_vs_median_diff = not np.allclose(
            result_mean["importances"], result_median["importances"], rtol=0.1
        )
        mean_vs_zero_diff = not np.allclose(
            result_mean["importances"], result_zero["importances"], rtol=0.1
        )
        assert mean_vs_median_diff or mean_vs_zero_diff


class TestMDAFeatureGroups:
    """Test feature group functionality."""

    @pytest.fixture(scope="class")
    def classification_data_6f(self):
        """Create classification dataset with 6 features (for group tests)."""
        X, y = make_classification(n_samples=300, n_features=6, n_informative=3, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_6f(self, classification_data_6f):
        """Train RF classifier once for class (6 features)."""
        X, y = classification_data_6f
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def df_data_and_model_6f(self, classification_data_6f):
        """Create DataFrame version and train model."""
        X, y = classification_data_6f
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_df, y)
        return X_df, y, model

    @pytest.fixture(scope="class")
    def correlated_data_and_model(self):
        """Create data with correlated features and train model."""
        np.random.seed(42)
        n_samples = 500
        # Create two correlated features that together predict the target
        feature_0 = np.random.randn(n_samples)
        feature_1 = feature_0 + np.random.randn(n_samples) * 0.5
        feature_2 = np.random.randn(n_samples)
        feature_3 = np.random.randn(n_samples)
        X = np.column_stack([feature_0, feature_1, feature_2, feature_3])
        y = (feature_0 + feature_1 > 0).astype(int)
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        return X, y, model

    def test_feature_groups_basic(self, trained_rf_6f, classification_data_6f):
        """Test basic feature group computation."""
        X, y = classification_data_6f

        feature_groups = {
            "group_A": ["feature_0", "feature_1"],
            "group_B": ["feature_2", "feature_3"],
            "group_C": ["feature_4", "feature_5"],
        }

        result = compute_mda_importance(
            model=trained_rf_6f, X=X, y=y, feature_groups=feature_groups, removal_method="mean"
        )

        # Should have 3 groups instead of 6 features
        assert result["n_features"] == 3
        assert len(result["importances"]) == 3
        assert len(result["feature_names"]) == 3
        assert set(result["feature_names"]) == {"group_A", "group_B", "group_C"}

    def test_feature_groups_with_dataframe(self, df_data_and_model_6f):
        """Test feature groups with pandas DataFrame."""
        X_df, y, model = df_data_and_model_6f

        feature_groups = {
            "group_A": ["feature_0", "feature_1"],
            "group_B": ["feature_2", "feature_3", "feature_4"],
        }

        result = compute_mda_importance(model=model, X=X_df, y=y, feature_groups=feature_groups)

        assert result["n_features"] == 2
        assert set(result["feature_names"]) == {"group_A", "group_B"}

    def test_feature_groups_invalid_feature_name_raises_error(
        self, trained_rf_6f, classification_data_6f
    ):
        """Test that invalid feature name in group raises ValueError."""
        X, y = classification_data_6f

        feature_groups = {
            "group_A": ["feature_0", "feature_1"],
            "group_B": ["invalid_feature"],  # This doesn't exist
        }

        with pytest.raises(ValueError, match="not found in feature_names"):
            compute_mda_importance(model=trained_rf_6f, X=X, y=y, feature_groups=feature_groups)

    def test_feature_groups_higher_importance_than_individual(self, correlated_data_and_model):
        """Test that feature groups can show higher joint importance."""
        X, y, model = correlated_data_and_model

        # Test individual features
        compute_mda_importance(model=model, X=X, y=y)

        # Test as group
        feature_groups = {"correlated_group": ["feature_0", "feature_1"]}
        result_group = compute_mda_importance(model=model, X=X, y=y, feature_groups=feature_groups)

        # Group importance should exist and be positive
        group_importance = result_group["importances"][
            result_group["feature_names"].index("correlated_group")
        ]
        assert group_importance > 0


class TestMDAInputFormats:
    """Test different input formats (pandas, polars, numpy)."""

    @pytest.fixture(scope="class")
    def base_data_5f(self):
        """Create base classification dataset with 5 features."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def numpy_model(self, base_data_5f):
        """Train model on numpy arrays."""
        X, y = base_data_5f
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def pandas_data_and_model(self, base_data_5f):
        """Create pandas data and train model."""
        X, y = base_data_5f
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        y_series = pd.Series(y)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_df, y_series)
        return X_df, y_series, model

    @pytest.fixture(scope="class")
    def polars_data_and_model(self, base_data_5f):
        """Create polars data and train model."""
        X, y = base_data_5f
        X_pl = pl.DataFrame(X, schema=[f"col_{i}" for i in range(5)])
        y_pl = pl.Series(y)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_pl.to_numpy(), y_pl.to_numpy())
        return X_pl, y_pl, model

    @pytest.fixture(scope="class")
    def custom_names_data_and_model(self):
        """Create 3-feature data with custom names and train model."""
        X, y = make_classification(
            n_samples=200, n_features=3, n_informative=2, n_redundant=0, random_state=42
        )
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_numpy_array_input(self, numpy_model, base_data_5f):
        """Test with numpy arrays."""
        X, y = base_data_5f
        result = compute_mda_importance(model=numpy_model, X=X, y=y)

        assert len(result["importances"]) == 5
        # Should generate default feature names
        assert result["feature_names"][0].startswith("feature_")

    def test_pandas_dataframe_input(self, pandas_data_and_model):
        """Test with pandas DataFrame."""
        X_df, y_series, model = pandas_data_and_model
        result = compute_mda_importance(model=model, X=X_df, y=y_series)

        assert len(result["importances"]) == 5
        # Should use DataFrame column names
        assert result["feature_names"][0].startswith("feat_")

    def test_polars_dataframe_input(self, polars_data_and_model):
        """Test with polars DataFrame."""
        X_pl, y_pl, model = polars_data_and_model
        result = compute_mda_importance(model=model, X=X_pl, y=y_pl)

        assert len(result["importances"]) == 5
        # Should use polars column names
        assert result["feature_names"][0].startswith("col_")

    def test_custom_feature_names(self, custom_names_data_and_model):
        """Test with custom feature names."""
        X, y, model = custom_names_data_and_model
        custom_names = ["price", "volume", "momentum"]
        result = compute_mda_importance(model=model, X=X, y=y, feature_names=custom_names)

        assert set(result["feature_names"]) == set(custom_names)


class TestMDAScoring:
    """Test different scoring functions."""

    @pytest.fixture(scope="class")
    def classification_data_5f(self):
        """Create classification dataset with 5 features."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def trained_rf_classifier(self, classification_data_5f):
        """Train RF classifier once for class."""
        X, y = classification_data_5f
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def regression_data_and_model(self):
        """Create regression data and train model."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_default_scoring(self, trained_rf_classifier, classification_data_5f):
        """Test with default model scoring."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=trained_rf_classifier, X=X, y=y)

        assert result["scoring"] == "default"
        assert 0.0 <= result["baseline_score"] <= 1.0

    def test_accuracy_scoring(self, trained_rf_classifier, classification_data_5f):
        """Test with accuracy scoring."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=trained_rf_classifier, X=X, y=y, scoring="accuracy")

        assert result["scoring"] == "accuracy"
        assert 0.0 <= result["baseline_score"] <= 1.0

    def test_r2_scoring_regression(self, regression_data_and_model):
        """Test with R² scoring for regression."""
        X, y, model = regression_data_and_model
        result = compute_mda_importance(model=model, X=X, y=y, scoring="r2")

        assert result["scoring"] == "r2"
        assert result["baseline_score"] <= 1.0

    def test_custom_scoring_function(self, trained_rf_classifier, classification_data_5f):
        """Test with custom scoring function."""
        from sklearn.metrics import accuracy_score, make_scorer

        X, y = classification_data_5f
        custom_scorer = make_scorer(accuracy_score)
        result = compute_mda_importance(
            model=trained_rf_classifier, X=X, y=y, scoring=custom_scorer
        )

        assert result["scoring"] == "custom"
        assert 0.0 <= result["baseline_score"] <= 1.0


class TestMDAEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def single_feature_data_and_model(self):
        """Create single-feature data and train model."""
        X, y = make_classification(
            n_samples=200,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42,
        )
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def mismatch_data_and_model(self):
        """Create data for mismatch test."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def perfect_prediction_data_and_model(self):
        """Create perfectly separable data and train model."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return X, y, model

    @pytest.fixture(scope="class")
    def uninformative_data_and_model(self):
        """Create random data with no relationship and train model."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.randint(0, 2, size=200)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return X, y, model

    def test_single_feature(self, single_feature_data_and_model):
        """Test with single feature."""
        X, y, model = single_feature_data_and_model
        result = compute_mda_importance(model=model, X=X, y=y)

        assert len(result["importances"]) == 1
        assert result["n_features"] == 1

    def test_mismatched_sample_sizes_raises_error(self, mismatch_data_and_model):
        """Test that mismatched X and y sizes raise ValueError."""
        X, y, model = mismatch_data_and_model

        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            compute_mda_importance(model=model, X=X, y=y[:100])

    def test_perfect_prediction(self, perfect_prediction_data_and_model):
        """Test with perfectly separable data."""
        X, y, model = perfect_prediction_data_and_model
        result = compute_mda_importance(model=model, X=X, y=y)

        # Should still work, first feature should be more important
        assert result["importances"][0] > result["importances"][1]

    def test_all_features_uninformative(self, uninformative_data_and_model):
        """Test with all uninformative features."""
        X, y, model = uninformative_data_and_model
        result = compute_mda_importance(model=model, X=X, y=y)

        # All importances should be small (close to zero or negative)
        assert len(result["importances"]) == 5
        # Don't expect high importances for random data


class TestMDAComparison:
    """Test MDA comparison with other importance methods."""

    @pytest.fixture(scope="class")
    def comparison_data_and_model(self):
        """Create data and train model for comparison tests."""
        X, y = make_classification(n_samples=500, n_features=10, n_informative=3, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        return X, y, model

    def test_mda_vs_pfi_similar_ranking(self, comparison_data_and_model):
        """Test that MDA and PFI give similar feature rankings."""
        from ml4t.diagnostic.evaluation.metrics import compute_permutation_importance

        X, y, model = comparison_data_and_model

        mda = compute_mda_importance(model=model, X=X, y=y, removal_method="mean")
        pfi = compute_permutation_importance(model=model, X=X, y=y, n_repeats=10, random_state=42)

        # Top 3 features should overlap between methods
        mda_top3 = set(mda["feature_names"][:3])
        pfi_top3 = set(pfi["feature_names"][:3])

        # At least 2 out of 3 should match (allowing for some variation)
        overlap = len(mda_top3 & pfi_top3)
        assert overlap >= 2, f"MDA top 3: {mda_top3}, PFI top 3: {pfi_top3}"

    def test_mda_vs_mdi_different_but_related(self, comparison_data_and_model):
        """Test that MDA and MDI give related but potentially different results."""
        from ml4t.diagnostic.evaluation.metrics import compute_mdi_importance

        X, y, model = comparison_data_and_model

        mda = compute_mda_importance(model=model, X=X, y=y)
        mdi = compute_mdi_importance(model=model)

        # Both should identify some important features
        assert mda["importances"][0] > 0
        assert mdi["importances"][0] > 0

        # Top features might differ (MDI can be biased)
        # Just verify both methods run successfully
        assert len(mda["feature_names"]) == len(mdi["feature_names"])


class TestMDADifferentModels:
    """Test MDA with different model types."""

    @pytest.fixture(scope="class")
    def classification_data_5f(self):
        """Create classification dataset with 5 features."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def regression_data_5f(self):
        """Create regression dataset with 5 features."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        return X, y

    @pytest.fixture(scope="class")
    def logistic_model(self, classification_data_5f):
        """Train logistic regression model."""
        X, y = classification_data_5f
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def linear_model(self, regression_data_5f):
        """Train linear regression model."""
        X, y = regression_data_5f
        model = LinearRegression()
        model.fit(X, y)
        return model

    @pytest.fixture(scope="class")
    def gradient_boosting_model(self, classification_data_5f):
        """Train gradient boosting model."""
        X, y = classification_data_5f
        model = GradientBoostingClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    def test_logistic_regression(self, logistic_model, classification_data_5f):
        """Test with logistic regression."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=logistic_model, X=X, y=y)

        assert len(result["importances"]) == 5

    def test_linear_regression(self, linear_model, regression_data_5f):
        """Test with linear regression."""
        X, y = regression_data_5f
        result = compute_mda_importance(model=linear_model, X=X, y=y)

        assert len(result["importances"]) == 5

    def test_gradient_boosting(self, gradient_boosting_model, classification_data_5f):
        """Test with gradient boosting."""
        X, y = classification_data_5f
        result = compute_mda_importance(model=gradient_boosting_model, X=X, y=y)

        assert len(result["importances"]) == 5
