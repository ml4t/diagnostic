"""Tests for feature-outcome analysis (Module C orchestration)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.config.feature_config import (
    DiagnosticConfig,
    ICSettings,
    MLDiagnosticsSettings,
)
from ml4t.diagnostic.evaluation.feature_outcome import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcome,
    FeatureOutcomeResult,
)


class TestFeatureICResults:
    """Test FeatureICResults dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureICResults(
            feature="test_feature",
            ic_mean=0.15,
            ic_std=0.05,
            ic_ir=3.0,
            p_value=0.001,
            n_observations=1000,
        )

        assert result.feature == "test_feature"
        assert result.ic_mean == 0.15
        assert result.ic_ir == 3.0
        assert result.p_value == 0.001

    def test_defaults(self):
        """Test default values."""
        result = FeatureICResults(feature="test")

        assert result.ic_mean == 0.0
        assert result.ic_std == 0.0
        assert result.ic_ir == 0.0
        assert result.p_value == 1.0


class TestFeatureImportanceResults:
    """Test FeatureImportanceResults dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureImportanceResults(
            feature="test_feature",
            mdi_importance=0.35,
            permutation_importance=0.28,
            rank_mdi=1,
            rank_permutation=2,
        )

        assert result.feature == "test_feature"
        assert result.mdi_importance == 0.35
        assert result.rank_mdi == 1


class TestFeatureOutcomeResult:
    """Test FeatureOutcomeResult aggregation."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        assert len(result.features) == 3
        assert result.ic_results == {}
        assert result.importance_results == {}
        assert result.drift_results is None

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        result = FeatureOutcomeResult(features=["f1", "f2"])

        # Add some IC results
        result.ic_results["f1"] = FeatureICResults(
            feature="f1", ic_mean=0.15, ic_std=0.05, ic_ir=3.0, p_value=0.001
        )
        result.ic_results["f2"] = FeatureICResults(
            feature="f2", ic_mean=-0.08, ic_std=0.04, ic_ir=-2.0, p_value=0.05
        )

        # Add importance results
        result.importance_results["f1"] = FeatureImportanceResults(
            feature="f1", mdi_importance=0.35, rank_mdi=1
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "feature" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "mdi_importance" in df.columns

        # Check values
        f1_row = df[df["feature"] == "f1"].iloc[0]
        assert f1_row["ic_mean"] == 0.15
        assert f1_row["ic_ir"] == 3.0
        assert f1_row["mdi_importance"] == 0.35

    def test_get_top_features(self):
        """Test top feature selection."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        # Add IC results with different values
        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=3.0)
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=1.5)
        result.ic_results["f3"] = FeatureICResults(feature="f3", ic_ir=2.0)

        top_features = result.get_top_features(n=2, by="ic_ir")

        assert len(top_features) == 2
        assert top_features[0] == "f1"  # Highest IC IR
        assert top_features[1] == "f3"  # Second highest

    def test_get_top_features_with_errors(self):
        """Test that features with errors are excluded."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=3.0)
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=2.0)
        # f3 has error
        result.errors["f3"] = "Test error"

        top_features = result.get_top_features(n=3, by="ic_ir")

        # Should only return 2 features (f3 excluded due to error)
        assert len(top_features) == 2
        assert "f3" not in top_features

    def test_get_recommendations(self):
        """Test recommendation generation."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        # f1: Strong signal
        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=2.5, p_value=0.001)

        # f2: Weak signal
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=0.3, p_value=0.2)

        # f3: Error
        result.errors["f3"] = "Insufficient data"

        recommendations = result.get_recommendations()

        assert len(recommendations) > 0
        # Should mention strong signal
        assert any("f1" in rec and "Strong" in rec for rec in recommendations)
        # Should mention weak signals
        assert any("weak" in rec.lower() for rec in recommendations)
        # Should mention errors
        assert any("failed" in rec.lower() or "error" in rec.lower() for rec in recommendations)


class TestFeatureOutcome:
    """Test FeatureOutcome orchestration class."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        analyzer = FeatureOutcome()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, DiagnosticConfig)
        assert analyzer.config.ic.enabled is True

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = DiagnosticConfig(
            ic=ICSettings(enabled=True, hac_adjustment=True),
            ml_diagnostics=MLDiagnosticsSettings(drift_detection=True),
        )
        analyzer = FeatureOutcome(config=config)

        assert analyzer.config.ic.hac_adjustment is True
        assert analyzer.config.ml_diagnostics.drift_detection is True

    def test_run_analysis_basic(self):
        """Test basic analysis with synthetic data."""
        np.random.seed(42)

        # Create synthetic data
        n = 500
        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(0, 1, n),
                "f3": np.random.normal(0, 1, n),
            },
            index=pd.date_range("2020-01-01", periods=n),
        )

        # Create outcomes correlated with f1
        outcomes = pd.Series(
            features["f1"] * 0.5 + np.random.normal(0, 0.5, n), index=features.index
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert isinstance(result, FeatureOutcomeResult)
        assert len(result.features) == 3
        assert "f1" in result.ic_results
        assert "f2" in result.ic_results
        assert "f3" in result.ic_results

        # f1 should have higher IC than f2/f3 (correlated with outcome)
        ic_f1 = result.ic_results["f1"].ic_mean
        ic_f2 = result.ic_results["f2"].ic_mean
        assert abs(ic_f1) > abs(ic_f2)

    def test_run_analysis_with_numpy_outcomes(self):
        """Test with numpy array outcomes."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
            }
        )

        outcomes = np.random.normal(0, 1, 100)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert isinstance(result, FeatureOutcomeResult)
        assert len(result.features) == 2

    def test_run_analysis_specific_features(self):
        """Test analyzing only specific features."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
                "f3": np.random.normal(0, 1, 100),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes, feature_names=["f1", "f2"])

        assert len(result.features) == 2
        assert "f1" in result.features
        assert "f2" in result.features
        assert "f3" not in result.features

    def test_run_analysis_with_missing_features(self):
        """Test error handling for missing features."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        outcomes = pd.Series([1, 2, 3])

        analyzer = FeatureOutcome()

        with pytest.raises(ValueError, match="Features not found"):
            analyzer.run_analysis(features, outcomes, feature_names=["f1", "missing"])

    def test_run_analysis_with_misaligned_data(self):
        """Test error handling for misaligned data."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        outcomes = pd.Series([1, 2])  # Different length!

        analyzer = FeatureOutcome()

        with pytest.raises(ValueError, match="must have same length"):
            analyzer.run_analysis(features, outcomes)

    def test_run_analysis_with_nans(self):
        """Test handling of NaN values."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": [1, 2, np.nan, 4, 5],
                "f2": [np.nan, 2, 3, 4, 5],
            }
        )

        outcomes = pd.Series([1, 2, 3, np.nan, 5])

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        # Should handle NaNs gracefully - insufficient data leads to errors
        assert isinstance(result, FeatureOutcomeResult)
        # With only 5 samples and NaNs, should have errors for insufficient data
        assert "f1" in result.errors or "f1" in result.ic_results
        assert "f2" in result.errors or "f2" in result.ic_results

    def test_run_analysis_insufficient_data(self):
        """Test handling of insufficient data."""
        # Only 5 samples - too few for analysis
        features = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5],
                "f2": [np.nan, np.nan, np.nan, np.nan, np.nan],  # All NaN
            }
        )

        outcomes = pd.Series([1, 2, 3, 4, 5])

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        # f2 should have error due to all NaN
        assert "f2" in result.errors

    def test_run_analysis_with_importance(self):
        """Test ML importance analysis."""
        np.random.seed(42)

        # Need more samples for importance analysis
        n = 200
        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(0, 1, n),
                "f3": np.random.normal(0, 1, n),
            }
        )

        # Outcomes correlated with f1
        outcomes = features["f1"] * 2 + np.random.normal(0, 0.5, n)

        config = DiagnosticConfig(
            ml_diagnostics=MLDiagnosticsSettings(feature_importance=True, drift_detection=False)
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcomes)

        # Check if importance was computed (depends on LightGBM availability)
        if len(result.importance_results) > 0:
            assert "f1" in result.importance_results
            # f1 should have higher importance (correlated with outcome)
            imp_f1 = result.importance_results["f1"].mdi_importance
            assert imp_f1 > 0

    def test_run_analysis_with_drift_detection(self):
        """Test drift detection integration."""
        np.random.seed(42)

        n = 500
        # Create data with drift in second half
        f1_first = np.random.normal(0, 1, n // 2)
        f1_second = np.random.normal(0.5, 1, n // 2)  # Mean shift
        f1 = np.concatenate([f1_first, f1_second])

        features = pd.DataFrame(
            {
                "f1": f1,
                "f2": np.random.normal(0, 1, n),  # No drift
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, n))

        config = DiagnosticConfig(
            ic=ICSettings(enabled=False),  # Disable IC for speed
            ml_diagnostics=MLDiagnosticsSettings(feature_importance=False, drift_detection=True),
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcomes)

        assert result.drift_results is not None
        # f1 should show drift
        drift_df = result.drift_results.to_dataframe()
        # Convert to pandas if polars
        if isinstance(drift_df, pl.DataFrame):
            drift_df = drift_df.to_pandas()
        f1_drift = drift_df[drift_df["feature"] == "f1"]
        # Depending on threshold, may or may not flag (just check it ran)
        assert len(f1_drift) > 0

    def test_summary_dataframe_completeness(self):
        """Test that summary DataFrame contains all expected columns."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 200),
                "f2": np.random.normal(0, 1, 200),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 200))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        df = result.summary

        assert df is not None
        assert "feature" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "error" in df.columns

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        np.random.seed(42)

        features = pd.DataFrame({"f1": np.random.normal(0, 1, 100)})

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert "n_features" in result.metadata
        assert result.metadata["n_features"] == 1
        assert "n_observations" in result.metadata
        assert result.metadata["n_observations"] == 100
        assert "computation_time" in result.metadata
        assert result.metadata["computation_time"] > 0

    def test_verbose_mode(self, capsys):
        """Test verbose output."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        analyzer.run_analysis(features, outcomes, verbose=True)

        captured = capsys.readouterr()
        assert "Analyzing" in captured.out
        assert "features" in captured.out
        assert "complete" in captured.out


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self):
        """Test complete workflow from data to recommendations."""
        np.random.seed(42)

        # Create realistic synthetic data
        n = 500
        features = pd.DataFrame(
            {
                "momentum": np.random.normal(0, 1, n),
                "value": np.random.normal(0, 1, n),
                "quality": np.random.normal(0, 1, n),
                "sentiment": np.random.normal(0, 1, n),
            },
            index=pd.date_range("2020-01-01", periods=n),
        )

        # Create returns with some correlation to momentum
        returns = (
            features["momentum"] * 0.3 + features["quality"] * 0.15 + np.random.normal(0, 1, n)
        )

        # Run full analysis
        config = DiagnosticConfig(
            ic=ICSettings(enabled=True),
            ml_diagnostics=MLDiagnosticsSettings(feature_importance=True, drift_detection=True),
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, returns, verbose=False)

        # Validate results
        assert len(result.features) == 4
        assert len(result.ic_results) == 4

        # Get top features
        top_features = result.get_top_features(n=2, by="ic_ir")
        assert len(top_features) <= 2

        # Get recommendations
        recommendations = result.get_recommendations()
        assert isinstance(recommendations, list)

        # Export to DataFrame
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
