"""Tests for unified drift analysis (analyze_drift).

Tests complete in ~17s - acceptable for CI without slow marker.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.drift import (
    DriftSummaryResult,
    FeatureDriftResult,
    analyze_drift,
)


class TestAnalyzeDriftBasic:
    """Basic functionality tests for analyze_drift."""

    def test_no_drift_all_methods(self):
        """Test that no drift is detected when distributions are identical."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )

        # Use higher domain classifier threshold to avoid false positives
        result = analyze_drift(reference, test, domain_classifier_config={"threshold": 0.9})

        assert isinstance(result, DriftSummaryResult)
        assert result.n_features == 2
        # With same distribution, should detect minimal or no drift
        assert result.n_features_drifted <= 1  # Allow for sampling variation
        assert result.consensus_threshold == 0.5
        assert set(result.methods_used) == {"psi", "wasserstein", "domain_classifier"}

    def test_drift_detected_all_methods(self):
        """Test that drift is detected when distributions differ."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(1.0, 1, 1000),  # Mean shifted
                "feature2": np.random.normal(0, 2, 1000),  # Variance changed
            }
        )

        result = analyze_drift(reference, test)

        assert result.n_features == 2
        # Both features should drift (strong shifts)
        assert result.n_features_drifted >= 1
        assert result.overall_drifted is True
        assert len(result.drifted_features) >= 1

    def test_psi_only(self):
        """Test analyze_drift with only PSI method."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(reference, test, methods=["psi"])

        assert result.methods_used == ["psi"]
        assert result.univariate_methods == ["psi"]
        assert result.multivariate_methods == []
        assert result.domain_classifier_result is None

        # Check feature result has PSI but not Wasserstein
        feature_result = result.feature_results[0]
        assert feature_result.psi_result is not None
        assert feature_result.wasserstein_result is None

    def test_wasserstein_only(self):
        """Test analyze_drift with only Wasserstein method."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(reference, test, methods=["wasserstein"])

        assert result.methods_used == ["wasserstein"]
        assert result.univariate_methods == ["wasserstein"]
        assert result.multivariate_methods == []
        assert result.domain_classifier_result is None

        # Check feature result has Wasserstein but not PSI
        feature_result = result.feature_results[0]
        assert feature_result.psi_result is None
        assert feature_result.wasserstein_result is not None

    def test_domain_classifier_only(self):
        """Test analyze_drift with only domain classifier method."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(0, 1, 500),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 500),
                "feature2": np.random.normal(0, 1, 500),
            }
        )

        result = analyze_drift(reference, test, methods=["domain_classifier"])

        assert result.methods_used == ["domain_classifier"]
        assert result.univariate_methods == []
        assert result.multivariate_methods == ["domain_classifier"]
        assert result.domain_classifier_result is not None

        # Feature results exist but have no univariate method results
        assert len(result.feature_results) == 2
        for feature_result in result.feature_results:
            assert feature_result.psi_result is None
            assert feature_result.wasserstein_result is None
            assert feature_result.n_methods_run == 0

    def test_psi_and_wasserstein_no_domain_classifier(self):
        """Test analyze_drift with PSI and Wasserstein but no domain classifier."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )

        result = analyze_drift(reference, test, methods=["psi", "wasserstein"])

        assert set(result.methods_used) == {"psi", "wasserstein"}
        assert result.domain_classifier_result is None
        assert len(result.feature_results) == 2

        # Both methods should have results
        for feature_result in result.feature_results:
            assert feature_result.psi_result is not None
            assert feature_result.wasserstein_result is not None


class TestConsensusLogic:
    """Test consensus drift flagging logic."""

    def test_consensus_threshold_50_percent(self):
        """Test default consensus threshold of 50%."""
        np.random.seed(42)
        # Create data where PSI detects drift but Wasserstein may not
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.3, 1, 1000)})

        result = analyze_drift(
            reference,
            test,
            methods=["psi", "wasserstein"],
            consensus_threshold=0.5,
            wasserstein_config={"threshold_calibration": False},
        )

        # With threshold=0.5, need at least 1/2 methods to detect drift
        feature_result = result.feature_results[0]
        assert feature_result.n_methods_run == 2

        # If at least one method detected drift, consensus should be True
        if feature_result.n_methods_detected >= 1:
            assert feature_result.drift_probability >= 0.5

    def test_consensus_threshold_100_percent(self):
        """Test consensus threshold requiring all methods to agree."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(1.0, 1, 1000)})

        result = analyze_drift(
            reference,
            test,
            methods=["psi", "wasserstein"],
            consensus_threshold=1.0,
        )

        feature_result = result.feature_results[0]

        # With threshold=1.0, all methods must detect drift
        if feature_result.drifted:
            assert feature_result.n_methods_detected == feature_result.n_methods_run
            assert feature_result.drift_probability == 1.0

    def test_consensus_threshold_33_percent(self):
        """Test low consensus threshold."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(
            reference,
            test,
            methods=["psi", "wasserstein"],
            consensus_threshold=0.33,
        )

        feature_result = result.feature_results[0]

        # With threshold=0.33, need at least 1/3 methods
        # If any method detected drift, should be flagged
        if feature_result.n_methods_detected > 0:
            assert feature_result.drift_probability >= 0.33


class TestDriftScenarios:
    """Test various drift detection scenarios."""

    def test_single_feature_drift(self):
        """Test drift in only one of multiple features."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(1.5, 1, 1000),  # Strong drift
                "feature2": np.random.normal(0, 1, 1000),  # No drift
                "feature3": np.random.normal(0, 1, 1000),  # No drift
            }
        )

        result = analyze_drift(reference, test)

        # At least feature1 should drift
        assert "feature1" in result.drifted_features
        assert result.n_features_drifted >= 1
        assert result.overall_drifted is True

    def test_all_features_drift(self):
        """Test drift in all features."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(1.5, 1, 1000),  # Strong drift
                "feature2": np.random.normal(-1.5, 1, 1000),  # Strong drift
            }
        )

        result = analyze_drift(reference, test)

        # Both features should drift
        assert result.n_features_drifted == 2
        assert set(result.drifted_features) == {"feature1", "feature2"}
        assert result.overall_drifted is True

    def test_multivariate_drift_only(self):
        """Test multivariate drift detected but not univariate."""
        np.random.seed(42)
        # Create correlated features where individual features may not drift
        # but their joint distribution does
        n = 500
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, n),
                "feature2": np.random.normal(0, 1, n),
            }
        )

        # Rotate the distribution (multivariate drift)
        theta = np.pi / 4
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        test_rotated = reference.values @ rotation.T
        test = pd.DataFrame(test_rotated, columns=["feature1", "feature2"])

        result = analyze_drift(reference, test, methods=["domain_classifier"])

        # Domain classifier should detect the multivariate drift
        assert result.domain_classifier_result is not None
        # AUC should be reasonably high (distribution rotated)
        assert result.domain_classifier_result.auc > 0.5


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_method_name(self):
        """Test that invalid method names raise ValueError."""
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})
        test = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})

        with pytest.raises(ValueError, match="Invalid methods"):
            analyze_drift(reference, test, methods=["invalid_method"])

    def test_none_inputs(self):
        """Test that None inputs raise ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            analyze_drift(None, pd.DataFrame({"feature1": [1, 2, 3]}))

        with pytest.raises(ValueError, match="must not be None"):
            analyze_drift(pd.DataFrame({"feature1": [1, 2, 3]}), None)

    def test_missing_features(self):
        """Test that missing features raise ValueError."""
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})
        test = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})

        with pytest.raises(ValueError, match="Features not found"):
            analyze_drift(reference, test, features=["missing_feature"])

    def test_empty_features_list(self):
        """Test that empty features list raises ValueError."""
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})
        test = pd.DataFrame({"feature1": np.random.normal(0, 1, 100)})

        with pytest.raises(ValueError, match="No features to analyze"):
            analyze_drift(reference, test, features=[])

    def test_no_numeric_features(self):
        """Test with non-numeric features."""
        reference = pd.DataFrame({"category": ["A", "B", "C"] * 10})
        test = pd.DataFrame({"category": ["A", "B", "C"] * 10})

        # Should raise error since no numeric features
        with pytest.raises(ValueError, match="No features to analyze"):
            analyze_drift(reference, test)


class TestMethodConfigs:
    """Test method-specific configurations."""

    def test_custom_psi_config(self):
        """Test custom PSI configuration."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(
            reference,
            test,
            methods=["psi"],
            psi_config={"n_bins": 20, "psi_threshold_red": 0.15},
        )

        # Check that custom config was used
        feature_result = result.feature_results[0]
        assert feature_result.psi_result is not None
        # Should have used 20 bins
        assert feature_result.psi_result.n_bins == 20

    def test_custom_wasserstein_config(self):
        """Test custom Wasserstein configuration."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(
            reference,
            test,
            methods=["wasserstein"],
            wasserstein_config={
                "p": 2,
                "threshold_calibration": True,
                "n_permutations": 500,
            },
        )

        feature_result = result.feature_results[0]
        assert feature_result.wasserstein_result is not None
        # Should have used p=2
        assert feature_result.wasserstein_result.p == 2

    def test_custom_domain_classifier_config(self):
        """Test custom domain classifier configuration."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(0, 1, 500),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 500),
                "feature2": np.random.normal(0, 1, 500),
            }
        )

        result = analyze_drift(
            reference,
            test,
            methods=["domain_classifier"],
            domain_classifier_config={
                "model_type": "sklearn",
                "threshold": 0.7,
                "n_estimators": 50,
            },
        )

        assert result.domain_classifier_result is not None
        assert result.domain_classifier_result.model_type == "sklearn"
        assert result.domain_classifier_result.threshold == 0.7


class TestResultSchema:
    """Test result schema and helper methods."""

    def test_feature_drift_result_summary(self):
        """Test FeatureDriftResult.summary() method."""
        from ml4t.diagnostic.evaluation.drift import PSIResult, WassersteinResult

        psi_result = PSIResult(
            psi=0.15,
            bin_psi=np.array([0.05, 0.10]),
            bin_edges=np.array([0, 1, 2]),
            reference_counts=np.array([50, 50]),
            test_counts=np.array([60, 40]),
            reference_percents=np.array([0.5, 0.5]),
            test_percents=np.array([0.6, 0.4]),
            n_bins=2,
            is_categorical=False,
            alert_level="yellow",
            interpretation="Small change detected",
        )

        wasserstein_result = WassersteinResult(
            distance=0.25,
            p=1,
            threshold=0.2,
            p_value=0.03,
            drifted=True,
            n_reference=100,
            n_test=100,
            reference_stats={"mean": 0.0, "std": 1.0},
            test_stats={"mean": 0.5, "std": 1.0},
            threshold_calibration_config=None,
            interpretation="Drift detected",
            computation_time=0.1,
        )

        feature_result = FeatureDriftResult(
            feature="test_feature",
            psi_result=psi_result,
            wasserstein_result=wasserstein_result,
            drifted=True,
            n_methods_run=2,
            n_methods_detected=2,
            drift_probability=1.0,
            interpretation="Both methods detected drift",
        )

        summary = feature_result.summary()
        assert "test_feature" in summary
        assert "2/2 methods" in summary
        assert "PSI: 0.15" in summary
        assert "Wasserstein: 0.25" in summary

    def test_drift_summary_result_summary(self):
        """Test DriftSummaryResult.summary() method."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(1.0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )

        result = analyze_drift(reference, test)
        summary = result.summary()

        assert "Drift Analysis Summary" in summary
        assert "Methods Used:" in summary
        assert "Total Features: 2" in summary
        assert "Computation Time:" in summary

    def test_drift_summary_to_dataframe(self):
        """Test DriftSummaryResult.to_dataframe() method."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(1.0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )

        result = analyze_drift(reference, test, methods=["psi", "wasserstein"])
        df = result.to_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "feature" in df.columns
        assert "drifted" in df.columns
        assert "drift_probability" in df.columns
        assert "psi" in df.columns
        assert "wasserstein_distance" in df.columns
        assert len(df) == 2  # Two features

    def test_polars_input(self):
        """Test that polars DataFrames work as input."""
        np.random.seed(42)
        reference = pl.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )
        test = pl.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
            }
        )

        result = analyze_drift(reference, test)

        assert isinstance(result, DriftSummaryResult)
        assert result.n_features == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        """Test with a single feature."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0.5, 1, 1000)})

        result = analyze_drift(reference, test)

        assert result.n_features == 1
        assert len(result.feature_results) == 1

    def test_small_sample_size(self):
        """Test with small sample size."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 50),
                "feature2": np.random.normal(0, 1, 50),
            }
        )

        # Should still work, though results may be less reliable
        result = analyze_drift(reference, test)

        assert result.n_features == 2
        assert isinstance(result, DriftSummaryResult)

    def test_feature_subset(self):
        """Test analyzing a subset of features."""
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            }
        )
        test = pd.DataFrame(
            {
                "feature1": np.random.normal(0.5, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            }
        )

        result = analyze_drift(reference, test, features=["feature1", "feature2"])

        assert result.n_features == 2
        assert len(result.feature_results) == 2
        feature_names = [r.feature for r in result.feature_results]
        assert set(feature_names) == {"feature1", "feature2"}

    def test_computation_time_tracking(self):
        """Test that computation time is tracked."""
        np.random.seed(42)
        reference = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
        test = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})

        result = analyze_drift(reference, test)

        assert result.computation_time > 0
        assert result.computation_time < 60  # Should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
