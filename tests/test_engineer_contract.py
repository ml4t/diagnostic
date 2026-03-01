"""Tests for ML4T Engineer integration contract."""

import pytest
from pydantic import ValidationError

from ml4t.diagnostic.integration import (
    EngineerConfig,
    PreprocessingRecommendation,
    TransformType,
)
from ml4t.diagnostic.results.feature_results import (
    FeatureDiagnosticsResultSchema,
    StationarityTestResult,
)


class TestTransformType:
    """Test TransformType enum."""

    def test_all_values(self):
        """Test all transform types are defined."""
        expected = {"none", "log", "sqrt", "standardize", "normalize", "winsorize", "diff"}
        actual = {t.value for t in TransformType}
        assert actual == expected

    def test_enum_is_string(self):
        """Test TransformType inherits from str."""
        assert isinstance(TransformType.DIFF, str)
        assert TransformType.DIFF == "diff"


class TestPreprocessingRecommendation:
    """Test PreprocessingRecommendation schema."""

    def test_basic_recommendation(self):
        """Test creating basic recommendation."""
        rec = PreprocessingRecommendation(
            feature_name="returns",
            transform=TransformType.DIFF,
            reason="Non-stationary feature",
            confidence=0.9,
        )
        assert rec.feature_name == "returns"
        assert rec.transform == TransformType.DIFF
        assert rec.reason == "Non-stationary feature"
        assert rec.confidence == 0.9
        assert rec.diagnostics is None

    def test_recommendation_with_diagnostics(self):
        """Test recommendation with diagnostic details."""
        rec = PreprocessingRecommendation(
            feature_name="rsi_14",
            transform=TransformType.WINSORIZE,
            reason="Outliers detected",
            confidence=0.85,
            diagnostics={"p99": 95.2, "p1": 5.1},
        )
        assert rec.diagnostics == {"p99": 95.2, "p1": 5.1}

    def test_confidence_validation(self):
        """Test confidence must be in [0, 1]."""
        # Valid
        PreprocessingRecommendation(
            feature_name="test",
            transform=TransformType.NONE,
            reason="test",
            confidence=0.0,
        )
        PreprocessingRecommendation(
            feature_name="test",
            transform=TransformType.NONE,
            reason="test",
            confidence=1.0,
        )

        # Invalid
        with pytest.raises(ValidationError):
            PreprocessingRecommendation(
                feature_name="test",
                transform=TransformType.NONE,
                reason="test",
                confidence=1.5,
            )


class TestEngineerConfig:
    """Test EngineerConfig schema."""

    def test_basic_config(self):
        """Test creating basic config."""
        recommendations = [
            PreprocessingRecommendation(
                feature_name="f1",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.9,
            ),
            PreprocessingRecommendation(
                feature_name="f2",
                transform=TransformType.NONE,
                reason="Good as-is",
                confidence=0.95,
            ),
        ]
        config = EngineerConfig(recommendations=recommendations)
        assert len(config.recommendations) == 2
        assert config.metadata is None

    def test_config_with_metadata(self):
        """Test config with metadata."""
        config = EngineerConfig(
            recommendations=[],
            metadata={"timestamp": "2024-01-01", "version": "2.0.0"},
        )
        assert config.metadata == {"timestamp": "2024-01-01", "version": "2.0.0"}

    def test_to_dict_basic(self):
        """Test export to dictionary format."""
        recommendations = [
            PreprocessingRecommendation(
                feature_name="rsi_14",
                transform=TransformType.WINSORIZE,
                reason="Outliers detected",
                confidence=0.85,
            ),
            PreprocessingRecommendation(
                feature_name="log_returns",
                transform=TransformType.NONE,
                reason="Already good",
                confidence=0.90,
            ),
        ]
        config = EngineerConfig(recommendations=recommendations)
        result = config.to_dict()

        # Check structure
        assert "rsi_14" in result
        assert "log_returns" in result

        # Check rsi_14
        assert result["rsi_14"]["transform"] == "winsorize"
        assert result["rsi_14"]["reason"] == "Outliers detected"
        assert result["rsi_14"]["confidence"] == 0.85

        # Check log_returns
        assert result["log_returns"]["transform"] == "none"
        assert result["log_returns"]["reason"] == "Already good"
        assert result["log_returns"]["confidence"] == 0.90

    def test_to_dict_with_diagnostics(self):
        """Test export includes diagnostics."""
        recommendations = [
            PreprocessingRecommendation(
                feature_name="returns",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.95,
                diagnostics={"adf_pvalue": 0.82, "kpss_pvalue": 0.01},
            )
        ]
        config = EngineerConfig(recommendations=recommendations)
        result = config.to_dict()

        assert "returns" in result
        assert result["returns"]["diagnostics"] == {
            "adf_pvalue": 0.82,
            "kpss_pvalue": 0.01,
        }

    def test_get_recommendations_by_transform(self):
        """Test filtering recommendations by transform type."""
        recommendations = [
            PreprocessingRecommendation(
                feature_name="f1",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.9,
            ),
            PreprocessingRecommendation(
                feature_name="f2",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.85,
            ),
            PreprocessingRecommendation(
                feature_name="f3",
                transform=TransformType.NONE,
                reason="Stationary",
                confidence=0.95,
            ),
        ]
        config = EngineerConfig(recommendations=recommendations)

        # Get DIFF recommendations
        diff_recs = config.get_recommendations_by_transform(TransformType.DIFF)
        assert len(diff_recs) == 2
        assert all(r.transform == TransformType.DIFF for r in diff_recs)

        # Get NONE recommendations
        none_recs = config.get_recommendations_by_transform(TransformType.NONE)
        assert len(none_recs) == 1
        assert none_recs[0].feature_name == "f3"

        # Get non-existent transform
        log_recs = config.get_recommendations_by_transform(TransformType.LOG)
        assert len(log_recs) == 0

    def test_summary(self):
        """Test summary generation."""
        recommendations = [
            PreprocessingRecommendation(
                feature_name="f1",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.95,
            ),
            PreprocessingRecommendation(
                feature_name="f2",
                transform=TransformType.DIFF,
                reason="Non-stationary",
                confidence=0.92,
            ),
            PreprocessingRecommendation(
                feature_name="f3",
                transform=TransformType.WINSORIZE,
                reason="Outliers",
                confidence=0.88,
            ),
            PreprocessingRecommendation(
                feature_name="f4",
                transform=TransformType.NONE,
                reason="Good",
                confidence=0.75,
            ),
        ]
        config = EngineerConfig(recommendations=recommendations)
        summary = config.summary()

        # Check key elements
        assert "Total features: 4" in summary
        assert "DIFF: 2 features" in summary
        assert "WINSORIZE: 1 feature" in summary
        assert "NONE: 1 feature" in summary
        assert "High-confidence recommendations" in summary


class TestFeatureDiagnosticsIntegration:
    """Test FeatureDiagnosticsResult.to_engineer_config()."""

    def test_non_stationary_recommendation(self):
        """Test non-stationary features get DIFF recommendation."""
        # Create non-stationary result
        stationarity = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-1.5,
            adf_pvalue=0.82,
            adf_is_stationary=False,
            kpss_statistic=2.5,
            kpss_pvalue=0.01,
            kpss_is_stationary=False,
            pp_statistic=-1.2,
            pp_pvalue=0.75,
            pp_is_stationary=False,
        )

        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=[stationarity])
        config = diagnostics.to_engineer_config()

        assert len(config.recommendations) == 1
        rec = config.recommendations[0]
        assert rec.feature_name == "returns"
        assert rec.transform == TransformType.DIFF
        assert "non-stationary" in rec.reason.lower()
        assert rec.confidence >= 0.8

    def test_stationary_recommendation(self):
        """Test stationary features get NONE recommendation."""
        stationarity = StationarityTestResult(
            feature_name="log_returns",
            adf_statistic=-5.2,
            adf_pvalue=0.001,
            adf_is_stationary=True,
            kpss_statistic=0.2,
            kpss_pvalue=0.15,
            kpss_is_stationary=True,
        )

        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=[stationarity])
        config = diagnostics.to_engineer_config()

        assert len(config.recommendations) == 1
        rec = config.recommendations[0]
        assert rec.feature_name == "log_returns"
        assert rec.transform == TransformType.NONE
        assert "stationary" in rec.reason.lower()

    def test_mixed_signals_recommendation(self):
        """Test features with mixed stationarity signals."""
        # ADF says non-stationary, KPSS says stationary
        stationarity = StationarityTestResult(
            feature_name="volume",
            adf_statistic=-2.0,
            adf_pvalue=0.6,
            adf_is_stationary=False,
            kpss_statistic=0.3,
            kpss_pvalue=0.12,
            kpss_is_stationary=True,
        )

        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=[stationarity])
        config = diagnostics.to_engineer_config()

        assert len(config.recommendations) == 1
        rec = config.recommendations[0]
        # Should recommend DIFF but with lower confidence
        assert rec.transform == TransformType.DIFF
        assert rec.confidence < 0.8

    def test_multiple_features(self):
        """Test recommendations for multiple features."""
        tests = [
            StationarityTestResult(
                feature_name="f1",
                adf_is_stationary=False,
                kpss_is_stationary=False,
            ),
            StationarityTestResult(
                feature_name="f2",
                adf_is_stationary=True,
                kpss_is_stationary=True,
            ),
            StationarityTestResult(
                feature_name="f3",
                adf_is_stationary=False,
                kpss_is_stationary=True,
            ),
        ]

        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=tests)
        config = diagnostics.to_engineer_config()

        assert len(config.recommendations) == 3
        feature_names = {r.feature_name for r in config.recommendations}
        assert feature_names == {"f1", "f2", "f3"}

    def test_export_to_dict(self):
        """Test full workflow: diagnostics → config → QFeatures dict."""
        stationarity = StationarityTestResult(
            feature_name="price",
            adf_statistic=-1.2,
            adf_pvalue=0.85,
            adf_is_stationary=False,
            kpss_statistic=2.8,
            kpss_pvalue=0.005,
            kpss_is_stationary=False,
        )

        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=[stationarity])
        config = diagnostics.to_engineer_config()
        eng_dict = config.to_dict()

        # Check final format
        assert "price" in eng_dict
        assert eng_dict["price"]["transform"] == "diff"
        assert "confidence" in eng_dict["price"]
        assert "reason" in eng_dict["price"]
        assert "diagnostics" in eng_dict["price"]

    def test_metadata_included(self):
        """Test metadata is included in config."""
        stationarity = StationarityTestResult(feature_name="test", adf_is_stationary=True)
        diagnostics = FeatureDiagnosticsResultSchema(stationarity_tests=[stationarity])
        config = diagnostics.to_engineer_config()

        assert config.metadata is not None
        assert "created_at" in config.metadata
        assert "diagnostic_version" in config.metadata


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    def test_example_workflow(self):
        """Test the workflow shown in docstrings."""
        # Simulate evaluation results
        diagnostics = FeatureDiagnosticsResultSchema(
            stationarity_tests=[
                StationarityTestResult(
                    feature_name="rsi_14",
                    adf_is_stationary=True,
                    kpss_is_stationary=True,
                ),
                StationarityTestResult(
                    feature_name="price",
                    adf_is_stationary=False,
                    kpss_is_stationary=False,
                    pp_is_stationary=False,
                ),
            ]
        )

        # Step 1: Get QFeatures config
        eng_config = diagnostics.to_engineer_config()
        assert len(eng_config.recommendations) == 2

        # Step 2: Export for QFeatures
        preprocessing_dict = eng_config.to_dict()
        assert "rsi_14" in preprocessing_dict
        assert "price" in preprocessing_dict

        # Step 3: Verify format is correct for QFeatures
        for _feature_name, config_dict in preprocessing_dict.items():
            assert "transform" in config_dict
            assert "reason" in config_dict
            assert "confidence" in config_dict
            assert isinstance(config_dict["transform"], str)
            assert isinstance(config_dict["confidence"], float)

        # Step 4: Print summary
        summary = eng_config.summary()
        assert "Total features: 2" in summary
