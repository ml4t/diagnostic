"""Test configuration file I/O operations.

This test suite verifies:
1. YAML loading and saving
2. JSON loading and saving
3. Dictionary loading and export
4. Auto-detection from file extension
5. Round-trip fidelity (save → load → compare)
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml


class TestYAMLIO:
    """Test YAML serialization and deserialization."""

    def test_yaml_roundtrip_diagnostic(self):
        """Test YAML roundtrip for DiagnosticConfig."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig.for_research()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Load and compare
            loaded = DiagnosticConfig.from_yaml(yaml_path)
            assert loaded == config

    def test_yaml_roundtrip_portfolio(self):
        """Test YAML roundtrip for PortfolioConfig."""
        from ml4t.diagnostic.config import PortfolioConfig

        config = PortfolioConfig.for_research()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = PortfolioConfig.from_yaml(yaml_path)
            assert loaded == config

    def test_yaml_roundtrip_statistical(self):
        """Test YAML roundtrip for StatisticalConfig."""
        from ml4t.diagnostic.config import StatisticalConfig

        config = StatisticalConfig.for_research()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = StatisticalConfig.from_yaml(yaml_path)
            assert loaded == config

    def test_yaml_roundtrip_report_config(self):
        """Test YAML roundtrip for ReportConfig."""
        from ml4t.diagnostic.config import ReportConfig

        config = ReportConfig.for_publication()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = ReportConfig.from_yaml(yaml_path)
            assert loaded == config

    def test_yaml_creates_parent_directories(self):
        """Test that to_yaml creates parent directories."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "subdir" / "nested" / "config.yaml"
            config.to_yaml(yaml_path)

            assert yaml_path.exists()
            assert yaml_path.parent.exists()

    def test_yaml_file_not_found(self):
        """Test that from_yaml raises FileNotFoundError for missing file."""
        from ml4t.diagnostic.config import DiagnosticConfig

        with pytest.raises(FileNotFoundError):
            DiagnosticConfig.from_yaml("/nonexistent/config.yaml")

    def test_yaml_preserves_nested_structure(self):
        """Test that YAML preserves nested configuration structure."""
        from ml4t.diagnostic.config import DiagnosticConfig, ICSettings
        from ml4t.diagnostic.config.validation import CorrelationMethod

        config = DiagnosticConfig(
            ic=ICSettings(
                method=CorrelationMethod.SPEARMAN,
                lag_structure=[0, 1, 5, 10, 21],
                hac_adjustment=True,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Read raw YAML to verify structure
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            assert "ic" in data
            assert data["ic"]["method"] == "spearman"
            assert data["ic"]["hac_adjustment"] is True

            # Verify round-trip
            loaded = DiagnosticConfig.from_yaml(yaml_path)
            assert loaded.ic.method == CorrelationMethod.SPEARMAN


class TestJSONIO:
    """Test JSON serialization and deserialization."""

    def test_json_roundtrip_diagnostic(self):
        """Test JSON roundtrip for DiagnosticConfig."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig.for_quick_analysis()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = DiagnosticConfig.from_json(json_path)
            assert loaded == config

    def test_json_roundtrip_portfolio(self):
        """Test JSON roundtrip for PortfolioConfig."""
        from ml4t.diagnostic.config import PortfolioConfig

        config = PortfolioConfig.for_production()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = PortfolioConfig.from_json(json_path)
            assert loaded == config

    def test_json_roundtrip_statistical(self):
        """Test JSON roundtrip for StatisticalConfig."""
        from ml4t.diagnostic.config import StatisticalConfig

        config = StatisticalConfig.for_publication()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = StatisticalConfig.from_json(json_path)
            assert loaded == config

    def test_json_roundtrip_report_config(self):
        """Test JSON roundtrip for ReportConfig."""
        from ml4t.diagnostic.config import ReportConfig

        config = ReportConfig.for_programmatic_access()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = ReportConfig.from_json(json_path)
            assert loaded == config

    def test_json_indentation(self):
        """Test that JSON is properly indented."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path, indent=4)

            with open(json_path) as f:
                content = f.read()

            # Should be pretty-printed (not single line)
            assert "\n" in content
            assert content.count("\n") > 10

    def test_json_creates_parent_directories(self):
        """Test that to_json creates parent directories."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "subdir" / "nested" / "config.json"
            config.to_json(json_path)

            assert json_path.exists()
            assert json_path.parent.exists()

    def test_json_file_not_found(self):
        """Test that from_json raises FileNotFoundError for missing file."""
        from ml4t.diagnostic.config import DiagnosticConfig

        with pytest.raises(FileNotFoundError):
            DiagnosticConfig.from_json("/nonexistent/config.json")


class TestDictIO:
    """Test dictionary loading and export."""

    def test_from_dict_diagnostic(self):
        """Test loading DiagnosticConfig from dict."""
        from ml4t.diagnostic.config import DiagnosticConfig

        data = {
            "stationarity": {"adf_enabled": True, "kpss_enabled": False},
        }

        config = DiagnosticConfig.from_dict(data)
        assert config.stationarity.adf_enabled is True
        assert config.stationarity.kpss_enabled is False

    def test_from_dict_portfolio(self):
        """Test loading PortfolioConfig from dict."""
        from ml4t.diagnostic.config import PortfolioConfig

        data = {
            "metrics": {"risk_free_rate": 0.02, "periods_per_year": 252},
        }

        config = PortfolioConfig.from_dict(data)
        assert config.metrics.risk_free_rate == 0.02

    def test_to_dict_python_mode(self):
        """Test to_dict() in python mode."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()
        d = config.to_dict(mode="python")

        assert isinstance(d, dict)
        assert "stationarity" in d
        assert "ic" in d

    def test_to_dict_json_mode(self):
        """Test to_dict() in json mode."""
        from ml4t.diagnostic.config import PortfolioConfig

        config = PortfolioConfig()
        d = config.to_dict(mode="json")

        assert isinstance(d, dict)

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str is not None

    def test_to_dict_exclude_none(self):
        """Test to_dict() with exclude_none."""
        from ml4t.diagnostic.config import ThresholdAnalysisSettings

        config = ThresholdAnalysisSettings(constraint_metric=None, constraint_value=None)

        d_with_none = config.to_dict(exclude_none=False)
        config.to_dict(exclude_none=True)

        assert "constraint_metric" in d_with_none
        assert "constraint_value" in d_with_none


class TestAutoDetection:
    """Test auto-detection of file format."""

    def test_from_file_yaml(self):
        """Test from_file() auto-detects YAML."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig.for_quick_analysis()

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = DiagnosticConfig.from_file(yaml_path)
            assert loaded == config

    def test_from_file_yml(self):
        """Test from_file() auto-detects .yml extension."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            yml_path = Path(tmpdir) / "config.yml"
            config.to_yaml(yml_path)

            loaded = DiagnosticConfig.from_file(yml_path)
            assert loaded == config

    def test_from_file_json(self):
        """Test from_file() auto-detects JSON."""
        from ml4t.diagnostic.config import PortfolioConfig

        config = PortfolioConfig.for_research()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = PortfolioConfig.from_file(json_path)
            assert loaded == config

    def test_from_file_unsupported_extension(self):
        """Test from_file() rejects unsupported extensions."""
        from ml4t.diagnostic.config import DiagnosticConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "config.txt"
            txt_path.write_text("some content")

            with pytest.raises(ValueError, match="Unsupported file type"):
                DiagnosticConfig.from_file(txt_path)

    def test_from_file_nonexistent(self):
        """Test from_file() raises FileNotFoundError for missing file."""
        from ml4t.diagnostic.config import DiagnosticConfig

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            DiagnosticConfig.from_file("/nonexistent/config.yaml")


class TestRoundTripFidelity:
    """Test that configurations maintain fidelity through save/load cycles."""

    def test_roundtrip_with_enums(self):
        """Test roundtrip with enum fields."""
        from ml4t.diagnostic.config import (
            CorrelationSettings,
            DiagnosticConfig,
            DistributionSettings,
        )
        from ml4t.diagnostic.config.validation import CorrelationMethod, NormalityTest

        config = DiagnosticConfig(
            correlation=CorrelationSettings(
                methods=[
                    CorrelationMethod.PEARSON,
                    CorrelationMethod.SPEARMAN,
                ]
            ),
            distribution=DistributionSettings(
                normality_tests=[
                    NormalityTest.JARQUE_BERA,
                    NormalityTest.SHAPIRO,
                ]
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = DiagnosticConfig.from_json(json_path)
            assert loaded.correlation.methods == config.correlation.methods
            assert loaded.distribution.normality_tests == config.distribution.normality_tests

    def test_roundtrip_with_nested_dicts(self):
        """Test roundtrip with nested dictionary fields."""
        from ml4t.diagnostic.config import PortfolioBayesianSettings
        from ml4t.diagnostic.config.validation import BayesianPriorDistribution

        config = PortfolioBayesianSettings(
            prior_distribution=BayesianPriorDistribution.STUDENT_T,
            prior_params={"df": 3, "loc": 0.0, "scale": 1.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            loaded = PortfolioBayesianSettings.from_yaml(yaml_path)
            assert loaded.prior_params == config.prior_params

    def test_multiple_roundtrips(self):
        """Test that multiple save/load cycles maintain fidelity."""
        from ml4t.diagnostic.config import StatisticalConfig

        config_original = StatisticalConfig.for_research()

        with tempfile.TemporaryDirectory() as tmpdir:
            # First roundtrip
            path1 = Path(tmpdir) / "config1.yaml"
            config_original.to_yaml(path1)
            config1 = StatisticalConfig.from_yaml(path1)

            # Second roundtrip
            path2 = Path(tmpdir) / "config2.yaml"
            config1.to_yaml(path2)
            config2 = StatisticalConfig.from_yaml(path2)

            # Third roundtrip
            path3 = Path(tmpdir) / "config3.yaml"
            config2.to_yaml(path3)
            config3 = StatisticalConfig.from_yaml(path3)

            # All should be equal
            assert config_original == config1
            assert config1 == config2
            assert config2 == config3

    def test_cross_format_roundtrip(self):
        """Test saving in one format and loading in another."""
        from ml4t.diagnostic.config import ReportConfig

        config = ReportConfig.for_publication()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as YAML
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            # Load and save as JSON
            loaded_from_yaml = ReportConfig.from_yaml(yaml_path)
            json_path = Path(tmpdir) / "config.json"
            loaded_from_yaml.to_json(json_path)

            # Load from JSON
            loaded_from_json = ReportConfig.from_json(json_path)

            # Should be equal
            assert config == loaded_from_yaml
            assert loaded_from_yaml == loaded_from_json


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_validate_fully_valid_config(self):
        """Test validate_fully() on valid config returns empty list."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config = DiagnosticConfig()
        errors = config.validate_fully()
        assert errors == []

    def test_validate_fully_with_invalid_modification(self):
        """Test validate_fully() catches validation errors."""
        from pydantic import ValidationError

        from ml4t.diagnostic.config import StatisticalConfig

        config = StatisticalConfig()

        # Try to set an invalid value (negative significance level)
        # This should fail at assignment time due to validate_assignment=True
        with pytest.raises(ValidationError):
            config.dsr.significance_level = -0.5


class TestStatisticalTestConfig:
    """Test StatisticalTestConfig base class."""

    def test_statistical_test_config_defaults(self):
        """Test default values for StatisticalTestConfig."""
        from ml4t.diagnostic.config.base import StatisticalTestConfig

        config = StatisticalTestConfig()
        assert config.enabled is True
        assert config.significance_level == 0.05

    def test_statistical_test_config_custom(self):
        """Test custom values for StatisticalTestConfig."""
        from ml4t.diagnostic.config.base import StatisticalTestConfig

        config = StatisticalTestConfig(enabled=False, significance_level=0.01)
        assert config.enabled is False
        assert config.significance_level == 0.01

    def test_statistical_test_config_validation(self):
        """Test validation of significance_level bounds."""
        from pydantic import ValidationError

        from ml4t.diagnostic.config.base import StatisticalTestConfig

        # Valid values
        StatisticalTestConfig(significance_level=0.001)
        StatisticalTestConfig(significance_level=0.10)

        # Invalid values
        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=0.0001)  # Too low

        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=0.5)  # Too high


class TestRuntimeConfig:
    """Test RuntimeConfig class."""

    def test_runtime_config_defaults(self):
        """Test default values for RuntimeConfig."""
        from ml4t.diagnostic.config.base import RuntimeConfig

        config = RuntimeConfig()
        assert config.n_jobs == -1
        assert config.cache_enabled is True
        assert config.cache_ttl is None
        assert config.verbose is False

    def test_runtime_config_custom(self):
        """Test custom values for RuntimeConfig."""
        from pathlib import Path

        from ml4t.diagnostic.config.base import RuntimeConfig

        config = RuntimeConfig(
            n_jobs=4,
            cache_enabled=False,
            cache_dir=Path("/tmp/custom_cache"),
            cache_ttl=3600,
            verbose=True,
        )
        assert config.n_jobs == 4
        assert config.cache_enabled is False
        assert config.cache_dir == Path("/tmp/custom_cache")
        assert config.cache_ttl == 3600
        assert config.verbose is True

    def test_runtime_config_creates_cache_dir(self):
        """Test that cache_dir is created when cache_enabled is True."""
        from ml4t.diagnostic.config.base import RuntimeConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "cache"
            RuntimeConfig(cache_enabled=True, cache_dir=cache_path)
            # The model_post_init should have created the directory
            assert cache_path.exists()

    def test_runtime_config_no_cache_dir_when_disabled(self):
        """Test that cache_dir is not created when cache_enabled is False."""
        from ml4t.diagnostic.config.base import RuntimeConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "nested" / "no_cache"
            RuntimeConfig(cache_enabled=False, cache_dir=cache_path)
            # Should NOT create the directory when disabled
            assert not cache_path.exists()

    def test_runtime_config_n_jobs_validation(self):
        """Test validation of n_jobs bounds."""
        from pydantic import ValidationError

        from ml4t.diagnostic.config.base import RuntimeConfig

        # Valid values
        RuntimeConfig(n_jobs=-1)
        RuntimeConfig(n_jobs=1)
        RuntimeConfig(n_jobs=16)

        # Invalid value
        with pytest.raises(ValidationError):
            RuntimeConfig(n_jobs=-2)


class TestConfigDiff:
    """Test configuration comparison functionality."""

    def test_diff_identical_configs(self):
        """Test diff() on identical configs returns empty dict."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config1 = DiagnosticConfig()
        config2 = DiagnosticConfig()

        diff = config1.diff(config2)
        assert diff == {}

    def test_diff_different_configs(self):
        """Test diff() identifies differences."""
        from ml4t.diagnostic.config import DiagnosticConfig

        config1 = DiagnosticConfig.for_quick_analysis()
        config2 = DiagnosticConfig.for_research()

        diff = config1.diff(config2)
        assert len(diff) > 0

        # Should have differences in HAC adjustment
        assert any("hac_adjustment" in key for key in diff)

    def test_diff_nested_changes(self):
        """Test diff() detects nested changes."""
        from ml4t.diagnostic.config import PortfolioConfig, PortfolioMetricsSettings

        config1 = PortfolioConfig(metrics=PortfolioMetricsSettings(risk_free_rate=0.0))
        config2 = PortfolioConfig(metrics=PortfolioMetricsSettings(risk_free_rate=0.02))

        diff = config1.diff(config2)
        assert "metrics.risk_free_rate" in diff
        assert diff["metrics.risk_free_rate"] == (0.0, 0.02)

    def test_diff_type_mismatch(self):
        """Test diff() raises TypeError on type mismatch."""
        from ml4t.diagnostic.config import DiagnosticConfig, PortfolioConfig

        config1 = DiagnosticConfig()
        config2 = PortfolioConfig()

        with pytest.raises(TypeError, match="Cannot compare"):
            config1.diff(config2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
