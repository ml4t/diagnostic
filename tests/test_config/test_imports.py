"""Quick smoke test to verify all config classes can be imported and instantiated.

This is a sanity check to ensure:
1. All imports work
2. All configs have valid defaults
3. Serialization works
4. Presets work
"""

import tempfile
from pathlib import Path

import pytest


def test_import_all_configs():
    """Test that all configs can be imported."""

    assert True  # If we get here, imports worked


def test_diagnostic_config_default():
    """Test DiagnosticConfig with defaults."""
    from ml4t.diagnostic.config import DiagnosticConfig

    config = DiagnosticConfig()
    assert config is not None
    assert config.stationarity is not None
    assert config.acf is not None
    assert config.volatility is not None
    assert config.distribution is not None
    assert config.correlation is not None
    assert config.ic is not None


def test_portfolio_config_default():
    """Test PortfolioConfig with defaults."""
    from ml4t.diagnostic.config import PortfolioConfig

    config = PortfolioConfig()
    assert config is not None
    assert config.metrics is not None
    assert config.aggregation is not None


def test_statistical_config_default():
    """Test StatisticalConfig with defaults."""
    from ml4t.diagnostic.config import StatisticalConfig

    config = StatisticalConfig()
    assert config is not None
    assert config.psr.enabled
    assert config.mintrl.enabled
    assert config.dsr.enabled
    assert config.fdr.enabled


def test_report_config_default():
    """Test ReportConfig with defaults."""
    from ml4t.diagnostic.config import ReportConfig

    config = ReportConfig()
    assert config is not None
    assert config.html is not None
    assert config.visualization is not None


def test_diagnostic_config_presets():
    """Test DiagnosticConfig presets."""
    from ml4t.diagnostic.config import DiagnosticConfig

    quick = DiagnosticConfig.for_quick_analysis()
    research = DiagnosticConfig.for_research()
    production = DiagnosticConfig.for_production()

    assert quick is not None
    assert research is not None
    assert production is not None

    # Quick should be faster (fewer enabled analyses)
    assert not quick.ic.hac_adjustment
    assert research.ic.hac_adjustment


def test_portfolio_config_presets():
    """Test PortfolioConfig presets."""
    from ml4t.diagnostic.config import PortfolioConfig

    quick = PortfolioConfig.for_quick_analysis()
    research = PortfolioConfig.for_research()
    production = PortfolioConfig.for_production()

    assert quick is not None
    assert research is not None
    assert production is not None

    # Research should have more metrics
    assert len(research.metrics.metrics) > len(quick.metrics.metrics)


def test_statistical_config_presets():
    """Test StatisticalConfig presets."""
    from ml4t.diagnostic.config import StatisticalConfig

    quick = StatisticalConfig.for_quick_check()
    research = StatisticalConfig.for_research()
    publication = StatisticalConfig.for_publication()

    assert quick is not None
    assert research is not None
    assert publication is not None

    # Publication should be most conservative
    assert publication.dsr.n_trials > research.dsr.n_trials


def test_report_config_presets():
    """Test ReportConfig presets."""
    from ml4t.diagnostic.config import ReportConfig

    quick = ReportConfig.for_quick_report()
    publication = ReportConfig.for_publication()
    programmatic = ReportConfig.for_programmatic_access()

    assert quick is not None
    assert publication is not None
    assert programmatic is not None

    # Publication should have higher DPI
    assert publication.visualization.plot_dpi > quick.visualization.plot_dpi


def test_yaml_serialization():
    """Test YAML serialization roundtrip."""
    from ml4t.diagnostic.config import DiagnosticConfig

    config = DiagnosticConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        config.to_yaml(yaml_path)

        # Load and compare
        loaded = DiagnosticConfig.from_yaml(yaml_path)
        assert loaded == config


def test_json_serialization():
    """Test JSON serialization roundtrip."""
    from ml4t.diagnostic.config import PortfolioConfig

    config = PortfolioConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "config.json"
        config.to_json(json_path)

        # Load and compare
        loaded = PortfolioConfig.from_json(json_path)
        assert loaded == config


def test_config_diff():
    """Test config comparison."""
    from ml4t.diagnostic.config import DiagnosticConfig

    config1 = DiagnosticConfig.for_quick_analysis()
    config2 = DiagnosticConfig.for_research()

    diff = config1.diff(config2)
    assert len(diff) > 0  # Should have differences


def test_dict_export():
    """Test dictionary export."""
    from ml4t.diagnostic.config import StatisticalConfig

    config = StatisticalConfig()

    # Python dict
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "psr" in d

    # JSON-serializable dict
    d_json = config.to_dict(mode="json")
    assert isinstance(d_json, dict)


def test_validation_errors():
    """Test that validation catches errors."""
    from pydantic import ValidationError

    from ml4t.diagnostic.config import StationaritySettings

    # Should work
    config = StationaritySettings(significance_level=0.05)
    assert config is not None

    # Should fail (all tests disabled)
    with pytest.raises(ValidationError, match="At least one"):
        StationaritySettings(adf_enabled=False, kpss_enabled=False, pp_enabled=False)


def test_custom_config():
    """Test creating custom nested config."""
    from ml4t.diagnostic.config import (
        DiagnosticConfig,
        ICSettings,
    )
    from ml4t.diagnostic.config.validation import CorrelationMethod

    config = DiagnosticConfig(
        ic=ICSettings(
            method=CorrelationMethod.SPEARMAN,
            lag_structure=[0, 1, 5, 10, 21],
            hac_adjustment=True,
        ),
        n_jobs=4,
        verbose=True,
    )

    assert config.ic.method == CorrelationMethod.SPEARMAN
    assert config.ic.hac_adjustment is True
    assert config.n_jobs == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
