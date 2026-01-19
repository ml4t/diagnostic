# ML4T Diagnostic Configuration System

Type-safe, validated configuration using Pydantic v2.

## Overview

The configuration system provides 10 primary config classes:

| Config | Purpose |
|--------|---------|
| `DiagnosticConfig` | Feature diagnostics (stationarity, IC, volatility) |
| `StatisticalConfig` | Statistical tests (PSR, DSR, MinTRL, FDR) |
| `PortfolioConfig` | Portfolio analysis (metrics, Bayesian, drawdown) |
| `TradeConfig` | Trade analysis (extraction, SHAP, clustering) |
| `SignalConfig` | Signal analysis (IC, quantiles, RAS) |
| `EventConfig` | Event studies |
| `BarrierConfig` | Triple barrier analysis |
| `ReportConfig` | Report generation (HTML, JSON, output) |
| `RuntimeConfig` | Execution settings (n_jobs, cache, verbose) |
| `MultiSignalAnalysisConfig` | Multi-signal comparison |

## Quick Start

```python
from ml4t.diagnostic.config import (
    DiagnosticConfig,
    PortfolioConfig,
    StatisticalConfig,
    RuntimeConfig,
)

# Use defaults (sensible out-of-the-box)
config = DiagnosticConfig()
portfolio_config = PortfolioConfig()

# Use presets
quick_config = DiagnosticConfig.for_quick_analysis()
research_config = DiagnosticConfig.for_research()
production_config = DiagnosticConfig.for_production()

# Load from YAML
config = DiagnosticConfig.from_yaml("config.yaml")

# Save to YAML
config.to_yaml("config.yaml")
```

## Architecture

### File Structure

```
config/
├── __init__.py              # Public API exports
├── base.py                  # BaseConfig, RuntimeConfig
├── validation.py            # Custom validators and types
├── feature_config.py        # DiagnosticConfig + Settings
├── portfolio_config.py      # PortfolioConfig + Settings
├── sharpe_config.py         # StatisticalConfig + Settings
├── signal_config.py         # SignalConfig + Settings
├── trade_analysis_config.py # TradeConfig + Settings
├── event_config.py          # EventConfig + WindowSettings
├── barrier_config.py        # BarrierConfig + Settings
├── multi_signal_config.py   # MultiSignalAnalysisConfig
└── report_config.py         # ReportConfig + Settings
```

### Design Pattern: Single-Level Nesting

All configs use a flat structure with Settings classes for grouping:

```python
from ml4t.diagnostic.config import DiagnosticConfig, StationaritySettings

config = DiagnosticConfig(
    stationarity=StationaritySettings(
        enabled=True,
        significance_level=0.01,
    )
)

# Access: config.stationarity.enabled
```

## Module Configurations

### Feature Diagnostics

```python
from ml4t.diagnostic.config import (
    DiagnosticConfig,
    StationaritySettings,
    ICSettings,
)

config = DiagnosticConfig(
    stationarity=StationaritySettings(
        significance_level=0.01,
        adf_enabled=True,
        kpss_enabled=True,
    ),
    ic=ICSettings(
        lag_structure=[0, 1, 5, 10, 21],
        hac_adjustment=True,
    ),
)
```

**Settings**: StationaritySettings, ACFSettings, VolatilitySettings, DistributionSettings,
CorrelationSettings, PCASettings, ClusteringSettings, RedundancySettings, ICSettings,
BinaryClassificationSettings, ThresholdAnalysisSettings, MLDiagnosticsSettings

### Portfolio Analysis

```python
from ml4t.diagnostic.config import (
    PortfolioConfig,
    MetricsSettings,
    PortfolioMetric,
)

config = PortfolioConfig(
    metrics=MetricsSettings(
        metrics=[
            PortfolioMetric.SHARPE,
            PortfolioMetric.SORTINO,
            PortfolioMetric.MAX_DRAWDOWN,
        ],
        risk_free_rate=0.02,
        periods_per_year=252,
    ),
)
```

**Settings**: MetricsSettings, BayesianSettings, TimeAggregationSettings, DrawdownSettings

### Statistical Testing

```python
from ml4t.diagnostic.config import (
    StatisticalConfig,
    PSRSettings,
    DSRSettings,
)

config = StatisticalConfig(
    psr=PSRSettings(
        target_sharpe=1.0,
        confidence_level=0.95,
    ),
    dsr=DSRSettings(
        n_trials=500,
        prob_zero_sharpe=0.5,
    ),
)
```

**Settings**: PSRSettings, MinTRLSettings, DSRSettings, FDRSettings

### Trade Analysis

```python
from ml4t.diagnostic.config import (
    TradeConfig,
    ExtractionSettings,
    ClusteringSettings,
)

config = TradeConfig(
    extraction=ExtractionSettings(n_worst=50, n_best=20),
    clustering=ClusteringSettings(min_cluster_size=10),
)
```

**Settings**: ExtractionSettings, FilterSettings, AlignmentSettings, ClusteringSettings, HypothesisSettings

### Signal Analysis

```python
from ml4t.diagnostic.config import SignalConfig, ICSignalSettings

config = SignalConfig(
    ic=ICSignalSettings(
        method="spearman",
        periods=[1, 5, 10, 21],
    ),
)
```

**Settings**: ICSignalSettings, QuantileSettings, RASSettings, VisualizationSettings

## Presets

Each config provides common presets:

```python
# Quick exploratory analysis
config = DiagnosticConfig.for_quick_analysis()
config = PortfolioConfig.for_quick_analysis()

# Comprehensive research
config = DiagnosticConfig.for_research()
config = StatisticalConfig.for_research()

# Production monitoring
config = DiagnosticConfig.for_production()
config = TradeConfig.for_production()
```

## Serialization

```python
# YAML (recommended for human editing)
config.to_yaml("config.yaml")
config = DiagnosticConfig.from_yaml("config.yaml")

# JSON (better for APIs)
config.to_json("config.json")
config = DiagnosticConfig.from_json("config.json")

# Auto-detect from extension
config = DiagnosticConfig.from_file("config.yaml")

# Dictionary
config = DiagnosticConfig.from_dict({"verbose": True})
d = config.to_dict()
```

## Validation

```python
# Automatic validation on construction
from pydantic import ValidationError

try:
    config = StationaritySettings(significance_level=0.5)  # Invalid
except ValidationError as e:
    print(e)  # "significance_level must be <= 0.10"

# Manual validation
config = DiagnosticConfig()
errors = config.validate_fully()
```

## Runtime Configuration

Runtime settings are separate to avoid coupling with analysis configs:

```python
from ml4t.diagnostic.config import RuntimeConfig

runtime = RuntimeConfig(
    n_jobs=-1,           # Use all CPU cores
    cache_enabled=True,  # Cache expensive computations
    verbose=True,        # Show progress
    random_state=42,     # Reproducibility
)

# Pass as separate parameter
result = analyze_features(df, config=DiagnosticConfig(), runtime=runtime)
```

## References

- **Pydantic v2**: https://docs.pydantic.dev/latest/
- **López de Prado, M.**: "Advances in Financial Machine Learning"
- **Bailey & López de Prado**: Multiple testing papers
