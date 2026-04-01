# config/ - Configuration Surface

Pydantic-based configuration for diagnostics, statistical testing, reporting, and validation workflows.

## Primary Configs

- `DiagnosticConfig` for feature diagnostics
- `StatisticalConfig` for DSR, FDR, MinTRL, and related inference
- `SignalConfig` for signal analysis
- `TradeConfig` for trade and Trade-SHAP workflows
- `PortfolioConfig` for performance and risk evaluation
- `ReportConfig` for report output settings
- `BarrierConfig` and `EventConfig` for specialized workflows
- `RuntimeConfig` for execution-level defaults
- `ValidatedCrossValidationConfig` for the CPCV plus DSR wrapper
- `MultiSignalAnalysisConfig` for cross-signal comparison

## Common Patterns

```python
from ml4t.diagnostic.config import DiagnosticConfig, ValidatedCrossValidationConfig

diag = DiagnosticConfig.for_research()
vcv = ValidatedCrossValidationConfig(n_groups=10, n_test_groups=2, label_horizon=5)
```

## Related Guides

- Splitter serialization and fold persistence live in [../../../../docs/user-guide/cv-configuration.md](../../../../docs/user-guide/cv-configuration.md)
- Validation workflow guidance lives in [../../../../docs/user-guide/validation-tiers.md](../../../../docs/user-guide/validation-tiers.md)
