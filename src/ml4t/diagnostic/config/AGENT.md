# config/ - Pydantic Configuration

10 primary configs with `.for_quick_analysis()`, `.for_research()` presets.

## Primary Configs

| Config | Purpose |
|--------|---------|
| DiagnosticConfig | Feature diagnostics |
| StatisticalConfig | DSR, RAS, FDR |
| PortfolioConfig | Portfolio analysis |
| SignalConfig | Signal analysis |
| TradeConfig | Trade analysis |
| EventConfig | Event studies |
| BarrierConfig | Barrier analysis |
| ReportConfig | Report generation |
| RuntimeConfig | Execution settings |

## Pattern

```python
config = DiagnosticConfig.for_research()
config.stationarity.enabled  # Single-level nesting
```
