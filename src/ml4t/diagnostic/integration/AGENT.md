# integration/ - 1261 Lines

Cross-library integration contracts for ml4t ecosystem.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| backtest_contract.py | 671 | ml4t-backtest integration (TradeRecord, EvaluationExport) |
| data_contract.py | 316 | ml4t-data integration (DataQualityReport, anomaly detection) |
| engineer_contract.py | 226 | ml4t-engineer integration (preprocessing recommendations) |
| __init__.py | 48 | Public exports |

## ml4t-backtest Integration

**Primary use case**: Evaluate backtest results and compare live vs backtest performance.

### Key Classes

| Class | Purpose |
|-------|---------|
| `TradeRecord` | Individual trade data with PnL validation |
| `StrategyMetadata` | Strategy identification (id, version, environment) |
| `EvaluationExport` | Complete evaluation results for storage |
| `ComparisonRequest` | Live vs backtest comparison input |
| `ComparisonResult` | Comparison output with recommendation |
| `PromotionWorkflow` | Paper-to-live promotion criteria |

### Workflow Example

```python
# 1. Convert backtest trades (use ml4t.backtest.analytics.bridge)
from ml4t.backtest.analytics.bridge import to_trade_records
records = to_trade_records(engine.broker.trades)

# 2. Create TradeRecord objects
from ml4t.diagnostic.integration import TradeRecord
trade_records = [TradeRecord(**r) for r in records]

# 3. Analyze with diagnostic tools
from ml4t.diagnostic.evaluation import TradeAnalysis
analyzer = TradeAnalysis(trade_records)
```

### Validation Features

- PnL consistency check (entry/exit prices vs reported PnL)
- Timestamp ordering (entry < exit)
- Duration validation (must be positive)
- 1% tolerance for fees/slippage discrepancies

## ml4t-data Integration

| Class | Purpose |
|-------|---------|
| `DataQualityReport` | Quality assessment results |
| `DataValidationRequest` | Validation request parameters |
| `DataAnomaly` | Individual anomaly record |
| `Severity` | Anomaly severity levels |

## ml4t-engineer Integration

| Class | Purpose |
|-------|---------|
| `PreprocessingRecommendation` | Feature preprocessing suggestions |
| `EngineerConfig` | Feature engineering configuration |
| `TransformType` | Transformation types enum |
