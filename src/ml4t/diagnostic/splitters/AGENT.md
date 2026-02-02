# splitters/ - Cross-Validation

Time-series CV with purging and embargo.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| combinatorial.py | 1392 | `CombinatorialCV` (CPCV) |
| walk_forward.py | 757 | `WalkForwardCV` |
| base.py | 501 | `BaseSplitter` abstract |
| calendar.py | 421 | `TradingCalendar` |
| config.py | 315 | Configuration classes |
| group_isolation.py | 329 | Multi-asset isolation |
| persistence.py | 316 | Fold save/load |

## Key Classes

`CombinatorialCV`, `WalkForwardCV`, `TradingCalendar`
