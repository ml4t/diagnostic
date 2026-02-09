# splitters/ - Cross-Validation

Time-series CV with purging and embargo.

## Documentation

**User Guide**: [docs/user-guide/cv-configuration.md](../../../../docs/user-guide/cv-configuration.md)
- JSON/YAML configuration format
- Fold persistence format
- Reproducible experiment workflows
- **Held-out test configuration**
- **Trading-day-aware gaps**

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| combinatorial.py | 1392 | `CombinatorialCV` (CPCV) |
| walk_forward.py | ~1100 | `WalkForwardCV` with held-out test & backward validation |
| base.py | 501 | `BaseSplitter` abstract |
| calendar.py | ~550 | `TradingCalendar` + trading day utilities |
| config.py | ~450 | Configuration classes (WalkForwardConfig, CombinatorialConfig) |
| group_isolation.py | 329 | Multi-asset isolation |
| persistence.py | 316 | Fold save/load |
| calendar_config.py | 92 | `CalendarConfig` + presets |

## Key Classes

`CombinatorialCV`, `WalkForwardCV`, `TradingCalendar`, `WalkForwardConfig`, `CombinatorialConfig`

## Key Features

### Held-Out Test Period
Reserve most recent data for final evaluation:
```python
cv = WalkForwardCV(
    n_splits=5,
    test_period="52D",  # Last 52 days for held-out test
    fold_direction="backward",  # Folds step backward from test boundary
)
for train_idx, val_idx in cv.split(X):
    # Validation folds
    pass
# Final evaluation
final_score = model.score(X.iloc[cv.test_indices_], y.iloc[cv.test_indices_])
```

### Trading-Day-Aware Gaps
When `calendar_id` is set, `label_horizon` is interpreted as trading days:
```python
cv = WalkForwardCV(
    label_horizon=5,  # 5 TRADING days, not calendar days
    calendar="NYSE",
)
```

### Calendar Utilities
```python
from ml4t.diagnostic.splitters.calendar import TradingCalendar, trading_days_to_timedelta

calendar = TradingCalendar('NYSE')
calendar.previous_trading_day(ref_date, n=5)  # 5 trading days back
calendar.next_trading_day(ref_date, n=3)      # 3 trading days forward
calendar.trading_days_between(start, end)     # Count trading days
```
