# splitters/ - Time-Series Cross-Validation

Leakage-aware cross-validation for financial time series.

## Main Components

- `WalkForwardCV` for rolling or expanding validation with optional held-out test periods
- `CombinatorialCV` for CPCV with purging, embargo, and path distributions
- `TradingCalendar` and `CalendarConfig` for session-aware splitting
- `WalkForwardConfig` and `CombinatorialConfig` for reproducible config-first workflows
- `save_folds`, `load_folds`, `verify_folds` for persistence and reruns

## Notes

- `WalkForwardCV` supports held-out test windows via `test_period` or `test_start` / `test_end`
- `CombinatorialCV` is the path used by `ValidatedCrossValidation`
- The public guide is [../../../../docs/user-guide/cv-configuration.md](../../../../docs/user-guide/cv-configuration.md)
