# Cross-Validation Configuration Guide

This guide documents the JSON/YAML configuration format for ml4t-diagnostic's
cross-validation splitters, enabling reproducible experiments.

**See also**: [cross-validation.md](cross-validation.md) - Conceptual guide to CV methods

## Overview

The splitters module uses Pydantic-based configuration classes that support:

- **Serialization**: Save/load configs as JSON or YAML
- **Validation**: Automatic validation with clear error messages
- **Reproducibility**: Exact fold recreation from saved configs

## Configuration Classes

| Class | Purpose |
|-------|---------|
| `WalkForwardConfig` | Walk-forward (rolling/expanding window) CV |
| `CombinatorialConfig` | Combinatorial Purged Cross-Validation (CPCV) |
| `CalendarConfig` | Trading calendar for session-aware splits |

---

## WalkForwardConfig

Walk-forward validation where training data always precedes test data.

Supports two modes:
1. **Traditional walk-forward**: Validation folds step forward through time
2. **Held-out test with backward validation**: Reserve most recent data for final
   evaluation, validation folds step backward from the test boundary

### JSON Schema

```json
{
  "n_splits": 5,
  "test_size": 100,
  "train_size": null,
  "step_size": null,
  "label_horizon": 0,
  "embargo_td": null,
  "align_to_sessions": false,
  "session_col": "session_date",
  "timestamp_col": null,
  "isolate_groups": false,
  "test_period": null,
  "test_start": null,
  "test_end": null,
  "fold_direction": "forward",
  "calendar_id": null
}
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_splits` | int | 5 | Number of CV folds |
| `test_size` | int/float/str/null | null | Validation fold size (see below) |
| `train_size` | int/float/str/null | null | Training set size (null = expanding) |
| `step_size` | int/null | null | Step between splits (null = test_size) |
| `label_horizon` | int/str | 0 | Forward-looking label window |
| `embargo_td` | int/str/null | null | Post-test embargo buffer |
| `align_to_sessions` | bool | false | Align to trading sessions |
| `session_col` | str | "session_date" | Session column name |
| `timestamp_col` | str/null | null | Timestamp column (Polars) |
| `isolate_groups` | bool | false | Prevent group leakage |
| `test_period` | int/str/null | null | Held-out test period (see below) |
| `test_start` | date/str/null | null | Explicit held-out test start date |
| `test_end` | date/str/null | null | Explicit held-out test end date |
| `fold_direction` | "forward"/"backward" | "forward" | Direction of validation folds |
| `calendar_id` | str/null | null | Trading calendar for trading-day gaps |

### Held-Out Test Configuration

The held-out test feature allows you to reserve the most recent data for final
evaluation, while using earlier data for cross-validation.

**Two ways to specify held-out test:**

1. **`test_period`**: Reserve the most recent N days/samples
   - String: `"52D"` reserves last 52 calendar days
   - Integer: `52` with `calendar_id` reserves last 52 trading days
   - Integer: `52` without calendar reserves last 52 samples

2. **`test_start`/`test_end`**: Explicit date range
   - `test_start="2024-03-01"` starts held-out test on March 1
   - `test_end` defaults to end of data if not specified

**Note**: `test_period` and `test_start` are mutually exclusive.

### Fold Direction

With a held-out test period, you can choose how validation folds are generated:

- **`"forward"`** (default): Traditional walk-forward, stopping before held-out test
- **`"backward"`**: Folds step backward from the held-out test boundary

**Backward direction is recommended** for proper chronological validation:

```
[train1][val1][train2][val2][train3][val3] | [HELD-OUT TEST]
        ←     ←     ←     ←     ←     ←     test_start
```

### Trading Calendar for label_horizon

When `calendar_id` is set (e.g., `"NYSE"`, `"CME_Equity"`), integer `label_horizon`
values are interpreted as **trading days** instead of calendar days.

```json
{
  "label_horizon": 5,
  "calendar_id": "NYSE"
}
```

This correctly handles weekends and holidays when calculating the purge window.

### Size Specifications

`test_size` and `train_size` accept multiple formats:

| Format | Example | Meaning |
|--------|---------|---------|
| Integer | `100` | 100 samples (or sessions if aligned) |
| Float | `0.1` | 10% of dataset |
| String | `"4W"` | 4 weeks (pandas offset alias) |
| null | `null` | Auto-calculate |

**Time-based strings** use pandas offset aliases: `"1D"`, `"4W"`, `"3M"`, `"1Y"`, etc.

**Note**: Time-based strings are NOT compatible with `align_to_sessions=true`.

### Examples

**Basic walk-forward**:
```json
{
  "n_splits": 5,
  "test_size": 252
}
```

**Fixed rolling window** (train never expands):
```json
{
  "n_splits": 10,
  "test_size": 63,
  "train_size": 252
}
```

**Time-based splits**:
```json
{
  "n_splits": 12,
  "test_size": "1M",
  "train_size": "12M"
}
```

**With label purging** (5-day forward returns):
```json
{
  "n_splits": 5,
  "test_size": 63,
  "label_horizon": 5
}
```

**Session-aligned** (for intraday data):
```json
{
  "n_splits": 5,
  "test_size": 20,
  "align_to_sessions": true,
  "session_col": "session_date"
}
```

**Held-out test with backward validation** (recommended for production):
```json
{
  "n_splits": 5,
  "test_period": "52D",
  "test_size": 20,
  "train_size": 252,
  "label_horizon": 5,
  "fold_direction": "backward",
  "calendar_id": "NYSE"
}
```

**Held-out test with explicit dates**:
```json
{
  "n_splits": 3,
  "test_start": "2024-10-01",
  "test_end": "2024-12-31",
  "test_size": 20,
  "fold_direction": "backward"
}
```

**Trading-day-aware purging**:
```json
{
  "n_splits": 5,
  "test_size": 63,
  "label_horizon": 5,
  "calendar_id": "NYSE"
}
```

---

## CombinatorialConfig

Combinatorial Purged Cross-Validation (CPCV) for multi-asset strategies.

Reference: Bailey & López de Prado (2014) - "The Deflated Sharpe Ratio"

### JSON Schema

```json
{
  "n_splits": 28,
  "n_groups": 8,
  "n_test_groups": 2,
  "max_combinations": null,
  "contiguous_test_blocks": false,
  "label_horizon": 0,
  "embargo_td": null,
  "embargo_pct": null,
  "align_to_sessions": false,
  "session_col": "session_date",
  "timestamp_col": null,
  "isolate_groups": true,
  "random_state": null
}
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_splits` | int | C(n,k) | Auto-calculated from n_groups/n_test_groups |
| `n_groups` | int | 8 | Timeline partitions (typically 8-12) |
| `n_test_groups` | int | 2 | Groups per test set (typically 2-3) |
| `max_combinations` | int/null | null | Limit folds (random sample if exceeded) |
| `contiguous_test_blocks` | bool | false | Only contiguous test groups |
| `embargo_pct` | float/null | null | Embargo as % of samples |
| `random_state` | int/null | null | Seed for reproducible sampling |
| `isolate_groups` | bool | **true** | Prevent group leakage (default ON for CPCV) |

**Note**: `embargo_td` and `embargo_pct` are mutually exclusive.

### Examples

**Standard CPCV** (28 folds = C(8,2)):
```json
{
  "n_groups": 8,
  "n_test_groups": 2
}
```

**Reduced overfitting** (contiguous only, fewer folds):
```json
{
  "n_groups": 10,
  "n_test_groups": 2,
  "contiguous_test_blocks": true
}
```

**With embargo** (2% buffer):
```json
{
  "n_groups": 8,
  "n_test_groups": 2,
  "embargo_pct": 0.02
}
```

**Limited folds** (reproducible subset):
```json
{
  "n_groups": 12,
  "n_test_groups": 3,
  "max_combinations": 50,
  "random_state": 42
}
```

---

## CalendarConfig

Trading calendar configuration for session-aware splitting.

### JSON Schema

```json
{
  "exchange": "CME_Equity",
  "timezone": "America/Chicago",
  "localize_naive": true
}
```

### Preset Configurations

| Preset | Exchange | Timezone |
|--------|----------|----------|
| `CME_CONFIG` | CME_Equity | America/Chicago |
| `NYSE_CONFIG` | NYSE | America/New_York |
| `NASDAQ_CONFIG` | NASDAQ | America/New_York |
| `LSE_CONFIG` | LSE | Europe/London |

### Usage with WalkForwardCV

```python
from ml4t.diagnostic.splitters import WalkForwardCV
from ml4t.diagnostic.splitters.calendar_config import CME_CONFIG

cv = WalkForwardCV(
    n_splits=5,
    test_size="4W",
    train_size="52W",
    calendar=CME_CONFIG
)
```

---

## Fold Persistence

Save and load computed folds for reproducibility.

### Fold File Format (v1.0)

```json
{
  "version": "1.0",
  "n_folds": 5,
  "n_samples": 10000,
  "timestamps": ["2020-01-02", "2020-01-03", ...],
  "metadata": {
    "strategy": "walk_forward",
    "config": {...}
  },
  "folds": [
    {
      "fold_id": 0,
      "train_indices": [0, 1, 2, ...],
      "test_indices": [1000, 1001, ...],
      "train_size": 1000,
      "test_size": 200,
      "train_start": "2020-01-02",
      "train_end": "2021-06-30",
      "test_start": "2021-07-01",
      "test_end": "2021-09-30"
    },
    ...
  ]
}
```

### Saving Folds

```python
from ml4t.diagnostic.splitters import (
    WalkForwardCV,
    WalkForwardConfig,
    save_folds,
    save_config
)

# Create and compute folds
config = WalkForwardConfig(n_splits=5, test_size=252)
cv = WalkForwardCV.from_config(config)
folds = list(cv.split(X))

# Save folds with metadata
save_folds(
    folds,
    X,
    "experiment_folds.json",
    metadata={
        "experiment": "factor_selection_v3",
        "config": config.to_dict()
    }
)

# Also save config separately
save_config(config, "cv_config.json")
```

### Loading Folds

```python
from ml4t.diagnostic.splitters import (
    load_folds,
    load_config,
    WalkForwardConfig,
    verify_folds
)

# Load saved folds
folds, metadata = load_folds("experiment_folds.json")

# Verify integrity
stats = verify_folds(folds, n_samples=len(X))
if not stats["valid"]:
    raise ValueError(f"Fold validation failed: {stats['errors']}")

# Use folds directly
for train_idx, test_idx in folds:
    X_train, X_test = X[train_idx], X[test_idx]
    # ... train and evaluate

# Or reload config
config = load_config("cv_config.json", WalkForwardConfig)
```

### Verification Stats

`verify_folds()` returns:

```python
{
    "valid": True,                  # Overall validity
    "errors": [],                   # List of error messages
    "n_folds": 5,
    "n_samples": 10000,
    "train_sizes": [1000, 1200, ...],
    "test_sizes": [200, 200, ...],
    "coverage": 0.95,               # Total sample coverage
    "train_coverage": 0.85,
    "test_coverage": 0.40,
    "avg_train_size": 1500.0,
    "std_train_size": 200.0,
    "avg_test_size": 200.0,
    "std_test_size": 0.0
}
```

---

## Complete Workflow Example

### 1. Define and Save Configuration

```python
from ml4t.diagnostic.splitters import WalkForwardConfig

# Create config
config = WalkForwardConfig(
    n_splits=5,
    test_size=63,           # Quarterly test sets
    train_size=252,         # 1 year training window
    label_horizon=5,        # 5-day forward returns
    isolate_groups=True     # Multi-asset validation
)

# Validate
errors = config.validate_fully()
if errors:
    raise ValueError(f"Invalid config: {errors}")

# Save to JSON
config.to_json("experiments/factor_v3/cv_config.json")

# Or YAML
config.to_yaml("experiments/factor_v3/cv_config.yaml")
```

### 2. Load and Use Configuration

```python
from ml4t.diagnostic.splitters import (
    WalkForwardConfig,
    WalkForwardCV,
    save_folds
)

# Load config (auto-detect format)
config = WalkForwardConfig.from_file("experiments/factor_v3/cv_config.json")

# Create splitter
cv = WalkForwardCV.from_config(config)

# Generate and save folds
folds = list(cv.split(X, groups=df["symbol"]))
save_folds(
    folds, X,
    "experiments/factor_v3/folds.json",
    metadata={"config": config.to_dict()}
)
```

### 3. Reproduce Experiment

```python
from ml4t.diagnostic.splitters import load_folds, verify_folds

# Load exact folds from previous run
folds, metadata = load_folds("experiments/factor_v3/folds.json")

# Verify against current data
stats = verify_folds(folds, n_samples=len(X))
assert stats["valid"], f"Fold mismatch: {stats['errors']}"

# Run experiment with identical splits
for fold_idx, (train_idx, test_idx) in enumerate(folds):
    print(f"Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
    # ... exact reproduction
```

---

## Label Horizon and Embargo Explained

### Label Horizon (Purging)

Removes training samples whose labels overlap with test data.

```
Timeline: [---training---][test]

With 5-day forward returns:
Sample at day 95 predicts day 100 return
If test starts at day 98, sample 95 "sees" test data

label_horizon=5 removes samples 93-97 from training
```

### Embargo (Post-Test Buffer)

Removes training samples immediately after test periods.
**Only affects CombinatorialCV** where training can follow test.

```
CPCV with test groups 3-4:
[train][train][TEST][TEST][train][train]

Samples after TEST have autocorrelated features.
embargo_td=10 removes 10 samples after each test block.
```

---

## API Reference

### Config Serialization Methods

All config classes inherit these methods from `BaseConfig`:

| Method | Description |
|--------|-------------|
| `to_json(path)` | Save as JSON |
| `to_yaml(path)` | Save as YAML |
| `to_dict()` | Convert to dictionary |
| `from_json(path)` | Load from JSON |
| `from_yaml(path)` | Load from YAML |
| `from_dict(data)` | Create from dictionary |
| `from_file(path)` | Auto-detect and load |
| `validate_fully()` | Return list of validation errors |
| `diff(other)` | Compare two configs |

### Persistence Functions

| Function | Description |
|----------|-------------|
| `save_folds(folds, X, path, metadata)` | Save fold indices |
| `load_folds(path)` | Load folds and metadata |
| `save_config(config, path)` | Save config (wrapper) |
| `load_config(path, cls)` | Load config (wrapper) |
| `verify_folds(folds, n_samples)` | Validate fold integrity |
