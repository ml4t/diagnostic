# Cross-Validation Configuration

Splitter configuration objects validate settings and serialize them for
reproducible runs. Pass the loaded configuration to the splitter constructor.

## Save and reload a walk-forward configuration

```python
from pathlib import Path

import numpy as np

from ml4t.diagnostic.splitters import WalkForwardCV
from ml4t.diagnostic.splitters.config import WalkForwardConfig

config = WalkForwardConfig(
    n_splits=4,
    train_size=120,
    test_size=40,
    label_horizon=5,
    calendar_id=None,
)

path = Path("walk_forward.json")
config.to_json(path)
reloaded = WalkForwardConfig.from_json(path)
cv = WalkForwardCV(config=reloaded)

features = np.arange(600, dtype=float).reshape(300, 2)
splits = list(cv.split(features))
assert len(splits) == 4
assert reloaded.model_dump() == config.model_dump()
print(path.read_text())
```

Use `to_yaml` and `from_yaml` for YAML. Both formats preserve the validated
configuration values.

## Persist generated folds

Persist the actual train/test indices when an audit or later model run must use
the same observations, not just the same splitter settings.

```python
from ml4t.diagnostic.splitters import load_folds, save_folds

fold_path = Path("walk_forward_folds.json")
save_folds(splits, features, fold_path, metadata={"dataset": "example-v1"})
loaded_folds, metadata = load_folds(fold_path)

assert metadata["dataset"] == "example-v1"
assert all(
    np.array_equal(saved_train, loaded_train)
    and np.array_equal(saved_test, loaded_test)
    for (saved_train, saved_test), (loaded_train, loaded_test) in zip(
        splits, loaded_folds, strict=True
    )
)
```

## Configuration classes

| Splitter | Configuration |
|----------|---------------|
| `WalkForwardCV` | `WalkForwardConfig` |
| `CombinatorialCV` | `CombinatorialConfig` |

Do not pass individual splitter parameters together with `config`. The
constructor rejects conflicting sources of settings.
