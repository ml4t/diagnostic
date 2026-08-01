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

## Configuration classes

| Splitter | Configuration |
|----------|---------------|
| `WalkForwardCV` | `WalkForwardConfig` |
| `CombinatorialCV` | `CombinatorialConfig` |

Do not pass individual splitter parameters together with `config`. The
constructor rejects conflicting sources of settings.
