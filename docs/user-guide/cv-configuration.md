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
    calendar=None,
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

`label_horizon` accepts an integer number of observations or a fixed duration
such as `5D` or `1W`. Monthly research configurations may use `1M` or the ISO
form `P1M`; each month is deliberately normalized to 30 calendar days because
purging requires a fixed duration. Use `30D` directly when you want that
approximation to be explicit.

## Select the walk-forward calendar

Use `calendar` in `WalkForwardConfig` and `WalkForwardCV`. `calendar_id` remains
a supported configuration input/property alias, but serialized output uses
`calendar`. Supplying both names with conflicting values raises an error.
An undeclared calendar stays absent on configuration serialization; explicit
`None` stays explicit, including when other null fields are excluded.
This also holds for `cv.config` after construction and for JSON/YAML/copy
round trips. Inspect `cv.calendar_id` for the resolved calendar; an undeclared
configuration can inherit different metadata when reused with another feed.

Walk-forward selection uses an explicit `calendar` argument first, then an
explicit configuration declaration, then the supplied `FeedSpec` metadata,
then `NYSE`. Explicit `None` disables calendar filtering. A named `24/7`
calendar still uses a schedule; it does not mean observation order.

```python
from ml4t.specs import FeedSpec

feed = FeedSpec(time_contract_version=2, calendar="24/7")
assert WalkForwardCV(feed_spec=feed).calendar.config.exchange == "24/7"
assert WalkForwardCV(feed_spec=feed, calendar=None).calendar is None
assert WalkForwardCV(
    feed_spec=FeedSpec(time_contract_version=2, calendar=None)
).calendar is None
```

Without a calendar, integer label horizons and embargo sizes in a row-mode
purge request count observed rows, even across weekends, holidays or missing
dates. Integer held-out sizes also count rows. Standard and forward/backward
held-out folds retain positional boundaries, including duplicate timestamps.
An explicit `pd.Timedelta` still means elapsed time. Pure row-count splitting
does not need timezone-aware timestamps; elapsed-time calculations retain their
timezone checks. With a selected calendar, the existing session-aware
walk-forward path remains active. Mixed-unit requests remain part of the
broader unit audit described below.

These are unreleased walk-forward changes. Qualification pins merged Specs
source `701bde68744035050ede386efce498234c661c21`, not the published Specs
0.1.5 package, which lacks the v2 interface. A published dependency must replace
that source pin before release. See the
[shared time contract](https://github.com/ml4t/specs/blob/701bde68744035050ede386efce498234c661c21/docs/contracts.md).
This does not establish schedule-grid, expected-bar or elapsed-unit conformance
for every Diagnostic workflow. Combinatorial and Backtest metadata adoption
remain separate parts of the calendar audit.

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
constructor rejects conflicting sources of settings, except the explicit
walk-forward `calendar` override, which leaves the caller's configuration
unchanged. `feed_spec` supplies metadata rather than another splitter config.
