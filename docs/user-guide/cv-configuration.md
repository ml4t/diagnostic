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
configuration values. Parsing legacy aliases leaves the supplied mapping
unchanged, including when validation rejects the configuration.

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
held-out walk-forward folds and ordinary CPCV row groups retain positional
purge/embargo boundaries, including duplicate timestamps.
An explicit `pd.Timedelta` still means elapsed time. Pure row-count splitting
does not need timezone-aware timestamps; elapsed-time calculations retain their
timezone checks. With a selected calendar, the existing session-aware
walk-forward path remains active.

The low-level `calculate_embargo_indices` helper also accepts an already
resolved `TradingCalendar` or `None`. Without a calendar, integer embargoes
and percentages count observations, including across missing dates. An
elapsed purge does not convert a separate integer embargo into days; an
elapsed embargo likewise does not change an integer purge's row units.
Explicit durations retain timezone validation. With a resolved calendar,
integer embargoes use the same trading-day conversion as purging.
Named-calendar percentage buffers retain their legacy elapsed-span convention;
this change does not establish the broader session/expected-bar unit contract.

These Diagnostic calendar-intake changes are unreleased after 0.1.9. They
require published `ml4t-specs>=0.2.0`, which provides the v2 interface; Specs
0.1.5 does not. See the
[shared time contract](https://github.com/ml4t/specs/blob/e1a05a8065c1988a6bac414e50f8f0474e3b6ad4/docs/contracts.md).
This does not establish schedule-grid, expected-bar or elapsed-unit conformance
for every Diagnostic workflow. Backtest metadata adoption and full CPCV
session/expected-bar window semantics remain separate parts of the audit.

## Pass calendar intent to CPCV

Unreleased after 0.1.9, `CombinatorialCV` and `CombinatorialConfig` accept the
same calendar declarations and `feed_spec` precedence. Explicit `None`
overrides named feed metadata without changing caller-owned configuration:

```python
import pandas as pd
from ml4t.specs import FeedSpec

from ml4t.diagnostic.splitters import CombinatorialConfig, CombinatorialCV

cpcv_config = CombinatorialConfig(
    n_groups=3,
    n_test_groups=1,
    label_horizon=1,
    isolate_groups=False,
    calendar=None,
)
cpcv = CombinatorialCV(
    config=cpcv_config,
    feed_spec=FeedSpec(time_contract_version=2, calendar="NYSE"),
)
cpcv_features = pd.DataFrame(
    {"feature": range(6)},
    index=pd.to_datetime(["2025-07-03"] * 3 + ["2025-07-07"] * 3, utc=True),
)
cpcv_folds = [(train.tolist(), test.tolist()) for train, test in cpcv.split(cpcv_features)]
assert cpcv.calendar is None
assert cpcv_config.to_dict(mode="json")["calendar"] is None
assert cpcv_folds == [
    ([2, 3, 4, 5], [0, 1]),
    ([0, 4, 5], [2, 3]),
    ([0, 1, 2], [4, 5]),
]
```

Omitted declarations stay omitted through runtime configuration and JSON/YAML/copy
round trips, so a reused configuration can inherit another feed's calendar.
`calendar_id` remains a configuration alias; conflicting aliases fail before
mutation. Inspect `cpcv.calendar` for the resolved provider, not an undeclared
configuration's default value.

CPCV retains its existing integer observation counts and elapsed `Timedelta`
buffers, even when a named calendar is selected. This intake does not introduce
schedule filtering, infer missing sessions/bars, or turn its integer horizons
into session counts. `align_to_sessions=True` still groups the supplied
`session_col` labels. Full authoritative schedule/unit adoption remains separate.

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
`calendar` override in either splitter, which leaves the caller's configuration
unchanged. `feed_spec` supplies metadata rather than another splitter config.
