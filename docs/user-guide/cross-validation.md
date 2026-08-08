# Cross-Validation

Use `WalkForwardCV` for chronological model evaluation. Use `CombinatorialCV`
when you need multiple backtest paths and a distribution of out-of-sample
results.

## Run purged walk-forward validation

```python
import numpy as np

from ml4t.diagnostic.splitters import WalkForwardCV

features = np.arange(600, dtype=float).reshape(300, 2)
cv = WalkForwardCV(
    n_splits=4,
    train_size=120,
    test_size=40,
    label_horizon=5,
    expanding=False,
)

walk_forward_splits = list(cv.split(features))
assert len(walk_forward_splits) == 4
assert all(train.max() < test.min() for train, test in walk_forward_splits)
assert all(test.min() - train.max() > cv.label_horizon for train, test in walk_forward_splits)

for fold, (train, test) in enumerate(walk_forward_splits, start=1):
    print(f"Fold {fold}: {len(train)} train, {len(test)} test")
```

`label_horizon=5` removes training observations whose five-period forward
labels would overlap the test set.

## Run combinatorial purged validation

```python
import math

import numpy as np

from ml4t.diagnostic.splitters import CombinatorialCV

features = np.arange(600, dtype=float).reshape(300, 2)
cpcv = CombinatorialCV(
    n_groups=6,
    n_test_groups=2,
    label_horizon=5,
    embargo_size=2,
    isolate_groups=False,
)

combinatorial_splits = list(cpcv.split(features))
assert len(combinatorial_splits) == math.comb(6, 2)
assert all(
    len(np.intersect1d(train, test)) == 0
    for train, test in combinatorial_splits
)
assert all(
    not any(
        np.intersect1d(np.arange(index + 1, index + cpcv.label_horizon + 1), test).size
        for index in train
    )
    for train, test in combinatorial_splits
)
print(f"CPCV combinations: {len(combinatorial_splits)}")
```

## Combine CPCV results with DSR

`ValidatedCrossValidation` summarizes fold Sharpe ratios and corrects the best
observed result for the number of trials.

```python
from ml4t.diagnostic import ValidatedCrossValidation

validation = ValidatedCrossValidation()
validation_result = validation.evaluate_sharpes([0.42, 0.51, 0.37, 0.48, 0.45])

assert validation_result.n_folds == 5
print(validation_result.summary())
```

## Select a splitter

| Requirement | Splitter |
|-------------|----------|
| Train only on observations before each test fold | `WalkForwardCV` |
| Measure performance across many test-group combinations | `CombinatorialCV` |
| Preserve a final untouched period | `WalkForwardCV` with `test_period` or `test_start` |
| Bound CPCV computation | `CombinatorialCV` with `max_combinations` |

Serialize splitter settings with the [CV configuration guide](cv-configuration.md).
The [CPCV method page](../methods/cpcv.md) explains the group and combination
parameters.
