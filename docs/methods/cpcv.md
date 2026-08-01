# Combinatorial Purged Cross-Validation

Combinatorial purged cross-validation (CPCV) divides an ordered sample into
contiguous groups and evaluates every selected combination of test groups.
Purging removes training labels that overlap a test period. An embargo can
remove training observations immediately after a test period.

## Run CPCV

```python
import math

import numpy as np

from ml4t.diagnostic.splitters import CombinatorialCV

features = np.arange(240, dtype=float).reshape(120, 2)
cv = CombinatorialCV(
    n_groups=6,
    n_test_groups=2,
    label_horizon=5,
    embargo_size=2,
    isolate_groups=False,
)

splits = list(cv.split(features))
assert len(splits) == math.comb(6, 2)
assert all(len(np.intersect1d(train, test)) == 0 for train, test in splits)

first_train, first_test = splits[0]
print(f"Combinations: {len(splits)}")
print(f"First split: {len(first_train)} train, {len(first_test)} test")
```

`n_groups=6` and `n_test_groups=2` produce 15 combinations. Set
`max_combinations` and `random_state` when the full combination count is too
large.

## Choose leakage controls

- Set `label_horizon` to the number of observations used by each forward label.
- Set either `embargo_size` or `embargo_pct`, not both.
- Pass ordered data. CPCV partitions rows in their existing order.
- For panel data, pass asset identifiers through `groups` to apply purging per asset.
- Set `isolate_groups=True` only when train and test sets must contain different assets.

Use [cross-validation](../user-guide/cross-validation.md) for a comparison with
walk-forward validation. See the [API reference](../api/index.md) for exact
constructor parameters.
