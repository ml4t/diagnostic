# Cross-Validation

ML4T Diagnostic provides time-series aware cross-validation methods that prevent data leakage through proper purging and embargo.

**See also**: [cv-configuration.md](cv-configuration.md) - JSON/YAML configuration format and fold persistence

## The Leakage Problem

Standard k-fold cross-validation leaks information in financial time series:

- **Temporal Leakage**: Future information leaks into training
- **Label Overlap**: Overlapping return windows contaminate folds
- **Autocorrelation**: Nearby samples are not independent

## Combinatorial Purged Cross-Validation (CPCV)

CPCV addresses these issues with purging and embargo:

```python
from ml4t.diagnostic.splitters import CombinatorialCV

cv = CombinatorialCV(
    n_groups=10,
    n_test_groups=2,      # Test on 2 groups at a time
    embargo_pct=0.01,     # 1% embargo after each test fold
    label_horizon=5       # Purging horizon around test boundaries
)

for train_idx, test_idx in cv.split(X, y, times):
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_groups` | Number of groups to split data into | 5-10 |
| `n_test_groups` | Groups in test set per iteration | 1-2 |
| `embargo_pct` | % of data to exclude after test | 0.01-0.02 |
| `label_horizon` | Samples/days to purge before test | 1-20 |

### How Purging Works

```
     Train           Purge    Test     Embargo    Train
├──────────────────┤ ░░░░░ ├────────┤ ░░░░░░░ ├──────────┤
                    ↑                          ↑
              Remove samples             Remove samples
              overlapping with           that could leak
              test labels                into next fold
```

## Purged Walk-Forward Cross-Validation

Rolling window approach with purging:

```python
from ml4t.diagnostic.splitters import WalkForwardCV

cv = WalkForwardCV(
    n_splits=5,
    train_size=252,       # 1 year training
    test_size=63,         # 1 quarter test
    embargo_pct=0.01
)

for train_idx, test_idx in cv.split(X, y, times):
    # Walk-forward validation
    pass
```

### Walk-Forward Diagram

```
Split 1: [====Train====][Test]
Split 2:     [====Train====][Test]
Split 3:         [====Train====][Test]
Split 4:             [====Train====][Test]
```

## Calendar-Aware and Group-Isolated Splitting

Use `WalkForwardCV` with `calendar` and `isolate_groups` options:

```python
from ml4t.diagnostic.splitters import WalkForwardCV

cv = WalkForwardCV(
    n_splits=5,
    train_size=252,
    test_size=63,
    calendar="NYSE",
    isolate_groups=True,
)
```

## Validated Cross-Validation

Combine CPCV with DSR validation in one step:

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.api import ValidatedCrossValidationConfig

config = ValidatedCrossValidationConfig(n_groups=10, n_test_groups=2, embargo_pct=0.01)
vcv = ValidatedCrossValidation(config=config)

result = vcv.fit_evaluate(X, y, model, times=times)

# Returns include both CV metrics AND DSR statistics
print(f"Mean Sharpe: {result.mean_sharpe:.4f}")
print(f"DSR: {result.dsr:.4f}")
print(f"Is Significant: {result.is_significant}")
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7
- Bailey et al. (2017). "The Probability of Backtest Overfitting"
