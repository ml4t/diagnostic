# Cross-Validation

ML4T Diagnostic provides time-series aware cross-validation methods that prevent data leakage through proper purging and embargo.

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
    n_splits=10,
    n_test_splits=2,      # Test on 2 groups at a time
    embargo_pct=0.01,     # 1% embargo after each test fold
    purge_pct=0.05        # 5% purging around test boundaries
)

for train_idx, test_idx in cv.split(X, y, times):
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_splits` | Number of groups to split data into | 5-10 |
| `n_test_splits` | Groups in test set per iteration | 1-2 |
| `embargo_pct` | % of data to exclude after test | 0.01-0.02 |
| `purge_pct` | % of data to exclude around test | 0.01-0.10 |

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
    train_period=252,     # 1 year training
    test_period=63,       # 1 quarter test
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

## Calendar-Aware Splitting

Respect trading calendar and market holidays:

```python
from ml4t.diagnostic.splitters import CalendarSplitter

cv = CalendarSplitter(
    calendar='NYSE',
    train_years=3,
    test_months=3
)
```

## Group Isolation

Prevent leakage when multiple assets share features:

```python
from ml4t.diagnostic.splitters import GroupIsolationSplitter

cv = GroupIsolationSplitter(
    groups=ticker_series,  # Ensure same ticker not in train AND test
    n_splits=5
)
```

## Validated Cross-Validation

Combine CPCV with DSR validation in one step:

```python
from ml4t.diagnostic import ValidatedCrossValidation

vcv = ValidatedCrossValidation(
    n_splits=10,
    embargo_pct=0.01,
    dsr_significance=0.05
)

result = vcv.fit_validate(model, X, y, times)

# Returns include both CV metrics AND DSR statistics
print(f"Mean CV Score: {result.cv_score:.4f}")
print(f"DSR: {result.dsr:.4f}")
print(f"Is Significant: {result.is_significant}")
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7
- Bailey et al. (2017). "The Probability of Backtest Overfitting"
