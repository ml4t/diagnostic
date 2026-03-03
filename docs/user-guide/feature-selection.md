# Feature Selection

Reduce a large set of candidate features to a focused, production-ready
set using systematic filtering.

---

## Quick Start

> **I have** feature-outcome analysis results (IC, importance, drift).
> **I want** to select the best features for my ML model.

```python
from ml4t.diagnostic.selection import (
    FeatureSelector,
    FeatureOutcomeResult,
    FeatureICResults,
    FeatureImportanceResults,
)

# Build outcome results from your analysis
outcome = FeatureOutcomeResult(
    features=feature_names,
    ic_results=ic_results,           # dict[str, FeatureICResults]
    importance_results=imp_results,  # dict[str, FeatureImportanceResults]
    drift_results=drift_results,     # from analyze_drift() — optional
)

# Create selector with correlation matrix
selector = FeatureSelector(outcome, correlation_matrix=corr_df)

# Run pipeline
selector.run_pipeline([
    ("drift", {"threshold": 0.2, "method": "psi"}),
    ("ic", {"threshold": 0.02}),
    ("correlation", {"threshold": 0.8}),
    ("importance", {"threshold": 0.01, "method": "mdi", "top_k": 20}),
])

selected = selector.get_selected_features()
print(selector.get_selection_report().summary())
```

---

## Pipeline Stages

The `FeatureSelector` provides four filtering methods. Apply them individually
or chain them with `run_pipeline()`.

### IC Filtering

Keep features with absolute Information Coefficient above a threshold.
IC measures the Spearman rank correlation between a feature and forward returns.

```python
selector.filter_by_ic(
    threshold=0.02,   # minimum |IC|
    min_periods=20,   # minimum observations
    lag=5,            # specific forward lag (None = mean across lags)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | — | Minimum absolute IC to keep |
| `min_periods` | 1 | Minimum observation count |
| `lag` | None | Specific lag to filter on (None = mean IC) |

### Importance Filtering

Keep features by ML importance score. Supports MDI, permutation, and SHAP.

```python
# Threshold-based
selector.filter_by_importance(threshold=0.01, method="mdi")

# Top-K (keeps the K most important, ignoring threshold)
selector.filter_by_importance(threshold=0, method="shap", top_k=20)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | — | Minimum importance value |
| `method` | `"mdi"` | `"mdi"`, `"permutation"`, or `"shap"` |
| `top_k` | None | Keep only top K features |

### Correlation Filtering

Remove redundant features. When two features exceed the correlation threshold,
keep the one with higher IC (or importance, or alphabetical order).

```python
selector.filter_by_correlation(
    threshold=0.8,
    keep_strategy="higher_ic",  # "higher_ic", "higher_importance", or "first"
)
```

The correlation matrix can be:

- A Polars DataFrame with a `"feature"` index column
- A Polars DataFrame where column names are feature names (e.g. from `df.corr()`)

### Drift Filtering

Remove features with unstable distributions. Requires `drift_results` in
the outcome.

```python
# PSI-based: removes features with red alert (PSI >= 0.2)
selector.filter_by_drift(method="psi")

# Consensus-based: removes features where majority of methods detect drift
selector.filter_by_drift(threshold=0.5, method="consensus")
```

Drift results come from `analyze_drift()`:

```python
from ml4t.diagnostic.evaluation.drift import analyze_drift

drift = analyze_drift(train_df.to_pandas(), test_df.to_pandas(), methods=["psi"])
outcome = FeatureOutcomeResult(features=names, drift_results=drift, ...)
```

---

## Method Chaining

All filter methods return `self`, so you can chain them:

```python
selected = (
    FeatureSelector(outcome, corr_matrix)
    .filter_by_drift(method="psi")
    .filter_by_ic(threshold=0.02)
    .filter_by_correlation(threshold=0.8)
    .filter_by_importance(threshold=0.05, method="mdi")
    .get_selected_features()
)
```

---

## Selection Report

After filtering, generate a report showing each step:

```python
report = selector.get_selection_report()
print(report.summary())
```

Output:

```text
======================================================================
Feature Selection Report
======================================================================
Initial Features: 76
Final Features: 12
Removed: 64 (84.2%)

Selection Pipeline:
----------------------------------------------------------------------

Step 1: Drift Filtering: 76 → 74 (2 removed, 2.6%)
  Parameters: {'threshold': 0.2, 'method': 'psi'}
  Reasoning: Removed features with psi drift >= 0.2

Step 2: IC Filtering: 74 → 45 (29 removed, 39.2%)
  Parameters: {'threshold': 0.02, 'min_periods': 1, 'lag': None}
  Reasoning: Removed features with |IC| < 0.02

Step 3: Correlation Filtering: 45 → 32 (13 removed, 28.9%)
  Parameters: {'threshold': 0.8, 'keep_strategy': 'higher_ic'}
  Reasoning: Removed features with correlation > 0.8

Step 4: Importance Filtering (MDI): 32 → 12 (20 removed, 62.5%)
  Parameters: {'threshold': 0, 'method': 'mdi', 'top_k': 12}
  Reasoning: Kept top 12 features by mdi importance
======================================================================
```

---

## Building FeatureOutcomeResult

The `FeatureOutcomeResult` aggregates IC, importance, and drift analysis
into the interface that `FeatureSelector` consumes.

### From manual analysis

```python
from ml4t.diagnostic.selection.types import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcomeResult,
)

# After computing IC cross-sectionally
ic_results = {}
for feature in features:
    ic_results[feature] = FeatureICResults(
        feature=feature,
        ic_mean=mean_ic,
        ic_std=std_ic,
        ic_ir=mean_ic / std_ic,
        t_stat=t_stat,
        p_value=p_val,
        ic_by_lag={1: ic_1d, 5: ic_5d, 21: ic_21d},
        n_observations=n_obs,
    )

# After fitting a tree model
importance_results = {}
for i, feature in enumerate(features):
    importance_results[feature] = FeatureImportanceResults(
        feature=feature,
        mdi_importance=model.feature_importances_[i],
        permutation_importance=perm_imp[i],
        permutation_std=perm_std[i],
    )

outcome = FeatureOutcomeResult(
    features=features,
    ic_results=ic_results,
    importance_results=importance_results,
)
```

### With drift detection

```python
from ml4t.diagnostic.evaluation.drift import analyze_drift

drift = analyze_drift(
    reference=train_features.to_pandas(),
    test=test_features.to_pandas(),
    methods=["psi", "wasserstein"],
)

outcome = FeatureOutcomeResult(
    features=features,
    ic_results=ic_results,
    importance_results=importance_results,
    drift_results=drift,
)
```

---

## Reset and Re-run

```python
selector.reset()  # restores initial feature set, clears history

# Try a different pipeline
selector.run_pipeline([
    ("ic", {"threshold": 0.03}),
    ("importance", {"threshold": 0, "method": "shap", "top_k": 10}),
])
```

---

## API Reference

::: ml4t.diagnostic.selection.FeatureSelector
    options:
      show_root_heading: true
      members:
        - filter_by_ic
        - filter_by_importance
        - filter_by_correlation
        - filter_by_drift
        - run_pipeline
        - get_selected_features
        - get_removed_features
        - get_selection_report
        - reset

::: ml4t.diagnostic.selection.types.FeatureOutcomeResult
    options:
      show_root_heading: true

::: ml4t.diagnostic.selection.types.FeatureICResults
    options:
      show_root_heading: true

::: ml4t.diagnostic.selection.types.FeatureImportanceResults
    options:
      show_root_heading: true
