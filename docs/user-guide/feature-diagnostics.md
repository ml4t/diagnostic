# Feature Diagnostics

Analyze feature quality, importance, and interactions before modeling.

## Quick Start

```python
from ml4t.diagnostic.evaluation import FeatureDiagnostics
from ml4t.diagnostic.config import DiagnosticConfig

config = DiagnosticConfig.for_research()
fd = FeatureDiagnostics(config=config)
result = fd.analyze(features_df, target, dates)
```

## Information Coefficient (IC)

Measure predictive power via rank correlation:

```python
from ml4t.diagnostic.evaluation.metrics import compute_ic_series

ic_result = compute_ic_series(
    predictions=pred_df,          # date, symbol, prediction
    returns=ret_df,               # date, symbol, forward_return
    pred_col="prediction",
    ret_col="forward_return",
    date_col="date",
    entity_col="symbol",
    method="spearman",
)

print(ic_result.head())
```

### IC Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| IC Mean | mean(IC) | Average predictive power |
| IC Std | std(IC) | Consistency |
| IC IR | mean/std | Risk-adjusted IC |
| IC t-stat | mean / (std/√n) | Statistical significance |

## Feature Importance

Seven methods with consensus ranking:

### Mean Decrease Impurity (MDI)

```python
from ml4t.diagnostic.evaluation.metrics import compute_mdi_importance

importance = compute_mdi_importance(
    model=trained_tree_model,
    feature_names=feature_names
)
```

### Permutation Feature Importance (PFI)

```python
from ml4t.diagnostic.evaluation.metrics import compute_permutation_importance

importance = compute_permutation_importance(
    model=model,
    X=X_test,
    y=y_test,
    n_repeats=10
)
```

### SHAP Importance

```python
from ml4t.diagnostic.evaluation.metrics import compute_shap_importance

importance = compute_shap_importance(
    model=model,
    X=X_background,
    n_samples=100
)
```

### Consensus Ranking

Run a combined tear-sheet style comparison:

```python
from ml4t.diagnostic.evaluation.metrics import analyze_ml_importance

analysis = analyze_ml_importance(
    model=model,
    X=X_train,
    y=y_train,
    methods=["mdi", "pfi", "shap"],
)

print(analysis["top_features_consensus"])
```

## Feature Interactions

Detect non-linear interactions using H-statistic:

```python
from ml4t.diagnostic.evaluation.metrics import compute_h_statistic

h_stat = compute_h_statistic(
    model=model,
    X=X,
    features=['momentum', 'volatility']
)

print(f"Interaction strength: {h_stat:.3f}")
# > 0.1 indicates meaningful interaction
```

## Stationarity Tests

Ensure features are stationary:

```python
from ml4t.diagnostic.evaluation import analyze_stationarity

result = analyze_stationarity(
    series=feature_series,
    tests=['adf', 'kpss', 'pp']
)

print(f"ADF p-value: {result.adf_pvalue:.4f}")
print(f"Is stationary: {result.is_stationary}")
```

## Distribution Analysis

Check for heavy tails and normality:

```python
from ml4t.diagnostic.evaluation.distribution import analyze_distribution

result = analyze_distribution(feature_series)

print(f"Skewness: {result.moments_result.skewness:.2f}")
print(f"Excess Kurtosis: {result.moments_result.excess_kurtosis:.2f}")  # Fisher convention (normal=0)
print(f"Jarque-Bera p-value: {result.jarque_bera_result.p_value:.4f}")
print(f"Is normal: {result.is_normal}")
print(f"Recommendation: {result.recommended_distribution}")
```

## Drift Detection

Monitor feature distribution changes:

```python
from ml4t.diagnostic.evaluation.drift import analyze_drift

result = analyze_drift(
    train_features=X_train,
    test_features=X_test
)

print(f"PSI: {result.psi:.4f}")
# > 0.25 indicates significant drift
```

## Complete Workflow

```python
from ml4t.diagnostic.evaluation import FeatureDiagnostics

fd = FeatureDiagnostics()
result = fd.analyze(features_df, target, dates)

# Review all diagnostics
print(result.summary())

# Get warnings
for warning in result.warnings:
    print(f"⚠️ {warning}")

# Export report
result.to_html("feature_diagnostics.html")
```
