# Quickstart

Get started with ML4T Diagnostic in 5 minutes.

## Basic Usage

### 1. Validated Cross-Validation

The simplest way to validate a strategy with proper statistical testing:

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.api import ValidatedCrossValidationConfig

# Create validator with CPCV and DSR
config = ValidatedCrossValidationConfig(
    n_groups=10,
    n_test_groups=2,
    embargo_pct=0.01,
    label_horizon=5,
)
vcv = ValidatedCrossValidation(config=config)

# Fit and validate in one step
result = vcv.fit_evaluate(X, y, model, times=times)

# Check significance
if result.is_significant:
    print(f"Strategy is statistically significant!")
    print(f"  Mean Sharpe: {result.mean_sharpe:.2f}")
    print(f"  DSR probability: {result.dsr:.4f}")
else:
    print(f"Strategy may be overfit (DSR={result.dsr:.4f})")
```

### 2. Signal Analysis

Analyze factor/signal quality (replaces Alphalens):

```python
from ml4t.diagnostic import analyze_signal

# Analyze signal predictive power
result = analyze_signal(
    factor=factor_df,      # columns: date, asset, factor
    prices=prices_df,      # columns: date, asset, price
    periods=(1, 5, 21),
)

# Key metrics
print(f"IC Mean (1D): {result.ic['1D']:.4f}")
print(f"IC IR (1D): {result.ic_ir['1D']:.2f}")
print(f"IC t-stat (1D): {result.ic_t_stat['1D']:.2f}")
```

### 3. Feature Diagnostics

Analyze feature importance and interactions:

```python
from ml4t.diagnostic.evaluation import FeatureDiagnostics, FeatureDiagnosticsConfig

config = FeatureDiagnosticsConfig(run_stationarity=True, run_distribution=True)
fd = FeatureDiagnostics(config=config)
result = fd.run_diagnostics(features_df["feature_1"], name="feature_1")
print(result.summary())
```

### 4. Trade Error Analysis

Identify why trades fail using SHAP:

```python
from ml4t.diagnostic.evaluation import TradeAnalysis, TradeShapAnalyzer

# Find worst trades
analyzer = TradeAnalysis(trade_records)
worst_trades = analyzer.worst_trades(n=20)

# Explain with SHAP
shap_analyzer = TradeShapAnalyzer(model, features_df, shap_values)
result = shap_analyzer.explain_worst_trades(worst_trades)

# Get actionable insights
for pattern in result.error_patterns:
    print(f"Error Pattern: {pattern.hypothesis}")
    print(f"  Suggested Action: {pattern.actions}")
```

## Configuration Presets

Use presets for common scenarios:

```python
from ml4t.diagnostic.config import DiagnosticConfig

# Quick exploratory analysis
config = DiagnosticConfig.for_quick_analysis()

# Thorough research
config = DiagnosticConfig.for_research()

# Production validation
config = DiagnosticConfig.for_production()
```

## Next Steps

- [Cross-Validation Guide](../user-guide/cross-validation.md) - CPCV and walk-forward details
- [Statistical Tests](../user-guide/statistical-tests.md) - DSR, RAS, FDR explained
- [Examples](https://github.com/stefan-jansen/ml4t-diagnostic/tree/main/examples) - Jupyter notebooks
