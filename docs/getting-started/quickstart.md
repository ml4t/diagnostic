# Quickstart

Get started with ML4T Diagnostic in 5 minutes.

## Basic Usage

### 1. Validated Cross-Validation

The simplest way to validate a strategy with proper statistical testing:

```python
from ml4t.diagnostic import ValidatedCrossValidation

# Create validator with CPCV and DSR
vcv = ValidatedCrossValidation(
    n_splits=10,
    embargo_pct=0.01,  # 1% embargo between folds
    purge_pct=0.05     # 5% purging around test set
)

# Fit and validate in one step
result = vcv.fit_validate(model, X, y, times)

# Check significance
if result.is_significant:
    print(f"Strategy is statistically significant!")
    print(f"  Sharpe Ratio: {result.sharpe:.2f}")
    print(f"  DSR p-value: {result.dsr_pvalue:.4f}")
else:
    print(f"Strategy may be overfit (p={result.dsr_pvalue:.4f})")
```

### 2. Signal Analysis

Analyze factor/signal quality (replaces Alphalens):

```python
from ml4t.diagnostic import analyze_signal

# Analyze signal predictive power
result = analyze_signal(
    factor=signal_series,           # Your alpha signal
    forward_returns=returns_df,     # Forward returns (1d, 5d, 10d)
    group_by='sector'               # Optional grouping
)

# Key metrics
print(f"IC Mean: {result.ic_mean:.4f}")
print(f"IC IR: {result.ic_ir:.2f}")
print(f"IC t-stat: {result.ic_tstat:.2f}")
```

### 3. Feature Diagnostics

Analyze feature importance and interactions:

```python
from ml4t.diagnostic.evaluation import FeatureDiagnostics
from ml4t.diagnostic.config import DiagnosticConfig

# Configure analysis
config = DiagnosticConfig.for_research()

# Run diagnostics
fd = FeatureDiagnostics(config=config)
result = fd.analyze(features_df, target, dates)

# View importance ranking
print(result.importance_ranking)

# Check for problematic interactions
print(result.interaction_warnings)
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
