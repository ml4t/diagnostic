# ML4T Diagnostic: Comprehensive Diagnostics for Quantitative Finance

**Statistical rigor meets actionable insights for ML trading strategies**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

---

## What is ML4T Diagnostic?

**ML4T Diagnostic** is a comprehensive evaluation library for quantitative trading strategies, spanning the entire ML workflow from feature analysis to portfolio performance.

### Key Improvements

| Capability | What's New |
|------------|------------|
| **Performance** | Polars-powered for 10-100x faster analysis |
| **Visualizations** | Interactive Plotly charts |
| **Insights** | Auto-interpretation with warnings |
| **Statistics** | DSR, CPCV, RAS, PBO, FDR corrections |
| **Exploratory** | Stationarity, ACF, volatility, distribution tests |
| **Signal Analysis** | Multi-signal comparison and selection |
| **Trade Diagnostics** | SHAP-based error pattern discovery |
| **Binary Metrics** | Precision, recall, lift, coverage with Wilson intervals |
| **Threshold Analysis** | Threshold sweep, optimization, monotonicity checks |

---

## Quick Start

### Installation

```bash
# Core library
pip install ml4t-diagnostic

# With ML dependencies (for SHAP, importance, interactions)
pip install ml4t-diagnostic[ml]

# With visualization (for interactive reports)
pip install ml4t-diagnostic[viz]

# Everything (ML + viz + dashboard)
pip install ml4t-diagnostic[all]
```

### Example 1: Trade Diagnostics

**Close the ML→Trading feedback loop**: Understand why specific trades fail and get actionable improvement suggestions.

```python
from ml4t.diagnostic.evaluation import TradeAnalysis, TradeShapAnalyzer
from ml4t.diagnostic.config import TradeShapConfig

# 1. Identify worst trades from backtest
analyzer = TradeAnalysis(trade_records)
worst_trades = analyzer.worst_trades(n=20)

# 2. Explain with SHAP
config = TradeShapConfig.for_quick_diagnostics()
shap_analyzer = TradeShapAnalyzer(
    model=trained_model,
    features_df=features_df,  # Features with timestamps
    shap_values=shap_values,   # Precomputed SHAP values
    config=config
)

# 3. Discover error patterns
result = shap_analyzer.explain_worst_trades(worst_trades)

# 4. Get actionable hypotheses
for pattern in result.error_patterns:
    print(f"Pattern {pattern.cluster_id}: {pattern.hypothesis}")
    print(f"  Actions: {pattern.actions}")
    print(f"  Confidence: {pattern.confidence:.2%}")
    print(f"  Potential savings: ${pattern.potential_impact:,.2f}")
```

**Output example**:
```
Pattern 1: High momentum + Low volatility → Reversals
  Actions: ['Add volatility regime filter', 'Shorten holding period in low vol']
  Confidence: 85%
  Potential savings: $12,450.00

Pattern 2: Low liquidity + Wide spreads → Poor execution
  Actions: ['Add minimum liquidity filter', 'Widen entry criteria']
  Confidence: 78%
  Potential savings: $8,230.00
```

See **[examples/trade_diagnostics_example.ipynb](examples/trade_diagnostics_example.ipynb)** for complete end-to-end workflow.

### Example 2: Feature Importance Analysis

```python
import polars as pl
from ml4t.diagnostic.evaluation import analyze_ml_importance

# Your data
X = pl.read_parquet("features.parquet")
y = pl.read_parquet("labels.parquet")

# Analyze feature importance (combines MDI, PFI, MDA, SHAP)
results = analyze_ml_importance(model, X, y)

# Get consensus ranking
print(results.consensus_ranking)
# [('momentum', 1.2), ('volatility', 2.1), ...]

# Check warnings
print(results.warnings)
# ["High SHAP importance but low PFI for 'spread' - possible overfitting"]

# Get interpretation
print(results.interpretation)
# "Strong consensus across methods. Top 3 features: momentum, volatility, volume..."
```

### Example 3: Feature Interactions

```python
from ml4t.diagnostic.evaluation import analyze_interactions

# Detect feature interactions (Conditional IC, H-stat, SHAP)
results = analyze_interactions(model, X, y)

# Top interactions by consensus
print(results.top_interactions_consensus)
# [('momentum', 'volatility'), ('volume', 'spread'), ...]

# Method agreement
print(results.method_agreement)
# {('h_statistic', 'shap'): 0.85, ...}  # High agreement = robust finding
```

### Example 4: Statistical Validation (DSR)

```python
from ml4t.diagnostic.evaluation import stats

# Your backtest results
returns = strategy.compute_returns()

# Statistical validation with multiple testing correction
dsr_result = stats.compute_dsr(
    returns=returns,
    benchmark_sr=0.0,
    n_trials=100,  # Number of strategies tested
    expected_max_sharpe=1.5
)

print(f"Sharpe Ratio: {dsr_result['sr']:.2f}")
print(f"Deflated Sharpe: {dsr_result['dsr']:.2f}")  # Accounts for multiple testing
print(f"p-value: {dsr_result['pval']:.4f}")
print(f"Significant: {dsr_result['is_significant']}")
```

### Example 5: Binary Classification Metrics

**Evaluate discrete trading signals** with proper statistical inference:

```python
from ml4t.diagnostic.evaluation import (
    binary_classification_report,
    precision, recall, lift, coverage, f1_score,
    wilson_score_interval,
    binomial_test_precision,
)

# Your signals and outcomes
signals = momentum > threshold  # Binary signals
labels = forward_returns > 0     # Binary outcomes (profitable or not)

# Comprehensive report with confidence intervals
report = binary_classification_report(signals, labels)

print(f"Precision: {report['precision']:.2%} ± {report['precision_ci_width']:.2%}")
print(f"Lift: {report['lift']:.2f}x (vs random)")
print(f"Coverage: {report['coverage']:.1%} of observations")
print(f"Statistically significant: {report['binomial_pvalue'] < 0.05}")
```

**Key metrics**:
- **Precision**: When you signal, how often are you right?
- **Lift**: How much better than random selection?
- **Coverage**: What fraction of time are you in a position?
- **Wilson interval**: Accurate confidence bounds for proportions

### Example 6: Threshold Optimization

**Find optimal signal thresholds** with train-only selection:

```python
from ml4t.diagnostic.evaluation import (
    evaluate_threshold_sweep,
    find_optimal_threshold,
    check_monotonicity,
)

# Sweep thresholds and compute metrics at each
results = evaluate_threshold_sweep(
    indicator=momentum_values,
    label=future_profitable,
    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    direction='above'
)

# Find optimal with constraints
optimal = find_optimal_threshold(
    indicator=momentum_values,
    label=future_profitable,
    metric="f1_score",
    min_coverage=0.02,           # At least 2% signal frequency
    require_significant=True     # Must pass binomial test
)

print(f"Optimal threshold: {optimal['threshold']:.2f}")
print(f"F1 Score: {optimal['f1_score']:.2%}")

# Check if relationship is monotonic (good) or non-monotonic (investigate)
mono = check_monotonicity(results, metric="precision")
print(f"Monotonicity score: {mono['score']:.2f}")
```

**Critical**: Use train-only threshold selection in cross-validation to prevent leakage.

---

## Library Overview

ML4T Diagnostic provides **three complementary capabilities** across **four application domains**:

### Three Pillars of Analysis

| Pillar | Purpose | Examples |
|--------|---------|----------|
| **Explore** | Understand patterns before modeling | Stationarity tests, ACF/PACF, distribution analysis |
| **Validate** | Test significance and prevent overfitting | DSR, CPCV, RAS, FDR corrections |
| **Visualize** | Communicate findings effectively | Interactive Plotly charts, dashboards, reports |

### Four Application Domains

| Domain | Stage | Key Classes |
|--------|-------|-------------|
| **Features & Data** | Pre-modeling | `FeatureDiagnostics`, `analyze_stationarity()` |
| **Signals & Models** | Modeling | `SignalAnalysis`, `MultiSignalAnalysis` |
| **Trades & Backtest** | Post-modeling | `TradeAnalysis`, `TradeShapAnalyzer` |
| **Portfolio** | Production | `PortfolioAnalysis`, rolling metrics |

This architecture ensures you can **explore, validate, and visualize** at every stage of the ML workflow.

---

## Architecture: Four-Tier Diagnostic Framework

ML4T Diagnostic covers **four tiers** of the quantitative workflow:

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1: Feature Analysis (Pre-Modeling)                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Time series diagnostics (stationarity, ACF, volatility)    │
│ • Distribution analysis (moments, normality, tails)          │
│ • Feature-outcome predictiveness (IC, MI, quantiles)         │
│ • Feature importance (MDI, PFI, MDA, SHAP consensus)         │
│ • Feature interactions (Conditional IC, H-stat, SHAP)        │
│ • Drift detection (PSI, domain classifier)                   │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ Tier 2: Signal Analysis (Model Outputs)                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • IC analysis (time series, histogram, heatmap)              │
│ • Quantile returns (bar, violin, cumulative)                 │
│ • Turnover analysis (top/bottom basket, autocorrelation)     │
│ • Multi-signal comparison and ranking                        │
│ • Signal selection framework                                 │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ Tier 3: Backtest Analysis (Post-Modeling)                   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Trade analysis (win/loss, PnL, holding periods)            │
│ • Statistical validity (DSR, RAS, PBO, FDR corrections)      │
│ • Trade-SHAP diagnostics (error pattern discovery)           │
│ • Excursion analysis (TP/SL parameter optimization)          │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ Tier 4: Portfolio Analysis (Production)                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Performance metrics (Sharpe, Sortino, Calmar, Omega)       │
│ • Drawdown analysis (underwater curve, top drawdowns)        │
│ • Rolling metrics (Sharpe, volatility, beta windows)         │
│ • Risk metrics (VaR, CVaR, tail ratio)                       │
│ • Monthly/annual returns visualization                       │
└──────────────────────────────────────────────────────────────┘
```

**See [docs/architecture.md](docs/architecture.md) for complete technical details.**

---

## Key Features

### Trade-Level Diagnostics

Connect SHAP explanations to trade outcomes for systematic continuous improvement.

**Core workflow**:
1. Extract worst trades from backtest
2. Align SHAP values to trade entry timestamps
3. Cluster trades by SHAP similarity (hierarchical clustering)
4. Generate actionable hypotheses for improvement
5. Iterate: Adjust features/model based on insights

**Benefits**:
- **Systematic debugging**: Understand exactly why trades fail
- **Pattern discovery**: Find recurring error modes
- **Actionable insights**: Get specific improvement suggestions
- **Continuous improvement**: Close the ML→trading feedback loop

### Performance (10-100x Faster)

**Polars + Numba optimization** for blazing fast analysis:

| Operation | Dataset | Time |
|-----------|---------|------|
| 5-fold CV | 1M rows | <10 seconds |
| Feature importance | 100 features | <5 seconds |
| CPCV backtest | 100K bars | <30 seconds |
| DSR calculation | 252 returns | <50ms |

### Interactive Visualizations

**Modern Plotly charts** (not outdated matplotlib):
- Hover for details
- Zoom and pan
- Responsive design
- Publication-ready
- Export to HTML/PDF

### Auto-Interpretation

**Human-readable insights**, not just numbers:
```python
results.warnings
# ["High Conditional IC but low H-statistic for (momentum, volatility)",
#  "Suggests regime-specific interaction - investigate market conditions"]

results.interpretation
# "Strong consensus across 3 methods. Top interaction: momentum × volatility.
#  High agreement (Spearman 0.85+) indicates robust finding."
```

### Advanced Statistics

**State-of-the-art methods** from López de Prado and others:
- **DSR** (Deflated Sharpe Ratio) - Corrects for multiple testing
- **CPCV** (Combinatorial Purged Cross-Validation) - Leak-free validation
- **RAS** (Rademacher Anti-Serum) - Backtest overfitting detection
- **PBO** (Probability of Backtest Overfitting) - Overfitting probability
- **HAC-adjusted IC** - Autocorrelation-robust information coefficient
- **FDR control** (Benjamini-Hochberg) - Multiple comparisons
- **SHAP-based diagnostics** - Trade-level error analysis

### Time Series Diagnostics

**Understand your data before making decisions**:

```python
from ml4t.diagnostic.evaluation import (
    analyze_stationarity,
    analyze_autocorrelation,
    analyze_volatility,
    analyze_distribution,
)

# Stationarity: ADF, KPSS, Phillips-Perron with consensus
result = analyze_stationarity(returns)
print(f"Consensus: {result.consensus}")  # 'stationary', 'non_stationary', 'inconclusive'
print(f"ADF p-value: {result.adf_result.pvalue:.4f}")

# Autocorrelation: ACF/PACF with significance bands
acf_result = analyze_autocorrelation(returns, nlags=20)
print(f"Significant lags: {acf_result.significant_lags}")

# Volatility: ARCH-LM test, GARCH(1,1) fitting
vol_result = analyze_volatility(returns)
print(f"ARCH effects: {vol_result.has_arch_effects}")

# Distribution: moments, normality, tail analysis
dist_result = analyze_distribution(returns)
print(f"Skewness: {dist_result.skewness:.3f}")
print(f"Jarque-Bera p-value: {dist_result.jb_pvalue:.4f}")
```

### Signal Analysis

**Full signal evaluation framework**:

```python
from ml4t.diagnostic.evaluation import SignalAnalysis, MultiSignalAnalysis

# Single signal analysis
signal_analyzer = SignalAnalysis(
    signal=factor_data,
    returns=forward_returns,
    periods=[1, 5, 21],  # 1D, 1W, 1M
)

# IC analysis with HAC adjustment
ic_result = signal_analyzer.compute_ic_analysis()
print(f"IC Mean: {ic_result.ic_mean:.4f}")
print(f"IC IR: {ic_result.ic_ir:.4f}")
print(f"HAC t-stat: {ic_result.hac_tstat:.2f}")

# Quantile returns
quantile_result = signal_analyzer.compute_quantile_analysis()
print(f"Q5-Q1 spread: {quantile_result.spread:.2%}")

# Turnover analysis
turnover = signal_analyzer.compute_turnover_analysis()

# Multi-signal comparison
multi_analyzer = MultiSignalAnalysis(signals_dict, returns)
ranking = multi_analyzer.rank_signals(metric='ic_ir')
```

### Portfolio Analysis

**Full portfolio tear sheet** with modern visualizations:

```python
from ml4t.diagnostic.evaluation import PortfolioAnalysis

# Initialize with returns
portfolio = PortfolioAnalysis(returns, benchmark=spy_returns)

# Summary statistics
metrics = portfolio.compute_summary_stats()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Sortino: {metrics.sortino_ratio:.2f}")
print(f"Calmar: {metrics.calmar_ratio:.2f}")
print(f"Omega: {metrics.omega_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")

# Rolling metrics
rolling = portfolio.compute_rolling_metrics(window=252)
rolling_df = rolling.to_dataframe()  # rolling Sharpe, vol, beta

# Drawdown analysis
drawdowns = portfolio.compute_drawdown_analysis(top_n=5)
print(f"Worst drawdown: {drawdowns.max_drawdown:.2%}")
print(f"Recovery days: {drawdowns.max_duration}")

# Generate tear sheet
portfolio.generate_tear_sheet()  # Interactive Plotly dashboard
```

### Seamless Integration

**Works with your existing tools**:
```python
# Supports pandas, polars, numpy
X_pandas = pd.DataFrame(...)
X_polars = pl.DataFrame(...)
X_numpy = np.array(...)

# All work seamlessly
analyze_ml_importance(model, X_pandas, y)
analyze_ml_importance(model, X_polars, y)
analyze_ml_importance(model, X_numpy, y)
```

**Integrates with popular backtesting engines**:
- ml4t-backtest (native support)
- zipline-reloaded (via adapter)
- VectorBT (via adapter)
- Custom engines (implement TradeRecord schema)

---

## Modular Design

Like AlphaLens, **every function works standalone or composed**:

```python
# Use individual metrics
from ml4t.diagnostic.evaluation import compute_ic_series, compute_h_statistic

ic = compute_ic_series(features, returns)
h_stat = compute_h_statistic(model, X)

# Or use tear sheets (combines multiple metrics)
from ml4t.diagnostic.evaluation import analyze_ml_importance

importance = analyze_ml_importance(model, X, y)
# → Combines MDI, PFI, MDA, SHAP
# → Consensus ranking
# → Warnings and interpretation

# Or use full workflow
from ml4t.diagnostic.evaluation import TradeShapAnalyzer

analyzer = TradeShapAnalyzer(model, features_df, shap_values, config)
result = analyzer.explain_worst_trades(worst_trades)
# → Trade analysis + SHAP + clustering + hypotheses
```

---

## Documentation

### User Guides
- **[Trade Diagnostics Example](examples/trade_diagnostics_example.ipynb)** - Complete end-to-end tutorial
- **[Architecture Guide](docs/architecture.md)** - Technical deep dive
- **[Visualization Strategy](docs/visualization_strategy.md)** - Plotly + reporting
- **[Data Schemas](docs/schemas.md)** - Integration contracts
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed installation options

### Academic References
- **[Academic References](docs/REFERENCES.md)** - Comprehensive citations for all implemented methods

### Integration Guides
- **[Book Integration](docs/book_integration.md)** - ML4T 3rd Edition alignment
- **[Backtest Integration](docs/integration_backtest.md)** - Backtesting engine
- **[Engineer Integration](docs/integration_engineer.md)** - Feature engineering

### Technical Documentation
- **[Optional Dependencies](docs/OPTIONAL_DEPENDENCIES.md)** - ML libraries and graceful degradation
- **[Dashboard Guide](docs/DASHBOARD.md)** - Interactive Streamlit dashboard
- **[Error Handling](docs/error_handling.md)** - Best practices
- **[Logging](docs/logging.md)** - Structured logging with structlog

### Migration
- **[Migration Guide](docs/MIGRATION.md)** - Upgrade from ml4t.evaluation to ml4t.diagnostic

---

## Optional Dependencies

ML4T Diagnostic is designed with minimal required dependencies. Optional ML libraries enhance functionality but are NOT required:

**Available Features**:
- **Core Analysis** - Always available (IC, statistics, distributions, DSR, RAS)
- **ML Importance** - Requires `lightgbm` or `xgboost`
- **SHAP Analysis** - Requires `shap` (interpretability)
- **Deep Learning** (v1.1+) - Requires `tensorflow` or `pytorch`
- **GPU Acceleration** (v1.1+) - Requires `cupy`
- **Dashboards** - Requires `streamlit` (interactive viz)

**Quick Check**:
```python
from ml4t.diagnostic.utils import get_dependency_summary
print(get_dependency_summary())
```

**Installation Options**:
```bash
# Core library (no ML dependencies)
pip install ml4t-diagnostic

# Standard ML support (Tree, Linear, Kernel explainers)
pip install ml4t-diagnostic[ml]      # LightGBM, XGBoost, SHAP

# Neural network support (adds Deep explainer)
pip install ml4t-diagnostic[deep]    # + TensorFlow

# GPU acceleration (10-50x speedup for large datasets)
pip install ml4t-diagnostic[gpu]     # + cupy

# Visualization and dashboards
pip install ml4t-diagnostic[viz]     # + Plotly, Streamlit

# Everything (all explainers + GPU + viz)
pip install ml4t-diagnostic[all-ml]  # ml + deep + gpu
pip install ml4t-diagnostic[all]     # all-ml + viz
```

**Explainer Availability (v1.1)**:

| Explainer | Dependency Group | Required Packages |
|-----------|-----------------|-------------------|
| TreeExplainer | `[ml]` | shap, lightgbm/xgboost |
| LinearExplainer | `[ml]` | shap, scikit-learn |
| KernelExplainer | `[ml]` | shap, scikit-learn |
| DeepExplainer | `[deep]` | shap, tensorflow or pytorch |
| GPU Support | `[gpu]` | cupy |

**Graceful Degradation**: Missing dependencies trigger clear warnings, not crashes. See [docs/OPTIONAL_DEPENDENCIES.md](docs/OPTIONAL_DEPENDENCIES.md) for details.

---

## API Stability

**ML4T Diagnostic follows [Semantic Versioning](https://semver.org/)**.

| Version Type | API Changes | Examples |
|--------------|-------------|----------|
| **Patch** (1.3.x) | Bug fixes only | Performance improvements, docs |
| **Minor** (1.x.0) | Backward compatible | New features, new config options |
| **Major** (x.0.0) | Breaking changes | Removed functions, renamed params |

**Public API**: Everything in `__all__` exports is considered stable. Internal modules (prefixed with `_`) may change without notice.

**Current Stability**: As of v1.3.0, the API is considered **stable** for production use.

---

## Development Status

**Current**: v0.1.0a1

### v1.3 - Module Decomposition & UX Improvements

**Major Feature**: Large monolithic modules decomposed into focused submodules for better maintainability.

**Key improvements**:
- **Module Decomposition**: 5 large modules (~12,000 lines) split into focused submodules
  - `metrics.py` (5,643 lines) → 13 modules in `metrics/`
  - `distribution.py`, `drift.py`, `stationarity.py` → dedicated packages
- **ValidatedCrossValidation**: One-step CPCV + DSR validation (20 lines → 5 lines)
- **Result.interpret()**: Human-readable insights on all key result classes
- **Data Quality Integration**: `DataQualityReport` contract with ml4t-data
- **Backward Compatible**: All old imports still work via `__init__.py` exports
- **Type Stubs**: Added `py.typed` marker for better IDE support

**New Usage Pattern**:
```python
from ml4t.diagnostic import ValidatedCrossValidation

# One-step validated cross-validation (combines CPCV + DSR)
vcv = ValidatedCrossValidation(n_splits=10, embargo_pct=0.01)
result = vcv.fit_evaluate(X, y, model, times=times)

if result.is_significant:
    print(f"Strategy passes DSR at {result.significance_level:.0%} confidence")
    print(result.summary())
else:
    print("Strategy may be overfit - DSR test failed")
    for insight in result.interpretation:
        print(f"  • {insight}")
```

### v1.2 - Configuration Consolidation

**Major Feature**: Reduced 61+ config classes to 10 primary configs with single-level nesting.

**Key improvements**:
- **Config Consolidation**: `DiagnosticConfig`, `StatisticalConfig`, `TradeConfig`, etc.
- **Single-Level Nesting**: `config.stationarity.enabled` (not deeply nested)
- **Presets Preserved**: `for_quick_analysis()`, `for_research()`, `for_production()`
- **Backward Compatible**: Old class names work as deprecated aliases

### v1.1 - Model-Agnostic SHAP Support

**Major Feature**: SHAP importance now works with **ANY sklearn-compatible model**, not just tree models!

**Key improvements**:
- **Multi-Explainer Support**: Auto-selects best explainer (Tree, Linear, Kernel, Deep)
- **Universal Compatibility**: Works with SVM, KNN, neural networks, ANY model
- **Smart Performance**: Automatic cascade (Tree → Linear → Kernel)
- **GPU Acceleration**: Optional GPU support for large datasets
- **Backward Compatible**: 100% compatible with v1.0 API

**Explainer Comparison**:

| Explainer | Models | Speed | Quality | Installation |
|-----------|--------|-------|---------|--------------|
| **Tree** | LightGBM, XGBoost, RF | <10ms/sample | Exact | `[ml]` |
| **Linear** | LogisticReg, Ridge, Lasso | <100ms/sample | Exact | `[ml]` |
| **Deep** | TensorFlow, PyTorch | <500ms/sample | Approx | `[deep]` |
| **Kernel** | ANY sklearn model | 100-5000ms/sample | Approx | `[ml]` |

**Installation**:
```bash
# Standard ML support (Tree, Linear, Kernel explainers)
pip install ml4t-diagnostic[ml]

# Neural network support (adds Deep explainer)
pip install ml4t-diagnostic[deep]

# GPU acceleration (10-50x speedup for large datasets)
pip install ml4t-diagnostic[gpu]

# Everything (all explainers + GPU)
pip install ml4t-diagnostic[all-ml]
```

**Migration from v1.0**:
- **Zero changes required** - All v1.0 code works unchanged
- **Auto-selection** - Tree models automatically use TreeExplainer
- **New models supported** - Linear and other models now work automatically
- **Explicit control** - Set `explainer_type='kernel'` to force specific explainer
- **Check explainer** - Use `result['explainer_type']` to see which was used

**Example (New in v1.1)**:
```python
from sklearn.svm import SVC
from ml4t.diagnostic.evaluation import compute_shap_importance

# Train ANY model (SVM example - not supported in v1.0!)
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Compute SHAP importance (auto-selects KernelExplainer)
result = compute_shap_importance(model, X_test, max_samples=100)
print(f"Explainer used: {result['explainer_type']}")  # 'kernel'

# Works with linear models too
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
result = compute_shap_importance(model, X_test)  # Auto-selects LinearExplainer
```

### v1.0 - Trade Diagnostics Framework
- Trade analysis framework (TradeAnalysis, TradeMetrics)
- Trade-SHAP diagnostics (TradeShapAnalyzer)
- Error pattern clustering (hierarchical clustering)
- Hypothesis generation (rule-based templates)
- Interactive dashboard (Streamlit)
- Feature importance analysis (MDI, PFI, MDA, SHAP consensus)
- Feature interactions (Conditional IC, H-statistic, SHAP)
- Statistical framework (CPCV, DSR, RAS, FDR, HAC-adjusted IC)
- Time-series cross-validation (purging, embargo)
- Comprehensive example notebook

### Roadmap
- **v0.1**: Alpha release - Core diagnostics framework
- **v0.2**: Event studies and barrier analysis
- **v1.0**: Full book integration (ML4T 3rd Edition)

---

## Performance Benchmarks

**Rigorous time-series validation** (After Numba optimization):

| Operation | Dataset Size | Time | vs Pandas |
|-----------|-------------|------|-----------|
| Maximum Drawdown | 10K points | 2ms | **6x faster** |
| Block Bootstrap | 100K samples | 30ms | **5x faster** |
| Rolling Sharpe | 50K window | 8ms | **12x faster** |
| Information Coefficient | 1M points | 10ms | **5x faster** |
| DSR Calculation | 252 returns | 50ms | **10x faster** |

**Target achieved**: 5-fold CV on 1M rows < 10 seconds

---

## Leakage Prevention

**Information leakage** in validation causes inflated performance estimates. ML4T Diagnostic provides tools to prevent common validation pitfalls:

### 1. Cross-Validation Leakage

**Wrong**: Using standard k-fold on time-series data
```python
# BAD - future data leaks into training
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    model.fit(X[train], y[train])  # WRONG: Train may contain future data
```

**Right**: Purged walk-forward or CPCV
```python
# GOOD - proper temporal separation with purging
from ml4t.diagnostic.splitters import PurgedWalkForwardCV

cv = PurgedWalkForwardCV(
    n_splits=5,
    embargo_pct=0.01,  # Gap between train/test
    purge_pct=0.02     # Remove overlapping labels
)
for train, test in cv.split(X, y, times):
    model.fit(X[train], y[train])  # Strictly past data only
```

### 2. Threshold Selection Leakage

**Wrong**: Optimizing thresholds on full dataset
```python
# BAD - uses test data to select threshold
from sklearn.metrics import f1_score
best_threshold = max(thresholds, key=lambda t: f1_score(y, pred > t))  # WRONG
```

**Right**: Train-only threshold optimization
```python
# GOOD - optimize on training fold only
from ml4t.diagnostic.evaluation import find_optimal_threshold

for train_idx, test_idx in cv.split(X, y, times):
    # Find optimal threshold using ONLY training data
    optimal = find_optimal_threshold(
        indicator=predictions[train_idx],
        label=y[train_idx],
        metric="f1_score",
        min_coverage=0.02
    )
    # Apply to test set
    test_signals = predictions[test_idx] > optimal['threshold']  # OK
```

### 3. Multiple Testing Correction

**Wrong**: Ignoring number of strategies tested
```python
# BAD - reports raw Sharpe without correction
sharpe = returns.mean() / returns.std() * np.sqrt(252)
print(f"Sharpe: {sharpe:.2f}")  # WRONG: May be spurious from many trials
```

**Right**: Deflated Sharpe Ratio accounts for trials
```python
# GOOD - corrects for multiple testing
from ml4t.diagnostic.evaluation import comprehensive_sharpe_evaluation

results = comprehensive_sharpe_evaluation(
    returns=strategy_returns,
    SR_benchmark=0.0,
    K_trials=100,         # Account for all strategies tested
    variance_trials=0.1,  # Variance across trials
    alpha=0.05
)
print(f"Raw Sharpe: {results['SR_observed']:.2f}")
print(f"Deflated Sharpe: {results['DSR']:.2f}")  # Adjusted for trials
print(f"Significant: {results['is_significant']}")
```

### Best Practice: Use CPCV for All Validation

The `CombinatorialPurgedCV` ensures leak-proof validation by construction:

```python
from ml4t.diagnostic.splitters import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_splits=10,
    embargo_pct=0.01,   # Gap after test period
    purge_pct=0.05      # Remove label overlap
)

# Each fold is leak-proof by design
for train_idx, test_idx in cv.split(X, y, timestamps):
    # Training data strictly precedes test data
    # Embargo prevents information bleeding
    # Purging handles overlapping label windows
    pass
```

---

## For ML4T Book Readers

ML4T Diagnostic is the **reference implementation** for the ML4T 3rd Edition book.

**Chapter mapping** (ML4T 3rd Edition):
- Chapter 6 (Alpha Factor Engineering) → `FeatureDiagnostics`, feature importance, interactions
- Chapter 7 (Evaluating Alpha Factors) → `SignalAnalysis`, IC analysis, RAS
- Chapter 9 (Backtesting) → `TradeAnalysis`, DSR, CPCV, `TradeShapAnalyzer`
- Chapter 10 (Portfolio Construction) → `PortfolioAnalysis`, rolling metrics, drawdowns
- Chapter 12 (Risk Management) → Risk metrics, VaR, stress tests

See [docs/book_integration.md](docs/book_integration.md) for complete mapping.

---

## Contributing

We welcome contributions! See [CLAUDE.md](CLAUDE.md) for:
- Development setup
- Code standards (ruff, mypy, pytest)
- Architecture principles
- How to add new metrics/tear sheets

---

## Citation

If you use ML4T Diagnostic in your research, please cite:

```bibtex
@software{ml4t_diagnostic2025,
  author = {Stefan Jansen},
  title = {ML4T Diagnostic: Comprehensive Diagnostics for Quantitative Finance},
  year = {2025},
  version = {0.1.0a1},
  publisher = {GitHub},
  url = {https://github.com/stefan-jansen/ml4t-diagnostic}
}
```

For academic references to the statistical methods implemented in this library, see **[docs/REFERENCES.md](docs/REFERENCES.md)**.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Related Projects

Part of the **ML4T ecosystem**:
- **[ml4t-data](../data/)** - Market data infrastructure
- **[ml4t-engineer](../engineer/)** - Feature engineering toolkit
- **[ml4t-backtest](../backtest/)** - Event-driven backtest engine
- **[ml4t-diagnostic](../diagnostics/)** - This library

---

**Ready to get started?** See [Quick Start](#quick-start) above or dive into the [Trade Diagnostics Example](examples/trade_diagnostics_example.ipynb).
