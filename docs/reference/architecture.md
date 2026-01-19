# ML4T Diagnostic Architecture

**Version**: 2.0.0-alpha
**Last Updated**: 2025-11-04
**Status**: Living document

---

## Table of Contents

1. [Vision & Value Proposition](#vision--value-proposition)
2. [Four-Tier Architecture](#four-tier-architecture)
3. [Modular Tear Sheet Pattern](#modular-tear-sheet-pattern)
4. [Visualization Layer](#visualization-layer)
5. [Data Schemas & Contracts](#data-schemas--contracts)
6. [Design Principles](#design-principles)
7. [Technology Stack](#technology-stack)
8. [Integration Points](#integration-points)

---

## Vision & Value Proposition

### What is ML4T Diagnostic?

**ML4T Diagnostic is the modern Alphalens + Pyfolio replacement for the machine learning era.**

### What We're NOT

- ❌ A wrapper library for sklearn/scipy
- ❌ Another backtesting engine (VectorBT, zipline already exist)
- ❌ Just a metrics collection

### What We ARE

**A comprehensive tear sheet system** spanning the entire ML4T workflow:
- Pre-modeling: Feature evaluation (Alphalens++)
- During modeling: Model diagnostics (NEW capability)
- Post-modeling: Backtest evaluation (Pyfolio++)
- Portfolio-level: Performance and risk analysis

### Key Differentiators

| Feature | Alphalens/Pyfolio | ML4T Diagnostic |
|---------|-------------------|-------|
| **Performance** | Pandas-based, slow | Polars-powered, 10-100x faster |
| **Visualizations** | Static matplotlib | Interactive Plotly |
| **Insights** | Raw metrics only | Auto-interpretation + warnings |
| **Statistics** | Basic tests | DSR, CPCV, RAS, HAC-adjusted, FDR control |
| **Scope** | Feature OR backtest | Full workflow (feature → model → backtest → portfolio) |
| **Model Diagnostics** | None | Comprehensive (pred vs actual, errors, regimes) |

### The "Tear Sheet" is the Product

**One command → comprehensive analysis → professional report**

```python
# Feature evaluation
from ml4t-diagnostic import analyze_feature_evaluation
results = analyze_feature_evaluation(X, y)
results.generate_report('factor_analysis.html')

# Model diagnostics
from ml4t-diagnostic import analyze_model_diagnostics
results = analyze_model_diagnostics(y_true, y_pred, X)
results.generate_report('model_diagnostics.html')

# Backtest evaluation
from ml4t-diagnostic import analyze_backtest_evaluation
results = analyze_backtest_evaluation(backtest_results)
results.generate_report('backtest_analysis.html')
```

---

## Four-Tier Architecture

ML4T Diagnostic is organized into four major analysis tiers, each addressing a distinct phase of the quantitative workflow.

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 1: Feature Analysis (Pre-Modeling)                        │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Individual feature characteristics (time-series stats)        │
│ • Feature relationships (correlation, orthogonality)            │
│ • Feature-outcome relationships (IC, MI, quantile analysis)     │
│ • Multivariate relationships (importance, interactions)         │
│ • Drift detection (PSI, domain classifier)                      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Tier 2: Model Diagnostics (During/After Modeling) [NEW!]       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Prediction quality (scatter, R², residuals)                   │
│ • Error distribution (histograms, outliers, bias)               │
│ • Top mistakes analysis (largest errors, patterns)              │
│ • Regime-specific performance (bull/bear, vol regimes)          │
│ • Temporal stability (rolling metrics, drift)                   │
│ • Calibration analysis (reliability diagrams, Brier score)      │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Tier 3: Backtest Analysis (Post-Modeling)                      │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Performance analysis (Pyfolio-style tear sheet)               │
│ • Statistical validity (DSR, RAS, FDR corrections)              │
│ • Trade analytics (entry/exit, holding periods)                 │
│ • Signal quality (decay, coverage, turnover)                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Tier 4: Portfolio Analysis (Production)                        │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Composition analysis (weights, concentration, turnover)       │
│ • Risk analysis (factor exposures, VaR, stress tests)          │
│ • Attribution analysis (returns decomposition)                  │
│ • Comparison framework (method horse races)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Tier 1: Feature Analysis

**Purpose**: Evaluate model inputs before training

**Module structure**:
```
ml4t-diagnostic/feature_analysis/
├── characteristics.py    # Time-series stats, distributions
├── relationships.py      # Correlation, orthogonality, PCA
├── predictiveness.py     # IC, MI, quantile analysis (Alphalens-style)
├── importance.py         # MDI, PFI, MDA, SHAP (✅ TASK-049)
├── interactions.py       # Conditional IC, H-stat, SHAP (✅ TASK-053)
└── drift.py             # PSI, KS, domain classifier
```

**Key tear sheets**:
- `analyze_feature_characteristics()` - Stationarity, autocorrelation, outliers
- `analyze_feature_predictiveness()` - Alphalens-style factor analysis
- `analyze_feature_importance()` - Consensus importance across methods
- `analyze_feature_interactions()` - Consensus interaction detection
- `analyze_feature_drift()` - Distribution shift detection

### Tier 2: Model Diagnostics (NEW - Differentiator)

**Purpose**: Systematic model output analysis

**Module structure**:
```
ml4t-diagnostic/model_diagnostics/
├── prediction_quality.py  # R², scatter, residuals
├── error_analysis.py      # Distribution, outliers, top mistakes
├── calibration.py         # Calibration curves, Brier score
├── regime_analysis.py     # Performance by market regime
└── stability.py           # Temporal drift, rolling metrics
```

**Key tear sheets**:
- `analyze_prediction_quality()` - Predictions vs actuals
- `analyze_error_distribution()` - Error patterns and outliers
- `analyze_top_mistakes()` - Largest errors with feature characterization
- `analyze_regime_performance()` - Bull/bear, high/low vol splits
- `analyze_prediction_stability()` - Rolling metrics, drift detection
- `analyze_prediction_calibration()` - Reliability diagrams

**Why this is critical**: Neither the book nor competitors (Alphalens, Pyfolio) address systematic model diagnostics. This is a huge gap and opportunity.

### Tier 3: Backtest Analysis

**Purpose**: Validate strategy performance

**Module structure**:
```
ml4t-diagnostic/backtest_analysis/
├── performance.py        # Pyfolio-style tear sheet
├── validity.py          # DSR, RAS, FDR corrections
├── trade_analytics.py   # Entry/exit, holding periods
└── signal_analysis.py   # Signal quality, decay, coverage (NEW!)
```

**Key tear sheets**:
- `analyze_backtest_performance()` - Returns, drawdowns, risk metrics
- `analyze_backtest_validity()` - Statistical corrections (DSR, RAS, FDR)
- `analyze_trade_quality()` - Trade-level analytics
- `analyze_signal_decay()` - Signal strength over time

**Integration**: Designed to work seamlessly with qengine backtest engine (our own) and zipline-reloaded.

### Tier 4: Portfolio Analysis

**Purpose**: Production portfolio monitoring and optimization

**Module structure**:
```
ml4t-diagnostic/portfolio_analysis/
├── composition.py    # Weights, concentration, turnover
├── risk.py          # Factor exposures, VaR, stress tests
└── attribution.py   # Returns decomposition
```

**Key tear sheets**:
- `analyze_portfolio_composition()` - Weight analysis, concentration
- `analyze_portfolio_risk()` - Factor exposures, tail risk
- `analyze_portfolio_attribution()` - Performance decomposition
- `compare_portfolio_methods()` - Horse race framework

---

## Modular Tear Sheet Pattern

### Three-Layer Architecture

ML4T Diagnostic follows a strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Ingredient Metrics (compute_*)                    │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Low-level calculations                                    │
│ • Single metric, single method                              │
│ • Pure functions (no side effects)                          │
│ • Examples: compute_ic(), compute_h_statistic()            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Tear Sheets (analyze_*)                          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Orchestrate multiple ingredient metrics                   │
│ • Consensus analysis across methods                         │
│ • Auto-generated warnings and interpretation                │
│ • Return structured dicts (no visualization)                │
│ • Examples: analyze_ml_importance(), analyze_interactions() │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Orchestrators (Full Workflow)                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • Combine multiple tear sheets                              │
│ • End-to-end analysis pipelines                             │
│ • Examples: analyze_feature_evaluation(),                   │
│             analyze_model_diagnostics()                     │
└─────────────────────────────────────────────────────────────┘
```

### AlphaLens-Style Composability

Like AlphaLens, every tear sheet can be used:
- **Standalone**: `analyze_feature_importance(model, X, y)`
- **Composed**: As part of `analyze_feature_evaluation(X, y)`
- **Custom**: Users can mix and match as needed

### Example: Feature Importance

```python
# Layer 1: Ingredient metrics
mdi_results = compute_mdi_importance(model, X, y)
pfi_results = compute_permutation_importance(model, X, y)
shap_results = compute_shap_importance(model, X, y)

# Layer 2: Tear sheet (consensus)
importance = analyze_feature_importance(model, X, y)
# → Combines MDI, PFI, MDA, SHAP
# → Consensus ranking
# → Method agreement
# → Warnings and interpretation

# Layer 3: Orchestrator (full workflow)
full_analysis = analyze_feature_evaluation(X, y)
# → Includes importance, interactions, drift, etc.
```

### Standard Output Format

Every `analyze_*()` function returns a standardized dict:

```python
{
    "method_results": {...},      # Individual method outputs
    "consensus": {...},           # Cross-method synthesis
    "warnings": [...],            # Auto-generated alerts
    "interpretation": "...",      # Human-readable summary
    "metadata": {...}             # Execution details
}
```

**No visualization in analysis layer** - that's separate (see next section).

---

## Visualization Layer

### Layered Visualization Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Analysis Engine (Pure Python)                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • analyze_*() functions return dicts                        │
│ • No visualization dependencies                             │
│ • Maximum flexibility                                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Visualization (Plotly)                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • plot_*() functions consume analysis dicts                 │
│ • Return Plotly Figure objects                              │
│ • Display anywhere (Jupyter, HTML, embed in apps)           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Reporting (Static)                                │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • generate_html_report() - Shareable reports                │
│ • generate_pdf_report() - Publication quality               │
│ • Jinja2 templates + embedded Plotly                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Dashboard (Optional, Separate Package)            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ • ml4t-diagnostic-streamlit (separate package)                        │
│ • Interactive exploration UI                                │
│ • For power users who need interactivity                    │
└─────────────────────────────────────────────────────────────┘
```

### Why Plotly (Not Matplotlib)?

| Criterion | Matplotlib | Plotly |
|-----------|------------|--------|
| Interactivity | Static | Native interactive (zoom, pan, hover) |
| Web-native | Requires conversion | Native HTML/JavaScript |
| Modern look | Dated styling | Professional, modern |
| Speed | Slow for large data | Fast, GPU-accelerated |
| Mobile | Poor | Responsive |

### Why Not Streamlit (In Core)?

Streamlit is **optional** (Layer 4) because:
- Heavy dependency (entire framework)
- Forces specific deployment model
- Not suitable for programmatic use
- Better as separate package: `ml4t-diagnostic-streamlit`

Users who want dashboards can:
1. Use Plotly figures in their own frameworks
2. Install optional `ml4t-diagnostic-streamlit` package
3. Build custom dashboards with ml4t-diagnostic as backend

### Browser-First Delivery

**Primary interface**: HTML reports with embedded Plotly

```python
results = analyze_feature_importance(model, X, y)

# Generate HTML report
results.generate_report('importance_report.html')
# → Opens in browser
# → Fully interactive Plotly charts
# → No server required
# → Shareable as file
```

**Advantages**:
- Works anywhere (no server)
- Shareable (just send HTML file)
- Interactive (Plotly charts work offline)
- Professional (clean, modern styling)

---

## Data Schemas & Contracts

### Why Schemas?

**Problem**: ml4t-diagnostic integrates with multiple tools (qengine, zipline, vectorbt, user code).

**Solution**: Define standard schemas for data exchange.

**Benefits**:
- Type safety (Pydantic validation)
- Clear contracts
- Easy integration
- Interoperability

### Core Schemas

#### 1. BacktestResults

```python
from pydantic import BaseModel
from typing import Optional
import polars as pl

class BacktestResults(BaseModel):
    """Standard schema for backtest outputs"""

    # Portfolio-level
    returns: pl.DataFrame          # timestamp, returns
    positions: pl.DataFrame        # timestamp, asset, quantity, value
    equity_curve: pl.DataFrame     # timestamp, portfolio_value

    # Transaction-level
    transactions: pl.DataFrame     # timestamp, asset, quantity, price, fees

    # Signal-level (NEW - expanding Pyfolio)
    signals: Optional[pl.DataFrame]  # timestamp, asset, signal_value

    # Metadata
    start_date: datetime
    end_date: datetime
    initial_capital: float
    strategy_name: str
    parameters: dict
```

**Usage**:
```python
# qengine outputs this format natively
results = backtest.run(strategy)

# zipline needs conversion
zipline_results = run_algorithm(...)
results = BacktestResults.from_zipline(zipline_results)

# ml4t-diagnostic analyzes
tear_sheet = analyze_backtest_performance(results)
```

#### 2. ModelPredictions

```python
class ModelPredictions(BaseModel):
    """Standard schema for model outputs"""

    predictions: pl.DataFrame      # timestamp/id, prediction
    actuals: pl.DataFrame          # timestamp/id, actual
    features: Optional[pl.DataFrame]  # timestamp/id, features

    model_name: str
    model_type: str  # 'classification', 'regression'
    feature_names: list[str]
    prediction_date: datetime
```

#### 3. FeatureData (AlphaLens-compatible)

```python
class FeatureData(BaseModel):
    """Standard schema for factor/feature analysis"""

    # Factor values (MultiIndex: date, asset)
    factor_data: pl.DataFrame

    # Forward returns at multiple horizons
    forward_returns: dict[str, pl.DataFrame]  # {'1D': df, '5D': df}

    # Optional grouping
    groups: Optional[pl.DataFrame]  # asset, sector/industry
```

### Schema Versioning

Schemas can evolve:
```python
class BacktestResults(BaseModel):
    schema_version: str = "2.0"  # Track version

    # Use validator for migrations
    @validator('schema_version')
    def migrate_if_needed(cls, v):
        if v == "1.0":
            # Migrate from old format
            ...
        return v
```

---

## Design Principles

### 1. Separation of Concerns

**Analysis ≠ Visualization ≠ Reporting**

- Analysis functions return dicts (no viz dependencies)
- Visualization functions consume dicts (no analysis logic)
- Reporting combines both (orchestration only)

### 2. Composability Over Monoliths

**Small, focused functions** that can be:
- Used standalone
- Combined in tear sheets
- Orchestrated in workflows

NOT: One giant function that does everything.

### 3. Convention Over Configuration

**Sensible defaults**, rare overrides:
```python
# Works out of the box
analyze_feature_importance(model, X, y)

# Can configure if needed
analyze_feature_importance(
    model, X, y,
    methods=['mdi', 'shap'],  # Subset
    max_samples=500           # Override default
)
```

### 4. Fail Gracefully

**Partial success > complete failure**

```python
# If SHAP fails, continue with MDI, PFI, MDA
results = analyze_feature_importance(model, X, y)
# → results['methods_failed'] = ['shap']
# → Still get useful output from other methods
```

### 5. Auto-Interpretation

**Humans want insights, not just numbers**

Every tear sheet includes:
- **Warnings**: Detected issues
- **Interpretation**: Human-readable summary
- **Recommendations**: What to do next

Example:
```python
{
    "warnings": [
        "High Conditional IC but low H-statistic for (momentum, volatility)",
        "Suggests regime-specific interaction - investigate market conditions"
    ],
    "interpretation": "Strong consensus across 3 methods. Top interaction: "
                     "momentum × volatility (avg rank 1.2). High agreement "
                     "(Spearman 0.85+) indicates robust finding.",
    "recommendations": [
        "Focus on top 3 interactions for modeling",
        "Investigate momentum-volatility regime dependence",
        "Consider interaction terms in final model"
    ]
}
```

### 6. Performance Matters

**10-100x faster than pandas alternatives**

- Polars for data manipulation (parallel, lazy)
- NumPy for numerics (vectorized)
- Efficient algorithms (avoid O(n²) when possible)
- Benchmark and optimize hot paths

### 7. Type Safety

**Type hints everywhere**

```python
def analyze_interactions(
    model: Any,
    X: Union[pl.DataFrame, pd.DataFrame, np.ndarray],
    y: Union[pl.Series, pd.Series, np.ndarray],
    feature_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    ...
```

Benefits:
- IDE autocomplete
- Type checking (mypy)
- Self-documenting code

### 8. Test Everything

**80%+ coverage target**

- Unit tests for every function
- Integration tests for workflows
- Regression tests for visualizations
- Property-based tests where appropriate

---

## Technology Stack

### Core Libraries

| Component | Library | Version | Why |
|-----------|---------|---------|-----|
| Data manipulation | Polars | 1.0+ | 10-100x faster than pandas |
| Numerics | NumPy | 1.24+ | Standard numerical computing |
| Statistics | SciPy, statsmodels | Latest | Statistical tests |
| ML models | scikit-learn | 1.3+ | Standard ML library |
| Advanced ML | LightGBM, XGBoost, SHAP | Latest | Tree models, explainability |
| Visualization | Plotly | 5.0+ | Interactive, web-native |
| Schemas | Pydantic | 2.0+ | Data validation |
| Testing | pytest | 7.0+ | Test framework |

### Optional Dependencies

Grouped by use case:
```toml
[project.optional-dependencies]
ml = ["lightgbm>=4.0.0", "xgboost>=2.0.0", "shap>=0.43.0"]
viz = ["plotly>=5.0.0", "kaleido>=0.2.1"]  # PDF export
reporting = ["jinja2>=3.1.0", "markdown>=3.4.0"]
all = ["ml4t-diagnostic[ml,viz,reporting]"]
```

### Development Tools

- **Linting**: ruff (fast, comprehensive)
- **Type checking**: mypy (strict mode)
- **Formatting**: ruff format (Black-compatible)
- **Pre-commit**: Enforce quality gates

---

## Integration Points

### 1. Book Integration (ML4T 3rd Edition)

**Reciprocal relationship**:
- Book explains concepts → Library implements
- Library provides tools → Book demonstrates usage

**Chapter mapping**: See `docs/book_integration.md`

### 2. QEngine Integration

**Backtest engine integration**:
```python
from qengine import Backtest
from ml4t-diagnostic import analyze_backtest_performance

# Run backtest
results = backtest.run(strategy)
# → Returns BacktestResults schema

# Analyze with ml4t-diagnostic
tear_sheet = analyze_backtest_performance(results)
```

### 3. Other Backtesters

**Support for**:
- zipline-reloaded (via conversion)
- VectorBT (via conversion)
- Custom engines (implement BacktestResults schema)

### 4. Jupyter Integration

**Native support**:
```python
# Display Plotly figures inline
results = analyze_feature_importance(model, X, y)
fig = plot_importance_summary(results)
fig.show()  # Renders in notebook

# Or generate full report
results.generate_notebook_report()  # Markdown + embedded charts
```

### 5. Production Monitoring

**Live trading integration**:
```python
# Log predictions and actuals in same format
predictions = ModelPredictions(
    predictions=live_predictions,
    actuals=realized_outcomes,
    model_name="production_model_v2"
)

# Monitor with ml4t-diagnostic
diagnostics = analyze_prediction_stability(predictions)
# → Detects drift, performance degradation
```

---

## Roadmap & Evolution

### Current State (v2.0.0-alpha)

**Completed**:
- ✅ Feature importance (TASK-049)
- ✅ Feature interactions (TASK-050-053)
- ✅ CPCV, DSR, FDR, HAC-adjusted IC

**In Progress**:
- ⏳ Module C (drift detection remaining)
- ⏳ Documentation (this document)

### Next Priorities

**Phase 3.5** (16h) - Documentation ← CURRENT
- ✅ Architecture (this document)
- ⏸️ README with value proposition
- ⏸️ Book integration mapping
- ⏸️ Visualization strategy
- ⏸️ Data schemas

**Phase 4** (40h) - Visualization Layer ← CRITICAL NEXT
- Prove the tear sheet → visualization pipeline works
- Implement plot_*() functions
- HTML report generation
- Example reports

**Phase 5-7** - Complete Analysis Modules
- Finish Module C (drift)
- Module A (feature diagnostics)
- Module B (cross-feature analysis)
- Enhanced Sharpe framework

**Phase 8** (60h) - Model Diagnostics ← DIFFERENTIATOR
- Prediction quality, error analysis
- Regime performance
- Calibration
- **Fills critical gap**

**Phase 9-10** - Portfolio & Polish
- Portfolio analysis module
- Integration and polish
- Performance benchmarks
- Release preparation

### Long-Term Vision

**v2.1**: Core modules complete, visualization layer stable
**v2.2**: Model diagnostics module complete
**v2.5**: Full book integration, all examples working
**v3.0**: Production-ready, performance optimized, comprehensive docs

**Optional future**: `ml4t-diagnostic-streamlit` package for interactive dashboards

---

## Contributing

### How to Extend ML4T Diagnostic

**Adding a new metric**:
1. Implement as `compute_*()` function (ingredient)
2. Add tests
3. Export in `__init__.py`
4. Consider adding to existing tear sheet

**Adding a new tear sheet**:
1. Implement as `analyze_*()` function
2. Combine multiple ingredient metrics
3. Add consensus analysis, warnings, interpretation
4. Create comprehensive tests (15-20)
5. Follow output format standard

**Adding visualization**:
1. Implement as `plot_*()` function (after Layer 2 built)
2. Consume analyze_*() output dict
3. Return Plotly Figure
4. Add to report templates

### Code Standards

- Type hints required
- Docstrings required (NumPy style)
- Tests required (80%+ coverage)
- Linting must pass (ruff)
- Follow existing patterns

---

## References

**Key papers**:
- López de Prado (2018): Advances in Financial Machine Learning
- Bailey & López de Prado (2014): The Deflated Sharpe Ratio
- Friedman & Popescu (2008): Predictive Learning via Rule Ensembles
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions

**Inspirations**:
- AlphaLens: Modular tear sheet design
- Pyfolio: Performance reporting
- SHAP: Unified explainability framework
- Polars: Modern data processing

---

**Document Status**: ✅ Complete
**Last Review**: 2025-11-04
**Next Review**: After Phase 4 (visualization layer complete)
