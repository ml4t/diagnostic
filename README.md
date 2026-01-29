# ml4t-diagnostic

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive diagnostics and statistical validation for quantitative trading strategies. Covers the complete ML workflow from feature analysis through portfolio performance.

## Features

- **Feature Analysis**: Importance (MDI, PFI, MDA, SHAP), interactions, drift detection
- **Signal Analysis**: IC analysis, quantile returns, turnover, multi-signal comparison
- **Trade Diagnostics**: SHAP-based error pattern discovery, trade analysis
- **Portfolio Analysis**: Rolling metrics, drawdown analysis, risk metrics
- **Statistical Validation**: DSR, CPCV, RAS, PBO, FDR corrections
- **Time Series Diagnostics**: Stationarity, ACF, volatility, distribution tests
- **Binary Metrics**: Precision, recall, lift, coverage with Wilson intervals
- **Performance**: Polars-powered for 10-100x faster analysis than pandas

## Installation

```bash
# Core library
pip install ml4t-diagnostic

# With ML dependencies (SHAP, importance analysis)
pip install ml4t-diagnostic[ml]

# With visualization (Plotly reports)
pip install ml4t-diagnostic[viz]

# Everything
pip install ml4t-diagnostic[all]
```

## Quick Start

### Trade Diagnostics

```python
from ml4t.diagnostic.evaluation import TradeAnalysis, TradeShapAnalyzer

# Identify worst trades from backtest
analyzer = TradeAnalysis(trade_records)
worst_trades = analyzer.worst_trades(n=20)

# Explain with SHAP
shap_analyzer = TradeShapAnalyzer(model, features_df, shap_values)
result = shap_analyzer.explain_worst_trades(worst_trades)

# Get actionable hypotheses
for pattern in result.error_patterns:
    print(f"Pattern: {pattern.hypothesis}")
    print(f"  Actions: {pattern.actions}")
    print(f"  Potential savings: ${pattern.potential_impact:,.2f}")
```

### Feature Importance

```python
from ml4t.diagnostic.evaluation import analyze_ml_importance

# Combines MDI, PFI, MDA, SHAP methods
results = analyze_ml_importance(model, X, y)

# Consensus ranking
print(results.consensus_ranking)
# [('momentum', 1.2), ('volatility', 2.1), ...]

# Warnings and interpretation
print(results.warnings)
print(results.interpretation)
```

### Statistical Validation (DSR)

```python
from ml4t.diagnostic.evaluation import stats

# Deflated Sharpe Ratio - accounts for multiple testing
dsr_result = stats.compute_dsr(
    returns=strategy_returns,
    benchmark_sr=0.0,
    n_trials=100,         # Number of strategies tested
    expected_max_sharpe=1.5
)

print(f"Sharpe Ratio: {dsr_result['sr']:.2f}")
print(f"Deflated Sharpe: {dsr_result['dsr']:.2f}")
print(f"Significant: {dsr_result['is_significant']}")
```

### Signal Analysis

```python
from ml4t.diagnostic.evaluation import SignalAnalysis

analyzer = SignalAnalysis(
    signal=factor_data,
    returns=forward_returns,
    periods=[1, 5, 21],  # 1D, 1W, 1M
)

# IC analysis with HAC adjustment
ic_result = analyzer.compute_ic_analysis()
print(f"IC Mean: {ic_result.ic_mean:.4f}")
print(f"HAC t-stat: {ic_result.hac_tstat:.2f}")

# Quantile returns
quantile_result = analyzer.compute_quantile_analysis()
print(f"Q5-Q1 spread: {quantile_result.spread:.2%}")
```

### Portfolio Analysis

```python
from ml4t.diagnostic.evaluation import PortfolioAnalysis

portfolio = PortfolioAnalysis(returns, benchmark=spy_returns)

# Summary metrics
metrics = portfolio.compute_summary_stats()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")

# Rolling metrics
rolling = portfolio.compute_rolling_metrics(window=252)

# Generate tear sheet
portfolio.generate_tear_sheet()
```

### Time Series Diagnostics

```python
from ml4t.diagnostic.evaluation import (
    analyze_stationarity,
    analyze_autocorrelation,
    analyze_volatility,
)

# Stationarity: ADF, KPSS, Phillips-Perron with consensus
result = analyze_stationarity(returns)
print(f"Consensus: {result.consensus}")  # 'stationary', 'non_stationary'

# Autocorrelation: ACF/PACF with significance
acf_result = analyze_autocorrelation(returns, nlags=20)
print(f"Significant lags: {acf_result.significant_lags}")

# Volatility: ARCH-LM test, GARCH fitting
vol_result = analyze_volatility(returns)
print(f"ARCH effects: {vol_result.has_arch_effects}")
```

## Four-Tier Framework

```
Tier 1: Feature Analysis (Pre-Modeling)
├── Time series diagnostics (stationarity, ACF, volatility)
├── Distribution analysis (moments, normality, tails)
├── Feature importance (MDI, PFI, MDA, SHAP)
└── Feature interactions (Conditional IC, H-stat)

Tier 2: Signal Analysis (Model Outputs)
├── IC analysis (time series, histogram, heatmap)
├── Quantile returns (bar, violin, cumulative)
├── Turnover analysis (autocorrelation)
└── Multi-signal comparison and ranking

Tier 3: Backtest Analysis (Post-Modeling)
├── Trade analysis (win/loss, PnL, holding periods)
├── Statistical validity (DSR, RAS, PBO)
├── Trade-SHAP diagnostics (error patterns)
└── Excursion analysis (TP/SL optimization)

Tier 4: Portfolio Analysis (Production)
├── Performance metrics (Sharpe, Sortino, Calmar)
├── Drawdown analysis (underwater, top drawdowns)
├── Rolling metrics (Sharpe, volatility, beta)
└── Risk metrics (VaR, CVaR, tail ratio)
```

## Statistical Methods

| Method | Purpose |
|--------|---------|
| **DSR** (Deflated Sharpe) | Corrects for multiple testing bias |
| **CPCV** (Combinatorial Purged CV) | Leak-free time series validation |
| **RAS** (Rademacher Anti-Serum) | Backtest overfitting detection |
| **PBO** | Probability of backtest overfitting |
| **HAC-adjusted IC** | Autocorrelation-robust information coefficient |
| **FDR Control** | Multiple comparisons (Benjamini-Hochberg) |

## Performance

| Operation | Dataset | Time |
|-----------|---------|------|
| 5-fold CV | 1M rows | <10 seconds |
| Feature importance | 100 features | <5 seconds |
| CPCV backtest | 100K bars | <30 seconds |
| DSR calculation | 252 returns | <50ms |

## API Reference

### Feature Analysis

```python
from ml4t.diagnostic.evaluation import (
    analyze_ml_importance,      # Combined importance analysis
    compute_shap_importance,    # SHAP values
    analyze_interactions,       # Feature interactions
    analyze_stationarity,       # Stationarity tests
    analyze_autocorrelation,    # ACF/PACF
    analyze_volatility,         # ARCH effects
    analyze_distribution,       # Distribution tests
)
```

### Signal Analysis

```python
from ml4t.diagnostic.evaluation import (
    SignalAnalysis,             # Single signal analysis
    MultiSignalAnalysis,        # Multi-signal comparison
    compute_ic_series,          # IC time series
)
```

### Trade Analysis

```python
from ml4t.diagnostic.evaluation import (
    TradeAnalysis,              # Trade statistics
    TradeShapAnalyzer,          # SHAP-based diagnostics
)
```

### Portfolio Analysis

```python
from ml4t.diagnostic.evaluation import (
    PortfolioAnalysis,          # Portfolio metrics
)
```

### Statistical Validation

```python
from ml4t.diagnostic.evaluation import stats
from ml4t.diagnostic.splitters import (
    PurgedWalkForwardCV,        # Walk-forward with purging
    CombinatorialPurgedCV,      # CPCV
)
```

### Binary Metrics

```python
from ml4t.diagnostic.evaluation import (
    binary_classification_report,
    precision, recall, lift, coverage,
    wilson_score_interval,
    find_optimal_threshold,
)
```

## Integration with ML4T Libraries

```python
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.backtest import Engine
from ml4t.diagnostic.evaluation import TradeAnalysis, PortfolioAnalysis

# Complete workflow
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")
features = compute_features(data, ["rsi", "macd", "atr"])
# ... train model ...
result = engine.run()

# Analyze trades
trade_analysis = TradeAnalysis(result.trades)
print(f"Win rate: {trade_analysis.win_rate:.1%}")

# Portfolio analysis
portfolio = PortfolioAnalysis(result.returns)
portfolio.generate_tear_sheet()
```

## Ecosystem

- **ml4t-data**: Market data acquisition and storage
- **ml4t-engineer**: Feature engineering and indicators
- **ml4t-diagnostic**: Statistical validation and evaluation (this library)
- **ml4t-backtest**: Event-driven backtesting
- **ml4t-live**: Live trading platform

## Testing

```bash
# Run tests (4,887 tests)
uv run pytest tests/ -q -n auto

# Type checking
uv run ty check

# Linting
uv run ruff check src/
```

## Development

```bash
git clone https://github.com/applied-ai/ml4t-diagnostic.git
cd ml4t-diagnostic

# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -q -n auto

# Type checking
uv run ty check
```

## Optional Dependencies

| Group | Packages | Features |
|-------|----------|----------|
| `[ml]` | shap, lightgbm, xgboost | SHAP importance, tree explainers |
| `[viz]` | plotly, streamlit | Interactive visualizations |
| `[deep]` | tensorflow | Deep learning explainers |
| `[gpu]` | cupy | GPU acceleration |
| `[all]` | All above | Everything |

```python
# Check what's available
from ml4t.diagnostic.utils import get_dependency_summary
print(get_dependency_summary())
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
- Bailey, D., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
- Bailey, D., et al. (2014). "The Deflated Sharpe Ratio."

See [docs/REFERENCES.md](docs/REFERENCES.md) for complete academic citations.

## Current Status

**Version**: 0.1.0a1 (Alpha)

The library is functional and tested (4,887 tests) but still in alpha. API may change before 1.0.

## License

MIT License - see [LICENSE](LICENSE) for details.
