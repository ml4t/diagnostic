# Four-Tier Validation Framework

ML4T Diagnostic implements a comprehensive Four-Tier Validation Framework to prevent overfitting, data leakage, and false discoveries in quantitative trading strategies.

!!! info "See it in the book"
    Ch06 covers leakage-aware cross-validation, Ch07 covers statistical inference,
    Ch16-Ch17 cover backtest and portfolio diagnostics, and Ch19 covers trade and
    risk analysis. Use the [Book Guide](../book-guide/index.md) for the exact chapter
    and case-study map.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FOUR-TIER VALIDATION                         │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1: Feature Analysis     │  Pre-modeling diagnostics       │
│  Tier 2: Model Diagnostics    │  During/after modeling          │
│  Tier 3: Backtest Analysis    │  Post-modeling validation       │
│  Tier 4: Portfolio Analysis   │  Production monitoring          │
└─────────────────────────────────────────────────────────────────┘
```

## Tier 1: Feature Analysis

**When**: Before training any model
**Goal**: Ensure features have predictive power and are suitable for modeling

### Key Analyses

| Analysis | Purpose |
|----------|---------|
| **Information Coefficient (IC)** | Correlation between features and future returns |
| **IC Decay** | How quickly predictive power decays over time |
| **Feature Importance** | MDI, PFI, MDA, SHAP rankings |
| **Feature Interactions** | H-statistic for detecting interactions |
| **Stationarity Tests** | ADF, KPSS, Phillips-Perron |
| **Distribution Analysis** | Jarque-Bera, tail risk |
| **Drift Detection** | PSI, Wasserstein distance |

### Example

```python
from ml4t.diagnostic.evaluation import FeatureDiagnostics

fd = FeatureDiagnostics()
result = fd.run_diagnostics(features_df["feature_1"], name="feature_1")

# Check feature quality
print(result.summary())
print(result.health_score)
```

## Tier 2: Signal And Model Diagnostics

**When**: During and after model training
**Goal**: Validate predictions, factor behavior, and stability

### Key Analyses

| Analysis | Purpose |
|----------|---------|
| **Signal IC** | Cross-sectional predictive power |
| **Quantile Spreads** | Economic monotonicity and long-short separation |
| **Turnover / Half-Life** | Stability and trading cost pressure |
| **Feature Importance** | MDI, PFI, SHAP, interaction structure |
| **Event / Barrier Analysis** | Targeted model-behavior diagnostics |

### Example

```python
from ml4t.diagnostic import analyze_signal

signal_result = analyze_signal(
    factor=factor_df,
    prices=prices_df,
    periods=(1, 5, 21),
)

print(signal_result.summary())
```

## Tier 3: Backtest Analysis

**When**: After backtesting
**Goal**: Validate performance is not due to overfitting

### Key Analyses

| Analysis | Purpose |
|----------|---------|
| **Deflated Sharpe Ratio (DSR)** | Adjust for multiple testing |
| **Probability of Backtest Overfitting (PBO)** | Likelihood of overfitting |
| **Rademacher Anti-Serum (RAS)** | Complexity-adjusted returns |
| **Minimum Track Record Length** | Required history for significance |
| **Trade Error Analysis** | SHAP-based failure patterns |

### Example

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

result = deflated_sharpe_ratio(
    returns=strategy_returns,
    n_trials=100,
    frequency="daily",
)

print(f"Probability of skill: {result.probability:.1%}")
print(f"Deflated Sharpe: {result.deflated_sharpe:.4f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Is significant: {result.is_significant}")
```

## Tier 4: Portfolio Analysis

**When**: In production
**Goal**: Monitor live performance and risk

### Key Analyses

| Analysis | Purpose |
|----------|---------|
| **Rolling Metrics** | Sharpe, Sortino, Calmar over time |
| **Drawdown Analysis** | Max drawdown, recovery time |
| **Risk Attribution** | Factor exposures |
| **Performance Attribution** | By sector, region, etc. |

### Example

```python
from ml4t.diagnostic.evaluation import PortfolioAnalysis

analysis = PortfolioAnalysis(returns=strategy_returns, benchmark=benchmark_returns)
metrics = analysis.compute_summary_stats()

print(metrics.sharpe_ratio)
print(metrics.max_drawdown)
```

## How This Relates To `Evaluator`

The library-level research workflow above is four-tiered. The `Evaluator` class is a
separate, lower-level validation orchestrator that implements a three-tier statistical
evaluation framework around splitters, metrics, and tests.

Use `Evaluator` when you want a configurable validation engine for model evaluation:

```python
from ml4t.diagnostic import Evaluator

evaluator = Evaluator(tier=2)
result = evaluator.evaluate(X, y, model)

print(result.summary())
```

Use the higher-level workflows like `analyze_signal()`, `FeatureDiagnostics`,
`ValidatedCrossValidation`, and `PortfolioAnalysis` when you want task-specific APIs
instead of a generic evaluation engine.
