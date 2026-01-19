# Four-Tier Validation Framework

ML4T Diagnostic implements a comprehensive Four-Tier Validation Framework to prevent overfitting, data leakage, and false discoveries in quantitative trading strategies.

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
result = fd.analyze(features_df, target, dates)

# Check feature quality
print(result.stationarity_summary)
print(result.importance_ranking)
print(result.drift_warnings)
```

## Tier 2: Model Diagnostics

**When**: During and after model training
**Goal**: Validate model predictions and calibration

### Key Analyses

| Analysis | Purpose |
|----------|---------|
| **Prediction Distribution** | Check for overconfidence |
| **Calibration Curves** | Predicted vs actual probabilities |
| **Stability Analysis** | Model consistency over time |
| **Autocorrelation** | ACF/PACF of residuals |
| **Volatility Clustering** | GARCH effects in errors |

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
    num_trials=100,  # Number of backtests tried
    frequency='daily'
)

print(f"DSR: {result.dsr:.4f}")
print(f"p-value: {result.pvalue:.4f}")
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

## Integration

All four tiers work together:

```python
from ml4t.diagnostic import Evaluator

# Run complete validation pipeline
evaluator = Evaluator()

# Tier 1: Feature analysis
feature_result = evaluator.analyze_features(features_df, target)

# Tier 2: Model diagnostics
model_result = evaluator.analyze_model(model, X_test, y_test)

# Tier 3: Backtest validation
backtest_result = evaluator.validate_backtest(
    returns=strategy_returns,
    num_trials=50
)

# Tier 4: Portfolio monitoring
portfolio_result = evaluator.analyze_portfolio(positions, prices)
```
