# ML4T Diagnostic

`ml4t.diagnostic` tests signals, models, and backtest results for leakage,
overfitting, and multiple-testing bias. It is the statistical validation
library in the ML4T package suite.

## Start with an executable check

This example asks whether the best of three strategy variants remains
significant after accounting for selection.

```python
import numpy as np

from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

rng = np.random.default_rng(42)
returns = rng.normal(
    loc=[0.0002, 0.0005, 0.0001],
    scale=0.01,
    size=(252, 3),
)

result = deflated_sharpe_ratio(
    returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)

print(f"Probability of skill: {result.probability:.3f}")
print(f"Expected maximum Sharpe from noise: {result.expected_max_sharpe:.3f}")
print(f"Significant: {result.is_significant}")
```

## Choose the guide for your task

| Task | Guide |
|------|-------|
| Analyze cross-sectional predictions | [Quickstart](getting-started/quickstart.md) |
| Prevent leakage in time-series validation | [Cross-validation](user-guide/cross-validation.md) |
| Correct Sharpe and IC significance | [Statistical tests](user-guide/statistical-tests.md) |
| Diagnose feature quality | [Feature diagnostics](user-guide/feature-diagnostics.md) |
| Select features systematically | [Feature selection](user-guide/feature-selection.md) |
| Inspect trades and recurring losses | [Trade analysis](user-guide/trade-analysis.md) |
| Generate HTML backtest reports | [Backtest tearsheets](user-guide/backtest-tearsheets.md) |

## Validation areas

The package separates four stages of analysis:

1. Feature diagnostics test stationarity, autocorrelation, distribution, and volatility.
2. Signal analysis measures information coefficient, quantile returns, spread, and turnover.
3. Backtest analysis applies DSR, PBO, RAS, FDR control, and trade-level diagnostics.
4. Portfolio analysis measures returns, drawdowns, risk, and factor attribution.

The [API reference](api/index.md) lists exact public imports. The
[book guide](book-guide/index.md) maps the library to *Machine Learning for
Trading, Third Edition*.

## Install

```bash
pip install ml4t-diagnostic
```

See the [installation guide](getting-started/installation.md) for optional
visualization, dashboard, backtest, and data integrations.
