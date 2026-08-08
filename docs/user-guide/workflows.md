# End-to-End Validation Workflow

This workflow starts with synthetic factor and price data, measures signal
quality, creates purged validation folds, corrects a strategy comparison for
multiple testing, and reports portfolio statistics. It runs without external
files or network access.

## Run the workflow

```python
import numpy as np
import polars as pl

from ml4t.diagnostic import analyze_signal
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio
from ml4t.diagnostic.splitters import WalkForwardCV

rng = np.random.default_rng(42)

# Build a cross-sectional signal dataset.
dates = pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 3, 31), eager=True)[:60]
assets = [f"asset_{index:02d}" for index in range(24)]
factor_rows = []
price_rows = []
prices = np.full(len(assets), 100.0)
for date in dates:
    scores = rng.normal(size=len(assets))
    factor_rows.extend(
        {"date": date, "asset": asset, "factor": score}
        for asset, score in zip(assets, scores, strict=True)
    )
    price_rows.extend(
        {"date": date, "asset": asset, "price": price}
        for asset, price in zip(assets, prices, strict=True)
    )
    prices *= 1 + 0.002 * scores + rng.normal(scale=0.006, size=len(assets))

signal = analyze_signal(
    factor=pl.DataFrame(factor_rows),
    prices=pl.DataFrame(price_rows),
    periods=(1, 5),
)
assert signal.ic["1D"] > 0.1
print(f"Signal IC: {signal.ic['1D']:.4f}")
print(f"Signal spread: {signal.spread['1D']:.2%}")

# Build leakage-aware chronological folds for the model stage.
features = rng.normal(size=(300, 6))
cv = WalkForwardCV(
    n_splits=4,
    train_size=120,
    test_size=40,
    label_horizon=5,
    expanding=False,
)
folds = list(cv.split(features))
assert len(folds) == 4
assert all(train.max() < test.min() for train, test in folds)
print(f"Validation folds: {len(folds)}")

# Correct selection among four strategy variants.
strategy_returns = rng.normal(
    loc=[0.0002, 0.0005, 0.0001, 0.0003],
    scale=0.01,
    size=(504, 4),
)
dsr = deflated_sharpe_ratio(
    strategy_returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)
print(f"DSR probability: {dsr.probability:.3f}")
print(f"Effective trials: {dsr.n_trials_effective:.2f}")

# Report the selected strategy's portfolio statistics.
portfolio = PortfolioAnalysis(strategy_returns[:, 1])
metrics = portfolio.compute_summary_stats()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Sortino: {metrics.sortino_ratio:.2f}")
print(f"Maximum drawdown: {metrics.max_drawdown:.2%}")
```

## Replace the synthetic inputs

1. Replace the factor and price tables with one row per date and asset.
2. Fit and score the model inside each yielded cross-validation split.
3. Store the return series for every strategy variant considered.
4. Pass the full return matrix to `deflated_sharpe_ratio`.
5. Generate a report only after the statistical checks use out-of-sample data.

Use [CPCV](cross-validation.md) when you need a distribution across many test
group combinations. Use [backtest tearsheets](backtest-tearsheets.md) to render
the final metrics and return series as HTML.
