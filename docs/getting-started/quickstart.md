# Quickstart

This tutorial analyzes a synthetic cross-sectional signal and then corrects a
strategy comparison for multiple testing. It runs without external data.

## Analyze a signal

Use the base package with Polars and NumPy. Inputs need one row per date
and asset; each factor observation must precede the price change it predicts.

Create 40 daily observations for 20 assets. The synthetic factor affects the
next price change, so the analysis has a known relation to detect.

```python
import numpy as np
import polars as pl

from ml4t.diagnostic import analyze_signal

rng = np.random.default_rng(7)
dates = pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 2, 28), eager=True)[:40]
assets = [f"asset_{index:02d}" for index in range(20)]

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
    prices *= 1 + 0.002 * scores + rng.normal(scale=0.005, size=len(assets))

result = analyze_signal(
    factor=pl.DataFrame(factor_rows),
    prices=pl.DataFrame(price_rows),
    periods=(1, 5),
)

assert result.ic["1D"] > 0.1

print(f"1-day IC: {result.ic['1D']:.4f}")
print(f"1-day IC t-stat: {result.ic_t_stat['1D']:.2f}")
print(f"1-day top-minus-bottom spread: {result.spread['1D']:.2%}")
```

`analyze_signal` expects one row per date and asset. The factor table needs
`date`, `asset`, and `factor` columns. The price table needs `date`, `asset`,
and `price` columns.

The assertion checks that the synthetic one-day signal has positive
cross-sectional information coefficient (IC). On real data, inspect IC
alongside its t-statistic and quantile spread. A positive value alone is not
evidence of a deployable strategy. See the [signal API](../api/index.md#signal-analysis)
and the book's [direct Diagnostic signal checks](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/08_financial_features/06_robustness_sensitivity.ipynb).

## Correct for multiple testing

Use Deflated Sharpe Ratio when you selected the best result from several
strategy variants. Passing a two-dimensional array treats each column as one
tested strategy.

```python
import numpy as np

from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

rng = np.random.default_rng(42)
strategy_returns = rng.normal(
    loc=[0.0003, 0.0005, 0.0002],
    scale=0.01,
    size=(252, 3),
)

dsr = deflated_sharpe_ratio(
    strategy_returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)

assert dsr.n_trials_raw == 3
assert 0 <= dsr.probability <= 1
print(f"Observed Sharpe: {dsr.sharpe_ratio_annualized:.2f}")
print(f"Probability after correction: {dsr.probability:.3f}")
print(f"Effective trials: {dsr.n_trials_effective:.2f}")
print(f"Significant: {dsr.is_significant}")
```

The probability reflects the supplied tested variants, not unrecorded
variants from earlier research. The [statistical-tests guide](../user-guide/statistical-tests.md)
explains how to retain the full trial history.

## Continue with a focused guide

- [Cross-validation](../user-guide/cross-validation.md) covers purged
  walk-forward validation and combinatorial purged cross-validation.
- [Statistical tests](../user-guide/statistical-tests.md) covers DSR, HAC IC,
  false discovery rate control, and PBO.
- [Backtest tearsheets](../user-guide/backtest-tearsheets.md) creates an HTML
  report from synthetic trades and returns.
- [Trade analysis](../user-guide/trade-analysis.md) identifies recurring
  losses in normalized trade records.
- [API reference](../api/index.md) lists the supported import surfaces.
- [Book Guide](../book-guide/index.md) maps direct Diagnostic calls to pinned
  companion notebooks.
