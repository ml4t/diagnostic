# Migrating an Alphalens or Pyfolio Task

Diagnostic is not a drop-in replacement for either package. Choose a task below,
convert its input shape, and compare result definitions before comparing numbers.
The examples use synthetic data and run with the base `ml4t-diagnostic` package.

## Alphalens factor analysis

[Alphalens](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/README.rst) accepts
a factor Series indexed by date and asset plus a wide price DataFrame, then
`get_clean_factor_and_forward_returns()` constructs forward returns and
quantiles. Its full tear sheet includes returns, IC, turnover, and grouped
analysis. Diagnostic's `analyze_signal()` takes two long tables and returns
IC, quantile returns, spread, and turnover metrics. It recomputes forward
returns from prices, so an Alphalens *cleaned* factor table by itself is
not an input substitute. Preserve the original point-in-time prices.
See the [Alphalens input contract](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/alphalens/utils.py)
and the [Diagnostic signal API](../api/index.md#signal-analysis).

| Alphalens input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Factor Series with `(date, asset)` index | `analyze_signal(factor=...)` | Reset the index to columns `date`, `asset`, `factor`; keep one value per pair. |
| Wide asset-price table | `analyze_signal(prices=...)` | Stack to `date`, `asset`, `price` rows, including enough future dates for the largest horizon. |
| IC, quantile returns, and spread | `SignalResult.ic`, `.quantile_returns`, `.spread` | Specify `periods`, `quantiles`, and `ic_method`; check timestamp alignment and compare each statistic's definition. |
| Group-neutral or zero-aware bucketing and the complete Alphalens tear sheet | No one-call `analyze_signal` equivalent | Preprocess groups or buckets separately if needed; do not treat Diagnostic's default quantiles or reports as equivalent. |

Here is the shape conversion with data whose factor predicts the next price
change. Replace the synthetic source objects with your Alphalens-shaped inputs:

```python
import numpy as np
import pandas as pd

from ml4t.diagnostic import analyze_signal

dates = pd.date_range("2025-01-01", periods=20, freq="D")
assets = [f"asset_{i:02d}" for i in range(12)]
rng = np.random.default_rng(7)
scores = rng.normal(size=(len(dates), len(assets)))
factor_index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
factor_series = pd.Series(scores.ravel(), index=factor_index, name="factor")
price_values = np.full(scores.shape, 100.0)
for day in range(1, len(dates)):
    price_values[day] = price_values[day - 1] * (
        1 + 0.002 * scores[day - 1] + rng.normal(scale=0.005, size=len(assets))
    )
wide_prices = pd.DataFrame(price_values, index=dates, columns=assets)

factor_long = factor_series.reset_index()
price_long = (
    wide_prices.rename_axis("date")
    .stack()
    .rename("price")
    .rename_axis(index=["date", "asset"])
    .reset_index()
)
signal = analyze_signal(factor_long, price_long, periods=(1,), min_assets=10)
assert signal.n_assets == len(assets)
assert signal.ic["1D"] > 0
print(f"One-day IC: {signal.ic['1D']:.3f}")
```

The assertion establishes the expected direction on these synthetic data;
it is not a significance test for a real factor. Use the
[signal quickstart](../getting-started/quickstart.md) to interpret IC and spread.
Alphalens also supports grouped and group-adjusted IC, configurable
zero-aware bucketing, and its own complete factor tear sheet. Those options
are not implied by a Diagnostic signal result.

Factor *construction* is a separate task. If needed,
[ML4T Engineer's feature guide](https://github.com/ml4t/engineer/blob/145ccafd2ca36dc61f976cdbeaaf147af21e7c2e/docs/user-guide/features.md)
documents `compute_features()`; Diagnostic does not require Engineer, and
users can supply factors produced elsewhere. Engineer owns feature
preparation, while Diagnostic evaluates the resulting factor.

## Pyfolio portfolio and performance analysis

[Pyfolio's `create_full_tear_sheet()`](https://github.com/quantopian/pyfolio/blob/4b901f6d73aa02ceb6d04b7d83502e5c6f2e81aa/pyfolio/tears.py)
takes noncumulative daily returns, with optional positions, transactions,
benchmark returns, and other inputs. Diagnostic separates numeric analysis
from HTML rendering. It does not accept Pyfolio's tear-sheet call or a
Zipline backtest object directly.

| Pyfolio input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Daily noncumulative return Series | `PortfolioAnalysis(returns=..., dates=...)` | Pass decimal return values and matching dates; set `periods_per_year` and risk-free convention explicitly. |
| Benchmark return Series | `PortfolioAnalysis(benchmark=...)` | Reindex to the strategy dates before passing values; verify missing dates and currency. |
| Daily position values and fill transactions | `PortfolioAnalysis(positions=..., transactions=...)` | Positions may be wide by asset or long `date`, `asset`, `value`; transactions need `date`, `asset`, `quantity`, `price`, `commission`. Reconcile cash, signs, and fees. |
| Performance HTML | `generate_backtest_tearsheet(returns=..., metrics=...)` | Install the `viz` extra; supply a return array and any precomputed metrics. This is not a Pyfolio report template. |
| A completed `ml4t-backtest` result | `generate_tearsheet_from_result(result)` | Only this bridge takes an `ml4t-backtest` result; it does not ingest a Pyfolio or Zipline result. |
| Trade-level ranking | `TradeAnalysis` on `TradeRecord` objects | Pair fills into completed trades and calculate PnL and duration first. Raw Pyfolio transaction rows are not completed trades. |

The minimal return conversion can be checked without visualization packages:

```python
import numpy as np
import pandas as pd

from ml4t.diagnostic.evaluation import PortfolioAnalysis

daily_returns = pd.Series(
    np.random.default_rng(42).normal(0.0005, 0.01, size=252),
    index=pd.bdate_range("2025-01-01", periods=252),
)
portfolio = PortfolioAnalysis(
    returns=daily_returns.to_numpy(),
    dates=daily_returns.index.to_numpy(),
    periods_per_year=252,
)
summary = portfolio.compute_summary_stats()
assert np.isfinite(summary.sharpe_ratio)
print(f"Portfolio Sharpe: {summary.sharpe_ratio:.2f}")
```

Use the [portfolio workflow](workflows.md) and
[backtest report guide](backtest-tearsheets.md) for the supported output checks.
Pyfolio's full sheet can additionally include return cones, significant
events, round trips reconstructed from fills, capacity analysis using
market data, and optional factor-attribution panels. Diagnostic has related
statistics and reports but no one-call equivalence for that full collection.
Keep Pyfolio or implement the specific analysis you need if those sections
are part of your acceptance criteria. Compare annualization, benchmark
alignment, cash, and cost conventions before comparing metrics.
