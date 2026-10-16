# Migrating an Alphalens or Pyfolio Task

Diagnostic is not a drop-in replacement for either package. Choose a task below,
convert its input shape, and compare result definitions before comparing numbers.
The examples use synthetic data and run with the base `ml4t-diagnostic` package.
For selected performance metrics, see the separate
[Empyrical migration assessment](empyrical-migration.md).

One reason to evaluate Diagnostic beyond these matching values is its
[purged cross-validation workflow](cross-validation.md), which combines
chronological folds with multiple-trial Sharpe assessment. For portfolio
reporting, its [backtest bridge](backtest-tearsheets.md) also accepts a
completed `ml4t-backtest` result. These are separate capabilities, not
claims that the legacy tear sheets or factor panels are reproduced.

## Run the shared-input comparison

The package-only examples below show each Diagnostic input shape. To compare the
published legacy and Diagnostic APIs on the same deterministic inputs, run this
script from a checkout of this repository on Python 3.12:

```bash
uv run --no-project --python 3.12 \
  --with 'ml4t-diagnostic==0.1.8' \
  --with 'pyfolio-reloaded==0.9.9' \
  --with 'alphalens-reloaded==0.4.6' \
  --with 'empyrical-reloaded==0.5.12' \
  python examples/legacy_migration_comparison.py --task all
```

The comparison script at `examples/legacy_migration_comparison.py` uses the
published packages, not the source checkout. It asserts the shared
Sharpe, drawdown, and IC results and prints other measured values. The figures
below are from that command with the stated versions; they are not performance
benchmarks or a promise that every legacy output matches.

## Alphalens factor analysis

[Alphalens](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/README.rst) accepts
a factor Series indexed by date and asset plus a wide price DataFrame, then
`get_clean_factor_and_forward_returns()` constructs forward returns and
quantiles. Its full tear sheet includes returns, IC, turnover, and grouped
analysis. Diagnostic's `analyze_signal()` takes two long tables and returns
IC, quantile returns, spread, and turnover metrics. It recomputes forward
returns from prices. Current unreleased source also provides
`analyze_prepared_signal()` for an existing cleaned factor table, retaining its
supplied forward returns and bucket assignments without fabricated prices.
See the [Alphalens input contract](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/alphalens/utils.py)
and the [Diagnostic signal API](../api/index.md#signal-analysis).

<div class="migration-table" markdown="1">

| Alphalens input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Factor Series with `(date, asset)` index | `analyze_signal(factor=...)` | Reset the index to columns `date`, `asset`, `factor`; keep one value per pair. |
| Wide asset-price table | `analyze_signal(prices=...)` | Stack to `date`, `asset`, `price` rows, including enough future dates for the largest horizon. |
| Already-cleaned factor table | `analyze_prepared_signal(data, ...)` | Accepts the ordinary pandas `(date, asset)` index, `factor_quantile` and `1D`/`5D` columns, or native flat columns. Preserves supplied buckets; rejects duplicate keys and incomplete requested data rather than cleaning it. Unreleased after 0.1.9. |
| IC, quantile returns, and spread | `SignalResult.ic`, `.quantile_returns`, `.spread` | Specify `periods`, `quantiles`, and `ic_method`; check timestamp alignment and compare each statistic's definition. |
| Quantile turnover | `SignalResult.turnover` | From 0.1.9, each horizon key uses that lag in sorted observed factor dates. The direct `compute_turnover(..., lag=...)` helper defaults to one observation; neither route establishes calendar-aware turnover. |
| Factor HTML report | `SignalResult.to_tear_sheet()` and `SignalDashboard` | This bridge populates IC and quantile summaries, but leaves `turnover_analysis` and quantile `cumulative_returns` unavailable. Existing plotting components do not make these paths part of this workflow. |
| Group-neutral or zero-aware bucketing and the complete Alphalens tear sheet | No complete equivalent | The prepared route can retain supplied buckets. It does not implement group-adjusted IC/returns, group-neutral portfolios or the complete tear sheet. |

</div>

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

For exact integer-count qcut membership, unreleased source after 0.1.9 accepts
`quantile_method="qcut"` in both `prepare_data` and `analyze_signal`. The existing
backend uses pandas qcut for boundary semantics while retaining Polars grouping,
row order and categorical output. Twelve
distinct ordered assets and five buckets have counts `3,2,2,2,3`; legacy
`quantile` rank buckets instead have counts `3,2,3,2,2`. The default remains the
legacy rank method, and `uniform` remains the legacy epsilon-adjusted width
method. These are not pandas qcut/cut aliases. Unknown method names now raise
instead of silently selecting uniform bins. The qcut option requires a positive
integer count and finite factors, and rejects duplicate edges rather than
splitting ties. It is not a cleaner: explicit probability/cut edges, zero-aware
group buckets, complete-case filtering and loss provenance still require
separate implementation. Other native signal statistics retain their stated
aggregation conventions.

Qcut boundary rounding depends on pandas/NumPy versions. For values `[0,1,2,3]`
and three buckets, the tested pandas 2.3.3/NumPy 2.5.4 environment assigns
`[1,2,3,3]`; pandas 3.0.5 corrects quantile rounding and assigns `[1,1,2,3]`.
Both retain right-closed bins, but the floating-point edge differs. Compare
Alphalens and Diagnostic in the same pinned environment. Qcut analysis records
the method and pandas/NumPy versions in `result.preparation_metadata` and JSON.
Do not replace this path with Polars 2's floor-selected quantile bins: tested
membership and duplicate-endpoint rejection differ.

For an already-cleaned panel, current source can retain its sample and supplied
buckets directly. This example does not need Alphalens at runtime:

```python
import numpy as np
import pandas as pd

from ml4t.diagnostic import analyze_prepared_signal

index = pd.MultiIndex.from_product(
    [pd.bdate_range("2025-01-02", periods=3), [f"A{i:02}" for i in range(12)]],
    names=["date", "asset"],
)
prepared = pd.DataFrame(
    {
        "factor": np.tile(np.arange(12, dtype=float), 3),
        "factor_quantile": np.tile(np.repeat([1, 2, 3], 4), 3),
        "1D": np.random.default_rng(120).normal(0.001, 0.01, 36),
    },
    index=index,
)
result = analyze_prepared_signal(prepared, periods=(1,), quantiles=3, min_assets=2)
assert result.prepared_data.height == 36
assert result.preparation_metadata["excluded_rows"] == 0
print(f"Prepared observations retained: {result.prepared_data.height}")
```

`result.prepared_data` returns an independent canonical frame. JSON round-trips
store it as a base64 Arrow IPC payload to preserve schema, nanoseconds and
timezone. The strict boundary rejects duplicate keys, incomplete/nonfinite
requested values and invalid buckets. It does not replace the full Alphalens
cleaner or produce a loss audit for excluded rows: it excludes no rows.

Precomputed `1D`/`5D` labels are selectors here, not newly calculated durations
or evidence of exchange-session/bar completeness. Native pooled quantile
statistics and two-sample spread inference still differ from Alphalens'
equal-date aggregation and propagated standard errors. Turnover is disabled by
default on this prepared route; enabling it uses the existing observed-date
lag method. Group-adjusted/neutral calculations and complete reports remain
unsupported by this route. No full-replacement claim follows from retaining a
prepared panel.

Timestamp grouping can also differ. On a prepared sample containing three
distinct nanosecond instants, Alphalens 0.4.6 regularizes IC to its inferred
`CustomBusinessDay` frequency and retains only the first instant. Diagnostic
retains all three supplied timestamp groups. Its correlations match independent
per-timestamp SciPy calculations, not a three-row Alphalens output. This does
not establish calendar or schedule equivalence.

In the shared-input comparison, 600 factor rows across 50 business dates and
12 assets use one-day forward returns from the same wide price table. Both calls
use five quantiles, no outlier filter, and ungrouped Spearman IC. Alphalens
kept all 600 rows; its mean IC and Diagnostic's `SignalResult.ic["1D"]` were
both `0.318881`. Diagnostic also returned a top-minus-bottom quantile spread
of `0.004636`. The agreement is for this ungrouped input and definition.

### Quantiles, turnover, and report boundaries

Quantile returns need not agree on other panels. In a separate shared-input
comparison with Alphalens 0.4.6 and Diagnostic 0.1.8, 40 assets on every date
produced matching raw 1D/5D quantile means. With 12 assets in five quantiles,
the packages allocated the remainder differently. With a changing 20/40-asset
universe, Diagnostic pooled observations within each quantile, while Alphalens
averaged per-date quantile means. Choose the bucketing and date-weighting
conventions explicitly before comparing those outputs. A raw top-minus-bottom
mean is also not a 50/50 unit-gross long-short portfolio return.

On 20 assets whose four-asset quantiles rotate by one asset per observation,
Alphalens turnover was `0.25` at lag 1 and `1.0` at lag 5. Diagnostic 0.1.8
incorrectly repeated `0.25` under both `1D` and `5D`; 0.1.9 corrects the
horizon-labelled result to use the requested lag. It retains the normalization
`1 - overlap / max(previous_count, current_count)`, which can differ from
Alphalens when membership sizes change. Lags count observed factor dates,
not elapsed calendar days; insufficient history returns NaN.

`SignalResult.to_tear_sheet()` supplies IC and quantile analysis to the HTML
dashboard. It does not supply turnover analysis or cumulative quantile-return
paths. Do not infer those panels from a successful HTML render or the presence
of related plotting functions. Keep Alphalens when those report sections,
factor-weighted portfolios, or event-aligned cumulative paths are required.

Alphalens also supports grouped and group-adjusted IC, configurable
zero-aware bucketing, and its own complete factor tear sheet. Those options
are not implied by a Diagnostic signal result. In particular, do not interpret
Diagnostic's default global quantiles as Alphalens `group_adjust=True` or
`binning_by_group=True`. Keep the legacy path when those panels or conventions
are required.

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

<div class="migration-table" markdown="1">

| Pyfolio input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Daily noncumulative return Series | `PortfolioAnalysis(returns=..., dates=...)` | A pandas DatetimeIndex is retained automatically. Explicit dates must agree with it; set `periods_per_year` and risk-free convention explicitly. |
| Benchmark return Series | `PortfolioAnalysis(benchmark=...)` | A dated Series is aligned to every strategy observation, with extra benchmark dates excluded. Missing/duplicate keys and timezone mismatches raise; verify currency separately. Undated arrays must have equal lengths. |
| Daily position values and fill transactions | `PortfolioAnalysis(positions=..., transactions=...)` | Positions may be wide by asset or long `date`, `asset`, `value`; transactions need `date`, `asset`, `quantity`, `price`, `commission`. Reconcile cash, signs, and fees. |
| Performance HTML | `generate_backtest_tearsheet(returns=..., metrics=...)` | Install the `viz` extra; supply a return array and any precomputed metrics. This is not a Pyfolio report template. |
| A completed `ml4t-backtest` result | `generate_tearsheet_from_result(result)` | Only this bridge takes an `ml4t-backtest` result; it does not ingest a Pyfolio or Zipline result. |
| Trade-level ranking | `TradeAnalysis` on `TradeRecord` objects | Pair fills into completed trades and calculate PnL and duration first. Raw Pyfolio transaction rows are not completed trades. |

</div>

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
    returns=daily_returns,
    periods_per_year=252,
)
summary = portfolio.compute_summary_stats()
assert np.isfinite(summary.sharpe_ratio)
print(f"Portfolio Sharpe: {summary.sharpe_ratio:.2f}")
```

Current source validates nonempty one-dimensional real simple returns, finite
values no lower than -1, a positive integer `periods_per_year`, and a finite
annual `risk_free > -1`. Missing observations raise instead of becoming cash
returns. Zero-return observations remain in the sample. Inputs and returned
arrays/results are independent snapshots, so caller mutations cannot change
cached metrics.

NumPy and Polars return arrays without dates still support numerical analysis
and observation-indexed charts. They no longer acquire fictional year-2000
dates. Monthly/annual tables and complete calendar dashboards require actual,
unique, increasing dates. Date preservation does not infer an exchange calendar
or convert event timestamps to session labels. A raw-return backtest tearsheet
without dated equity data marks calendar panels unavailable while retaining
observation-based analysis. These boundary corrections are
unreleased source changes after 0.1.9; earlier published behavior is not a
supported missing-data or positional-alignment policy.

Use the [portfolio workflow](workflows.md) and
[backtest report guide](backtest-tearsheets.md) for the supported output checks.
The shared-input comparison passes 252 aligned business-day returns and benchmark
returns to Pyfolio `perf_stats()` and Diagnostic `PortfolioAnalysis`, with zero
risk-free rate and 252 periods per year. Return, volatility, drawdown, and
alpha values use decimal-return units; Sharpe ratio and beta are dimensionless:

<div class="migration-table" markdown="1">

| Metric | Pyfolio 0.9.9 | Diagnostic 0.1.8 |
|---|---:|---:|
| Annual return | 0.079730 | 0.079730 |
| Annual volatility | 0.150155 | 0.150155 |
| Sharpe ratio | 0.585568 | 0.585568 |
| Maximum drawdown | -0.110216 | -0.110216 |
| Alpha | 0.100369 | 0.095663 |
| Beta | 0.795601 | 0.795601 |

</div>

The alpha difference is expected under the two implementations: Pyfolio's
Empyrical calculation compounds mean periodic alpha, while Diagnostic
multiplies its fitted daily intercept by 252. Do not compare annualized alpha
without choosing a convention. The example has no positions or transactions;
Pyfolio's optional Zipline position multipliers were unavailable and did not
affect these returns-based values.

For initial losses, the [drawdown correction](https://github.com/ml4t/diagnostic/issues/118)
uses initial capital as the first high-water mark. In an executed Pyfolio 0.9.9
comparison, returns `[-0.2, 0.1, 0.05]` gave its episode table 0% drawdown even
though Empyrical's scalar maximum drawdown was -20%. The independent wealth path
`100, 80, 88, 92.4` requires -20%; current Diagnostic source uses that result
instead of reproducing the episode error. A pre-first-return peak has no invented
date. Native episode thresholds and observation-count durations still differ
from Pyfolio's top-only selection and business-date recovery durations. This
focused correction does not qualify every episode option or the full report.

Pyfolio's full sheet can additionally include return cones, significant
events, round trips reconstructed from fills, capacity analysis using
market data, and optional factor-attribution panels. Diagnostic has related
statistics and reports but no one-call equivalence for that full collection.
Keep Pyfolio or implement the specific analysis you need if those sections
are part of your acceptance criteria. Compare annualization, benchmark
alignment, cash, and cost conventions before comparing metrics.
