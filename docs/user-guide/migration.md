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
For unbucketed factors and supplied outcomes, `prepare_signal_panel()` selects
one common sample and records exclusions before assigning buckets.
See the [Alphalens input contract](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/alphalens/utils.py)
and the [Diagnostic signal API](../api/index.md#signal-analysis).

<div class="migration-table" markdown="1">

| Alphalens input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Factor Series with `(date, asset)` index | `analyze_signal(factor=...)` | Reset the index to columns `date`, `asset`, `factor`; keep one value per pair. |
| Wide asset-price table | `analyze_signal(prices=...)` | Stack to `date`, `asset`, `price` rows, including enough future dates for the largest horizon. |
| Already-cleaned factor table | `analyze_prepared_signal(data, ...)` | Accepts the ordinary pandas `(date, asset)` index, `factor_quantile` and `1D`/`5D` columns, or native flat columns. Preserves supplied buckets; rejects duplicate keys and incomplete requested data rather than cleaning it. Unreleased after 0.1.9. |
| Unbucketed factors plus supplied returns | `signal.prepare_signal_panel(factor, forward_returns, ...)` | Owns one complete finite sample, then assigns qcut/cut buckets and records factor/return/group/bucketing losses. Supports static or dated groups and group-relative/zero-aware buckets. Unreleased after 0.1.9; does not compute returns or validate a calendar. |
| IC, quantile returns, and spread | `SignalResult.ic`, `.quantile_returns`, `.spread` | Specify `periods`, `quantiles`, and `ic_method`; check timestamp alignment and compare each statistic's definition. |
| Grouped or group-adjusted IC observations | `panel.ic_by_date(group_adjust=..., by_group=...)` | Unreleased after 0.1.9. Uses supplied returns and reports per-date/group paired counts and undefined reasons. Inspect the shared adjusted outcomes with `panel.demean_returns(by_group=True)`. Does not construct a group-neutral portfolio. |
| Daily quantile means/errors and equal-date summaries | `panel.quantile_statistics(by_date=..., by_group=..., demeaned=..., group_adjust=...)` | Unreleased after 0.1.9. Reports sample standard deviation, standard error and asset/date counts; does not change the native pooled `SignalResult` fields or spread inference. |
| Quantile turnover | `SignalResult.turnover` | From 0.1.9, each horizon key uses that lag in sorted observed factor dates. The direct `compute_turnover(..., lag=...)` helper defaults to one observation; neither route establishes calendar-aware turnover. |
| Factor HTML report | `SignalResult.to_tear_sheet()` and `SignalDashboard` | This bridge populates IC and quantile summaries, but leaves `turnover_analysis` and quantile `cumulative_returns` unavailable. Existing plotting components do not make these paths part of this workflow. |
| Group-neutral portfolios and the complete Alphalens tear sheet | No complete equivalent | Grouped IC and return demeaning do not implement portfolio weights, holdings, event paths or the complete tear sheet. |

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

Current source rejects duplicate or missing date/asset keys in factor and price
inputs before computing or joining forward returns. Deduplicate overlapping
batches upstream with an explicit quote/aggregation policy; Diagnostic does not
silently choose first or last. The lower-level
`metrics.compute_forward_returns(..., date_col=None)` retains explicit input-row
order instead, including repeated timestamps within an asset. These are
unreleased input-boundary changes after 0.1.9, not calendar/grid validation.
For metrics-layer IC, decay and feature-outcome panels, pass the asset key as
`group_col`; `None` denotes one series per date. These workflows validate
prediction keys and forward the declared `date_col` to the return producer,
including custom time-column names.
Equivalent millisecond, microsecond and nanosecond datetime keys in the same
timezone join at the finer unit without rounding. An unrepresentable conversion
raises instead of wrapping timestamps; this does not infer a timezone policy or
validate calendar sessions.

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
splitting ties. It is not a cleaner: use `prepare_signal_panel` for explicit
probability/cut edges, zero-aware group buckets, complete-case filtering and
loss provenance. Other native signal statistics retain their stated
aggregation conventions.

Qcut boundary rounding depends on pandas/NumPy versions. For values `[0,1,2,3]`
and three buckets, the tested pandas 2.3.3/NumPy 2.5.4 environment assigns
`[1,2,3,3]`; pandas 3.0.5 corrects quantile rounding and assigns `[1,1,2,3]`.
Both retain right-closed bins, but the floating-point edge differs. Alphalens
0.4.6 declares `pandas<3`, so its actual comparisons use pandas 2.3.3. Diagnostic
on pandas 3 is checked separately against runtime qcut and the independent
boundary expectations, not an unsupported Alphalens installation. Qcut analysis records
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
equal-date aggregation and propagated spread errors. The separate
`panel.quantile_statistics()` route below exposes daily/equal-date bucket
statistics without changing those native defaults. Turnover is disabled by
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

Unreleased source after 0.1.9 retains the mean and positive fraction of a
single defined date-level IC. Its sample standard deviation and t-test remain
undefined. An empty IC sample has no positive-observation fraction, and a
zero/undefined IC standard deviation has no IC information ratio; neither is
reported as a fabricated zero. Signal results express positive observations
as a percentage, while the direct metric summary uses a fraction. JSON uses
null for unavailable values and restores them as NaN on the result round trip.

### Prepare and audit supplied returns

For an unbucketed factor Series and a precomputed return panel, unreleased
source after 0.1.9 provides `ml4t.diagnostic.signal.prepare_signal_panel`.
It does not need Alphalens or prices at runtime. From a source checkout, run the
[self-contained example](https://github.com/ml4t/diagnostic/blob/3ff496a079e73d03b4c6cf716c142441693ce097/examples/signal_panel_cleaning.py):

```bash
uv run python examples/signal_panel_cleaning.py
```

The example constructs its supplied outcomes with the existing return producer.
It starts with 24 factor observations and retains 12: four initial feature
warmup observations and eight incomplete forward tails are excluded. It asserts
those losses, then passes the owned panel to the existing analysis engine.
It also prints group-adjusted outcomes and twelve grouped IC observations
(three dates, two groups and two horizons), each with two paired assets.
The printed IC values describe synthetic data, not evidence of a profitable factor.

Inspect `panel.data`, `panel.excluded_data` and `panel.metadata["loss_by_stage"]`
before calling `panel.analyze()`. Each accessor returns an independent copy.
Excluded keys retain their timestamp precision, first failing stage and reason;
analysis JSON retains both Arrow snapshots and the cleaning audit. Loss limits use the original
factor count. `max_loss=0` exposes bucket errors and permits no exclusions;
values outside supplied bin/probability edges count as bucketing losses.
Numeric factor/outcome columns are normalized to Float64 without modifying the
caller or recovering precision already lost in Float32 inputs.

The [signal API](../api/index.md#signal-analysis) documents static/dated groups,
probability qcut, cut edges and zero-aware options. Preparing group-relative
buckets alone does not provide group-neutral inference. The analysis shortcut currently
requires positive integer `D` selectors and contiguous buckets starting at one.
Other supplied labels and bucket sets remain inspectable in `panel.data`; the
shortcut rejects them before writing a report. No calendar grid is inferred or
validated, and the native pooled/equal-date/report differences below still apply.

### Inspect grouped IC and adjusted outcomes

Use the same owned panel to compare the observations underlying a grouped IC.
These unreleased methods after 0.1.9 do not require Alphalens. Using `prepared`
from the earlier example, first attach explicit asset groups:

```python
from ml4t.diagnostic.signal import prepare_signal_panel

panel = prepare_signal_panel(
    prepared["factor"],
    prepared[["1D"]],
    groups={f"A{i:02}": "first" if i < 6 else "second" for i in range(12)},
    quantiles=3,
    max_loss=0.0,
)
adjusted = panel.demean_returns(by_group=True)
ic = panel.ic_by_date(group_adjust=True, by_group=True, min_obs=2)
print(adjusted)
print(ic.select("date", "group", "return_column", "ic", "n_obs", "status"))
```

`group_adjust=True` subtracts each date/group's paired mean return before
correlation. `by_group=True` computes separate correlations for each date/group.
The flags are independent of each other and of `binning_by_group`, which only
assigns buckets. Factors are not demeaned, and no portfolio weights are created.
The default method is Spearman. Pearson is an explicit option, not an Alphalens
default. Both correlation methods use the existing cross-sectional IC engine.

`panel.demean_returns(by_group=True)` exposes the same adjusted outcomes used by
IC. With `by_group=False`, it subtracts date means instead. Keys, factors,
groups, buckets and row order remain unchanged. The standalone
`signal.demean_signal_returns` and `signal.compute_signal_ic` accept raw unique
date/asset inputs and explicit `return_columns`. Each horizon uses its own
finite factor/outcome pairs; unavailable pairs do not enter another asset's
group mean. Demeaning retains those rows with NaN outcomes. A prepared panel
already has one common finite sample across all selected horizons.

IC observations retain undefined dates with null IC, actual `n_obs` and a
reason such as `no_pairs`, `insufficient_pairs`, `constant_factor` or
`constant_outcome`. The minimum is applied within each output group.
`ic_by_date` accepts arbitrary supplied return selectors; the integer-D and
contiguous-bucket restrictions apply to the separate `analyze()` shortcut.
Output dimensions are canonical `date` and, when requested, `group`, even with
custom physical input names.

By default, Diagnostic reports supplied observed dates. Alphalens 0.4.6 can
regularize global IC to its inferred frequency and insert undefined dates.
Pass an explicit `date_grid` DataFrame (`date`, or `date` and `group`) to request
particular output keys, including empty dates with zero pairs and null IC.
This grid controls output keys only. It does not validate sessions, construct
returns, fill observations or establish an exchange calendar.

### Inspect daily and equal-date quantile statistics

The owned panel also exposes a separate bucket-statistics table. Using the
`panel` constructed above:

```python
daily = panel.quantile_statistics(by_date=True, demeaned=False)
equal_date = panel.quantile_statistics(demeaned=False)
assert set(daily["sample_unit"].to_list()) == {"asset"}
assert set(equal_date["sample_unit"].to_list()) == {"date"}
print(daily)
print(equal_date)
```

Daily rows retain date, bucket, horizon and optional group. Each mean uses
finite factor/outcome pairs, with sample standard deviation (`ddof=1`) and
standard error `std / sqrt(n_obs)`. With `by_date=False` (the default), the
method reduces those daily means instead. Each date gets equal weight and
`n_obs` counts defined dates, not pooled asset-date rows. A singleton has a
defined mean but null sample deviation/error; an observed empty sample has
zero count and null mean. Buckets absent from a date are not invented.

The new method defaults to `demeaned=True`, subtracting each date's paired
mean return. `group_adjust=True` takes precedence and subtracts date/group
means even when `demeaned=False`. `by_group=True` retains group dimensions
independently. Use `demeaned=False, group_adjust=False` for raw returns.
Inspect the same adjusted observations with `demean_returns`; neither route
constructs portfolio weights or changes bucket assignments.

The direct `signal.compute_quantile_statistics` route accepts explicit return
selectors and physical column names. Both routes preserve timestamp precision
and accept positive bucket labels even when some labels are absent.
`SignalResult.quantile_returns` remains the existing pooled result. In a
quote-derived changing-universe sample, upper-bucket daily means of `0.3125`
and `0.5` give an equal-date mean of `0.40625` and a standard error of `0.09375`.
Pooling its three asset-date returns instead gives `0.375`. Choose the
estimand explicitly; neither number establishes a self-financing portfolio.
This bucket standard error is not a propagated spread error or a replacement
for the existing two-sample spread test. Full portfolio/event/report paths
remain separate work.

### Quantiles, turnover, and report boundaries

In unreleased source after 0.1.9, direct `metrics.cross_sectional_ic_series()`
and its summary route validate unique, nonmissing observation keys on both
inputs. Multi-asset panels require `entity_col`; overlapping batches must be
deduplicated upstream under an explicit policy. Each input supplies only its
selected value column, so extra historical-return columns cannot replace the
requested future outcomes, and both selected columns may share a name.
Equivalent same-timezone datetime storage units align losslessly without
rounding distinct observations. These boundary corrections do not change the
finite-pair, minimum-sample or correlation definitions.

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

Alphalens also supplies grouped summaries and its own complete factor tear
sheet. Those capabilities are not implied by grouped IC observations or by a Diagnostic signal result.
Diagnostic's default global quantiles do not mean `group_adjust=True` or
`binning_by_group=True`; the latter can now be requested explicitly during
supplied-panel preparation. Request grouped IC explicitly through `ic_by_date`.
Retain the legacy route for the missing analytical
and report sections, not merely for constructing those buckets.

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

For uncertainty estimates, current source
`evaluation.compute_backtest_uncertainty()` retains observed zero-return cash
sessions instead of treating them as warmup. In a tested NAV-to-return sample,
100 observations include 20 initial cash sessions. Their annualized Sharpe is
1.910147 at 252 periods per year; silently removing those sessions gave
2.136821. Explicitly slicing the dated input to the last 80 observations still
analyzes that shorter window. Retain the window-selection reason with your
research inputs; the result does not record it. The shared correction also
preserves zeros in selection adjustment and automatic bootstrap block selection.
This is unreleased source work after 0.1.9. Paired uncertainty, selection adjustment
and Reality Check align pandas DatetimeIndex inputs with the same unique date
labels chronologically, then exclude nonfinite observations with one common mask.
Entirely positional inputs require equal original lengths. Mixed dated/positional
inputs and different dated windows raise instead of guessing observation identity.
Single-series uncertainty retains its finite-value filter; neither route validates
a calendar grid or records excluded dates in its result. Their stationary-bootstrap intervals are not Pyfolio's IID bootstrap
intervals; matching point estimates does not establish interval equivalence.

Pyfolio's full sheet can additionally include return cones, significant
events, round trips reconstructed from fills, capacity analysis using
market data, and optional factor-attribution panels. Diagnostic has related
statistics and reports but no one-call equivalence for that full collection.
Keep Pyfolio or implement the specific analysis you need if those sections
are part of your acceptance criteria. Compare annualization, benchmark
alignment, cash, and cost conventions before comparing metrics.
