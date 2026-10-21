# Migrating an Alphalens or Pyfolio Task

Diagnostic is not a drop-in replacement for either package. Choose a task below,
convert its input shape, and compare result definitions before comparing numbers.
Unreleased source after 0.1.9 requires Polars 1.36.1 or newer. The earlier 0.20
minimum lacked APIs already used by prepared signal and factor-weight workflows.
Polars 1.0 also fails existing empty-sample and numerical/report contracts.
The complete signal suite passes at 1.36.1, including actual reference and browser
workflows. This floor also satisfies the tested Backtest integration requirement.
Core examples use synthetic data and the base package. HTML and real-Backtest
tasks name their optional dependencies below.
For selected performance metrics, see the separate
[Empyrical migration assessment](empyrical-migration.md).

One reason to evaluate Diagnostic beyond these matching values is its
[purged cross-validation workflow](cross-validation.md), which combines
chronological folds with multiple-trial Sharpe assessment. For portfolio
reporting, its [backtest bridge](backtest-tearsheets.md) also accepts a
completed `ml4t-backtest` result. These are separate capabilities, not
claims that the legacy tear sheets or factor panels are reproduced.

## Run the shared-input comparison

The package-only examples below show each Diagnostic input shape. The recorded
published 0.1.8 baseline uses this
[frozen comparison script](https://github.com/ml4t/diagnostic/blob/942f367f2f6aff269804d2600ba34814c661bf61/examples/legacy_migration_comparison.py).
Run it with Python 3.12 and the matching pandas/legacy dependencies:

```bash
uv run --no-project --python 3.12 \
  --with 'ml4t-diagnostic==0.1.8' \
  --with 'pyfolio-reloaded==0.9.9' \
  --with 'alphalens-reloaded==0.4.6' \
  --with 'empyrical-reloaded==0.5.12' \
  --with 'pandas==2.3.3' --with 'pytz==2026.5' \
  https://raw.githubusercontent.com/ml4t/diagnostic/942f367f2f6aff269804d2600ba34814c661bf61/examples/legacy_migration_comparison.py \
  --task all
```

That command uses published packages, not the source checkout. It asserts the
shared Sharpe, drawdown and IC results and prints other measured values. The
figures below describe that published baseline, not a performance benchmark or
a promise that every legacy output matches.

The current checkout's script also exercises unreleased prepared-input routes.
Do not combine it with the older installed package. From the current checkout,
run the complete source comparison instead:

```bash
uv run --with 'pyfolio-reloaded==0.9.9' --with 'alphalens-reloaded==0.4.6' \
  --with 'empyrical-reloaded==0.5.12' --with 'pandas==2.3.3' \
  python examples/legacy_migration_comparison.py --task all
```

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
| Empirical quantile return distributions | `plot_quantile_returns_violin(result.to_quantile_result(), period=...)` | Unreleased after 0.1.9. Uses actual dated bucket means retained by the analysis/report bridge, with explicit adjustment and percentile options. No summary-derived simulation. Pooled bars and density presentation remain distinct. |
| Dated spread/error chart | `plot_spread_timeseries(result.to_quantile_result(), observations=...)` | Unreleased after 0.1.9. Uses actual dated bucket means and independent propagated errors, with explicit pair, bandwidth and observation-window controls. No cumulative-increment reconstruction or inferred confidence coverage. |
| Arbitrary quantile mean differences and propagated errors | `panel.quantile_spread(upper_quantile=..., lower_quantile=..., ...)` | Unreleased after 0.1.9. Reuses daily/equal-date statistics and propagates independent leg errors with `hypot`. Raw +1 upper / -1 lower exposure is gross two, not a unit-gross portfolio return or a Student test. |
| Factor weights and return contributions | `panel.factor_weights(...)`, `panel.factor_returns(..., by_asset=...)` | Unreleased after 0.1.9. Shares one magnitude/equal-sign/group weight calculation across supplied horizons, with explicit zero-signal handling. Does not construct dated holdings or expiration accounting. |
| Factor alpha and beta | `panel.factor_alpha_beta(annualization_factors=...)` | Unreleased after 0.1.9. Reuses factor returns and OLS against the same universe's equal-asset mean. Retains dated joint samples/counts and explicitly compounded annual alpha; constant benchmarks and unavailable dates remain explicit. Does not add a complete returns table or tear sheet. |
| Selected-factor cumulative wealth | `panel.factor_cumulative_returns(return_column=..., quantiles=..., groups=...)` | Unreleased after 0.1.9. Selects rows before shared weighting, then compounds one supplied horizon in observation order. Retains actual returns/counts and a labelled flat-wealth policy for unavailable returns. Not self-financing daily holdings or an inferred calendar. |
| Quantile turnover | `SignalResult.turnover` | From 0.1.9, each horizon key uses that lag in sorted observed factor dates. Unreleased after 0.1.9, persistence summaries also preserve sub-microsecond observation identity. The direct `compute_turnover(..., lag=...)` helper defaults to one observation; neither route establishes calendar-aware turnover. |
| Dated turnover by bucket | `panel.quantile_turnover(lags=..., quantiles=..., date_grid=...)` | Unreleased after 0.1.9. Reports entrants/current-membership fractions, leg/universe counts and unavailable statuses on an observed or explicitly supplied clock. The scalar larger-membership definition remains separate. |
| Dated rank persistence | `panel.rank_autocorrelation(lags=..., min_obs=..., date_grid=...)` | Unreleased after 0.1.9. Ranks each full scored date before pairing assets and computing Pearson correlation. Retains counts, statuses and explicit clocks; native paired-sample Spearman summaries remain separate. |
| Factor HTML report | `SignalResult.to_tear_sheet()` and `SignalDashboard` | Unreleased after 0.1.9, persistence-enabled analyses also retain dated turnover/rank tables and explicit method/count summaries. Quantile cumulative paths and complete reference chart/report composition remain unavailable. |
| Dated holdings and the complete Alphalens tear sheet | No complete equivalent | Analytical factor weights and supplied-return contributions do not establish capital/expiry accounting, event paths or the complete tear sheet. |

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
lag method. Grouped IC, adjusted quantile statistics and factor weights require
the explicit owned-panel methods below; they are not enabled by this shortcut.
Complete reports remain unsupported by this route. No full-replacement claim follows from retaining a
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

### Match IC report statistics to selected dates

Unreleased after 0.1.9, `result.to_ic_result()` and the tear-sheet bridge compute
their summaries from the common finite IC dates they retain across requested
horizons. The original `SignalResult.ic`, standard deviation and inference maps
still describe each horizon's full finite sample. Select one period explicitly
to inspect that horizon without another horizon's date restriction:

```python
single_ic = result.to_ic_result(period=1)
values = np.asarray(single_ic.ic_by_date["1D"])
assert np.isclose(single_ic.ic_mean["1D"], values.mean())
print(single_ic.get_dataframe("summary"))
```

For example, a quote-derived one-period IC series `[1, -1]` can be restricted to
`[1]` when another horizon lacks the second date. The report mean is then `1`,
not the original full-sample mean `0`. A singleton keeps its mean and positive
percentage but no sample deviation, t/p or information ratio. An empty common
intersection has unavailable statistics, not values copied from discarded
observations. Required numeric summaries serialize unavailable values as null
and restore NaN on JSON read; optional HAC/RAS fields are unchanged.

The native IC time-series band is the zero-centered normal approximation
`z * std / sqrt(n_finite)`, not HAC-adjusted inference. Undefined variance
displays `CI unavailable` without a band; `show_significance=False` omits both.

Alphalens Reloaded 0.4.6 retains per-date undefined IC values in its table.
The native facade still extracts finite IC dates before this bridge. Compare
the same explicitly selected dates and finite observations; the bridge is not
a complete dated-status/count export or a union-clock report. Calendar/bar
semantics and full report composition remain separate accepted work. The
[runnable supplied-panel example](https://github.com/ml4t/diagnostic/blob/0ef65182e33fb661cb9cb8ccaf3aa0d01d123a63/examples/signal_panel_cleaning.py)
checks report means against their actual series and prints the report summaries.

### Inspect monthly IC on the selected sample

With the visualization extra, unreleased source after 0.1.9 plots arithmetic
monthly means of the IC observations already selected by the report bridge:

```python
from ml4t.diagnostic.visualization.signal import plot_monthly_ic_heatmap

monthly_ic = plot_monthly_ic_heatmap(single_ic, period="1D")
print(monthly_ic.data[0].x)
print(monthly_ic.data[0].z)
```

IC and its means are dimensionless correlations. Each timestamp's supplied
civil year and month determine its cell, including timestamps whose UTC offset
changes. This grouping does not infer an exchange session or bar calendar.
The plot labels actual observed months; January and March do not become
January and February. Years run newest first. Missing year/month intersections
stay unavailable rather than zero; a defined monthly mean of zero remains a
visible cell. An empty retained sample displays `Monthly IC unavailable`.
Cell labels are trace-native, avoiding numeric year annotations that previously
collapsed the visible heatmap. These values and labels also reach the existing
`SignalDashboard` through `result.to_tear_sheet()`.

For Alphalens Reloaded 0.4.6, compare
`performance.mean_information_coefficient(selected_factor_data, by_time="ME")`
on the same explicitly selected factor/outcome rows and finite IC dates before
calling its `plotting.plot_monthly_ic_heatmap`. Its month-end grouping inserts
entirely absent months between observations; Diagnostic shows only observed
month columns. Neither definition fabricates a zero IC for a missing month.
Shared-input checks cover raw, prepared and owned workflows, their cleaning
keys, buckets, counts and IC values before monthly reduction. They do not
establish forward-calendar equivalence, identical renderer presentation or
complete information tear-sheet coverage. The existing
[supplied-panel example](https://github.com/ml4t/diagnostic/blob/b65d0b5246c8a00ad232c0277c59c84aa5238431/examples/signal_panel_cleaning.py)
checks and prints monthly plot values when run with
`uv run --extra viz python examples/signal_panel_cleaning.py --plots`.

### Prepare and audit supplied returns

For an unbucketed factor Series and a precomputed return panel, unreleased
source after 0.1.9 provides `ml4t.diagnostic.signal.prepare_signal_panel`.
It does not need Alphalens or prices at runtime. From a source checkout, run the
[self-contained example](https://github.com/ml4t/diagnostic/blob/3231786bd9ad1ff0e47cea7c2e3b336ff6519893/examples/signal_panel_cleaning.py):

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
buckets alone does not provide group-neutral inference. Unreleased after 0.1.9,
`panel.analyze()` retains arbitrary supplied return columns and sparse positive
bucket IDs without rebucketing. `result.horizon_labels` preserves selector order;
`result.quantile_ids` preserves the actual IDs in summaries, JSON and quantile
plots. The strict standalone route accepts `return_columns=("30m", "1ns", "1D1h")`
instead of integer `periods`; supplying both arguments is an error. Without a
`quantiles` limit, that route uses the buckets actually present. Existing integer
`periods` calls retain their five-bucket default and daily-column aliases.

These labels still do not establish durations or validate a calendar grid.
When persistence is enabled, only positive exact-day labels such as `1D` produce
automatic turnover lags in observed factor timestamps. Subday/composite selectors
remain analyzable without invented daily lags. For explicit observation lags use
`panel.quantile_turnover(lags=(1, 2))`. The executable
`examples/signal_panel_cleaning.py --plots` includes supplied intraday outcomes,
sparse cut buckets and actual plotted bucket means. Native pooled/equal-date/report
differences below still apply; this change does not complete all seven tear sheets.

The installed-artifact comparison with Alphalens 0.5.0 uses shared `30m`, `1ns`
and `1D1h` outcome columns, nanosecond keys, ordinary/sparse cut buckets and a
missing-outcome case. Keys, excluded rows, bucket membership and dated ICs agree
with matched options. Native pooled means remain different from Alphalens'
equal-date means when date counts differ: in the tested low bucket, seven asset
outcomes average `-0.02/7`, whereas four date means average `-0.0025`.
Qualification used Python 3.12.11, pandas 3.0.6, NumPy 2.5.4, Polars 2.0.0 and
SciPy 1.18.1. This comparison qualifies supplied-panel analysis, not price
endpoint construction, holding expiry or complete report replacement.

Prepared-input entry points also accept the shared `feed_spec` and canonical
`timestamp_col`/`entity_col` keywords. Explicit column names override feed mappings;
existing `date_col`/`asset_col` aliases remain supported and conflicting explicit
names raise. Calendar precedence is explicit argument, declared feed metadata,
then NYSE; `calendar=None` always disables calendars. The audit and JSON preserve
an omitted version-2 calendar separately from explicit null. Recording this
declaration does not validate supplied outcomes against a schedule or infer their
endpoints. Version-1 metadata retains its original null-as-unspecified convention.

For supplied long outcomes, set `horizon_col="horizon"` and, if needed,
`return_col="forward_return"`. `return_columns` then selects horizon names,
not physical value columns. The cleaner pivots once and still selects one common
finite sample. `panel.outcome_data` and `result.outcome_data` retain the original
selected source/end/availability fields and statuses, including excluded outcomes.
Their derived `duration_ns` and `return_duration_ns` are exact integer nanoseconds
encoded as Arrow decimal values; session labels and naive keys have no inferred
physical duration. Analysis JSON retains the owned Arrow snapshot without losing
nanoseconds. Declared log returns are rejected, not silently compounded as decimal
returns. Conflicting declared producer/feed calendars raise unless an explicit
calendar argument chooses the analytical setting; that never recomputes outcomes.

`examples/signal_panel_cleaning.py --plots --engineer-outcomes` optionally runs
Engineer's native producer and compares its retained keys, buckets and outcomes
with the standalone supplied-return route. Engineer remains optional. Captured
producer tests also exercise exact `1ns`/`2ns` elapsed endpoints, missing quotes
and overlapping input batches without installing Engineer in the consumer.

Standalone captures from the qualified Engineer source also exercise native
session labels, validated expected bars, CME Sunday-night bars and NYSE's July 3
early close. Diagnostic retains the producer's exact endpoints and declared
units; it does not relabel session dates as midnight instants. Session-label
durations and unknown publications remain unknown. These captures use Engineer
`0.1.7.dev35+g071b87cef`, Specs 0.2.0 and pandas-market-calendars 5.5.0, not a
claim that all later producer revisions or every option has been qualified.

### Prepare the same sample directly from prices

Unreleased after 0.1.9, `prepare_signal_panel(factor, prices=prices,
horizons={"1D": 1, "2D": 2}, price_basis="raw", calendar=None)` feeds the
same cleaner. Supply prices or realized outcomes, not both. Prices may be keyed
long quotes or a pandas wide table; the declared basis is retained, not adjusted.
`examples/signal_panel_cleaning.py --plots` executes both routes and compares
the retained sample. Neither a provider nor Engineer is required.

The new price route defaults to the union of supplied price timestamps
(`index_mode="common_grid"`). An absent asset row stays an unavailable endpoint;
it is not silently compressed into the next observed asset row. Explicit
`index_mode="per_asset"` retains per-asset observation shifts. A present row with
a missing quote is distinguished from an absent row. Neither clock establishes
exchange-session or expected-bar completeness. Supplied long outcomes can carry
those separately declared producer units without changing their endpoints.

`endpoint_method="cumulative"` computes entry-to-exit change;
`endpoint_method="exit_bar"` computes the final single-observation return instead.
For prices `[100, 110, 121]`, a two-observation outcome is respectively `0.21`
or `0.10`, with different reference timestamps. `fill_method="ffill"` is an
explicit analytical view, not the native default. Its original row/price flags
remain false where quotes were imputed. A wide missing cell alone does not
establish whether an original source row existed before reshaping.

The optional `return_filter_zscore` fits each asset's returns over all factor
analysis dates before cleaning, including dates without a factor for that asset.
It retains unfiltered values and exclusion statuses. This lookahead-sensitive
research filter is not a causal feature or executable-price policy. Counts,
losses, exact endpoints and these choices survive analysis JSON.

With `available_at_col`, an unfilled, unfiltered outcome is available only when
both original reference and exit publications are known. Missing publications
remain unknown. Imputed outcomes and sample-fitted filtering leave
`outcome_available_at` unknown: a past quote's publication does not prove that
the future quote will be missing or that a future-fitted filter will retain it.
The primitive publication timestamps remain inspectable.

The shared public return kernel now subtracts integer prices before floating
conversion, so valid one-tick changes above Float64's integer precision are not
lost. This kernel emits Float64 returns; it cannot recover input quantization.

### Keep Alphalens preparation calls during migration

Unreleased after 0.1.9, change the preparation import while keeping the pandas
MultiIndex inputs and familiar cleaning arguments:

```python
# Previously: from alphalens import utils as al
from ml4t.diagnostic.compat.alphalens import utils as al
from ml4t.diagnostic import analyze_prepared_signal
import pandas as pd

adapter_dates = pd.bdate_range("2025-07-07", periods=4, tz="UTC")
adapter_prices = pd.DataFrame(
    {"A": [100, 110, 121, 133.1], "B": [100, 120, 144, 172.8]},
    index=adapter_dates,
)
adapter_factor = pd.Series(
    [-1.0, 1.0] * 2,
    index=pd.MultiIndex.from_product(
        [adapter_dates[:2], ["A", "B"]], names=["date", "asset"]
    ),
    name="factor",
)
adapter_clean = al.get_clean_factor_and_forward_returns(
    adapter_factor, adapter_prices, periods=(1, 2), quantiles=2, max_loss=0,
    filter_zscore=None, price_basis="raw", calendar=None,
)
adapter_result = analyze_prepared_signal(
    adapter_clean, return_columns=["1D", "2D"], min_assets=2, calendar=None,
)
```

The labels in this example require daily observations. Intraday inputs retain
labels such as `30m`, `1ns` and composites. The adapter also exposes
`compute_forward_returns` and `get_clean_factor` for the two-stage workflow or
supplied returns. Numeric and opaque hashable asset IDs are restored without
string conversion; static or dated groups, labels, qcut/cut, custom edges,
zero-aware buckets and the common-sample loss limit use the native cleaner.
The returned `factor_quantile` column is accepted directly by prepared analysis.
Direct `al.quantize_factor` uses the same native qcut/cut backend without
discarding rows for missing future outcomes. Its `no_raise=True` option excludes
failed date/group buckets only; strict duplicate-edge failures remain actionable
`ValueError`s at the compatibility boundary.
The runnable `examples/signal_panel_cleaning.py --plots` compares price,
supplied-return and adapted samples before executing native diagnostics.

These are explicit legacy preparation defaults, not native defaults:

- Price views forward-fill before computing observation-count returns. The
  convenience cleaner defaults to `filter_zscore=20`; the standalone return
  function defaults to no filter. Pass `None` to avoid the lookahead filter.
  Legacy filtering uses all retained pandas date-index levels before selecting
  factor rows, including unused levels left after dropping an entire date. The
  native price route instead fits actual factor dates. The audit records the
  selected filter clock's count and hash; no scored observations are invented.
- Inferred weekday/holiday calendars determine legacy duration labels only.
  They are not an exchange schedule. A nonconforming intraday index retains its
  observed frequency and emits a warning, as corrected Alphalens 0.5.0 does.
- Without an explicit basis the adapter records `legacy_unspecified`, not raw
  or adjusted. Supply `price_basis` when known. Use native `outcome_data` when
  original quote masks, exact endpoints and publication instants are needed.
- The pandas result retains cleaning audit metadata in `attrs["diagnostic"]`.
  Passing the return adapter's table to its cleaner preserves declared calendar
  metadata, including explicit `None`. A conflicting new feed declaration
  raises unless an explicit `calendar` argument resolves it.
- `MaxLossExceededError` and `NonMatchingTimezoneError` are available in this
  namespace. The adapter does not print legacy progress messages. Ambiguous
  inferred labels raise instead of silently overwriting a horizon; explicit
  native selectors avoid that inference.

Preparation comparisons use corrected Alphalens 0.5.0, including retained keys,
missingness, bucket/group membership, cumulative versus terminal-bar returns and
filter options. Integer-tick/Float32 accuracy is checked against independent
fractions rather than reproducing legacy cancellation. These preparation calls
do not yet provide every performance function or all seven legacy report routes.
The keyword-only `reference_version` defaults to `"0.5.0"`; `"0.4.6"` explicitly
selects the frozen profile. These preparation definitions are shared; the
historical intraday frequency-assignment failure is not reproduced. The profile
is retained and threaded through the convenience call without global state.
The installed preparation profile uses Python 3.12.11, pandas 3.0.6, NumPy 2.5.4,
Polars 2.0.0, SciPy 1.18.1 and Plotly 7.1.0. A separate consumer/reference profile
uses pandas 2.3.3, NumPy 2.2.6, Polars 1.36.1 and SciPy 1.17.0. The pinned Engineer
wheel requires Polars 2, so its live regeneration is not claimed at that floor;
standalone ingestion of its frozen captures needs no Engineer installation.

The same namespace also provides explicit legacy clock/selector conversions:
`backshift_returns_series`, `get_forward_returns_columns`,
`turnover_periods_from_columns`, `timedelta_to_string`,
`timedelta_strings_to_integers`, `make_naive_ts`, `infer_trading_calendar`, and
the custom-calendar timedelta add/difference helpers. Their shared implementation
is `ml4t.diagnostic.utils.legacy_time`, not another label or execution engine.
Backward-return rekeying follows the common date-level clock, preserving missing
asset rows and session-date labels rather than compressing per asset. Zero shift
is an identity and an overlong clock gives an empty result; valid sparse input
does not reproduce the reference helper's first-row assertion failure.

Legacy business-day offset addition preserves local wall-clock time. Pandas
changed `Day` from a 24-hour tick to a calendar day in
[version 3](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html); this explicit
legacy helper follows the supplied pandas offset, not a version-independent
physical-duration contract. Both dependency profiles are tested. The legacy duration
difference counts UTC-naive dates and retains a physical subday remainder.
They are not inverses across DST and are not authoritative session arithmetic.
Negative duration formatting uses an exact global sign: the independent `-1ns`
round trip corrects Alphalens 0.5.0's mixed-sign component string. Floored day
components are a separate utility; automatic turnover selects only positive
exact-day labels, never a zero lag inferred from `30m` or `1ns`.

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
Both `ic_by_date` and `analyze()` accept arbitrary supplied return selectors;
the former additionally exposes grouped/adjusted observations and statuses.
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
zero count and null mean. Daily tables do not invent missing buckets. Across-date
group summaries retain the product of observed group and bucket labels, including
empty combinations with zero dates and null statistics, as in Alphalens 0.4.6.

The new method defaults to `demeaned=True`, subtracting each date's paired
mean return. `group_adjust=True` takes precedence and subtracts date/group
means even when `demeaned=False`. `by_group=True` retains group dimensions
independently. Use `demeaned=False, group_adjust=False` for raw returns.
Inspect the same adjusted observations with `demean_returns`; neither route
constructs portfolio weights or changes bucket assignments.

The direct `signal.compute_quantile_statistics` route accepts explicit return
selectors and physical column names. Both routes preserve timestamp precision
and accept positive bucket labels even when some labels are absent. Horizon
rows follow the requested return-column order, not alphabetical labels.
`SignalResult.quantile_returns` remains the existing pooled result. In a
quote-derived changing-universe sample, upper-bucket daily means of `0.3125`
and `0.5` give an equal-date mean of `0.40625` and a standard error of `0.09375`.
Pooling its three asset-date returns instead gives `0.375`. Choose the
estimand explicitly; neither number establishes a self-financing portfolio.
This bucket standard error is not a propagated spread error or a replacement
for the existing two-sample spread test. Full portfolio/event/report paths
remain separate work.

Unreleased fixes after 0.1.9 stabilize the shared finite-return mean, sample
deviation and standard error. Identical large finite returns retain their
finite mean and observation count instead of overflowing into a missing daily
mean. Standard error is rescaled separately, so it can remain finite even when
the sample deviation exceeds Float64 range. Ordinary finite NumPy means retain
their existing result. The pooled spread and its existing two-sample Student
test use common scaled, shifted observations to avoid intermediate overflow
or underflow; their estimand and test method are unchanged.

### Inspect arbitrary quantile spreads and propagated errors

Select the buckets explicitly. This unreleased method after 0.1.9 delegates to
`quantile_statistics` and preserves its date/group sampling and adjustment flags:

```python
spreads = panel.quantile_spread(
    upper_quantile=2,
    lower_quantile=1,
    by_date=True,
    demeaned=False,
)
assert set(spreads["sample_unit"].to_list()) == {"asset"}
assert set(spreads["exposure_convention"].to_list()) == {
    "long_upper_short_lower_unit_each"
}
print(spreads)
```

Each row retains the upper/lower means, sample errors and observation counts.
The difference is upper minus lower. Its standard error is
`hypot(upper_standard_error, lower_standard_error)`, the independent-error
propagation used by Alphalens 0.4.6. Selecting the same bucket twice gives zero
difference but still propagates the two reported errors independently; it does
not estimate their covariance. `include_standard_error=False` leaves joint
error null, not zero. Missing counterpart legs and undefined singleton errors
remain unavailable. Across-date group/bucket combinations with no observations
retain zero leg counts; absent per-date legs have null counts.

The direct `signal.compute_quantile_spread(statistics, ...)` consumes the
existing canonical statistics table, including positive noncontiguous bucket
labels, without resampling. Requested horizon order and precise timestamp keys
survive the join. Entirely absent requested buckets in nonempty statistics fail
clearly; a cleaned empty sample returns a typed empty table. Caller inputs stay
unchanged.

The raw spread represents +1 upper / -1 lower exposure, gross two. It is not a
unit-gross portfolio return, a self-financing wealth path or a time-series t
test. The native pooled `SignalResult.spread_t_stat` still uses its existing
two-sample method. Released Alphalens squares its error inputs before taking a
square root; the new helper uses `hypot` to avoid intermediate overflow and
underflow when the final error is representable. This numerical reconciliation
does not imply full reference parity. Plot bands, period conversion and complete
report/event/portfolio workflows remain separate capabilities.

### Inspect dated quantile turnover

Use the owned sample's bucket assignments without recomputing returns or
buckets. This unreleased method after 0.1.9 reports dimensionless fractions:

```python
dated_turnover = panel.quantile_turnover(lags=(1, 2))
assert set(dated_turnover["turnover_method"].to_list()) == {"entrants_over_current"}
assert set(dated_turnover["lag_unit"].to_list()) == {"observations"}
print(dated_turnover)
```

Each date/bucket/lag observation retains `n_current`, `n_previous`,
`n_entered`, `n_overlap` and both date-level universe sizes. Turnover is the
number of entrants divided by current bucket membership, as in Alphalens
0.4.6. A bucket shrinking from `{C, D}` to `{D}` has no entrants and turnover
zero. The native scalar helper's larger-membership definition instead gives
that bucket `0.5`; if the other bucket is unchanged, its two-bucket mean is
`0.25`. Neither definition is silently substituted for the other.

The default clock contains sorted observed input dates. A unique, explicitly
supplied `date_grid` must cover every observed date and defines lag positions,
including missing observations. Its date column uses the panel's physical date
name. It does not infer holidays, choose an exchange calendar or substitute the
next observed date for a missing one. Lags count clock observations, not days or
sessions. `clock_source` distinguishes observed and supplied grids.

Insufficient history has null previous counts. Empty current/previous buckets
have zero membership counts; their turnover, overlap and entrant estimates stay
null, with explicit statuses rather than zero turnover. Missing interchange
bucket labels are excluded from membership while their dates and universe sizes
remain visible. Positive noncontiguous labels and requested lag/bucket order
are preserved. `quantiles=None` selects observed labels; explicit selections
can retain unavailable buckets on a supplied clock even for a cleaned-empty
sample.

The direct `signal.compute_quantile_turnover` accepts native or pandas
date/asset-indexed bucket tables, including an Alphalens bucket Series converted
to a DataFrame. Physical column selectors are explicit, inputs stay unchanged,
and duplicate keys or incomplete grids fail before computation. No factors or
future return columns are required by this consumer. Report/plot integration
and authoritative calendar/time contracts remain separate capabilities.

The shared membership cache also isolates caller column names from its internal
aggregation names. A native `quantile_col="assets"` previously raised a duplicate
column error; that valid selector now retains the same scalar definition.

### Inspect dated full-date rank persistence

Use the same owned sample to rank each date before intersecting assets. This
unreleased route after 0.1.9 returns dimensionless correlations:

```python
dated_rank = panel.rank_autocorrelation(lags=(1, 2), min_obs=2)
assert set(dated_rank["rank_scope"].to_list()) == {"date"}
assert set(dated_rank["correlation_method"].to_list()) == {"pearson"}
print(dated_rank)
```

As in Alphalens 0.4.6, average tied ranks are computed over each complete scored
date, then Pearson correlation uses common ranked assets. For prior full ranks
`A=1, B=2, C=3, D=4` and current ranks `A=1, B=2, D=3`, the paired vectors are
`[1, 2, 4]` and `[1, 2, 3]`, giving `9 / sqrt(84) = 0.981980506...`.
The existing `compute_autocorrelation` instead reranks the paired sample for
Spearman correlation, giving `1.0`. Its scalar averaging, missing-value policy
and default `min_obs=10` are unchanged. Neither definition replaces the other.

The dated route retains `n_obs` common ranked assets, `n_current` and
`n_previous` rankable counts, and both raw date-level universe sizes. Null/NaN
scores are omitted before ranking; infinite raw scores retain ordered extreme
ranks. The direct `signal.compute_rank_autocorrelation` needs only numeric
factors and unique date/asset keys, not quantiles or future returns. Irrelevant
missing future returns never select its raw sample. An owned panel has already
removed nonfinite factors and incomplete common outcomes during preparation;
its cleaned sample can therefore differ from the direct raw-factor route.

`min_obs` must be at least two. Insufficient common assets or constant paired
ranks yield null correlations with explicit statuses, not zero; actual counts
remain visible. Insufficient history has null previous counts. Clock rules,
physical selectors, exact timestamps, caller ownership and supplied missing
dates follow the dated turnover route above. In particular, lags count
observations, not inferred sessions or days. Comparing Alphalens requires
declaring the same observation frequency on its date index; default calendar
parity is not established. Standalone dated calls return tables; the
persistence-enabled analysis workflow below retains them in the report bridge.
Neither route establishes full Alphalens/Pyfolio replacement.

### Inspect dated persistence in a result and report

Enable persistence in the actual analysis workflow to retain the dated tables
as well as the existing native scalar summaries. This unreleased workflow after
0.1.9 works through `analyze_signal`, `analyze_prepared_signal` or the owned panel:

```python
report_result = panel.analyze(
    min_assets=2, compute_turnover_flag=True, autocorrelation_lags=2
)
persistence = report_result.to_turnover_result()
assert persistence is not None
turnover_observations = persistence.get_dataframe("dated_turnover")
rank_observations = persistence.get_dataframe("dated_autocorrelation")
print(persistence.get_dataframe("summary"))
assert report_result.to_tear_sheet().turnover_analysis is not None
```

The properties `report_result.dated_turnover` and
`report_result.dated_autocorrelation` return independent table copies with exact
timestamps, counts, unavailable statuses and method/clock metadata. Signal JSON
and the structured result's JSON preserve those observations as Arrow IPC/base64,
including sub-microsecond timestamps and null estimates. Missing or malformed
paired persistence payloads fail instead of silently discarding a report section.
The new reader accepts older saved results without these observations. New
persistence-enabled JSON adds two encoded fields and requires the new reader;
upgrade the reader before loading that JSON in an older environment.

The existing native `turnover`, `autocorrelation` and `half_life` attributes keep
their definitions. Dated turnover uses entrants/current membership, and dated
rank persistence uses full-date ranks before paired Pearson correlation. The
result's `turnover` DataFrame view averages defined date pairs within each
bucket/lag; its `autocorrelation` view averages defined date pairs per lag with
`n_dates` counts. The `summary` view averages bucket means and labels lags
`1obs`, `2obs`, and so on, not as validated return horizons or sessions. No
half-life or uncertainty is inferred for these dated summaries. Default analysis
clocks remain observed dates; authoritative calendar adoption is separate.

With the declared visualization extra, pass `report_result.to_tear_sheet()` to
the existing `SignalDashboard` to render dated tables, method labels and summary
values. Unavailable values display as `N/A`; the workflow does not create
confidence bands from a fixed 252-observation assumption. These are numeric
table/report paths, not complete dated chart composition or all seven Alphalens
tear sheets. Quantile cumulative paths, holdings and event workflows remain
separate accepted work. Persistence-disabled analyses and historical saved
results without dated observations return no turnover component; scalar means
cannot reconstruct missing dates, buckets or reference ranks.

### Plot dated rank and membership persistence

Unreleased after 0.1.9, the existing plotting APIs consume the actual dated
component retained by persistence-enabled analysis:

```python
from ml4t.diagnostic.visualization.signal import (
    plot_autocorrelation,
    plot_top_bottom_turnover,
)
from ml4t.diagnostic.visualization.signal.turnover_plots import plot_turnover_heatmap

dated = report_result.to_turnover_result()
membership_plot = plot_top_bottom_turnover(dated, lag=1)
rank_plot = plot_autocorrelation(dated, lag=1)
lag_bucket_means = plot_turnover_heatmap(dated)
print(membership_plot.layout.meta)
print(rank_plot.layout.meta)
```

For dated inputs, `lag` counts recorded clock observations, not return days or
exchange sessions. The default selects the first recorded lag. Membership and
rank computations can retain different lag sets; inspect `dated_turnover` and
`dated_autocorrelation` before selecting one. Missing or invalid lags raise
`ValueError`. Return-horizon `period` and lag-summary `max_lags` selectors do not
apply to a dated rank time series. Historical legacy components retain their
existing horizon bars and lag-summary plots; they are not converted into dated
observations.

Top/bottom plots show actual entrants/current-membership fractions for the
extreme recorded buckets; `show_all_quantiles=True` includes every recorded
bucket. Rank plots show the existing full-date average-rank Pearson estimates,
not reranked paired-sample Spearman correlations. With changing membership,
the paired ranks `[1, 2, 4]` and `[1, 2, 3]` yield `9 / sqrt(84)`, not `1`.
Both charts preserve null gaps and show actual counts and statuses on hover.
Sub-millisecond, single-date and entirely unavailable samples use a categorical
observed-order axis with exact
timestamps on hover, avoiding millisecond collapse in the browser. An empty
warmup sample displays unavailable; a supplied empty clock retains missing
observations rather than fabricated zero estimates.

The heatmap reuses recorded bucket/lag means over defined dates and shows their
date counts. A defined zero turnover remains zero; missing means stay blank.
No confidence interval or half-life is inferred from the dated tables. The rank
plot displays `CI unavailable` when `show_significance=True`; setting it to
`False` omits that notice. The existing dashboard renders both dated charts
at their first recorded lag and retains the complete tables and summaries.

Shared-input checks compare the same membership, rank universe and observation
clock with Alphalens Reloaded 0.4.6 `quantile_turnover` and
`factor_rank_autocorrelation` before their dated plotting routes. The native
scalar larger-membership turnover and paired-sample Spearman definitions remain
separate. These chart workflows do not establish exchange-calendar defaults,
complete turnover-sheet composition or full Alphalens/Pyfolio replacement.
Run the existing
[supplied-panel example](https://github.com/ml4t/diagnostic/blob/8d52739bb133e5c5d3540c8cf268ea1a8d3c7722/examples/signal_panel_cleaning.py)
with `--plots` to inspect the actual
trace values, recorded methods and hover counts without a legacy dependency.

### Plot empirical dated quantile observations

Unreleased after 0.1.9, the actual raw, prepared and owned analysis routes retain
`result.quantile_observations`. Each row identifies a date, bucket and supplied
return column, its mean, asset count and adjustment. The independent table
survives `SignalResult` JSON, `to_quantile_result()` and `to_tear_sheet()`.
With the visualization extra installed:

```python
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_quantile_returns_violin

quantiles = report_result.to_quantile_result()
observations = quantiles.get_dataframe("observations")
fig = plot_quantile_returns_violin(quantiles, period="1D", ylim_percentiles=(10, 90))
adjusted = panel.quantile_statistics(by_date=True, demeaned=True, group_adjust=True)
adjusted_fig = plot_quantile_returns_violin(quantiles, observations=adjusted, period="1D")
html = SignalDashboard().generate(report_result.to_tear_sheet())
```

The default violin sample contains actual date/bucket means, not pooled asset
returns or normal samples inferred from scalar summaries. Unequal bucket sizes
can make its equally weighted date mean differ from the existing pooled bar.
Neither the plot nor report changes NumPy's global random state. Select one
group explicitly before plotting grouped statistics; missing observations are
unavailable, not fabricated. Historical summary-only results need an explicit
observations table or supplied values. The existing `factor_data=` path remains
an explicit raw-value sample, not an implicit date reduction.

Alphalens Reloaded 0.4.6 accepts dated quantile means and displays basis points.
Diagnostic trace values remain decimal returns with percentage axis labels;
`ylim_percentiles` uses the selected horizon without dropping trace samples.
Group adjustment takes precedence over date demeaning, as in the shared
statistics method. Both density renderers stop at the observed sample bounds,
but Plotly and Seaborn smoothing/box presentation are not claimed identical.
Plot separate horizons explicitly; the current dashboard renders its first
horizon and labels its selector as not interactive. A matching bundled Plotly
runtime makes its HTML self-contained, including NumPy-backed trace arrays.
Full multi-horizon chart composition, cumulative paths, event/portfolio
workflows and all seven Alphalens tear sheets remain separate accepted work.

### Distinguish unavailable monotonicity from a negative decision

Unreleased after 0.1.9, use the structured result's availability-aware accessor
before interpreting a quantile monotonicity flag. Construct two buckets
explicitly from the supplied `prepared` observations above:

```python
two_bucket_panel = prepare_signal_panel(
    prepared["factor"], prepared[["1D"]], quantiles=2, max_loss=0
)
quantiles = two_bucket_panel.analyze(min_assets=2).to_quantile_result()
assert quantiles.get_monotonicity("1D") == (None, "unavailable")
print(quantiles.get_dataframe("spread").select("period", "is_monotonic"))
print(quantiles.summary())
```

The computed DataFrame uses a nullable Boolean; the text summary, bar annotation
and dashboard summary card display `N/A`. Unknown periods raise `ValueError`.
The native rule leaves rank correlation undefined below three defined bucket
means or with constant means. This repair does not change that rule or the
finite-score decision threshold. A finite negative decision still displays
`False`/`No`; increasing and decreasing finite decisions retain their direction.

Raw `is_monotonic` and `monotonicity_direction` maps retain their historical
saved-model shape, including `False`/`"none"` for undefined scores. Use
`get_monotonicity()` or the computed DataFrame instead of interpreting those
raw flags without their rank correlation. Actual quote-derived samples,
bucket membership and mean returns are checked against Alphalens Reloaded
0.4.6; the native minimum-three-bucket decision is not an Alphalens definition
or a statistical significance claim. The
[runnable supplied-panel example](https://github.com/ml4t/diagnostic/blob/b94d7017b03353ad0f0b0c41f939217888106a90/examples/signal_panel_cleaning.py)
prints the unavailable decisions alongside the actual observations.

### Plot dated spreads and propagated errors

Unreleased after 0.1.9, the existing spread chart uses actual dated bucket means
retained by the analysis or supplied explicitly. With the visualization extra:

```python
from ml4t.diagnostic.visualization.signal import SignalDashboard, plot_spread_timeseries

quantiles = report_result.to_quantile_result()
fig = plot_spread_timeseries(
    quantiles, period="1D", upper_quantile=2, lower_quantile=1,
    rolling_window=22, bandwidth=1.0,
)
adjusted = panel.quantile_statistics(by_date=True, demeaned=True, group_adjust=True)
adjusted_fig = plot_spread_timeseries(quantiles, observations=adjusted, period="1D")
html = SignalDashboard().generate(report_result.to_tear_sheet())
```

The curve is the raw upper-minus-lower mean difference, with +1/-1 (gross-two)
exposure, not a unit-gross portfolio return. It delegates to
`compute_quantile_spread`, retaining each leg's actual asset count and shared
return adjustment. Select one group before passing grouped statistics. Missing
counterparts stay unavailable rather than becoming a zero mean. The automatic
top/bottom route requires at least two buckets; explicit arbitrary/reversed/same
bucket pairs retain the shared method's independent-error convention.

`rolling_window` counts observations and requires a complete defined window.
The native default is 21; choose 22 to match Alphalens 0.4.6's plotted moving
mean. `bandwidth` multiplies the actual propagated standard error at each date,
not a rolling sample deviation. The historical `show_confidence` argument now
controls these error bands; its name does not establish a confidence level.
Unavailable errors create gaps and an explicit note, not a fabricated 95% band.
Pooled scalar/Student summaries remain separate and are not used to annotate
this dated sample.

Trace returns stay decimal with percentage ticks. The reference shows basis
points and clips its default axis to the 95th percentile of absolute spreads;
Diagnostic retains every trace value and uses its automatic range. Callers can
set a different range on the returned figure. Non-millisecond timestamps use
labelled observation order, with exact timestamps/counts on hover, instead of
silently collapsing distinct keys through JavaScript dates.

Historical summaries or cumulative-only models must supply actual canonical
statistics; neither cumulative increments nor aggregate means reconstruct the
missing sample and uncertainty. With the source checkout's visualization extra,
`uv run --extra viz python examples/signal_panel_cleaning.py --plots` prints
actual plot arrays and provenance alongside the existing core-only example.
This adds a dated chart to the first-horizon report, not the complete reference
multi-horizon, cumulative, holdings or event workflows.

### Construct factor weights and supplied-return contributions

Use the owned panel's factor observations to construct exposure independently of
its bucket labels. These unreleased methods after 0.1.9 share one normalization
calculation. Using the `panel` above:

```python
from ml4t.diagnostic.evaluation import PortfolioAnalysis

weights = panel.factor_weights(group_adjust=True)
contributions = panel.factor_returns(group_adjust=True, by_asset=True)
aggregate = panel.factor_returns(group_adjust=True)
one_period = aggregate.to_pandas()
one_period = one_period[one_period["return_column"] == "1D"].set_index("date")["weighted_return"]
summary = PortfolioAnalysis(one_period, periods_per_year=252).compute_summary_stats()
print(weights)
print(aggregate.select("date", "return_column", "weighted_return", "n_obs"))
print(summary.total_return)
```

Magnitude weights default to `demeaned=True`, subtracting the factor mean and
normalizing absolute exposure to one. With `equal_weight=True`, the method
subtracts the median, assigns signs, normalizes each active sign side and then
normalizes gross exposure. Exact zeros and median ties stay zero. If only one
side remains after median ties, equal-sign weights are one-sided, not dollar
neutral. With `demeaned=False`, signed magnitude or sign weights retain their
net exposure.

`group_adjust=True` first applies that rule within date/group, then normalizes
across the date. Active groups receive equal gross exposure; inactive groups
remain zero. This changes factor weights, unlike the date/group return
demeaning used by IC and quantile statistics. It does not infer group-neutral
holdings from group-relative bucket assignment.

Weights are dimensionless exposures. `weighted_return` contains decimal
supplied-horizon contributions, or their date-level sum with at least one
defined contribution. `n_obs` counts defined contributions, including defined
zero contributions. Unavailable future outcomes do not trigger weight
reallocation; an entirely unavailable horizon/date has null return and zero
count. The direct `signal.compute_factor_weights` route needs no forward
returns. It and `signal.compute_factor_returns` require unique date/asset keys
and finite factors rather than silently cleaning or filtering a weight universe.

Zero signal produces zero weights and, when outcomes are finite, zero
contributions. Alphalens 0.4.6 can return undefined weights for zero dispersion.
The reference maintenance repair in
[Alphalens PR 48](https://github.com/stefan-jansen/alphalens-reloaded/pull/48)
corrects that behavior. Diagnostic follows the independently checked zero
exposure rule; it does not claim those outputs match released 0.4.6.
Normalization also rescales finite magnitudes and removes a common offset
before centering. This avoids overflow or lost centered exposure for extreme
or near-constant scores; raw reference differences are retained in the
shared-input qualification rather than hidden by summary correlations.

The example passes a one-horizon arithmetic series into the existing
`PortfolioAnalysis` consumer. The illustrative `periods_per_year=252` is an
explicit cadence assumption, not inferred from the label `1D` or validated by
an exchange calendar. Overlapping supplied horizons are not automatically a
self-financing daily wealth path. Output dates are observed dates unless the
caller supplies an explicit aggregate `date_grid`; missing requested dates
have null return and zero count. Capital, signal expirations, dated holdings
and the complete Alphalens/Pyfolio reports remain separate work.

### Inspect factor alpha and beta on the same universe

Unreleased after 0.1.9, use the same `panel` to inspect a factor portfolio's
regression against its supplied asset universe. Declare annualization rather
than inferring a duration from a column label:

```python
regression = panel.factor_alpha_beta(
    annualization_factors={"1D": 252},
    demeaned=True,
    group_adjust=True,
    equal_weight=False,
)
print(regression.statistics)
print(regression.observations)
restored_regression = type(regression).from_json(regression.to_json())
assert restored_regression.observations.equals(regression.observations)
```

The independent variable is the equal-asset mean of each supplied horizon in
the same input sample, not an external index or a group-neutral benchmark.
The dependent variable reuses `factor_returns` with the selected weighting
options. The observation table retains both dated returns, actual finite
asset counts, the joint finite-date mask, fitted returns and residuals.
Statistics retain pair counts, scaled-design rank/condition number, period alpha,
beta, compound annual alpha and availability statuses. Beta is dimensionless;
both alpha fields are decimal returns at their stated period/annual scale.
The existing OLS kernel runs in unit-max-absolute coordinates and restores the
coefficient units afterwards; `fit_coordinate_policy` identifies those units
for the condition diagnostics. A finite universe mean that overflows its first
aggregation is recomputed with the existing stable sample-statistics helper.
Neither adjustment removes an otherwise finite date. Price-derived tests retain
three joint dates for identical asset returns `[1e308, -0.5, 0.5]`, with known
beta 1 and alpha 0, and preserve a supplied portfolio's tiny beta `1e-308`.
On those same six date/asset observations and weights of 0.5, released
Alphalens 0.4.6 with pandas 2.3.3 retains the portfolio returns but overflows
the first universe mean to infinity. Its regression raises `MissingDataError`.
The independent expected mean of two identical finite returns is that return;
regressing the same nonconstant series on itself has beta 1 and intercept 0.
This stress-boundary reconciliation does not establish full reference parity.

To reproduce Alphalens 0.4.6's fixed 252-day annualization convention for known
one-day and five-day outcomes, explicitly supply `{"1D": 252, "5D": 50.4}`.
This does not infer exchange sessions, validate bars or annualize overlapping
holdings. The formula is `expm1(log1p(alpha_per_period) * scale)`, unlike the
existing generic `alpha_beta` metric's unchanged linear scaling. In a real
price-derived two-asset test, period alpha `0.001` and beta `0.6` give annual
alpha `0.28643404437618775` at scale 252, not `0.252`.

Supply `returns=` to regress caller-computed portfolio returns instead. Use a
wide pandas/Polars table with the same horizon columns and date keys, or an
indexed pandas Series for one horizon. Dates align by identity, never row
position. Supplied weighting flags are then unused and metadata records no
applied weight options. Extra portfolio dates remain visible with unavailable
benchmark values; missing horizons retain their own joint samples. Source
Series names, input frames, owned panel data and result snapshots stay unchanged.
The direct `signal.compute_factor_alpha_beta` also accepts a raw unique
date/asset panel and explicit return-column selectors without re-cleaning it.

Unlike released Alphalens 0.4.6, an entirely missing portfolio date remains
null, not a fabricated zero. Zero signal follows Diagnostic's checked
zero-exposure rule. A constant benchmark leaves both coefficients unavailable;
the released reference's constant-zero design can return a minimum-norm beta0
even though the slope is unidentified. Fewer than two joint dates also remain
unavailable. Negative gross intercepts or compound overflow retain defined
period coefficients but no annual alpha, with a separate reason. The reference
can apply integer powers to negative gross intercepts; Diagnostic does not
interpret that algebraic value as a compounded investment return.

Run the current comparison command above with `--task all` to check actual
Alphalens 0.4.6 and Diagnostic weights, dated returns, universe means, membership,
counts and alpha/beta for all eight weighting variants on one shared
price-generated 600-row sample with dated unequal groups and two horizons.
It also checks reverse-ordered provided returns and horizon-specific missingness.
These comparisons cover this numerical workflow, not all input/reference
options or the complete returns/full tear sheets. No t-statistic is synthesized
from the reference docstring; its actual alpha/beta table contains no t-statistic.

### Compound selected factor returns without inferring holdings

Unreleased after 0.1.9, compose the existing factor weights and returns with the
shared NaN-aware cumulative product. Choose one exact supplied return column:

```python
wealth = panel.factor_cumulative_returns(
    return_column="1D",
    demeaned=True,
    group_adjust=False,
    equal_weight=False,
    quantiles=[1],
)
print(wealth.select("date", "weighted_return", "n_obs", "wealth"))
```

`quantiles=` and `groups=` select rows before mean/median centering and gross
normalization. They are not filters on a portfolio constructed from the full
universe. The direct `signal.compute_factor_cumulative_returns` route also
accepts pandas indexed or native flat data and explicit physical column names.
Return selectors such as `3h15m` are exact column names, not inferred durations.

`wealth` starts at one and compounds in observation order. It is a wealth
multiple, not an excess-return percentage or an annualized statistic.
`weighted_return` remains a decimal supplied-horizon return; `n_obs` counts
defined contributions, including zero exposure. An entirely unavailable return
or an unscored requested grid date stays null with count zero. The explicit
`missing_return_policy="flat_wealth"` retains the previous wealth value rather
than claiming an observed zero return. Empty selections create no artificial
initial-capital row. Constant/singleton centered signals retain zero exposure;
the released Alphalens undefined-weight difference described above still applies.

Actual Alphalens 0.4.6 comparisons match the selected memberships, finite
weights, contributions and Empyrical 0.5.12 compounding with a matching clock.
Without date-level frequency metadata, its `asfreq(None)` can expand civil-day
dates or collapse a subday sample. Diagnostic defaults to observed keys; an
explicit `date_grid` reproduces a requested reference clock, including null
holes, and labels `compounding_clock="explicit_grid"` instead of `"observed"`.
Neither behavior validates sessions or bars. Compounding overlapping
forward outcomes is a mathematical horizon-return path, not self-financing
daily holdings. Explicit synthetic expiration accounting is available below;
cumulative quantile report panels and complete Alphalens/Pyfolio report coverage
remain separate accepted work.

### Turn explicit expiries into synthetic portfolio inputs

Unreleased after 0.1.9, `panel.positions()` consumes supplied endpoint keys and
uses the existing factor-weight kernel. Group and bucket selections occur before
sizing. Its owned `allocations`, `positions` and `timeline` retain each allocation
ID, exact entry/expiry key, active count and exposure status. At each unique union
event, every due allocation expires and coincident new entries are added before
net-gross normalization. No active allocation, zero signal and offsetting active
allocations are different states. Date labels remain dates; nanosecond instants
are not rounded to Python microseconds.

The native price and long-outcome routes already retain endpoints. Choose a
supplied return column and, by default, `requested_exit_time`; a requested key is
not evidence of an execution. Wide returns require an explicit expiry table:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.signal import prepare_signal_panel
from ml4t.diagnostic.compat.alphalens import performance as al_performance

portfolio_dates = pd.DatetimeIndex([
    "2025-07-04T10:00Z", "2025-07-04T14:00Z", "2025-07-07T10:00Z",
])
portfolio_index = pd.MultiIndex.from_product(
    [portfolio_dates, ["A", "B"]], names=["date", "asset"],
)
portfolio_factor = pd.Series([1., 3.] * 3, index=portfolio_index, name="factor")
portfolio_forward = pd.DataFrame(
    {"2D": [.1, .1, .2, .2, .05, .05]}, index=portfolio_index,
)
portfolio_panel = prepare_signal_panel(
    portfolio_factor, portfolio_forward, quantiles=2, calendar=None,
)
portfolio_expiries = portfolio_factor.reset_index()[["date", "asset"]]
portfolio_expiries["requested_exit_time"] = portfolio_expiries["date"] + pd.Timedelta("2D")
portfolio_native = portfolio_panel.to_portfolio_inputs(
    return_column="2D", expiries=portfolio_expiries, demeaned=False,
)
np.testing.assert_allclose(portfolio_native.returns["return"], [.32, 0., 0., .05])
portfolio_analysis = portfolio_native.analyze(periods_per_year=365)
assert portfolio_analysis.compute_summary_stats().total_return > 0

# Keep the legacy call while migrating imports, without installing Alphalens.
portfolio_legacy_table = portfolio_panel.data.to_pandas().rename(
    columns={"quantile": "factor_quantile"},
).set_index(["date", "asset"])
legacy_returns, legacy_positions, legacy_benchmark = al_performance.create_pyfolio_input(
    portfolio_legacy_table, "2D", long_short=False, calendar=None,
)
np.testing.assert_allclose(legacy_returns, [0., 0., 0., .05])
assert legacy_benchmark is None  # No supplied1Dbenchmark column.
```

For reports with the `viz` extra, call
`portfolio_analysis.create_tear_sheet().save_html("synthetic-portfolio.html")`.
This uses the existing portfolio report and caller-selected annualization scale,
not another metric engine. Returns here are civil-day observations, including
zero weekend changes; the two-day holding duration does not set annualization.
Run the complete price-to-preparation-to-holdings example with
`uv run python examples/signal_panel_cleaning.py`, optionally `--plots` or
`--plots --engineer-outcomes` with the declared optional dependencies. The latter
consumes actual native Engineer outcomes; ordinary use requires no producer package.

The native daily defaults retain the final active holdings each day and the
first wealth change from initial wealth one. Legacy interchange instead uses
`position_method="legacy_event_sum"` and `first_return="zero"`. It daily-sums
event weights and renormalizes: an empty Saturday bin becomes all cash, even
when an allocation is active, and a partial-day Sunday allocation may remain
invested in that day's event sum after its final expiry. These are Alphalens's
mathematical conventions, not executed self-financing holdings. Cash is
`1 - net asset weight`; `capital=` scales holdings by the observed wealth path.
Marks beyond the last return date remain unknown with capital, not zero dollars.
The benchmark uses the original full universe before strategy selections,
with positive equal-sign weights after taking absolute factors; zero factors
receive zero weight. Its independent full-universe date range is retained even
when a strategy selection starts later. `observations` retains the original
pre-resampling outcomes, finite counts and wealth. No transactions, fees or
slippage are fabricated.

The adapters also provide `factor_weights`, `factor_returns` (including
`by_asset=True`), `cumulative_returns`, `positions`, `factor_positions` and
`factor_cumulative_returns`. Corrected Alphalens 0.5.0 comparisons exercise actual
weights, contributions, events, daily returns, cash, benchmark and capital,
including all long-short/group-neutral/equal-weight combinations and selections.
The separate frozen 0.4.6 fixture exposes duplicate coincident expiry keys and
single-pop removal; neither defect is reproduced, even with
`reference_version="0.4.6"`. A test mutating the actual native expiry producer
to remove only one due allocation fails the independent active-set oracle.

Native endpoints are never manufactured from a selector such as `custom` or
`2D`. The legacy `positions(weights, period, freq=...)` adapter alone translates
an explicit duration using its named pandas offset convention. Canonical
`calendar=None`, including declared feed metadata, disables these offsets and
uses elapsed duration; supplying `freq` simultaneously is rejected. Other calendar
declarations remain provenance, not a claim of exchange-schedule validation.
Pandas 2.3 versus 3 civil/elapsed Day differences described above still apply.
Observed offset multiples such as `3B` remain material in the legacy convention:
two declared offset days mean two such offsets, not an inferred two-session hold.
The corrected reference itself rejects an inferred hourly/nanosecond offset in
its daily-offset helper. That default error remains explicit; use `calendar=None`
for the adapter's working elapsed-duration workflow, or supply actual endpoints
through the native route. The tested reference `positions(..., freq=Day())`
subday path remains available. No authoritative trading grid is inferred.
The returned `SyntheticPortfolioInputs` owns Arrow/JSON tables and exports fresh
Pyfolio-shaped tuples. Native asset IDs never collide with separate cash fields;
the legacy wide export rejects an actual asset named `cash`, which that format reserves.

This qualifies the synthetic allocation/interchange route, not every legacy
report, account normalization, round-trip, attribution or capacity workflow.

### Align observed-clock event paths

This how-to replaces the aligned-value workflow of Alphalens 0.4.6
`common_start_returns` and `average_cumulative_return_by_quantile`, not its
complete event tear sheets. Diagnostic's existing abnormal-return
`EventStudyAnalysis` remains a different method.

Pass wide dated prices/levels with `input_kind="levels"`. They are not rebased
at event zero, matching the reference's `cumulative=True` behavior. With
`input_kind="returns"`, the complete supplied clock is compounded from one;
missing simple returns keep wealth flat. Missing supplied levels remain missing.

```python
from pathlib import Path

import pandas as pd

from ml4t.diagnostic.signal import AlignedEventResult, analyze_signal_events
from ml4t.diagnostic.visualization.signal import generate_aligned_event_report

event_dates = pd.bdate_range("2025-01-02", periods=5, tz="UTC")
event_prices = pd.DataFrame({"A": [10.0] * 5, "B": [100.0] * 5}, index=event_dates)
event_input = (
    event_prices.loc[event_dates[[0, 2, 4]]].stack()
    .rename("factor").rename_axis(["date", "asset"])
)
event_result = analyze_signal_events(
    event_input, event_prices, before=1, after=1,
    input_kind="levels", demeaned=False,
)
assert event_result.statistics["mean"].to_list() == [55.0, 55.0, 55.0]
assert event_result.statistics["n_obs"].to_list() == [2, 3, 2]
assert event_result.metadata["event_zero_rebased"] is False
assert AlignedEventResult.from_json(event_result.to_json()).paths.equals(event_result.paths)
generate_aligned_event_report(event_result, output_file=Path("aligned_events.html"))
print("Aligned event levels: 55.0; finite date paths: 2/3/2")
```

For a price-produced panel with dynamic groups and buckets, run the existing
migration example with an explicit task:

```bash
uv run python examples/legacy_migration_comparison.py --task events --output aligned-events.html
```

It checks 24 declared events, 96 requested path cells and a typed JSON roundtrip
before writing the report. It needs Diagnostic's `viz` extra, not Alphalens or
Pyfolio. Existing `--task all` comparisons are unchanged.

An existing owned panel can call `panel.event_study(event_prices, ...)` with
the same options. Its supplied buckets/groups are retained. `demeaned=True`
subtracts the entire event-date universe's level path; `group_adjust=True`
instead uses each event-date group's universe and takes precedence.
`by_group=True` keeps separate group summaries. Date/group means are formed
before their final mean and sample standard deviation. `mean_by_date=False`
instead weights individual events. `paths`, `observations`, `statistics` and
`event_counts` expose these intermediate samples and counts. The two-panel
HTML report shows actual means/sample deviations and event counts; error bars
are not SE or confidence intervals. Rendering completes before an atomic file
replacement, so invalid rendering settings do not overwrite an existing report.

Diagnostic deliberately retains every declared event ID, including repeated
date/asset pairs. Released Alphalens conditionally deduplicates assets in the
unadjusted path. For two A events at level 10 and one B event at 100, Diagnostic
returns 40, whereas the legacy unadjusted date mean is 55. This is an explicit
event-multiplicity difference, not a tolerance adjustment or a reference-error
claim. Boundary-clipped and off-clock event cells remain inspectable as missing;
they do not become zero observations.

Offsets count rows of the supplied value clock, not calendar days or sessions.
No holidays, missing scheduled bars or authoritative session grid are inferred.
Full calendar adoption, all event/report options and the four canonical notebooks
remain separately required. The existing statistical-validation and Backtest
workflows described below are additional capabilities; aligned event paths alone
do not establish complete Alphalens/Pyfolio replacement.

### Complete event-returns report

Unreleased after Diagnostic 0.1.9, `analyze_event_returns` or
`panel.event_returns` prepares the global sample and, with `by_group=True`,
separate group samples using the same event engine. The group panels supplement
the global view. `generate_event_returns_report` renders the global overview,
then one sample-deviation panel per quantile when `std_bar=True`, then group
overviews. This covers the panel selections of Alphalens 0.4.6
`create_event_returns_tear_sheet`; it is not the separate event-study sheet or
all seven Alphalens reports.

```python
from ml4t.diagnostic.signal import EventReturnsResult, analyze_event_returns
from ml4t.diagnostic.visualization.signal import generate_event_returns_report

complete_events = analyze_event_returns(
    event_input, event_prices, before=1, after=1,
    input_kind="levels", demeaned=False,
)
generate_event_returns_report(
    complete_events, std_bar=True, output_file="event_returns.html",
)
event_export = complete_events.to_json()
assert EventReturnsResult.from_json(event_export).overall.paths.equals(
    complete_events.overall.paths
)
print("Event-returns report: global view and per-quantile sample deviations")
```

For the complete grouped, price-produced example and its typed JSON data export:

```bash
uv run python examples/legacy_migration_comparison.py --task event-returns --output event-returns.html
```

The command prints package versions and checks 24 events, 96 requested path
cells, two quantiles and two changing, unequal groups. Its five panels use the
actual overall/group samples; `event-returns.json` retains both samples' paths,
observations, counts, deviations and settings. No Alphalens, Pyfolio or Backtest
runtime dependency is required. `plot_event_returns` returns the individual
figures for notebooks or the existing PDF exporter.

`demeaned` corresponds to reference `long_short`; `group_adjust` corresponds
to `group_neutral` and takes precedence. The report keeps supplied levels or
full-clock wealth in their declared units. The reference plot multiplies its
input by 10,000 for a basis-point display, even when supplied values are prices;
Diagnostic does not relabel those levels as basis-point returns. Demeaned
values are labelled level or wealth differences, not wealth starting at one.
The rendered report states the window and actual demeaning choice. Group plots
and the combined overview show means without deviations; only the optional
separate quantile plots show sample-standard-deviation bars. Clipped and missing
cells, duplicate-event weighting and observation-clock limitations above remain
explicit. Full event-study, calendar and canonical-notebook qualification remain
separate requirements.

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

The published baseline's `SignalResult.to_tear_sheet()` supplies IC and quantile
analysis but not turnover or cumulative quantile-return paths. The unreleased
persistence-enabled workflow above adds actual dated turnover/rank tables and
summaries, not cumulative paths or complete reference chart composition. Do not
infer complete panels from a successful HTML render. Keep Alphalens when those report sections,
dated holdings/expiration accounting, or event-aligned cumulative paths are required.

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
| Complete dated signed holdings | `PortfolioAnalysis(positions=...).compute_holdings_analysis(valuations, complete_snapshots=True)` | Unreleased after 0.1.9. Wide string asset columns or long `date`, `asset`, `value`; supply exact observed keys and reported account values. See [prepared holdings](#analyze-prepared-holdings-on-observed-keys). |
| Fill transactions | `PortfolioAnalysis(transactions=...)` | Currently stored only, not a direct transaction-analysis or fill-to-trade workflow. Passing a table does not enable those analyses. Reconcile fees and executed quantities upstream. |
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

`PortfolioAnalysis` profit factor divides the sum of positive period returns by
the absolute sum of negative period returns, without annualization. In
unreleased source after 0.1.9, a flat sample reports `NaN` for undefined `0/0`,
instead of positive infinity. Gains without losses still produce infinity;
losses without gains produce zero. This return-based ratio does not use
trade-lifecycle cash P&L and is not a Pyfolio metric-parity claim.

The returns-distribution normal reference uses the supplied returns' sample
mean and sample standard deviation, without annualization. Fewer than two
observations or identical returns do not define that fit. The histogram and
quantiles remain available; the plot, dashboard, HTML, notebook and exported
availability explain why the normal curve is absent. Tiny but genuinely varying
returns are not discarded by a dispersion tolerance. This optional normal
reference is not evidence that returns follow a normal distribution.

For dated returns, `plot_monthly_returns_heatmap(portfolio)` retains every
observed year, including zero or tiny returns, without inventing intervening
years. Missing months stay blank, not zero. Pyfolio 0.9.9's plot rounds monthly
returns to three decimal places and fills missing cells with zero for display;
Diagnostic retains the unrounded values and missing cells. Compare the dated
monthly aggregates before those presentation differences.

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

### Analyze prepared holdings on observed keys

In unreleased source after 0.1.9, use the existing `PortfolioAnalysis` workflow
for complete, caller-prepared signed market values in a single account currency.
Supply dated account valuations separately. Reported `equity` is authoritative;
Diagnostic does not infer NAV from cash plus holdings, which need not describe
account equity for every instrument. Missing equity is allowed and leaves only
NAV-dependent ratios unavailable.

Run the existing example without Pyfolio, Backtest or visualization packages:

```bash
uv run python examples/legacy_migration_comparison.py --task holdings
```

The example supplies three dated snapshots, first all-cash and then A=400,
B=-300, cash=900 and NAV=1000. The resulting long/short magnitudes are .4/.3,
gross/net exposures .7/.1, largest absolute-invested-capital share 4/7 and
gross HHI 25/49. Per-asset `nav_weight` preserves signed shorts (-.3).
Pyfolio 0.9.9's `pos.get_long_short_pos()` instead reports short exposure as
a negative number. `pos.get_percent_alloc()` includes cash; Diagnostic's
`gross_share` and concentration exclude cash and use absolute asset values.
These are declared conventions, not interchangeable columns. The example also
checks canonical long inputs with integer asset IDs (A=1, B=2).

Positions may be wide pandas/Polars tables or canonical long `date`, `asset`,
`value` rows. Pandas DatetimeIndex labels are preserved. Wide tables reserve
`date` and optional `cash`; use long format for integer asset IDs or assets
named `asset`/`value`. Values must already be marked market values, not
quantities or unconverted foreign currencies. All frames must cover the same
exact, unique return observation keys and timezone. Intraday keys are not
normalized to civil dates. Different datetime storage precision is accepted
only when the actual keys agree.

`complete_snapshots=True` explicitly declares absent assets zero within each
snapshot; it does not fill missing dates or unknown marks. Long format still
needs a row for every observation, including an explicit zero holding for an
all-cash observation. Wide cash-only rows also represent complete all-cash
snapshots. Null/nonfinite asset marks and duplicate date/asset IDs raise.
Optional wide cash must agree with cash supplied in valuations. Missing cash
stays unknown. Caller inputs are copied before conversion, and every result
owns independent tables.

The typed `HoldingsAnalysisResult` contains `positions`, `timeline`, shared
occupancy `metrics`, `coverage` and caller-declared `snapshot_kind`
(`observed`, `end_of_day` or `intraday`). No kind infers a calendar or validates
a session boundary. Exposure ratios reuse the existing
[Backtest occupancy kernel](#retain-raw-exposure-when-equity-cannot-normalize-it),
which provides a tested reason to evaluate this route alongside Backtest
analysis without a fabricated engine result. Counts and averages use supplied
observations, not elapsed days or inferred sessions.

Zero, negative or missing NAV leaves ratios null while raw values, counts and
gross-capital concentration remain available. All-cash positive NAV has known
zero exposure but undefined concentration, not HHI=0. Inspect normalization
coverage before using averages. `snapshot_kind="intraday"` describes observed
snapshots only; it does not reconstruct missing intraday holdings.

#### Rank holdings across observations

Call `holdings.top_positions(top=10, side="absolute")` on the same result to
rank each asset's largest absolute marked value across the supplied sample.
`side="long"` selects each asset's largest positive value; `side="short"`
selects its most negative value, retaining the negative sign. The returned
`asset`, `value` table uses account-currency values, not NAV weights or
time-weighted averages. An asset that changes side can appear in both signed
rankings. Account cash is excluded; explicit zero-only assets appear only in
the absolute ranking. Ties sort by stable asset ID, independently of input
column order. `top=0` returns a typed empty table; other limits must be
nonnegative integers, excluding booleans.

The existing `--task holdings` example also checks the timeline's
`max_long_weight`, `median_long_weight`, `median_short_weight` and
`min_short_weight`. These dimensionless statistics use the reported NAV and
the strictly positive or negative asset values at each observation. For
A=400/C=100 and B=-300/D=-100 at NAV=1000, they are .4, .25, -.2 and -.3.
Missing/nonpositive NAV or an empty side leaves its statistics null. Cash and
zero assets do not enter either side's median. These signed NAV weights are
distinct from `largest_gross_share` and `gross_hhi`, which use absolute
invested capital and can remain defined without NAV.

Pinned Pyfolio 0.9.9 `pos.get_top_long_short_abs()` and
`pos.get_max_median_position_concentration()` provide the reference routes.
Compare identical marked values and observation keys; Pyfolio derives NAV
from holdings plus cash while Diagnostic uses reported equity. Diagnostic
resolves tied top membership by stable ID; Pyfolio retains input-column order.
Neither route implies a calendar/session conversion or an intraday estimate.

#### Classify holdings by sector

Pass `sector_mapping` to the same `compute_holdings_analysis()` call to request
the typed `sectors` table. The existing `--task holdings` example checks this
with both wide string IDs and canonical integer IDs. A static dictionary,
pandas Series or `asset`, `sector` table applies labels to every observation.
Add a `date` column or pandas DatetimeIndex for exact dated classifications.
Dates use the holdings timezone and lossless storage precision. No calendar
grouping, as-of lookup or forward/backward filling occurs. A future label
cannot classify earlier observations. Provide labels actually known at each
observation; this API does not infer publication history.

Sector long/short magnitudes, gross/net values and nonzero counts come from
the owned holdings. NAV fractions reuse their existing normalization status.
Raw sector totals reconcile to the asset totals even when NAV is missing;
their ratios then remain null. The returned `positions` also carries each
matched sector. Sparse zero groups may be absent from the sector table; the
complete snapshot timeline still supplies every observed date.

Unmapped or missing labels form a visible null-sector group. A literal
`"Unknown"` label stays distinct from null. Empty mappings request unknown
groups for every holding; omitted `sector_mapping=None` does not request sector
analysis. Sector labels may be homogeneous strings or finite numeric codes;
NaN codes become missing labels, not zero. Ambiguous duplicate keys, missing
asset/date keys, incompatible IDs, lossy timestamp conversion and unsupported
extra mapping columns raise without changing inputs.

Account cash remains in `timeline.cash` and `timeline.cash_fraction`, outside
the sector table. A sector literally named `"cash"` can therefore hold net 100
and gross 700 while the account separately reports cash 900. On the tested
A=400/B=-300 example, pinned Pyfolio 0.9.9
`pos.get_sector_exposures()` matches the fully mapped raw net sector totals,
but it overwrites a `"cash"` sector with account cash and omits an unmapped
short (with a warning when the key is absent). Diagnostic retains that short
as null-sector net -300/gross 300. These differences are reconciled against the
independent input values, not treated as parity targets. Cash-only complete
snapshots have an empty requested sector table and unchanged cash in timeline.

The [prepared holdings dashboard](#render-prepared-holdings-in-the-portfolio-dashboard)
renders these sector tables. This route does not provide intraday reconstruction, automatic
per-asset Backtest holdings exports, calendar/session-aware `from_pyfolio()` or
`from_zipline()`, transaction costs/scenarios or complete portfolio reports.
Those remain accepted coverage requirements. No full replacement is claimed.

#### Render prepared holdings in the portfolio dashboard

This unreleased source route is not included in published 0.1.9.
Pass the prepared result to the existing dashboard. Plotly is optional; no
Backtest, Pyfolio or Empyrical runtime is required for caller-prepared inputs:

```python
import pandas as pd
from ml4t.diagnostic.evaluation import PortfolioAnalysis
from ml4t.diagnostic.visualization.portfolio import create_portfolio_dashboard

dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
positions = pd.DataFrame(
    {"A": [0.0, 400.0, 400.0], "B": [0.0, -300.0, -300.0]}, index=dates
)
valuations = pd.DataFrame(
    {"equity": [1000.0] * 3, "cash": [1000.0, 900.0, 900.0]}, index=dates
)
analysis = PortfolioAnalysis(pd.Series([0.0] * 3, index=dates), positions=positions)
holdings = analysis.compute_holdings_analysis(
    valuations, complete_snapshots=True, sector_mapping={"A": "Technology"}
)
report = create_portfolio_dashboard(
    analysis, holdings=holdings, include_transactions=False
)
report.save_html("portfolio-holdings.html", include_plotlyjs=True)
data = report.to_dict()["holdings"]
assert data["timeline"][-1]["cash"] == 900.0
assert data["top_positions"]["short"] == [{"asset": "B", "value": -300.0}]
```

Run the existing example from the source checkout:

```bash
uv run --with plotly python examples/legacy_migration_comparison.py \
  --task holdings-report --output portfolio-holdings.html
```

The report renders exposure, counts, gross-capital concentration, signed
maximum/median NAV weights, top long/short/absolute marked values and requested
sector net/gross exposure. It uses the prepared results without a second
analytical engine. Top charts show up to ten sample extrema, not final holdings
or time-weighted rankings. HHI is a dimensionless concentration index, not a
return or NAV allocation. Account cash and unclassified asset sectors stay
distinct.

When some NAV fractions are unavailable, charts retain their gaps. If none are
defined, exposure/sector charts show raw account-currency values with explicit
units. Undefined concentration or absent sides have an explanation instead of
an empty chart. `report.availability` distinguishes requested unavailable data
from disabled analysis; `include_positions=True` without `holdings` supplies an
actionable preparation reason. Transaction reporting in this dashboard remains
explicitly unavailable, not silently implemented by the flag.

The report owns its prepared snapshot and validates exact observation keys and
timezone. Date labels cannot alias midnight timestamps; timestamp transport
and exported tables preserve nanosecond keys. Browser date axes have their own
display resolution; inspect the original keys in the exported data. The export
contains positions, account timeline, sectors, top tables, coverage and snapshot
kind. HTML uses atomic replacement, leaving an existing report intact if a
write fails. Default-width HTML reflows at phone width. The wide monthly-return
matrix and explicitly fixed-width figures scroll within their panels to keep
values readable without widening the page. `include_plotlyjs=True` embeds the
runtime for offline use; the default `"cdn"` instead needs network access.

In a notebook, `report.show()` displays each figure and the same availability
reasons as HTML. Install IPython and nbformat in the notebook kernel and use a
Plotly-compatible frontend. `report.show(compact=True)` remains the four-panel
returns overview; it does not display the prepared holdings panels.

These are observed prepared-holdings panels, not the complete Pyfolio positions
tear sheet or all portfolio reports. Calendar-aware ingestion, intraday
reconstruction, transactions/cost scenarios, risk/capacity/round-trip workflows
and full Alphalens/Pyfolio composition remain accepted work.

### Retain raw exposure when equity cannot normalize it

For an actual `ml4t-backtest` result, use
`analyze_backtest_result(result).occupancy`. Its `timeline` retains observed
timestamps, equity, cash, gross/net exposure in account-currency units, and open
position counts. Gross and net exposure fractions are dimensionless. Ratios
require finite positive equity and finite exposures; nonpositive equity leaves
them null rather than reporting zero risk. Positive-equity cash-only snapshots
still have known zero exposure. The input/result state is not mutated.

`coverage` reports `n_observations`, `n_normalized_observations`,
`normalization_coverage` and `status`. Normalized averages/maxima use defined
ratios only; position counts and time in market use the observed state. An empty
selected window has no estimates, not cash-only zero estimates. Inspect coverage
before comparing windows. The exposure plot retains gaps and raw-state hover
details, and the full report states its normalization sample explicitly.

The existing migration example runs a real short order through the Backtest
engine, verifies the actual fill, and checks marks of 150, 200 and 250 after
shorting 10 shares at 100 from initial equity 1000:

```bash
uv run --with ml4t-backtest==0.1.12 --with plotly \
  python examples/legacy_migration_comparison.py --task exposure
```

Cash is 2000 after the fill. At the final marks, equity is respectively 500, 0
and -500 while gross exposure is 1500, 2000 and 2500. The first case has gross
ratio 3 and net ratio -3. The other final ratios are unavailable; each window
has two defined snapshots out of three. This capability reuses Backtest's
observed state and Diagnostic's common profile/report analysis.

This is an unreleased correction after 0.1.9, not direct ingestion of a Pyfolio
positions table. For the same positive-equity positions/cash, pinned Pyfolio
0.9.9 `timeseries.gross_lev()` and `pos.get_long_short_pos()` agree on gross/net
ratios. In the zero-equity short example, Pyfolio produces infinite gross
leverage, negative-infinite short allocation and NaN net exposure (the long
allocation is 0/0). At negative equity its division reverses the gross/net signs;
Diagnostic deliberately leaves these ratios unavailable. The full performance
report still rejects simple returns below
-100%, including the mark-250 case. Standalone occupancy support does not remove
that restriction or qualify complete positions, transactions or round-trip
reports.

### Retain executed turnover when equity cannot normalize it

For a Backtest result, `analyze_backtest_result(result).activity` retains raw
fill/rebalance totals and provides `turnover_timeline`, `metrics` and `coverage`.
Its current turnover divides absolute executed notional by the account equity
at the same observed timestamp. Cost drag divides observed implementation cost
by that equity. Nonpositive or unusable equity leaves both ratios null, not
zero. Positive-equity observations without fills retain known zeros when an
activity timeline is available.

`avg_turnover` uses defined active observations (positive filled notional);
`max_turnover` uses all defined observations. Coverage reports both the total
and normalized observation counts and the separate `n_active_observations`,
`n_normalized_active_observations` and `active_normalization_coverage`. Inspect
both sample sizes before comparing windows. Raw execution amounts/counts stay
available when none of the selected ratios are defined. The full report's
Activity strip states both normalization samples, and the Turnover & Costs
panel reuses `visualization.backtest.plot_activity_overview(profile)` to retain
chart gaps and raw execution hover details. Report turnover uses the same
Diagnostic active-observation mean as the profile, even when the producer's
summary uses another convention. The producer's metrics are not mutated.

Run the existing migration example with actual next-bar short and cover fills:

```bash
uv run --with ml4t-backtest==0.1.12 --with plotly \
  python examples/legacy_migration_comparison.py --task activity
```

The short fills 10 shares at 100; the cover fills at 150, 200 or 250. With
initial equity 1000, the final equity is 500, 0 or -500 and closing notional is
1500, 2000 or 2500. The positive-equity close has turnover 3 and active mean 2.
The other closing ratios are unavailable; their active mean is 1 from one of
two active observations. Their three-row windows have two normalized
observations, and the zero-equity example generates the actual full report.
This is an unreleased correction after 0.1.9.

On shared positive-equity inputs, pinned Pyfolio 0.9.9
`txn.get_turnover(..., denominator="portfolio_value")` agrees with the current
per-observation ratios. Its default is `"AGB"`, which averages consecutive
gross books and is a different denominator. Pyfolio also normalizes timestamps
to civil dates and fills NaN ratios with zero; Diagnostic's current bridge
retains exact observation timestamps. This example has one observation per
explicit daily input date. It does not establish exchange-session alignment,
intraday aggregation or AGB support. Invalid-NAV reference divisions are
recorded as differences, not parity targets.

Direct Pyfolio transaction ingestion, complete unknown-fee handling, cost
scenarios, flow adjustments and calendar/session grouping remain accepted work.
The full performance report still requires valid simple returns: a negative-NAV
step below -100%, or a later 0/0 return after NAV reaches zero, is not supported
by the current performance workflow. Standalone activity analysis does not
remove those restrictions.

Pyfolio's full sheet can additionally include return cones, significant
events, round trips reconstructed from fills, capacity analysis using
market data, and optional factor-attribution panels. Diagnostic has related
statistics and reports but no one-call equivalence for that full collection.
Keep Pyfolio or implement the specific analysis you need if those sections
are part of your acceptance criteria. Compare annualization, benchmark
alignment, cash, and cost conventions before comparing metrics.
