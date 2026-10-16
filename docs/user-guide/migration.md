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
| Empirical quantile return distributions | `plot_quantile_returns_violin(result.to_quantile_result(), period=...)` | Unreleased after 0.1.9. Uses actual dated bucket means retained by the analysis/report bridge, with explicit adjustment and percentile options. No summary-derived simulation. Pooled bars and density presentation remain distinct. |
| Dated spread/error chart | `plot_spread_timeseries(result.to_quantile_result(), observations=...)` | Unreleased after 0.1.9. Uses actual dated bucket means and independent propagated errors, with explicit pair, bandwidth and observation-window controls. No cumulative-increment reconstruction or inferred confidence coverage. |
| Arbitrary quantile mean differences and propagated errors | `panel.quantile_spread(upper_quantile=..., lower_quantile=..., ...)` | Unreleased after 0.1.9. Reuses daily/equal-date statistics and propagates independent leg errors with `hypot`. Raw +1 upper / -1 lower exposure is gross two, not a unit-gross portfolio return or a Student test. |
| Factor weights and return contributions | `panel.factor_weights(...)`, `panel.factor_returns(..., by_asset=...)` | Unreleased after 0.1.9. Shares one magnitude/equal-sign/group weight calculation across supplied horizons, with explicit zero-signal handling. Does not construct dated holdings or expiration accounting. |
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
Sub-millisecond observations use a categorical observed-order axis with exact
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
Run the existing supplied-panel example with `--plots` to inspect the actual
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
have null return and zero count. Capital, signal expirations, dated holdings,
factor alpha/beta regression and the complete Alphalens/Pyfolio reports remain
separate work.

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
daily holdings. Capital, expiration accounting, cumulative quantile report
panels and full Alphalens/Pyfolio report workflows remain separate accepted work.

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
