# Migrating an Alphalens or Pyfolio Task

Diagnostic is not a drop-in replacement for either package. Choose a task below,
convert its input shape, and compare result definitions before comparing numbers.
Diagnostic 0.2.0 requires Polars 1.36.1 or newer. The earlier 0.20
minimum lacked APIs already used by prepared signal and factor-weight workflows.
Polars 1.0 also fails existing empty-sample and numerical/report contracts.
The complete signal suite passes at 1.36.1, including actual reference and browser
workflows. This floor also satisfies the tested Backtest integration requirement.
Core examples use synthetic data and the base package. HTML and real-Backtest
tasks name their optional dependencies below.
For selected performance metrics, see the separate
[Empyrical migration assessment](empyrical-migration.md).

## Choose a migration route

Diagnostic 0.2.0 provides the following complete analytical
report families. Start with the familiar adapter to retain a notebook's calling
pattern, or use the native result when you need its samples, validation and
serialization. The adapters share native calculations; they do not install a
second analytical engine or require a Reloaded package at runtime.

`from ml4t.diagnostic import compat` exposes `compat.alphalens`,
`compat.empyrical` and `compat.pyfolio`. Pyfolio loads when requested;
its numerical calls do not require Plotly or a visualization extra.

| Existing workflow | Diagnostic route | Runnable migration |
|---|---|---|
| Alphalens returns, summary, full, information, turnover, event-returns and event-study sheets | `ml4t.diagnostic.compat.alphalens.tears` | [Original Alphalens notebooks](#run-preserved-original-alphalens-notebooks), including the factor-to-Pyfolio journey |
| Pyfolio returns, positions, transactions, simple/full accounts, attribution, capacity, round trips and historical-event sheets | `ml4t.diagnostic.compat.pyfolio` | [Complete account reports](#compose-a-familiar-simple-or-full-account-report) and [original Pyfolio notebooks](#replay-the-original-pyfolio-analytical-notebooks) |
| Empyrical metrics, benchmarks, rolling windows, attribution and tail risk | `ml4t.diagnostic.compat.empyrical` | [Metric methods, units and shape contracts](empyrical-migration.md) |
| Acquisition, factor preparation and execution outputs | Existing Data providers, Engineer outcomes and Backtest account captures | [Acquisition-to-report recipe](#acquire-inputs-with-data-then-analyze-them-in-diagnostic), [price preparation](#prepare-the-same-sample-directly-from-prices) and [actual Backtest capture](#consume-actual-backtest-account-capture) |

Compatibility is analytical, not identical presentation. Figures are Plotly;
reports use HTML/notebook display and explicit export methods instead of mutable
Matplotlib axes, seaborn global styles or printed terminal tables. Requested
panels still require their real inputs: returns alone cannot supply holdings,
executed fills, factor loadings or market volume. Synthetic factor portfolios
remain research outcomes, not executed account P&L. Each recipe below names
its reference definitions, required inputs and reconciled numerical differences.

Native defaults are not silently changed to match legacy definitions. For example,
grouped IC is separate from group-neutral weights, an observation lag is not an
exchange-session lag, and periodic risk-free returns are not annual yield quotes.
Use the adapter's named reference profile or the native explicit method described
in the relevant section. The published 0.1.9 wheel does not include these new routes.

### Continue joint rolling-factor regression

For Pyfolio's `timeseries.rolling_regression`, change the import to
`ml4t.diagnostic.compat.pyfolio.timeseries`. This fits all factors jointly;
separate rolling beta calls do not remove confounding between factors.
Run the supplied example without a Reloaded dependency:

```bash
uv run python examples/pyfolio_rolling_regression.py
```

The recipe has seven fitted windows with final coefficients `alpha=0.001`,
`market=1.5`, `style=-0.75`. Alpha is a periodic decimal return and betas are
dimensionless. The executable reference cases compare Pyfolio 0.10.0 and
Empyrical 0.7.0 on pandas 2.3.3, including nonconstant fitted coefficients,
missing returns, reversed factor input order and rank-deficient designs.

The familiar `rolling_window=3` selects four observations, including both
endpoints. Missing returns are removed before selecting each interval. The
adapter checks each factor's missing fraction against `nan_threshold`, then
fits only rows complete across all factors in that interval. Missing factor
date labels raise; they do not silently compress the clock. Warmup remains
undefined on the surviving return index. Constant or collinear factor fits use
centered minimum-norm coefficients; a single complete row has zero slopes and
its return as intercept. These are estimates, not evidence of identified risk
exposures. Output coefficients are numeric Float64, rather than the reference's
initially object-typed table. Invalid thresholds and ambiguous duplicate keys
raise before caller inputs change.

Native `evaluation.factor.compute_rolling_exposures` remains available for
its explicit `FactorData` inputs, expanding windows, R-squared and stability
diagnostics. Its default window counts the complete rows after native input
filtering, and singular windows remain unavailable. Neither route changes
native annualization or silently subtracts a risk-free series. Supply the
intended raw or excess response and factors explicitly.

The same `timeseries` adapter exposes the original scalar metrics, alpha/beta,
`cum_returns` and `aggregate_returns` through the existing Empyrical adapters.
`omega_ratio` retains the original `annual_return_threshhold` spelling and its
annual threshold convention. `common_sense_ratio` follows the original code,
`tail_ratio * (1 + annual_return)`, despite its old gain-to-pain description.
`normalize` divides a supplied path by its first row; it does not compound
periodic returns. `var_cov_var_normal` delegates to the native Gaussian cash
VaR definition and validates its nondegenerate confidence/scale inputs. These
routes do not alter the native defaults described in the metric guide.

### Replace presentation and error helpers

These supporting helpers do not require another analytical implementation.
Preserve the calculations and change the surrounding presentation or error handling:

| Legacy helper | Migration |
|---|---|
| `plotting.customize`, `plotting_context`, `axes_style` | Remove the Matplotlib/seaborn decorator or context manager. Select `theme=` on the report, or update the returned Plotly figure's layout. No global Matplotlib style is changed. |
| `tears.GridFigure` | Use the complete report composer or the returned Plotly figures. The Matplotlib subplot-allocation class is not reproduced. |
| `utils.print_table` | For terminal output use `table.to_string()`. For notebooks use `IPython.display.HTML(table.to_html(escape=True))`; select numeric display formatting explicitly. This changes presentation, not the table's values or units. |
| `utils.rethrow` | Inside the original exception handler call `error.add_note(context)` and use a bare `raise`. This preserves the exception's type and cause instead of wrapping or suppressing it; the contextual note is separate from its original message. |

The preparation adapter retains `MaxLossExceededError`, `NonMatchingTimezoneError`
and `non_unique_bin_edges_error`. Internal reference helpers for NaN reductions,
IC column selection and frequency padding are implementation details of the
native calculations and adapters, not additional public services to migrate.

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

The current checkout's script also exercises 0.2.0 prepared-input routes.
Do not combine it with the older installed package. From the current checkout,
run the complete source comparison instead:

```bash
uv run --with 'pyfolio-reloaded==0.9.9' --with 'alphalens-reloaded==0.4.6' \
  --with 'empyrical-reloaded==0.5.12' --with 'pandas==2.3.3' \
  python examples/legacy_migration_comparison.py --task all
```

## Acquire inputs with Data, then analyze them in Diagnostic

Use this route when migrating an Empyrical/Pyfolio notebook that downloaded
prices or Fama-French factors before analyzing returns. Acquisition and caching
remain in Data. Diagnostic consumes the returned frames through
`PortfolioAnalysis.from_frames` and the familiar complete report adapter; it
does not implement another provider, cache, broker or returns engine.

The recipe uses Data 0.3.0, yfinance 1.7.0 and pandas 3.0.6. Its tests replay
pinned authentic source responses through Data's actual providers and caches,
then consume those outputs through Diagnostic's complete reports. Captured
responses do not establish live service availability or historical publication
instants.

From a source checkout containing this example, replay the bundled,
provider-derived CSVs without Data, Reloaded packages or network access:

```bash
uv run python examples/acquisition_to_portfolio_report.py --output acquisition-report.html
```

For an explicit live request, install Data's optional Yahoo dependency and run:

```bash
uv run --with 'ml4t-data[yahoo]==0.3.0' python examples/acquisition_to_portfolio_report.py --fetch --symbol SPY --start 2024-01-02 --end 2024-01-06 --output acquisition-report.html
```

The original acquisition names live in `ml4t.data.compat.empyrical`:
`get_symbol_returns_from_yahoo`, `get_fama_french`, `get_treasury_yield` and
`get_returns_cached`. The example delegates to them, then calls Diagnostic's
`create_full_tear_sheet` with the supplied returns and benchmark. It also
returns a native `PortfolioAnalysis` for further analysis without a report.

Yahoo's request end is exclusive; the Fama-French and FRED bounds are inclusive.
Yahoo uses adjusted closes and preserves missing observations. The CSV cache
names include symbol and requested bounds. Its freshness follows Data's file
mtime contract, not historical publication time. FRED is fetched explicitly on
each `--fetch` call; it is not part of those CSV caches. For reproducible offline
analysis, retain acquired inputs or use the bundled captured-input route.

Data's native Parquet factor cache is separate from the legacy CSV helper.
The producer regressions consume its actual returned factor frames through
the same full report after miss, hit, refresh and offline stale-policy checks.
`FamaFrenchProvider.fetch` requires explicit `allow_stale=True` to accept an
offline cache older than the requested `max_age`.

Fama-French `Mkt-RF`, `RF` and factor columns are decimal periodic returns.
The total-market benchmark is `Mkt-RF + RF`, not market excess alone. The separate
FRED `DGS3MO` column is an annual-percent quoted yield. A quote of `5.46` is not
a daily return of `0.0546`, and the recipe does not automatically turn it into a
periodic risk-free series. The report retains the familiar zero-RF statistics;
an additional labelled Sharpe uses the actual dated periodic `RF` observations.
Conflicting declared units raise rather than silently relabelling the input.

The captured example produces three return observations, four factor rows and
three Treasury quotes. It does not create a Treasury observation on January 5.
Ending wealth is independently determined by adjusted prices:
`453.09259033203125 / 457.67266845703125 = 0.9899926772108962`.
The dated-RF Sharpe is approximately `-11.7810178587`, dimensionless at 252
observations per year. Missing positions and transactions remain not requested;
no portfolio holdings or executed fills are inferred from a price series.

This route adds native dated-input validation and a reusable native analysis
object to the familiar report workflow. See the existing dated portfolio-input
and complete-account sections for the independently exercised input validation,
annualization and export contracts. It is not a claim of live-provider fidelity,
historical availability or complete Reloaded replacement.

## Alphalens factor analysis

[Alphalens](https://github.com/quantopian/alphalens/blob/77084f1e4c2c0be407e032d444fb19e4be4b0f37/README.rst) accepts
a factor Series indexed by date and asset plus a wide price DataFrame, then
`get_clean_factor_and_forward_returns()` constructs forward returns and
quantiles. Its full tear sheet includes returns, IC, turnover, and grouped
analysis. Diagnostic's `analyze_signal()` takes two long tables and returns
IC, quantile returns, spread, and turnover metrics. It recomputes forward
returns from prices. Current 0.2.0 source also provides
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
| Already-cleaned factor table | `analyze_prepared_signal(data, ...)` | Accepts the ordinary pandas `(date, asset)` index, `factor_quantile` and `1D`/`5D` columns, or native flat columns. Preserves supplied buckets; rejects duplicate keys and incomplete requested data rather than cleaning it. 0.2.0 after 0.1.9. |
| Unbucketed factors plus supplied returns | `signal.prepare_signal_panel(factor, forward_returns, ...)` | Owns one complete finite sample, then assigns qcut/cut buckets and records factor/return/group/bucketing losses. Supports static or dated groups and group-relative/zero-aware buckets. 0.2.0 after 0.1.9; does not compute returns or validate a calendar. |
| IC, quantile returns, and spread | `SignalResult.ic`, `.quantile_returns`, `.spread` | Specify `periods`, `quantiles`, and `ic_method`; check timestamp alignment and compare each statistic's definition. |
| Grouped or group-adjusted IC observations | `panel.ic_by_date(group_adjust=..., by_group=...)` | 0.2.0 after 0.1.9. Uses supplied returns and reports per-date/group paired counts and undefined reasons. Inspect the shared adjusted outcomes with `panel.demean_returns(by_group=True)`. Does not construct a group-neutral portfolio. |
| Daily quantile means/errors and equal-date summaries | `panel.quantile_statistics(by_date=..., by_group=..., demeaned=..., group_adjust=...)` | 0.2.0 after 0.1.9. Reports sample standard deviation, standard error and asset/date counts; does not change the native pooled `SignalResult` fields or spread inference. |
| Empirical quantile return distributions | `plot_quantile_returns_violin(result.to_quantile_result(), period=...)` | 0.2.0 after 0.1.9. Uses actual dated bucket means retained by the analysis/report bridge, with explicit adjustment and percentile options. No summary-derived simulation. Pooled bars and density presentation remain distinct. |
| Dated spread/error chart | `plot_spread_timeseries(result.to_quantile_result(), observations=...)` | 0.2.0 after 0.1.9. Uses actual dated bucket means and independent propagated errors, with explicit pair, bandwidth and observation-window controls. No cumulative-increment reconstruction or inferred confidence coverage. |
| Arbitrary quantile mean differences and propagated errors | `panel.quantile_spread(upper_quantile=..., lower_quantile=..., ...)` | 0.2.0 after 0.1.9. Reuses daily/equal-date statistics and propagates independent leg errors with `hypot`. Raw +1 upper / -1 lower exposure is gross two, not a unit-gross portfolio return or a Student test. |
| Factor weights and return contributions | `panel.factor_weights(...)`, `panel.factor_returns(..., by_asset=...)` | 0.2.0 after 0.1.9. Shares one magnitude/equal-sign/group weight calculation across supplied horizons, with explicit zero-signal handling. Does not construct dated holdings or expiration accounting. |
| Factor alpha and beta | `panel.factor_alpha_beta(annualization_factors=...)` | 0.2.0 after 0.1.9. Reuses factor returns and OLS against the same universe's equal-asset mean. Retains dated joint samples/counts and explicitly compounded annual alpha; constant benchmarks and unavailable dates remain explicit. The owned returns/report composition below includes these samples and tables. |
| Selected-factor cumulative wealth | `panel.factor_cumulative_returns(return_column=..., quantiles=..., groups=...)` | 0.2.0 after 0.1.9. Selects rows before shared weighting, then compounds one supplied horizon in observation order. Retains actual returns/counts and a labelled flat-wealth policy for unavailable returns. Not self-financing daily holdings or an inferred calendar. |
| Quantile turnover | `SignalResult.turnover` | From 0.1.9, each horizon key uses that lag in sorted observed factor dates. 0.2.0 after 0.1.9, persistence summaries also preserve sub-microsecond observation identity. The direct `compute_turnover(..., lag=...)` helper defaults to one observation; neither route establishes calendar-aware turnover. |
| Dated turnover by bucket | `panel.quantile_turnover(lags=..., quantiles=..., date_grid=...)` | 0.2.0 after 0.1.9. Reports entrants/current-membership fractions, leg/universe counts and unavailable statuses on an observed or explicitly supplied clock. The scalar larger-membership definition remains separate. |
| Dated rank persistence | `panel.rank_autocorrelation(lags=..., min_obs=..., date_grid=...)` | 0.2.0 after 0.1.9. Ranks each full scored date before pairing assets and computing Pearson correlation. Retains counts, statuses and explicit clocks; native paired-sample Spearman summaries remain separate. |
| Factor HTML report | `SignalResult.to_tear_sheet()` and `SignalDashboard` for selected components; `generate_signal_report(...)` for owned composition | 0.2.0 after 0.1.9. The owned composer renders actual returns, information and dated persistence results, including requested mathematical quantile paths. Familiar report routes and their option/sample differences are described below. |
| Expiring allocations and daily portfolio interchange | `panel.positions(...)`, `panel.to_portfolio_inputs(...)`, `compat.alphalens.performance.create_pyfolio_input(...)` | 0.2.0 after 0.1.9. Explicit expiry, capital and daily conventions remain distinct from executed holdings/account P&L. Native end-of-day and familiar event-sum definitions are named separately. |

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
0.2.0 input-boundary changes after 0.1.9, not calendar/grid validation.
For metrics-layer IC, decay and feature-outcome panels, pass the asset key as
`group_col`; `None` denotes one series per date. These workflows validate
prediction keys and forward the declared `date_col` to the return producer,
including custom time-column names.
Equivalent millisecond, microsecond and nanosecond datetime keys in the same
timezone join at the finer unit without rounding. An unrepresentable conversion
raises instead of wrapping timestamps; this does not infer a timezone policy or
validate calendar sessions.

For exact integer-count qcut membership, 0.2.0 source after 0.1.9 accepts
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
This shortcut does not itself compose a complete report. For that workflow use
the [owned result composition](#compose-owned-return-information-and-turnover-results)
or the familiar tear-sheet adapters listed above.

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

0.2.0 source after 0.1.9 retains the mean and positive fraction of a
single defined date-level IC. Its sample standard deviation and t-test remain
undefined. An empty IC sample has no positive-observation fraction, and a
zero/undefined IC standard deviation has no IC information ratio; neither is
reported as a fabricated zero. Signal results express positive observations
as a percentage, while the direct metric summary uses a fraction. JSON uses
null for unavailable values and restores them as NaN on the result round trip.

### Match IC report statistics to selected dates

0.2.0 after 0.1.9, `result.to_ic_result()` and the tear-sheet bridge compute
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
a complete dated-status/count export or a union-clock report. Use the
[complete information analysis](#complete-information-analysis-report) for its
owned samples and the [shared time contract](cv-configuration.md) for calendar selection. The
[runnable supplied-panel example](https://github.com/ml4t/diagnostic/blob/0ef65182e33fb661cb9cb8ccaf3aa0d01d123a63/examples/signal_panel_cleaning.py)
checks report means against their actual series and prints the report summaries.

### Inspect monthly IC on the selected sample

With the visualization extra, 0.2.0 source after 0.1.9 plots arithmetic
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

For an unbucketed factor Series and a precomputed return panel, 0.2.0
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
buckets alone does not provide group-neutral inference. 0.2.0 after 0.1.9,
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
differences below still apply. Complete sheets use the explicitly selected
methods in the [owned report composition](#compose-owned-return-information-and-turnover-results).

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

0.2.0 after 0.1.9, `prepare_signal_panel(factor, prices=prices,
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

0.2.0 after 0.1.9, change the preparation import while keeping the pandas
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
do not establish every performance/plotting adapter or original notebook workflow.
The report compositions below reuse this preparation without another cleaner.
The keyword-only `reference_version` defaults to `"0.5.0"`; `"0.4.6"` explicitly
selects the frozen profile without global state. Actual 0.4.6 comparisons match
daily price returns, retained keys and buckets for cumulative and terminal-bar
outcomes. Historical intraday labels are corrected, not copied: on the shared
regular 30-minute input, 0.4.6 calls one observation `1D30m` and two `2D1h`.
The independently measured endpoints are 30 minutes and one hour, so Diagnostic
retains `30m` and `1h`. Nanosecond, overnight and DST-fold cases likewise retain
their actual subday lengths rather than adding one fictitious day per observation.
Return values and sample/bucket membership match after reconciling those labels.
The historical intraday frequency-assignment failure is not reproduced either.
The installed preparation profile uses Python 3.12.11, pandas 3.0.6, NumPy 2.5.4,
Polars 2.0.0, SciPy 1.18.1 and Plotly 7.1.0. A separate consumer/reference profile
uses pandas 2.3.3, NumPy 2.2.6, Polars 1.36.1 and SciPy 1.17.0. The pinned Engineer
wheel requires Polars 2, so its live regeneration is not claimed at that floor;
standalone ingestion of its frozen captures needs no Engineer installation.

Published Engineer 0.2.0 also regenerates the retained nanosecond, missing-price,
cumulative/exit-bar, session-label, expected-bar, overnight and early-close
captures. Their original fields, keys, endpoints and values match exactly.
That producer adds six Boolean row/price-observation flags. Diagnostic preserves
them in `outcome_data` and JSON rather than dropping them or treating them as
additional return horizons. The historical capture metadata continues to name
its original development producer; it is not relabelled as a 0.2.0 capture.
The dedicated published-producer checks require all twelve actual cases, with
no optional-producer skips accepted in that profile.

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

### Keep familiar IC and quantile-statistics calls

0.2.0 after 0.1.9, the analytical adapters reuse the native IC, equal-date
quantile statistics and propagated-error kernels. Continue from `adapter_clean`
above with a changed import:

```python
# Previously: from alphalens import performance
from ml4t.diagnostic.compat.alphalens import performance

adapter_ic = performance.factor_information_coefficient(adapter_clean)
adapter_mean_ic = performance.mean_information_coefficient(adapter_clean)
adapter_means, adapter_errors = performance.mean_return_by_quantile(
    adapter_clean, by_date=True, demeaned=False,
)
adapter_spread, adapter_spread_error = performance.compute_mean_returns_spread(
    adapter_means, 2, 1, adapter_errors,
)
assert np.allclose(adapter_ic["1D"], 1)
assert np.allclose(adapter_spread["1D"], 0.05)
assert adapter_spread_error["1D"].isna().all()
print("Alphalens statistical adapters: IC and explicit spread exposure checked")
```

IC and its mean/deviation are dimensionless. Quantile means and spreads are
decimal returns; their errors are not fabricated for singleton samples.
Corrected `reference_version="0.5.0"` returns a unit-gross spread, one half of
the upper-minus-lower return and propagated error. Explicit `"0.4.6"` retains
the historical raw gross-two difference. Native `compute_quantile_spread`
keeps its existing raw-difference default. No input is re-bucketed or mutated.

The same namespace exposes `factor_alpha_beta`, using the native same-universe
regression and compounded intercept. The following supplied outcomes define an
independently known beta of 0.6 and a per-observation intercept of 0.001:

```python
regression_dates = pd.bdate_range("2025-02-03", periods=3, tz="UTC")
regression_index = pd.MultiIndex.from_product(
    [regression_dates, ["A", "B"]], names=["date", "asset"],
)
universe_returns = np.array([-0.03, 0.01, 0.04])
regression_data = pd.DataFrame({
    "factor": np.tile([1.0, 0.0], 3),
    "1D": np.column_stack([
        0.001 + 0.6 * universe_returns,
        -0.001 + 1.4 * universe_returns,
    ]).ravel(),
}, index=regression_index)
adapter_regression = performance.factor_alpha_beta(regression_data, demeaned=False)
assert np.isclose(adapter_regression.loc["beta", "1D"], 0.6)
assert np.isclose(
    adapter_regression.loc["Ann. alpha", "1D"], np.expm1(np.log1p(0.001) * 252),
)
print("Familiar factor regression: same-universe beta and compounded annual alpha")
```

Legacy duration labels default to a 252-day year, so `5D` uses 50.4, not 252.
Explicit `annualization_factors={...}` selects other horizon scales. Supplied
`returns=` bypasses factor weighting and may omit the unused factor column;
dates align by key, and a Series selects the first forward-return column without
renaming the caller's Series. The returned pandas rows are `Ann. alpha` and
`beta`. Neither reference actually supplies its docstring's claimed t-statistic.
All-missing outcomes and constant benchmarks remain unidentified rather than
copying frozen zero-return or zero-beta behavior. Use the
[owned native regression](#inspect-factor-alpha-and-beta-on-the-same-universe)
to inspect paired observations, finite counts, rank and unavailability reasons.

The overview notebook's direct persistence calls also use the native membership
and full-date rank kernels:

```python
adapter_membership = performance.quantile_turnover(adapter_clean["factor_quantile"], 1)
adapter_rank = performance.factor_rank_autocorrelation(adapter_clean, period=1)
assert np.allclose(adapter_membership.iloc[1:], 0)
assert np.allclose(adapter_rank.iloc[1:], 1)
print("Familiar persistence: actual bucket membership and full-date ranks")
```

With a declared index frequency, shifts count positions on that grid. Without
one, the corrected profile uses observed bucket dates for turnover and observed
factor dates for rank correlation. The explicit frozen `reference_version="0.4.6"`
profile inserts its historical daily grid. Missing comparisons remain unknown.
Zero and negative legacy shifts retain self and future comparisons; they do not
establish causal features. Native public persistence still requires positive lags.
Pandas frequency metadata and all-unavailable object dtype follow the legacy
sparse/full-grid alignment; native typed counts and statuses remain available
through the separately named panel methods.

The original overview's direct charts use the same analytical tables:

```python
from ml4t.diagnostic.compat.alphalens import plotting as familiar_plots

adapter_bar = familiar_plots.plot_quantile_returns_bar(
    performance.mean_return_by_quantile(adapter_clean, demeaned=False)[0],
)
assert np.allclose(adapter_bar.data[0].y, [1000, 2000])
adapter_bar.write_html("familiar_quantile_means.html", include_plotlyjs=True)
adapter_ic_figures = familiar_plots.plot_ic_ts(adapter_ic)
assert len(adapter_ic_figures) == 2
adapter_ic_figures[0].write_html("familiar_ic.html", include_plotlyjs=True)
print("Familiar overview charts: supplied samples, bps and explicit Plotly exports")
```

These functions return Plotly figures, or a list for per-horizon/group panels;
use `.show()` explicitly in notebook cells ending with a semicolon. Matplotlib
`ax` reuse raises before output. Bar and violin ordinates are bps. Cumulative
figures compound supplied raw observations into wealth multiples with the shared
product kernel; their linear display replaces the reference's symmetric-log
quantile axis. They do not infer daily execution accounts from a return label.
Unknown group/bucket means remain absent, rather than copying the zero-height
bars produced by pandas/Matplotlib for missing cells. No sample counts or standard
errors are invented from supplied means. Native IC histograms use 25 bins, rather
than the reference's automatic bin selection; the empirical missing-value mask
is retained. Q-Q, rolling IC and month heatmaps reuse the existing native charts.
The original `by_time="M"` call translates to an explicit month-end offset on
pandas 3; it preserves the observation's civil month, not an exchange calendar.
The four direct table functions use the same native report statistics:

```python
adapter_factor_table = familiar_plots.plot_quantile_statistics_table(
    adapter_clean, return_df=True,
)
adapter_information_table = familiar_plots.plot_information_table(adapter_ic, return_df=True)
adapter_return_table = familiar_plots.plot_returns_table(
    adapter_regression,
    performance.mean_return_by_quantile(adapter_clean, demeaned=False)[0],
    adapter_spread,
    return_df=True,
)
adapter_turnover_table, adapter_rank_table = familiar_plots.plot_turnover_table(
    {1: adapter_rank}, {1: pd.concat([adapter_membership], axis=1)}, return_df=True,
)
assert adapter_factor_table["count"].sum() == len(adapter_clean)
assert adapter_information_table.shape == (2, 7)
assert adapter_return_table.loc["beta", "1D"] == adapter_regression.loc["beta", "1D"]
assert adapter_turnover_table.iloc[0, 0] == 0
print("Familiar tables: original labels, actual counts and unmodified regression units")
```

`return_df=True` returns pandas tables; the default returns escaped HTML strings,
or two strings for turnover. Display them with `IPython.display.HTML` in a notebook
or include them in a report. These functions do not reproduce terminal printouts
or Matplotlib artists. Alpha and beta retain their supplied units; only the supplied
quantile means and spread convert to bps. The return-table `reference_version`
labels the caller's existing spread convention without rescaling it. Native IC
summaries omit missing dates separately per column. Unlike frozen 0.4.6, a missing
IC does not invalidate the remaining dates' inference. Constant and empty samples
keep unavailable inference rather than reporting infinite significance.

Direct cumulative, spread and persistence figures retain exact sub-millisecond
timestamps with labelled observation-order axes and exact timestamps on hover.
Distinct nanosecond keys are not passed through JavaScript's date conversion.
Group titles wrap, horizons/quantiles remain visible in legends, and supplied
mean/std event figures keep unknown deviations separate from known zero deviations.
These checks cover the declared report options; qualification of every original
journey also requires the installed-package notebook checks below.

### Run preserved original Alphalens notebooks

0.2.0 source includes `scripts/migrate_alphalens_notebooks.py` for the
original overview, intraday-factor, event-study and Pyfolio-integration notebooks. The unmodified
upstream fixtures and their Apache 2.0 license are retained under
`tests/fixtures/alphalens_original/`, pinned to Alphalens source
`1bc181de4aafbe33437ed0e362ff96aa7ce20acd`. The generator checks the original
hash, records every import/I/O/missing-value-method/presentation edit and retains
the original analytical expressions. It replaces the obsolete Yahoo acquisition call, not the
subsequent date conversion or stack operations. It does not download data.

Supply a captured parquet file with unique date rows and Yahoo-shaped
`(field, asset)` columns containing `Open` and `Close`. Those input dates have
the provider's original daily-label convention; the original notebook's explicit
UTC conversion remains in place. A long Data/Engineer frame must first be
explicitly reshaped into that source-provider form. This narrow fixture contract
is not a new requirement on Diagnostic's native long-frame analytical APIs.

From a checkout of this matching source revision, run:

```bash
uv run --with nbclient --with ipykernel python scripts/migrate_alphalens_notebooks.py --source-notebook tests/fixtures/alphalens_original/overview.ipynb --ohlc captured_ohlc.parquet --output migrated_overview.ipynb --execute
```

Use the corresponding `intraday_factor.ipynb`, `event_study.ipynb` or
`pyfolio_integration.ipynb` source and
a new output filename for those journeys. Existing outputs and the original
notebook are never overwritten. Modified outputs include a visible migration
notice, the original license and the exact edit record. Optional Jupyter tools
execute the generated notebook; no Alphalens, Empyrical or Pyfolio installation
is required by the migrated analyses.

Pandas 3's stack implementation retains missing values that the original
notebooks' default omitted. The recorded `_stack_observed` method drops missing
Series values and all-missing DataFrame rows explicitly, preserving absent-event
membership instead of counting every missing event as a supplied factor. Native
cleaner loss limits and missing-value accounting are unchanged. This converter
requires external inputs; it is a migration tool, not a no-argument demo script.

The tested overview preserves eleven independently compared intermediate tables
and Series against actual Alphalens 0.5.0 on shared captured inputs. Diagnostic
renders 33 Plotly MIME outputs and six HTML reports instead of the reference's
24 Matplotlib images. Standalone installed-runtime tests also exercise the
original intraday and event-study calculations, including their actual forward
endpoints, retained keys, explicit UTC conversion and unchanged input files.
Their numerical methods still use the separately tested native kernels; a
successful notebook alone does not establish every legacy option. The original
Alphalens/Pyfolio integration notebook uses the same native-backed complete
Pyfolio reports for its daily and Monday-only factor selections. Its synthetic
research allocations are labelled separately from execution-account facts.
The legacy five-day offset on the Monday-only sample retains its original
multiple: positions can extend beyond the return clock. It does not declare
five NYSE sessions or justify clipping trailing unknown marks.
Availability in published 0.1.9 is not claimed.

IC preserves legacy per-column NaN propagation. Quantile means instead retain
each supplied outcome's own available sample and give defined date means equal
weight. Raw `compute_forward_returns` plus `quantize_factor` tables may retain
incomplete forward tails; those are not silently cleaned across horizons.
For native statistics on already supplied buckets without a score column, use
`compute_quantile_statistics(..., factor_col=None)`; its default still selects
finite score/outcome pairs. Group adjustment and group output remain independent.

To inspect sample counts and uncertainty alongside these familiar pandas tables,
use the existing [native quantile statistics](#inspect-daily-and-equal-date-quantile-statistics).
For dependence-aware IC inference, use the separately selected
[HAC validation](#complete-information-analysis-report); the adapter's ordinary
IC mean is not a HAC estimate. Complete sheet composition is available through
the [owned results](#compose-owned-return-information-and-turnover-results).

### Inspect grouped IC and adjusted outcomes

Use the same owned panel to compare the observations underlying a grouped IC.
These 0.2.0 methods after 0.1.9 do not require Alphalens. Using `prepared`
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

### Complete information-analysis report

0.2.0 source after 0.1.9 composes the owned panel's full information analysis.
Using `panel` from the grouped-IC example above:

```python
from ml4t.diagnostic.signal import SignalInformationResult, analyze_signal_information
from ml4t.diagnostic.visualization.signal import generate_signal_information_report

information = analyze_signal_information(panel, group_adjust=True, by_group=True)
generate_signal_information_report(information, output_file="signal-information.html")
restored_information = SignalInformationResult.from_json(information.to_json())
assert restored_information.get_table("prepared_data").height == panel.data.height
print("Information report samples:", information.get_table("ic_by_date").height)
```

An already-clean factor table can keep its supplied buckets through the native
`PreparedSignalPanel.from_prepared` constructor or the familiar sheet adapter:

```python
from ml4t.diagnostic.compat import alphalens as al
from ml4t.diagnostic.signal import PreparedSignalPanel

familiar_information = (
    panel.data.to_pandas().set_index(["date", "asset"])
    .rename(columns={"quantile": "factor_quantile"})
)
owned_information = PreparedSignalPanel.from_prepared(
    familiar_information, return_columns=["1D"], calendar=None,
)
assert owned_information.metadata["bucket_source"] == "supplied, not recomputed"
information_figures = al.tears.create_information_tear_sheet(
    familiar_information, group_neutral=True, by_group=True,
    output_file="adapted-information.html", return_fig=True,
)
assert len(information_figures) == 4
print("Alphalens information adapter preserves the prepared sample and group options")
```

The legacy call is `alphalens.tears.create_information_tear_sheet(factor_data,
group_neutral=..., by_group=...)`; change its import to
`ml4t.diagnostic.compat.alphalens.tears`. Default notebook calls display the
complete HTML report and return `None`. The explicit `return_fig=True` extension
returns Plotly figures, not Matplotlib figures or axes. `output_file` saves the
complete report atomically. `set_context` accepts the legacy boolean without
changing global Matplotlib/seaborn state; use `theme` for native presentation.
The same namespace also exposes turnover, event-returns and event-study sheets.
Returns, summary and full-sheet composition use the shared owned results below;
their plotting API and numerical method differences remain explicit.

Continue from the price-produced `adapter_clean` and `adapter_prices` above:

```python
from ml4t.diagnostic.compat.alphalens import tears as familiar_sheets

familiar_turnover = familiar_sheets.create_turnover_tear_sheet(
    adapter_clean, turnover_periods=["1D", "2D"],
    output_file="adapted-turnover.html", return_fig=True,
)
familiar_events = familiar_sheets.create_event_returns_tear_sheet(
    adapter_clean, adapter_prices, avgretplot=(1, 1), long_short=False,
    std_bar=False, output_file="adapted-events.html", return_fig=True,
)
familiar_study = familiar_sheets.create_event_study_tear_sheet(
    adapter_clean, adapter_prices, avgretplot=(1, 1), n_bars=2,
    output_file="adapted-study.html", return_fig=True,
)
assert len(familiar_turnover) == 4 and len(familiar_events) == 1
assert len(familiar_study) == 6
print("Familiar turnover and event sheets retain native samples and supplied levels")
```

The historical event argument is named `returns`, but both reference sheets
pass it to `common_start_returns(cumulative=True)`. These adapters therefore
preserve supplied levels without compounding or event-zero rebasing. For simple
returns, use the native `panel.event_returns(..., input_kind="returns")` instead.
`long_short`, `group_neutral`, `by_group` and `std_bar` retain their analytical
roles; displayed error bars are sample deviations, not confidence intervals.
Reference event plots multiply these values by 10,000 for BPS display;
Diagnostic labels supplied-level or level-difference units. Diagnostic sorts
event offsets chronologically; comparisons use exact offset labels when clipped
reference windows retain concatenation order.
Event study retains unadjusted date/bucket observations, optional level paths
and elapsed-time event bins. `rate_of_ret=True` converts display means to the
first supplied horizon using duration labels; the native route still requires
explicit conversion factors for arbitrary selectors. No path is invented when
`returns` or `avgretplot` is `None` in the event-study call.

Turnover day labels select observed-grid lags, not exchange-validated durations.
Both reference profiles preserve a declared pandas frequency. Without one,
`reference_version="0.5.0"` retains observed dates while `"0.4.6"` selects the
frozen civil-daily reporting grid. Native methods continue to require an explicit
grid when a different output clock is wanted.

Both routes require complete finite prepared factors/outcomes and positive bucket
IDs; they do not silently clean or rebucket a supplied table. The constructor
retains sparse bucket IDs, groups, exact keys and calendar declarations in owned
data/audit copies. It does not manufacture original price endpoints or causal
availability that the supplied table does not contain.

The adapter preserves an input pandas frequency as an explicit output grid,
including unknown ICs on absent factor dates. Preparation retains unused date
levels without inventing factor rows. A missing declared date therefore remains
unknown inside the complete 22-date rolling window, rather than compressing it
into a 22-scored-date window. This is a named legacy reporting convention, not
exchange-calendar validation. Native `analyze_signal_information` still defaults
to observed dates; pass `date_grid` explicitly to request other output keys.

`group_adjust` demeans each date/group's outcomes before correlation. Independently,
`by_group=True` adds group means without replacing the global analysis; `False`
shows civil-month heatmaps instead. Every supplied horizon gets an IC time series,
actual-sample histogram and fitted Q-Q panel. The default is Spearman with at least
two paired assets. Pearson remains an explicit native alternative.

The result exports prepared keys/groups/buckets, global and requested group ICs,
paired counts and undefined statuses, summary inference and exact month/group
aggregates through `get_table`. Undefined observations remain missing, not zero.
Empty, singleton and constant samples cannot produce KDE or fitted quantiles.
Rolling means require all 22 finite ICs in a 22-observation window, not an elapsed
month. Nanosecond keys remain in the typed export and chart payloads.

IC, its sample deviation and mean/deviation ratio are dimensionless. The report
uses naive two-sided t inference and biased skew/Fisher excess kurtosis. It keeps
constant-sample inference undefined rather than adopting the reference's infinite
t statistic and zero p value. This is an explicit degenerate-sample convention,
not a claim that a differing definition proves a reference error.

The fitted Q-Q convention matches Alphalens's standardized observations and
estimated distribution. Standalone `plot_ic_qq` retains its existing native
midpoint-probability/sample-deviation normal default; select `fitted=True` and,
optionally, `theoretical_dist=scipy.stats.t` to request the fitted alternative.
Civil-month grouping and supplied horizon labels establish no session calendar,
expected-bar grid or annualization.

The offline recipe runs all four adjustment/grouping combinations and exports
four native HTML/typed-JSON pairs plus four familiar HTML reports without requiring
Alphalens, Backtest or Engineer:

```bash
uv run --extra viz python examples/legacy_migration_comparison.py --task information --output signal-information.html
```

Its fixture uses 224 price-produced observations over 32 dates, changing unequal
groups and supplied one-/five-observation returns. The `1D`/`5D` labels are supplied
selectors, not validated elapsed-day or exchange-session declarations. Comparisons
use Alphalens Reloaded 0.4.6, pandas 2.3.3, Polars 1.36.1 and Plotly 6.1.2, checking
keys, buckets, groups, ICs, table values, rolling samples and actual plotted Q-Q data.
The familiar adapter additionally executes the actual whole information sheet
from Alphalens 0.5.0/pandas 3.0.6 and frozen 0.4.6/pandas 2.3.3, using shared
price-produced six-asset samples and all four group-flag combinations. Original
Matplotlib rendering is executed before comparing its IC observations, native
histogram inputs and group/civil-month means. The minimum-renderer profile uses
Plotly 6.1.2; the current profile uses Plotly 7.1.0. Neither reference is a native
runtime dependency, and these tests do not qualify the other six sheets.

The same retained IC observations can also feed Diagnostic's
[HAC-adjusted inference](statistical-tests.md#hac-adjusted-ic). The recipe executes
that separate analysis with an explicit five-observation label horizon; it does
not relabel the report's naive t-test as HAC. Other report families use their
own samples and methods; see [migration routes](#choose-a-migration-route).

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
for the existing two-sample spread test. Portfolio, event and report composition
use these observations through the explicitly selected workflows below.

0.2.0 fixes after 0.1.9 stabilize the shared finite-return mean, sample
deviation and standard error. Identical large finite returns retain their
finite mean and observation count instead of overflowing into a missing daily
mean. Standard error is rescaled separately, so it can remain finite even when
the sample deviation exceeds Float64 range. Ordinary finite NumPy means retain
their existing result. The pooled spread and its existing two-sample Student
test use common scaled, shifted observations to avoid intermediate overflow
or underflow; their estimand and test method are unchanged.

### Inspect arbitrary quantile spreads and propagated errors

Select the buckets explicitly. This 0.2.0 method after 0.1.9 delegates to
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
buckets. This 0.2.0 method after 0.1.9 reports dimensionless fractions:

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
0.2.0 route after 0.1.9 returns dimensionless correlations:

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
as well as the existing native scalar summaries. This 0.2.0 workflow after
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
confidence bands from a fixed 252-observation assumption. The
[complete turnover report](#complete-turnover-report) below composes all recorded
lags. Other Alphalens tear sheets and dated holdings remain separate accepted
work. Persistence-disabled analyses and historical saved
results without dated observations return no turnover component; scalar means
cannot reconstruct missing dates, buckets or reference ranks.

### Plot dated rank and membership persistence

0.2.0 after 0.1.9, the existing plotting APIs consume the actual dated
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
exchange sessions. The individual plot default selects the smallest recorded lag. Membership and
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
`False` omits that notice. The dashboard renders membership and rank charts
for every recorded lag and retains the complete tables and summaries.

Shared-input checks compare the same membership, rank universe and observation
clock with Alphalens Reloaded 0.4.6 `quantile_turnover` and
`factor_rank_autocorrelation` before their dated plotting routes. The native
scalar larger-membership turnover and paired-sample Spearman definitions remain
separate. These chart workflows do not establish exchange-calendar defaults
or full Alphalens/Pyfolio replacement.
Run the existing
[supplied-panel example](https://github.com/ml4t/diagnostic/blob/8d52739bb133e5c5d3540c8cf268ea1a8d3c7722/examples/signal_panel_cleaning.py)
with `--plots` to inspect the actual
trace values, recorded methods and hover counts without a legacy dependency.

### Complete turnover report

0.2.0 after 0.1.9, `generate_signal_turnover_report` composes all recorded
membership and rank lags, all-quantile mean/count tables, and the exact dated
observations. It consumes the existing owned `TurnoverAnalysisResult`, without
an Alphalens, Pyfolio, Backtest or Engineer runtime dependency. Use this route
when you need to inspect missing dates, denominators and lag clocks alongside
the charts. The result uses the same
[dated persistence calculations](#inspect-dated-persistence-in-a-result-and-report)
and JSON/Arrow snapshots as the analysis workflow.

With the visualization extra installed, this price-produced example requests
three lags on the observed-date clock:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.signal import prepare_signal_panel
from ml4t.diagnostic.visualization.signal import (
    generate_signal_turnover_report,
    plot_signal_turnover,
)

turnover_dates = pd.bdate_range("2025-03-03", periods=14, tz="UTC")
turnover_changes = np.array([
    np.roll([0.01, 0.02, -0.01, -0.02], i) for i in range(13)
])
turnover_prices = pd.DataFrame(
    np.vstack([np.full(4, 100.0), 100 * np.cumprod(1 + turnover_changes, axis=0)]),
    index=turnover_dates,
    columns=list("ABCD"),
)
turnover_factor = (
    turnover_prices.pct_change(fill_method=None).iloc[1:-1]
    .stack(future_stack=True).rename("factor")
)
turnover_forward = (
    (turnover_prices.shift(-1) / turnover_prices - 1)
    .stack(future_stack=True).rename("1D").to_frame()
)
turnover_factor.index.names = turnover_forward.index.names = ["date", "asset"]
turnover_panel = prepare_signal_panel(
    turnover_factor, turnover_forward, quantiles=2, max_loss=0
)
turnover_result = TurnoverAnalysisResult.from_observations(
    turnover_panel.quantile_turnover(lags=(1, 2, 5)),
    turnover_panel.rank_autocorrelation(lags=(1, 2, 5)),
)
turnover_means = turnover_result.get_dataframe("autocorrelation")
np.testing.assert_allclose(turnover_means["autocorrelation"], [0, -1, 0], atol=1e-12)
assert turnover_means["n_dates"].to_list() == [11, 10, 7]
assert len(plot_signal_turnover(turnover_result)) == 6
turnover_report = generate_signal_turnover_report(turnover_result)
turnover_restored = TurnoverAnalysisResult.model_validate_json(turnover_result.model_dump_json())
assert turnover_restored.get_dataframe("dated_turnover").equals(
    turnover_result.get_dataframe("dated_turnover")
)
print("Turnover report: 6 panels; defined rank dates 11/10/7")
```

In a notebook, call `figure.show()` for each figure from
`plot_signal_turnover(turnover_result)` and display the returned HTML. To write
HTML atomically, pass `output_file="turnover-report.html"` to the generator.
The HTML composer loads Plotly from its CDN; exporting this HTML alone does
not make it an offline bundle. `show_all_quantiles=True` includes every recorded
bucket in each membership chart. The default charts show the top and bottom
buckets, while the tables always include all buckets. Custom lag order and
different membership/rank lag sets are retained independently.
Alphalens's default `turnover_periods=None` derives its lags from the forward-return
column labels. Diagnostic takes explicit observation lags when constructing this
result; a label such as `1D` does not establish a validated session duration.

The runnable command also prints installed package versions and writes typed
JSON alongside the HTML. Run it from a checkout containing this 0.2.0 API:

```bash
uv run --extra viz python examples/legacy_migration_comparison.py --task turnover --output turnover-report.html
```

Clock choice changes the sample. Corrected Alphalens Reloaded 0.5.0 retains the
12 observed scored dates when the prepared index has no frequency, matching
Diagnostic's default. Frozen Alphalens Reloaded 0.4.6 with pandas 2.3.3
calls `asfreq(None)` on the prepared index used in the comparison, inserting
civil daily dates. This is a different sample, not a numerical tolerance issue.
To compare on that reference's 16-date clock, supply the same grid to both
Diagnostic calculations:

```python
turnover_grid = pd.DataFrame({
    "date": pd.date_range(turnover_dates[1], turnover_dates[-2], freq="D")
})
turnover_daily = TurnoverAnalysisResult.from_observations(
    turnover_panel.quantile_turnover(lags=(1, 2, 5), date_grid=turnover_grid),
    turnover_panel.rank_autocorrelation(lags=(1, 2, 5), date_grid=turnover_grid),
)
assert turnover_daily.get_dataframe("autocorrelation")["n_dates"].to_list() == [9, 6, 6]
```

The rotating sample has the same mean rank correlations under these two
clocks, but different dates, missingness and counts. The reference comparison
checks those intermediates in separate corrected and frozen profiles and calls
the complete reference turnover sheet
for rotating, changing-membership, constant-rank and single-asset samples.
Inserted weekends remain unavailable, not zero turnover or zero correlation.
Choose the intended clock explicitly; neither observation lags nor this civil
grid establish exchange-session validation. Native scalar definitions remain
unchanged. This workflow does not qualify the other six Alphalens sheets,
dated holdings/expiration accounting or full Pyfolio replacement.

### Plot empirical dated quantile observations

0.2.0 after 0.1.9, the actual raw, prepared and owned analysis routes retain
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
For multi-horizon chart composition, cumulative paths and event/portfolio
workflows use the [complete report routes](#choose-a-migration-route), not this
single-component dashboard.

### Distinguish unavailable monotonicity from a negative decision

0.2.0 after 0.1.9, use the structured result's availability-aware accessor
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

0.2.0 after 0.1.9, the existing spread chart uses actual dated bucket means
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
its bucket labels. These 0.2.0 methods after 0.1.9 share one normalization
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
have null return and zero count. Capital, expirations and dated holdings belong
to the separate [synthetic portfolio-input route](#turn-explicit-expiries-into-synthetic-portfolio-inputs).

### Inspect factor alpha and beta on the same universe

0.2.0 after 0.1.9, use the same `panel` to inspect a factor portfolio's
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

0.2.0 after 0.1.9, compose the existing factor weights and returns with the
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
cumulative quantile report panels use the
[owned returns result](#retain-the-complete-return-analysis-sample-before-rendering).

### Turn explicit expiries into synthetic portfolio inputs

0.2.0 after 0.1.9, `panel.positions()` consumes supplied endpoint keys and
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

`to_pyfolio()` retains a minimal `attrs["diagnostic"]` origin marker on each
returned object. Familiar simple/full account reports recognize synthetic
forward-outcome returns separately from synthetic holdings, including mixed
supplied/research inputs and hidden-position reports. They label these as
research allocations and returns, not executed account P&L. Removing the marker
removes that provenance; ordinary unmarked frames remain supplied observations,
not evidence of executions. The marker does not change values or clocks.

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

### Retain the complete return-analysis sample before rendering

0.2.0 source after 0.1.9 composes the existing analytical kernels into one
owned return-analysis result. Continue from `adapter_clean` above:

```python
from ml4t.diagnostic.signal import PreparedSignalPanel, SignalReturnsResult, analyze_signal_returns

return_panel = PreparedSignalPanel.from_prepared(
    adapter_clean, return_columns=["1D", "2D"], calendar=None,
)
return_analysis = analyze_signal_returns(
    return_panel,
    annualization_factors={"1D": 252, "2D": 126},
    rate_conversion_factors={"1D": 1, "2D": 0.5},
)
return_copy = SignalReturnsResult.from_json(return_analysis.to_json())
assert return_copy.get_table("prepared_data").height == len(adapter_clean)
assert return_copy.metadata["annualization_factors"] == {"1D": 252, "2D": 126}
assert return_analysis.get_table("quantile_observations")["standard_error"].is_null().all()
print("Owned signal returns preserve explicit scales and unavailable per-date errors")
```

The example has one asset per date/bucket, so per-date sample errors are
unavailable, not zero. Across-date means and errors remain separate statistics.
Annual alpha compounds the fitted intercept with the caller's explicit factor;
beta is dimensionless. A constant benchmark leaves the fit unavailable. Display
means use the first supplied horizon, while descriptive errors scale by the
square root of its declared conversion power. Raw statistics and counts remain
inspectable; this conversion does not establish confidence coverage.

`table_names` and `get_table` expose the prepared input, actual weights and
returns, regression observations/statistics, global/group quantile statistics,
dated observations, the requested output grid, propagated spreads and factor/
quantile wealth paths. Every read and JSON restoration owns independent typed
data. No price, provider or backtest installation is needed for supplied panels.

Native spreads default to the raw +1/-1 mean difference. Select
`spread_exposure="unit_gross"` to scale both spread and propagated error by one
half. `date_grid` must retain every observed input date; additional requested
cells have unknown returns, zero observed counts and `row_present=False` in
`quantile_output_grid`. Wealth follows the existing explicit NaN-as-flat product
without relabeling missing returns as observed zeros. These horizon-return paths
are not self-financing accounts.

Render the retained samples through the existing quantile charts:

```python
from ml4t.diagnostic.visualization.signal import generate_signal_returns_report, plot_signal_returns

return_figures = plot_signal_returns(return_analysis)
assert len(return_figures) == 10  # Five actual panels per retained horizon.
generate_signal_returns_report(return_analysis, output_file="owned_signal_returns.html")
print("Owned signal returns report: raw wealth, dated distributions and descriptive errors")
```

The standalone HTML includes Plotly without requiring an external script request.
Each horizon has mean/error bars, the actual dated-mean distribution, descriptive
spread bands and any requested factor/quantile paths. Group summaries add mean
bars without inventing dated group paths. Beta and wealth multiples are
dimensionless; quantile cumulative returns are wealth minus one. Sparse bucket
IDs and full timestamp precision survive the chart bridge. Sub-millisecond keys
use observation-order axes with exact keys retained for inspection, not browser
date rounding. `to_quantile_result(return_column)` adapts those same owned samples
for existing charts; uncomputed inference remains unavailable. Native spread
rolling windows default to 21 observations and bands to one propagated SE.
For the familiar returns-sheet entry point, retain the prepared pandas table:

```python
from ml4t.diagnostic.compat.alphalens import tears

familiar_return_figures = tears.create_returns_tear_sheet(
    adapter_clean, long_short=True, group_neutral=False, by_group=False,
    reference_version="0.5.0", return_fig=True, set_context=False,
    output_file="familiar_signal_returns.html",
)
assert len(familiar_return_figures) == 8
print("Familiar returns sheet: retained samples and explicit corrected spread exposure")
```

The adapter retains Alphalens's first-horizon rate powers and 252-day horizon
alpha scales; `annualization_factors=` overrides those scales explicitly.
These duration ratios do not validate an exchange calendar. Corrected 0.5.0
uses unit-gross spreads and errors; `reference_version="0.4.6"` keeps the raw
gross-two convention. Both use complete 22-observation rolling means and
half-SE descriptive bands. Familiar cumulative panels require a supplied `1D`
column, unlike the native explicit arbitrary-horizon selection.

Plotly/HTML replaces Matplotlib. Horizons have separate panels and distribution
axis percentiles, and native mean bars additionally display the retained sample
errors. Actual samples and counts are not changed by these presentation choices.
Group mean bars supplement the global analysis. As with the other adapters,
default notebook calls display the report and return `None`; `return_fig=True`
returns native Plotly figures. Summary/full-sheet compositions and original
notebook qualification remain separate unfinished acceptance.

### Compose owned return, information and turnover results

For a native summary or full report, reuse the same prepared sample rather than
joining independently cleaned analyses:

```python
from ml4t.diagnostic.results.signal_results import TurnoverAnalysisResult
from ml4t.diagnostic.signal import analyze_signal_information
from ml4t.diagnostic.visualization.signal import generate_signal_report

report_information = analyze_signal_information(return_panel, min_obs=2)
report_turnover = TurnoverAnalysisResult.from_observations(
    return_panel.quantile_turnover(lags=[1, 2]),
    return_panel.rank_autocorrelation(lags=[1, 2]),
)
full_signal_html = generate_signal_report(return_analysis, report_information, report_turnover)
assert full_signal_html.count('class="plot-container"') == 22
generate_signal_report(
    return_analysis, report_information, report_turnover,
    summary=True, output_file="signal_summary.html",
)
print("Composed native signal report: owned returns, information and recorded persistence")
```

The full view renders the existing component collections. The summary retains
mean bars plus actual factor, alpha/beta, quantile/spread, IC and persistence
tables. Original factor statistics reuse the event-study calculation, including
sample deviation and actual counts. Return and information components must own
identical prepared rows and horizon order; a mismatch raises before replacing an
output file. Supply dated turnover/rank components from that same sample; their
recorded lag clocks are not inferred return horizons. Native information
adjustment follows the supplied information result explicitly.

The familiar compositions enter through the same prepared-data boundary:

```python
summary_figures = tears.create_summary_tear_sheet(
    adapter_clean, return_fig=True, output_file="familiar_summary.html", set_context=False,
)
full_figures = tears.create_full_tear_sheet(
    adapter_clean, return_fig=True, output_file="familiar_full.html", set_context=False,
)
assert len(summary_figures) == 2 and len(full_figures) == 20
print("Familiar summary and full sheets: shared native samples and original IC adjustment")
```

Both references leave summary-sheet IC unadjusted even when `group_neutral=True`
adjusts returns. The full sheet applies `group_neutral` to both returns and IC;
`by_group=True` adds group summaries without discarding the global sample.
Factor, return, IC and persistence tables retain their original analysis inputs;
the HTML display rounds statistics to six significant digits, while the native
owned results retain full precision. Native themes and Plotly figures replace
Matplotlib context/artist behavior. Event-return and event-study sheets remain
separate routes, as in the reference full-sheet composition.

All seven familiar sheet names now have source implementations. This does not
qualify every option, plotting/performance helper or the four original notebooks;
those workflows and final artifact/platform checks remain required before any
full Alphalens replacement claim.

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

0.2.0 after 0.1.9, the direct familiar calls now delegate to that same
native event engine. Neither call requires a provider, forward returns or a
factor score. Reuse the supplied events and levels above:

```python
from ml4t.diagnostic.compat.alphalens import performance as familiar_events

aligned = familiar_events.common_start_returns(
    event_input, event_prices, 1, 1, cumulative=True, mean_by_date=True,
)
event_buckets = event_input.to_frame().assign(factor_quantile=1)
average = familiar_events.average_cumulative_return_by_quantile(
    event_buckets, event_prices, periods_before=1, periods_after=1, demeaned=False,
)
assert aligned.shape == (3, 3)
assert average.loc[(1, "mean")].tolist() == [55.0, 55.0, 55.0]
assert average.loc[(1, "std")].tolist() == [0.0, 0.0, 0.0]
event_plot = familiar_plots.plot_quantile_average_cumulative_return(average, std_bar=True)
assert len(event_plot.data) == 2
event_plot.write_html("familiar_event_paths.html", include_plotlyjs=True)
event_bins = familiar_plots.plot_events_distribution(event_input, num_bars=4)
assert sum(event_bins.data[0].y) == len(event_input)
print("Familiar event paths: observed clocks, date means and supplied levels retained")
```

`common_start_returns(cumulative=False)` first compounds periodic returns over
the entire supplied clock. `demean_by` may include peer assets outside the event
cohort; its event-date universe is held fixed across each window. Individual
paths have repeated asset columns across event dates, matching the familiar
layout. Assets within a date follow input order instead of Alphalens's
unspecified Python-set order. Clipped offsets follow first-seen order, while
genuinely missing supplied cells stay NaN. The native result additionally
retains every requested offset, its status and its unique event ID.

The average call consumes levels despite the legacy argument name `returns`.
It weights date means equally, or group/date means with `group_adjust=True`;
sample deviations use ddof=1, not standard error. `by_group=True` preserves the
additional group level. Both Alphalens 0.5.0/pandas 3.0.6 and frozen
0.4.6/pandas 2.3.3 have been compared on clipped/missing paths, demeaning,
group options and duplicate events. The familiar no-demeaning aligned-path
call retains the reference's within-date asset deduplication; native event
analysis deliberately retains individual event multiplicity as described below.
Missing assets, invalid clocks and a sample with no event on the value clock
raise rather than inventing paths. These adapters do not perform calendar-grid
validation or convert supplied levels into event-zero abnormal returns.
`plot_quantile_average_cumulative_return` consumes the returned mean/std table
directly. Use `by_quantile=True` for separate figures and select one group before
plotting a grouped table. Ordinates follow the reference's bps convention, even
when the caller supplied levels. Error bars show the supplied sample deviation,
not SE, confidence coverage or an inferred number of paths. Event distributions
reuse elapsed-time bins, count nonmissing supplied event values and retain empty
bins. Numbered bin labels keep phone-sized reports readable; exact starts remain
on hover. A one-instant sample has one bin instead of an invalid zero-frequency
grouping. Windows too fine for the supplied timestamp resolution raise explicitly.

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

0.2.0 after Diagnostic 0.1.9, `analyze_event_returns` or
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

### Signal event-study report

0.2.0 after 0.1.9, `analyze_signal_event_study` composes the separate
Alphalens event-study workflow from an owned `PreparedSignalPanel`. It retains
factor statistics, elapsed-time event bins and each event's bin membership,
raw prepared events, equal-date quantile statistics and empirical dated return
samples. Optional event paths reuse the preceding event-returns workflow.
This is not Diagnostic's abnormal-return/CAR model.

Reference comparisons use corrected Alphalens Reloaded 0.5.0 and a separate
frozen 0.4.6 profile on pandas 2.3.3. Diagnostic-only workflows also run without
Alphalens; the legacy package is a comparison dependency, not a report runtime.

```python
from ml4t.diagnostic.signal import analyze_signal_event_study, prepare_signal_panel
from ml4t.diagnostic.visualization.signal import generate_signal_event_study_report

study_index = pd.MultiIndex.from_product(
    [event_dates[[0, 2, 4]], ["A", "B"]], names=["date", "asset"]
)
study_scores = pd.Series(range(1, 7), index=study_index, name="factor")
study_returns = pd.DataFrame({"3D": [0.03] * 6, "1D": [0.01] * 6}, index=study_index)
study_panel = prepare_signal_panel(study_scores, study_returns, quantiles=1, max_loss=0)
study_result = analyze_signal_event_study(
    study_panel, rate_conversion_factors={"3D": 1, "1D": 3}, num_bars=3,
)
assert study_result.metadata["base_period"] == "3D"
assert study_result.event_distribution["n_events"].sum() == 6
generate_signal_event_study_report(study_result, output_file="signal_event_study.html")
print("Signal event study: owned factor statistics, bins and dated return samples")
```

The first supplied horizon is the rate base, not necessarily the shortest one.
Declare dimensionless `rate_conversion_factors` from known compatible horizon
units. For a three-observation base and one-observation return, the powers are
one and three. Labels such as `3D` alone establish neither elapsed days nor
sessions. No calendar or annualization factor is inferred. With
`rate_of_return=False`, raw horizon means are displayed and no factors are
required. Raw means, deviations, errors and counts remain unchanged; separate
`display_mean`, `conversion_ratio` and `display_status` fields expose conversion.
Undefined real-valued rates remain missing rather than becoming zero returns.

Supply `values` and an explicit `input_kind` for optional aligned paths. Omit
values or pass `event_window=None` to exclude them; the report explains their
absence and does not invent data. The window counts observed rows. Histogram
bins instead divide the actual elapsed timestamp span; they retain empty bins
and may include an endpoint bin beyond the requested `num_bars`. A singleton
timestamp uses one observed bin, whereas the reference zero-width grouper fails.

The native factor table reuses stable sample statistics. It retains finite
two-point deviations for large finite factors where the reference overflows.
Identity rate conversion preserves a supplied tiny return such as `1e-20`;
the reference add/power/subtract calculation rounds it to zero. These are
explicit numerical reconciliations, not blanket equivalence claims. Violin
samples are the actual date/bucket means. Their visible limits use the 1st/99th
percentiles without trimming the sample or generating observations; Plotly and
Seaborn density rendering are not claimed identical.

Run the original event-study notebook's threshold-crossing sequence on bounded
offline prices, including one bin, all seven horizons and positive/negative
event factors:

```bash
uv run --extra viz python examples/legacy_migration_comparison.py --task event-study --output event-study.html
```

The command checks nine events and 144 requested path cells per sign and writes
positive/negative reports with typed JSON exports. Negative event factors change
the factor statistics, not the sign of raw forward-outcome panels. This is the
reference event-study convention, not a short-strategy portfolio simulation.
Provider downloads are replaced by deterministic offline prices; no legacy
library, Backtest or Engineer is required at runtime. The other canonical
notebooks and report families remain separately required.

### Quantiles, turnover, and report boundaries

In 0.2.0 source after 0.1.9, direct `metrics.cross_sectional_ic_series()`
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
analysis but not turnover or cumulative quantile-return paths. The 0.2.0
persistence-enabled workflow above adds actual dated turnover/rank tables and
summaries. The complete turnover report composes every recorded lag. For dated
holdings and expiration accounting use the separate
[synthetic portfolio-input route](#turn-explicit-expiries-into-synthetic-portfolio-inputs),
not the turnover table or changes in bucket membership.

Alphalens also supplies grouped summaries and its own complete factor tear
sheet. Those capabilities are not implied by grouped IC observations or by a Diagnostic signal result.
Diagnostic's default global quantiles do not mean `group_adjust=True` or
`binning_by_group=True`; the latter can now be requested explicitly during
supplied-panel preparation. Request grouped IC explicitly through `ic_by_date`.
Use the [complete familiar sheets](#choose-a-migration-route) for grouped report
composition; selecting buckets alone does not select those additional methods.

Factor *construction* is a separate task. If needed,
[ML4T Engineer's feature guide](https://github.com/ml4t/engineer/blob/145ccafd2ca36dc61f976cdbeaaf147af21e7c2e/docs/user-guide/features.md)
documents `compute_features()`; Diagnostic does not require Engineer, and
users can supply factors produced elsewhere. Engineer owns feature
preparation, while Diagnostic evaluates the resulting factor.

## Pyfolio portfolio and performance analysis

[Pyfolio's `create_full_tear_sheet()`](https://github.com/quantopian/pyfolio/blob/4b901f6d73aa02ceb6d04b7d83502e5c6f2e81aa/pyfolio/tears.py)
takes noncumulative daily returns, with optional positions, transactions,
benchmark returns, and other inputs. Diagnostic separates numeric analysis
from HTML rendering. Development source includes the nine familiar
`compat.pyfolio` report calls, complete simple/full account composition and a
normalized Zipline performance-frame adapter. These analytical routes do not
implement an execution engine. Published 0.1.9 does not contain the additions below.

<div class="migration-table" markdown="1">

| Pyfolio input or task | Diagnostic route | Conversion or difference |
|---|---|---|
| Daily noncumulative return Series | `PortfolioAnalysis(returns=..., dates=...)` | A pandas DatetimeIndex is retained automatically. Explicit dates must agree with it; set `periods_per_year` and risk-free convention explicitly. |
| Benchmark return Series | `PortfolioAnalysis(benchmark=...)` | A dated Series is aligned to every strategy observation, with extra benchmark dates excluded. Missing/duplicate keys and timezone mismatches raise; verify currency separately. Undated arrays must have equal lengths. |
| Complete dated signed holdings | `PortfolioAnalysis(positions=...).compute_holdings_analysis(valuations, complete_snapshots=True)` | 0.2.0 after 0.1.9. Wide string asset columns or long `date`, `asset`, `value`; supply exact observed keys and reported account values. See [prepared holdings](#analyze-prepared-holdings-on-observed-keys). |
| Fill transactions | `PortfolioAnalysis.from_frames(..., transactions=...).compute_transaction_analysis(...)` | 0.2.0 owned actual-fill analysis; select the denominator and known fee/cost declarations. [Actual transactions](#analyze-actual-transactions-and-incremental-costs) and [fractional FIFO](#match-actual-fractional-fifo-lots-from-an-explicitly-flat-book) are separate calculations, not inferred from changing weights. |
| Performance HTML | `generate_backtest_tearsheet(returns=..., metrics=...)` | Install the `viz` extra; supply a return array and any precomputed metrics. This is not a Pyfolio report template. |
| A completed `ml4t-backtest` result | `generate_tearsheet_from_result(result)` or [normalized captured account](#consume-actual-backtest-account-capture) | The existing Backtest report bridge is distinct from supplied-account and Zipline performance-frame adapters. |
| Trade-level ranking | `TradeAnalysis` on `TradeRecord` objects | Consume the explicitly prepared [FIFO result](#match-actual-fractional-fifo-lots-from-an-explicitly-flat-book); raw transaction rows are not completed trades. |

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
0.2.0 source changes after 0.1.9; earlier published behavior is not a
supported missing-data or positional-alignment policy.

`PortfolioAnalysis` profit factor divides the sum of positive period returns by
the absolute sum of negative period returns, without annualization. In
0.2.0 source after 0.1.9, a flat sample reports `NaN` for undefined `0/0`,
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
This is 0.2.0 source work after 0.1.9. Paired uncertainty, selection adjustment
and Reality Check align pandas DatetimeIndex inputs with the same unique date
labels chronologically, then exclude nonfinite observations with one common mask.
Entirely positional inputs require equal original lengths. Mixed dated/positional
inputs and different dated windows raise instead of guessing observation identity.
Single-series uncertainty retains its finite-value filter; neither route validates
a calendar grid or records excluded dates in its result. Their stationary-bootstrap intervals are not Pyfolio's IID bootstrap
intervals; matching point estimates does not establish interval equivalence.

### Keep familiar performance tables and IID summaries

0.2.0 after 0.1.9, `compat.pyfolio.timeseries.perf_stats` uses the existing
native metric kernels with Pyfolio's thirteen labels and per-metric defaults.
It does not require Pyfolio or a producer package:

```python
from ml4t.diagnostic.compat.pyfolio import timeseries as familiar_performance

performance_prices = pd.Series(
    [100.0, 110.0, 99.0, 103.95],
    index=pd.date_range("2025-02-03", periods=4, tz="UTC"),
)
performance_returns = performance_prices.pct_change(fill_method=None).iloc[1:]
performance_table = familiar_performance.perf_stats(performance_returns)
assert np.isclose(performance_table["Cumulative returns"], 0.0395)
assert np.isclose(performance_table["Max drawdown"], -0.1)
assert np.isclose(
    performance_table["Daily value at risk"], 1 / 60 - 2 * np.sqrt(13 / 1200),
)
performance_bootstrap = familiar_performance.perf_stats_bootstrap(
    daily_returns, n_samples=16, random_seed=42,
)
assert performance_bootstrap.shape == (13, 4)
assert performance_bootstrap.attrs["diagnostic"]["method"] == "iid"
print("Familiar performance table: metric-specific missingness and Gaussian periodic VaR")
```

Returns, volatility, drawdown and alpha use their declared return scales;
Sharpe, leverage and beta are dimensionless. The table's Gaussian daily VaR is
mean minus two sample deviations, not Empyrical's empirical quantile. The fixed
legacy annualization default is 252; explicit `periods_per_year=12` is an
extension for a differently declared observation cadence. Native annual-rate,
linear-alpha and summary defaults are unchanged. Optional benchmark alpha/beta
reuse the existing paired regression adapter, including compounded annual alpha.
Supplied positions add original gross leverage, and actual transactions add the
existing explicit AGB/closing-value turnover. Legacy leverage retains signed
and zero-NAV division; native positive-NAV holdings ratios remain separate.

Missingness is per statistic, not one global drop or fill. CAGR retains missing
rows as elapsed observations; means/deviations omit them where their kernels
define that behavior. The original SciPy skew/kurtosis profile propagates a
missing return. These raw analytical helpers do not relax the native supplied
account's finite-return validation.

`calc_bootstrap` resamples paired positional rows and resets labels before calling
the statistic, as in the reference. Its optional `random_seed` uses a local stream
without changing global NumPy state. Unseeded draws are not promised to equal a
reference process's global-RNG draws. `perf_stats_bootstrap` advances the same
stream across metrics and returns raw draws with `return_stats=False`, or mean,
median and 5/95 percentiles. These are IID sampling summaries, not stationary/HAC
inference or calibrated confidence coverage. Degenerate/infinite metric samples
can leave summaries undefined; they are not zero risk. Holdings and transactions
are not bootstrapped, matching the original return-only bootstrap. Unlike the
original function's silently ignored kwargs, explicit sample count, seed and scale
are honored. Full report composition and original notebook qualification remain
separate acceptance work.

The `compat.pyfolio.plotting` return panels accept the same raw dated Series.
They return Plotly figures, not reusable Matplotlib axes. For example:

```python
from ml4t.diagnostic.compat.pyfolio import plotting as familiar_return_plots

wealth_panel = familiar_return_plots.plot_rolling_returns(
    performance_returns, live_start_date=performance_returns.index[-1],
    cone_std=(1, 2), num_samples=16, random_seed=42,
)
assert np.isclose(wealth_panel.data[0].y[-1], 0.99)
assert np.isclose(wealth_panel.data[1].y[-1], 1.0395)
assert wealth_panel.layout.meta["training_observations"] == 2
split_performance = familiar_return_plots.show_perf_stats(
    performance_returns, live_start_date=performance_returns.index[-1], return_df=True,
)
assert split_performance.loc["Cumulative returns", "All"] == "3.95%"
print("Familiar return panels: full-clock wealth and training-only live forecasts")
```

The historical name `plot_rolling_returns` denotes cumulative wealth starting at
one, not rolling CAGR. Benchmark matching uses sample volatility; logarithmic
scale changes presentation, not returns. Raw `plot_returns` keeps NaNs as gaps,
while wealth compounding treats them as neutral product factors. Date spacing
counts observed rows; exact timestamps, including nanoseconds, remain on hover.
`cone_std=None` is the direct plotting default. Explicit cones receive only the
pre-live observations and their final wealth. A custom cone callback receives a
copy and owns its own random seed. Native layout options are `theme`, `height`
and `width`; arbitrary Matplotlib style keywords and `ax` are not accepted.

`show_perf_stats` returns a pandas table with `return_df=True`, or escaped HTML
otherwise. Percentage display applies only to the named return/volatility/
drawdown/turnover rows; alpha and ratios retain their original numeric display.
Bootstrap summaries keep mean, median and 5/95 columns. Actual Pyfolio 0.10.0
instead reindexes the no-live bootstrap table to a nonexistent `Backtest`
column and displays `nan%` even for known positive CAGR. Diagnostic retains the
computed values. Live bootstrap samples pair the benchmark on each split's
actual dates. Neither correction changes native inference defaults or claims
complete portfolio-report coverage.

### Run a complete familiar returns report

The 0.2.0 `compat.pyfolio.create_returns_tear_sheet` (also available under
`compat.pyfolio.tears`) selects the original returns-report financial panels
and uses the existing `PortfolioTearSheet` renderer. Supplied raw returns need
no producer or Reloaded runtime:

```python
from ml4t.diagnostic.compat import pyfolio as familiar_portfolio

returns_sheet = familiar_portfolio.create_returns_tear_sheet(
    performance_returns, return_fig=True, periods_per_year=12,
)
assert len(returns_sheet.figures) == 14
assert np.isclose(returns_sheet.figures["Underwater"].data[0].y[1], -0.1)
assert returns_sheet.availability["volatility_match"]["status"] == "not_requested"
assert returns_sheet.figures["Rolling volatility"].layout.meta["status"] == "unavailable"
print("Complete familiar returns manifest: existing native renderer and explicit availability")
```

The report contains performance and drawdown tables, wealth/log/live/cone views,
raw returns, rolling volatility and Sharpe, drawdown episodes/underwater, monthly
heatmap/distribution, annual returns and daily/weekly/monthly return quantiles.
A benchmark adds rolling beta and enables volatility matching. `bootstrap=True`
requires a benchmark and adds actual IID metric distributions. A missing benchmark
or undefined dispersion leaves volatility matching explicitly unavailable rather
than displaying an unmatched curve under that name. `return_fig=True` returns the
native report with `show()`, `save_html()` and `to_dict()`, not a Matplotlib figure;
the default displays the report and returns `None`.

The runnable price-to-report example produces 280 observations and a 260/20
training/live split, with all JavaScript embedded for offline viewing:

```bash
uv run --extra viz python examples/pyfolio_returns_report.py --bootstrap --output returns-report.html
```

The optional reference run uses published Pyfolio 0.10.0 / Empyrical 0.7.0 and
their supported pandas 2.3.3 profile, separately from native pandas 3 workflows:

```bash
uv run --isolated --no-dev --python 3.12 --extra viz --with pyfolio-reloaded==0.10.0 --with empyrical-reloaded==0.7.0 --with pandas==2.3.3 python examples/pyfolio_returns_report.py --compare --bootstrap --output returns-reference.html
```

Windows count observations, not inferred sessions or rebalances. The original
126/252 rolling-beta arguments use inclusive endpoints and therefore select
127/253 rows; volatility and Sharpe require complete 126-row samples. The same
explicit `periods_per_year` reaches both summary and rolling risk calculations.
Rolling panels with no finite estimates show an unavailable explanation instead
of an empty auto-scaled risk axis. Observed zero volatility remains a plotted
zero, and an available benchmark is not hidden by incomplete strategy windows.
Calendar-year/month products include partial boundary periods and are not CAGR.
Live quantiles aggregate independently inside each split. Monthly heatmap
display retains the reference's three-decimal return rounding but leaves absent
months blank instead of displaying invented zero returns. Tables and monthly
heatmaps scroll horizontally on narrow screens to preserve readable values.

Drawdown tables, shading and underwater curves include initial capital, matching
Diagnostic's existing scalar drawdown. Pyfolio 0.10.0's report excludes that peak:
returns `[-.2, .25, -.1]` produce a correct scalar drawdown of -20%, but its table's
largest loss is only 10%. Diagnostic reports both 20% and 10% episodes. The initial
peak timestamp is not present in a raw return Series, so this familiar table
leaves it blank rather than guessing the first observation. Native supplied-account
analysis can retain an initial valuation timestamp. The familiar duration column counts
inclusive Monday-Friday civil dates only when peak and recovery are known,
not exchange sessions. Existing native episode durations still count observations.
The combined account route below reuses these panels. The original-notebook
replay and producer examples below exercise their distinct input contracts.

### Compose a familiar simple or full account report

0.2.0 after 0.1.9, `compat.pyfolio.create_simple_tear_sheet` and
`create_full_tear_sheet` accept supplied pandas returns, optional marked dollar
positions with separate `cash`, and an optional actual-fill ledger. No Data,
Engineer, Backtest, Zipline or Reloaded runtime is required. This route is useful
when migrating an existing account report to Diagnostic's owned report exports
and its [supplied-account analysis](#normalize-a-supplied-account-without-a-backtest-engine).
It does not reconstruct a broker account from a partial ledger.

```python
account_report_dates = pd.bdate_range("2025-02-04", periods=3, tz="UTC")
account_report_returns = pd.Series([.1, -.1, .05], index=account_report_dates)
account_report_nav = pd.Series([1100., 990., 1039.5], index=account_report_dates)
account_report_positions = pd.DataFrame({
    "Core": .6 * account_report_nav, "Hedge": -.2 * account_report_nav,
    "cash": .6 * account_report_nav,
})
account_report_fills = pd.DataFrame({
    "symbol": ["Core", "Core"], "amount": [1., -1.], "price": [110., 99.],
}, index=account_report_dates[:2] + pd.Timedelta(hours=15))
full_account_sheet = familiar_portfolio.create_full_tear_sheet(
    account_report_returns, account_report_positions, account_report_fills,
    turnover_denom="portfolio_value", estimate_intraday=False,
    periods_per_year=12, return_fig=True,
)
assert np.isclose(full_account_sheet.figures["Returns/Cumulative wealth"].data[0].y[-1], 1.0395)
assert np.allclose(full_account_sheet.figures["Transactions/Turnover"].data[0].y, [.1, .1, 0.])
simple_account_sheet = familiar_portfolio.create_simple_tear_sheet(
    account_report_returns, account_report_positions, account_report_fills,
    slippage=10, estimate_intraday=False, return_fig=True,
)
assert np.isclose(simple_account_sheet.figures["Returns/Cumulative wealth"].data[0].y[-1], 1.0392900105)
print("Familiar account composition: supplied returns, actual fills and options reach nested reports")
```

The simple report selects performance statistics, wealth, rolling Sharpe,
underwater, optional benchmark beta, four position views and actual turnover/
timing. The full report composes the complete returns, stress, positions and
transaction manifests. Supply `market_data` for the original fixed-capital
capacity sweep and all four threshold tables. Supply both `factor_returns` and
`factor_loadings` for attribution statistics, factor statistics and partitioned
paths/exposures. `factor_partitions=None` selects all factors; `{}` suppresses
partition panels without suppressing the summary. `round_trips=True` additionally
requires a flat opening state and complete actual execution history, as described
in the [complete round-trip sheet](#use-the-familiar-complete-round-trip-sheet).
The partial ledger in this example does not establish that condition.

`return_fig=True` returns the existing `PortfolioTearSheet`; otherwise the
functions display it and return `None`. Save offline HTML with
`.save_html(path, include_plotlyjs=True)`. The full report retains optional
tables in figures/export, not only an HTML string or a list of plots. Each
component keeps its own keys, missingness and membership rules. Returns-only
input does not create an empty transaction ledger. Invalid incomplete requested
inputs raise before display. Matplotlib artist mutation and global style
contexts do not migrate; `set_context` is accepted without mutating Matplotlib.

`periods_per_year` reaches all selected annualized statistics, rolling risk,
slippage and capacity curves and attribution statistics. It counts observations
in the supplied return series, not rebalance dates. `turnover_denom` reaches both
statistics and transaction panels. Pyfolio 0.10.0's full wrapper drops that
argument when calling its transaction sheet, so its chart always uses AGB even
when statistics use closing portfolio value. Diagnostic preserves the requested
denominator. UTC fill instants use the supplied book's timezone before daily
aggregation; naive/aware ambiguity is rejected rather than guessed.

Additional `slippage` is a hypothetical cash-bps penalty, never a second charge
for observed fees. Full composition preserves the original order: adjust on
the supplied closing book before intraday selection, only when no explicit
`unadjusted_returns` baseline was supplied. Simple composition selects intraday
positions before applying additional costs. Tied equal-NAV peak candidates
remain separate exposure observations but cannot duplicate a daily return or
charge. The shared slippage adapter collapses their equal NAV denominator once;
conflicting same-date NAV raises. Pyfolio's duplicate-index projection instead
duplicates that day's adjusted return and compounds the cost twice.

With `hide_positions=True`, full composition anonymizes asset and sector
identities before preparing positions, fills, capacity, attribution or FIFO
results. That includes exported tables and notebook payloads. Caller inputs
remain unchanged. User-supplied free text such as `header_rows` is not a secret
redaction service; do not place confidential identities there.

Run the existing price-to-report example end to end:

```bash
uv run --extra viz python examples/pyfolio_returns_report.py --simple-output simple-account.html --full-output full-account.html --output returns-report.html
```

Add `--compare` under the published Pyfolio 0.10.0 / Empyrical 0.7.0 / pandas
2.3.3 reference environment shown above to execute the actual original complete
simple and full sheets as well. Native pandas 3 support is qualified separately
from that reference's declared pandas range. These supplied-account examples
do not establish every original notebook/provider workflow or full replacement.

### Replay the original Pyfolio analytical notebooks

0.2.0 after 0.1.9, `scripts/migrate_pyfolio_notebooks.py` converts the five
original Pyfolio examples at
[`185eaa1`](https://github.com/stefan-jansen/pyfolio-reloaded/tree/185eaa1657a845a5e1c8c82809248eac7ce99340/src/pyfolio/examples).
The output-cleared copies in `tests/fixtures/pyfolio_original/` retain every original
markdown/code cell and include the original license and provenance. The command
accepts these copies or the originals, rejects changed analytical cells, checks
captured account hashes, records every import/I/O/display change, and never
overwrites an existing notebook. Conversion alone does not execute anything.

From this repository, run the original sector, slippage/full-report or round-trip
workflow without installing Pyfolio, Alphalens, Empyrical, Data, Engineer or Backtest:

```bash
uv run --extra viz --with nbclient --with ipykernel python scripts/migrate_pyfolio_notebooks.py --source-notebook tests/fixtures/pyfolio_original/slippage_example.ipynb --output migrated-slippage.ipynb --execute
```

Substitute `sector_mappings_example.ipynb` or
`round_trip_tear_sheet_example.ipynb` and a new output filename for those workflows.
The original slippage account has 1,511 return observations, 1,506 closing position
rows and 9,367 fills; its ending wealth multiple is 1.6540425002394212. Diagnostic
uses the same report renderer and numerical adapters as the supplied-account
examples above, with interactive Plotly figures and explicit unavailable panels.
Matplotlib artist mutation is not preserved.

The single-stock example requires your captured parquet history with a `Close`
column and unique ordered timezone-naive dates, because its original financial
cell explicitly localizes dates to UTC. No Yahoo request is substituted silently:

```bash
uv run --extra viz --with nbclient --with ipykernel python scripts/migrate_pyfolio_notebooks.py --source-notebook tests/fixtures/pyfolio_original/single_stock_example.ipynb --history captured-history.parquet --output migrated-single-stock.ipynb --execute
```

The `zipline_algo_example.ipynb` route replays the original analytical cells against
the original captured
[`results.pickle`](https://github.com/stefan-jansen/pyfolio-reloaded/blob/185eaa1657a845a5e1c8c82809248eac7ce99340/src/pyfolio/examples/results.pickle),
not a new trading simulation. `--zipline-output` must match SHA256
`7f6480c177869eb73fc7bf6cb0715924282d02b4fa507f128f182553967ba946`;
other pickles are rejected before conversion or deserialization. Only this
producer-object replay needs Zipline to deserialize its original asset objects:

```bash
uv run --extra viz --with nbclient --with ipykernel --with 'zipline-reloaded==3.2.0' --with 'pandas==2.3.3' --with 'numpy==2.2.6' python scripts/migrate_pyfolio_notebooks.py --source-notebook tests/fixtures/pyfolio_original/zipline_algo_example.ipynb --zipline-output results.pickle --output migrated-zipline.ipynb --execute
```

That original account is distinct from the CSV example: 1,511 account rows, 7,871
fills and reported ending NAV multiple 1.6120420817. Zipline's declared Empyrical
dependency belongs to this optional producer environment, not Diagnostic's
analytical runtime. The command removes the producer magic cell explicitly and
records it as not rerun. Conversion does not make arbitrary pickles safe.

The four standalone analytical notebooks have installed-wheel evidence on
Python 3.12, pandas 3.0.6 and a pandas 2.3.3 / NumPy 2.2.6 / Polars 1.36.1 /
Plotly 6.1.2 floor. Their actual notebook Plotly payload counts are 7 (round-trip),
17 (sector), 40 (slippage) and 14 (single-stock). The captured Zipline replay has
80, with the same reported account wealth. These proofs cover captured analytical
replay, not live Yahoo acquisition, cache behavior or rerunning the original
Zipline algorithm. Those producer/provider journeys and full replacement remain
separate acceptance work.

### Normalize a supplied account without a backtest engine

0.2.0 after 0.1.9, `PortfolioAnalysis.from_frames()` accepts dated returns,
an optional dated benchmark, signed marked holdings, reported cash/NAV and
actual transaction records. It uses the existing portfolio metrics and holdings
analysis. Ordinary supplied-account analysis needs no Pyfolio, Zipline,
Backtest, Engineer or Data installation.

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.evaluation import PortfolioAnalysis

account_dates = pd.date_range("2025-07-07", periods=2, tz="UTC")
account_returns = pd.Series([.01, -.02], index=account_dates)
account_positions = pd.DataFrame(
    {"A": [600., 500.], "B": [-100., -50.], "cash": [510., 539.8]},
    index=account_dates,
)
account_valuations = pd.DataFrame({"date": account_dates, "equity": [1010., 989.8]})
account_transactions = pd.DataFrame({
    "timestamp": account_dates, "asset": ["A", "B"],
    "quantity": [.5, -.25], "price": [100., 200.],
    "commission": [None, .25],
})
account_analysis = PortfolioAnalysis.from_frames(
    account_returns, positions=account_positions, valuations=account_valuations,
    transactions=account_transactions, complete_snapshots=True,
    calendar=None, periods_per_year=252,
)
np.testing.assert_allclose(account_analysis.returns, [.01, -.02])
account_holdings = account_analysis.compute_holdings_analysis()
np.testing.assert_allclose(account_holdings.timeline["gross_exposure"], [700., 550.])
np.testing.assert_allclose(account_holdings.timeline["cash"], [510., 539.8])
assert account_analysis.input_data.transactions["commission"].null_count() == 1
print("Supplied account preserves net returns and unknown fees")
```

Returns are authoritative net simple returns. NAV changes can include unknown
flows and never replace them. Wide `cash` is reported cash; NAV is not inferred
from cash plus signed holdings or futures notional. Missing NAV leaves raw
exposure available and dependent ratios unknown. `complete_snapshots=True`
declares absent assets zero within each supplied date, not across missing dates.
Incomplete holdings remain incomplete: no forward fill or inferred liquidation.
The existing array constructor remains available with its original defaults.

`input_data` owns Arrow/JSON snapshots of normalized returns, the full benchmark,
positions, valuations and transactions. Frame/metadata accessors return independent
values. Exact nanosecond keys and string/integer asset IDs are retained. Numeric
pandas wide IDs are not stringified. Use `positions_format="wide"` for ambiguous
wide asset names such as `asset` and `value`; long inputs can represent an actual
asset named `cash` separately from account cash. Cash-only wide dates remain
explicit snapshots without a fabricated cash asset.

Supply `feed_spec`, `timestamp_col` and `entity_col` for declared physical names;
`return_col` selects a table's return values. Calendar argument precedence and
explicit None follow the shared contract, but normalization itself does not
resample returns or establish schedule completeness. Fills can use a separate
`transaction_timestamp_col`. `quantity` is signed; a supplied buy/sell side is
reconciled and contradictory signs rejected. Integer quantities and prices are
retained before accounting, including values beyond Float64's exact integer
range. `multiplier` must be finite/positive; absent multipliers use the documented
one cash-price unit per quantity unit convention. Futures inputs must supply
their actual multiplier. Unknown commission/slippage remain unknown, not zero.
No execution is reconstructed from portfolio weights.

The existing `--task holdings` and `--task holdings-report` migration examples now
use this boundary. With `viz`, `account_analysis.create_tear_sheet()` also prepares
the declared complete holdings for the existing portfolio dashboard. Source input
normalization is not qualification of all Zipline, round-trip,
attribution or full legacy report workflows; those remain accepted implementation
and integration work.

### Analyze actual transactions and incremental costs

0.2.0 after 0.1.9, supplied accounts can prepare actual execution activity
through `compute_transaction_analysis()`. This recipe uses two real-format
execution records and explicitly synthetic net returns/NAV. Neither NAV changes
nor target weights determine the returns or fills.

```python
import pandas as pd
import polars as pl
from ml4t.diagnostic.evaluation import PortfolioAnalysis

cost_dates = pd.date_range("2025-07-07", periods=3, tz="UTC")
cost_analysis = PortfolioAnalysis.from_frames(
    pd.Series([.01, -.02, .03], index=cost_dates),
    positions=pd.DataFrame({"A": [200., 400., 600.]}, index=cost_dates),
    valuations=pd.DataFrame({
        "date": pd.DatetimeIndex(["2025-07-03T00:00Z", *cost_dates]),
        "equity": [1000., 1100., 1200., 1300.],
    }),
    transactions=pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-07-06T23:30Z", "2025-07-07T00:30Z"]),
        "asset": ["A", "A"], "quantity": [.5, -.5], "price": [2., 2.],
        "multiplier": [100., 100.], "commission": [.1, .1], "slippage": [0., 0.],
    }),
    complete_snapshots=True, transaction_timestamp_col="timestamp",
    feed_spec={
        "timestamp_col": "date", "entity_col": "asset", "calendar": "CME_Equity",
        "time_contract_version": 2, "timestamp_semantics": "session_label",
    },
    periods_per_year=252,
)
cost_result = cost_analysis.compute_transaction_analysis(incremental_bps=[0, 10])
assert cost_result.timeline["turnover"].to_list() == [.2, 0, 0]
extra_cost = cost_result.scenarios.filter(pl.col("incremental_bps") == 10)
assert extra_cost["incremental_cost_return"][0] == .0002
assert extra_cost["net_return"][0] == .0098
print("200 actual notional / 1000 beginning NAV; extra 10bps costs .0002 return")
```

The native turnover default is gross traded cash notional divided by beginning
reported NAV. `turnover_method="one_way"` explicitly halves that ratio. Choose
`turnover_denom="average_gross_book"` for the mean of adjacent complete closing
gross exposures. The first gross-book denominator is unknown without a previous
snapshot; it is not silently half the first exposure. Full-clock and active-fill
means/counts are both returned. These choices differ from Pyfolio's default
average-gross-book turnover and its first-observation convention. No single
turnover definition is implied by the name alone.

Notional is `abs(quantity * price * multiplier)`, including negative execution
prices. Commission is cash; slippage is a per-unit price difference multiplied
by absolute quantity and multiplier. Unknown commission or slippage leaves
observed total cost unknown. A zero-fill ledger proves zero activity, whereas
an absent ledger cannot. Missing or nonpositive NAV leaves raw dollars available
and active cost-return scenarios undefined. A known zero extra cost leaves net
returns unchanged even without NAV. Already-observed fees are never subtracted
from supplied net returns again. Extra-bps scenarios always use beginning NAV,
even when turnover uses gross book or the one-way convention.

With a named calendar, declared `session_label` return/NAV keys stay labels;
timezone-aware event instants use actual exchange intervals, including overnight
sessions and scheduled breaks. The Sunday and Monday fills above share the
July 7 CME session. Event-time returns compound within a session; original
observations and exact execution timestamps remain in the results. Off-session
fills or fills without a matching return session raise instead of disappearing.
Declared `bar_close` observations at touching 24-hour boundaries belong to the
completed session. Actual executions at that instant belong to the newly open
session. A declared feed timezone localizes naive event instants; without a
timezone declaration, event inputs must already be timezone-aware.
With `calendar=None`, no exchange grouping runs: fills must match exact return
observation keys, not a guessed civil date. No intraday NAV is reconstructed.

For an interactive report, use `cost_analysis.create_tear_sheet(transactions=cost_result)`.
The existing report renders prepared notional, absolute execution units, turnover,
all-clock versus active turnover distributions, execution timing, known costs and
extra-cost scenarios without selecting methods again. Absolute units use the
supplied instrument units, not an assumption that every instrument is a share.
`cost_result.execution_timing(bin_minutes=5, timezone="America/New_York")`
returns counts, absolute units, actual notional and notional fractions for the
whole civil day. The optional timezone changes display only; overnight fills are
not discarded. Native default timing retains the source instant timezone, or a
declared feed timezone for naive inputs. Date labels and undeclared naive instants
leave timing unavailable without removing raw activity or other report panels.
Pyfolio's time histogram instead defaults to New York and retains only
09:30 through 16:00, so that filtered distribution differs for overnight fills.
Returns-only/holdings reports remain
independently usable. From the source checkout, run
`uv run python examples/legacy_migration_comparison.py --task transactions`, or
`uv run python examples/legacy_migration_comparison.py --task transactions-report --output transactions-report.html`
with the visualization dependencies installed. These checks cover the named
supplied-input workflow, not Zipline extraction, lots, capacity or complete Pyfolio
report parity.

### Keep familiar Pyfolio transaction calculations

0.2.0 after 0.1.9, replace `from pyfolio import txn` with
`from ml4t.diagnostic.compat.pyfolio import txn` for `map_transaction`,
`make_transaction_frame`, `get_txn_vol`, `get_turnover` and
`adjust_returns_for_slippage`. These pandas-facing adapters use the same native
execution-value, gross-exposure and cost calculations; Pyfolio is not a runtime
dependency. Use the native owned-account workflow above for exchange-session
semantics, contract multipliers and reported beginning NAV.

The adapter intentionally retains the legacy daily definitions: grouping by
civil date in the input timezone, signed execution price for cash volume,
absolute units for share volume, gross two-sided turnover, default `AGB` with
the first previous book seeded at zero, and `portfolio_value` as the closing
sum of signed positions plus cash. Slippage bps adjusts PnL using that closing
value, not native beginning NAV. Existing net returns retain observed fees
once. Metric-specific missingness and IEEE division by zero remain explicit
legacy behavior; this is not the native positive-NAV normalization contract.

The runnable `examples/pyfolio_transaction_adapters.py` supplies three daily
observations, two cash-volume totals of 200 and two unit-volume totals of 2.
Independent expected AGB turnover is `[2, 0, .4]`; 10bps extra slippage changes
net returns `[.01, -.02, .03]` to `[.0098, -.02, .0298]`. Run without Pyfolio:

```bash
uv run python examples/pyfolio_transaction_adapters.py
```

The same sample and both denominator options are compared with published
Pyfolio 0.10.0 using pandas 2.3.3; the reference package requires pandas below 3.
From this source checkout, the optional reference command is:

```bash
uv run --isolated --no-dev --python 3.12 --with pyfolio-reloaded==0.10.0 --with pandas==2.3.3 python examples/pyfolio_transaction_adapters.py --compare
```

Two deliberate corrections are not output-parity claims: nested transaction
conversion preserves the actual timezone and nanoseconds rather than losing
them through a datetime ndarray, and widened integer accumulation avoids
signed-64-bit absolute-value overflow. An actually empty nested ledger returns
an empty table instead of the reference's missing-column error. The reference
tests cover intermediate amounts, prices, date keys, missingness and labels,
not just aggregate correlation.

With the `viz` extra, `from ml4t.diagnostic.compat.pyfolio import plotting`
also provides `plot_turnover`, `plot_daily_volume`,
`plot_daily_turnover_hist`, `plot_txn_time_hist`, `plot_slippage_sweep` and
`plot_slippage_sensitivity`. They return native Plotly figures with
`show()`, `write_html()` and `to_dict()`, not Matplotlib axes. Remove `ax=`
when migrating; Matplotlib axis reuse/style callbacks and legend placement are
not preserved. Histogram bin presentation may differ, but the underlying
turnover sample is retained, with any excluded nonfinite values counted.

Daily turnover retains civil-month and all-time mean overlays. Daily volume
uses the mean over actual activity dates, not the native full return clock.
Legacy timing defaults to five-minute New York buckets and includes only
09:30 through the entire 16:00 minute. It normalizes selected actual notional,
counts excluded fills and leaves an empty selection undefined. Native complete-day
timing above remains the route for overnight strategies.

Slippage sweep defaults remain `(3, 8, 10, 12, 15, 20, 50)` cash bps. Wealth
starts at 1 and compounds the exact adjusted return path, with legacy missing
returns contributing no change. Sensitivity evaluates the original 1 through 99 bps
sample using native CAGR, not arithmetic mean return. Its legacy daily scale
is 252; pass `periods_per_year=` for a different declared return cadence.
Three observations with `periods_per_year=3` use one exact window product;
`periods_per_year=6` squares that product. This is not rolling-return
annualization or an assumption about exchange sessions.

Run the six-plot, self-contained HTML recipe without Pyfolio:

```bash
uv run python examples/pyfolio_transaction_adapters.py --report transaction-adapters.html
```

The optional `--compare` flag checks both the published helpers and the complete
reference transaction sheet on those inputs. The example now calls the same
report entry point used by migrating callers:

```python
import pandas as pd
from ml4t.diagnostic.compat import pyfolio as pf

txn_report_dates = pd.date_range("2025-07-07", periods=3, tz="America/New_York")
txn_report_returns = pd.Series([.01, -.02, .03], index=txn_report_dates)
txn_report_positions = pd.DataFrame(
    {"A": [200.0, 400.0, 600.0], "cash": [800.0, 600.0, 400.0]}, index=txn_report_dates
)
txn_report_fills = pd.DataFrame(
    {"amount": [1.0, -1.0, 2.0], "price": [100.0] * 3},
    index=pd.to_datetime(["2025-07-07T15:00Z", "2025-07-07T16:00Z", "2025-07-09T15:00Z"]),
)
txn_report = pf.create_txn_tear_sheet(
    txn_report_returns, txn_report_positions, txn_report_fills,
    unadjusted_returns=txn_report_returns,
    estimate_intraday=False, return_fig=True, periods_per_year=3,
)
assert len(txn_report.figures) == 6
assert txn_report.availability["transactions"]["n_actual_fills"] == 3
txn_report.save_html("transaction-report.html", include_plotlyjs=True)
print("Complete familiar transaction manifest: actual fills and explicit extra-cost scenarios")
```

Without `unadjusted_returns`, the report includes turnover, activity-date unit
volume, the finite turnover distribution and selected New York timing. Supplying
it adds the original seven bps wealth paths and all 99 sensitivity scenarios;
`periods_per_year` reaches the sensitivity CAGR calculation. Original net fees
are not subtracted again. An empty actual ledger remains empty, never fills
inferred from changing positions.

`turnover_denom` selects `AGB` or `portfolio_value` on the whole report. The
intraday selector uses the same explicit estimator as the positions report;
estimated positions do not turn transactions into estimates. Plotly histogram
bins and Matplotlib presentation differ. For full/simple account composition and
the original portfolio notebooks, use the separate migration recipes above.

The report converts owned aware fill timestamps to the supplied position
clock's timezone before civil-day grouping. Instants, nanoseconds and caller
frames remain unchanged. This corrects mixed UTC-fill/New-York-book alignment:
the reference creates five unmatched daily keys with zero turnover for the
known `[2, 0, .4]` sample. Naive/aware ambiguity requires an explicit timezone;
the report does not guess UTC or exchange sessions. Standalone numerical
adapters retain their documented input-clock definitions.

### Match actual fractional FIFO lots from an explicitly flat book

0.2.0 after 0.1.9, `analysis.compute_round_trips(initial_state="flat")`
matches actual owned executions in timestamp order, preserving source order at
the same instant. The opening flat declaration is required; a first sale could
otherwise be either a new short or an unknown prior long's closing sale. This
route also accepts explicitly supplied complete opening lots below. Do not use
either declaration with an unknown opening book.

Run `uv run python examples/portfolio_fifo.py`. The independent example buys
10 at 100, buys 5 at 110, then sells 12 at 120. It realizes 220 gross and leaves 3
at 110. Entry cash commissions 10 and 10 plus the sell commission 12 allocate 26
to matched segments, giving 194 net; 6 of entry commission remain with the open
lot. No execution is inferred from holdings, and supplied net returns are not
rewritten from this attribution. Price slippage is already included in actual
execution prices, not subtracted again from lot PnL.

`segments` retains each matched entry/exit pair, source record identities and
nanosecond timestamps. `open_lots` retains actual unmatched units and entry fees,
without marks, guessed liquidation prices or synthetic terminal closes.
Matching uses exact rational units from decimal input representations, not an
integer share expansion or an arbitrary residual tolerance. Calculated quantity
and cash columns use Float64; source integer prices and record IDs remain exact.
`quantity_exact` preserves rational units that
Float64 cannot exactly represent. Original executions remain available unchanged.
Unknown commissions leave net PnL unknown while preserving gross PnL. Zero entry
notional leaves percentage lot return undefined, not zero. Zero-quantity cash
fees are reported separately as unallocated events, not turned into trades.

On identical integer fills separated by more than eight hours, using the legacy
display symbol `A` for native stable asset ID `7`, published
Pyfolio 0.10.0 reports one closing transaction with 220 PnL; native reports two
matched lot segments totaling 220. Its closing-transaction return 220/1220 is
not the mean of the two segment returns. Pyfolio additionally merges consecutive
same-direction fills within eight hours into VWAP groups; that is a different
analytic convention from this exact-execution FIFO route. Its integer share
expansion also truncates the fractional .1/.2/-.3 fixture and raises instead of
reporting the independently calculable 4 profit. Native fractional matching
retains that profit and leaves no phantom open lot; this is an explicit
reference-defect reconciliation, not a full round-trip-report parity claim.
Using the integer directly as the reference's symbol also triggers a row-upcast
TypeError on these inputs. Native IDs stay integers; the reference comparison
uses the explicit one-to-one display-label mapping, not a changed economic sample.

The named native option `matching="consecutive_vwap"` applies that grouping
convention before the same FIFO matcher; `actual_fifo` remains the default.
`max_gap` defaults to an elapsed eight hours between adjacent same-direction
fills, with exactly eight hours included. It is not a total group-age limit or
a calendar-day boundary. For buys10@100 and5@110 one hour apart, then sell12@120
the next day, grouping realizes200gross/172net and leaves3 at grouped cost310/3
with4 entry fees. Actual FIFO realizes220/194 and leaves3@110 with6 entry fees.
Both give280gross/248net total marked PnL at a final price130. These are different
valid realized-cost conventions, not a tolerance adjustment.

`matching_groups` retains exact quantities/prices, original member record IDs,
first timestamps and last-member `available_at`. Group IDs are separate from
actual execution IDs; a multi-fill group has no invented single execution ID.
The original `executions` remain unchanged. Completed tables/statistics count
closing groups under this explicit profile, rather than collapsing null source
IDs or counting matched entry segments. Fractional matching uses rational units
and corrects the reference's integer truncation in either profile.

Consecutive grouping is retrospective. Extending history can change the last
group until its sign/gap boundary is known. Its representative first timestamp
does not make the eventual VWAP available at that earlier instant. Entry/exit
availability remains in segment/completed tables; an explicit `as_of` filters
raw future fills before grouping. Do not use grouped prices as a causal fill or
intraday state stream. Unknown fees and zero-quantity cash events retain their
native missingness/separate accounting rather than becoming zero costs.

For the familiar extraction workflow, use
`from ml4t.diagnostic.compat import pyfolio as pf`, then
`pf.round_trips.extract_round_trips(transactions, portfolio_value=beginning_nav)`.
The adapter accepts the original pandas `amount`/`price`/`symbol` ledger and
DatetimeIndex, without a return-series placeholder or mandatory legacy install.
It selects the legacy flat-book/eight-hour-group profile over the same native
normalizer and matcher. Gross PnL ignores commission/slippage fields, as the
reference does; use native analysis for cost-aware net results and supplied
opening books. Legacy prices are cash per execution unit, not unscaled contract
points; use native declared multipliers for point-price contracts. Original
symbol objects and integer/string distinctions survive
projection without an execution-package import or row-wise float upcast.

The returned `pnl`, `open_dt`, `close_dt`, `long`, `rt_returns`, `symbol`, `duration`
and optional `returns` preserve the familiar symbol-sorted sample and exact
timestamps. `rt_returns` uses invested entry notional; `returns` uses the supplied
beginning portfolio NAV on the explicit civil-date clock. The classic NAV array
retains IEEE NaN/Inf/signed-denominator division. Native positive-NAV and defined
lot-return defaults remain unchanged; an explicit `nav_method="legacy_ieee"`
selects the array profile on the native contribution method, with native NAV
validity still reported separately. A zero-entry-price trade retains its actual
cash PnL and undefined lot return instead of the reference's `ZeroDivisionError`.
Fractional/negative-price reference defects are corrected, not ported;
no-completion inputs return a typed empty frame.

The example exercises both interfaces and can compare shared inputs with real
published Pyfolio0.10.0/pandas2.3.3 on Python3.12:

```bash
uv run --isolated --python 3.12 --with pyfolio-reloaded==0.10.0 --with pandas==2.3.3 python examples/portfolio_fifo.py --compare
```

The comparison explicitly maps integer ID7 to display labelA on both sides to
avoid the reference's integer-symbol upcast defect; it changes no economic
values or sample. Default native/extraction recipes need no Pyfolio installation.
`pf.round_trips.gen_round_trip_stats(trades)` returns the five familiar `pnl`,
`summary`, `duration`, `returns` and `symbols` tables with their original labels.
It requires the NAV-relative `returns` field. Available `rt_returns` do not replace
missing NAV returns. The classic "Avg. trade net profit" label describes the
supplied `pnl` column; extraction uses gross PnL and does not deduct costs.

For the native tables, call
`result.statistics_tables(pnl="net", return_basis="nav", portfolio_value=beginning_nav)`.
Its default `return_basis="lot"` instead uses invested entry notional. Native
long-form tables keep asset IDs as values. Both interfaces share completed-cash
statistics, retain typed nanosecond durations and expose per-metric coverage.
Missing NAV does not remove known cash PnL; unknown entry time does not become
zero elapsed time. Native unknown cash is rejected rather than reported as zero.
The explicit `method="legacy_pyfolio"` uses the reference's independent masks,
all-row profitable fraction and all-loss profit factor of zero. Native
`TradeStatistics` definitions remain unchanged. Familiar table metadata lives in
each pandas table's `attrs["diagnostic"]`; native metadata and coverage are
separate fields on the returned result. The example compares all five tables.

For caller-defined scalar reductions, use
`pf.round_trips.agg_all_long_short(trades, "pnl", [("sum", "sum"), ("mean", "mean")])`.
Named aggregation pairs, function lists and name-to-reduction mappings retain
their metric-specific samples. Unknown NAV does not remove known cash PnL.
Published Pyfolio 0.10.0 accepts named pairs, but its dictionary route raises
`SpecificationError` on pandas 2.3.3; the adapter supports that dictionary input
without copying the reference error.

`pf.round_trips.apply_sector_mappings_to_round_trips(trades, mapping)` preserves
the original whole-row complete-case behavior. A known cash PnL row with missing
NAV return is therefore excluded. This is not the native label-only attribution
method: use `result.profit_attribution(sector_mappings=mapping)` to retain known
cash without the legacy complete-case mask. The facade's
`attrs["diagnostic_sector_projection"]` reports input/output counts and excluded
source indices. Missing sector keys become `"No Sector Mapping"` as in Pyfolio.

Run `uv run python examples/portfolio_fifo.py --print-tables` to print all five
native-backed familiar tables. The callable
`pf.round_trips.print_round_trip_stats(trades, hide_pos=True)` omits the symbol
table. It writes plain text, not IPython display objects; cash is labeled supplied
account currency rather than assumed USD. Return tables display NAV-relative
percent, while `gen_round_trip_stats` keeps decimal fractions. For the complete
report, use the [familiar round-trip sheet](#use-the-familiar-complete-round-trip-sheet).

`pf.round_trips.add_closing_transactions(positions, transactions)` supplies the
familiar optional closing projection for a declared-flat opening book and a
complete actual ledger. Prices are implied from ending marked cash values divided
by exact remaining units, not claimed observed quotes. Unlike the published
midnight-plus-one-second rule, daily labels resolve to the declared session close,
including early closes. Calendar precedence is explicit argument, `FeedSpec`, then
NYSE. Explicit `calendar=None` disables that inference; pass an actual
`as_of=` if the last label would precede fills. Naive execution instants require
a declared timezone for calendar inference; only that explicit declaration
localizes their copied projection, leaving the source untouched.

The returned rows have an `is_synthetic` flag and authoritative `amount_exact`
for hypothetical quantities. Passing them to `extract_round_trips` keeps closing
cash/duration marked synthetic and retains actual FIFO executions, actual
realization and open lots separately. Gross statistics include only hypothetical
rows that the caller explicitly added; native defaults still exclude them.
Zero-valued marked assets with known nonzero quantity are closed at zero rather
than erased. Missing marks, an inconsistent opening/ending book, double closing
or a close preceding real fills raise instead of inventing state. Use native
supplied-opening inventory and known-at marks for carried books, net costs or
point-price multipliers. The default recipe verifies 200 actual plus 80
hypothetical gross cash PnL under its explicitly selected VWAP profile, with the
three actual open units unchanged.

To save a native round-trip report, run
`uv run python examples/portfolio_fifo.py --report round-trips.html` with Plotly
installed. `create_round_trip_dashboard(analysis, result, ...)` composes the
existing portfolio renderer with all five statistics tables, coverage, trade
lifetimes, Beta(1,1) win-probability uncertainty, fractional elapsed-duration and
cash/return distributions, and asset/optional-sector cash attribution. It returns
the same `PortfolioTearSheet` with `.show()`, `.save_html()` and `.to_dict()`;
prepared observations, raw fills, open lots, synthetic rows and exact nanosecond
keys remain in its owned export. Supplied net account returns are not reconstructed
from realized trade PnL. The default report selects net cash and lot returns;
choose `return_basis="nav", portfolio_value=beginning_nav` explicitly for NAV
contributions. `sector_mappings` maps original IDs to nonempty strings; unmapped
assets remain in "No Sector Mapping" without dropping rows for missing NAV/time.

Asset/sector attribution divides cash PnL by signed total PnL, not by NAV. Shares
may be negative or exceed one; a zero total leaves them undefined. Unknown cash
raises instead of being displayed as zero. Missing entry time omits only its
lifeline/duration, and missing denominators omit only their return observations.
The report displays small samples rather than Pyfolio's fewer-than-five-trades
suppression, and uses the declared Beta(1,1) prior rather than the reference's
undefined all-win/all-loss beta parameters. Every asset is displayed without
random sampling. Browser lifetime axes have millisecond display precision;
hover and exports preserve the original nanosecond instants.

Native reports do not infer terminal fills from end-of-day position labels.
Optional `include_synthetic=True` includes only synthetic segments already
requested from known-at marks in reconstruction, with separate histogram traces,
dashed lifelines and an explicit sample count. Original fills, actual realized
PnL and remaining open lots do not change. Observed and explicit cash-price
intraday routes and the bounded familiar round-trip sheet are described below.
The [full account report](#compose-a-familiar-simple-or-full-account-report)
composes these native results when their inputs are supplied.

### Analyze observed intraday snapshots on their own clock

0.2.0 after 0.1.9,
`ml4t.diagnostic.evaluation.portfolio_analysis.analyze_intraday_exposure`
accepts complete observed marked holdings and optional reported valuations
without a return series. It reuses the existing holdings/occupancy calculations.
Call it with `complete_snapshots=True`; use actual Datetime instants and signed
marked cash values, with cash and NAV in the same currency. It does not infer
quantities, prices, expected bars or missing positions. Missing NAV stays unknown
even when cash plus holdings could be summed. Explicit completeness declares
absent assets zero within each supplied snapshot, not missing snapshots zero.

The result exposes native `positions`, `timeline`, `metrics`, `coverage` and
`metadata`. Maxima describe the supplied snapshots, not continuous-time peaks;
averages/time-in-market count observations rather than elapsed time. `FeedSpec`
and explicit column mappings use the same account boundary. Version 2 preserves
an explicitly declared metadata calendar of `None`. Calendar precedence
is explicit argument, declared metadata, then NYSE; explicit `calendar=None`
disables schedules. Session validation does not replace original instants or
compress overnight observations. Date/session labels cannot stand in for
intraday instants. An explicitly right-labeled closing-bar declaration rejects
an opening-account anchor; a snapshot is not automatically a completed bar.

Run `uv run python examples/portfolio_intraday.py` for the complete observed
workflow. The shared account starts with A = 400, B = -300, cash = 900 and NAV = 1,000.
Two same-instant one-unit buys at 100 with 0.5 cash commission each produce
A = 600, B = -300, cash = 699 and NAV = 999. Gross exposure is 700/900/900 over
three supplied observations; the reported peak fraction is 900/999. The example verifies that omitting
NAV retains raw exposure and leaves normalized ratios unavailable.

With Plotly installed, add `--report intraday.html`. Pass the prepared result as
`intraday=observed` to `analysis.create_tear_sheet`, `create_portfolio_dashboard`
or `create_round_trip_dashboard`. The existing renderer keeps its independent
intraday clock alongside the daily return clock without resampling. The example
uses observation-order axes for intraday time series; exact instants stay on
hover and in the export, without implying equal elapsed spacing. It
uses one separately supplied daily net return of -0.001, three observed snapshots,
the two actual fills and complete carried quantities. No completed trades or
terminal fills are invented. Intraday panels, availability, provenance and owned
export remain separate from return/EOD holdings panels. End-of-day or Date-label
results cannot be relabeled as intraday report peaks.

### Reconstruct an explicit cash-price intraday account

0.2.0 after 0.1.9, `RoundTripResult.reconstruct_intraday_exposure` replays
its owned original fills, complete opening quantities and supplied known-at marks.
Run `uv run python examples/portfolio_intraday.py --reconstruct`; add
`--report reconstructed.html` for the combined report. No Backtest, Data, Engineer
or legacy-package installation is required. This is a separate opt-in route from
reported snapshots: it derives cash-price NAV, never labels it reported NAV.

Call `trips.reconstruct_intraday_exposure(opening_cash=900.0, cash_model="cash_price")`
on the example's supplied opening book. Opening cash already includes historical
fees. Each actual fill changes units and full-notional cash, with its commission
charged once; actual fill prices already include slippage. Original lot entry
basis is not needed to mark holdings and is not invented. The shared fixture gives
cash 900/799.5/699/699, gross marked exposure 700/800/900/900 and cash-price NAV
1,000/999.5/999/999. Both 14:00 fills remain separate states, identified by
`observation_id`, with the same original timestamp. They are not grouped into a
VWAP or assigned fabricated nanoseconds. Counts use exact nonzero quantities,
including held assets marked at zero; exposure signs follow signed marked cash.

Supply an opening instant and terminal `as_of`. A supplied inventory already owns
its opening instant; a declared-flat book needs `opening_as_of=` here. Replay
uses original executions even if FIFO matching selected `consecutive_vwap`.
Fills before the opening anchor or after `as_of` stay excluded. Each held asset
requires a supplied mark known at or before each sampled instant; a terminal
quote cannot value earlier inventory. Fill prices are not silently substituted
for marks. Same-instant marks are declared known before fills; finer quote/fill
sequence is not inferred. Quote timestamps and nanosecond ages remain in the
export, so a stale last-known quote is visible rather than claimed current.
Numeric projections are Float64; `quantity_exact`, `price_exact`,
`multiplier_exact`, `value_exact`, `cash_exact` and `equity_exact` retain the
decimal-input rational values, including large integer prices.
Report exports serialize every temporal field, including secondary quote
timestamps, as ISO values with nanoseconds intact. Native frames retain their
original typed clocks. Read saved HTML as UTF-8, as declared by its charset.

Samples are the opening state, after every actual fill, relevant mark-only
instants and the terminal state if not already sampled. A mark-only quote for
an asset not currently held does not create an account observation. Maxima and
averages describe those event observations, not continuous peaks or time-weighted
occupancy. Calendar resolution keeps the same explicit argument, metadata, NYSE
precedence; `None` disables it. Original instants, identity and execution order
remain intact. The report uses unique observation-order display keys while
exporting the unchanged repeated timestamps and observation IDs.

Unknown opening cash or commission leaves subsequent cash/NAV unknown, not zero;
raw marked exposure stays available. Negative commission rebates are retained.
The explicit full-notional cash-price model is not futures margin accounting.
It requires one consistent contract multiplier per asset and does not infer FX,
dividends, external cash flows or corporate actions. Use supplied observed
snapshots for accounts that require those additional states.

Published Pyfolio 0.10.0's `estimate_intraday` is a different daily heuristic,
not this event-state contract. An actual published-version regression shows its
default pivot mean turns two same-instant one-unit buys at 100 into 100 of
exposure rather than the independently calculated 200. The same two fills at
distinct instants give the correct 200 in that reference. Diagnostic preserves
both actual quantities and returns 200 in both cases; it does not copy that
reference error. The explicit daily projection below is a different method.
The familiar intraday helpers and
[complete account report](#compose-a-familiar-simple-or-full-account-report)
select that daily estimate explicitly, without treating it as observed account state.

### Keep the daily intraday estimate separate from account facts

0.2.0 after 0.1.9, `analysis.compute_intraday_estimate(method="legacy_peak_notional")`
implements the retrospective daily transaction-notional estimate. It is not
observed holdings or a quantity/quote reconstruction. Run
`uv run python examples/portfolio_intraday.py --estimate --report estimated.html`.
The shared carried account's estimate is A = 200, B = 0 and cash = 800, not its
actual A = 600, B = -300 and cash = 699. The report labels these panels
"Estimated Intraday" and keeps the supplied EOD holdings separate.

The method cumulates signed execution cash values within each civil day and
selects every candidate tied for maximum absolute transacted notional. It adds
each candidate to the previous supplied EOD marked values, not to a guessed unit
book. The first risky book is assumed zero; initial cash is inferred from first
EOD marked total divided by one plus the first EOD return. Transaction fees are
ignored by this explicit legacy estimate. It does not compute continuous marked
peaks, actual cash, time-weighted occupancy or causal daily information. A later
known quote can change actual marked exposure without changing this estimate.

Daily `date` values are labels, not intraday instants. Distinct tied candidates
retain observation IDs, original nanosecond `candidate_timestamp` and exact source
membership. Metadata `execution_order`, filtered by civil date and the first
`member_stop` records, identifies the cumulative sample without copying every
prefix. Dates with no candidate carry the shifted book and an unavailable
candidate timestamp, not an invented execution. Simultaneous fills are summed;
the actual published reference's pivot-mean error is not copied.

The independently callable
`ml4t.diagnostic.evaluation.portfolio_analysis.estimate_intraday_positions`
accepts supplied positions, transactions, optional valuations and an explicit
`first_return` scalar with `complete_snapshots=True`. It reuses normalizers,
cash-unit calculations and holdings analysis. The scalar is a decimal portfolio
return for the first EOD period, not an annual rate. `None`/NaN means unknown; no fabricated
finite return series is needed. Native `from_frames` finite-return validation
remains unchanged. Use explicit `calendar=None` or declared metadata `None` for
this civil-date profile; a resolved exchange calendar is not silently disabled.
The same FeedSpec column and timezone declarations are consumed.

For the familiar interface, this flat-at-EOD round trip starts with 1000 cash,
buys two shares at 100 and sells at 102. The selected estimate is 200 in the
asset and 800 cash, not the actual terminal book of 0 and 1004:

```python
import pandas as pd

from ml4t.diagnostic.compat import pyfolio as pf

estimate_dates = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
estimate_returns = pd.Series([0.004], index=estimate_dates)
estimate_positions = pd.DataFrame({"A": [0.0], "cash": [1004.0]}, index=estimate_dates)
estimate_transactions = pd.DataFrame(
    {"symbol": ["A", "A"], "amount": [2.0, -2.0], "price": [100.0, 102.0]},
    index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"]),
)
is_intraday = pf.utils.detect_intraday(estimate_positions, estimate_transactions)
estimated = pf.utils.check_intraday(
    "infer", estimate_returns, estimate_positions, estimate_transactions
)
forced = pf.utils.estimate_intraday(
    estimate_returns, estimate_positions, estimate_transactions, EOD_hour=23
)
assert is_intraday and forced["A"].iloc[0] == 200.0
assert forced["cash"].iloc[0] == 800.0
```

The classifier retains the nonzero-EOD-position count divided by summed daily
unique-transacted-symbol counts and its default threshold of 0.25. Disabled
selection returns the supplied object unchanged; `"infer"` only estimates when
both inputs exist and classification succeeds. `EOD_hour` is accepted but unused,
as actual Pyfolio 0.10.0 probes confirm; it does not define a new session close.
Original asset objects and integer/string distinctions survive projection. The
wide facade reserves `cash` for account cash; native long inputs can represent a
security with that ID separately.

Native `capital_method="defined_only"` keeps unidentified initial capital unknown.
The familiar facade explicitly selects `"legacy_zero_fill"`, matching the reference's
NaN-capital-to-zero projection. With an unknown first return and a 100 cash buy,
that legacy profile can produce estimated cash -100; it is not a reported cash
balance. Infinite implied capital stays unavailable rather than becoming a
financial cash value. The facade also retains the original profile that treats
missing position values as zero; its `attrs["diagnostic"]` discloses methods, filled-cell counts,
candidate clocks and membership. Native unknown holdings are still rejected.
All marked values, executions and cash must use one supplied cash currency;
this method does not provide FX conversion, external-flow accounting or margin
reconstruction. The round-trip adapter below composes these named helpers;
the [complete account report](#compose-a-familiar-simple-or-full-account-report)
composes the other requested portfolio sections from their actual inputs.

### Use the familiar complete round-trip sheet

0.2.0 after 0.1.9, `ml4t.diagnostic.compat.pyfolio.tears.create_round_trip_tear_sheet`
accepts daily returns, complete wide marked positions/cash, actual transactions,
`sector_mappings`, `estimate_intraday` and `return_fig`. Install the visualization
extra. Run `uv run --extra viz python examples/portfolio_fifo.py --familiar-report round-trips.html`
for the supplied account -> native FIFO -> tables/panels -> offline HTML workflow.
The example retains three actual open units while separately showing an optional
hypothetical close. It does not import Pyfolio or require a producer installation.

For a flat-at-EOD strategy, change the import rather than rebuilding its inputs:

```python
import pandas as pd

from ml4t.diagnostic.compat import pyfolio as pf

sheet_dates = pd.DatetimeIndex(["2025-07-07"], tz="UTC")
sheet_returns = pd.Series([0.004], index=sheet_dates)
sheet_positions = pd.DataFrame({"A": [0.0], "cash": [1004.0]}, index=sheet_dates)
sheet_transactions = pd.DataFrame(
    {"symbol": ["A", "A"], "amount": [2.0, -2.0], "price": [100.0, 102.0]},
    index=pd.to_datetime(["2025-07-07T14:00Z", "2025-07-07T15:00Z"]),
)
sheet = pf.tears.create_round_trip_tear_sheet(
    sheet_returns, sheet_positions, sheet_transactions,
    sector_mappings={"A": "Supplied sector"}, estimate_intraday=False, return_fig=True,
)
assert sheet.round_trip_statistics.observations["gross_pnl"].to_list() == [4.0]
sheet.save_html("round-trips.html", include_plotlyjs=True)
```

`return_fig=True` now returns a native `PortfolioTearSheet`, not a Matplotlib
figure. Its `figures`, `round_trip_statistics`, `round_trips`, `to_dict()` and
`save_html()` expose results and sample coverage. The default displays the report
and returns `None`. This execution-only sheet has `metrics=None`: supplied returns
are used for the explicitly selected daily denominator, not replaced with a dummy
series to manufacture a performance summary. Native reports can likewise call
`create_round_trip_dashboard(None, prepared_round_trips, ...)` without returns.

The familiar profile uses gross cash-price PnL, an explicitly flat opening and
consecutive-VWAP grouping. Point multipliers, net fees, supplied opening inventory
and known-at marks belong in the native route. Synthetic terminal closes use the
original supplied ending book, never the estimated peak book, and remain separate
from actual fills/realization. `as_of`, `feed_spec` and `calendar` preserve the
existing closing-helper time contract; naive execution clocks require a declared
timezone for exchange closes. No midnight-plus-one-second close is invented.

`estimate_intraday="infer"` retains the original classifier; `True` forces the
retrospective daily estimate and `False`/`None` retains supplied positions. The
selected daily position-plus-cash sum divided by `1 + supplied return` is the
explicit legacy beginning-NAV profile. An estimated denominator is labeled as
such, never reported NAV. Unknown denominators do not remove known cash PnL.
Sector attribution deliberately uses the original whole-row complete-case mask;
its excluded rows are disclosed. Native label-only attribution stays distinct.
Tied peak candidates remain separate exposure observations. Their equal daily
denominator is used once per actual closing trade, not joined repeatedly to
multiply the trade count. The candidate clocks and collapse count remain in
metadata; disagreement between candidate denominators raises explicitly.
The shared two-day regression exposes a published Pyfolio 0.10.0 error: after
two day-one peak candidates, its positional join assigns a day-one zero to the
next day's return observation despite retaining the 5-unit cash profit. Diagnostic
keeps the three actual closes and the selected-denominator return `5 / (1000 / 1.005) = 0.005025`.
That denominator is the explicit legacy estimate, not the supplied net return.

All five tables and lifetime, probability, holding-time, PnL, return and asset/sector
panels use native calculations. The familiar probability is the explicit
zero-prior `Beta(wins, nonwins)` rather than native `Beta(1,1)` updating. Empty or
one-sided zero-prior samples have no proper posterior and show an unavailable
note; the reference's NaN-density/axis failure is not reproduced. Fewer than five
trades remain visible with a warning. Familiar duration histograms use whole days
and returns display percent of the named NAV denominator; raw durations remain
nanoseconds and raw returns remain fractions. Native fractional elapsed-day,
fraction-return and uniform-prior defaults are unchanged.

The familiar `plotting.plot_round_trip_lifetimes(trades, disp_amount=16, lsize=18, ax=None)`,
`plotting.plot_prob_profit_trade(trades, ax=None)` and `plotting.show_profit_attribution(trades)`
also consume completed tables without inventing execution prices. Figures are
Plotly; an optional `ax` is a Plotly figure, not Matplotlib axes. Sampling uses a
local `RandomState(1)` and never resets global NumPy state. Exact clocks remain on
hover/export. This covers the bounded round-trip route, not all Pyfolio reports.

### Supply opening inventory and known-at FIFO marks

For a non-flat book, pass `initial_state="supplied"`, `opening_lots` and
`opening_as_of`. The opening table requires stable `asset` IDs and signed remaining
`quantity`. Optional `lot_id`, `entry_timestamp`, `entry_price`, `multiplier` and
`remaining_commission` describe the remaining lots, not the full fee of a previously
partially closed fill. Row order explicitly declares FIFO order. Unknown basis,
entry time or fees remain unknown; they do not become zero cost or invented
duration. A known entry cannot occur after `opening_as_of`. The book is complete
immediately before that instant, so earlier executions are excluded from replay.
Opening lots have their own IDs and no fabricated entry execution record.

For known-at valuation, supply `marks` (`timestamp`, `asset`, `price`) and explicit
`as_of`. Only the last quote at or before that instant is used; quote age and source
row remain visible. The same cutoff excludes future executions. Timestamps retain
nanoseconds; undeclared mixed naive/aware clocks raise rather than acquire an
assumed timezone. Marks are not interpolated or fetched. Missing marks/basis keep
unrealized totals unknown, with coverage counts. Marked notional is not inferred
cash or NAV, particularly for futures. Mixed integer/float price types that would
round an original integer price raise; supply a common lossless numeric dtype.

The example also marks the three remaining units at 130, giving 60 gross/54 net
unrealized PnL, separately from actual realized 220/194. Optional
`synthetic_close=True` adds a separate `synthetic_segments` table at `as_of`, with
`is_synthetic=True`, no actual exit record ID and explicitly hypothetical zero
additional closing fees. Actual realized PnL, open inventory and original fills
are unchanged. Every open asset needs a supplied known-at quote for this option.
`marked_gross_pnl`/`marked_net_pnl` sum realized and unrealized PnL; they do not
reclassify hypothetical gains as actual wins. Original opening/mark tables remain
available on the result. These are source-qualified, 0.2.0 routes, not a
claim that published 0.1.9 accepts these arguments.

`result.completed_trades()` aggregates entry-lot segments into one observation per
actual closing fill. The example's two segments therefore give one trade, with
absolute entry notional 1220 and gross/net lot returns 220/1220 and 194/1220.
They are not the mean of segment returns, and neither ratio is a portfolio NAV
contribution. The average entry price is an analytical projection, not a new
execution. Exact rational quantity/cash columns retain components before derived
Float64 conversion; original execution prices remain in the source segments.

`result.to_trade_analysis(pnl="net")` feeds those observations to the existing
`TradeAnalysis` statistics, without a second summary implementation. Native
`TradeMetrics` still defaults to price return; this bridge explicitly selects
PnL over absolute entry notional. Unknown selected PnL raises by default. Select
`pnl="gross"` when fees are unknown, or explicitly choose `missing="exclude"` and
inspect `source_metadata` for excluded counts. Known PnL from a carried lot with
unknown entry time stays in PnL statistics, but duration summaries report only
known durations and their count. No fictitious zero-day holding period is added.

`result.win_probability()` reports a separate Beta(1,1) posterior for strictly
positive PnL on the same closing-fill sample. One winning observation gives
Beta(2,1), posterior mean 2/3 and equal-tail 95% interval
`[sqrt(.025), sqrt(.975)]`. Break-even observations are nonwins; an empty known
sample retains the declared uniform prior. Missing-PnL handling is explicit.
The Bernoulli model assumes a common win probability; it does not establish
independence or account for serially dependent trades.
The existing frequentist `TradeStatistics.win_rate` is unchanged. Synthetic rows
are excluded from tables/statistics/posteriors unless `include_synthetic=True`
is explicitly selected; they remain labeled and do not alter actual realization.

`result.portfolio_contributions(portfolio_value)` takes an explicitly supplied
beginning-period portfolio NAV series, including cash. A frame uses canonical
`date`/`equity`, or explicit `timestamp_col`/`value_col` aliases. The example's
NAV1000 gives .22 gross and .194 net portfolio contributions, alongside unchanged
lot returns 220/1220 and 194/1220. This method neither shifts the supplied NAV nor
infers it from returns, quantities or holdings. For owned account inputs, the
existing transaction timeline's `beginning_nav` supplies the corresponding
already-prepared denominator. By default, missing/nonpositive NAV leaves the contribution
undefined; missing dates are not forward-filled, and unrepresentable ratios raise.
These ratios attribute holding-period PnL and allocated entry fees to the closing
observation. They are not a reconstruction of contemporaneous daily account
returns; the supplied net return path remains authoritative.

The default NAV alignment reuses the owned input's calendar/FeedSpec declaration;
an explicit method `calendar` overrides it. `calendar=None` matches exact row keys,
without turning date labels into midnight events. Exchange calendars use the
existing authoritative session mapping, including overnight CME sessions.
Duplicate NAV observations after session alignment raise instead of multiplying
trade rows. A separately selected `alignment="civil_date", calendar=None` matches
the legacy own-timezone civil-date profile. The published Pyfolio0.10.0 comparison
matches .22 on this common sample; its `returns` here use gross PnL without the
supplied commissions. Native net contribution .194 remains separately named.

The existing [completed-trade consumer](trade-analysis.md#preserve-execution-units-and-timing)
now accepts same-instant closures, explicit point multipliers, actual zero/negative
prices and signed cash-fee rebates. Producer-to-consumer tests pass these real
FIFO rows through `TradeRecord` and `TradeAnalysis` without altering the execution
quantity, prices or fees. Undefined price/duration ratios remain undefined;
nanosecond instants survive metric tables and JSON. Use the
[complete account report](#compose-a-familiar-simple-or-full-account-report)
to combine those results with the other requested portfolio sections.
### Consume a Zipline performance result without a legacy analytics dependency

The familiar extraction call is also available as
`ml4t.diagnostic.compat.pyfolio.utils.extract_rets_pos_txn_from_zipline(performance)`.
It projects the same normalized native inputs into a pandas `(returns, positions,
transactions)` tuple. Actual Asset objects need no Zipline import in the consumer;
portable dictionaries retain stable integer SID labels. Raw integer SIDs use
the legacy cash-price-per-unit convention, or explicitly supplied
`asset_metadata={sid: {"multiplier": ...}}` for point-price contracts.
Closing cash comes from the producer, not an inferred exposure residual.
If a partial export omits a cash observation, it remains unknown, not zero.
This deliberately corrects the reference extractor's blanket zero fill; the
normalizer already supplies zero only for assets absent from a complete book.

Daily tuple return/position labels follow the legacy civil-date/naive-UTC
convention. Original event timestamps remain timezone/ns precise in the
transaction ledger. Unlike the reference extractor, it does not mutate the
source index or drop a complete all-cash date before the first position. The
captured first-day cash1000 is therefore retained, not replaced with a missing
book. Unknown commissions remain unknown. Native `portfolio_analysis_from_zipline`
below preserves the original observation clock instead of this explicit daily
compatibility projection. Both routes are exercised by
`examples/zipline_portfolio_inputs.py` on portable captures and actual engine output.

0.2.0 after 0.1.9, `integration.zipline.portfolio_analysis_from_zipline()`
extracts `run_algorithm` results into the same `PortfolioAnalysis.from_frames`
boundary. It imports neither Zipline nor Pyfolio. The result's reported returns,
`portfolio_value` and `ending_cash` remain authoritative. Nested positions are
complete snapshots; empty lists prove an all-cash date, while missing lists are
invalid rather than silently empty. Missing optional columns disable their
dependent analyses without blocking returns-only analysis.

```python
import pandas as pd
from ml4t.diagnostic.integration.zipline import portfolio_analysis_from_zipline

zipline_dates = pd.to_datetime(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
zipline_example = pd.DataFrame({
    "returns": [0., -.0003], "portfolio_value": [1000., 999.7],
    "ending_cash": [1000., 699.7],
    "positions": [[], [{
        "sid": {"sid": 7, "symbol": "A", "multiplier": 1.},
        "amount": 3, "last_sale_price": 100.,
    }]],
    "transactions": [[], [{
        "sid": {"sid": 7, "symbol": "A", "multiplier": 1.},
        "amount": 3, "price": 100., "dt": zipline_dates[1],
        "order_id": "observed-order", "commission": None,
    }]],
}, index=zipline_dates)
zipline_analysis = portfolio_analysis_from_zipline(zipline_example, calendar="NYSE")
zipline_holdings = zipline_analysis.compute_holdings_analysis()
assert zipline_holdings.timeline["gross_exposure"].to_list() == [0., 300.]
assert zipline_holdings.timeline["cash"].to_list() == [1000., 699.7]
assert zipline_analysis.input_data.transactions["asset"].to_list() == [7]
assert zipline_analysis.input_data.transactions["commission"][0] is None
print("Zipline input preserves sid, reported cash/NAV and unknown fill costs")
```

The two-row snippet illustrates the format. The repository fixture
`tests/test_integration/data/zipline320_account.json` is captured from an actual
published Zipline 3.2.0 `run_algorithm` execution using pandas 2.3.3, NumPy 2.4.6
and exchange-calendars 4.13.2. It includes an opening buy and partial sale,
reported cash/NAV, an all-cash initial date and unknown per-fill commission.
The optional producer example
[zipline_portfolio_inputs.py](https://github.com/ml4t/diagnostic/blob/6ac080f97917e5a149f881edd32553d74a86c575/examples/zipline_portfolio_inputs.py)
regenerates that local deterministic bundle without market downloads and consumes
the actual in-memory result, including Zipline Asset objects:

```bash
uv run --isolated --no-dev --python 3.12 --with zipline-reloaded==3.2.0 --with pandas==2.3.3 --with plotly python examples/zipline_portfolio_inputs.py --capture zipline-account.json --report zipline-account.html
```

For the ordinary consumer path, run
`uv run --no-dev python examples/zipline_portfolio_inputs.py --input tests/test_integration/data/zipline320_account.json`.
No execution package is needed to consume the capture. The four-date result has
420 of actual traded notional; unknown fill commission is not replaced by zero
or inferred from aggregate account charges. The returns already include the
observed commissions once. The same prepared transaction results can be passed
to the existing portfolio report as in the preceding workflow.

Stable integer `sid` identifies each instrument, even if display symbols change
or are reused. Asset objects supply their declared `price_multiplier`; portable
asset dictionaries carry `sid`, `symbol` and `multiplier`. Bare integer IDs need
an explicit `asset_metadata` mapping, such as `{7: {"symbol": "A", "multiplier": 1}}`.
Marked notional is amount times last sale price times multiplier. That value
does not replace the producer's cash or NAV for futures or leveraged accounts.
Actual execution `dt`, order ID and signed amount are retained; no fill is
reconstructed from a position change. Feed declarations and explicit calendar
precedence use the same shared contract. This extraction does not implement a
Zipline broker, data bundle service, Pipeline, capacity model, or full Pyfolio
tear sheet.

### Consume actual Backtest account capture

The existing `integration.portfolio_analysis_from_result()` now uses the same
owned input boundary for producer returns, reported account values and actual
fills. Explicit `calendar=None` retains observations and requires an explicit
`periods_per_year`. An omitted calendar continues to resolve producer metadata.
Dated benchmarks align by keys; explicitly positional benchmark arrays retain
their order and must match in length.

For complete per-asset holdings, the producer must actually enable
`Engine(..., holdings_sampling="bar")`. With observation returns, the bridge
consumes those complete bar snapshots, including all-cash observations. It checks
reported completeness and position counts before filling absent assets with
zero. It does not reconstruct holdings from trades or target weights. Older
producers without captured holdings still support returns/account/fill analysis;
the dependent holdings report remains unavailable. Aggregated-return conversion
of non-bar or differently sampled holdings remains unqualified.

The complete-capture route works with published Backtest 0.2.0 and Specs 0.2.0.
The earlier capture was qualified against immutable Backtest source
`da223f904cfa03744152311e3c617c15fdc1ddcd`; its retained provenance is unchanged.
The executable
[backtest_portfolio_inputs.py](https://github.com/ml4t/diagnostic/blob/6ac080f97917e5a149f881edd32553d74a86c575/examples/backtest_portfolio_inputs.py)
runs an actual Engine and checks returns `[0, 0, .02]`, cash `[1000, 800, 800]`,
marked exposure `[0, 200, 220]`, NAV `[1000, 1000, 1020]` and 200 actual filled
notional. Ten additional bps costs `.0002` of beginning NAV. The source-qualified
recipe using the published producer is:

```bash
uv run --isolated --no-dev --python 3.12 --with 'ml4t-backtest==0.2.0' --with plotly python examples/backtest_portfolio_inputs.py --run-engine --report backtest-account.html
```

Backtest 0.2.0 satisfies Diagnostic's Specs 0.2 requirement; older Backtest
0.1.12 requires Specs below 0.2 and cannot be combined with this Diagnostic
source. Legacy frame exports
that dropped timezone information recover it only from matching actual public
Fill/account record timestamps, never from a guessed timezone or return date.
Standalone supplied-account and captured Zipline analysis need no Backtest
installation.
Both producer scripts consume bundled actual captures when run without arguments,
so ordinary examples need no execution engine. `--capture` explicitly regenerates
Zipline input; Backtest regeneration requires `--run-engine`. The Backtest minute
example declares 252*390 periods per year, rather than using an hourly scale.

The bridge retains the producer's known initial capital as `initial_nav` for
the first transaction denominator, including a same-bar opening fill. Supplied
account users can pass that value to `PortfolioAnalysis.from_frames` explicitly.
It does not invent an initial timestamp or fill later missing NAV observations.
Net supplied returns already include actual fees; a zero-extra-cost scenario
leaves them unchanged. Additional bps are separate hypothetical execution costs.

Actual Backtest external capital records survive in the owned input metadata and
the report's `external_capital_flows` availability/export entry. The HTML states
their count and net account-currency amount. Reported NAV includes deposits and
withdrawals; cumulative investment returns use the producer's flow-adjusted
return series. They are not recomputed as a ratio of ending to initial NAV.
Original position quantities, marks and multipliers remain in owned inputs;
reported liquidation value and notional exposure are kept distinct.

### Analyze dated stress, live splits and training-only wealth cones

Current 0.2.0 source can evaluate an existing dated portfolio without a
backtest engine. The same owned account boundary feeds native summaries,
stress/live analysis and the existing report. Stationary resampling provides
an explicit dependence-aware alternative to IID draws; this is a tested reason
to evaluate the native workflow, not a claim of superior forecast accuracy.
Install the `viz` extra for the report calls below; numeric analysis remains core-only.

```python
import pandas as pd

from ml4t.diagnostic.evaluation import PortfolioAnalysis

regime_dates = pd.date_range("2025-01-13", periods=12, tz="UTC")
regime_returns = pd.Series(
    [0.01, -0.02, 0.005, 0.02, -0.01, 0.03,
     -0.005, 0.01, -0.01, 0.02, 0.04, -0.03],
    index=regime_dates,
)
regime_account = PortfolioAnalysis.from_frames(
    regime_returns, calendar=None, periods_per_year=252,
)
regime_stress = regime_account.compute_stress({
    "selected": (regime_dates[1], regime_dates[4]),
    "unobserved": ("1990", "1990"),
})
regime_live = regime_account.compute_live_split(live_start=regime_dates[8])
regime_cones = regime_account.compute_wealth_cones(
    live_start=regime_dates[8], horizon=7, n_samples=256,
    method="stationary", block_size=3, seed=42, initial_capital=1000,
)
assert regime_stress.coverage["n_observations"].to_list() == [4, 0]
assert regime_cones.training.height == 8 and regime_cones.live.height == 4
assert regime_cones.paths.height == 1792
regime_report = regime_account.create_tear_sheet(
    stress=regime_stress, live=regime_live, wealth_cones=regime_cones,
)
regime_report.save_html("regimes.html", include_plotlyjs=True)
print("Stress/live/cones: 4 selected, 8 training, 4 live; steps count observations")
```

Stress datetime endpoints are inclusive instants. Civil dates and partial
year/month strings select their full civil interval in the input timezone.
Coverage distinguishes unobserved, partially covered and observed extents;
it does not infer a trading grid or declare missing sessions. Native default
historical windows use the supplied sample extent for open-ended events.
Empty explicit `periods={}` requests no windows. Benchmarks retain the same
native aligned observations and scale as the account. Returns, drawdowns and
volatility are fractions; Sharpe and beta are dimensionless.

The live split uses `<live_start` and `>=live_start`, preserving actual fill
nanoseconds. Cone training uses only pre-live returns. Each simulated row
retains its source training-row identity and sampled return; benchmark draws
use that same row. Changing all live outcomes leaves draws and bands unchanged.
Wealth compounds from positive initial capital, including absorbing total loss.
The seed controls a local Generator, not global random state. Stationary
`block_size` counts training observations and wraps circularly.

Forecast steps count observations, not days or inferred future sessions.
`calendar` follows explicit argument, then input declaration, then NYSE;
explicit `None` disables it. For this observation forecast it is provenance
only and never invents future dates. Quantile lines use linear quantiles.
Mean-plus/minus-population-standard-deviation bounds are separately labeled,
may be negative and are not mean confidence intervals. Actual live overlays
retain supplied dates; simulated future steps have no fabricated timestamps.

The same prepared report renders coverage/statistics tables in notebook
`regime_report.show()` and saved HTML, alongside the actual Plotly paths.
Result exports retain the owned Arrow clocks and method declarations. The
native prepared-report notebook test executes a real kernel; it is not a
qualification of every original Pyfolio notebook or combined report.

Run the shared-input recipe, including optional offline reports:

```bash
uv run python examples/portfolio_regimes.py --task all
uv run --extra viz python examples/portfolio_regimes.py --task all \
  --output regimes.html --familiar-output stress.html
```

The familiar `compat.pyfolio.timeseries` calls are `simulate_paths`,
`summarize_paths`, `forecast_cone_bootstrap` and
`extract_interesting_date_ranges`. Their explicit profile preserves Pyfolio
0.10.0's local RandomState stream, NaN-neutral path products, numeric std
columns and `starting_value=0` net-return convention. Native finite-input
and positive-capital defaults do not change. Familiar event selection retains
partial pandas label slices and missing observations; unobserved periods are
omitted there, unlike native coverage. Invalid boundaries raise rather than
reproducing the reference's blanket suppression. Default Covid ends at call
time instead of module import; supply explicit periods to freeze the sample.

`compat.pyfolio.tears.create_interesting_times_tear_sheet` includes original
mean/min/max percent statistics, all event paths and available benchmarks.
It returns offline HTML, or per-event Plotly figures with `return_fig=True`;
`output_file` saves the complete report in either mode. Missing benchmark
events are unavailable, not zero-return curves. `legend_loc` maps to documented
native positions without changing global Matplotlib state. Familiar
`plotting.plot_cones` supports zero through three original lower-two-std
crossing redraws. They additively shift the same supplied bounds from realized
wealth, not retrained forecasts; the original `num_samples` and `random_seed`
arguments do not sample anything on that plotting route.

Shared-input tests execute actual Pyfolio 0.10.0 / Empyrical 0.7.0 with pandas
2.3.3, including raw draws, all bounds, event samples, original printed stress
statistics and actual Matplotlib redraw artists. Combined returns/full-sheet,
bootstrap and original notebook routes have their own regression corpora and
recipes above; the stress/cone comparison does not substitute for those checks.

### Analyze prepared holdings on observed keys

In 0.2.0 source after 0.1.9, use the existing `PortfolioAnalysis` workflow
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
renders these sector tables. It consumes owned holdings rather than simulating an account.
Use [captured Backtest holdings](#consume-actual-backtest-account-capture) for producer
snapshots, the [supplied-account boundary](#normalize-a-supplied-account-without-a-backtest-engine)
for declared clocks and external inputs, and the
[complete account report](#compose-a-familiar-simple-or-full-account-report) for
requested transactions, cost scenarios and other portfolio sections.

#### Render prepared holdings in the portfolio dashboard

This 0.2.0 source route is not included in published 0.1.9.
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
actionable preparation reason. Supply an owned transaction analysis through
`transactions=` to include actual turnover and incremental-cost panels; the
positions flag alone does not create transaction data.

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

These native prepared-holdings panels use positive-NAV definitions. The familiar
position manifest below instead selects the legacy signed-NAV profile. Neither
route establishes all portfolio reports. The separately tested 0.2.0
[supplied-account boundary](#normalize-a-supplied-account-without-a-backtest-engine),
[actual transaction analysis](#analyze-actual-transactions-and-incremental-costs) and
[capacity workflow](#supplied-market-capacity-and-impact-scenarios-unreleased) have their own
methods and coverage below or above. The [familiar account composition](#compose-a-familiar-simple-or-full-account-report)
reuses those components. The [original notebook](#replay-the-original-pyfolio-analytical-notebooks)
and [provider](#acquire-inputs-with-data-then-analyze-them-in-diagnostic) recipes
exercise their complete journeys separately from this holdings dashboard.

#### Keep familiar position projections without changing native account defaults

0.2.0 after 0.1.9, `compat.pyfolio.pos` provides `get_percent_alloc`,
`get_top_long_short_abs`, `get_max_median_position_concentration`,
`get_long_short_pos`, `get_sector_exposures` and `extract_pos`. These adapters
reuse the native holdings extrema, concentration, exposure and marked-value
calculations. They retain pandas labels and original asset IDs, including
input-column tie order for top positions. This is the numerical input layer
for familiar reports, not qualification of the complete positions tear sheet.

```python
import pandas as pd
from ml4t.diagnostic.compat import pyfolio as pf

position_dates = pd.date_range("2025-02-03", periods=2, tz="UTC")
position_values = pd.DataFrame(
    {203: [100.0, -20.0], 101: [-40.0, 10.0], "cash": [140.0, -40.0]},
    index=position_dates,
)
allocation = pf.pos.get_percent_alloc(position_values)
long_short = pf.pos.get_long_short_pos(position_values)
assert allocation.loc[position_dates[0], 203] == 0.5
assert long_short.loc[position_dates[0], "short"] == -0.2
assert allocation.loc[position_dates[1], 203] == 0.4
assert long_short.loc[position_dates[1], "long"] == -0.2
top_long, top_short, top_absolute = pf.pos.get_top_long_short_abs(position_values)
assert top_absolute.index[0] == 203
print("Familiar positions: original assets, signed NAV and per-metric missingness")
```

The negative-NAV row deliberately shows the familiar sign reversal. Native
holdings results still require positive finite NAV for normalized fractions;
their raw marked exposures remain available independently. Missing positions
are omitted from a row sum, not globally replaced with zero: allocation keeps
each missing cell, while `get_long_short_pos` leaves all ratios missing when
account cash is missing. Concentration selects signs of the normalized
allocation, which can differ from the marked-position sign at negative NAV.
Empty concentration tables retain the reference's empty RangeIndex.

Sector projections warn and exclude unmapped assets, matching the familiar
workflow. They reject a sector literally named `cash`: Pyfolio 0.10.0 overwrites
that sector with account cash, so 60 of holdings plus 40 of cash becomes only
40. Use the [native sector result](#render-prepared-holdings-in-the-portfolio-dashboard)
to retain both namespaces and unknown classifications.

`extract_pos` consumes actual `sid`, `amount` and `last_sale_price` records and
supplied cash, without inventing returns. Declared asset-object multipliers
are honored without installing Zipline. Repeated date/asset records retain the
reference's pivot-mean convention, not additive fills. The supplied cash clock
must cover every position date; complete all-cash dates remain in the result
with zero asset values. This corrects the reference's left join, which drops
cash-only dates. Inputs remain unchanged, including the cash Series name and
frequency metadata; the projected index has no inferred frequency, as in the
reference. Absent or NaN marked cells use the familiar zero-filled projection;
do not apply it to incomplete holdings. Missing cash instead remains unknown,
not the reference's invented zero balance. This projection is not account
replay or inferred portfolio NAV.

#### Run a complete familiar positions report

This 0.2.0 route composes the original exposure, top allocations,
maximum/median concentration, holdings counts, long/short counts and gross
leverage panels. Requested sector paths and top long/short/absolute tables use
the same native kernels as the numerical adapters. It requires dated returns
and a wide marked-value frame with separate `cash`; no broker or Reloaded
runtime is required. Example snapshots below are explicitly constructed inputs,
not observed fills or a reconstructed account:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.compat import pyfolio as pf

position_report_prices = pd.Series(
    [100.0, 101.0, 99.0, 103.0],
    index=pd.bdate_range("2025-02-03", periods=4, tz="UTC"),
)
position_report_returns = position_report_prices.pct_change(fill_method=None).iloc[1:]
example_nav = 10000 * position_report_prices.iloc[1:] / position_report_prices.iloc[0]
position_report_values = pd.DataFrame(
    {"Core": .6 * example_nav, "Hedge": -.2 * example_nav, "cash": .6 * example_nav}
)
position_report = pf.create_position_tear_sheet(
    position_report_returns,
    position_report_values,
    sector_mappings={"Core": "Core", "Hedge": "Hedge"},
    estimate_intraday=False,
    return_fig=True,
)
np.testing.assert_allclose(position_report.figures["Gross leverage"].data[0].y, .8)
assert len(position_report.figures) == 10
position_report.save_html("positions-report.html", include_plotlyjs=True)
print("Complete familiar positions manifest: shared kernels, owned identities and explicit estimates")
```

Run the existing whole-return example with its companion position report:

```bash
uv run --extra viz python examples/pyfolio_returns_report.py \
  --positions-output positions-report.html --output returns-report.html
uv run --extra viz --with pyfolio-reloaded==0.10.0 \
  --with empyrical-reloaded==0.7.0 --with pandas==2.3.3 \
  python examples/pyfolio_returns_report.py --compare \
  --positions-output positions-reference.html --output returns-reference.html
```

`show_and_plot_top_pos=0` selects top paths, `1` selects top tables and `2`
selects both. `hide_positions=True` forces plot-only output and anonymizes asset
and sector labels before building any figure. Original IDs do not remain in
HTML, data exports or Plotly notebook MIME. Caller inputs remain unchanged.

The default `estimate_intraday="infer"` preserves the existing Pyfolio heuristic
when actual transactions are supplied. `True` forces the explicit daily
transaction-notional estimate; `False` or `None` keeps supplied positions. The
report labels estimates `estimated_intraday`, preserves tied candidate rows
with separate observation identities, and does not call them observed marked
holdings or continuous-time peaks.

Legacy normalization stays panel-specific. Exposure plotting omits missing cash
from the closing row sum, unlike `pos.get_long_short_pos`, which leaves those
ratios unknown. Undefined fractions remain gaps, not zero risk. Sector plots
exclude unmapped assets with a warning and normalize mapped sectors plus cash;
use native prepared holdings for unknown-sector retention and full-NAV weights.

`return_fig=True` returns the existing `PortfolioTearSheet`, not a Matplotlib
figure. Use `.save_html()`, `.show()` or `.to_dict()`; Matplotlib axes, artist
mutation and plotting-style kwargs do not migrate. Notebook display requires
a Plotly-compatible frontend. Exact keys remain in hover/export; horizontal
spacing follows categorical observation order, not elapsed exchange sessions.
This positions workflow is also used by the familiar full account report.
The [original notebook replays](#replay-the-original-pyfolio-analytical-notebooks)
exercise the complete report compositions with their pinned analytical inputs.

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

The exposure and activity recipes below work with published Backtest 0.2.0 and
Specs 0.2.0. Upgrade older Backtest versions whose Specs bounds are incompatible;
do not override their dependency bounds. Earlier captured qualification against
`c6feb24084ec89c3342f5703cc47ea2524b521f4` remains historical producer evidence.
The supplied-frame examples above do not require a Backtest installation.

```bash
uv run --isolated --no-dev --python 3.12 --with plotly \
  --with 'ml4t-backtest==0.2.0' \
  python examples/legacy_migration_comparison.py --task exposure
```

Cash is 2000 after the fill. At the final marks, equity is respectively 500, 0
and -500 while gross exposure is 1500, 2000 and 2500. The first case has gross
ratio 3 and net ratio -3. The other final ratios are unavailable; each window
has two defined snapshots out of three. This capability reuses Backtest's
observed state and Diagnostic's common profile/report analysis.

This is an 0.2.0 correction after 0.1.9, not direct ingestion of a Pyfolio
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
uv run --isolated --no-dev --python 3.12 --with plotly \
  --with 'ml4t-backtest==0.2.0' \
  python examples/legacy_migration_comparison.py --task activity
```

The short fills 10 shares at 100; the cover fills at 150, 200 or 250. With
initial equity 1000, the final equity is 500, 0 or -500 and closing notional is
1500, 2000 or 2500. The positive-equity close has turnover 3 and active mean 2.
The other closing ratios are unavailable; their active mean is 1 from one of
two active observations. Their three-row windows have two normalized
observations, and the zero-equity example generates the actual full report.
This is an 0.2.0 correction after 0.1.9.

On shared positive-equity inputs, pinned Pyfolio 0.9.9
`txn.get_turnover(..., denominator="portfolio_value")` agrees with the current
per-observation ratios. Its default is `"AGB"`, which averages consecutive
gross books and is a different denominator. Pyfolio also normalizes timestamps
to civil dates and fills NaN ratios with zero; Diagnostic's current bridge
retains exact observation timestamps. This example has one observation per
explicit daily input date. It does not establish exchange-session alignment,
intraday aggregation or AGB support. Invalid-NAV reference divisions are
recorded as differences, not parity targets.

For direct supplied-frame ingestion, unknown fees, extra-cost scenarios and explicit
observation/session grouping, use the newer
[actual-transaction workflow](#analyze-actual-transactions-and-incremental-costs).
The earlier bridge example here does not qualify those methods. External-flow
adjustments use the supplied-account boundary; complete sheets retain their
per-component input and numerical contracts.
The full performance report still requires valid simple returns: a negative-NAV
step below -100%, or a later 0/0 return after NAV reaches zero, is not supported
by the current performance workflow. Standalone activity analysis does not
remove those restrictions.

### Supplied-market capacity and impact scenarios (0.2.0) {#supplied-market-capacity-and-impact-scenarios-unreleased}

Evaluate Diagnostic when you want to combine supplied account inputs, explicit market-coverage
checks and typed capacity results in the existing portfolio report. The native result can be
exported as JSON without losing nanosecond keys or unbounded durations. This is a hypothetical
liquidity analysis, not a fitted execution model or a speed claim.

The runnable [capacity example](https://github.com/ml4t/diagnostic/blob/5bc2dbbce947d072b500cbc88a6508021eceac23/examples/portfolio_capacity.py)
constructs a consistent two-asset toy account with actual stated fills, then runs:

```python
import pandas as pd
from ml4t.diagnostic.evaluation import PortfolioAnalysis

dates = pd.date_range("2025-01-13", periods=3, tz="UTC")
empty_fills = pd.DataFrame({
    "timestamp": pd.DatetimeIndex([], tz="UTC"),
    "asset": pd.Series(dtype=str),
    "quantity": pd.Series(dtype=float),
    "price": pd.Series(dtype=float),
})
analysis = PortfolioAnalysis.from_frames(
    pd.Series([0.0, 0.001, -1 / 1001], index=dates),
    positions=pd.DataFrame({"A": [100.0, 101.0, 100.0], "cash": 900.0}, index=dates),
    valuations=pd.DataFrame({"equity": [1000.0, 1001.0, 1000.0], "cash": 900.0}, index=dates),
    transactions=empty_fills,
    transaction_timestamp_col="timestamp",
    complete_snapshots=True,
    calendar=None,
)
market_data = pd.DataFrame({
    "date": dates, "asset": ["A"] * 3,
    "price": [10.0, 10.1, 10.0], "volume": [10_000.0] * 3,
})
capacity_result = analysis.compute_capacity(
    market_data,
    currency="USD",
    price_basis="raw",
    volume_basis="raw",
    participation=0.1,
    lookback=1,
    base_capital=1000,
    scenario_capitals=[1000, 2000, 3000],
)
analysis.create_tear_sheet(capacity=capacity_result).save_html("capacity.html")
print("Capacity report preserves the explicit empty actual ledger")
```

`analysis` uses the owned `PortfolioAnalysis.from_frames` boundary. Positions are complete,
signed account-currency marked values; fills retain actual quantities and point multipliers.
Market data uses the existing `FeedSpec` timestamp, asset, price and volume mappings. Declare one
common currency and matching price/share-volume adjustment basis. Diagnostic performs no FX,
corporate-action adjustment, provider acquisition or trade simulation here. A market multiplier
defaults to one; supplied execution multipliers must agree. Missing market data is not filled.

Run the complete account-to-report workflow, and optionally compare the actual reference:

```bash
uv run --extra viz python examples/portfolio_capacity.py --output capacity.html
uv run --extra viz --with pyfolio-reloaded==0.10.0 --with empyrical-reloaded==0.7.0 --with pandas==2.3.3 python examples/portfolio_capacity.py --compare --output capacity-reference.html
uv run --extra viz --with ml4t-data==0.2.1 python examples/portfolio_capacity.py --data-provider --output provider-capacity.html
uv run --extra viz python examples/portfolio_capacity.py --familiar-report familiar-capacity.html
```

The provider command uses the actual offline `SyntheticProvider`, not observed exchange liquidity.
If an older single-symbol provider omits its identity column, the example carries the explicit
requested symbol alongside those bars. It never infers an asset from prices or row positions,
and it rejects a contradictory returned identity. Published Data 0.2.1 supplies that column itself.
No Data or Reloaded installation is required for the supplied-input command. The example prints
installed package versions. These new APIs belong to an 0.2.0 source/wheel candidate, not
the immutable published Diagnostic 0.1.9 distribution.

For familiar calls, change the import to
`from ml4t.diagnostic.compat.pyfolio import capacity`. The five capacity functions retain pandas
labels and their distinct legacy conventions: fixed capital multiplied by signed NAV allocation,
trimmed row windows, inclusive recent liquidation versus strict recent transaction civil-day
cutoffs, and compounded closing-NAV impact. `plot_capacity_sweep` in `compat.pyfolio.plotting`
returns Plotly instead of Matplotlib; it retains the capital grid and stop-before-Sharpe-below-minus-one
rule, with explicit optional `periods_per_year`. Sharpe is dimensionless. Native extra-impact
scenarios instead use beginning-period NAV and actual marked position dollars. Neither route
deducts observed fees in supplied net returns again.

Liquidation uses `mean(price * volume * multiplier).shift(1)`, not a product of separate means
or future volume. Native observation windows retain the common supplied clock and asset holes;
`window_unit="session"` also retains missing scheduled rows. Calendar resolution is explicit
argument, then declared market/account feed, then NYSE; explicit `calendar=None` preserves exact
row keys. Delayed supplied `known_at_col` observations cannot enter an earlier decision.
Recent native summaries count observations, not elapsed civil days.
Capacity plots retain distinct nanosecond observations on equal-spaced observation axes with
exact keys on hover. A complete-book absent asset is plotted at zero exposure; an unknown
market denominator or unbounded liquidation time remains a gap, not zero. Finer market-key
precision is used internally without changing the authoritative account/report observation keys.

Material corrections relative to Pyfolio 0.10.0 are deliberate. Its daily transaction helper
raises on two traded assets sharing a date; the adapter retains both asset/date records.
Its liquidation helper replaces known zero volume with NaN; Diagnostic reports an open position
as unbounded. Missing active market volume remains unavailable in impact scenarios rather than
being summed into a zero penalty. Native reports name unavailable and unbounded cases and do
not draw unbounded duration as zero. Defined reference liquidation, daily transaction, summary
and impact cases are compared separately from these independent corrections.

The 0.2.0 `compat.pyfolio.tears.create_capacity_tear_sheet` composes all four familiar
full/recent liquidation/participation tables and the capital-performance sweep. It retains
`liquidation_daily_vol_limit`, `trade_daily_vol_limit`, `last_n_days`,
`days_to_liquidate_limit` and the original `estimate_intraday="infer"`, forced and disabled
choices. It reuses the estimator described above, not a second account-reconstruction engine.
Tied peak candidates remain separate dated observations; the worst-row allocation comes from
that same candidate, not a Series of ambiguous duplicate-date values. Such retrospective
estimates are not observed inventory or causal intraday liquidity.

Both liquidation views honor the caller's threshold. Actual Pyfolio 0.10.0 instead hardcodes
`> 1` for the recent view and prints a 20% assumption even when a different participation limit
was requested. The guide tests the actual original full sheet, its four rendered tables and
Matplotlib curve against the same supplied inputs, and separately verifies those corrections.
Recent helper windows count civil days from their respective supplied endpoints, not exchange
sessions. Unknown estimates remain visible as unavailable; unbounded risk is not zero.

The familiar function returns offline HTML or its saved path, replacing implicit notebook
display. `return_fig=True` returns the single Plotly sweep, not a Matplotlib Figure;
`output_file` still saves every table in that mode. Optional `periods_per_year` and `impact`
make the annualization scale and hypothetical coefficient explicit. The original fixed-capital,
five-market-row liquidation and closing-NAV impact conventions remain different from the native
result. Use the [complete account report](#compose-a-familiar-simple-or-full-account-report)
to combine capacity with supplied returns, holdings, transactions and attribution.

The development routes in this guide provide the seven Alphalens and nine Pyfolio
analytical sheet families, with preserved original-notebook recipes and explicit
native/legacy method choices. They are not part of the published 0.1.9 wheel.
Keep the installed legacy workflow until a successor distribution containing the
required route has been published and verified in your environment. Compare
annualization, benchmark alignment, cash, costs and the stated sample conventions
when migrating; matching API names alone does not establish equivalent inputs or results.
