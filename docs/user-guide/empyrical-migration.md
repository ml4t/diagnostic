# Migrating Selected Empyrical Metrics

Diagnostic covers the portfolio metrics listed below, but it does not replace
every Empyrical workflow. Move one calculation at a time, retain its input
cadence and missing-value policy, and compare the result on your own returns.
For Pyfolio returns analysis or Alphalens factor analysis, use the
[separate migration guide](migration.md).

If portfolio metrics are one part of a larger research workflow, Diagnostic
also documents [purged cross-validation and multiple-trial Sharpe
assessment](cross-validation.md). That capability is a reason to evaluate it,
not evidence that every Empyrical function has an equivalent.

## Run the comparison

From a checkout of this repository, run the published packages on one
deterministic returns series with Python 3.12:

```bash
uv run --no-project --python 3.12 \
  --with 'ml4t-diagnostic==0.1.8' \
  --with 'empyrical-reloaded==0.5.12' \
  --with 'pytz==2026.5' \
  python examples/legacy_migration_comparison.py --task empyrical
```

The script at `examples/legacy_migration_comparison.py` calls real Empyrical
functions and Diagnostic `PortfolioAnalysis` on the same 252 daily decimal
returns. It checks complete-data agreement, then replaces one return with
NaN. These are observed results for versions 0.5.12 and 0.1.8, not a general
equivalence guarantee.
The separate `pytz` pin supplies a runtime import used by Empyrical 0.5.12
but absent from that release's dependency metadata.

<div class="migration-table" markdown="1">

| Empyrical call | Diagnostic result from `PortfolioAnalysis(...).compute_summary_stats()` | Complete-input result in both |
|---|---|---:|
| `annual_return(..., annualization=252)` | `.annual_return` | 0.079730 |
| `annual_volatility(..., annualization=252)` | `.annual_volatility` | 0.150155 |
| `sharpe_ratio(..., annualization=252)` | `.sharpe_ratio` | 0.585568 |
| `sortino_ratio(..., annualization=252)` | `.sortino_ratio` | 0.890397 |
| `max_drawdown(...)` | `.max_drawdown` | -0.110216 |
| `calmar_ratio(..., annualization=252)` | `.calmar_ratio` | 0.723394 |
| `omega_ratio(...)` | `.omega_ratio` | 1.098490 |
| `tail_ratio(...)` | `.tail_ratio` | 1.218861 |
| `value_at_risk(..., cutoff=0.05)` | `.var_95` | -0.013656 |
| `conditional_value_at_risk(..., cutoff=0.05)` | `.cvar_95` | -0.017801 |

</div>

Both drawdown and the observed 5% VaR/CVaR are **negative return values**, not
positive loss magnitudes. Empyrical uses a left-tail `cutoff=0.05` while
Diagnostic labels the same selection as 95% confidence. Check the sign before
feeding either result into a risk limit.

## Preserve cadence, risk-free units, and missing-value behavior

Empyrical defaults to `period="daily"` (252 periods per year). Diagnostic
defaults to `periods_per_year=252`. For monthly returns, pass
`period="monthly"` to Empyrical and `periods_per_year=12` to Diagnostic.
The script's separate twelve-month input produced annual return `0.037051`
in both. Neither API can infer the intended annualization from an unlabelled
NumPy return array.

Both Sharpe defaults use zero risk-free rate. For nonzero rates, Empyrical's
`risk_free` argument is a **per-period return**; Diagnostic's `risk_free`
argument is an **annual rate**. The script converted a 2% annual rate to
`(1 + 0.02) ** (1 / 252) - 1` before calling Empyrical and passed `0.02`
to Diagnostic. Both produced Sharpe `0.453682`. Passing `0.02` directly to
both functions would not compare the same assumption.

With one NaN at return index 10, nine of the ten listed metric pairs still
matched on this input. Empyrical `value_at_risk` returned `nan` because it
uses a percentile that includes NaN; Diagnostic `.var_95` returned
`-0.013664` using a NaN-aware percentile. Both annual-return values were
`0.071988`; the input still had 252 periods and 251 valid returns. Choose a
missing-value policy before migration rather than relying on this one example.

Those NaN values describe the pinned published 0.1.8 workflow above. Current
Diagnostic 0.2.0 rejects missing/nonfinite portfolio returns at the
constructor boundary. The same comparison script reports that rejection and
still evaluates Empyrical's NaN behavior; it does not silently fill or drop a
return to make the inputs agree. The published comparison table remains a
historical measurement, not the current source's missing-data contract.

## Use columnwise scalar adapters (0.2.0) {#use-columnwise-scalar-adapters-unreleased}

The source after 0.1.9 adds `ml4t.diagnostic.compat.empyrical` for annual
return/CAGR, sample annual volatility, one-series Calmar, periodic-rate Sharpe,
Sortino and downside risk. PyPI 0.1.9 does not contain these adapters. Native
Diagnostic APIs keep their existing annual-rate defaults and sample guards.
This namespace does not redirect `import empyrical` or provide every legacy
function or metric registry.

This standalone workflow needs Diagnostic, not Empyrical, Backtest or Engineer:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.compat import empyrical as metrics

prices = pd.DataFrame(
    {"complete": [100, 102, 101, 103, 104], "missing": [50, 51, 52, 51, np.nan]},
    index=pd.bdate_range("2025-01-02", periods=5, tz="UTC"),
)
returns = prices.pct_change(fill_method=None).iloc[1:]
annual = metrics.annual_return(returns, annualization=12)
np.testing.assert_allclose(annual, [1.04**3 - 1, 1.02**3 - 1])
assert list(annual.index) == ["complete", "missing"]

out = np.empty(2)
sortino = metrics.sortino_ratio(
    returns, required_return=0.001, annualization=12, out=out,
)
assert list(sortino.index) == ["complete", "missing"]
assert np.shares_memory(sortino, out)
calmar = returns.apply(lambda column: metrics.calmar_ratio(column, annualization=12))
assert calmar.shape == (2,)
print("Empyrical scalar adapters: columnwise counts, periodic rates and labels checked")
```

Build a fresh unpublished wheel with `uv build --out-dir dist/empyrical-adapters`.
Run all seven routes and print the actual package versions from that wheel:

```bash
uv run --no-project --with dist/empyrical-adapters/ml4t_diagnostic-0.1.9-py3-none-any.whl \
  python -I examples/legacy_migration_comparison.py --task empyrical-native
```

The four return rows include one missing observation. Annual return treats
that observation as neutral compounding but still counts four elapsed rows.
Volatility uses sample deviation (`ddof=1`); means and downside deviation omit
missing adjusted observations separately in each column. Infinite returns are
not silently removed. A total loss remains -100%; a negative ending wealth
raised to a fractional annualization power is undefined (`NaN`), and a
floating overflow is `Inf`, not a complex value or Python overflow exception.

`risk_free` and `required_return` here are periodic returns. Explicit positive
finite `annualization` takes precedence over `period`, including noninteger
scales such as 52.5; no cadence or calendar is inferred from dates. For a
Series, a supplied rate Series aligns by date. For a DataFrame, a rate Series
must name its columns and aligns by column label, preserving the input column
order. Use a matching rate DataFrame for date-varying per-column rates rather
than passing a dated Series as if it were a column rate.

With enough input rows, annual return/CAGR, Sortino and downside risk return a
labeled Series for a DataFrame. Annual return/CAGR and downside risk accept
one input row; Sortino needs at least two raw rows, even if only one has a
defined adjusted return. Insufficient frame samples return an unlabeled
undefined array. Volatility and Sharpe return a column-ordered NumPy array.
Calmar is a one-series route; use `DataFrame.apply` as above. Empty annual
return/CAGR inputs return scalar `NaN`; the other vectorized routes preserve
the empty frame's column shape with undefined values. Supplied output buffers
must have the exact reduced shape, floating dtype, be writable and not overlap
the input. One-series calls return a scalar; columnwise array calls return the
buffer itself, and labeled calls share its storage. Sortino's optional
`_downside_risk` is an already annualized deviation.
Inputs are normalized to float64 for calculation; an explicitly supplied
floating output buffer retains its dtype and corresponding output precision.

Exactly identical finite excess returns have zero sample dispersion. With at
least two observations, Diagnostic returns signed infinity for nonzero mean
excess and `NaN` for zero mean excess. For example, ten or 252 copies of
`0.001` produce positive infinity, including through `excess_sharpe` and the
Empyrical rolling adapters. Published Empyrical 0.7.0 can instead produce a
huge finite value from reduction rounding. Diagnostic centers observations
before calculating dispersion; it does not use a tolerance that discards real
one-ULP variation. Native portfolio rolling metrics retain their distinct
undefined-flat-window convention, and the Numba kernel retains zero for an
exactly zero-return window. Pyfolio rolling reports still require complete
windows; Empyrical rolling adapters omit missing observations within a window.
The same sample-deviation kernel gives exactly zero volatility for finite
constant returns. Native information ratio remains undefined at zero tracking
error. Constant bootstrap resamples do not enter finite Sharpe confidence bounds.
Annual return, volatility and downside deviation use decimal return units;
Sharpe, Sortino and Calmar ratios are dimensionless.

Published Empyrical Reloaded 0.5.12 loses Sortino DataFrame column names. The
maintained reference has [repaired those labels](https://github.com/stefan-jansen/empyrical-reloaded/blob/a08e6fd123b74859c5e7e6747252ea9d2148cf67/src/empyrical/stats.py),
and this adapter retains the input names instead of copying the published
label defect. Numerical reference comparisons and independent price-product,
sample-deviation and downside formulas are separate checks. Matching these
scalar cases does not qualify the report, rolling or attribution workflows
listed below.

## Follow dated price and return paths (0.2.0) {#follow-dated-price-and-return-paths-unreleased}

The same namespace also exposes `simple_returns`, `cum_returns`,
`cum_returns_final`, `drawdown_series` and `max_drawdown`. These adapters use
Diagnostic's existing NaN-neutral compounding and initial-capital drawdown
calculations. They do not annualize a return path or infer a calendar. The
`--task empyrical-native` command above executes both the scalar workflow and
this price-to-wealth workflow without a legacy runtime dependency:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.compat import empyrical as metrics

prices = pd.DataFrame(
    {"strategy": [100.0, 101.0, 98.0, np.nan, 103.0], "benchmark": [50, 50, 52, 51, 53]},
    index=pd.date_range("2025-01-02", periods=5, tz="UTC", name="date"),
)
returns = pd.DataFrame(
    metrics.simple_returns(prices.to_numpy()),
    index=prices.index[1:], columns=prices.columns,
)
wealth = metrics.cum_returns(returns, starting_value=100)
np.testing.assert_allclose(wealth, [[101, 100], [98, 104], [98, 102], [98, 106]])
np.testing.assert_allclose(metrics.cum_returns_final(returns), [-0.02, 0.06], atol=1e-15)
drawdown = metrics.drawdown_series(returns)
np.testing.assert_allclose(metrics.max_drawdown(returns), [-3 / 101, -1 / 52], atol=1e-15)
assert drawdown.index.equals(returns.index)
assert list(wealth.columns) == ["strategy", "benchmark"]
print("Return paths: dated labels, missing adjacencies and initial capital checked")
```

The missing strategy price makes both adjacent ndarray returns undefined.
Compounding retains those rows as neutral growth, leaving wealth at 98;
it does not reconstruct the final price of 103. For `simple_returns`, pandas
inputs deliberately retain the installed pandas `pct_change()` default, as
Empyrical does. Pandas 2.3 pads that missing price, yielding final wealth 103;
pandas 3.0 propagates the gap, yielding 98. NumPy prices propagate missing
adjacencies under both versions. Use the ndarray route above, or calculate
`prices.pct_change(fill_method=None).iloc[1:]` explicitly, when the workflow
requires version-independent gap propagation. Integer ndarray differences are
computed before float conversion so small ticks above the float64 exact-integer
range and unsigned declines survive.

`cum_returns(..., starting_value=0)` reports cumulative decimal gains; a
nonzero starting value reports wealth in the units of that starting capital.
`cum_returns_final` returns the final gain or wealth, without annualization.
`drawdown_series` and `max_drawdown` are negative decimal declines from the
running peak, including the starting capital. All paths compound per column.
An ndarray drawdown has one row per input return, matching repaired Empyrical
0.7.0 rather than the extra anchor returned by 0.5.12.

Nonempty Series paths retain dates but omit the Series name, as the reference
does; DataFrame paths retain dates and columns. Final and maximum reductions
return a scalar for 1D inputs, a column-labeled Series for a DataFrame, or an
array for 2D ndarray input. With an explicit buffer, cumulative paths and
maximum reductions return the array rather than rebuilding labels; drawdown
paths rebuild pandas labels and share the buffer after its initial zero anchor.
That drawdown buffer must have one extra row. Buffer validation rejects wrong
shapes, nonfloating or read-only storage, and overlap with the original input.
Calculations use float64 and native unit-wealth drawdowns, so low-order rounding
can differ from reference float32 paths or its 100-unit drawdown calculation.

Empty cumulative paths preserve the empty input; empty final reductions return
scalar `NaN`. Empty drawdown paths return scalar `NaN` for 1D input or an
unlabeled `(1, n_columns)` NaN array for 2D input. Empty maximum reductions
return scalar `NaN` or an unlabeled `(n_columns,)` NaN array. These per-function
contracts are not a blanket DataFrame-support claim for other Empyrical routes.

## Compare arbitrary supplied benchmarks (0.2.0) {#compare-arbitrary-supplied-benchmarks-unreleased}

The source after 0.1.9 adds `alpha_beta`, `alpha`, `beta`, their `_aligned`
forms, `excess_sharpe`, `up_alpha_beta`, `down_alpha_beta` and `batting_average`
to the same explicit compatibility namespace. These methods reuse native
regression, compounding and periodic Sharpe kernels, not an Empyrical runtime.
Run the complete price -> return -> external benchmark workflow with:

```bash
uv run python examples/legacy_migration_comparison.py --task empyrical-benchmark
```

The supplied price path has periodic intercept 0.0003 and beta 1.2. Familiar
annual alpha is `(1.0003)**252 - 1`; native arithmetic annual alpha remains
`0.0003 * 252 = 0.0756`. Beta and active Sharpe are dimensionless, not decimal
return values. `excess_sharpe` does not apply the native information ratio's
square-root annualization. A nonzero `risk_free` is periodic and adjusts both
returns and benchmark in the alpha residual; fitted beta uses the original paired
returns. The reference accepts but does not use beta's `risk_free` argument.

Ordinary functions align labeled Series; `_aligned` functions consume rows
positionally. Matching ndarray inputs also remain positional. Combined regression
returns an array ending in `[alpha, beta]`. For a strategy matrix, supply the
benchmark as an explicit column, `benchmark[:, None]`, and retain the input column
order. The combined output has shape `(n_columns, 2)`; beta returns column-ordered
arrays, while supported `alpha_aligned` DataFrame inputs retain their column
labels. Unaligned multi-column DataFrames are not a universal supported shape:
apply the one-series function per column or use the explicit aligned matrix route.

Each column has its own paired missingness mask. The familiar beta variance floor
is explicitly `1e-30`; native nonconstant-benchmark regression keeps its existing
default. A supplied `_beta` lets alpha evaluate known residuals without pretending
that a singular benchmark can estimate a slope. Output-buffer validation also
checks original benchmark/rate storage before alignment copies can conceal an
alias. Numeric nullable pandas inputs retain separate data and missingness
buffers; both are protected, along with periodic-rate and supplied-downside
inputs. Independent output buffers remain supported for these inputs. For other
extension storage whose physical buffers cannot be verified, a provided `out`
requires an explicit owned dense snapshot, for example
`returns.to_numpy(dtype=float, na_value=np.nan, copy=True)`. The adapter raises
before writing instead of treating a copied numerical view as proof of no alias.

Calculations use float64 and cast once when committing an output buffer.
This differs from the reference's intermediate float32 arithmetic. For strategy
prices `[100, 101, 99, 102, 103, 101]` and benchmark prices
`[100, 100.5, 100, 101, 101.5, 100.8]`, both supplied as float32 pandas Series,
the qualified Empyrical 0.7.0 / pandas 2.3.3 / NumPy 2.5.4 profile gives:

| Output buffer | Diagnostic annual alpha | Empyrical annual alpha |
|---|---:|---:|
| float64 | -0.480214204627 | -0.480214100867 |
| float32 | -0.480214208364 | -0.480215013027 |

The benchmark recipe includes this price-produced case. An exact-rational OLS
oracle on the represented input returns supports Diagnostic's result; the final
float32 buffer receives that result rounded once. Beta differs by about
`2.35e-7` in this fixture. These measured precision differences are not hidden
by widening comparison tolerances or claiming bitwise parity.

Native `portfolio_analysis.benchmark_batting_average` returns fractions and
sample/win counts for finite active-return/benchmark pairs. Ties are nonwins and
empty subsets are undefined. Its up market is strictly positive by default;
`up_includes_zero=True` selects the legitimate Empyrical convention that includes
zero benchmark returns. The familiar facade returns the original three labeled
fractions and selects that inclusive convention. This is benchmark-relative
outperformance, not factor-monotonicity or a strategy's positive-return fraction.

The same example exercises `capture`, `up_capture`, `down_capture` and
`up_down_capture`. These divide annualized geometric return gains; they are
dimensionless ratios. The existing native `up_down_capture` continues to return
a tuple of terminal-wealth ratios. It is not the familiar scalar up/down ratio.
Full-stream capture compounds each supplied stream independently, counting its
missing rows as elapsed observations. Sign-subset one-series calls first select
present pairs using the benchmark's strict sign. Zero benchmark returns are
excluded from both subsets, and empty subsets are undefined.

Array matrix shapes remain function-specific. `capture` returns column ratios;
for up/down capture, a one-dimensional benchmark selects whole rows, while a
same-shaped benchmark matrix selects cells and pools them, as in the actual
reference. Use explicit per-column calls for independent sign-subset strategy
results. No blanket DataFrame or broadcast claim follows from these cases.

These benchmark workflows are not a full-replacement claim.

## Select metric definitions explicitly (0.2.0) {#select-metric-definitions-explicitly-unreleased}

The same source compatibility namespace also provides `omega_ratio`,
`tail_ratio`, `stability_of_timeseries`, `value_at_risk` and
`conditional_value_at_risk`. These adapters select explicit options in the
native kernels. They do not require Empyrical at runtime or change native
portfolio-summary defaults. Run their standalone examples with:

```bash
uv run python examples/legacy_migration_comparison.py --task empyrical-definitions
```

| Statistic | Familiar definition | Existing native default |
|---|---|---|
| Omega | Periodic risk-free return, annual hurdle converted to the input period; no-loss result is NaN | Periodic `threshold`; positive gains without losses give infinity |
| Tail ratio | Magnitudes of both 95th and 5th percentiles; zero denominator can give infinity | Signed upper percentile divided by absolute lower percentile |
| Stability | Fit cumulative log returns on nonmissing observations | Fit compounded wealth, retaining missing rows as unchanged wealth |
| Empirical VaR | Lower-return quantile with `cutoff`; any NaN makes the result undefined | `confidence=1-cutoff`; omit NaNs |
| CVaR | Omit NaNs, then average the lowest `int((n-1)*cutoff)+1` observations | Include every observation at or below the interpolated VaR threshold |

For returns `[-0.2, -0.1, NaN, -0.1, 0.1]` at `cutoff=0.5`, fixed-count CVaR
is `-0.15`; threshold-tie CVaR is `-2/15`. Missing rows are excluded before
choosing the fixed tail count, following corrected Empyrical 0.7.0 rather than
the older published behavior. The original cutoff is retained directly when
choosing the count, avoiding a floating-point confidence-complement round trip.

VaR and CVaR here are decimal returns, not positive cash losses. Tail ratio,
stability and Omega are dimensionless. The qualifying reference workflow uses
Empyrical 0.7.0 with pandas 2.3.3. These scalar tail/stability functions pool
the supported array shapes rather than inventing per-column results. Use an
explicit per-column call for independent strategy statistics; support in the
earlier columnwise metric section does not generalize to these functions.

The native metric module exposes `omega_ratio(..., no_loss="nan")`,
`tail_ratio(..., absolute=True)`, `stability_of_timeseries(..., method="log")`,
`value_at_risk(..., nan_policy="propagate")` and
`conditional_var(..., method="fixed_count")` when selecting these definitions
without the familiar signatures. Gaussian return VaR and covariance-based cash
VaR remain separate workflows, not aliases for the empirical quantile.

The native portfolio-analysis module provides `gaussian_value_at_risk(returns,
sigma=2)` for Pyfolio-style mean-minus-sample-deviations return VaR, and
`normal_cash_value_at_risk(portfolio_value, confidence, mean=0, std=1)` for
covariance-model cash risk. Both are exercised by the same definition recipe.
For `[-0.02, 0, 0.02]`, Gaussian two-deviation return VaR is `-0.04`, whereas
the native empirical 95% return quantile is `-0.018`. For capital 1000 and
a modeled median return of `-0.01`, normal cash VaR at confidence 0.5 is a
loss of 10 cash units. Positive modeled lower returns can produce a negative
cash loss; the function does not clip gains to zero.

Pyfolio's `timeseries.value_at_risk(..., period=...)` first aggregates returns.
Preserve that choice explicitly with `aggregate_returns`, then pass the result
to `gaussian_value_at_risk`. The actual Pyfolio 0.10.0 / Empyrical 0.7.0 profile
qualifies weekly, monthly and yearly grouping as well as the unaggregated sigma
options. `var_cov_var_normal(P, c, mu, sigma)` maps to
`normal_cash_value_at_risk(P, c, mean=mu, std=sigma)` for a nondegenerate normal
model. These are separately named native routes; familiar Pyfolio report wiring
is tracked with its own workflow and is not implied by metric availability.

`aggregate_returns` compounds only observed rows into weekly, monthly, quarterly
or yearly groups. Weekly keys use ISO year together with ISO week, so the week
starting 2019-12-30 is `(2020, 1)`, including its January observations. Other
groups use the supplied civil date labels and timezone. The adapter does not
convert to UTC, resolve a trading calendar, fill absent holidays/months or infer
dates from an integer index. All-missing observed groups have unchanged wealth
(zero cumulative return), consistent with the return-path adapter. Genuine
dated pandas input is required; strategy columns and their labels are retained.
The same `--task empyrical-definitions` recipe checks the ISO-year boundary.

## Continue row-window research (0.2.0) {#continue-row-window-research-unreleased}

The familiar facade adds row-window drawdown, CAGR, volatility, Sharpe, Sortino,
downside risk, active Sharpe, alpha/beta and up/down capture functions. These use
the same native-backed metric kernels and a shared read-only row-window layout;
the existing optimized native rolling calculations retain their defaults.
The historical misspelling `roll_downsize_risk` remains a warning-emitting alias
of `roll_downside_risk`. Generic `roll`, `up`, `down` callback helpers and
`rolling_window` views support direct migration recipes.

```bash
uv run python examples/legacy_migration_comparison.py --task empyrical-rolling
```

| Workflow | Endpoint/sample contract |
|---|---|
| Vectorized `roll_*` risk/regression metrics | Trim warmup; when `window > n`, evaluate one short sample if input is nonempty |
| `roll_alpha_beta` | Align complete labeled streams before forming positional row windows |
| Other vectorized paired wrappers | Consume rows positionally, including labeled input; dates label output endpoints |
| `roll_up_capture`, `roll_down_capture`, `roll_up_down_capture`, generic `roll` | Evaluate full windows only; overlong windows have no output |
| Native `PortfolioAnalysis.compute_rolling_metrics` | Retain the input length and undefined warmup; overlong windows remain undefined |

Irregular or duplicate legacy endpoint labels do not make row windows into
calendar windows. Native portfolio inputs keep their existing unique-date
validation. Use a positive integer window; invalid windows and overlapping
output buffers are rejected without modifying caller data. A row-window view is
read-only by default; `mutable=True` explicitly allows shared-storage edits.
For independent strategies, apply these one-series rolling adapters per column;
no general multi-asset matrix rolling parity is claimed.

For a wide return frame, preserve strategy labels by collecting the one-series
results. The rolling recipe executes this path, and the reference tests check
each rolling function on price-produced columns with different missingness:

```python
import pandas as pd
from ml4t.diagnostic.compat import empyrical as ep

rolling_sharpe = pd.concat(
    {name: ep.roll_sharpe_ratio(values, window=63)
     for name, values in returns.items()},
    axis=1,
)
```

Paired wrappers also receive the same supplied benchmark per strategy. Combined
alpha/beta results retain a nested strategy/result column index. No asset is
pooled with another strategy merely to accept a matrix argument.

`roll_cagr` annualizes each exact compounded window product by its observed row
count. It is not a renamed native rolling cumulative return. In the tested
Empyrical 0.7.0 profile, its generated wrapper passes an unsupported `out`
keyword to `cagr` and raises `TypeError` for nonempty input. Diagnostic's route
is checked against independent window-product expectations instead of copying
that reference defect. Direct reference comparisons for the broken wrapper are
reported separately from passing comparisons for other rolling functions.

`SIMPLE_STAT_FUNCS` runs the complete simple-statistics registry, including the
native-backed definitions above and SciPy skew/kurtosis. `FACTOR_STAT_FUNCS`
preserves the complete Empyrical order: excess Sharpe, alpha, beta, beta
fragility, GPD risk estimates, capture, up capture and down capture. The registry
does not give these functions a uniform signature. GPD receives a univariate
return series and `var_p`, and returns five fields; the other entries receive
strategy and benchmark returns and return scalar statistics.

```python
registry_dates = pd.bdate_range("2025-01-02", periods=10, tz="UTC")
registry_returns = pd.Series(
    [0.01, -0.02, 0.03, -0.04, 0.02, -0.01, -0.03, 0.015, 0.005, -0.025],
    index=registry_dates,
)
registry_benchmark = pd.Series(
    [0.02, -0.01, 0.025, -0.03, 0.01, -0.02, -0.025, 0.01, 0.005, -0.015],
    index=registry_dates,
)
factor_statistics = {
    function.__name__: (
        function(registry_returns, var_p=0.01)
        if function.__name__ == "gpd_risk_estimates"
        else function(registry_returns, registry_benchmark)
    )
    for function in ep.FACTOR_STAT_FUNCS
}
```

Use one strategy series at a time. These statistics have different units:
alpha is an annual compounded return by default, beta/capture/excess Sharpe
are dimensionless, fragility is a periodic return, and GPD contains a
threshold, scale, dimensionless shape, loss VaR and expected shortfall.
The runnable `--task empyrical-tail` recipe also executes this complete
registry without requiring Empyrical at runtime.

## Attribute returns to supplied holdings and loadings (0.2.0) {#attribute-returns-to-supplied-holdings-and-loadings-unreleased}

Use this workflow when you already have asset NAV weights and factor loadings.
It does not estimate betas from portfolio returns. The same calculation serves
the native result and the familiar `perf_attrib` and `compute_exposures` adapters.
Neither route requires Data, Engineer, Backtest or a legacy package at runtime.

From this source revision, run the independently checked two-period example:

```bash
uv run --extra viz python examples/holdings_factor_attribution.py --report holdings-attribution.html
```

To compare the familiar public tables with Empyrical Reloaded 0.7.0, use its
qualified pandas 2.3.3 profile in an isolated environment:

```bash
uv run --no-project --isolated --python 3.12 --with '.[viz]' --with empyrical-reloaded==0.7.0 --with pandas==2.3.3 python examples/holdings_factor_attribution.py --compare --report holdings-attribution.html
```

The example prints actual package versions and checks exposures `[1.5, 0.25]`
and `[1, 0.5]`, common periodic returns `0.02` and `-0.015`, and specific
returns `0.01` and `0.005`. Wealth-linked common and specific contributions
finish at `0.00455` and `0.01515`; their sum is the portfolio return `0.0197`.
The report displays those stored contributions without annualizing or
compounding each component independently.

For an existing Empyrical workflow, change the import and retain the supplied
pandas inputs. This executable two-date book uses the same weights and loadings:

```python
import numpy as np
import pandas as pd
from ml4t.diagnostic.compat import empyrical as ep

attribution_dates = pd.DatetimeIndex(["2025-07-07T20:00Z", "2025-07-08T20:00Z"])
attribution_keys = pd.MultiIndex.from_product(
    [attribution_dates, ["A", "B"]], names=["dt", "ticker"]
)
attribution_returns = pd.Series([0.03, -0.01], index=attribution_dates)
positions = pd.Series([0.5, 0.5, 0.25, 0.75], index=attribution_keys)
factor_loadings = pd.DataFrame(
    {"market": [2.0, 1.0, 1.0, 1.0], "style": [0.0, 0.5, 2.0, 0.0]},
    index=attribution_keys,
)
factor_returns = pd.DataFrame(
    {"market": [0.01, -0.02], "style": [0.02, 0.01]}, index=attribution_dates
)
exposures, contributions = ep.perf_attrib(
    attribution_returns, positions, factor_returns, factor_loadings
)
np.testing.assert_allclose(exposures, [[1.5, 0.25], [1.0, 0.5]])
np.testing.assert_allclose(contributions["common_returns"], [0.02, -0.015])
```

Here `positions` is a fractional NAV-weight Series with a date/asset
MultiIndex, not dollar holdings. `factor_loadings` has that same kind of index
and one column per factor. `returns` and `factor_returns` have datetime
indexes. Original factor labels, including numeric labels or names such as
`asset`, survive conversion to the native table representation. The adapters
preserve the tested reference's observed-label alignment and skip-missing
sums, with warnings for uncovered loadings or missing weights. They do not
assert that the information was available before trading.

Asset matching uses one joint pandas identity index before native transport;
numeric asset `1` and text asset `"1"` remain distinct even when loading rows
arrive in a different asset order. Missing or empty asset labels are rejected.
The familiar date-sliced workflow requires chronological outer date groups in
the loading MultiIndex, as Empyrical/Pyfolio do; within-date asset order need
not match the holdings rows. Native frame inputs retain their declared asset
ID types rather than adopting these internal transport codes.

For explicit availability and coverage checks, use
`evaluation.factor.compute_holdings_attribution` or
`PortfolioAnalysis.compute_holdings_attribution`. Supply dated returns with
explicit `period_start`, fractional weights and loadings with `known_at`,
and same-period factor returns. Declare `complete_snapshots=True` only for
complete account observations. A genuine all-cash observation may have no
asset rows; pass its actual timeline through `observed_snapshot_dates` rather
than inventing asset rows or treating a missing observation as cash. The class
method accepts `period_starts`, or uses its actual `initial_timestamp` and
preceding return observations. It does not infer the first interval from a
calendar frequency.

Native defaults reject future information and uncovered active loadings.
`missing_loadings="legacy_skip"` and `timing="legacy_observed"` are explicitly
different choices. Native cumulative contributions use beginning wealth;
`cumulative_method="arithmetic"` selects a named sum of periodic contributions.
Full-sample mean exposure defines tilt, not a forecast. Specific returns are
unexplained returns, not evidence of skill. Without explicitly supplied cash
contributions, cash remains in the specific residual.

`result.to_dataframe()` returns an owned Polars contribution table with its
temporal dtype. `result.to_dict()` exports exact ISO observation keys, including
nanoseconds and timezone offsets. Explicit legacy Date labels remain dates;
the export does not turn them into invented midnight instants. Modifying either
export does not change the stored analytical result.

Pass the native result as `metrics={"holdings_attribution": result}` to the
existing [backtest report](backtest-tearsheets.md) function, builder or workspace
export. It produces separate supplied-exposure, attribution, and loading-coverage
sections. The report rejects a different return path or a serialized dictionary
in place of the typed result before changing output. This supplied-holdings
workflow remains distinct from the existing fitted factor regression. For the
complete familiar performance-attribution sheet, use
`compat.pyfolio.create_perf_attrib_tear_sheet` with supplied holdings and factors;
the native and familiar routes retain their separately documented timing and
missing-loading policies.

The same native calculation also backs
`ml4t.diagnostic.compat.pyfolio.perf_attrib.perf_attrib` and `compute_exposures`.
These accept the familiar wide dollar/fractional positions with an explicit
account `cash` column. `create_perf_attrib_stats` retains the reference's
independently compounded component statistics, with explicit
`periods_per_year=252` by default. These component growth paths do not add up to
the wealth-linked decomposition. `show_perf_attrib_stats` prints and returns
the tables without requiring IPython; it does not produce Pyfolio's notebook
display object. Use `--compare-pyfolio` in the example with optional Pyfolio
Reloaded 0.10.0, Empyrical 0.7.0 and pandas 2.3.3 to run both public workflows.

```bash
uv run --no-project --isolated --python 3.12 --with '.[viz]' --with pyfolio-reloaded==0.10.0 --with empyrical-reloaded==0.7.0 --with pandas==2.3.3 python examples/holdings_factor_attribution.py --compare --compare-pyfolio --report holdings-attribution.html
```

One numerical correction is intentional. Pyfolio 0.10.0 drops an asset without
loadings before normalizing dollar holdings. On the tested full-NAV-1000 book,
covered asset A has fractions `0.5, 0.25, 0.5` and market loadings `2, 1, 2`.
The correct covered exposure is `1, 0.25, 1`. The reference returns `2, 1, 2`
after removing B from its NAV denominator. Diagnostic retains the full account
denominator and warns about the uncovered loadings; dollar and equivalent
fractional inputs therefore agree. Unknown B exposure is still unknown, not
proved zero. Missing whole loading dates retain the familiar warned sample
exclusion. A high-turnover warning does not repair missing intraday exposures.
The same module provides `plot_returns`, `plot_alpha_returns`,
`plot_factor_contribution_to_perf` and `plot_risk_exposures`. These return Plotly
figures; remove a supplied Matplotlib `ax` and use `write_html` or `show`.
Explicit periodic costs subtract from gross total returns once, before
compounding; their separately plotted path is negative compounded cost.
The factor plot retains `specific_returns` alongside factors, as the reference
does, and omits total/common/tilt/timing aggregates. Specific-return histograms
use ten explicit equal-width bins and count nonfinite exclusions.

Run the complete familiar attribution report on the same supplied inputs:

```bash
uv run --extra viz python examples/holdings_factor_attribution.py --pyfolio-report pyfolio-attribution.html
```

This writes an offline full sheet and `pyfolio-attribution-specific-returns.html`.
`ml4t.diagnostic.compat.pyfolio.tears.create_perf_attrib_tear_sheet` includes the
summary/exposure tables, total/specific/common growth and factor-partition
contribution/exposure panels. It preserves `transactions`, `pos_in_dollars`
and `factor_partitions` input choices. Defaults retain Pyfolio's canonical
style/sector members; pass `factor_partitions=None` to show every supplied
factor. A custom mapping selects members while preserving input-column order.
An empty mapping produces only the returns panel; it does not reserve blank
axes. A named group with no matching supplied factor is visibly unavailable.

Presentation differs explicitly. By default the function returns HTML, or the
saved path when `output_file` is given, without triggering IPython display.
`return_fig=True` returns the ordered list of Plotly figures rather than one
Matplotlib figure. HTML output always contains the tables. `periods_per_year`
sets table annualization explicitly; it does not annualize already compounded
paths. `theme` uses Diagnostic's existing report themes.

Actual Pyfolio 0.10.0 with Matplotlib 3.11.2 plots the histogram but then raises
`AttributeError` while sorting rectangle legend handles. The factor/exposure
plots draw their paths but raise `ValueError` while constructing an empty
legend. Diagnostic compares those unmodified artist values and independent
bin/path expectations; it does not reproduce the rendering exceptions or
claim those reference calls completed. Complete attribution-report browser,
artifact and platform qualification is recorded separately from table matches.

## Inspect a fitted generalized-Pareto tail (0.2.0) {#inspect-a-fitted-generalized-pareto-tail-unreleased}

`metrics.estimate_gpd_tail_risk(returns, tail_probability=0.01, threshold=None)`
returns `GPDTailRiskResult`, including fit status, threshold/scale/shape,
positive decimal-loss VaR and expected shortfall, finite/excluded observation
counts, loss/exceedance counts and the actual probability denominator.

```bash
uv run python examples/legacy_migration_comparison.py --task empyrical-tail
```

With the optional visualization dependencies, add `--output gpd-tail-risk.html`
to generate the existing portfolio report with the supplied fit beneath its
return-distribution chart. Function, builder and performance-workspace export
routes accept `metrics={"gpd_tail_risk": result}` when `result` is the native
`GPDTailRiskResult` computed from the same returns. The report displays the fit
without refitting it, including sample counts, boundary fits, undefined risk and
positive infinite ES. These loss values are per observation and are not scaled
by the report's annualization setting. A serialized dictionary or the familiar
five-field array is not a substitute for the native result object.

The native `peaks_over_threshold` method fits strictly positive excesses over
the chosen threshold. When omitted, the threshold is the nonnegative 90th
percentile of all finite losses, where loss is `-return`. The exceedance
probability uses all finite observations, including gains. Appending gains
therefore changes unconditional risk without changing the selected tail fit.
There is no calendar inference or annualization.

The native likelihood fit constrains shape to at least -1. Below that bound,
the density at the sample maximum can grow without limit as the distribution
endpoint approaches that maximum. The result reports the bound and whether
the estimate is on it. A boundary fit and a small exceedance count are not
evidence of estimator precision or model adequacy.

Choose `method="empyrical_raw_tail"` to inspect the familiar method. It fits
raw positive losses, not excesses, at thresholds starting at 0.2 and halving
toward 1e-9; probabilities use the loss-only sample. Its positive-shape
selection and optimizer initialization differ from the native method. These
rules require three supplied observations, not three losses or exceedances.
Small fitted samples remain visible in the result counts. The two
methods need not produce equal parameter or risk estimates. The familiar
`gpd_risk_estimates` and `_aligned` calls project the named method into the
five legacy fields; a supplied pandas Series retains the reference's positional
result labels, not the input dates. Use one portfolio per call.

The likelihood and optimizer helpers also accept a single-column frame or
array. They reject multiple portfolio columns rather than combine them into one
tail fit. Apply the same helper independently to each strategy. For example,
`{name: ep.gpd_loglikelihood_minimizer_aligned(-values[values < 0])
for name, values in returns.items()}` fits the positive raw losses per strategy;
it does not estimate a joint portfolio loss distribution. The top-level
`gpd_risk_estimates` continues to require a one-dimensional return series.

Both methods exclude and count nonfinite observations. Insufficient or
degenerate tails, a failed fit, or a requested probability outside the fitted
tail produce undefined risk with a native status. Familiar unsuccessful
projections contain NaNs instead of the reference's misleading zero-risk
sentinel. Familiar likelihood helpers retain their nonnegative-shape model;
native fitting also permits bounded negative-shape tails. Invalid likelihood
domains return negative infinity rather than a finite near-overflow sentinel.
Distribution helpers support the exponential shape-zero limit;
nonpositive scales and invalid probability domains are rejected. Expected
shortfall is positive infinity for fitted shape at least one, as in the
corrected Empyrical 0.7.0 workflow, not zero or a negative loss.

These source APIs implement the named GPD methods and explicit failure statuses.
They do not describe the published 0.1.9 distribution. Select and compare the
tail definition before replacing an existing Empyrical calculation.

## Inspect benchmark fragility (0.2.0) {#inspect-benchmark-fragility-unreleased}

`ml4t.diagnostic.metrics.beta_fragility` returns an immutable
`BetaFragilityResult` with the periodic return gap, usable/excluded counts,
selected input keys/ordinals, actual strategy/benchmark points and endpoint
weights. This makes the sample and tie convention inspectable instead of
returning only a scalar. The familiar `beta_fragility_heuristic` and
`beta_fragility_heuristic_aligned` calls return the scalar from the same native
interpolation kernel.

```bash
uv run python examples/legacy_migration_comparison.py --task empyrical-fragility
```

The price-produced example has benchmark returns `[-0.02, 0, 0.03]` and strategy
returns `[-0.0004, 0, -0.0009]`. Endpoint weights at the actual middle benchmark
coordinate are `[0.6, 0.4]`, giving a gap of `-0.0006`. The middle observation is
the sorted row at `n//2`, not an averaged coordinate for an even sample. Fewer
than three usable pairs give an undefined value and explicit counts, without
inventing another observation. Negative gaps describe concavity of these
selected observations. They are not annualized, fitted betas, significance tests
or prediction-accuracy scores. Use the existing [statistical validation
workflows](statistical-tests.md) for the distinct inference task.

The native result uses finite pairs and preserves input order within benchmark
ties. A constant benchmark has an explicitly flagged half/half endpoint
convention; the result is not permutation-invariant within ties. The familiar
calls select the actual Empyrical 0.7.0 present-pair and pandas quicksort tie
conventions. A real zero-price recovery can therefore produce an infinite
familiar result while the native result excludes that nonfinite pair and
reports its exclusion. This difference is declared, not silently filled.

Despite its name, the tested reference's `_aligned` fragility function still
concatenates labeled Series by their labels. The adapter preserves that actual
behavior. Matching arrays remain positional; use one series per strategy rather
than assuming all functions accept a multi-column matrix.
The native route also accepts one-dimensional Polars Series and numeric
sequences. These inputs are positional, must have equal lengths and are
normalized to float64 without changing the caller's data.

One reference numerical error is corrected rather than copied. With benchmark
coordinates `[-1e308, 0, 1e308]` and strategy values `[-0.01, 0.02, 0.03]`, the
independent symmetric interpolation gap is `-0.01`. Published Empyrical 0.7.0
returns `-0.02` because the finite coordinate range overflows before division.
The shared Diagnostic kernel rescales only that overflowing finite range; both
native and familiar routes retain the correct half/half weights. Infinite
benchmark coordinates are not misclassified as a constant finite design.

## Native definitions still need explicit choices

The native benchmark-relative `alpha` and `beta` remain available when you
pass aligned benchmark returns. On the historical [Pyfolio comparison](migration.md#pyfolio-portfolio-and-performance-analysis),
beta matched at `0.795601`, while annualized alpha was `0.100369` in Pyfolio's
Empyrical calculation and `0.095663` in Diagnostic. Empyrical compounds the
mean periodic alpha; Diagnostic multiplies the fitted daily intercept by 252.
Choose the annualization definition before replacing alpha in a report.

The similarly named stability statistics are not equivalent. On the complete
input, Empyrical `stability_of_timeseries` returned `0.252783` and Diagnostic
`.stability` returned `0.247302`. Empyrical fits cumulative log returns;
Diagnostic fits cumulative simple wealth. Do not relabel one as the other.

The GPD and supplied-holdings attribution routes above have explicit native
and familiar methods. They do not define a general multi-asset matrix rolling
contract: apply a documented one-portfolio helper separately to each column.
Compare input alignment, tail definitions and attribution timing when migrating.
These APIs are part of Diagnostic 0.2.0. They are absent from the immutable
0.1.9 distribution.

## Replace path, UTC and warning utilities

The small Empyrical/Pyfolio utilities need no new Diagnostic service. Use
`pathlib` for directory creation and paths, pandas for the existing timestamp
conversion, and `warnings` in a caller-owned deprecated wrapper.

| Legacy helper | Replacement and retained behavior |
|---|---|
| `cache_dir()` | Honor `EMPYRICAL_CACHE_DIR`, otherwise `XDG_CACHE_HOME/empyrical`, otherwise `~/.cache/empyrical` if retaining the old cache location. A new application may choose its own explicit path. |
| `data_path(name)` | Join the chosen cache root with `name` using `Path`. This chooses a path; it does not fetch or validate data. |
| `ensure_directory(path)` | `Path(path).mkdir(parents=True, exist_ok=True)`. An existing regular file remains an error. |
| `get_utc_timestamp(value)` | Localize a naive pandas timestamp/index to UTC; convert an aware value to UTC. Localizing a naive daily label is a declared convention, not a recovered exchange instant. |
| `deprecated(...)` | Emit `warnings.warn(message, DeprecationWarning, stacklevel=2)` from the caller-owned compatibility wrapper. Warning filters still control visibility. |

```python
import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd

cache_root = Path(os.environ.get(
    "EMPYRICAL_CACHE_DIR",
    str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "empyrical"),
))
cache_file = cache_root / "returns.csv"
with tempfile.TemporaryDirectory(prefix="migration-utils-") as scratch:
    directory = Path(scratch) / "nested" / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.txt").write_text("supplied report", encoding="utf-8")

naive_utc = pd.Timestamp("2025-01-02 12:00").tz_localize("UTC")
aware_utc = pd.Timestamp("2025-01-02 07:00", tz="America/New_York").tz_convert("UTC")
assert naive_utc == aware_utc
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always", DeprecationWarning)
    warnings.warn("Use the named replacement", DeprecationWarning, stacklevel=2)
    assert len(recorded) == 1 and recorded[0].category is DeprecationWarning
print("Utility migration: explicit paths, UTC convention and caller warnings")
```

These operations neither reinterpret a trading calendar nor annualize returns.
Acquisition and cache freshness remain the producer's responsibility.
