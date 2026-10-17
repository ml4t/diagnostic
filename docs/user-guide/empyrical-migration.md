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
unreleased source after 0.1.9 rejects missing/nonfinite portfolio returns at the
constructor boundary. The same comparison script reports that rejection and
still evaluates Empyrical's NaN behavior; it does not silently fill or drop a
return to make the inputs agree. The published comparison table remains a
historical measurement, not the current source's missing-data contract.

## Use columnwise scalar adapters (unreleased)

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
Annual return, volatility and downside deviation use decimal return units;
Sharpe, Sortino and Calmar ratios are dimensionless.

Published Empyrical Reloaded 0.5.12 loses Sortino DataFrame column names. The
maintained reference has [repaired those labels](https://github.com/stefan-jansen/empyrical-reloaded/blob/a08e6fd123b74859c5e7e6747252ea9d2148cf67/src/empyrical/stats.py),
and this adapter retains the input names instead of copying the published
label defect. Numerical reference comparisons and independent price-product,
sample-deviation and downside formulas are separate checks. Matching these
scalar cases does not qualify the report, rolling or attribution workflows
listed below.

## Metrics that need separate review

Benchmark-relative `alpha` and `beta` are available from Diagnostic when you
pass aligned benchmark returns. On the [Pyfolio comparison](migration.md#pyfolio-portfolio-and-performance-analysis),
beta matched at `0.795601`, while annualized alpha was `0.100369` in Pyfolio's
Empyrical calculation and `0.095663` in Diagnostic. Empyrical compounds the
mean periodic alpha; Diagnostic multiplies the fitted daily intercept by 252.
Choose the annualization definition before replacing alpha in a report.

The similarly named stability statistics are not equivalent. On the complete
input, Empyrical `stability_of_timeseries` returned `0.252783` and Diagnostic
`.stability` returned `0.247302`. Empyrical fits cumulative log returns;
Diagnostic fits cumulative simple wealth. Do not relabel one as the other.

This assessment does not cover Empyrical's `gpd_risk_estimates`,
`perf_attrib`, date aggregation, or the `roll_*` function family as drop-in
replacements. Diagnostic has other portfolio and factor analyses, but their
inputs and result definitions must be checked separately. Keep Empyrical for
those tasks until a specific migration is verified.
