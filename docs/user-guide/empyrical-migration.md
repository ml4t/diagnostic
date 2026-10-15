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
