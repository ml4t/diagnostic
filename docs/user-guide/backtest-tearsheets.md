# Backtest Tearsheets

Generate a standalone HTML report from normalized metrics, trades, and returns.
The minimal supported input is a metrics dictionary or a return series.
The example uses synthetic daily returns and does not require a backtest engine.
Install the `viz` extra for Plotly rendering. Use real returns at the frequency
assumed by the metrics and statistical corrections.

## Generate an HTML report

```python
from pathlib import Path

import numpy as np

from ml4t.diagnostic.visualization.backtest import generate_backtest_tearsheet

rng = np.random.default_rng(42)
daily_returns = rng.normal(loc=0.0005, scale=0.01, size=252)
metrics = {
    "n_trades": 80,
    "total_pnl": 12_500.0,
    "win_rate": 0.54,
    "profit_factor": 1.6,
    "sharpe_ratio": 1.3,
    "max_drawdown": -0.12,
}

output = Path("backtest_report.html")
html = generate_backtest_tearsheet(
    metrics=metrics,
    returns=daily_returns,
    template="quant_trader",
    theme="default",
    output_path=output,
    n_trials=25,
    periods_per_year=252,
)

assert output.exists()
assert "plotly" in html.lower()
print(f"Wrote {output}")
```

The file-existence and Plotly assertions check that rendering completed.
Inspect reported Sharpe, drawdown, and trial count against your source
backtest; HTML generation does not validate supplied metrics.

Set `periods_per_year` from the cadence of the return observations, not the
number of observations or the rebalance frequency. Use 12 for monthly returns,
252 for a declared daily trading-year convention, or 365 for daily continuous
returns. The value must be a positive integer. It is a numerical convention,
not evidence that an expected calendar/bar grid is complete.

The same override is available on `BacktestTearsheet`, `export_workspaces`,
`generate_tearsheet_from_result`, and `generate_tearsheet_from_run_artifacts`.
Without an override, profile reports inherit the profile's annualization scale;
raw-return reports use the documented 252-period daily convention at every
sample length. Computed metrics, benchmark comparisons, rolling plots, factor
analysis and statistical-validation sections use that scale. Supplied metrics
are retained and must already use the same convention. Reports record the scale
in their methodology section and the `ml4t-periods-per-year` HTML metadata.

In 0.1.9 and earlier, raw reports with fewer than 100 observations incorrectly
selected a 12-period scale. That published release remains unchanged; this
correction applies to subsequent source revisions. Portfolio input alignment,
calendar selection and full Pyfolio/Alphalens capability coverage are separate
requirements, not guarantees established by report generation.

### Keep calendar omission distinct from disabling aggregation

Unreleased after 0.1.9, the lazy Backtest bridge uses the shared calendar
contract: an explicit argument overrides declared producer feed metadata, then
the default is NYSE. Omitting `calendar` inherits that resolution. Explicit
`calendar=None` instead keeps the producer's observation-level returns and exact
timestamp keys, without daily/session aggregation. It does not mean "inherit the
engine calendar". No return is reconstructed from NAV by this choice.

For an existing `result = engine.run()`, choose the return scale explicitly:

```pycon
>>> from ml4t.diagnostic.integration import analyze_backtest_result
>>> observed = analyze_backtest_result(result, calendar=None, periods_per_year=252)
>>> observed.resolved_calendar is None
True
>>> observed.daily_returns.equals(result.to_returns_series().alias("daily_return"))
True
```

Here 252 declares a daily trading-year convention for already-daily observations;
use the scale appropriate to the actual return cadence. Observation-mode metrics
and reports reject a missing scale. The same `periods_per_year` override reaches
`compute_metrics_from_result`, `portfolio_analysis_from_result`, artifact profiles
and HTML reports. Passing a descriptive report calendar that conflicts with the
resolved analysis calendar is rejected before writing an output file.

The runnable `examples/legacy_migration_comparison.py --task exposure` exercises
this bridge on actual engine output. The captured calendar regression uses a
Sunday-night CME run: omission retains the declared CME calendar while explicit
None retains all three observed bars. Neither a selected scale nor three received
bars establishes a complete exchange grid. The optional producer remains optional
for ordinary supplied-return analysis.

## Choose a template

Profile attribution bars and contributor rankings use signed cash P&L. The
table's "Share of net P&L" is a separate ratio: a losing total can give a
losing asset a positive share, so its sign does not determine profit/loss
colors or rankings. Undefined shares display `N/A`; defined zero cash P&L
remains zero. Standalone share-only contributor inputs use neutral colors
and report unavailable values explicitly.
On narrow screens, the attribution panel scrolls horizontally to keep all table
labels and values readable.

| Template | Primary content |
|----------|-----------------|
| `quant_trader` | Trades, performance, and model diagnostics |
| `hedge_fund` | Performance, costs, and reporting context |
| `risk_manager` | Risk and statistical validation |
| `full` | Every available section |

Use `BacktestProfile` when you have normalized trades, returns, positions,
costs, predictions, and factor results. Use
`generate_tearsheet_from_result` for an `ml4t-backtest` result. The integration
normalizes the backtest object before rendering.

In unreleased source after 0.1.9, `include_statistical=False` on
`generate_tearsheet_from_result` or `generate_backtest_tearsheet` omits
inferential validation calculations, credibility/validity panels, confidence
intervals, minimum track record, Sharpe bootstrap and statistical summary KPIs.
Ordinary performance, trading and descriptive drawdown analysis remain.
The default is `True`. Previously the bridge silently ignored `False`.

For contract fills, provide the declared point-value multiplier. The Backtest
bridge uses a fill's explicit `multiplier` when present, otherwise the producer's
trade multiplier for that asset. Conflicting trade declarations or missing,
nonfinite or nonpositive declared multipliers raise an error. Direct
`compute_activity_metrics` callers can supply the corresponding `trades_df`.
Legacy inputs with no multiplier declaration retain cash-per-unit prices; they
do not establish support for contract prices quoted in points.

Filled notional is absolute quantity times execution price times multiplier.
Slippage quoted in price points uses the same multiplier; commission is already
cash and is not multiplied again. In the tested one-contract example, price
100 and multiplier 50 produce $5,000 notional and 0.50 turnover against $10,000
end-observation NAV. With 0.5 points of slippage and $2 commission, implementation
cost is $27, not $2.50. Activity and symbol attribution use the same units and
leave producer tables unchanged.

These activity ratios use the NAV at the matched observed timestamp. They do
not establish exchange-session aggregation, Pyfolio's average-gross-book
denominator, or beginning-session-NAV return scenarios. Those workflows require
separate input and calendar contracts.

Trade profit factor divides gross positive lifecycle cash P&L by gross loss
magnitude. Empty or all-flat samples have an unavailable `0/0` ratio, displayed
as `N/A`, rather than zero or infinity. Gains without losses retain infinity;
losses without gains retain zero. Unreleased source after 0.1.9 preserves this
availability through native profiles, the computed Backtest bridge and reports,
even when legacy producer metrics use zero for the missing sample. Producer
tables and metrics remain unchanged. Existing lifecycle eligibility is retained;
this is not the separate ratio of positive and negative period returns.
When the default bridge prefers the producer's closed-trade statistics, an
open-only account also has an unavailable ratio. Direct native lifecycle
metrics can still include marked P&L through their existing all-open fallback;
the repair does not reclassify either sample or change their trade counts.

PDF export also depends on the browser and rendering packages documented in the
[installation guide](../getting-started/installation.md).
A headless environment can verify HTML generation without claiming that
browser-dependent PDF export was exercised.

See the [visualization API](../api/index.md#visualization) and the book's
[performance reporting notebook](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/16_strategy_simulation/09_performance_reporting.ipynb),
which directly calls Diagnostic in a longer backtest workflow. For Pyfolio
inputs, use the [migration guide](migration.md#pyfolio-portfolio-and-performance-analysis).
