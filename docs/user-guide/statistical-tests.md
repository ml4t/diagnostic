# Statistical Tests

Use these tests to separate an observed backtest result from the research
process that selected it. Report the inputs and correction method with every
result.

## Deflated Sharpe Ratio

Pass one return series for Probabilistic Sharpe Ratio or a two-dimensional
array for Deflated Sharpe Ratio across tested variants.

```python
import numpy as np

from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

rng = np.random.default_rng(42)
returns = rng.normal(
    loc=[0.0003, 0.0006, 0.0001, 0.0004],
    scale=0.01,
    size=(504, 4),
)
dsr = deflated_sharpe_ratio(
    returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)

print(f"Annualized Sharpe: {dsr.sharpe_ratio_annualized:.2f}")
print(f"Probability after correction: {dsr.probability:.3f}")
print(f"Raw trials: {dsr.n_trials_raw}")
print(f"Effective trials: {dsr.n_trials_effective:.2f}")
```

Include every strategy variant considered during selection. Omitting failed or
discarded variants understates the multiple-testing penalty.

### Keep trial history across research sessions

The candidate set belongs to the selection exercise, not to one Python process.
Persist a stable strategy identifier and the evidence produced for every variant,
including variants that failed or were discarded. Do not reuse an identifier for a
different configuration.

When every variant uses the same observations, a wide Parquet file is a minimal
trial ledger. Its first column is the observation timestamp and each remaining
column is one immutable strategy variant:

```python
from pathlib import Path

import numpy as np
import polars as pl

from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

ledger_path = Path("research/strategy-returns.parquet")

# `timestamps`, `lookback_20_returns`, and `lookback_40_returns` are the
# results from the current research session.
current = pl.DataFrame(
    {
        "timestamp": timestamps,
        "session_2026_09_02__lookback_20": lookback_20_returns,
        "session_2026_09_02__lookback_40": lookback_40_returns,
    }
)

# WRONG: this forgets variants evaluated in earlier sessions.
current_only = deflated_sharpe_ratio(current.drop("timestamp").to_numpy())

if ledger_path.exists():
    history = pl.read_parquet(ledger_path)
    if not history["timestamp"].equals(current["timestamp"]):
        raise ValueError("all variants in this ledger must use the same observations")
else:
    history = current.select("timestamp")

for strategy_id in current.columns[1:]:
    if strategy_id in history.columns:
        raise ValueError(f"strategy identifier already exists: {strategy_id}")
    history = history.with_columns(current[strategy_id])

ledger_path.parent.mkdir(parents=True, exist_ok=True)
history.write_parquet(ledger_path)

# CORRECT: the correction sees every variant accumulated for this selection exercise.
all_returns = history.drop("timestamp").to_numpy()
accumulated = deflated_sharpe_ratio(
    all_returns,
    frequency="daily",
    correlation_method="effective_rank",
    min_k_eff=2.0,
)

print(f"Current-session trials: {current_only.n_trials_raw}")
print(f"Accumulated trials: {accumulated.n_trials_raw}")
```

Start a separate ledger when the target, evaluation window, data revision, or
selection decision changes. Store those values next to the file in the research
record. Appending a newly recomputed history to an older ledger without recording
the changed inputs mixes different experiments.

If retaining each return series is impractical, persist one row per trial with its
native-frequency Sharpe ratio and the selected strategy's sample count, skewness,
excess kurtosis, and autocorrelation. Compute the cross-sectional Sharpe variance
over all ledger rows, then call `deflated_sharpe_ratio_from_statistics()` with the
accumulated `n_trials` and `variance_trials`. This is the supported route for a
single selected strategy whose trial count is maintained outside the library.

PBO and Rademacher complexity cannot be reconstructed from a trial count alone.
For PBO, persist one row per fold and strategy with both in-sample and out-of-sample
performance, then load identically ordered `(n_folds, n_strategies)` matrices. For
Rademacher complexity, retain the aligned `(n_observations, n_strategies)` return or
IC matrix. In both cases, add new strategy columns across sessions and preserve the
same row index.

## False discovery rate control

Use Benjamini-Hochberg when testing many hypotheses and you want to control the
expected fraction of false discoveries among rejected hypotheses.

```python
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

p_values = [0.001, 0.012, 0.030, 0.080, 0.40]
fdr = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

print(f"Rejected hypotheses: {fdr['rejected'].tolist()}")
print(f"Adjusted p-values: {fdr['adjusted_p_values'].round(4).tolist()}")
```

Benjamini-Hochberg assumes independent or positively dependent tests. Use
`holm_bonferroni` when you need family-wise error control instead.

## HAC-adjusted IC

Use `compute_ic_hac_stats` for autocorrelated IC series. Always pass the
forward-return horizon when labels overlap. The [HAC IC method page](../methods/hac-ic.md)
contains a complete example and the returned fields.

## Probability of backtest overfitting

`compute_pbo` compares in-sample and out-of-sample performance matrices across
strategy variants. Use it when the same variants have been evaluated across
multiple partitions. PBO complements DSR; it tests ranking decay rather than
Sharpe significance.
