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
