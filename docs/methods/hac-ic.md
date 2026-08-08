# HAC-Adjusted Information Coefficient

An information coefficient (IC) series can be autocorrelated because adjacent
forward-return labels overlap. A naive t-test treats the observations as
independent and can understate uncertainty. Newey-West heteroskedasticity and
autocorrelation consistent (HAC) standard errors account for this dependence.

## Compute HAC statistics

```python
import numpy as np

from ml4t.diagnostic.metrics import compute_ic_hac_stats

rng = np.random.default_rng(42)
innovations = rng.normal(loc=0.015, scale=0.08, size=252)
ic_series = np.empty_like(innovations)
ic_series[0] = innovations[0]
for index in range(1, len(ic_series)):
    ic_series[index] = 0.5 * ic_series[index - 1] + innovations[index]

stats = compute_ic_hac_stats(
    ic_series,
    label_horizon=5,
    kernel="bartlett",
    use_correction=True,
)

print(f"Mean IC: {stats['mean_ic']:.4f}")
print(f"HAC standard error: {stats['hac_se']:.4f}")
print(f"HAC t-stat: {stats['t_stat']:.2f}")
print(f"Two-sided p-value: {stats['p_value']:.4f}")
print(f"Naive t-stat: {stats['naive_t_stat']:.2f}")
print(f"Lags used: {stats['effective_lags']}")
```

Pass the forward-return horizon through `label_horizon`. When `maxlags` is not
set, the implementation uses the larger of the automatic Newey-West lag and
`label_horizon - 1`, capped at half the sample size.

## Inputs and outputs

`compute_ic_hac_stats` accepts a NumPy array or a pandas or Polars DataFrame.
For a DataFrame, set `ic_col` to the IC column name.

The returned dictionary contains the mean IC, HAC standard error, t-statistic,
two-sided p-value, sample count, lag count, and the corresponding naive
standard error and t-statistic.

## Interpretation

- Compare `hac_se` with `naive_se` to measure the effect of serial dependence.
- Test significance with `p_value`, not a fixed t-statistic threshold.
- Report `label_horizon`, `effective_lags`, sample count, and kernel with results.
- Treat HAC as an inference correction. It does not correct biased labels or data leakage.

See [statistical tests](../user-guide/statistical-tests.md) for DSR and other
multiple-testing corrections.
