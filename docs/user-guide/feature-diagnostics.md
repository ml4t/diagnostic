# Feature Diagnostics

Run feature diagnostics before fitting a model. The default analysis checks
stationarity, autocorrelation, volatility, and distribution properties.

## Diagnose one feature

```python
import numpy as np

from ml4t.diagnostic.evaluation import FeatureDiagnostics

rng = np.random.default_rng(42)
feature = np.empty(500)
innovations = rng.normal(size=500)
feature[0] = innovations[0]
for index in range(1, len(feature)):
    feature[index] = 0.6 * feature[index - 1] + innovations[index]

diagnostics = FeatureDiagnostics()
result = diagnostics.run_diagnostics(feature, name="momentum_score")

print(result.summary())
print(f"Health score: {result.health_score:.2f}")
print(f"Stationarity: {result.stationarity.consensus}")
print(f"Flags: {result.flags}")
```

## Read the result

`run_diagnostics` returns one result object with these sections:

| Attribute | Contents |
|-----------|----------|
| `stationarity` | Unit-root tests and consensus |
| `autocorrelation` | ACF, PACF, and suggested ARIMA order |
| `volatility` | Conditional heteroskedasticity and persistence |
| `distribution` | Moments, normality, and tail analysis |
| `health_score` | Aggregate score from enabled checks |
| `flags` | Conditions that need review |

Run `FeatureDiagnostics.run_batch_diagnostics` for a pandas DataFrame. Configure
individual sections with `DiagnosticConfig` and the settings classes in
`ml4t.diagnostic.config`.

Feature quality does not establish predictive value. Use the
[quickstart](../getting-started/quickstart.md) to measure cross-sectional IC and
the [HAC IC method](../methods/hac-ic.md) for time-series inference.

## Profile labels by feature quantile

`quantile_profile` assigns quantiles within each timestamp by default. Pass
`by=None` only when pooled quantiles match the research question.

```python
import polars as pl

from ml4t.diagnostic import quantile_profile

panel = pl.DataFrame(
    {
        "timestamp": [date for date in range(3) for _ in range(10)],
        "asset": [f"asset_{asset}" for _ in range(3) for asset in range(10)],
        "feature": [float(asset) for _ in range(3) for asset in range(10)],
        "label": [0.01 * asset + 0.001 * date for date in range(3) for asset in range(10)],
    }
)

profile = quantile_profile(
    panel,
    feature="feature",
    label="label",
    n_quantiles=5,
    by="timestamp",
    keys=["timestamp", "asset"],
    min_per_bucket=6,
)

print(profile.means)
print(profile.counts)
print(f"Monotonicity: {profile.monotonicity:.2f}")
```

Rows with non-finite feature or label values are excluded. A group with fewer
valid rows than quantiles is excluded. Pooled input with fewer valid rows than
quantiles raises `ValueError`. Equal feature values share an average rank, so
their bucket does not depend on input order. Empty buckets can still occur when
many rows have the same feature value; in that case monotonicity is `NaN`.
