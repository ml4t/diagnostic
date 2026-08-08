# Feature Selection

`FeatureSelector` applies explicit filters to evaluated features. It consumes
`FeatureOutcomeResult`; it does not fit a model or compute IC values itself.

## Filter by IC and correlation

```python
import polars as pl

from ml4t.diagnostic.selection import FeatureSelector
from ml4t.diagnostic.selection.types import FeatureICResults, FeatureOutcomeResult

features = ["momentum_5d", "momentum_20d", "volatility", "noise"]
mean_ic = {
    "momentum_5d": 0.045,
    "momentum_20d": 0.038,
    "volatility": -0.030,
    "noise": 0.004,
}
ic_results = {
    feature: FeatureICResults(
        feature=feature,
        ic_mean=value,
        ic_std=0.02,
        ic_ir=value / 0.02,
        t_stat=value / 0.01,
        p_value=0.01 if abs(value) >= 0.02 else 0.60,
        ic_by_lag={1: value},
        n_observations=120,
    )
    for feature, value in mean_ic.items()
}
outcomes = FeatureOutcomeResult(features=features, ic_results=ic_results)
correlations = pl.DataFrame(
    {
        "feature": features,
        "momentum_5d": [1.0, 0.92, 0.10, 0.02],
        "momentum_20d": [0.92, 1.0, 0.12, 0.01],
        "volatility": [0.10, 0.12, 1.0, 0.05],
        "noise": [0.02, 0.01, 0.05, 1.0],
    }
)

selector = FeatureSelector(outcomes, correlations)
selector.run_pipeline(
    [
        ("ic", {"threshold": 0.02, "min_periods": 60}),
        ("correlation", {"threshold": 0.80, "keep_strategy": "higher_ic"}),
    ]
)

report = selector.get_selection_report()
print(f"Selected: {report.final_features}")
print(f"Removed: {selector.get_removed_features()}")
```

The IC filter uses absolute IC, so a stable negative relation can survive. The
correlation filter above keeps the member of each correlated pair with the
larger absolute IC.

## Available filters

| Filter name | Required results |
|-------------|------------------|
| `ic` | `FeatureICResults` |
| `importance` | `FeatureImportanceResults` |
| `correlation` | Correlation matrix |
| `drift` | Drift results on `FeatureOutcomeResult` |

Record thresholds and the final `SelectionReport` with each model run. Feature
selection on the full dataset leaks information; compute its inputs within the
training portion of each cross-validation fold.
