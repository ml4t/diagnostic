# selection/ - Feature Triage Pipeline

Systematic selection of candidate features using IC, importance, correlation, and drift evidence.

## Main Files

- `systematic.py` - `FeatureSelector`, `SelectionStep`, `SelectionReport`
- `types.py` - `FeatureOutcomeResult`, `FeatureICResults`, `FeatureImportanceResults`

## Typical Flow

```python
from ml4t.diagnostic.selection import FeatureSelector

selector = FeatureSelector(outcome_results, correlation_matrix=corr_df)
selector.run_pipeline([
    ("drift", {"threshold": 0.2, "method": "psi"}),
    ("ic", {"threshold": 0.02}),
    ("correlation", {"threshold": 0.8}),
    ("importance", {"threshold": 0.01, "method": "mdi"}),
])
```

## Related Docs

- `docs/user-guide/feature-selection.md`
- `docs/user-guide/feature-diagnostics.md`
