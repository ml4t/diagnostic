# metrics/ - Feature, Signal, And Risk Metrics

Feature-level and signal-level metric functions used by diagnostics, selection, and reporting workflows.

## Main Modules

- `ic.py` - IC computation primitives
- `ic_inference.py` - HAC-adjusted IC statistics and summary helpers
- `conditional.py` - conditional IC analysis
- `importance_classical.py`, `importance_mda.py`, `importance_shap.py` - importance methods
- `importance_analysis.py` - multi-method comparison
- `interactions.py` - H-statistic and SHAP interaction workflows
- `feature_outcome.py` - combined feature-outcome analysis
- `monotonicity.py` and `risk_adjusted.py` - ranking and portfolio-style metrics

## Common Entry Points

- `cross_sectional_ic_series`
- `cross_sectional_ic`
- `pooled_ic`
- `compute_ic_hac_stats`
- `analyze_ml_importance`
- `compute_shap_importance`
- `compute_h_statistic`
- `analyze_feature_outcome`
