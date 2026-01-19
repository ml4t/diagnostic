# metrics/ - Feature Metrics

IC, importance, and interaction analysis.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| ic.py | 530 | Core IC functions |
| ic_statistics.py | 446 | HAC-adjusted IC, decay |
| conditional_ic.py | 469 | Conditional IC |
| importance_classical.py | 375 | PFI, MDI |
| importance_mda.py | 371 | Mean Decrease Accuracy |
| importance_shap.py | 715 | SHAP importance |
| importance_analysis.py | 338 | Multi-method comparison |
| interactions.py | 772 | H-statistic, SHAP interactions |
| feature_outcome.py | 475 | Feature-outcome analysis |
| monotonicity.py | 226 | Monotonicity tests |
| risk_adjusted.py | 324 | Sharpe, Sortino, drawdown |

## Key Functions

`information_coefficient()`, `compute_ic_series()`, `analyze_ml_importance()`, `compute_h_statistic()`, `compute_shap_importance()`
