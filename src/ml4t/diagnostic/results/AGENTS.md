# results/ - Result Schemas And Summaries

Structured result objects used across diagnostics, signal analysis, Sharpe inference, portfolio analysis, and event studies.

## Main Modules

- `base.py` - shared result base types
- `feature_results.py` - feature diagnostics and feature-outcome schemas
- `sharpe_results.py` - PSR, DSR, FDR, and MinTRL schemas
- `portfolio_results.py` - portfolio metrics and Bayesian comparison outputs
- `event_results.py` - event-study outputs
- `multi_signal_results.py` - multi-signal comparison and summary types

## Result Families

- `signal_results/` - signal IC, quantile, turnover, and tearsheet outputs
- `barrier_results/` - barrier-analysis outputs

## Notes

- Most result objects support `.to_dict()` and summary-style helpers.
- Newer factor and Trade-SHAP workflows also expose dedicated models from their own packages.
