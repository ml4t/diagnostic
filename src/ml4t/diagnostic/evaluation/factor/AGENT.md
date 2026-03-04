# Factor Exposure & Attribution

Returns-based factor analysis: OLS+HAC regression, rolling exposures, lagged return attribution, and variance-based risk decomposition.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| results.py | ~480 | 8 result dataclasses (FactorModelResult, RollingExposureResult, etc.) |
| data.py | ~250 | FactorData container + factory methods (from_dataframe, from_fama_french, from_aqr) |
| static_model.py | ~150 | OLS + HAC via statsmodels (compute_factor_model) |
| rolling_model.py | ~180 | Rolling OLS + stability diagnostics (compute_rolling_exposures) |
| attribution.py | ~220 | Return attribution with CIs + maximal attribution (Paleologo Ch 14) |
| risk.py | ~120 | Variance decomposition, MCTR, Euler (compute_risk_attribution) |
| validation.py | ~100 | QLIKE, MALV, Ljung-Box, Jarque-Bera model quality tests |
| analysis.py | ~200 | FactorAnalysis orchestrator class with caching |
| kalman.py | ~180 | Kalman filter time-varying betas (Tier 2) |
| regularized.py | ~130 | Ridge/LASSO/ElasticNet + bootstrap SEs (Tier 2) |
| timing.py | ~80 | Factor timing correlations (Tier 2) |

## Key Functions

- `compute_factor_model(returns, factor_data, hac=True)` → FactorModelResult
- `compute_rolling_exposures(returns, factor_data, window=63)` → RollingExposureResult
- `compute_return_attribution(returns, factor_data, window=63, lag=1)` → AttributionResult
- `compute_maximal_attribution(returns, factor_data, factors_of_interest)` → MaximalAttributionResult
- `compute_risk_attribution(returns, factor_data, shrinkage="ledoit_wolf")` → RiskAttributionResult
- `FactorAnalysis` — orchestrator class with caching and `generate_report()`
