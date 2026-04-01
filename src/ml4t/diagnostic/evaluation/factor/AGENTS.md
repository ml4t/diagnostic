# evaluation/factor/ - Factor Exposure And Attribution

Returns-based factor analysis with static OLS, rolling exposures, attribution, and risk decomposition.

## Core Modules

- `data.py` - `FactorData` container and input normalization
- `static_model.py` - `compute_factor_model`
- `rolling_model.py` - `compute_rolling_exposures`
- `attribution.py` - return and maximal attribution
- `risk.py` - variance and contribution-to-risk attribution
- `analysis.py` - `FactorAnalysis` orchestrator
- `validation.py` - model-quality checks
- `kalman.py`, `regularized.py`, `timing.py` - advanced extensions

## Key Exports

- `FactorData`
- `FactorAnalysis`
- `compute_factor_model`
- `compute_rolling_exposures`
- `compute_return_attribution`
- `compute_risk_attribution`

## Related Docs

- Architecture and docs integration are covered in `docs/reference/architecture.md`
- Book-facing usage is mapped in `docs/book-guide/`
