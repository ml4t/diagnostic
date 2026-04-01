# integration/ - Cross-Library Bridges

Integration points for `ml4t-data`, `ml4t-engineer`, and `ml4t-backtest`.

## Key Backtest Surface

- `analyze_backtest_result` -> lazy `BacktestProfile`
- `compute_metrics_from_result` -> normalized diagnostic metrics
- `generate_tearsheet_from_result` -> HTML tearsheet from `BacktestResult`
- `profile_from_run_artifacts` -> `BacktestProfile` from saved artifact directories
- `generate_tearsheet_from_run_artifacts` -> reporting from case-study outputs
- `BacktestReportMetadata` -> reporting metadata envelope

## Supporting Modules

- `backtest.py` - public bridge functions
- `backtest_profile.py` - lazy profile object
- `backtest_analytics.py` - normalized analytics assembly
- `report_metadata.py` - report metadata model
- `backtest_contract.py` - compatibility models for trade/result exchange
- `data_contract.py` and `engineer_contract.py` - upstream integration contracts

## Notes

- This package is where the recent tearsheet/reporting bridge lives.
- Rendering itself happens in `visualization/backtest/`.
