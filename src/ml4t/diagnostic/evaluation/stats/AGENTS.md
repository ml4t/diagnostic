# evaluation/stats/ - Statistical Inference

Multiple-testing correction and robust inference for strategy evaluation.

## Main Modules

- `deflated_sharpe_ratio.py` - PSR and DSR from returns or pre-computed statistics
- `minimum_track_record.py` - MinTRL calculations
- `backtest_overfitting.py` - Probability of Backtest Overfitting helpers
- `rademacher_adjustment.py` - RAS adjustments
- `false_discovery_rate.py` - FDR and FWER corrections
- `hac_standard_errors.py` - bootstrap-based robust IC inference
- `bootstrap.py`, `moments.py`, `sharpe_inference.py` - supporting math
- `reality_check.py` - White's Reality Check

## Common Entry Points

- `deflated_sharpe_ratio`
- `deflated_sharpe_ratio_from_statistics`
- `compute_min_trl`
- `compute_pbo`
- `ras_sharpe_adjustment`
- `benjamini_hochberg_fdr`
- `robust_ic`

## Conventions

- Public kurtosis inputs use Fisher/excess kurtosis (`normal = 0`)
- `DSRResult`, `MinTRLResult`, and related result objects carry richer interpretation than raw scalars
