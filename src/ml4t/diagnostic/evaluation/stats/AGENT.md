# stats/ - Statistical Tests

Multiple testing corrections and robust inference.

## Modules

| File | Lines | Purpose |
|------|-------|---------|
| dsr.py | 590 | Deflated Sharpe Ratio - orchestration layer |
| moments.py | 164 | Return statistics (Sharpe, skewness, kurtosis, autocorrelation) |
| sharpe_inference.py | 220 | Variance estimation, expected max Sharpe, rescaling |
| min_trl.py | 407 | Minimum Track Record Length calculation |
| pbo.py | 219 | Probability of Backtest Overfitting |
| ras.py | 436 | Rademacher Anti-Serum |
| fdr.py | 295 | FDR/FWER corrections |
| hac.py | 108 | HAC standard errors |
| bootstrap.py | 228 | Stationary bootstrap |
| reality_check.py | 155 | White's Reality Check |

## Key Functions

- `deflated_sharpe_ratio()` - DSR from return series
- `deflated_sharpe_ratio_from_statistics()` - DSR from pre-computed stats
- `compute_min_trl()` - Minimum Track Record Length
- `min_trl_fwer()` - MinTRL with FWER correction
- `compute_pbo()` - Probability of Backtest Overfitting
- `ras_sharpe_adjustment()`, `ras_ic_adjustment()` - RAS adjustments
- `benjamini_hochberg_fdr()`, `holm_bonferroni_fwer()` - Multiple testing
- `robust_ic()` - HAC-adjusted IC with bootstrap

## Result Dataclasses

- `DSRResult` - Full DSR analysis results
- `MinTRLResult` - MinTRL calculation results
- `PBOResult` - PBO analysis results

## API Convention: Kurtosis

All **public functions** use **Fisher/excess kurtosis** (normal=0):
- Parameter: `excess_kurtosis`
- Matches `scipy.stats.kurtosis()` and `pandas.DataFrame.kurtosis()` defaults

Internal functions use Pearson kurtosis (normal=3) for mathematical formulas.
