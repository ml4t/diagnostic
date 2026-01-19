"""Statistical tests for financial ML evaluation.

This package implements advanced statistical tests used in ml4t-diagnostic's
Three-Tier Framework:

**Multiple Testing Corrections**:
- Deflated Sharpe Ratio (DSR) for selection bias correction
- Rademacher Anti-Serum (RAS) for correlation-aware multiple testing
- False Discovery Rate (FDR) and Family-Wise Error Rate (FWER) corrections

**Time Series Inference**:
- HAC-adjusted Information Coefficient for autocorrelated data
- Stationary bootstrap for temporal dependence preservation

**Strategy Comparison**:
- White's Reality Check for multiple strategy comparison
- Probability of Backtest Overfitting (PBO)

All tests are implemented with:
- Mathematical correctness validated against academic references
- Proper handling of autocorrelation and heteroskedasticity
- Numerical stability for edge cases
- Support for both single and multiple hypothesis testing

Module Decomposition (v1.4+)
----------------------------
The stats package is organized into focused modules:

**Sharpe Ratio Analysis**:
- moments.py: Return statistics (Sharpe, skewness, kurtosis, autocorr)
- sharpe_inference.py: Variance estimation, expected max calculation
- minimum_track_record.py: Minimum Track Record Length
- backtest_overfitting.py: Probability of Backtest Overfitting
- deflated_sharpe_ratio.py: DSR/PSR orchestration layer (main entry points)

**Other Statistical Tests**:
- rademacher_adjustment.py: Rademacher complexity and RAS adjustments
- bootstrap.py: Stationary bootstrap methods
- hac_standard_errors.py: HAC-adjusted IC estimation
- false_discovery_rate.py: FDR and FWER corrections
- reality_check.py: White's Reality Check

All original imports are preserved for backward compatibility.
"""

# =============================================================================
# MOMENTS AND RETURN STATISTICS
# =============================================================================
# =============================================================================
# BOOTSTRAP METHODS
# =============================================================================
# =============================================================================
# PROBABILITY OF BACKTEST OVERFITTING
# =============================================================================
from ml4t.diagnostic.evaluation.stats.backtest_overfitting import (
    PBOResult,
    compute_pbo,
)
from ml4t.diagnostic.evaluation.stats.bootstrap import (
    _optimal_block_size,
    _stationary_bootstrap_indices,
    stationary_bootstrap_ic,
)

# =============================================================================
# DSR/PSR (MAIN ENTRY POINTS)
# =============================================================================
from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import (
    DSRResult,
    Frequency,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics,
)

# =============================================================================
# FDR CORRECTIONS
# =============================================================================
from ml4t.diagnostic.evaluation.stats.false_discovery_rate import (
    benjamini_hochberg_fdr,
    holm_bonferroni,
    multiple_testing_summary,
)

# =============================================================================
# ROBUST IC ESTIMATION
# =============================================================================
from ml4t.diagnostic.evaluation.stats.hac_standard_errors import (
    hac_adjusted_ic,
    robust_ic,
)

# =============================================================================
# MINIMUM TRACK RECORD LENGTH
# =============================================================================
from ml4t.diagnostic.evaluation.stats.minimum_track_record import (
    DEFAULT_PERIODS_PER_YEAR,
    MinTRLResult,
    compute_min_trl,
    min_trl_fwer,
)
from ml4t.diagnostic.evaluation.stats.moments import (
    compute_autocorrelation,
    compute_kurtosis,
    compute_return_statistics,
    compute_sharpe,
    compute_skewness,
)

# =============================================================================
# RADEMACHER ANTI-SERUM
# =============================================================================
from ml4t.diagnostic.evaluation.stats.rademacher_adjustment import (
    RASResult,
    rademacher_complexity,
    ras_ic_adjustment,
    ras_sharpe_adjustment,
)

# =============================================================================
# WHITE'S REALITY CHECK
# =============================================================================
from ml4t.diagnostic.evaluation.stats.reality_check import (
    whites_reality_check,
)

# =============================================================================
# SHARPE RATIO INFERENCE
# =============================================================================
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    EULER_GAMMA,
    VARIANCE_RESCALING_FACTORS,
    compute_expected_max_sharpe,
    compute_sharpe_variance,
    get_variance_rescaling_factor,
)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# Old private names for variance rescaling
_VARIANCE_RESCALING_FACTORS = VARIANCE_RESCALING_FACTORS
_get_variance_rescaling_factor = get_variance_rescaling_factor

__all__ = [
    # Moments and return statistics
    "compute_return_statistics",
    "compute_sharpe",
    "compute_skewness",
    "compute_kurtosis",
    "compute_autocorrelation",
    # Sharpe inference
    "compute_sharpe_variance",
    "compute_expected_max_sharpe",
    "get_variance_rescaling_factor",
    "EULER_GAMMA",
    "VARIANCE_RESCALING_FACTORS",
    # MinTRL
    "MinTRLResult",
    "compute_min_trl",
    "min_trl_fwer",
    "DEFAULT_PERIODS_PER_YEAR",
    # PBO
    "PBOResult",
    "compute_pbo",
    # DSR/PSR
    "DSRResult",
    "Frequency",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_statistics",
    # RAS
    "RASResult",
    "rademacher_complexity",
    "ras_ic_adjustment",
    "ras_sharpe_adjustment",
    # Bootstrap
    "stationary_bootstrap_ic",
    "_stationary_bootstrap_indices",
    "_optimal_block_size",
    # Robust IC (bootstrap-based)
    "robust_ic",
    "hac_adjusted_ic",
    # FDR
    "benjamini_hochberg_fdr",
    "holm_bonferroni",
    "multiple_testing_summary",
    # Reality Check
    "whites_reality_check",
    # Backward compat aliases
    "_get_variance_rescaling_factor",
    "_VARIANCE_RESCALING_FACTORS",
]
