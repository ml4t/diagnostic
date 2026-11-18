"""Validation of the Sharpe ratio inference functions against López de Prado et al.

Reference: López de Prado, M., Lipton, A., & Zoonekynd, V., "How to Use the
Sharpe Ratio", ADIA Lab Research Paper Series, No. 19, version of 7 March 2026.
Equation and exhibit numbers refer to that version.

Every expected value is compared with the output of a library function:

- Standard deviation of the Sharpe ratio estimator (Equation 5):
  compute_sharpe_variance
- PSR (Equation 9): deflated_sharpe_ratio_from_statistics with one trial
- MinTRL (Equation 11): compute_min_trl
- Expected maximum Sharpe ratio (Equation 28): compute_expected_max_sharpe
- Re-scaling factors (Exhibit 4): get_variance_rescaling_factor
- DSR (Equation 30): deflated_sharpe_ratio_from_statistics

The reference values and their sources are in
tests/test_evaluation/fixtures/dsr_reference.py.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from ml4t.diagnostic.evaluation.stats import (
    DSRResult,
    compute_min_trl,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_statistics,
    min_trl_fwer,
)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    VARIANCE_RESCALING_FACTORS as LIBRARY_RESCALING_FACTORS,
)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    compute_deflated_sharpe_std,
)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    compute_expected_max_sharpe as _compute_expected_max_sharpe,
)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    compute_sharpe_variance as _compute_sharpe_variance,
)
from ml4t.diagnostic.evaluation.stats.sharpe_inference import (
    get_variance_rescaling_factor as _get_variance_rescaling_factor,
)
from tests.test_evaluation.fixtures.dsr_reference import (
    DSR_FEW_TRIALS,
    DSR_MANY_TRIALS,
    DSR_NORMAL,
    DSR_PAPER_EXAMPLE,
    MINTRL_CASES,
    PAPER_SHARPE,
    PSR_CASES,
    VARIANCE_RESCALING_FACTORS,
)

# =============================================================================
# GOLDEN TESTS - 2025 PAPER REFERENCE VALUES
# =============================================================================


class TestGoldenPaperExamples:
    """Published values of the paper's worked example, computed by the library.

    The example is a two-year monthly track record with
    (μ, σ, γ₃, γ₄, ρ, T) = (0.036%, 0.079%, -2.448, 10.164, 0.2, 24).
    """

    def test_sharpe_ratio_standard_deviation(self):
        """Golden test: standard deviation of the Sharpe ratio estimator (Equation 5).

        Reference: numerical example after Equation 5.

        The paper evaluates Equation 5 at the estimated Sharpe ratio and reports
        σ[SR] = 0.379. Assuming i.i.d. normal returns it reports 0.214.
        """
        with_moments = _compute_sharpe_variance(
            sharpe=PAPER_SHARPE,
            n_samples=24,
            skewness=-2.448,
            kurtosis=10.164,
            autocorrelation=0.2,
        )
        iid_normal = _compute_sharpe_variance(
            sharpe=PAPER_SHARPE,
            n_samples=24,
            skewness=0.0,
            kurtosis=3.0,
            autocorrelation=0.0,
        )

        assert abs(np.sqrt(with_moments) - 0.379) < 0.0005, (
            f"σ[SR] should be 0.379, got {np.sqrt(with_moments):.4f}"
        )
        assert abs(np.sqrt(iid_normal) - 0.214) < 0.0005, (
            f"σ[SR] under i.i.d. normal returns should be 0.214, got {np.sqrt(iid_normal):.4f}"
        )

    def test_expected_max_sharpe_k10(self):
        """Golden test: E[max{SR}] for K=10, V[{SR_k}]=0.1.

        Reference: Equation 28 and the numerical example after Exhibit 5.

        E[max{SR_k}] ≈ √V[{SR_k}] × ((1-γ)Φ⁻¹(1-1/K) + γΦ⁻¹(1-1/(Ke))) = 0.498

        where γ ≈ 0.5772 is the Euler-Mascheroni constant.
        """
        expected_max = _compute_expected_max_sharpe(10, 0.1)

        assert abs(expected_max - 0.498) < 0.0005, (
            f"E[max{{SR}}] should be 0.498, got {expected_max:.4f}"
        )

    def test_variance_rescaling_factors_exhibit4(self):
        """Golden test: re-scaling factors √V[max{X_k}] for K = 1 to 100.

        Reference: Exhibit 4, which prints five decimals.
        """
        assert sorted(VARIANCE_RESCALING_FACTORS) == list(range(1, 101))

        for k, expected_factor in VARIANCE_RESCALING_FACTORS.items():
            actual_factor = _get_variance_rescaling_factor(k)

            assert abs(actual_factor - expected_factor) < 0.6e-5, (
                f"Rescaling factor mismatch for K={k}\n"
                f"Expected: {expected_factor:.5f}\n"
                f"Got: {actual_factor:.5f}"
            )

    def test_library_rescaling_table_matches_exhibit4(self):
        """The values the library exports are the exhibit's values for the same K.

        Exhibit 4 has ten values per row. Reading its first column as K = 20,
        30, ... assigns each of those K the value of K + 1.
        """
        for k, factor in LIBRARY_RESCALING_FACTORS.items():
            assert factor == VARIANCE_RESCALING_FACTORS[k], f"K={k}"

    def test_dsr_denominator_uses_cross_sectional_variance(self):
        """Golden test: with K>1 the DSR denominator is the dispersion across trials.

        Reference: Equation 30 and the numerical example after Exhibit 5.

        σ[SR₀,K] = √V[{SR_k}] × √V[max_k{X_k}]

        The paper rescales √V[{SR_k}] = 0.316 to σ[SR₀,K] = 0.186 and reports
        DSR = 0.410. Rescaling the standard deviation of the Sharpe ratio
        estimator instead gives 0.230 and DSR = 0.427 for the same inputs.
        """
        case = DSR_PAPER_EXAMPLE

        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=case.observed_sharpe,
            n_trials=case.n_trials,
            variance_trials=case.variance_trials,
            n_samples=case.n_samples,
            skewness=case.skewness,
            excess_kurtosis=case.kurtosis - 3.0,  # Convert Pearson to Fisher
            autocorrelation=case.autocorrelation,
        )

        assert abs(np.sqrt(case.variance_trials) - 0.316) < 0.0005

        std_sr = compute_deflated_sharpe_std(
            threshold_sharpe=result.expected_max_sharpe,
            n_samples=case.n_samples,
            skewness=case.skewness,
            kurtosis=case.kurtosis,
            autocorrelation=case.autocorrelation,
            n_trials=case.n_trials,
            variance_trials=case.variance_trials,
        )
        assert abs(std_sr - 0.186) < 0.0005, f"σ[SR₀,K] should be 0.186, got {std_sr:.4f}"

        # The z-score the function reports uses the same standard deviation
        implied_std = (case.observed_sharpe - result.expected_max_sharpe) / result.z_score
        assert abs(implied_std - 0.186) < 0.0005, (
            f"DSR z-score implies σ = {implied_std:.4f}, paper says 0.186"
        )

        assert abs(result.probability - 0.410) < 0.0005, (
            f"Implementation DSR does not match paper value\n"
            f"Paper says: 0.410\n"
            f"Implementation gives: {result.probability:.4f}"
        )

    def test_dsr_full_example(self):
        """Full validation of the paper's DSR example with all components.

        Reference: numerical example after Exhibit 5.

        1. E[max{SR}] = 0.498
        2. Adjusted threshold = SR₀ + E[max{SR}] = 0 + 0.498
        3. DSR = 0.410, against a one-trial PSR of 0.966
        """
        case = DSR_PAPER_EXAMPLE
        inputs = {
            "observed_sharpe": case.observed_sharpe,
            "n_samples": case.n_samples,
            "skewness": case.skewness,
            "excess_kurtosis": case.kurtosis - 3.0,  # Convert Pearson to Fisher
            "autocorrelation": case.autocorrelation,
        }

        result = deflated_sharpe_ratio_from_statistics(
            n_trials=case.n_trials, variance_trials=case.variance_trials, **inputs
        )
        single_trial = deflated_sharpe_ratio_from_statistics(
            n_trials=1, variance_trials=0.0, **inputs
        )

        assert abs(result.expected_max_sharpe - 0.498) < 0.0005, (
            f"E[max{{SR}}] should be 0.498, got {result.expected_max_sharpe:.4f}"
        )
        assert abs(result.probability - 0.410) < 0.0005, (
            f"DSR should be 0.410, got {result.probability:.4f}"
        )
        assert abs(single_trial.probability - 0.966) < 0.0005, (
            f"One-trial PSR should be 0.966, got {single_trial.probability:.4f}"
        )

        # Deflated Sharpe should be observed - E[max]
        expected_deflated = case.observed_sharpe - result.expected_max_sharpe
        assert abs(result.deflated_sharpe - expected_deflated) < 1e-12, "Deflated Sharpe mismatch"

    def test_dsr_single_trial_equals_psr(self):
        """DSR with K=1 is the PSR: no deflation, and the paper's PSR of 0.966."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=PAPER_SHARPE,
            n_trials=1,
            variance_trials=0.0,
            n_samples=24,
            skewness=-2.448,
            excess_kurtosis=10.164 - 3.0,  # Convert Pearson to Fisher
            autocorrelation=0.2,
        )

        assert result.expected_max_sharpe == 0.0
        assert result.n_trials == 1
        assert abs(result.probability - 0.966) < 0.0005, (
            f"One-trial PSR should be 0.966, got {result.probability:.4f}"
        )

    def test_min_trl_fwer_multiple_testing(self):
        """Test min_trl_fwer adjusts target for multiple testing.

        For K>1 strategies, the MinTRL should be higher than the single-strategy
        MinTRL due to the adjusted threshold (SR₀|K = SR₀ + E[max{SR}]).
        """
        observed_sharpe = 0.456
        n_trials = 10
        variance_trials = 0.1

        # Single strategy MinTRL
        single_result = compute_min_trl(
            observed_sharpe=observed_sharpe,
            target_sharpe=0.0,
            confidence_level=0.95,
            frequency="monthly",
        )

        # Multiple testing MinTRL
        fwer_result = min_trl_fwer(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            variance_trials=variance_trials,
            target_sharpe=0.0,
            confidence_level=0.95,
            frequency="monthly",
        )

        # FWER result should have adjusted target (SR₀ + E[max])
        expected_max = _compute_expected_max_sharpe(n_trials, variance_trials)
        assert abs(fwer_result.target_sharpe - expected_max) < 0.001, (
            f"Adjusted target should be {expected_max:.4f}, got {fwer_result.target_sharpe:.4f}"
        )

        # MinTRL under FWER should be >= single strategy MinTRL
        # (actually will be much higher due to reduced gap: observed - adjusted_target)
        assert fwer_result.min_trl >= single_result.min_trl, (
            f"FWER MinTRL ({fwer_result.min_trl}) should be >= "
            f"single MinTRL ({single_result.min_trl})"
        )

    def test_min_trl_fwer_single_trial_reduces_to_standard(self):
        """For K=1, min_trl_fwer should give same result as compute_min_trl."""
        observed_sharpe = 0.5
        target_sharpe = 0.0

        single_result = compute_min_trl(
            observed_sharpe=observed_sharpe,
            target_sharpe=target_sharpe,
            confidence_level=0.95,
            frequency="monthly",
        )

        fwer_result = min_trl_fwer(
            observed_sharpe=observed_sharpe,
            n_trials=1,
            variance_trials=0.0,  # No variance for single trial
            target_sharpe=target_sharpe,
            confidence_level=0.95,
            frequency="monthly",
        )

        # For K=1, E[max]=0, so results should match
        assert fwer_result.min_trl == single_result.min_trl, (
            f"FWER with K=1 should match single: {fwer_result.min_trl} != {single_result.min_trl}"
        )
        assert fwer_result.target_sharpe == target_sharpe, "Target should not be adjusted for K=1"


# =============================================================================
# PSR VALIDATION TESTS
# =============================================================================


def _psr(
    sharpe_ratio: float,
    skewness: float,
    excess_kurtosis: float,
    n_samples: int,
    sharpe_star: float = 0.0,
    autocorrelation: float = 0.0,
) -> float:
    """PSR from the library: the deflated Sharpe ratio of a single trial (Equation 9).

    probabilistic_sharpe_ratio in the same package is the 2012 statistic. It
    evaluates the variance at the observed Sharpe ratio, divides by T - 1 and
    has no serial-correlation term, so it does not return the paper's values.
    """
    return deflated_sharpe_ratio_from_statistics(
        observed_sharpe=sharpe_ratio,
        n_samples=n_samples,
        n_trials=1,
        variance_trials=0.0,
        benchmark_sharpe=sharpe_star,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        autocorrelation=autocorrelation,
    ).probability


class TestPSRValidation:
    """Validate the library's PSR against López de Prado et al. reference cases."""

    @pytest.mark.parametrize("case", PSR_CASES, ids=lambda c: c.name)
    def test_psr_reference(self, case):
        """PSR of each reference case, through the library."""
        result = _psr(
            sharpe_ratio=case.sharpe_ratio,
            skewness=case.skewness,
            excess_kurtosis=case.kurtosis - 3.0,
            n_samples=case.n_samples,
            sharpe_star=case.sharpe_star,
            autocorrelation=case.autocorrelation,
        )

        error = abs(result - case.expected_psr)

        assert error < case.tolerance, (
            f"{case.name}: PSR validation failed\n"
            f"Expected: {case.expected_psr:.6f}\n"
            f"Got: {result:.6f}\n"
            f"Error: {error:.6f} (tolerance: {case.tolerance})\n"
            f"Reference: {case.reference}"
        )

    def test_psr_zero_sharpe(self):
        """Edge case: PSR with zero Sharpe ratio."""
        psr = _psr(
            sharpe_ratio=0.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=100,
            sharpe_star=0.0,
        )

        # With SR=0 and SR_0=0, PSR should be 0.5 (50% chance)
        assert abs(psr - 0.5) < 0.001

    def test_psr_negative_sharpe(self):
        """Edge case: PSR with negative Sharpe ratio."""
        psr = _psr(
            sharpe_ratio=-1.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=100,
            sharpe_star=0.0,
        )

        # Negative Sharpe should have very low PSR
        assert psr < 0.01

    def test_psr_extreme_kurtosis(self):
        """Edge case: PSR with very high kurtosis (fat tails).

        With SR₀=0 the kurtosis term of Equation 5 vanishes, so the benchmark
        is non-zero here.
        """
        psr_normal = _psr(
            sharpe_ratio=1.0,
            skewness=0.0,
            excess_kurtosis=0.0,  # Normal
            n_samples=100,
            sharpe_star=0.5,
        )

        psr_kurtotic = _psr(
            sharpe_ratio=1.0,
            skewness=0.0,
            excess_kurtosis=17.0,  # Extreme kurtosis (Fisher: Pearson 20 - 3)
            n_samples=100,
            sharpe_star=0.5,
        )

        # High kurtosis increases uncertainty, should reduce PSR
        assert psr_kurtotic < psr_normal

    def test_psr_serial_correlation_effect(self):
        """Positive serial correlation raises the estimator variance and lowers PSR."""
        psr_independent = _psr(
            sharpe_ratio=0.3,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=60,
        )
        psr_correlated = _psr(
            sharpe_ratio=0.3,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=60,
            autocorrelation=0.3,
        )

        assert psr_correlated < psr_independent

    def test_psr_sample_size_effect(self):
        """Verify PSR increases with sample size for same Sharpe."""
        sharpe_ratio = 0.5

        psr_small = _psr(
            sharpe_ratio=sharpe_ratio,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=12,
            sharpe_star=0.0,
        )

        psr_large = _psr(
            sharpe_ratio=sharpe_ratio,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=120,
            sharpe_star=0.0,
        )

        # Larger sample size should increase confidence
        assert psr_large > psr_small


# =============================================================================
# MinTRL VALIDATION TESTS
# =============================================================================


def _min_trl(
    sharpe_ratio: float,
    sharpe_star: float,
    skewness: float,
    excess_kurtosis: float,
    alpha: float = 0.05,
    autocorrelation: float = 0.0,
) -> float:
    """MinTRL from the library, in observations."""
    return compute_min_trl(
        observed_sharpe=sharpe_ratio,
        target_sharpe=sharpe_star,
        confidence_level=1 - alpha,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        autocorrelation=autocorrelation,
    ).min_trl


class TestMinTRLValidation:
    """Validate the library's MinTRL against López de Prado et al. reference cases."""

    @pytest.mark.parametrize("case", MINTRL_CASES, ids=lambda c: c.name)
    def test_mintrl_reference(self, case):
        """MinTRL of each reference case, through the library.

        Equation 11 gives a real number of observations (19.543 and 39.369
        months in the paper's example). compute_min_trl adds one observation
        and rounds up, so it returns ceil(1 + value), here 21 and 41.
        """
        result = _min_trl(
            sharpe_ratio=case.sharpe_ratio,
            sharpe_star=case.sharpe_star,
            skewness=case.skewness,
            excess_kurtosis=case.kurtosis - 3.0,
            alpha=case.alpha,
            autocorrelation=case.autocorrelation,
        )

        expected = math.ceil(1 + case.expected_mintrl)
        assert result == expected, (
            f"{case.name}: MinTRL validation failed\n"
            f"Equation 11: {case.expected_mintrl:.3f}, so expected {expected} observations\n"
            f"Got: {result}\n"
            f"Reference: {case.reference}"
        )

    @pytest.mark.parametrize("case", MINTRL_CASES, ids=lambda c: c.name)
    def test_mintrl_is_shortest_record_that_rejects(self, case):
        """MinTRL is the smallest T at which PSR reaches 1 - α.

        Reference: the sentence after Equation 11. With the library's extra
        observation, PSR reaches 1 - α at min_trl - 1 observations and not at
        min_trl - 2. The two results come from separate library functions.
        """
        inputs = {
            "sharpe_ratio": case.sharpe_ratio,
            "sharpe_star": case.sharpe_star,
            "skewness": case.skewness,
            "excess_kurtosis": case.kurtosis - 3.0,
            "autocorrelation": case.autocorrelation,
        }
        min_trl = int(_min_trl(alpha=case.alpha, **inputs))

        assert _psr(n_samples=min_trl - 1, **inputs) >= 1 - case.alpha
        assert _psr(n_samples=min_trl - 2, **inputs) < 1 - case.alpha

    def test_mintrl_zero_target(self):
        """Edge case: MinTRL with SR_0 = 0."""
        mintrl = _min_trl(
            sharpe_ratio=1.0,
            sharpe_star=0.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.05,
        )

        # (1.645 / 1.0)^2 = 2.706, plus one observation, rounded up
        assert mintrl == 4

    def test_mintrl_equal_sharpes(self):
        """Edge case: MinTRL when SR = SR_0."""
        mintrl = _min_trl(
            sharpe_ratio=1.0,
            sharpe_star=1.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.05,
        )

        # Should be infinite (cannot prove superiority)
        assert np.isinf(mintrl)

    def test_mintrl_below_target(self):
        """Edge case: MinTRL when SR < SR_0."""
        mintrl = _min_trl(
            sharpe_ratio=0.5,
            sharpe_star=1.0,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.05,
        )

        # Should be infinite (observed is worse than target)
        assert np.isinf(mintrl)

    def test_mintrl_alpha_sensitivity(self):
        """Verify MinTRL increases with stricter alpha."""
        sharpe_ratio = 1.0
        sharpe_star = 0.5

        mintrl_lenient = _min_trl(
            sharpe_ratio=sharpe_ratio,
            sharpe_star=sharpe_star,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.10,  # Lenient
        )

        mintrl_strict = _min_trl(
            sharpe_ratio=sharpe_ratio,
            sharpe_star=sharpe_star,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.01,  # Strict
        )

        # Stricter alpha requires more observations
        assert mintrl_strict > mintrl_lenient


# =============================================================================
# DSR VALIDATION TESTS
# =============================================================================


class TestDSRValidation:
    """Validate DSR against López de Prado et al. reference cases."""

    def test_extreme_p_value_does_not_underflow(self):
        """Representable DSR upper tails remain nonzero."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=0.4,
            n_trials=1,
            variance_trials=0.0,
            n_samples=1000,
            skewness=0.0,
            excess_kurtosis=0.0,
        )

        expected = norm.sf(result.z_score)
        assert expected > 0.0
        assert result.p_value > 0.0
        assert result.p_value == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("case", [DSR_PAPER_EXAMPLE], ids=lambda c: c.name)
    def test_dsr_paper_reference(self, case):
        """Test DSR calculation against the value the paper publishes.

        Reference: numerical example after Exhibit 5 (version of 7 March 2026).
        This is the only DSR value the paper states.
        """
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=case.observed_sharpe,
            n_trials=case.n_trials,
            variance_trials=case.variance_trials,
            n_samples=case.n_samples,
            skewness=case.skewness,
            excess_kurtosis=case.kurtosis - 3.0,
            autocorrelation=case.autocorrelation,
        )

        # Validate DSR (probability)
        dsr_error = abs(result.probability - case.expected_dsr)
        assert dsr_error < case.tolerance, (
            f"{case.name}: DSR validation failed\n"
            f"Expected DSR: {case.expected_dsr:.6f}\n"
            f"Got DSR: {result.probability:.6f}\n"
            f"Error: {dsr_error:.6f} ({dsr_error / case.expected_dsr * 100:.2f}%)\n"
            f"Tolerance: {case.tolerance}\n"
            f"Reference: {case.reference}"
        )

        # Validate E[max]
        max_sharpe_error = abs(result.expected_max_sharpe - case.expected_max_sharpe)
        assert max_sharpe_error < case.tolerance, (
            f"{case.name}: Expected max Sharpe validation failed\n"
            f"Expected: {case.expected_max_sharpe:.6f}\n"
            f"Got: {result.expected_max_sharpe:.6f}\n"
            f"Error: {max_sharpe_error:.6f}\n"
            f"Reference: {case.reference}"
        )

    @pytest.mark.parametrize(
        "case", [DSR_FEW_TRIALS, DSR_MANY_TRIALS, DSR_NORMAL], ids=lambda c: c.name
    )
    def test_dsr_computed_cases(self, case):
        """Test DSR on cases the paper does not state.

        Expected values come from Equations 28 and 30, evaluated outside the library.
        """
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=case.observed_sharpe,
            n_trials=case.n_trials,
            variance_trials=case.variance_trials,  # Empirical variance across K trials
            n_samples=case.n_samples,
            skewness=case.skewness,
            excess_kurtosis=case.kurtosis - 3.0,
            autocorrelation=case.autocorrelation,
        )

        dsr_error = abs(result.probability - case.expected_dsr)
        assert dsr_error < case.tolerance, (
            f"{case.name}: DSR validation failed\n"
            f"Expected DSR: {case.expected_dsr:.6f}\n"
            f"Got DSR: {result.probability:.6f}\n"
            f"Error: {dsr_error:.6f} ({dsr_error / case.expected_dsr * 100:.2f}%)\n"
            f"Tolerance: {case.tolerance}\n"
            f"Reference: {case.reference}"
        )

        max_sharpe_error = abs(result.expected_max_sharpe - case.expected_max_sharpe)
        assert max_sharpe_error < case.tolerance, (
            f"{case.name}: Expected max Sharpe validation failed\n"
            f"Expected: {case.expected_max_sharpe:.6f}\n"
            f"Got: {result.expected_max_sharpe:.6f}\n"
            f"Error: {max_sharpe_error:.6f} ({max_sharpe_error / case.expected_max_sharpe * 100:.2f}%)\n"
            f"Tolerance: {case.tolerance}\n"
            f"Reference: {case.reference}"
        )

    def test_dsr_components_structure(self):
        """Verify DSR returns all expected components via DSRResult."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=1.5,
            n_trials=100,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
            skewness=-1.0,
            excess_kurtosis=2.0,  # Pearson 5.0 → Fisher 2.0
        )

        # Check result is DSRResult with all expected attributes
        assert isinstance(result, DSRResult)
        assert hasattr(result, "probability")
        assert hasattr(result, "expected_max_sharpe")
        assert hasattr(result, "z_score")
        assert hasattr(result, "p_value")

        # Check types and ranges
        assert isinstance(result.probability, float)
        assert isinstance(result.expected_max_sharpe, float)
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)

        assert 0 <= result.probability <= 1
        assert 0 <= result.p_value <= 1

    def test_dsr_single_trial(self):
        """Edge case: DSR with single trial (no deflation) = PSR."""
        observed_sharpe = 1.5

        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=1,
            variance_trials=0.0,  # Single trial has no variance
            n_samples=252,  # One year of daily returns
        )

        # With single trial (PSR), expected_max should be 0 (no selection bias)
        assert result.expected_max_sharpe == 0.0
        # variance_trials is 0 for single trial
        assert result.variance_trials == 0.0

    def test_dsr_zero_sharpe(self):
        """Edge case: DSR with zero Sharpe ratio."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=0.0,
            n_trials=100,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )

        # Zero Sharpe should be heavily deflated (z-score should be negative)
        assert result.z_score < 0
        # Expected max under null should be positive
        assert result.expected_max_sharpe > 0

    def test_dsr_negative_sharpe(self):
        """Edge case: DSR with negative Sharpe ratio."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=-1.0,
            n_trials=100,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )

        # Negative Sharpe should result in strongly negative z-score
        assert result.z_score < -1.0
        # P-value should be very high (probability > benchmark is low)
        # Note: p_value = 1 - probability, so for bad strategies it's high
        assert result.probability < 0.01

    def test_dsr_extreme_trials(self):
        """Edge case: DSR with very large number of trials."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=2.0,
            n_trials=10000,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )

        # With many trials, expected max should be high
        assert result.expected_max_sharpe > 1.5
        # z-score should be less than observed Sharpe (deflation)
        assert result.z_score < 2.0

    def test_dsr_multiple_trials_ignores_selected_strategy_moments(self):
        """With K>1, skewness, kurtosis and autocorrelation do not enter the z-score.

        Equation 30 gives the standard deviation of the maximum Sharpe ratio as
        √V[{SR_k}] × √V[max_k{X_k}]. Neither factor depends on the moments of
        the selected strategy's returns, so the DSR for given K and V[{SR_k}]
        is the same for normal and non-normal returns.
        """
        observed_sharpe = 1.5
        n_trials = 50

        def dsr(**moments):
            return deflated_sharpe_ratio_from_statistics(
                observed_sharpe=observed_sharpe,
                n_trials=n_trials,
                variance_trials=1.0,
                n_samples=252,
                **moments,
            )

        result_normal = dsr(skewness=0.0, excess_kurtosis=0.0)
        expected_z = (observed_sharpe - result_normal.expected_max_sharpe) / (
            1.0 * VARIANCE_RESCALING_FACTORS[n_trials]
        )
        # The exhibit prints five decimals
        assert result_normal.z_score == pytest.approx(expected_z, rel=2e-5)

        for moments in (
            {"skewness": -2.0},
            {"excess_kurtosis": 7.0},  # Pearson 10.0 → Fisher 7.0
            {"autocorrelation": 0.3},
        ):
            result = dsr(**moments)
            assert result.expected_max_sharpe == result_normal.expected_max_sharpe, moments
            assert result.z_score == result_normal.z_score, moments

    def test_identical_trials_reduce_to_single_strategy(self):
        """Trials with equal Sharpe ratios leave nothing to deflate.

        Their dispersion is zero, so Equation 30 alone would give a zero standard
        deviation and a probability of exactly 0 or 1. The result must equal the
        single-strategy PSR of the same returns.
        """
        returns = np.random.default_rng(0).normal(0.001, 0.01, 500)

        single = deflated_sharpe_ratio(returns)
        repeated = deflated_sharpe_ratio([returns, returns.copy(), returns.copy()])

        assert repeated.n_trials_raw == 3
        assert repeated.variance_trials == 0.0
        assert repeated.expected_max_sharpe == 0.0
        assert repeated.z_score == pytest.approx(single.z_score, rel=1e-12)
        assert repeated.probability == pytest.approx(single.probability, rel=1e-12)

    def test_rounding_noise_in_trial_variance_is_not_dispersion(self):
        """Equal Sharpe ratios whose variance is not exactly zero still give the PSR.

        np.var([0.1, 0.1, 0.1], ddof=1) is about 3e-34, because the mean of the
        three values is not 0.1. Used in Equation 30, that variance gives a
        standard deviation of 1e-17 and a probability of exactly 0 or 1.
        """
        variance_trials = float(np.var([0.1, 0.1, 0.1], ddof=1))
        assert 0.0 < variance_trials < 1e-30

        single = deflated_sharpe_ratio_from_statistics(observed_sharpe=0.1, n_samples=252)
        repeated = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=0.1,
            n_samples=252,
            n_trials=3,
            variance_trials=variance_trials,
        )

        assert 0.9 < single.probability < 0.99
        assert repeated.probability == pytest.approx(single.probability, rel=1e-12)

    def test_dsr_increasing_trials(self):
        """Verify DSR decreases as number of trials increases."""
        observed_sharpe = 1.5

        result_few = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=5,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )
        result_many = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=500,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )

        # More trials should lead to more deflation (lower z-score)
        assert result_many.z_score < result_few.z_score

    def test_dsr_p_value_consistency(self):
        """Verify p-value is consistent with DSR statistic."""
        result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=1.5,
            n_trials=100,
            variance_trials=1.0,  # Standard normal
            n_samples=252,  # One year of daily returns
        )

        # P-value should match DSR z-score via standard normal CDF
        expected_p_value = 1 - norm.cdf(result.z_score)
        assert abs(result.p_value - expected_p_value) < 1e-10


# =============================================================================
# DSR PARAMETER SENSITIVITY TESTS
# =============================================================================


class TestDSRParameterSensitivity:
    """Test DSR sensitivity to parameters and validate internal calculations.

    These tests validate:
    1. E[max{Z}] calculation (expected maximum of K standard normals)
    2. Parameter sensitivity (n_trials, skewness, kurtosis effects)
    3. Monotonicity properties
    4. Edge cases and stability
    """

    def test_expected_max_increases_with_trials(self):
        """E[max{SR}] should increase with more trials (selection bias grows)."""
        test_points = [2, 5, 10, 20, 50, 100]
        variance_trials = 1.0

        expected_max_values = []
        for n_trials in test_points:
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=0.0,
                n_trials=n_trials,
                variance_trials=variance_trials,
                n_samples=252,
            )
            expected_max_values.append(result.expected_max_sharpe)

        # Verify increasing pattern (more trials = higher expected max)
        for i in range(len(expected_max_values) - 1):
            assert expected_max_values[i] < expected_max_values[i + 1], (
                f"E[max{{SR}}] should increase with more trials: "
                f"K={test_points[i]} E[max]={expected_max_values[i]:.5f}, "
                f"K={test_points[i + 1]} E[max]={expected_max_values[i + 1]:.5f}"
            )

    def test_sample_size_changes_z_score_for_one_trial_only(self):
        """Sample size enters the z-score through the estimator variance (K=1).

        For one trial the denominator is the standard deviation of the Sharpe
        ratio estimator, which falls as 1/√T, so |z| grows with T. For K>1 the
        denominator is √V[{SR_k}] × √V[max_k{X_k}] (Equation 30); with
        V[{SR_k}] held fixed, T does not change the z-score.
        """
        test_sample_sizes = [50, 100, 252, 500, 1000]
        observed_sharpe = 1.5  # Use positive Sharpe

        single_trial = [
            abs(
                deflated_sharpe_ratio_from_statistics(
                    observed_sharpe=observed_sharpe,
                    n_trials=1,
                    variance_trials=0.0,
                    n_samples=n_samples,
                ).z_score
            )
            for n_samples in test_sample_sizes
        ]
        for i in range(len(single_trial) - 1):
            assert single_trial[i] < single_trial[i + 1], (
                f"|z-score| should increase with more samples for one trial: "
                f"T={test_sample_sizes[i]} |z|={single_trial[i]:.5f}, "
                f"T={test_sample_sizes[i + 1]} |z|={single_trial[i + 1]:.5f}"
            )

        ten_trials = {
            deflated_sharpe_ratio_from_statistics(
                observed_sharpe=observed_sharpe,
                n_trials=10,
                variance_trials=1.0,
                n_samples=n_samples,
            ).z_score
            for n_samples in test_sample_sizes
        }
        assert len(ten_trials) == 1, f"z-score should not depend on T for K=10: {ten_trials}"

    def test_expected_max_formula_accuracy(self):
        """Validate E[max{Z}] against known values from extreme value theory."""
        # For K standard normals, E[max{Z}] ≈ Φ^(-1)(1 - 1/K) for large K
        # For small K, extreme value correction is needed

        test_cases = [
            (2, 0.56),  # Two normals: expect ~0.56
            (5, 1.16),  # Five normals: expect ~1.16
            (10, 1.54),  # Ten normals: expect ~1.54
            (100, 2.51),  # 100 normals: expect ~2.51
        ]

        for n_trials, expected_e_max in test_cases:
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=0.0,
                n_trials=n_trials,
                variance_trials=1.0,
                n_samples=252,
            )

            # E[max{SR}] = sqrt(variance_trials) * E[max{Z}]
            # With variance_trials=1.0, E[max{SR}] = E[max{Z}]
            e_max_z = result.expected_max_sharpe

            # Allow 10% tolerance (extreme value approximations vary)
            relative_error = abs(e_max_z - expected_e_max) / expected_e_max
            assert relative_error < 0.10, (
                f"E[max{{Z}}] approximation failed for K={n_trials}\n"
                f"Expected: {expected_e_max:.5f}\n"
                f"Got: {e_max_z:.5f}\n"
                f"Relative error: {relative_error:.2%}"
            )

    @pytest.mark.parametrize(
        "skewness,kurtosis,expected_effect",
        [
            (0.0, 3.0, "baseline"),  # Normal - baseline
            (0.0, 5.0, "less_extreme_z"),  # High kurtosis increases variance
            (0.0, 8.0, "less_extreme_z"),  # Very high kurtosis increases variance
        ],
    )
    def test_non_normality_affects_uncertainty(self, skewness, kurtosis, expected_effect):
        """Non-normal returns change the one-trial z-score through Equation 5.

        V[SR] = (1/T) * (a - b·γ₃·SR₀ + c·(γ₄-1)/4·SR₀²)

        Higher kurtosis (γ₄) increases the variance when SR₀ ≠ 0, which moves the
        z-score towards 0. The benchmark is therefore non-zero here. With more
        than one trial these moments do not enter the denominator; see
        TestDSRValidation.test_dsr_multiple_trials_ignores_selected_strategy_moments.
        """
        n_samples = 252
        observed_sharpe = 1.0
        benchmark_sharpe = 0.5  # Non-zero to see skew/kurt effects

        # Normal case (baseline)
        result_normal = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=1,
            variance_trials=0.0,
            n_samples=n_samples,
            benchmark_sharpe=benchmark_sharpe,
            skewness=0.0,
            excess_kurtosis=0.0,
        )

        # Non-normal case
        result_nonnormal = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=observed_sharpe,
            n_trials=1,
            variance_trials=0.0,
            n_samples=n_samples,
            benchmark_sharpe=benchmark_sharpe,
            skewness=skewness,
            excess_kurtosis=kurtosis - 3.0,  # Convert Pearson to Fisher
        )

        if expected_effect == "baseline":
            # Should be identical
            assert result_nonnormal.z_score == result_normal.z_score
        elif expected_effect == "less_extreme_z":
            # Higher kurtosis increases variance -> less extreme z-score (smaller magnitude)
            assert abs(result_nonnormal.z_score) < abs(result_normal.z_score), (
                f"Higher kurtosis should lead to less extreme z-score (more uncertainty)\n"
                f"Normal |z|: {abs(result_normal.z_score):.5f}\n"
                f"Non-normal |z|: {abs(result_nonnormal.z_score):.5f}\n"
                f"Skew={skewness}, Kurt={kurtosis}"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDSRIntegration:
    """Integration tests combining DSR with PSR and MinTRL concepts."""

    def test_dsr_psr_relationship(self):
        """One trial gives the PSR; more trials deflate it."""
        inputs = {
            "observed_sharpe": PAPER_SHARPE,
            "n_samples": 24,
            "skewness": -2.448,
            "excess_kurtosis": 10.164 - 3.0,  # Convert Pearson to Fisher
            "autocorrelation": 0.2,
        }

        single = deflated_sharpe_ratio_from_statistics(n_trials=1, variance_trials=0.0, **inputs)
        multiple = deflated_sharpe_ratio_from_statistics(n_trials=10, variance_trials=1.0, **inputs)

        # One trial: no selection bias, so the threshold stays at the benchmark
        assert single.expected_max_sharpe == 0.0
        assert single.probability == _psr(
            sharpe_ratio=PAPER_SHARPE,
            skewness=-2.448,
            excess_kurtosis=10.164 - 3.0,
            n_samples=24,
            autocorrelation=0.2,
        )

        # Ten trials: the threshold rises to E[max{SR}] and the probability falls
        assert multiple.expected_max_sharpe > 0
        assert multiple.probability < single.probability

    def test_extreme_value_theory_convergence(self):
        """Verify extreme value theory approximations converge for large K."""
        # Test that expected max converges to theoretical limit
        results = []
        for n_trials in [10, 50, 100, 500, 1000]:
            result = deflated_sharpe_ratio_from_statistics(
                observed_sharpe=1.0,
                n_trials=n_trials,
                variance_trials=1.0,  # Standard normal
                n_samples=252,  # One year of daily returns
            )
            results.append((n_trials, result.expected_max_sharpe))

        # Expected max should increase with K but at decreasing rate
        for i in range(len(results) - 1):
            k1, max1 = results[i]
            k2, max2 = results[i + 1]

            # Should increase
            assert max2 > max1

            # Rate of increase should decrease
            if i > 0:
                k0, max0 = results[i - 1]
                rate1 = (max1 - max0) / (k1 - k0)
                rate2 = (max2 - max1) / (k2 - k1)
                assert rate2 < rate1, "Expected decreasing rate of growth"

    def test_non_normality_consistent_across_functions(self):
        """Fat tails and negative skew lower PSR and lengthen MinTRL.

        With SR₀=0 the skewness and kurtosis terms of Equation 5 vanish, so the
        benchmark is non-zero here.
        """
        sharpe_ratio = 0.5
        skewness = -1.5
        excess_kurtosis = 3.0  # Fisher convention (Pearson 6.0 - 3)
        n_samples = 50
        sharpe_star = 0.1

        psr = _psr(
            sharpe_ratio=sharpe_ratio,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            n_samples=n_samples,
            sharpe_star=sharpe_star,
        )
        psr_normal = _psr(
            sharpe_ratio=sharpe_ratio,
            skewness=0.0,
            excess_kurtosis=0.0,
            n_samples=n_samples,
            sharpe_star=sharpe_star,
        )
        assert psr < psr_normal

        mintrl = _min_trl(
            sharpe_ratio=sharpe_ratio,
            sharpe_star=sharpe_star,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            alpha=0.05,
        )
        mintrl_normal = _min_trl(
            sharpe_ratio=sharpe_ratio,
            sharpe_star=sharpe_star,
            skewness=0.0,
            excess_kurtosis=0.0,
            alpha=0.05,
        )
        assert mintrl > mintrl_normal
