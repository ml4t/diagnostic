"""Comprehensive stationarity analysis combining ADF, KPSS, and PP tests.

This module provides unified stationarity analysis by combining multiple
tests with consensus-based interpretation.

Key Concept:
    Different tests have different null hypotheses:
    - ADF/PP: H0 = unit root (non-stationary), reject => stationary
    - KPSS: H0 = stationary, reject => non-stationary

    Strong evidence requires agreement between tests with opposite hypotheses.

Consensus Logic:
    - Strong stationary: All tests agree (ADF/PP reject, KPSS fails to reject)
    - Likely stationary: 2/3 tests agree on stationarity
    - Inconclusive: Tests evenly split (e.g., ADF/PP reject, KPSS rejects)
    - Likely non-stationary: 2/3 tests agree on non-stationarity
    - Strong non-stationary: All tests agree (ADF/PP fail to reject, KPSS rejects)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.evaluation.stationarity.augmented_dickey_fuller import ADFResult, adf_test
from ml4t.diagnostic.evaluation.stationarity.kpss_test import KPSSResult, kpss_test
from ml4t.diagnostic.evaluation.stationarity.phillips_perron import (
    PPResult,
    _check_arch_available,
    pp_test,
)
from ml4t.diagnostic.logging import get_logger

logger = get_logger(__name__)


class StationarityAnalysisResult:
    """Comprehensive stationarity analysis combining ADF, KPSS, and PP tests.

    Provides unified view of multiple stationarity tests with consensus interpretation.

    Attributes:
        adf_result: ADF test result (None if test not run or failed)
        kpss_result: KPSS test result (None if test not run or failed)
        pp_result: PP test result (None if test not run or failed)
        consensus: Consensus interpretation of stationarity
        summary_df: DataFrame with all test results in tabular form
        agreement_score: Agreement between tests (0.0 to 1.0)
        alpha: Significance level used for all tests
        n_tests_run: Number of tests successfully completed
    """

    def __init__(
        self,
        adf_result: ADFResult | None = None,
        kpss_result: KPSSResult | None = None,
        pp_result: PPResult | None = None,
        alpha: float = 0.05,
    ):
        """Initialize stationarity analysis result.

        Args:
            adf_result: ADF test result
            kpss_result: KPSS test result
            pp_result: PP test result
            alpha: Significance level used
        """
        self.adf_result = adf_result
        self.kpss_result = kpss_result
        self.pp_result = pp_result
        self.alpha = alpha

        # Count number of tests run
        self.n_tests_run = sum(
            [
                adf_result is not None,
                kpss_result is not None,
                pp_result is not None,
            ]
        )

        # Calculate consensus and agreement
        self.consensus = self._calculate_consensus()
        self.agreement_score = self._calculate_agreement()

        # Create summary DataFrame
        self.summary_df = self._create_summary_df()

    def _calculate_consensus(
        self,
    ) -> Literal[
        "strong_stationary",
        "likely_stationary",
        "inconclusive",
        "likely_nonstationary",
        "strong_nonstationary",
    ]:
        """Calculate consensus interpretation from all tests.

        Consensus Logic:
            - Strong stationary: All tests agree stationary
            - Likely stationary: 2/3 tests agree stationary
            - Inconclusive: Tests evenly split or only 2 tests with disagreement
            - Likely non-stationary: 2/3 tests agree non-stationary
            - Strong non-stationary: All tests agree non-stationary

        Returns:
            Consensus interpretation
        """
        # Collect stationarity results
        results = []
        if self.adf_result is not None:
            results.append(self.adf_result.is_stationary)
        if self.kpss_result is not None:
            results.append(self.kpss_result.is_stationary)
        if self.pp_result is not None:
            results.append(self.pp_result.is_stationary)

        if len(results) == 0:
            return "inconclusive"

        # Count votes
        stationary_votes = sum(results)

        # Determine consensus
        if len(results) == 3:
            if stationary_votes == 3:
                return "strong_stationary"
            elif stationary_votes == 2:
                return "likely_stationary"
            elif stationary_votes == 1:
                return "likely_nonstationary"
            else:  # stationary_votes == 0
                return "strong_nonstationary"
        elif len(results) == 2:
            if stationary_votes == 2:
                return "likely_stationary"
            elif stationary_votes == 0:
                return "likely_nonstationary"
            else:  # stationary_votes == 1 (disagreement)
                return "inconclusive"
        else:  # len(results) == 1
            # Single test - use its result but label as "likely" not "strong"
            if results[0]:
                return "likely_stationary"
            else:
                return "likely_nonstationary"

    def _calculate_agreement(self) -> float:
        """Calculate agreement score between tests.

        Agreement score ranges from 0.0 (complete disagreement) to 1.0 (complete agreement).

        For 3 tests:
            - All agree: 1.0
            - 2 agree: 0.67
            - None agree (all different): 0.33

        For 2 tests:
            - Both agree: 1.0
            - Disagree: 0.0

        For 1 test:
            - Always 1.0 (no disagreement possible)

        Returns:
            Agreement score between 0.0 and 1.0
        """
        # Collect stationarity results
        results = []
        if self.adf_result is not None:
            results.append(self.adf_result.is_stationary)
        if self.kpss_result is not None:
            results.append(self.kpss_result.is_stationary)
        if self.pp_result is not None:
            results.append(self.pp_result.is_stationary)

        if len(results) <= 1:
            return 1.0

        # Count how many agree with majority
        stationary_votes = sum(results)
        majority_count = max(stationary_votes, len(results) - stationary_votes)

        # Agreement score = proportion agreeing with majority
        return majority_count / len(results)

    def _create_summary_df(self) -> pd.DataFrame:
        """Create summary DataFrame with all test results.

        Returns:
            DataFrame with columns: test_name, test_statistic, p_value,
                                   is_stationary, conclusion, alpha
        """
        rows = []

        # Add ADF results
        if self.adf_result is not None:
            rows.append(
                {
                    "test_name": "ADF",
                    "test_statistic": self.adf_result.test_statistic,
                    "p_value": self.adf_result.p_value,
                    "is_stationary": self.adf_result.is_stationary,
                    "conclusion": "Stationary"
                    if self.adf_result.is_stationary
                    else "Non-stationary",
                    "alpha": self.alpha,
                }
            )

        # Add KPSS results
        if self.kpss_result is not None:
            rows.append(
                {
                    "test_name": "KPSS",
                    "test_statistic": self.kpss_result.test_statistic,
                    "p_value": self.kpss_result.p_value,
                    "is_stationary": self.kpss_result.is_stationary,
                    "conclusion": "Stationary"
                    if self.kpss_result.is_stationary
                    else "Non-stationary",
                    "alpha": self.alpha,
                }
            )

        # Add PP results
        if self.pp_result is not None:
            rows.append(
                {
                    "test_name": "PP",
                    "test_statistic": self.pp_result.test_statistic,
                    "p_value": self.pp_result.p_value,
                    "is_stationary": self.pp_result.is_stationary,
                    "conclusion": "Stationary"
                    if self.pp_result.is_stationary
                    else "Non-stationary",
                    "alpha": self.alpha,
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StationarityAnalysisResult("
            f"consensus={self.consensus}, "
            f"agreement={self.agreement_score:.2f}, "
            f"n_tests={self.n_tests_run})"
        )

    def summary(self) -> str:
        """Human-readable summary of comprehensive stationarity analysis."""
        lines = [
            "Comprehensive Stationarity Analysis",
            "=" * 60,
            f"Tests Run: {self.n_tests_run} | Significance Level: {self.alpha}",
            "",
        ]

        # Individual test results
        for name, res in [
            ("ADF Test", self.adf_result),
            ("KPSS Test", self.kpss_result),
            ("PP Test", self.pp_result),
        ]:
            if res is not None:
                status = "Stationary" if res.is_stationary else "Non-stationary"
                lines.append(
                    f"{name}: {status} (stat={res.test_statistic:.4f}, p={res.p_value:.4f})"
                )

        lines.append(
            f"\nAgreement Score: {self.agreement_score:.2f} ({int(self.agreement_score * 100)}%)"
        )

        consensus_labels = {
            "strong_stationary": "STRONG STATIONARY (all agree)",
            "likely_stationary": "LIKELY STATIONARY (majority)",
            "inconclusive": "INCONCLUSIVE (tests disagree)",
            "likely_nonstationary": "LIKELY NON-STATIONARY (majority)",
            "strong_nonstationary": "STRONG NON-STATIONARY (all agree)",
        }
        lines.append(f"Consensus: {consensus_labels[self.consensus]}")

        # Interpretation guidance matching test expectations
        lines.append("\nInterpretation:")
        if self.consensus == "strong_stationary":
            lines.append("  - Series exhibits strong evidence of stationarity")
            lines.append("  - Safe to use in models requiring stationarity")
        elif self.consensus == "likely_stationary":
            lines.append("  - Series likely stationary, but some uncertainty")
        elif self.consensus == "inconclusive":
            lines.append("  - Tests provide conflicting evidence")
            lines.append("  - Consider differencing or detrending")
        elif self.consensus == "likely_nonstationary":
            lines.append("  - Series likely has unit root")
            lines.append("  - Apply differencing before modeling")
        else:  # strong_nonstationary
            lines.append("  - Series exhibits strong evidence of unit root")
            lines.append("  - Requires differencing or cointegration approach")

        return "\n".join(lines)


def analyze_stationarity(
    data: pd.Series | np.ndarray,
    alpha: float = 0.05,
    include_tests: list[Literal["adf", "kpss", "pp"]] | None = None,
    **test_kwargs,
) -> StationarityAnalysisResult:
    """Perform comprehensive stationarity analysis with multiple tests.

    Runs ADF, KPSS, and PP tests (or subset) and provides consensus interpretation
    of stationarity. This is the recommended way to assess stationarity robustly.

    Key Concept:
        Different tests have different null hypotheses:
        - ADF/PP: H0 = unit root (non-stationary), reject => stationary
        - KPSS: H0 = stationary, reject => non-stationary

        Strong evidence requires agreement between tests with opposite hypotheses.

    Consensus Logic:
        - Strong stationary: All tests agree (ADF/PP reject, KPSS fails to reject)
        - Likely stationary: 2/3 tests agree on stationarity
        - Inconclusive: Tests evenly split (e.g., ADF/PP reject, KPSS rejects)
        - Likely non-stationary: 2/3 tests agree on non-stationarity
        - Strong non-stationary: All tests agree (ADF/PP fail to reject, KPSS rejects)

    Args:
        data: Time series data to test (1D array or Series)
        alpha: Significance level for all tests (default: 0.05)
        include_tests: List of tests to run. If None, runs all available tests.
                      Options: ["adf", "kpss", "pp"]. PP requires arch package.
        **test_kwargs: Additional keyword arguments passed to individual tests.
                      Common options:
                      - regression: 'c', 'ct', 'n' (for ADF/KPSS/PP)
                      - maxlag: int or None (for ADF)
                      - autolag: 'AIC', 'BIC', 't-stat' or None (for ADF)
                      - nlags: int, 'auto', or 'legacy' (for KPSS)
                      - lags: int or None (for PP)

    Returns:
        StationarityAnalysisResult with all test results, consensus, and summary

    Raises:
        ValidationError: If data is invalid (empty, wrong shape, etc.)
        ComputationError: If all tests fail to run

    Example:
        >>> import numpy as np
        >>> from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity
        >>> white_noise = np.random.randn(1000)
        >>> result = analyze_stationarity(white_noise)
        >>> print(f"Consensus: {result.consensus}, Agreement: {result.agreement_score:.2%}")
        >>> # With custom parameters
        >>> result = analyze_stationarity(white_noise, regression="ct", include_tests=["adf", "kpss"])

    Notes:
        - White noise: strong_stationary; Random walk: strong_nonstationary
        - PP test requires arch package (auto-skipped if unavailable)
        - Individual results: adf_result, kpss_result, pp_result; tabular: summary_df
    """
    # Validate data first
    if data is None:
        raise ValidationError("Data cannot be None", context={"function": "analyze_stationarity"})

    # Convert to numpy array for validation
    if isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValidationError(
            f"Data must be pandas Series or numpy array, got {type(data)}",
            context={"function": "analyze_stationarity", "data_type": type(data).__name__},
        )

    if arr.ndim != 1:
        raise ValidationError(
            f"Data must be 1-dimensional, got {arr.ndim}D",
            context={"function": "analyze_stationarity", "shape": arr.shape},
        )

    if len(arr) == 0:
        raise ValidationError(
            "Data cannot be empty", context={"function": "analyze_stationarity", "length": 0}
        )

    # Determine which tests to run
    if include_tests is None:
        # Run all available tests
        tests_to_run = ["adf", "kpss"]
        if _check_arch_available():
            tests_to_run.append("pp")
        else:
            logger.info(
                "PP test not available (arch package not installed), running ADF and KPSS only"
            )
    else:
        # Validate test names
        valid_tests: set[str] = {"adf", "kpss", "pp"}
        provided_tests: set[str] = set(include_tests)
        invalid = provided_tests - valid_tests
        if invalid:
            raise ValidationError(
                f"Invalid test names: {invalid}. Valid options: {valid_tests}",
                context={"function": "analyze_stationarity", "include_tests": include_tests},
            )

        tests_to_run = list(include_tests)

        # Warn if PP requested but not available
        if "pp" in tests_to_run and not _check_arch_available():
            logger.warning(
                "PP test requested but arch package not installed - skipping PP test. "
                "Install with: pip install arch or pip install ml4t-diagnostic[advanced]"
            )
            tests_to_run = [t for t in tests_to_run if t != "pp"]

    if len(tests_to_run) == 0:
        raise ValidationError(
            "No valid tests to run",
            context={"function": "analyze_stationarity", "include_tests": include_tests},
        )

    logger.info(
        "Running comprehensive stationarity analysis",
        n_obs=len(arr),
        tests=tests_to_run,
        alpha=alpha,
    )

    # Run tests and collect results
    adf_result = None
    kpss_result = None
    pp_result = None
    failed_tests = []

    # Define test configurations: (test_name, test_func, param_keys)
    test_configs = {
        "adf": (adf_test, ["maxlag", "regression", "autolag"]),
        "kpss": (kpss_test, ["regression", "nlags"]),
        "pp": (pp_test, ["lags", "regression", "test_type"]),
    }

    for test_name in tests_to_run:
        test_func, param_keys = test_configs[test_name]
        params = {k: test_kwargs[k] for k in param_keys if k in test_kwargs}

        # KPSS only supports 'c' and 'ct' regression
        if (
            test_name == "kpss"
            and "regression" in params
            and params["regression"] not in ("c", "ct")
        ):
            logger.warning(f"KPSS does not support regression='{params['regression']}', using 'c'")
            params.pop("regression")

        try:
            result = test_func(data, **params)
            logger.info(f"{test_name.upper()} test completed", stationary=result.is_stationary)
            if test_name == "adf":
                adf_result = result
            elif test_name == "kpss":
                kpss_result = result
            else:
                pp_result = result
        except Exception as e:
            logger.error(f"{test_name.upper()} test failed", error=str(e))
            failed_tests.append((test_name.upper(), str(e)))

    # Check if at least one test succeeded
    n_succeeded = sum([adf_result is not None, kpss_result is not None, pp_result is not None])

    if n_succeeded == 0:
        # All tests failed
        error_msg = "All stationarity tests failed:\n"
        for test_name, error in failed_tests:
            error_msg += f"  - {test_name}: {error}\n"
        raise ComputationError(
            error_msg.strip(),
            context={
                "function": "analyze_stationarity",
                "n_obs": len(arr),
                "tests_attempted": tests_to_run,
            },
        )

    # Log warnings for failed tests
    if failed_tests:
        logger.warning(
            f"{len(failed_tests)} test(s) failed but {n_succeeded} succeeded",
            failed_tests=[t[0] for t in failed_tests],
        )

    # Create analysis result
    result = StationarityAnalysisResult(
        adf_result=adf_result,
        kpss_result=kpss_result,
        pp_result=pp_result,
        alpha=alpha,
    )

    logger.info(
        "Stationarity analysis completed",
        n_tests_run=result.n_tests_run,
        consensus=result.consensus,
        agreement=result.agreement_score,
    )

    return result
