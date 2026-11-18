"""Reference values for the Sharpe ratio inference tests.

Published values come from López de Prado, M., Lipton, A., & Zoonekynd, V.,
"How to Use the Sharpe Ratio", ADIA Lab Research Paper Series, No. 19, version
of 7 March 2026. Equation and exhibit numbers refer to that version.

The paper's worked example is a two-year track record of monthly returns with
(mu, sigma, gamma_3, gamma_4, rho, T) = (0.036%, 0.079%, -2.448, 10.164, 0.2, 24).
The paper prints the Sharpe ratio as 0.456; its results follow from the
unrounded 0.036/0.079.

Cases marked "computed" are not in the paper. Their expected values were
computed from the cited equations outside the library.

Sharpe ratios are not annualized. Kurtosis is Pearson kurtosis (3 for normal
returns).
"""

from typing import NamedTuple

# Sharpe ratio of the paper's worked example
PAPER_SHARPE = 0.036 / 0.079


class PSRTestCase(NamedTuple):
    """Test case for the Probabilistic Sharpe Ratio.

    Attributes:
        name: Descriptive name of test case
        sharpe_ratio: Observed Sharpe ratio (SR_hat*)
        skewness: Sample skewness (gamma_3)
        kurtosis: Sample kurtosis (gamma_4, 3.0 for normal)
        n_samples: Number of observations (T)
        sharpe_star: Benchmark SR (SR_0) for null hypothesis H0: SR <= SR_0
        expected_psr: Expected PSR value
        reference: Where the expected value comes from
        tolerance: Acceptable numerical error for validation
        autocorrelation: First-order autocorrelation of returns (rho)
    """

    name: str
    sharpe_ratio: float
    skewness: float
    kurtosis: float
    n_samples: int
    sharpe_star: float
    expected_psr: float
    reference: str
    tolerance: float = 0.0005
    autocorrelation: float = 0.0


class DSRTestCase(NamedTuple):
    """Test case for Deflated Sharpe Ratio.

    Attributes:
        name: Descriptive name of test case
        observed_sharpe: Observed Sharpe ratio (SR_hat*)
        n_trials: Number of independent trials (K)
        variance_trials: Variance of Sharpe ratios across trials (V[{SR_k*}])
        skewness: Sample skewness (gamma_3)
        kurtosis: Sample kurtosis (gamma_4)
        n_samples: Number of observations per trial (T)
        expected_dsr: Expected DSR value (PSR adjusted for multiple testing)
        expected_max_sharpe: Expected maximum Sharpe ratio E[max{SR_k}]
        reference: Citation to paper location
        tolerance: Acceptable numerical error for validation
        autocorrelation: First-order autocorrelation of returns (rho)
    """

    name: str
    observed_sharpe: float
    n_trials: int
    variance_trials: float
    skewness: float
    kurtosis: float
    n_samples: int
    expected_dsr: float
    expected_max_sharpe: float
    reference: str
    tolerance: float = 0.001
    autocorrelation: float = 0.0


class MinTRLTestCase(NamedTuple):
    """Test case for the Minimum Track Record Length.

    Attributes:
        name: Descriptive name of test case
        sharpe_ratio: Observed Sharpe ratio (SR_hat*)
        sharpe_star: Benchmark SR (SR_0) for null hypothesis
        skewness: Sample skewness (gamma_3)
        kurtosis: Sample kurtosis (gamma_4)
        alpha: Significance level (false positive rate)
        expected_mintrl: Value of Equation 11, a real number of observations.
            compute_min_trl adds one observation and rounds up, so it returns
            ceil(1 + expected_mintrl).
        reference: Where the expected value comes from
        autocorrelation: First-order autocorrelation of returns (rho)
    """

    name: str
    sharpe_ratio: float
    sharpe_star: float
    skewness: float
    kurtosis: float
    alpha: float
    expected_mintrl: float
    reference: str
    autocorrelation: float = 0.0


# =============================================================================
# PSR TEST CASES
# =============================================================================

PSR_PAPER_EXAMPLE = PSRTestCase(
    name="Paper example, SR_0=0",
    sharpe_ratio=PAPER_SHARPE,
    skewness=-2.448,
    kurtosis=10.164,
    n_samples=24,
    sharpe_star=0.0,
    expected_psr=0.966,
    reference="Numerical example after Equation 9: PSR = 0.966 under SR_0 = 0",
    autocorrelation=0.2,
)

PSR_PAPER_EXAMPLE_BENCHMARK = PSRTestCase(
    name="Paper example, SR_0=0.1",
    sharpe_ratio=PAPER_SHARPE,
    skewness=-2.448,
    kurtosis=10.164,
    n_samples=24,
    sharpe_star=0.1,
    expected_psr=0.900,
    reference="Numerical example after Equation 9: PSR = 0.900 under SR_0 = 0.1",
    autocorrelation=0.2,
)

# The paper's example before serial correlation was added to it. The authors'
# companion code (github.com/zoonek/2025-sharpe-ratio) asserts 0.987 and 0.939.
PSR_NO_SERIAL_CORRELATION = PSRTestCase(
    name="Example without serial correlation, SR_0=0",
    sharpe_ratio=PAPER_SHARPE,
    skewness=-2.448,
    kurtosis=10.164,
    n_samples=24,
    sharpe_star=0.0,
    expected_psr=0.9872,
    reference="Computed from Equations 5 and 9 with rho = 0",
)

PSR_NO_SERIAL_CORRELATION_BENCHMARK = PSRTestCase(
    name="Example without serial correlation, SR_0=0.1",
    sharpe_ratio=PAPER_SHARPE,
    skewness=-2.448,
    kurtosis=10.164,
    n_samples=24,
    sharpe_star=0.1,
    expected_psr=0.9391,
    reference="Computed from Equations 5 and 9 with rho = 0",
)

PSR_NORMAL_CASE = PSRTestCase(
    name="Normal returns (skew=0, kurt=3)",
    sharpe_ratio=1.0,
    skewness=0.0,
    kurtosis=3.0,
    n_samples=100,
    sharpe_star=0.0,
    expected_psr=1.0,  # Z[SR * sqrt(T)] = Z[10]
    reference="Computed from Equations 5 and 9",
)

PSR_SMALL_SAMPLE = PSRTestCase(
    name="Small sample (T=12)",
    sharpe_ratio=0.5,
    skewness=-0.5,
    kurtosis=4.0,
    n_samples=12,
    sharpe_star=0.0,
    expected_psr=0.9584,  # Z[0.5 * sqrt(12)]; the moments drop out at SR_0 = 0
    reference="Computed from Equations 5 and 9",
)

PSR_NEG_SKEW = PSRTestCase(
    name="Negative skewness with SR_0=0.5",
    sharpe_ratio=0.8,
    skewness=-1.7,
    kurtosis=7.5,
    n_samples=100,
    sharpe_star=0.5,
    expected_psr=0.9771,  # Z[(0.8 - 0.5) / sqrt(2.25625 / 100)]
    reference="Computed from Equations 5 and 9",
)

PSR_CASES = [
    PSR_PAPER_EXAMPLE,
    PSR_PAPER_EXAMPLE_BENCHMARK,
    PSR_NO_SERIAL_CORRELATION,
    PSR_NO_SERIAL_CORRELATION_BENCHMARK,
    PSR_NORMAL_CASE,
    PSR_SMALL_SAMPLE,
    PSR_NEG_SKEW,
]


# =============================================================================
# MinTRL TEST CASES
# =============================================================================

MINTRL_PAPER_EXAMPLE = MinTRLTestCase(
    name="Paper example, SR_0=0",
    sharpe_ratio=PAPER_SHARPE,
    sharpe_star=0.0,
    skewness=-2.448,
    kurtosis=10.164,
    alpha=0.05,
    expected_mintrl=19.543,  # months
    reference="Numerical example after Equation 11: MinTRL = 19.543 months",
    autocorrelation=0.2,
)

MINTRL_PAPER_EXAMPLE_BENCHMARK = MinTRLTestCase(
    name="Paper example, SR_0=0.1",
    sharpe_ratio=PAPER_SHARPE,
    sharpe_star=0.1,
    skewness=-2.448,
    kurtosis=10.164,
    alpha=0.05,
    expected_mintrl=39.369,  # months
    reference="Numerical example after Equation 11: MinTRL = 39.369 months",
    autocorrelation=0.2,
)

MINTRL_NO_SERIAL_CORRELATION = MinTRLTestCase(
    name="Example without serial correlation, SR_0=0",
    sharpe_ratio=PAPER_SHARPE,
    sharpe_star=0.0,
    skewness=-2.448,
    kurtosis=10.164,
    alpha=0.05,
    expected_mintrl=13.029,
    reference="Computed from Equation 11 with rho = 0",
)

MINTRL_NO_SERIAL_CORRELATION_BENCHMARK = MinTRLTestCase(
    name="Example without serial correlation, SR_0=0.1",
    sharpe_ratio=PAPER_SHARPE,
    sharpe_star=0.1,
    skewness=-2.448,
    kurtosis=10.164,
    alpha=0.05,
    expected_mintrl=27.109,
    reference="Computed from Equation 11 with rho = 0",
)

MINTRL_NORMAL = MinTRLTestCase(
    name="Normal returns",
    sharpe_ratio=1.0,
    sharpe_star=0.5,
    skewness=0.0,
    kurtosis=3.0,
    alpha=0.05,
    expected_mintrl=12.175,  # (1 + 0.5 * 0.5**2) * (1.645 / 0.5)**2
    reference="Computed from Equation 11",
)

MINTRL_HIGH_TARGET = MinTRLTestCase(
    name="High target SR",
    sharpe_ratio=1.5,
    sharpe_star=1.0,
    skewness=-1.0,
    kurtosis=5.0,
    alpha=0.05,
    expected_mintrl=32.467,
    reference="Computed from Equation 11",
)

MINTRL_CASES = [
    MINTRL_PAPER_EXAMPLE,
    MINTRL_PAPER_EXAMPLE_BENCHMARK,
    MINTRL_NO_SERIAL_CORRELATION,
    MINTRL_NO_SERIAL_CORRELATION_BENCHMARK,
    MINTRL_NORMAL,
    MINTRL_HIGH_TARGET,
]


# =============================================================================
# DSR TEST CASES
# =============================================================================

# Worked example of the paper (version of 7 March 2026): K=10 trials, V[{SR_k}]=0.1.
# The Sharpe ratio is the unrounded 0.036/0.079; the paper prints it as 0.456.
DSR_PAPER_EXAMPLE = DSRTestCase(
    name="DSR with K=10 trials (paper example)",
    observed_sharpe=PAPER_SHARPE,
    n_trials=10,
    variance_trials=0.1,
    skewness=-2.448,
    kurtosis=10.164,
    n_samples=24,
    expected_dsr=0.410,
    expected_max_sharpe=0.498,  # E[max{SR_k}]
    reference=(
        "Numerical example after Exhibit 5: SR_0,K = 0.498, sigma[SR_0,K] = 0.186, "
        "DSR = 0.410 (vs one-trial PSR of 0.966)"
    ),
    tolerance=0.0005,  # the paper prints three decimals
    autocorrelation=0.2,
)

# The three cases below are not in the paper. Their expected values come from
# Equations 28 and 30, with V[max{X_k}] obtained by Gauss-Hermite quadrature
# outside the library.

# Small number of trials
DSR_FEW_TRIALS = DSRTestCase(
    name="DSR with K=3 trials",
    observed_sharpe=0.5,
    n_trials=3,
    variance_trials=0.05,
    skewness=-0.5,
    kurtosis=4.0,
    n_samples=50,
    expected_dsr=0.9678,
    expected_max_sharpe=0.1907,  # sqrt(0.05) * E[max{Z}] for K=3
    reference="Computed from Equations 28 and 30",
    tolerance=0.0005,
)

# Many trials
DSR_MANY_TRIALS = DSRTestCase(
    name="DSR with K=100 trials",
    observed_sharpe=1.0,
    n_trials=100,
    variance_trials=0.2,
    skewness=-1.0,
    kurtosis=6.0,
    n_samples=100,
    expected_dsr=0.2464,  # the observed Sharpe ratio is below the expected maximum
    expected_max_sharpe=1.1317,  # sqrt(0.2) * E[max{Z}] for K=100
    reference="Computed from Equations 28 and 30",
    tolerance=0.0005,
)

# Normal returns with multiple trials
DSR_NORMAL = DSRTestCase(
    name="DSR with normal returns, K=10",
    observed_sharpe=0.8,
    n_trials=10,
    variance_trials=0.1,
    skewness=0.0,
    kurtosis=3.0,
    n_samples=50,
    expected_dsr=0.9482,
    expected_max_sharpe=0.4979,  # sqrt(0.1) * E[max{Z}] for K=10
    reference="Computed from Equations 28 and 30",
    tolerance=0.0005,
)

DSR_CASES = [
    DSR_PAPER_EXAMPLE,
    DSR_FEW_TRIALS,
    DSR_MANY_TRIALS,
    DSR_NORMAL,
]


# =============================================================================
# STANDARD DEVIATION RE-SCALING FACTORS (Exhibit 4)
# =============================================================================

# sqrt(V[max_k{X_k}]) for K = 1 to 100 independent standard normals, as printed
# in Exhibit 4. The exhibit has ten rows of ten values: row r, column c is
# K = 10 * (r - 1) + c.
VARIANCE_RESCALING_FACTORS = {
    1: 1.00000, 2: 0.82565, 3: 0.74798, 4: 0.70122, 5: 0.66898,
    6: 0.64492, 7: 0.62603, 8: 0.61065, 9: 0.59779, 10: 0.58681,
    11: 0.57728, 12: 0.56889, 13: 0.56143, 14: 0.55473, 15: 0.54867,
    16: 0.54315, 17: 0.53808, 18: 0.53341, 19: 0.52909, 20: 0.52507,
    21: 0.52131, 22: 0.51780, 23: 0.51449, 24: 0.51138, 25: 0.50844,
    26: 0.50565, 27: 0.50301, 28: 0.50050, 29: 0.49811, 30: 0.49582,
    31: 0.49364, 32: 0.49155, 33: 0.48954, 34: 0.48762, 35: 0.48577,
    36: 0.48399, 37: 0.48228, 38: 0.48062, 39: 0.47903, 40: 0.47748,
    41: 0.47599, 42: 0.47455, 43: 0.47315, 44: 0.47180, 45: 0.47048,
    46: 0.46921, 47: 0.46797, 48: 0.46676, 49: 0.46559, 50: 0.46445,
    51: 0.46334, 52: 0.46226, 53: 0.46120, 54: 0.46017, 55: 0.45917,
    56: 0.45819, 57: 0.45723, 58: 0.45629, 59: 0.45538, 60: 0.45448,
    61: 0.45361, 62: 0.45275, 63: 0.45192, 64: 0.45109, 65: 0.45029,
    66: 0.44950, 67: 0.44873, 68: 0.44797, 69: 0.44723, 70: 0.44650,
    71: 0.44579, 72: 0.44508, 73: 0.44439, 74: 0.44372, 75: 0.44305,
    76: 0.44240, 77: 0.44175, 78: 0.44112, 79: 0.44050, 80: 0.43989,
    81: 0.43929, 82: 0.43870, 83: 0.43811, 84: 0.43754, 85: 0.43698,
    86: 0.43642, 87: 0.43587, 88: 0.43533, 89: 0.43480, 90: 0.43428,
    91: 0.43376, 92: 0.43325, 93: 0.43275, 94: 0.43226, 95: 0.43177,
    96: 0.43129, 97: 0.43081, 98: 0.43034, 99: 0.42988, 100: 0.42942,
}  # fmt: skip
