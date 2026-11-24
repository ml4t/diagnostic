"""Rademacher Anti-Serum (RAS) for multiple testing correction.

Implements Rademacher complexity-based corrections using the joint strategy
performance matrix to account for strategy correlation.

Identical strategies contribute zero additional complexity. Confidence statements
require the stated population, sampling and complexity assumptions.

References
----------
.. [1] Paleologo, G. (2025). "The Elements of Quantitative Investing",
       Wiley Finance, Rademacher Anti-Serum discussion.
.. [2] Bartlett, P.L. & Mendelson, S. (2002). "Rademacher and Gaussian
       Complexities: Risk Bounds and Structural Results", JMLR 3:463-482.
.. [3] Massart, P. (2000). "Some applications of concentration inequalities
       to statistics", Annales de la Faculté des Sciences de Toulouse.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class RASResult:
    """Result of Rademacher Anti-Serum adjustment.

    Attributes
    ----------
    adjusted_values : NDArray
        Lower bounds conditional on the adjustment's sampling, population and
        complexity assumptions.
    observed_values : NDArray
        Original observed values before adjustment.
    complexity : float
        Rademacher complexity R̂ used in adjustment.
    data_snooping_penalty : float
        Penalty from data snooping (2R̂).
    estimation_error : float
        Penalty from estimation uncertainty.
    n_significant : int
        Number of strategies with adjusted values > 0.
    significant_mask : NDArray[np.bool_]
        Boolean mask of significant strategies.
    massart_bound : float
        Population-range upper bound κ√(2 log N / T) for IC; the unit-scale
        diagnostic √(2 log N / T) for standardized Sharpe inputs.
    complexity_ratio : float
        R̂ / massart_bound (lower = more correlated strategies).
    """

    adjusted_values: NDArray[Any]
    observed_values: NDArray[Any]
    complexity: float
    data_snooping_penalty: float
    estimation_error: float
    n_significant: int
    significant_mask: NDArray[np.bool_]
    massart_bound: float
    complexity_ratio: float


def rademacher_complexity(
    X: NDArray[Any],
    n_simulations: int = 10000,
    random_state: int | None = None,
) -> float:
    """Compute empirical Rademacher complexity via Monte Carlo estimation.

    Measures a strategy set's capacity to fit random noise, quantifying
    overfitting risk when selecting among multiple candidates.

    **Convention used here** (non-absolute supremum, 1/T normalization):

        R̂_T(F) = E_σ[sup_{n} (1/T) Σᵢ σᵢ xᵢₙ]

    where σᵢ ∈ {-1, +1} with P(σᵢ = 1) = 0.5 (Rademacher distribution).

    **Interpretation**:
    - Identical columns have exact empirical complexity zero.
    - More distinct strategies can fit more noise; the upper bound also scales
      with column norms, not just the number of strategies.

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Performance matrix: T time periods × N strategies.
        Typically contains period-by-period ICs or returns.
    n_simulations : int, default=10000
        Monte Carlo samples. Higher = more accurate but slower.
        Accuracy depends on the data; no relative-error guarantee is supplied.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Monte Carlo estimate of empirical Rademacher complexity. Finite draws
        can underestimate the exact expectation, or even produce a negative
        estimate for a singleton class.

    Notes
    -----
    **Massart's Upper Bound** [3]:
        R̂ ≤ max_n ||xₙ||₂ × √(2 log N) / T

    For normalized data (||xₙ||₂ ≈ √T), this simplifies to √(2 log N / T).
    A Monte Carlo point estimate is not a certified upper bound. For confidence
    statements, use the exact expectation, a deterministic upper bound (such as
    Massart's bound above), or a separately justified Monte Carlo upper bound
    with its failure probability included in the total error budget.

    **Computational Complexity**: O(n_simulations × T × N)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(2500, 1000) * 0.02  # 1000 strategies, 2500 days
    >>> R_hat = rademacher_complexity(X, random_state=42)
    >>> massart = np.sqrt(2 * np.log(1000) / 2500)
    >>> print(f"R̂={R_hat:.4f}, Massart={massart:.4f}, ratio={R_hat/massart:.2f}")

    References
    ----------
    .. [2] Bartlett & Mendelson (2002), JMLR 3:463-482, Definition 2
           uses an absolute supremum and a different factor-two convention.
    .. [3] Massart (2000), Lemma 1.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be numpy array, got {type(X)}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D array (T, N), got shape {X.shape}")

    T, N = X.shape

    if T < 1 or N < 1:
        raise ValueError(f"X must have positive dimensions, got ({T}, {N})")

    rng = np.random.default_rng(random_state)

    # Monte Carlo estimation: E_σ[max_n (σ^T x_n / T)]
    max_correlations = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Rademacher vector: σᵢ ∈ {-1, +1} with P=0.5
        sigma = rng.choice([-1.0, 1.0], size=T)

        # Compute (σ^T x_n) / T for all strategies n
        correlations = sigma @ X / T

        # Take supremum over strategy set
        max_correlations[i] = np.max(correlations)

    return float(np.mean(max_correlations))


def ras_ic_adjustment(
    observed_ic: NDArray[Any],
    complexity: float,
    n_samples: int,
    delta: float = 0.05,
    kappa: float = 1.0,
    return_result: bool = False,
    *,
    ic_samples: NDArray[Any] | None = None,
) -> NDArray[Any] | RASResult:
    """Apply RAS adjustment for Information Coefficients (bounded metrics).

    Computes conservative lower bounds on true IC values accounting for
    data snooping and estimation error.

    **Formula** (empirical Rademacher bound for period ICs in [-κ, κ]):

        θₙ ≥ θ̂ₙ - 2R̂ - 3κ√(2 log(2/δ)/T)
               ───   ─────────────────
               (a)         (b)

    where:
        (a) = data snooping penalty from testing N strategies
        (b) = bounded-difference error, including estimation of complexity

    Parameters
    ----------
    observed_ic : ndarray of shape (N,)
        Means of the period-by-period ICs for N strategies, not pooled ICs.
    complexity : float
        Exact empirical Rademacher complexity R̂, or a conservative upper bound,
        for the same period-by-strategy matrix. The Monte Carlo point estimate
        from `rademacher_complexity()` alone does not certify coverage.
    n_samples : int
        Number of time periods T used to compute ICs.
    delta : float, default=0.05
        Significance level (1 - confidence). Lower = more conservative.
    kappa : float, default=1.0
        A priori population bound: every possible period IC must satisfy
        |x[t, n]| <= kappa almost surely. Pearson and rank correlations admit
        the safe bound 1.0. A smaller bound needs an external justification;
        a typical mean, standard deviation or observed sample maximum is not
        sufficient. This function never estimates kappa from the sample.
    return_result : bool, default=False
        If True, return RASResult dataclass with full diagnostics.
    ic_samples : ndarray of shape (n_samples, N), optional
        Period IC matrix used for the means and complexity. Checks shape,
        finite values, means and observed violations of kappa. Passing these
        checks cannot certify a bound on unobserved population outcomes.

    Returns
    -------
    ndarray or RASResult
        If return_result=False: Adjusted IC lower bounds (N,).
        If return_result=True: RASResult with full diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid or supplied period ICs exceed kappa.

    Warns
    -----
    UserWarning
        If any mean |observed_ic| > kappa without ic_samples. This detects only
        an obvious violation; means within kappa do not validate the bound.

    Notes
    -----
    For a fixed strategy class and independent identically distributed period
    vectors, the empirical Rademacher theorem gives a simultaneous one-sided
    bound for all strategies. Rescaling [-kappa, kappa] to [0, 1] gives the
    estimation term 3*kappa*sqrt(2*log(2/delta)/T) [3]. The extra concentration
    accounts for replacing population complexity with empirical complexity;
    Hoeffding's inequality for a single mean alone does not justify that step.
    Serially dependent or overlapping periods do not satisfy the IID assumption.
    No zero-false-positive guarantee is made. Negative lower bounds are retained.

    The previous default 0.02 and smaller estimation term have been replaced.
    Explicit kappa and positional return_result calls remain accepted, but
    lower bounds are more conservative even when kappa is supplied explicitly.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(42).uniform(-0.1, 0.1, (252, 5))
    >>> observed_ic = X.mean(axis=0)
    >>> upper_complexity = np.max(np.linalg.norm(X, axis=0)) * np.sqrt(2*np.log(5)) / 252
    >>> result = ras_ic_adjustment(
    ...     observed_ic, upper_complexity, 252, return_result=True, ic_samples=X
    ... )
    >>> print(f"Significant: {result.n_significant}/{len(observed_ic)}")

    References
    ----------
    .. [1] Paleologo (2025), Rademacher Anti-Serum discussion.
    .. [2] Hoeffding (1963), "Probability inequalities for sums of bounded
           random variables", JASA 58:13-30.
    .. [3] Mohri, Rostamizadeh & Talwalkar (2018), "Foundations of Machine
           Learning", 2nd ed., Theorem 3.3, empirical-complexity bound.
    """
    observed_ic = np.asarray(observed_ic, dtype=float)

    if observed_ic.ndim != 1:
        raise ValueError(f"observed_ic must be 1D, got shape {observed_ic.shape}")

    if observed_ic.size == 0 or not np.all(np.isfinite(observed_ic)):
        raise ValueError("observed_ic must contain finite, nonempty means")

    if not np.isfinite(complexity) or complexity < 0:
        raise ValueError(f"complexity must be finite and non-negative, got {complexity}")

    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 1
    ):
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    if kappa is None or not np.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"kappa must be positive and finite, got {kappa}")

    if ic_samples is not None:
        samples = np.asarray(ic_samples, dtype=float)
        if samples.shape != (n_samples, observed_ic.size):
            raise ValueError("ic_samples must have shape (n_samples, len(observed_ic))")
        if not np.all(np.isfinite(samples)):
            raise ValueError("ic_samples must contain only finite period ICs")
        if np.max(np.abs(samples)) > kappa:
            raise ValueError("period ICs in ic_samples exceed kappa; no confidence bound is valid")
        if not np.allclose(samples.mean(axis=0), observed_ic, rtol=1e-10, atol=1e-12):
            raise ValueError("observed_ic must equal the column means of ic_samples")

    # Warn if ICs exceed the bounded assumption
    max_abs_ic = np.max(np.abs(observed_ic))
    if ic_samples is None and max_abs_ic > kappa:
        warnings.warn(
            f"max(|mean IC|)={max_abs_ic:.4f} exceeds kappa={kappa}. "
            "The population bound on period ICs is violated; no confidence guarantee holds. "
            "Means within kappa would not certify that bound either.",
            UserWarning,
            stacklevel=2,
        )

    N = len(observed_ic)
    T = n_samples

    # (a) Data snooping penalty: 2R̂
    data_snooping = 2 * complexity

    # Empirical-complexity concentration, rescaled from [0, 1] to [-kappa, kappa].
    estimation_error = 3 * kappa * np.sqrt(2 * np.log(2 / delta) / T)

    # Conservative lower bound
    adjusted_ic = observed_ic - data_snooping - estimation_error

    if not return_result:
        return adjusted_ic

    # Compute diagnostics
    massart_bound = kappa * np.sqrt(2 * np.log(N) / T) if N > 1 else 0.0
    significant_mask = adjusted_ic > 0

    return RASResult(
        adjusted_values=adjusted_ic,
        observed_values=observed_ic,
        complexity=complexity,
        data_snooping_penalty=data_snooping,
        estimation_error=estimation_error,
        n_significant=int(np.sum(significant_mask)),
        significant_mask=significant_mask,
        massart_bound=massart_bound,
        complexity_ratio=complexity / massart_bound if massart_bound > 0 else 0.0,
    )


def ras_sharpe_adjustment(
    observed_sharpe: NDArray[Any],
    complexity: float,
    n_samples: int,
    n_strategies: int,
    delta: float = 0.05,
    return_result: bool = False,
) -> NDArray[Any] | RASResult:
    """Apply RAS adjustment for Sharpe ratios (sub-Gaussian metrics).

    Computes conservative lower bounds on true Sharpe ratios accounting for
    data snooping, estimation error, and multiple testing.

    **Formula** (sub-Gaussian concentration + union bound):

        θₙ ≥ θ̂ₙ - 2R̂ - 3√(2 log(2/δ)/T) - √(2 log(2N/δ)/T)
               ───   ─────────────────────────────────────
               (a)              (b)              (c)

    where:
        (a) = data snooping penalty
        (b) = sub-Gaussian estimation error (factor 3 for conservatism)
        (c) = union bound over N strategies

    Parameters
    ----------
    observed_sharpe : ndarray of shape (N,)
        Observed (annualized) Sharpe ratios for N strategies.
    complexity : float
        Rademacher complexity R̂ from `rademacher_complexity()`.
    n_samples : int
        Number of time periods T used to compute Sharpe ratios.
    n_strategies : int
        Total number of strategies N tested.
    delta : float, default=0.05
        Significance level (1 - confidence). Lower = more conservative.
    return_result : bool, default=False
        If True, return RASResult dataclass with full diagnostics.

    Returns
    -------
    ndarray or RASResult
        If return_result=False: Adjusted Sharpe lower bounds (N,).
        If return_result=True: RASResult with full diagnostics.

    Notes
    -----
    **Derivation**:
    1. Data snooping: 2R̂ (standard Rademacher bound)
    2. Sub-Gaussian error: For σ²-sub-Gaussian X, P(X > t) ≤ exp(-t²/2σ²).
       Daily returns typically have σ ≈ 1 when standardized.
       Factor 3 provides conservatism for heavier tails.
    3. Union bound: P(∃n: |X̂ₙ - Xₙ| > t) ≤ N × single-strategy bound.
       Contributes √(2 log(2N/δ)/T) term.

    **Comparison to DSR**:
    - DSR assumes independent strategies (overpenalizes correlated ones)
    - RAS captures correlation via R̂ (correlated → lower R̂ → less penalty)
    - RAS is non-asymptotic; DSR requires large T

    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.randn(252, 100) * 0.01  # 100 strategies, 1 year
    >>> observed_sr = returns.mean(axis=0) / returns.std(axis=0) * np.sqrt(252)
    >>> R_hat = rademacher_complexity(returns)
    >>> result = ras_sharpe_adjustment(
    ...     observed_sr, R_hat, 252, 100, return_result=True
    ... )
    >>> print(f"Significant: {result.n_significant}/100")

    References
    ----------
    .. [1] Paleologo (2024), Section 8.3.2, Procedure 8.2.
    """
    observed_sharpe = np.asarray(observed_sharpe)

    if observed_sharpe.ndim != 1:
        raise ValueError(f"observed_sharpe must be 1D, got shape {observed_sharpe.shape}")

    if complexity < 0:
        raise ValueError(f"complexity must be non-negative, got {complexity}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if n_strategies < 1:
        raise ValueError(f"n_strategies must be positive, got {n_strategies}")

    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    T = n_samples
    N = n_strategies

    # (a) Data snooping penalty: 2R̂
    data_snooping = 2 * complexity

    # (b) Sub-Gaussian estimation error (independent of N)
    # Factor 3 for conservatism with potential heavy tails
    error_term1 = 3 * np.sqrt(2 * np.log(2 / delta) / T)

    # (c) Union bound over N strategies
    error_term2 = np.sqrt(2 * np.log(2 * N / delta) / T)

    estimation_error = error_term1 + error_term2

    # Conservative lower bound
    adjusted_sharpe = observed_sharpe - data_snooping - estimation_error

    if not return_result:
        return adjusted_sharpe

    # Compute diagnostics
    massart_bound = np.sqrt(2 * np.log(N) / T) if N > 1 else 0.0
    significant_mask = adjusted_sharpe > 0

    return RASResult(
        adjusted_values=adjusted_sharpe,
        observed_values=observed_sharpe,
        complexity=complexity,
        data_snooping_penalty=data_snooping,
        estimation_error=estimation_error,
        n_significant=int(np.sum(significant_mask)),
        significant_mask=significant_mask,
        massart_bound=massart_bound,
        complexity_ratio=complexity / massart_bound if massart_bound > 0 else 0.0,
    )


__all__ = [
    "RASResult",
    "rademacher_complexity",
    "ras_ic_adjustment",
    "ras_sharpe_adjustment",
]
