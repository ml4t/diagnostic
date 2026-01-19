"""Rademacher Anti-Serum (RAS) for multiple testing correction.

Implements Rademacher complexity-based corrections that account for strategy
correlation, unlike traditional methods (DSR, Bonferroni) which assume independence.

**Key Advantage**: Zero false positive rate when strategies are correlated.
Identical strategies contribute zero additional complexity.

References
----------
.. [1] Paleologo, G. (2024). "The Elements of Quantitative Investing",
       Wiley Finance, Chapter 4.3 / Section 8.3.
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
        Conservative lower bounds on true performance metrics.
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
        Theoretical upper bound √(2 log N / T).
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

    **Definition** (Bartlett & Mendelson, 2002):

        R̂_T(F) = E_σ[sup_{n} (1/T) Σᵢ σᵢ xᵢₙ]

    where σᵢ ∈ {-1, +1} with P(σᵢ = 1) = 0.5 (Rademacher distribution).

    **Interpretation**:
    - R̂ ≈ 0: Strategies highly correlated (low overfitting risk)
    - R̂ → √(2 log N / T): Strategies uncorrelated (Massart upper bound)

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Performance matrix: T time periods × N strategies.
        Typically contains period-by-period ICs or returns.
    n_simulations : int, default=10000
        Monte Carlo samples. Higher = more accurate but slower.
        10000 provides ~1% relative error.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Empirical Rademacher complexity R̂ ∈ [0, √(2 log N / T)].

    Notes
    -----
    **Massart's Upper Bound** [3]:
        R̂ ≤ max_n ||xₙ||₂ × √(2 log N) / T

    For normalized data (||xₙ||₂ ≈ √T), this simplifies to √(2 log N / T).

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
    .. [2] Bartlett & Mendelson (2002), JMLR 3:463-482, Definition 2.
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
    kappa: float = 0.02,
    return_result: bool = False,
) -> NDArray[Any] | RASResult:
    """Apply RAS adjustment for Information Coefficients (bounded metrics).

    Computes conservative lower bounds on true IC values accounting for
    data snooping and estimation error.

    **Formula** (Hoeffding concentration for |IC| ≤ κ):

        θₙ ≥ θ̂ₙ - 2R̂ - 2κ√(log(2/δ)/T)
               ───   ─────────────────
               (a)         (b)

    where:
        (a) = data snooping penalty from testing N strategies
        (b) = estimation error for bounded r.v. (Hoeffding's inequality)

    Parameters
    ----------
    observed_ic : ndarray of shape (N,)
        Observed Information Coefficients for N strategies.
    complexity : float
        Rademacher complexity R̂ from `rademacher_complexity()`.
    n_samples : int
        Number of time periods T used to compute ICs.
    delta : float, default=0.05
        Significance level (1 - confidence). Lower = more conservative.
    kappa : float, default=0.02
        Bound on |IC|. **Critical parameter**.

        Practical guidance (Paleologo 2024, p.273):
        - κ=0.02: Typical alpha signals
        - κ=0.05: High-conviction signals
        - κ=1.0: Theoretical maximum (usually too conservative)
    return_result : bool, default=False
        If True, return RASResult dataclass with full diagnostics.

    Returns
    -------
    ndarray or RASResult
        If return_result=False: Adjusted IC lower bounds (N,).
        If return_result=True: RASResult with full diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid or observed ICs exceed kappa bound.

    Warns
    -----
    UserWarning
        If any |observed_ic| > κ (theoretical guarantee violated).

    Notes
    -----
    **Derivation**:
    1. Data snooping: Standard Rademacher generalization bound gives 2R̂.
    2. Estimation: For bounded r.v. |X| ≤ κ, Hoeffding gives
       P(|X̂ - X| > t) ≤ 2exp(-Tt²/2κ²). Setting RHS = δ yields
       t = κ√(2 log(2/δ)/T). Conservative factor 2 for two-sided.

    **Advantages over DSR**:
    - Accounts for strategy correlation (R̂ ↓ as correlation ↑)
    - Non-asymptotic (valid for any T)
    - Zero false positives in Paleologo's simulations

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(2500, 500) * 0.02
    >>> observed_ic = X.mean(axis=0)
    >>> R_hat = rademacher_complexity(X)
    >>> result = ras_ic_adjustment(observed_ic, R_hat, 2500, return_result=True)
    >>> print(f"Significant: {result.n_significant}/{len(observed_ic)}")

    References
    ----------
    .. [1] Paleologo (2024), Section 8.3.2, Procedure 8.1.
    .. [2] Hoeffding (1963), "Probability inequalities for sums of bounded
           random variables", JASA 58:13-30.
    """
    observed_ic = np.asarray(observed_ic)

    if observed_ic.ndim != 1:
        raise ValueError(f"observed_ic must be 1D, got shape {observed_ic.shape}")

    if complexity < 0:
        raise ValueError(f"complexity must be non-negative, got {complexity}")

    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")

    # Warn if ICs exceed the bounded assumption
    max_abs_ic = np.max(np.abs(observed_ic))
    if max_abs_ic > kappa:
        warnings.warn(
            f"max(|IC|)={max_abs_ic:.4f} exceeds kappa={kappa}. "
            "Theoretical guarantees may not hold. Consider increasing kappa.",
            UserWarning,
            stacklevel=2,
        )

    N = len(observed_ic)
    T = n_samples

    # (a) Data snooping penalty: 2R̂
    data_snooping = 2 * complexity

    # (b) Estimation error: 2κ√(log(2/δ)/T) from Hoeffding
    estimation_error = 2 * kappa * np.sqrt(np.log(2 / delta) / T)

    # Conservative lower bound
    adjusted_ic = observed_ic - data_snooping - estimation_error

    if not return_result:
        return adjusted_ic

    # Compute diagnostics
    massart_bound = np.sqrt(2 * np.log(N) / T) if N > 1 else 0.0
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
