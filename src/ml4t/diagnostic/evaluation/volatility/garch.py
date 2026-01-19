"""GARCH model fitting for time-varying volatility.

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
capture time-varying volatility in financial time series.

References:
    Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
    Journal of Econometrics, 31(3), 307-327. DOI: 10.1016/0304-4076(86)90063-1
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ml4t.diagnostic.errors import ComputationError, ValidationError
from ml4t.diagnostic.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# GARCH model fitting requires arch package (optional dependency)
# Lazy loading to avoid slow module-level import (~200ms)
HAS_ARCH: bool | None = None  # Will be set on first check
_arch_model_cache: Callable[..., Any] | None = None
_ARCHModelResult_cache: type[Any] | None = None


def _check_arch_available() -> bool:
    """Check if arch package is available and import it (lazy)."""
    global HAS_ARCH, _arch_model_cache, _ARCHModelResult_cache
    if HAS_ARCH is None:
        try:
            from arch import arch_model as _impl
            from arch.univariate.base import ARCHModelResult as _ARCHModelResultImpl

            _arch_model_cache = _impl
            _ARCHModelResult_cache = _ARCHModelResultImpl
            HAS_ARCH = True
        except ImportError:
            HAS_ARCH = False
            _arch_model_cache = None
            _ARCHModelResult_cache = None
    return HAS_ARCH


def _get_arch_model() -> Callable[..., Any]:
    """Get the arch_model function (lazy import)."""
    _check_arch_available()
    if _arch_model_cache is None:
        raise ImportError(
            "GARCH fitting requires the 'arch' package. Install with: pip install arch"
        )
    return _arch_model_cache


def _compute_skewness(data: np.ndarray) -> float:
    """Compute sample skewness."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 3))


def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute sample excess kurtosis."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 4) - 3)


class GARCHResult:
    """Results from GARCH model fitting.

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
    capture time-varying volatility in financial time series. The GARCH(p,q)
    model specifies conditional variance as:

        σ²ₜ = ω + Σ(αᵢ·ε²ₜ₋ᵢ) + Σ(βⱼ·σ²ₜ₋ⱼ)

    For GARCH(1,1):
        σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

    Attributes:
        omega: Constant term (long-run variance component)
        alpha: ARCH coefficient (impact of past squared errors)
        beta: GARCH coefficient (impact of past conditional variance)
        persistence: α + β (should be < 1 for stationarity)
        log_likelihood: Log-likelihood of fitted model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        conditional_volatility: Fitted conditional volatility (σₜ)
        standardized_residuals: Residuals divided by conditional volatility
        converged: Whether optimization converged successfully
        iterations: Number of iterations taken
        n_obs: Number of observations used in fitting
    """

    def __init__(
        self,
        omega: float,
        alpha: float | tuple[float, ...],
        beta: float | tuple[float, ...],
        persistence: float,
        log_likelihood: float,
        aic: float,
        bic: float,
        conditional_volatility: pd.Series,
        standardized_residuals: pd.Series,
        converged: bool,
        iterations: int,
        n_obs: int,
    ):
        """Initialize GARCH result.

        Args:
            omega: Constant term
            alpha: ARCH coefficient(s)
            beta: GARCH coefficient(s)
            persistence: Sum of alpha and beta (alpha + beta)
            log_likelihood: Log-likelihood value
            aic: Akaike Information Criterion
            bic: Bayesian Information Criterion
            conditional_volatility: Fitted conditional volatility series
            standardized_residuals: Standardized residuals
            converged: Whether optimization converged
            iterations: Number of iterations
            n_obs: Number of observations
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.persistence = persistence
        self.log_likelihood = log_likelihood
        self.aic = aic
        self.bic = bic
        self.conditional_volatility = conditional_volatility
        self.standardized_residuals = standardized_residuals
        self.converged = converged
        self.iterations = iterations
        self.n_obs = n_obs

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GARCHResult(omega={self.omega:.6f}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"persistence={self.persistence:.4f})"
        )

    def summary(self) -> str:
        """Human-readable summary of GARCH model results.

        Returns:
            Formatted summary string
        """
        lines = [
            "GARCH Model Fitting Results",
            "=" * 50,
            f"Observations:      {self.n_obs}",
            f"Converged:         {'Yes' if self.converged else 'No'}",
            f"Iterations:        {self.iterations}",
        ]

        lines.append("")
        lines.append("Model Parameters:")
        lines.append(f"  ω (omega):       {self.omega:.6f}")

        # Handle scalar or vector alpha/beta
        if isinstance(self.alpha, tuple | list):
            for i, a in enumerate(self.alpha, 1):
                lines.append(f"  α{i} (alpha[{i}]):  {a:.6f}")
        else:
            lines.append(f"  α (alpha):       {self.alpha:.6f}")

        if isinstance(self.beta, tuple | list):
            for i, b in enumerate(self.beta, 1):
                lines.append(f"  β{i} (beta[{i}]):   {b:.6f}")
        else:
            lines.append(f"  β (beta):        {self.beta:.6f}")

        lines.append("")
        lines.append(f"Persistence (α+β): {self.persistence:.6f}")

        if self.persistence >= 1.0:
            lines.append("  ⚠ WARNING: Persistence ≥ 1 (non-stationary)")
        elif self.persistence > 0.95:
            lines.append("  → High persistence (slow mean reversion)")
        else:
            lines.append("  → Stationary process")

        lines.append("")
        lines.append("Model Fit Statistics:")
        lines.append(f"  Log-Likelihood:  {self.log_likelihood:.4f}")
        lines.append(f"  AIC:             {self.aic:.4f}")
        lines.append(f"  BIC:             {self.bic:.4f}")

        lines.append("")
        lines.append("Conditional Volatility:")
        vol = np.asarray(self.conditional_volatility.to_numpy(), dtype=np.float64)
        lines.append(f"  Mean:            {float(np.mean(vol)):.6f}")
        lines.append(f"  Std Dev:         {float(np.std(vol)):.6f}")
        lines.append(f"  Min:             {np.min(vol):.6f}")
        lines.append(f"  Max:             {np.max(vol):.6f}")

        lines.append("")
        lines.append("Standardized Residuals:")
        resid = np.asarray(self.standardized_residuals.to_numpy(), dtype=np.float64)
        lines.append(f"  Mean:            {float(np.mean(resid)):.6f}")
        lines.append(f"  Std Dev:         {float(np.std(resid)):.6f}")
        lines.append(f"  Skewness:        {_compute_skewness(resid):.4f}")
        lines.append(f"  Kurtosis:        {_compute_kurtosis(resid):.4f}")

        lines.append("")
        lines.append("Interpretation:")
        lines.append("  - ω: Long-run unconditional variance = ω / (1 - α - β)")
        lines.append("  - α: Sensitivity to recent shocks (news impact)")
        lines.append("  - β: Persistence of past volatility")
        lines.append("  - α+β: Overall persistence (< 1 for stationarity)")

        return "\n".join(lines)


def fit_garch(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean_model: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ] = "Zero",
    dist: Literal[
        "normal", "gaussian", "t", "studentst", "skewstudent", "skewt", "ged", "generalized error"
    ] = "normal",
) -> GARCHResult:
    """Fit GARCH(p, q) model to returns series.

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
    are used to model time-varying volatility in financial time series. The
    GARCH(p,q) model specifies conditional variance as:

        σ²ₜ = ω + Σ(αᵢ·ε²ₜ₋ᵢ) + Σ(βⱼ·σ²ₜ₋ⱼ)

    For the common GARCH(1,1):
        σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁

    Where:
        - ω (omega): Constant term
        - α (alpha): ARCH coefficient (impact of past squared errors)
        - β (beta): GARCH coefficient (impact of past conditional variance)

    Persistence (α + β) should be < 1 for stationarity. Values close to 1
    indicate high volatility persistence.

    Args:
        returns: Returns series (NOT prices) to fit GARCH model
        p: ARCH order (number of lagged squared errors), default 1
        q: GARCH order (number of lagged conditional variances), default 1
        mean_model: Mean model specification, one of:
            - "Zero": Zero mean (default, common for returns)
            - "Constant": Constant mean
            - "AR": Autoregressive mean
            - "ARX": AR with exogenous regressors
            - "HAR": Heterogeneous AR
            - "LS": Least squares
        dist: Error distribution, one of:
            - "normal": Normal distribution (default)
            - "t": Student's t distribution (fat tails)
            - "skewt": Skewed Student's t distribution
            - "ged": Generalized Error Distribution

    Returns:
        GARCHResult with fitted parameters and diagnostics

    Raises:
        ValidationError: If data is invalid or arch package not installed
        ComputationError: If GARCH fitting fails

    Notes:
        - Requires arch package: pip install arch
        - GARCH(1,1) is sufficient for most financial applications
        - Higher orders (p>1, q>1) rarely improve fit significantly
        - Use ARCH-LM test first to check if GARCH is appropriate
        - Convergence can be sensitive to starting values
        - Consider Student's t or skewed t for fat-tailed returns

    References:
        Bollerslev, T. (1986). Generalized Autoregressive Conditional
        Heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
        DOI: 10.1016/0304-4076(86)90063-1
    """
    # Check if arch package is available (lazy check)
    if not _check_arch_available():
        raise ValidationError(
            "GARCH fitting requires the 'arch' package. Install with: pip install arch",
            context={"available": False},
        )
    logger.debug(f"Fitting GARCH({p},{q}) model with mean_model={mean_model}, dist={dist}")

    # Convert to numpy array if needed
    arr = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)

    # Validate input
    if arr.size == 0:
        raise ValidationError(
            "Cannot fit GARCH on empty data",
            context={"data_size": 0},
        )

    if arr.ndim != 1:
        raise ValidationError(
            f"Returns must be 1-dimensional, got shape {arr.shape}",
            context={"data_shape": arr.shape},
        )

    if np.any(~np.isfinite(arr)):
        n_invalid = np.sum(~np.isfinite(arr))
        raise ValidationError(
            f"Returns contain {n_invalid} NaN or infinite values",
            context={"n_invalid": n_invalid, "data_size": arr.size},
        )

    # Check minimum sample size
    min_obs = max(p, q) * 10 + 50  # Need sufficient data for estimation
    if arr.size < min_obs:
        raise ValidationError(
            f"Insufficient data for GARCH({p},{q}). Need at least {min_obs} observations, got {arr.size}",
            context={"n_obs": arr.size, "p": p, "q": q, "min_required": min_obs},
        )

    # Validate model parameters
    if p < 1:
        raise ValidationError(
            f"ARCH order (p) must be at least 1, got {p}",
            context={"p": p},
        )

    if q < 1:
        raise ValidationError(
            f"GARCH order (q) must be at least 1, got {q}",
            context={"q": q},
        )

    try:
        # Scale returns to percentage (arch works better with scaled data)
        # Convert to pandas Series if needed (arch requires Series or DataFrame)
        returns_series = (
            pd.Series(arr, name="returns") if not isinstance(returns, pd.Series) else returns.copy()
        )

        # Create and fit GARCH model using arch library (lazy import)
        model = _get_arch_model()(
            returns_series,
            mean=mean_model,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
        )

        # Fit model (may take time for complex models)
        fitted = model.fit(disp="off", show_warning=False)

        # Extract parameters
        params = fitted.params

        # For GARCH(1,1), parameters are typically:
        # omega (constant), alpha[1] (ARCH), beta[1] (GARCH)
        omega = float(params.get("omega", 0.0))

        # Extract ARCH coefficients (alpha)
        alpha_list = []
        for i in range(1, p + 1):
            key = f"alpha[{i}]"
            if key in params:
                alpha_list.append(float(params[key]))

        # Extract GARCH coefficients (beta)
        beta_list = []
        for i in range(1, q + 1):
            key = f"beta[{i}]"
            if key in params:
                beta_list.append(float(params[key]))

        # Handle scalar vs vector
        if len(alpha_list) == 1:
            alpha: float | tuple[float, ...] = alpha_list[0]
        else:
            alpha = tuple(alpha_list)

        if len(beta_list) == 1:
            beta: float | tuple[float, ...] = beta_list[0]
        else:
            beta = tuple(beta_list)

        # Compute persistence (sum of all alpha and beta coefficients)
        persistence = sum(alpha_list) + sum(beta_list)

        # Extract fitted values
        conditional_volatility = fitted.conditional_volatility
        standardized_residuals = fitted.std_resid

        # Extract convergence info
        converged = fitted.convergence_flag == 0  # 0 means success
        # fit_stop is a string (e.g., "Normal convergence"), not iteration count
        # Try to get actual iteration count from optimization result if available
        try:
            iterations = fitted.fit_info.get("iterations", 0)
            if not isinstance(iterations, int):
                iterations = 0
        except (AttributeError, TypeError):
            iterations = 0  # Fallback if not available

        logger.info(
            f"GARCH({p},{q}) fitted successfully",
            omega=omega,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            converged=converged,
        )

        return GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            persistence=persistence,
            log_likelihood=float(fitted.loglikelihood),
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            conditional_volatility=conditional_volatility,
            standardized_residuals=standardized_residuals,
            converged=converged,
            iterations=iterations,
            n_obs=arr.size,
        )

    except Exception as e:
        # Handle computation errors
        logger.error(f"GARCH fitting failed: {e}", p=p, q=q, n_obs=arr.size)
        raise ComputationError(  # noqa: B904
            f"GARCH({p},{q}) fitting failed: {e}",
            context={
                "n_obs": arr.size,
                "p": p,
                "q": q,
                "mean_model": mean_model,
                "dist": dist,
            },
            cause=e,
        )
