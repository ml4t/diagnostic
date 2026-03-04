"""Result dataclasses for factor exposure and attribution analysis.

All result types follow the library pattern: dataclass with to_dict(),
to_dataframe(), and summary() methods for serialization and display.

References
----------
- Paleologo, *Elements of Quantitative Investing* (2025), Ch 14: Attribution CIs
- Newey & West (1987): HAC standard errors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass
class FactorModelResult:
    """Static factor regression output (OLS or regularized).

    Parameters
    ----------
    alpha : float
        Per-period intercept (Jensen's alpha). Not annualized.
    alpha_se : float
        Standard error of alpha.
    alpha_t : float
        t-statistic for alpha.
    alpha_p : float
        p-value for alpha.
    betas : dict[str, float]
        Factor loadings (exposures).
    beta_ses : dict[str, float]
        Standard errors of betas (HAC if enabled).
    beta_ts : dict[str, float]
        t-statistics per factor.
    beta_ps : dict[str, float]
        p-values per factor.
    beta_cis : dict[str, tuple[float, float]]
        Confidence intervals per factor.
    r_squared : float
        Coefficient of determination.
    adj_r_squared : float
        Adjusted R-squared.
    residuals : np.ndarray
        Regression residuals.
    durbin_watson : float
        Durbin-Watson statistic for autocorrelation.
    factor_names : list[str]
        Factor names in regression order.
    n_obs : int
        Number of observations used.
    method : str
        Estimation method ("ols", "ridge", "lasso", "elastic_net").
    hac : bool
        Whether HAC standard errors were used.
    confidence_level : float
        Confidence level for CIs (e.g. 0.95).
    """

    alpha: float
    alpha_se: float
    alpha_t: float
    alpha_p: float
    betas: dict[str, float]
    beta_ses: dict[str, float]
    beta_ts: dict[str, float]
    beta_ps: dict[str, float]
    beta_cis: dict[str, tuple[float, float]]
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    durbin_watson: float
    factor_names: list[str]
    n_obs: int
    method: str = "ols"
    hac: bool = True
    confidence_level: float = 0.95

    @property
    def t_stat_pct_above_2(self) -> float:
        """Percentage of factor t-stats with |t| > 2 (Paleologo Ch 5)."""
        if not self.beta_ts:
            return 0.0
        count = sum(1 for t in self.beta_ts.values() if abs(t) > 2.0)
        return count / len(self.beta_ts)

    @property
    def significant_factors(self) -> list[str]:
        """Factors with p-value < (1 - confidence_level)."""
        threshold = 1.0 - self.confidence_level
        return [f for f, p in self.beta_ps.items() if p < threshold]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alpha": self.alpha,
            "alpha_se": self.alpha_se,
            "alpha_t": self.alpha_t,
            "alpha_p": self.alpha_p,
            "betas": dict(self.betas),
            "beta_ses": dict(self.beta_ses),
            "beta_ts": dict(self.beta_ts),
            "beta_ps": dict(self.beta_ps),
            "beta_cis": {k: list(v) for k, v in self.beta_cis.items()},
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "durbin_watson": self.durbin_watson,
            "factor_names": list(self.factor_names),
            "n_obs": self.n_obs,
            "method": self.method,
            "hac": self.hac,
            "confidence_level": self.confidence_level,
            "t_stat_pct_above_2": self.t_stat_pct_above_2,
            "significant_factors": self.significant_factors,
        }

    def to_dataframe(self) -> pl.DataFrame:
        """Factor-level results as a Polars DataFrame."""
        rows = []
        for f in self.factor_names:
            ci = self.beta_cis.get(f, (float("nan"), float("nan")))
            rows.append(
                {
                    "factor": f,
                    "beta": self.betas[f],
                    "se": self.beta_ses[f],
                    "t_stat": self.beta_ts[f],
                    "p_value": self.beta_ps[f],
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                }
            )
        return pl.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Factor Model Results",
            "=" * 50,
            f"Method: {self.method.upper()}" + (" (HAC SEs)" if self.hac else ""),
            f"Observations: {self.n_obs}",
            f"R²: {self.r_squared:.4f}  Adj R²: {self.adj_r_squared:.4f}",
            f"Durbin-Watson: {self.durbin_watson:.3f}",
            f"|t| > 2: {self.t_stat_pct_above_2:.0%} of factors",
            "",
            f"Alpha: {self.alpha:.6f} (t={self.alpha_t:.2f}, p={self.alpha_p:.4f})",
            "",
            f"{'Factor':<15} {'Beta':>8} {'SE':>8} {'t-stat':>8} {'p-value':>8}",
            "-" * 55,
        ]
        for f in self.factor_names:
            lines.append(
                f"{f:<15} {self.betas[f]:>8.4f} {self.beta_ses[f]:>8.4f} "
                f"{self.beta_ts[f]:>8.2f} {self.beta_ps[f]:>8.4f}"
            )
        return "\n".join(lines)


@dataclass
class StabilityDiagnostics:
    """Diagnostics for rolling beta stability.

    Parameters
    ----------
    beta_std : dict[str, float]
        Standard deviation of rolling betas per factor.
    sign_consistency : dict[str, float]
        Fraction of windows where beta has same sign as full-sample.
    max_abs_change : dict[str, float]
        Maximum single-step absolute change in beta.
    vif : dict[str, float] | None
        Variance inflation factors (None if not computed).
    r_squared_mean : float
        Mean rolling R².
    r_squared_std : float
        Std of rolling R².
    """

    beta_std: dict[str, float]
    sign_consistency: dict[str, float]
    max_abs_change: dict[str, float]
    vif: dict[str, float] | None
    r_squared_mean: float
    r_squared_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta_std": dict(self.beta_std),
            "sign_consistency": dict(self.sign_consistency),
            "max_abs_change": dict(self.max_abs_change),
            "vif": dict(self.vif) if self.vif else None,
            "r_squared_mean": self.r_squared_mean,
            "r_squared_std": self.r_squared_std,
        }

    def to_dataframe(self) -> pl.DataFrame:
        factors = list(self.beta_std.keys())
        rows = []
        for f in factors:
            row: dict[str, Any] = {
                "factor": f,
                "beta_std": self.beta_std[f],
                "sign_consistency": self.sign_consistency[f],
                "max_abs_change": self.max_abs_change[f],
            }
            if self.vif is not None:
                row["vif"] = self.vif.get(f, float("nan"))
            rows.append(row)
        return pl.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "Stability Diagnostics",
            "=" * 50,
            f"R² mean: {self.r_squared_mean:.4f} (std: {self.r_squared_std:.4f})",
            "",
            f"{'Factor':<15} {'Beta Std':>10} {'Sign Cons':>10} {'Max Chg':>10}",
            "-" * 50,
        ]
        for f in self.beta_std:
            lines.append(
                f"{f:<15} {self.beta_std[f]:>10.4f} "
                f"{self.sign_consistency[f]:>10.1%} "
                f"{self.max_abs_change[f]:>10.4f}"
            )
        if self.vif:
            lines.extend(["", "VIF:"])
            for f, v in self.vif.items():
                flag = " ⚠" if v > 5 else ""
                lines.append(f"  {f}: {v:.2f}{flag}")
        return "\n".join(lines)


@dataclass
class RollingExposureResult:
    """Time-varying factor exposure estimates.

    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps for each rolling window.
    rolling_betas : dict[str, np.ndarray]
        Rolling beta time series per factor.
    rolling_alpha : np.ndarray
        Rolling alpha time series.
    rolling_r_squared : np.ndarray
        Rolling R² time series.
    stability : StabilityDiagnostics
        Summary stability metrics.
    window : int
        Window size used.
    factor_names : list[str]
        Factor names.
    """

    timestamps: np.ndarray
    rolling_betas: dict[str, np.ndarray]
    rolling_alpha: np.ndarray
    rolling_r_squared: np.ndarray
    stability: StabilityDiagnostics
    window: int
    factor_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "factor_names": list(self.factor_names),
            "n_windows": len(self.timestamps),
            "stability": self.stability.to_dict(),
        }

    def to_dataframe(self) -> pl.DataFrame:
        """Rolling betas as a long-format DataFrame."""
        ts_series = pl.Series("timestamp", self.timestamps)
        frames = []
        for f in self.factor_names:
            betas = self.rolling_betas[f]
            frame = pl.DataFrame(
                {
                    "timestamp": ts_series,
                    "factor": [f] * len(ts_series),
                    "beta": betas.tolist(),
                }
            )
            frames.append(frame)
        return pl.concat(frames)

    def to_wide_dataframe(self) -> pl.DataFrame:
        """Rolling betas as a wide-format DataFrame (one column per factor)."""
        cols: dict[str, Any] = {"timestamp": pl.Series("timestamp", self.timestamps)}
        for f in self.factor_names:
            cols[f] = self.rolling_betas[f].tolist()
        cols["alpha"] = self.rolling_alpha.tolist()
        cols["r_squared"] = self.rolling_r_squared.tolist()
        return pl.DataFrame(cols)

    def summary(self) -> str:
        lines = [
            "Rolling Exposure Results",
            "=" * 50,
            f"Window: {self.window}  Periods: {len(self.timestamps)}",
            "",
        ]
        lines.append(self.stability.summary())
        return "\n".join(lines)


@dataclass
class AttributionResult:
    """Return attribution with confidence intervals (Paleologo Ch 14).

    Parameters
    ----------
    timestamps : np.ndarray
        Attribution period timestamps.
    factor_contributions : dict[str, np.ndarray]
        Per-period factor contributions (beta[t-lag] * factor_return[t]).
    alpha_contribution : np.ndarray
        Per-period alpha contribution.
    residual : np.ndarray
        Per-period unexplained residual.
    cumulative_factor : dict[str, np.ndarray]
        Cumulative compounded factor contributions.
    cumulative_alpha : np.ndarray
        Cumulative compounded alpha.
    cumulative_residual : np.ndarray
        Cumulative compounded residual.
    cumulative_total : np.ndarray
        Cumulative total portfolio return.
    summary_pct : dict[str, float]
        Percentage of total return attributed to each factor.
    attribution_se : dict[str, float]
        Standard error per factor's attributed PnL (Paleologo Ch 14).
    attribution_ci : dict[str, tuple[float, float]]
        Confidence intervals on factor attribution.
    idiosyncratic_se : float
        Standard error of idiosyncratic (alpha + residual) PnL.
    factor_names : list[str]
        Factor names.
    window : int
        Rolling window used for beta estimation.
    lag : int
        Lag applied to betas.
    confidence_level : float
        Confidence level for CIs.
    """

    timestamps: np.ndarray
    factor_contributions: dict[str, np.ndarray]
    alpha_contribution: np.ndarray
    residual: np.ndarray
    cumulative_factor: dict[str, np.ndarray]
    cumulative_alpha: np.ndarray
    cumulative_residual: np.ndarray
    cumulative_total: np.ndarray
    summary_pct: dict[str, float]
    attribution_se: dict[str, float]
    attribution_ci: dict[str, tuple[float, float]]
    idiosyncratic_se: float
    factor_names: list[str]
    window: int
    lag: int
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_pct": dict(self.summary_pct),
            "attribution_se": dict(self.attribution_se),
            "attribution_ci": {k: list(v) for k, v in self.attribution_ci.items()},
            "idiosyncratic_se": self.idiosyncratic_se,
            "factor_names": list(self.factor_names),
            "window": self.window,
            "lag": self.lag,
            "confidence_level": self.confidence_level,
            "n_periods": len(self.timestamps),
        }

    def to_dataframe(self) -> pl.DataFrame:
        """Per-period attribution as DataFrame."""
        cols: dict[str, Any] = {"timestamp": pl.Series("timestamp", self.timestamps)}
        for f in self.factor_names:
            cols[f"contrib_{f}"] = self.factor_contributions[f].tolist()
        cols["contrib_alpha"] = self.alpha_contribution.tolist()
        cols["residual"] = self.residual.tolist()
        return pl.DataFrame(cols)

    def summary(self) -> str:
        cl_pct = f"{self.confidence_level:.0%}"
        lines = [
            "Return Attribution",
            "=" * 60,
            f"Window: {self.window}  Lag: {self.lag}  Periods: {len(self.timestamps)}",
            "",
            f"{'Source':<15} {'% of Return':>12} {'SE':>10} {cl_pct + ' CI':>20}",
            "-" * 60,
        ]
        for f in self.factor_names:
            pct = self.summary_pct.get(f, 0.0)
            se = self.attribution_se.get(f, 0.0)
            ci = self.attribution_ci.get(f, (float("nan"), float("nan")))
            lines.append(f"{f:<15} {pct:>11.1%} {se:>10.4f} [{ci[0]:>8.4f}, {ci[1]:>8.4f}]")
        alpha_pct = self.summary_pct.get("alpha", 0.0)
        resid_pct = self.summary_pct.get("residual", 0.0)
        lines.append(f"{'Alpha':<15} {alpha_pct:>11.1%}")
        lines.append(f"{'Residual':<15} {resid_pct:>11.1%} {self.idiosyncratic_se:>10.4f}")
        return "\n".join(lines)


@dataclass
class MaximalAttributionResult:
    """Maximal attribution resolving correlated-factor ambiguity (Paleologo Ch 14).

    Given a subset S of factors of interest and remaining factors U,
    computes the maximum PnL attributable to S by accounting for
    indirect exposure via correlated factors.

    Parameters
    ----------
    adjusted_betas : dict[str, float]
        Betas adjusted for correlation with non-interest factors.
    maximal_pnl : dict[str, float]
        Maximum PnL attributable to each factor of interest.
    orthogonal_residual : float
        PnL orthogonal to all factors of interest.
    rotation_matrix : np.ndarray
        The rotation matrix A = Omega_{U,S} @ inv(Omega_{S,S}).
    factors_of_interest : list[str]
        The subset S of factors.
    all_factor_names : list[str]
        All factor names.
    """

    adjusted_betas: dict[str, float]
    maximal_pnl: dict[str, float]
    orthogonal_residual: float
    rotation_matrix: np.ndarray
    factors_of_interest: list[str]
    all_factor_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjusted_betas": dict(self.adjusted_betas),
            "maximal_pnl": dict(self.maximal_pnl),
            "orthogonal_residual": self.orthogonal_residual,
            "factors_of_interest": list(self.factors_of_interest),
            "all_factor_names": list(self.all_factor_names),
        }

    def summary(self) -> str:
        lines = [
            "Maximal Attribution (Paleologo Ch 14)",
            "=" * 50,
            f"Factors of interest: {', '.join(self.factors_of_interest)}",
            "",
            f"{'Factor':<15} {'Adj Beta':>10} {'Max PnL':>10}",
            "-" * 40,
        ]
        for f in self.factors_of_interest:
            lines.append(f"{f:<15} {self.adjusted_betas[f]:>10.4f} {self.maximal_pnl[f]:>10.4f}")
        lines.append(f"\nOrthogonal residual: {self.orthogonal_residual:.4f}")
        return "\n".join(lines)


@dataclass
class RiskAttributionResult:
    """Variance-based risk decomposition.

    Parameters
    ----------
    total_variance : float
        Total portfolio variance.
    factor_variance : float
        Variance explained by factors (beta' * Sigma_F * beta).
    idiosyncratic_variance : float
        Unexplained variance.
    factor_contributions : dict[str, float]
        Absolute variance contribution per factor (Euler decomposition).
    factor_contributions_pct : dict[str, float]
        Percentage of total variance per factor.
    mctr : dict[str, float]
        Marginal contribution to risk per factor.
    factor_names : list[str]
        Factor names.
    shrinkage : str
        Covariance shrinkage method used.
    """

    total_variance: float
    factor_variance: float
    idiosyncratic_variance: float
    factor_contributions: dict[str, float]
    factor_contributions_pct: dict[str, float]
    mctr: dict[str, float]
    factor_names: list[str]
    shrinkage: str = "ledoit_wolf"

    @property
    def factor_variance_pct(self) -> float:
        if self.total_variance == 0:
            return 0.0
        return self.factor_variance / self.total_variance

    @property
    def idiosyncratic_variance_pct(self) -> float:
        if self.total_variance == 0:
            return 0.0
        return self.idiosyncratic_variance / self.total_variance

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_variance": self.total_variance,
            "factor_variance": self.factor_variance,
            "idiosyncratic_variance": self.idiosyncratic_variance,
            "factor_variance_pct": self.factor_variance_pct,
            "idiosyncratic_variance_pct": self.idiosyncratic_variance_pct,
            "factor_contributions": dict(self.factor_contributions),
            "factor_contributions_pct": dict(self.factor_contributions_pct),
            "mctr": dict(self.mctr),
            "factor_names": list(self.factor_names),
            "shrinkage": self.shrinkage,
        }

    def to_dataframe(self) -> pl.DataFrame:
        rows = []
        for f in self.factor_names:
            rows.append(
                {
                    "factor": f,
                    "variance_contribution": self.factor_contributions[f],
                    "variance_pct": self.factor_contributions_pct[f],
                    "mctr": self.mctr[f],
                }
            )
        return pl.DataFrame(rows)

    def summary(self) -> str:
        total_vol = np.sqrt(self.total_variance) if self.total_variance > 0 else 0.0
        lines = [
            "Risk Attribution",
            "=" * 55,
            f"Total volatility: {total_vol:.4f} (variance: {self.total_variance:.6f})",
            f"Factor variance: {self.factor_variance_pct:.1%}  "
            f"Idiosyncratic: {self.idiosyncratic_variance_pct:.1%}",
            f"Shrinkage: {self.shrinkage}",
            "",
            f"{'Factor':<15} {'Var Contrib':>12} {'% of Total':>10} {'MCTR':>8}",
            "-" * 50,
        ]
        for f in self.factor_names:
            lines.append(
                f"{f:<15} {self.factor_contributions[f]:>12.6f} "
                f"{self.factor_contributions_pct[f]:>9.1%} "
                f"{self.mctr[f]:>8.4f}"
            )
        return "\n".join(lines)


@dataclass
class FactorTimingResult:
    """Factor timing analysis (Tier 2).

    Parameters
    ----------
    correlations : dict[str, float]
        Spearman rank correlation of beta_k[t] with F_k[t+1].
    p_values : dict[str, float]
        P-values for timing correlations.
    factor_names : list[str]
        Factor names.
    window : int
        Rolling window used for beta estimation.
    """

    correlations: dict[str, float]
    p_values: dict[str, float]
    factor_names: list[str]
    window: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlations": dict(self.correlations),
            "p_values": dict(self.p_values),
            "factor_names": list(self.factor_names),
            "window": self.window,
        }

    def to_dataframe(self) -> pl.DataFrame:
        rows = []
        for f in self.factor_names:
            rows.append(
                {
                    "factor": f,
                    "timing_correlation": self.correlations[f],
                    "p_value": self.p_values[f],
                }
            )
        return pl.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            "Factor Timing Analysis",
            "=" * 45,
            f"Window: {self.window}",
            "",
            f"{'Factor':<15} {'Corr':>8} {'p-value':>10} {'Signal':>8}",
            "-" * 45,
        ]
        for f in self.factor_names:
            corr = self.correlations[f]
            p = self.p_values[f]
            sig = "Yes" if p < 0.05 else "No"
            lines.append(f"{f:<15} {corr:>8.4f} {p:>10.4f} {sig:>8}")
        return "\n".join(lines)


@dataclass
class ModelValidationResult:
    """Model quality diagnostics (Paleologo Ch 5).

    Parameters
    ----------
    qlike : float
        Quasi-likelihood loss (lower is better).
    malv : float
        Mean absolute log variance ratio.
    r_squared : float
        Standard R² for reference.
    t_stat_pct : float
        Fraction of |t| > 2.
    durbin_watson : float
        Durbin-Watson statistic (2.0 = no autocorrelation).
    ljung_box_p : float
        Ljung-Box p-value for residual autocorrelation.
    jarque_bera_p : float
        Jarque-Bera p-value for residual normality.
    condition_number : float
        Condition number of the design matrix.
    """

    qlike: float
    malv: float
    r_squared: float
    t_stat_pct: float
    durbin_watson: float
    ljung_box_p: float
    jarque_bera_p: float
    condition_number: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "qlike": self.qlike,
            "malv": self.malv,
            "r_squared": self.r_squared,
            "t_stat_pct": self.t_stat_pct,
            "durbin_watson": self.durbin_watson,
            "ljung_box_p": self.ljung_box_p,
            "jarque_bera_p": self.jarque_bera_p,
            "condition_number": self.condition_number,
        }

    def summary(self) -> str:
        lines = [
            "Model Validation (Paleologo Ch 5)",
            "=" * 45,
            f"QLIKE:            {self.qlike:.6f}",
            f"MALV:             {self.malv:.6f}",
            f"R²:               {self.r_squared:.4f}",
            f"|t| > 2:          {self.t_stat_pct:.0%}",
            f"Durbin-Watson:    {self.durbin_watson:.3f}",
            f"Ljung-Box p:      {self.ljung_box_p:.4f}"
            + (" ⚠ autocorr" if self.ljung_box_p < 0.05 else ""),
            f"Jarque-Bera p:    {self.jarque_bera_p:.4f}"
            + (" ⚠ non-normal" if self.jarque_bera_p < 0.05 else ""),
            f"Condition number: {self.condition_number:.1f}"
            + (" ⚠ multicollinear" if self.condition_number > 30 else ""),
        ]
        return "\n".join(lines)
