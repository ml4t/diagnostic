"""Result schemas for Enhanced Sharpe Ratio Framework.

Implements proper statistical evaluation of Sharpe ratios following
López de Prado, Lipton & Zoonekynd (2025) "How to Use the Sharpe Ratio".

Includes:
- Probabilistic Sharpe Ratio (PSR)
- Minimum Track Record Length (MinTRL)
- Deflated Sharpe Ratio (DSR)
- Bayesian False Discovery Rate (FDR)
"""

from __future__ import annotations

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult


class PSRResult(BaseResult):
    """Probabilistic Sharpe Ratio (PSR) results.

    PSR accounts for non-normality and sample length to compute the
    probability that the true Sharpe ratio exceeds a target.

    Reference: López de Prado et al. (2025), Equation 9

    Attributes:
        observed_sharpe: Observed Sharpe ratio from returns
        target_sharpe: Target Sharpe ratio to exceed
        psr_value: Probability that true SR > target
        confidence_level: Confidence level used (typically 0.95)
        skewness: Return skewness (affects PSR)
        kurtosis: Return excess kurtosis (affects PSR)
        n_observations: Sample size
    """

    analysis_type: str = "probabilistic_sharpe_ratio"

    observed_sharpe: float = Field(..., description="Observed Sharpe ratio")
    target_sharpe: float = Field(..., description="Target Sharpe to exceed")
    psr_value: float = Field(..., ge=0.0, le=1.0, description="P(true SR > target)")
    confidence_level: float = Field(default=0.95, description="Confidence level")

    # Distribution parameters
    skewness: float = Field(..., description="Return skewness")
    kurtosis: float = Field(..., description="Excess kurtosis")
    n_observations: int = Field(..., gt=0, description="Sample size")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get PSR results as single-row DataFrame.

        Returns:
            DataFrame with PSR statistics
        """
        data = {
            "observed_sharpe": [self.observed_sharpe],
            "target_sharpe": [self.target_sharpe],
            "psr": [self.psr_value],
            "confidence_level": [self.confidence_level],
            "skewness": [self.skewness],
            "kurtosis": [self.kurtosis],
            "n_obs": [self.n_observations],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of PSR analysis."""
        lines = ["Probabilistic Sharpe Ratio (PSR)", "=" * 40]
        lines.append(f"Observed Sharpe: {self.observed_sharpe:.3f}")
        lines.append(f"Target Sharpe: {self.target_sharpe:.3f}")
        lines.append(f"PSR: {self.psr_value:.1%}")
        lines.append("")
        lines.append(f"Sample size: {self.n_observations}")
        lines.append(f"Skewness: {self.skewness:.3f}")
        lines.append(f"Kurtosis: {self.kurtosis:.3f}")
        lines.append("")

        # Interpretation
        for interp in self.interpret():
            lines.append(interp)

        return "\n".join(lines)

    def interpret(self) -> list[str]:
        """Get interpretation of PSR results.

        Returns:
            List of interpretation strings
        """
        if self.psr_value >= 0.95:
            return ["Conclusion: High confidence that SR exceeds target"]
        elif self.psr_value >= 0.80:
            return ["Conclusion: Moderate confidence that SR exceeds target"]
        elif self.psr_value >= 0.50:
            return ["Conclusion: Weak evidence that SR exceeds target"]
        else:
            return ["Conclusion: Unlikely that SR exceeds target"]


class MinTRLResult(BaseResult):
    """Minimum Track Record Length (MinTRL) results.

    MinTRL computes the minimum sample size needed to reject the null
    hypothesis that true Sharpe <= target Sharpe at a given confidence level.

    Reference: López de Prado et al. (2025), Equation 11

    Attributes:
        observed_sharpe: Observed Sharpe ratio
        target_sharpe: Target Sharpe ratio
        min_trl_days: Minimum track record length (days) needed
        actual_days: Actual sample size (days)
        is_sufficient: Whether actual >= min_trl
        confidence_level: Confidence level (typically 0.95)
        skewness: Return skewness
        kurtosis: Return excess kurtosis
    """

    analysis_type: str = "minimum_track_record_length"

    observed_sharpe: float = Field(..., description="Observed Sharpe ratio")
    target_sharpe: float = Field(..., description="Target Sharpe ratio")

    min_trl_days: int = Field(..., gt=0, description="Minimum TRL in days")
    actual_days: int = Field(..., gt=0, description="Actual sample size in days")
    is_sufficient: bool = Field(..., description="Whether sample is sufficient")

    confidence_level: float = Field(default=0.95, description="Confidence level")
    skewness: float = Field(..., description="Return skewness")
    kurtosis: float = Field(..., description="Excess kurtosis")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get MinTRL results as single-row DataFrame.

        Returns:
            DataFrame with MinTRL statistics
        """
        data = {
            "observed_sharpe": [self.observed_sharpe],
            "target_sharpe": [self.target_sharpe],
            "min_trl_days": [self.min_trl_days],
            "actual_days": [self.actual_days],
            "is_sufficient": [self.is_sufficient],
            "confidence_level": [self.confidence_level],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of MinTRL analysis."""
        lines = ["Minimum Track Record Length (MinTRL)", "=" * 40]
        lines.append(f"Observed Sharpe: {self.observed_sharpe:.3f}")
        lines.append(f"Target Sharpe: {self.target_sharpe:.3f}")
        lines.append("")
        lines.append(f"Minimum TRL: {self.min_trl_days} days")
        lines.append(f"Actual sample: {self.actual_days} days")
        lines.append(f"Sufficient: {'Yes' if self.is_sufficient else 'No'}")
        lines.append("")

        # Interpretation
        if self.is_sufficient:
            lines.append("Conclusion: Sample size is adequate")
        else:
            shortfall = self.min_trl_days - self.actual_days
            lines.append(
                f"Conclusion: Need {shortfall} more days of data ({shortfall / 252:.1f} years)"
            )

        return "\n".join(lines)


class DSRResult(BaseResult):
    """Deflated Sharpe Ratio (DSR) results.

    DSR adjusts for backtest overfitting by correcting for multiple testing.
    Accounts for number of trials and variance across trials.

    Reference: Bailey & López de Prado (2014), López de Prado et al. (2025)

    Attributes:
        observed_sharpe: Observed Sharpe ratio
        dsr_value: Deflated Sharpe ratio (adjusted for multiple testing)
        adjusted_pvalue: P-value adjusted for FWER control
        is_significant: Whether DSR is significant at alpha level
        n_trials: Number of trials/strategies tested
        variance_trials: Variance of Sharpe ratios across trials
        alpha: Significance level (typically 0.05)
    """

    analysis_type: str = "deflated_sharpe_ratio"

    observed_sharpe: float = Field(..., description="Observed Sharpe ratio")
    dsr_value: float = Field(..., description="Deflated Sharpe ratio")
    adjusted_pvalue: float = Field(..., ge=0.0, le=1.0, description="FWER-adjusted p-value")
    is_significant: bool = Field(..., description="Is significant at alpha")

    n_trials: int = Field(..., gt=0, description="Number of trials tested")
    variance_trials: float = Field(..., ge=0.0, description="Variance of Sharpe across trials")
    alpha: float = Field(default=0.05, description="Significance level")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get DSR results as single-row DataFrame.

        Returns:
            DataFrame with DSR statistics
        """
        data = {
            "observed_sharpe": [self.observed_sharpe],
            "dsr": [self.dsr_value],
            "adjusted_pvalue": [self.adjusted_pvalue],
            "is_significant": [self.is_significant],
            "n_trials": [self.n_trials],
            "variance_trials": [self.variance_trials],
            "alpha": [self.alpha],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of DSR analysis."""
        lines = ["Deflated Sharpe Ratio (DSR)", "=" * 40]
        lines.append(f"Observed Sharpe: {self.observed_sharpe:.3f}")
        lines.append(f"Deflated Sharpe: {self.dsr_value:.3f}")
        lines.append(f"Adjusted p-value: {self.adjusted_pvalue:.4f}")
        lines.append("")
        lines.append(f"Number of trials: {self.n_trials}")
        lines.append(f"Trial variance: {self.variance_trials:.4f}")
        lines.append(f"Significance level: {self.alpha}")
        lines.append("")

        # Interpretation
        for interp in self.interpret():
            lines.append(interp)

        return "\n".join(lines)

    def interpret(self) -> list[str]:
        """Get interpretation of DSR results.

        Returns:
            List of interpretation strings with recommendations
        """
        interpretations = []
        if self.is_significant:
            interpretations.append("Conclusion: Significant after multiple testing correction")
            if self.observed_sharpe > 1.0:
                interpretations.append(
                    "Recommendation: Strong evidence of skill, proceed to paper trading"
                )
            else:
                interpretations.append(
                    "Recommendation: Modest but real edge, investigate improvements"
                )
        else:
            interpretations.append("Conclusion: Not significant (likely due to overfitting)")
            interpretations.append(
                "Recommendation: Revisit feature selection or reduce strategy complexity"
            )
        return interpretations


class FDRResult(BaseResult):
    """Bayesian False Discovery Rate (FDR) results.

    Computes probability of no skill given observed data using Bayesian inference.
    More intuitive than p-values!

    Reference: López de Prado et al. (2025), Equations 21 & 24

    Attributes:
        observed_sharpe: Observed Sharpe ratio
        null_sharpe: Sharpe under H0 (no skill)
        alternative_sharpe: Sharpe under H1 (has skill)
        prior_h0: Prior probability of no skill
        ofdr: Observed FDR - P(H0 | data)
        pfdr: Planned FDR - expected FDR at planning stage
    """

    analysis_type: str = "bayesian_fdr"

    observed_sharpe: float = Field(..., description="Observed Sharpe ratio")
    null_sharpe: float = Field(..., description="Sharpe under H0 (no skill)")
    alternative_sharpe: float = Field(..., description="Sharpe under H1 (skill)")

    prior_h0: float = Field(..., ge=0.0, le=1.0, description="Prior P(no skill)")
    ofdr: float = Field(..., ge=0.0, le=1.0, description="Observed FDR: P(H0 | data)")
    pfdr: float | None = Field(None, ge=0.0, le=1.0, description="Planned FDR (at design stage)")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get FDR results as single-row DataFrame.

        Returns:
            DataFrame with FDR statistics
        """
        data = {
            "observed_sharpe": [self.observed_sharpe],
            "null_sharpe": [self.null_sharpe],
            "alt_sharpe": [self.alternative_sharpe],
            "prior_h0": [self.prior_h0],
            "ofdr": [self.ofdr],
            "pfdr": [self.pfdr],
        }
        return pl.DataFrame(data)

    def summary(self) -> str:
        """Human-readable summary of FDR analysis."""
        lines = ["Bayesian False Discovery Rate (FDR)", "=" * 40]
        lines.append(f"Observed Sharpe: {self.observed_sharpe:.3f}")
        lines.append(f"H0 Sharpe: {self.null_sharpe:.3f}")
        lines.append(f"H1 Sharpe: {self.alternative_sharpe:.3f}")
        lines.append("")
        lines.append(f"Prior P(no skill): {self.prior_h0:.1%}")
        lines.append(f"Observed FDR - P(no skill | data): {self.ofdr:.1%}")
        if self.pfdr is not None:
            lines.append(f"Planned FDR: {self.pfdr:.1%}")
        lines.append("")

        # Interpretation
        if self.ofdr < 0.05:
            lines.append("Conclusion: Strong evidence of skill")
        elif self.ofdr < 0.20:
            lines.append("Conclusion: Moderate evidence of skill")
        elif self.ofdr < 0.50:
            lines.append("Conclusion: Weak evidence of skill")
        else:
            lines.append("Conclusion: Likely no skill (false discovery)")

        return "\n".join(lines)


class SharpeFrameworkResult(BaseResult):
    """Complete results from Enhanced Sharpe Framework.

    Comprehensive Sharpe ratio evaluation including:
    - PSR (non-normality adjustment)
    - MinTRL (sample adequacy)
    - DSR (multiple testing correction)
    - FDR (Bayesian interpretation)

    Attributes:
        psr: Probabilistic Sharpe Ratio results
        min_trl: Minimum Track Record Length results
        dsr: Deflated Sharpe Ratio results
        fdr_results: Bayesian FDR results
    """

    analysis_type: str = "sharpe_framework"

    psr: PSRResult | None = Field(None, description="PSR analysis")
    min_trl: MinTRLResult | None = Field(None, description="MinTRL analysis")
    dsr: DSRResult | None = Field(None, description="DSR analysis")
    fdr_results: FDRResult | None = Field(None, description="Bayesian FDR")

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get framework results as DataFrame.

        Args:
            name: 'psr', 'min_trl', 'dsr', or 'fdr'

        Returns:
            Requested DataFrame
        """
        if name == "psr" and self.psr:
            return self.psr.get_dataframe()
        elif name == "min_trl" and self.min_trl:
            return self.min_trl.get_dataframe()
        elif name == "dsr" and self.dsr:
            return self.dsr.get_dataframe()
        elif name == "fdr" and self.fdr_results:
            return self.fdr_results.get_dataframe()
        else:
            # Return combined summary
            rows = []
            if self.psr:
                rows.append(
                    {
                        "metric": "PSR",
                        "value": self.psr.psr_value,
                        "sharpe": self.psr.observed_sharpe,
                    }
                )
            if self.min_trl:
                rows.append(
                    {
                        "metric": "MinTRL",
                        "value": self.min_trl.min_trl_days,
                        "sharpe": self.min_trl.observed_sharpe,
                    }
                )
            if self.dsr:
                rows.append(
                    {
                        "metric": "DSR",
                        "value": self.dsr.dsr_value,
                        "sharpe": self.dsr.observed_sharpe,
                    }
                )
            if self.fdr_results:
                rows.append(
                    {
                        "metric": "oFDR",
                        "value": self.fdr_results.ofdr,
                        "sharpe": self.fdr_results.observed_sharpe,
                    }
                )
            return pl.DataFrame(rows) if rows else pl.DataFrame()

    def summary(self) -> str:
        """Human-readable summary of complete framework analysis."""
        lines = ["Enhanced Sharpe Framework Summary", "=" * 60]

        if self.psr:
            lines.append("")
            lines.append(self.psr.summary())

        if self.min_trl:
            lines.append("")
            lines.append(self.min_trl.summary())

        if self.dsr:
            lines.append("")
            lines.append(self.dsr.summary())

        if self.fdr_results:
            lines.append("")
            lines.append(self.fdr_results.summary())

        # Overall conclusion
        lines.append("")
        lines.append("Overall Assessment")
        lines.append("-" * 60)

        checks = []
        if self.psr and self.psr.psr_value >= 0.80:
            checks.append("PSR: High probability of exceeding target")
        if self.min_trl and self.min_trl.is_sufficient:
            checks.append("MinTRL: Adequate sample size")
        if self.dsr and self.dsr.is_significant:
            checks.append("DSR: Significant after multiple testing")
        if self.fdr_results and self.fdr_results.ofdr < 0.20:
            checks.append("FDR: Low probability of false discovery")

        if checks:
            lines.extend(checks)
        else:
            lines.append("⚠ Concerns about statistical significance")

        return "\n".join(lines)
