"""Validated Cross-Validation combining CPCV with DSR for robust strategy assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ml4t.diagnostic.config import StatisticalConfig, ValidatedCrossValidationConfig
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio_from_statistics
from ml4t.diagnostic.splitters.combinatorial import CombinatorialCV

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for models that can be fit and predict."""

    def fit(self, X: Any, y: Any) -> Any:
        """Fit the model."""
        ...

    def predict(self, X: Any) -> Any:
        """Make predictions."""
        ...


@dataclass
class ValidationFoldResult:
    """Result from a single cross-validation fold."""

    fold_idx: int
    train_size: int
    test_size: int
    sharpe_ratio: float
    returns: np.ndarray
    predictions: np.ndarray | None = None


@dataclass
class ValidationResult:
    """Complete result from validated cross-validation.

    Combines cross-validation performance with statistical significance testing.

    Attributes
    ----------
    fold_results : list[ValidationFoldResult]
        Results from each CV fold
    n_folds : int
        Number of folds completed
    mean_sharpe : float
        Mean Sharpe ratio across folds
    std_sharpe : float
        Standard deviation of Sharpe ratios
    dsr : float
        Deflated Sharpe Ratio (probability true SR > 0)
    dsr_zscore : float
        DSR z-score
    expected_max_sharpe : float
        Expected maximum Sharpe under null hypothesis
    is_significant : bool
        Whether DSR > significance threshold
    interpretation : list[str]
        Human-readable interpretation of results
    """

    fold_results: list[ValidationFoldResult] = field(default_factory=list)
    n_folds: int = 0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    dsr: float = 0.0
    dsr_zscore: float = 0.0
    expected_max_sharpe: float = 0.0
    is_significant: bool = False
    significance_level: float = 0.95
    interpretation: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = [
            "=" * 50,
            "Validated Cross-Validation Results",
            "=" * 50,
            "",
            f"Folds completed: {self.n_folds}",
            f"Mean Sharpe:     {self.mean_sharpe:.4f}",
            f"Std Sharpe:      {self.std_sharpe:.4f}",
            "",
            "--- Statistical Significance ---",
            f"DSR (probability true SR > 0): {self.dsr:.4f}",
            f"DSR z-score:                   {self.dsr_zscore:.4f}",
            f"Expected max SR under null:    {self.expected_max_sharpe:.4f}",
            f"Significant at {self.significance_level:.0%}:      {'YES' if self.is_significant else 'NO'}",
            "",
            "--- Interpretation ---",
        ]

        for interp in self.interpretation:
            lines.append(f"  - {interp}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export to dictionary.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "n_folds": self.n_folds,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "dsr": self.dsr,
            "dsr_zscore": self.dsr_zscore,
            "expected_max_sharpe": self.expected_max_sharpe,
            "is_significant": self.is_significant,
            "significance_level": self.significance_level,
            "interpretation": self.interpretation,
            "fold_sharpes": [fr.sharpe_ratio for fr in self.fold_results],
        }


class ValidatedCrossValidation:
    """Orchestrates CPCV with DSR computation for robust strategy validation.

    Combines Combinatorial Purged Cross-Validation with Deflated Sharpe Ratio
    to provide statistically rigorous assessment of trading strategies.

    This addresses the workflow fragmentation where users must manually:
    1. Run CPCV
    2. Collect Sharpe ratios
    3. Compute DSR
    4. Interpret results

    Examples
    --------
    >>> # Basic usage with model
    >>> vcv = ValidatedCrossValidation(config)
    >>> result = vcv.fit_evaluate(X, y, model, times=dates)
    >>> print(result.summary())

    >>> # With custom returns computation
    >>> def compute_returns(y_true, y_pred, prices):
    ...     positions = np.sign(y_pred)
    ...     returns = positions * y_true  # Simple return
    ...     return returns
    >>> result = vcv.fit_evaluate(X, y, model, times=dates, returns_fn=compute_returns)

    >>> # Just evaluate pre-computed fold Sharpes
    >>> result = vcv.evaluate_sharpes([0.5, 0.6, 0.4, 0.7, 0.3])
    """

    def __init__(
        self,
        config: ValidatedCrossValidationConfig | None = None,
        statistical_config: StatisticalConfig | None = None,
    ):
        """Initialize ValidatedCrossValidation.

        Parameters
        ----------
        config : ValidatedCrossValidationConfig, optional
            CV and evaluation configuration
        statistical_config : StatisticalConfig, optional
            Statistical testing configuration (for advanced DSR settings)
        """
        self.config = config or ValidatedCrossValidationConfig()
        self.statistical_config = statistical_config or StatisticalConfig()

        # Initialize CPCV splitter
        self._cv = CombinatorialCV(
            n_groups=self.config.n_groups,
            n_test_groups=self.config.n_test_groups,
            embargo_pct=self.config.embargo_pct,
            label_horizon=self.config.label_horizon,
        )

    def fit_evaluate(
        self,
        X: np.ndarray | pl.DataFrame,
        y: np.ndarray | pl.Series,
        model: ModelProtocol,
        times: np.ndarray | pl.Series | None = None,
        returns_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> ValidationResult:
        """Run cross-validation and compute DSR in one call.

        Parameters
        ----------
        X : array-like
            Features matrix
        y : array-like
            Target variable (or returns if returns_fn not provided)
        model : ModelProtocol
            Model with fit/predict interface
        times : array-like, optional
            Timestamps for purging. Required for temporal purging.
        returns_fn : callable, optional
            Function(y_true, y_pred) -> returns.
            If None, assumes y contains returns and predictions are positions.

        Returns
        -------
        ValidationResult
            Complete validation results with DSR
        """
        import polars as pl

        # Convert to numpy if needed
        if isinstance(X, pl.DataFrame):
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(y, pl.Series):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        if times is not None:
            if isinstance(times, pl.Series):
                times_np = times.to_numpy()
            else:
                times_np = np.asarray(times)
        else:
            times_np = None

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(self._cv.split(X_np, y_np, times_np)):
            # Fit model
            model.fit(X_np[train_idx], y_np[train_idx])

            # Get predictions
            predictions = model.predict(X_np[test_idx])

            # Compute returns
            if returns_fn is not None:
                fold_returns = returns_fn(y_np[test_idx], predictions)
            else:
                # Default: assume y is returns, predictions are signals
                fold_returns = np.sign(predictions) * y_np[test_idx]

            # Compute Sharpe
            sharpe = self._compute_sharpe(fold_returns)

            fold_results.append(
                ValidationFoldResult(
                    fold_idx=fold_idx,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    sharpe_ratio=sharpe,
                    returns=fold_returns,
                    predictions=predictions,
                )
            )

        return self._compute_validation_result(fold_results)

    def evaluate_sharpes(self, sharpe_ratios: list[float]) -> ValidationResult:
        """Evaluate pre-computed Sharpe ratios with DSR.

        Use when you've already computed Sharpe ratios from custom evaluation.

        Parameters
        ----------
        sharpe_ratios : list[float]
            Sharpe ratios from each CV fold or strategy

        Returns
        -------
        ValidationResult
            Complete validation results with DSR

        Examples
        --------
        >>> sharpes = [0.5, 0.6, 0.4, 0.7, 0.3, 0.55]
        >>> result = vcv.evaluate_sharpes(sharpes)
        >>> print(f"DSR: {result.dsr:.4f}")
        """
        fold_results = [
            ValidationFoldResult(
                fold_idx=i,
                train_size=0,
                test_size=0,
                sharpe_ratio=sr,
                returns=np.array([]),
            )
            for i, sr in enumerate(sharpe_ratios)
        ]
        return self._compute_validation_result(fold_results)

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio.

        Parameters
        ----------
        returns : np.ndarray
            Period returns

        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        # Annualize
        sharpe = (mean_ret / std_ret) * np.sqrt(self.config.annualization_factor)
        return float(sharpe)

    def _compute_validation_result(
        self, fold_results: list[ValidationFoldResult]
    ) -> ValidationResult:
        """Compute final validation result with DSR.

        Parameters
        ----------
        fold_results : list[ValidationFoldResult]
            Results from each fold

        Returns
        -------
        ValidationResult
            Complete validation result
        """
        sharpes = [fr.sharpe_ratio for fr in fold_results]
        n_folds = len(sharpes)

        if n_folds == 0:
            return ValidationResult(interpretation=["No folds completed"])

        mean_sharpe = float(np.mean(sharpes))
        std_sharpe = float(np.std(sharpes, ddof=1)) if n_folds > 1 else 0.0
        max_sharpe = float(np.max(sharpes))

        # Compute variance of Sharpes
        var_sharpes = std_sharpe**2 if n_folds > 1 else 0.0

        # Compute DSR
        # We use max_sharpe as the "observed" Sharpe (the one we'd select)
        dsr_result = deflated_sharpe_ratio_from_statistics(
            observed_sharpe=max_sharpe,
            n_trials=n_folds,
            variance_trials=var_sharpes,
            n_samples=252,  # Assume annual Sharpes
            skewness=0.0,  # Assume symmetric
            excess_kurtosis=0.0,  # Assume normal (Fisher convention: normal=0)
        )

        dsr = dsr_result.probability
        dsr_zscore = dsr_result.z_score
        expected_max = dsr_result.expected_max_sharpe

        is_significant = dsr > self.config.significance_level

        # Generate interpretation
        interpretation = self._generate_interpretation(
            mean_sharpe=mean_sharpe,
            max_sharpe=max_sharpe,
            expected_max=expected_max,
            dsr=dsr,
            is_significant=is_significant,
        )

        return ValidationResult(
            fold_results=fold_results,
            n_folds=n_folds,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            dsr=dsr,
            dsr_zscore=dsr_zscore,
            expected_max_sharpe=expected_max,
            is_significant=is_significant,
            significance_level=self.config.significance_level,
            interpretation=interpretation,
        )

    def _generate_interpretation(
        self,
        mean_sharpe: float,
        max_sharpe: float,
        expected_max: float,
        dsr: float,
        is_significant: bool,
    ) -> list[str]:
        """Generate human-readable interpretation.

        Parameters
        ----------
        mean_sharpe : float
            Mean Sharpe across folds
        max_sharpe : float
            Maximum observed Sharpe
        expected_max : float
            Expected max under null
        dsr : float
            Deflated Sharpe Ratio
        is_significant : bool
            Whether result is significant

        Returns
        -------
        list[str]
            Interpretation strings
        """
        interp = []

        # Significance assessment
        if is_significant:
            interp.append(
                f"Strategy is statistically significant (DSR={dsr:.2%} > {self.config.significance_level:.0%})"
            )
        else:
            interp.append(
                f"Strategy is NOT significant (DSR={dsr:.2%} < {self.config.significance_level:.0%})"
            )

        # Overfitting assessment
        inflation = max_sharpe - expected_max
        if inflation > 0:
            interp.append(
                f"Potential overfitting: observed SR ({max_sharpe:.3f}) exceeds null expectation ({expected_max:.3f}) by {inflation:.3f}"
            )
        else:
            interp.append("No obvious overfitting: observed SR below null expectation")

        # Mean vs max
        if max_sharpe > 2 * mean_sharpe and mean_sharpe > 0:
            interp.append("High variance in fold performance suggests unstable strategy")
        elif mean_sharpe > 0.5:
            interp.append("Consistent positive performance across folds")

        # Recommendation
        if is_significant and mean_sharpe > 0.3:
            interp.append(
                "Recommendation: Strategy shows robust performance, consider paper trading"
            )
        elif is_significant:
            interp.append(
                "Recommendation: Significant but modest returns, investigate improvements"
            )
        else:
            interp.append(
                "Recommendation: Strategy likely overfit, revisit feature selection or model"
            )

        return interp


# Convenience function
def validated_cross_val_score(
    model: ModelProtocol,
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray | None = None,
    n_groups: int = 10,
    embargo_pct: float = 0.01,
) -> ValidationResult:
    """Convenience function for validated cross-validation.

    Parameters
    ----------
    model : ModelProtocol
        Model with fit/predict interface
    X : np.ndarray
        Features
    y : np.ndarray
        Target (or returns)
    times : np.ndarray, optional
        Timestamps for purging
    n_groups : int, default 10
        Number of CV groups
    embargo_pct : float, default 0.01
        Embargo fraction

    Returns
    -------
    ValidationResult
        Validation results with DSR

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> result = validated_cross_val_score(
    ...     model=RandomForestClassifier(),
    ...     X=features,
    ...     y=returns,
    ...     times=dates,
    ... )
    >>> print(f"DSR: {result.dsr:.4f}")
    """
    config = ValidatedCrossValidationConfig(
        n_groups=n_groups,
        embargo_pct=embargo_pct,
    )
    vcv = ValidatedCrossValidation(config)
    return vcv.fit_evaluate(X, y, model, times)
