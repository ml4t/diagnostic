"""ML4T Engineer integration contract for preprocessing recommendations.

This module defines the API contract between ML4T Diagnostic and ML4T Engineer for feature
preprocessing recommendations. After evaluating features, ML4T Diagnostic can recommend
transforms that ML4T Engineer should apply.

Example workflow:
    >>> from ml4t.diagnostic.evaluation import FeatureEvaluator
    >>> from ml4t.diagnostic.integration import EngineerConfig
    >>>
    >>> # 1. Evaluate features
    >>> evaluator = FeatureEvaluator(config)
    >>> results = evaluator.evaluate(features_df)
    >>>
    >>> # 2. Get preprocessing recommendations
    >>> eng_config = results.to_engineer_config()
    >>>
    >>> # 3. Export for ML4T Engineer
    >>> preprocessing_dict = eng_config.to_dict()
    >>>
    >>> # 4. Use with ML4T Engineer
    >>> # from ml4t.engineer import PreprocessingPipeline
    >>> # pipeline = PreprocessingPipeline(preprocessing_dict)
    >>> # transformed = pipeline.transform(features_df)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TransformType(str, Enum):
    """Supported transform types matching ML4T Engineer API.

    These transforms align with ML4T Engineer's PreprocessingPipeline.
    Each transform addresses specific statistical issues:

    - NONE: Feature is good as-is
    - LOG: Reduce right skew, stabilize variance
    - SQRT: Reduce right skew (milder than log)
    - STANDARDIZE: Zero mean, unit variance (z-score)
    - NORMALIZE: Scale to [0, 1] range
    - WINSORIZE: Cap outliers at percentiles
    - DIFF: First difference for non-stationary series
    """

    NONE = "none"
    LOG = "log"
    SQRT = "sqrt"
    STANDARDIZE = "standardize"
    NORMALIZE = "normalize"
    WINSORIZE = "winsorize"
    DIFF = "diff"


class PreprocessingRecommendation(BaseModel):
    """Recommendation for preprocessing a single feature.

    ML4T Diagnostic generates these recommendations based on diagnostics:
    - Stationarity tests → recommend differencing
    - Distribution analysis → recommend transforms for skew
    - Outlier detection → recommend winsorization
    - Scale issues → recommend normalization

    Attributes:
        feature_name: Name of the feature
        transform: Recommended transform type
        reason: Human-readable explanation of why this transform
        confidence: Confidence in recommendation [0.0, 1.0]
        diagnostics: Optional diagnostic details that led to recommendation

    Example:
        >>> rec = PreprocessingRecommendation(
        ...     feature_name="returns",
        ...     transform=TransformType.DIFF,
        ...     reason="Feature is non-stationary (ADF p=0.82)",
        ...     confidence=0.95
        ... )
    """

    feature_name: str = Field(..., description="Feature name")
    transform: TransformType = Field(..., description="Recommended transform")
    reason: str = Field(..., description="Explanation for recommendation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence [0.0, 1.0]")
    diagnostics: dict[str, float] | None = Field(
        None, description="Optional diagnostic values (e.g., {'adf_pvalue': 0.82})"
    )


class EngineerConfig(BaseModel):
    """Configuration for ML4T Engineer PreprocessingPipeline.

    This is the output format that ML4T Engineer can consume.
    Contains recommendations for all features that need preprocessing.

    Attributes:
        recommendations: List of feature preprocessing recommendations
        metadata: Optional metadata about evaluation context

    Example:
        >>> config = EngineerConfig(recommendations=[
        ...     PreprocessingRecommendation(
        ...         feature_name="rsi_14",
        ...         transform=TransformType.WINSORIZE,
        ...         reason="Outliers detected at 1st and 99th percentile",
        ...         confidence=0.85
        ...     ),
        ...     PreprocessingRecommendation(
        ...         feature_name="log_returns",
        ...         transform=TransformType.NONE,
        ...         reason="Already stationary and normally distributed",
        ...         confidence=0.90
        ...     )
        ... ])
        >>> eng_dict = config.to_dict()
    """

    recommendations: list[PreprocessingRecommendation] = Field(
        ..., description="Feature preprocessing recommendations"
    )
    metadata: dict[str, str] | None = Field(
        None, description="Optional metadata (e.g., eval timestamp, config)"
    )

    def to_dict(self) -> dict[str, dict[str, str | float | dict[str, float]]]:
        """Export to ML4T Engineer-compatible format.

        Returns dictionary mapping feature names to preprocessing configs:
            {
                "feature_name": {
                    "transform": "diff",
                    "reason": "Non-stationary",
                    "confidence": 0.95,
                    "diagnostics": {...}
                }
            }

        Returns:
            Dictionary in ML4T Engineer PreprocessingPipeline format

        Example:
            >>> config.to_dict()
            {
                'rsi_14': {
                    'transform': 'winsorize',
                    'reason': 'Outliers detected',
                    'confidence': 0.85
                }
            }
        """
        result: dict[str, dict[str, str | float | dict[str, float]]] = {}
        for rec in self.recommendations:
            feature_dict: dict[str, str | float | dict[str, float]] = {
                "transform": rec.transform.value,
                "reason": rec.reason,
                "confidence": rec.confidence,
            }
            if rec.diagnostics:
                feature_dict["diagnostics"] = rec.diagnostics
            result[rec.feature_name] = feature_dict
        return result

    def get_recommendations_by_transform(
        self, transform: TransformType
    ) -> list[PreprocessingRecommendation]:
        """Filter recommendations by transform type.

        Useful for analyzing patterns in recommendations:
        - How many features need differencing?
        - Which features can stay unchanged?

        Args:
            transform: Transform type to filter by

        Returns:
            List of recommendations with matching transform

        Example:
            >>> config.get_recommendations_by_transform(TransformType.DIFF)
            [PreprocessingRecommendation(feature_name='returns', ...)]
        """
        return [rec for rec in self.recommendations if rec.transform == transform]

    def summary(self) -> str:
        """Human-readable summary of recommendations.

        Returns:
            Formatted summary string

        Example:
            >>> print(config.summary())
            ML4T Engineer Preprocessing Recommendations
            ==========================================
            Total features: 5
            - DIFF: 2 features
            - WINSORIZE: 1 feature
            - NONE: 2 features
        """
        lines = ["ML4T Engineer Preprocessing Recommendations", "=" * 44]
        lines.append(f"Total features: {len(self.recommendations)}")
        lines.append("")

        # Count by transform type
        transform_counts: dict[TransformType, int] = {}
        for rec in self.recommendations:
            transform_counts[rec.transform] = transform_counts.get(rec.transform, 0) + 1

        # Sort by count (descending)
        for transform, count in sorted(transform_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {transform.value.upper()}: {count} features")

        # Show high-confidence recommendations
        high_conf = [rec for rec in self.recommendations if rec.confidence >= 0.9]
        if high_conf:
            lines.append("")
            lines.append(f"High-confidence recommendations (≥0.9): {len(high_conf)}")
            for rec in high_conf[:5]:  # Show top 5
                lines.append(
                    f"  {rec.feature_name}: {rec.transform.value} (conf={rec.confidence:.2f})"
                )
            if len(high_conf) > 5:
                lines.append(f"  ... and {len(high_conf) - 5} more")

        return "\n".join(lines)
