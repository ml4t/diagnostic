"""Importance data extraction for visualization layer.

Extracts comprehensive visualization data from feature importance analysis results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .types import (
    FeatureDetailData,
    ImportanceVizData,
    LLMContextData,
    MethodComparisonData,
    MethodImportanceData,
    UncertaintyData,
)
from .validation import _validate_lengths_match


def extract_importance_viz_data(
    importance_results: dict[str, Any],
    include_uncertainty: bool = True,
    include_distributions: bool = True,
    include_per_feature: bool = True,
    include_llm_context: bool = True,
) -> ImportanceVizData:
    """Extract comprehensive visualization data from importance analysis results.

    This function transforms raw importance analysis results into a structured
    format optimized for rich interactive visualization. It exposes all details
    including per-method breakdowns, uncertainty estimates, per-feature views,
    and auto-generated narratives.

    Parameters
    ----------
    importance_results : dict
        Results from analyze_ml_importance() containing:
        - 'consensus_ranking': list of features in importance order
        - 'method_results': dict of {method_name: method_result}
        - 'method_agreement': dict of pairwise correlations
        - 'interpretation': analysis interpretation
        - 'warnings': list of warning messages
    include_uncertainty : bool, default=True
        Whether to compute and include uncertainty metrics (stability, CI).
        Requires bootstrap or repeated analysis data.
    include_distributions : bool, default=True
        Whether to include full distributions (per-repeat values for PFI).
        Useful for detailed uncertainty visualization.
    include_per_feature : bool, default=True
        Whether to create per-feature aggregated views.
        Enables feature drill-down dashboards.
    include_llm_context : bool, default=True
        Whether to generate auto-narratives for LLM consumption.

    Returns
    -------
    ImportanceVizData
        Complete structured data package with all visualization details.
        See ImportanceVizData TypedDict for full structure.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization.data_extraction import extract_importance_viz_data
    >>>
    >>> # Analyze importance
    >>> results = analyze_ml_importance(model, X, y, methods=['mdi', 'pfi'])
    >>>
    >>> # Extract visualization data
    >>> viz_data = extract_importance_viz_data(results)
    >>>
    >>> # Access different views
    >>> print(viz_data['summary']['n_features'])  # High-level summary
    >>> print(viz_data['per_method']['mdi']['ranking'][:5])  # Top 5 by MDI
    >>> print(viz_data['per_feature']['momentum']['method_ranks'])  # Feature detail
    >>> print(viz_data['llm_context']['key_insights'])  # Auto-generated insights

    Notes
    -----
    - The extracted data is designed for both human visualization and LLM interpretation
    - Per-feature views enable drill-down dashboards
    - Uncertainty metrics enable confidence visualization
    - Auto-narratives prepare for future LLM integration
    """
    # Extract basic info
    consensus_ranking = importance_results.get("consensus_ranking", [])
    method_results = importance_results.get("method_results", {})
    method_agreement = importance_results.get("method_agreement", {})
    interpretation = importance_results.get("interpretation", {})
    warnings = importance_results.get("warnings", [])
    methods_run = importance_results.get("methods_run", list(method_results.keys()))

    n_features = len(consensus_ranking)
    n_methods = len(methods_run)

    # Build summary
    summary = _build_summary(
        consensus_ranking, method_agreement, methods_run, n_features, n_methods, warnings
    )

    # Extract per-method details
    per_method = _extract_per_method_data(
        method_results, include_distributions=include_distributions
    )

    # Build per-feature aggregations
    per_feature = {}
    if include_per_feature:
        per_feature = _build_per_feature_data(
            consensus_ranking, method_results, method_agreement, methods_run
        )

    # Compute uncertainty metrics
    uncertainty_data: UncertaintyData = {
        "method_stability": {},
        "rank_stability": {},
        "confidence_intervals": {},
        "coefficient_of_variation": {},
    }
    if include_uncertainty:
        uncertainty_data = _compute_uncertainty_metrics(method_results, consensus_ranking)

    # Build method comparison data
    method_comparison = _build_method_comparison(method_agreement, method_results, methods_run)

    # Build metadata
    metadata = {
        "n_features": n_features,
        "n_methods": n_methods,
        "methods_run": methods_run,
        "analysis_timestamp": datetime.now().isoformat(),
        "warnings": warnings,
        "interpretation": interpretation,
    }

    # Generate LLM context
    llm_context: LLMContextData = {
        "summary_narrative": "",
        "key_insights": [],
        "recommendations": [],
        "caveats": [],
        "analysis_quality": "medium",
    }
    if include_llm_context:
        llm_context = _generate_llm_context(
            summary, per_method, method_comparison, uncertainty_data, warnings
        )

    return ImportanceVizData(
        summary=summary,
        per_method=per_method,
        per_feature=per_feature,
        uncertainty=uncertainty_data,
        method_comparison=method_comparison,
        metadata=metadata,
        llm_context=llm_context,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _build_summary(
    consensus_ranking: list[str],
    method_agreement: dict[str, float],
    methods_run: list[str],
    n_features: int,
    n_methods: int,
    warnings: list[str],
) -> dict[str, Any]:
    """Build high-level summary statistics."""
    # Compute average agreement
    if method_agreement:
        avg_agreement = float(np.mean(list(method_agreement.values())))
    else:
        avg_agreement = 1.0 if n_methods == 1 else 0.0

    # Determine agreement level
    if avg_agreement > 0.8:
        agreement_level = "high"
    elif avg_agreement > 0.6:
        agreement_level = "medium"
    else:
        agreement_level = "low"

    return {
        "n_features": n_features,
        "n_methods": n_methods,
        "methods_run": methods_run,
        "top_feature": consensus_ranking[0] if consensus_ranking else None,
        "consensus_ranking": consensus_ranking,
        "avg_method_agreement": avg_agreement,
        "agreement_level": agreement_level,
        "has_warnings": len(warnings) > 0,
        "warnings_count": len(warnings),
    }


def _extract_per_method_data(
    method_results: dict[str, dict], include_distributions: bool = True
) -> dict[str, MethodImportanceData]:
    """Extract detailed per-method importance data with normalized values."""
    per_method: dict[str, MethodImportanceData] = {}

    for method_name, method_result in method_results.items():
        feature_names = method_result.get("feature_names", [])

        # Get importances based on method type
        if method_name == "pfi":
            importances_mean = method_result.get("importances_mean", [])
            importances_std = method_result.get("importances_std", [])
            importances_raw = method_result.get("importances_raw", [])

            # Validate length consistency for PFI data
            _validate_lengths_match(
                ("feature_names", feature_names),
                ("importances_mean", importances_mean),
                ("importances_std", importances_std),
            )

            # Normalize importances to sum to 1.0 (percentage basis)
            total = sum(importances_mean)
            if total > 0:
                importances_mean = [imp / total for imp in importances_mean]
                importances_std = [std / total for std in importances_std]

            # Convert to dicts (strict=True since we validated above)
            importances_dict = dict(zip(feature_names, importances_mean, strict=True))
            std_dict = dict(zip(feature_names, importances_std, strict=True))

            # Compute confidence intervals (95% assuming normal)
            # Use standard error (std / sqrt(n_repeats)) for CI of the mean
            n_repeats = method_result.get("n_repeats", 1)
            sqrt_n = np.sqrt(max(n_repeats, 1))
            ci_dict = {}
            for feat, mean, std in zip(
                feature_names, importances_mean, importances_std, strict=False
            ):
                se = std / sqrt_n  # Standard error of the mean
                ci_dict[feat] = (float(mean - 1.96 * se), float(mean + 1.96 * se))

            # Get raw values per repeat
            raw_list = None
            if include_distributions and importances_raw is not None and len(importances_raw) > 0:
                raw_list = []
                for repeat_values in importances_raw:
                    raw_list.append(dict(zip(feature_names, repeat_values, strict=False)))

            per_method[method_name] = MethodImportanceData(
                importances=importances_dict,
                ranking=sorted(feature_names, key=lambda f: importances_dict[f], reverse=True),
                std=std_dict,
                confidence_intervals=ci_dict,
                raw_values=raw_list,
                metadata={
                    "n_repeats": method_result.get("n_repeats", 1),
                    "scoring": method_result.get("scoring", "unknown"),
                },
            )

        else:
            # MDI, MDA, SHAP - single value per feature
            importances = method_result.get("importances", [])

            # Validate length consistency for non-PFI methods
            _validate_lengths_match(
                ("feature_names", feature_names),
                ("importances", importances),
            )

            # Normalize importances to sum to 1.0 (percentage basis)
            # MDI is already normalized, but SHAP and others may not be
            total = sum(importances)
            if total > 0 and abs(total - 1.0) > 0.01:  # Not already normalized
                importances = [imp / total for imp in importances]

            importances_dict = dict(zip(feature_names, importances, strict=True))

            per_method[method_name] = MethodImportanceData(
                importances=importances_dict,
                ranking=sorted(feature_names, key=lambda f: importances_dict[f], reverse=True),
                std=None,
                confidence_intervals=None,
                raw_values=None,
                metadata={},
            )

    return per_method


def _build_per_feature_data(
    consensus_ranking: list[str],
    method_results: dict[str, dict],
    _method_agreement: dict[str, float],
    methods_run: list[str],
) -> dict[str, FeatureDetailData]:
    """Build per-feature aggregated views for drill-down."""
    per_feature: dict[str, FeatureDetailData] = {}

    # Create importance and ranking dicts per method
    method_importances: dict[str, dict[str, float]] = {}
    method_rankings: dict[str, list[str]] = {}

    for method_name, method_result in method_results.items():
        feature_names = method_result.get("feature_names", [])

        if method_name == "pfi":
            importances = method_result.get("importances_mean", [])
        else:
            importances = method_result.get("importances", [])

        method_importances[method_name] = dict(zip(feature_names, importances, strict=False))
        method_rankings[method_name] = sorted(
            feature_names, key=lambda f: method_importances[method_name].get(f, 0), reverse=True
        )

    # Build per-feature data
    for consensus_rank, feature_name in enumerate(consensus_ranking, start=1):
        method_ranks = {}
        method_scores = {}
        method_stds = {}

        for method_name in methods_run:
            # Get rank in this method (with safe index lookup)
            try:
                ranking_list = method_rankings.get(method_name, [])
                method_ranks[method_name] = ranking_list.index(feature_name) + 1
            except ValueError:
                # Feature not found in ranking - assign last rank
                method_ranks[method_name] = len(method_rankings.get(method_name, [])) + 1

            # Get score in this method
            method_scores[method_name] = method_importances.get(method_name, {}).get(
                feature_name, 0.0
            )

            # Get std if available (PFI) - with bounds checking
            if method_name == "pfi":
                pfi_result = method_results.get("pfi", {})
                feature_names_pfi = pfi_result.get("feature_names", [])
                if feature_name in feature_names_pfi:
                    idx = feature_names_pfi.index(feature_name)
                    importances_std = pfi_result.get("importances_std", [])
                    # Check bounds before accessing
                    if idx < len(importances_std):
                        method_stds[method_name] = importances_std[idx]

        # Determine agreement level for this feature
        rank_variance = 0.0  # Initialize before conditional to avoid undefined
        if len(method_ranks) > 1:
            rank_variance = float(np.var(list(method_ranks.values())))
            if rank_variance < 2:
                agreement_level = "high"
            elif rank_variance < 10:
                agreement_level = "medium"
            else:
                agreement_level = "low"
        else:
            agreement_level = "n/a"

        # Compute stability score (inverse of rank variance, normalized)
        stability_score = 1.0 / (1.0 + rank_variance) if len(method_ranks) > 1 else 1.0

        # Generate interpretation
        interpretation = _generate_feature_interpretation(
            feature_name, consensus_rank, method_ranks, agreement_level
        )

        per_feature[feature_name] = FeatureDetailData(
            consensus_rank=consensus_rank,
            consensus_score=float(np.mean(list(method_scores.values()))),
            method_ranks=method_ranks,
            method_scores=method_scores,
            method_stds=method_stds,
            agreement_level=agreement_level,
            stability_score=float(stability_score),
            interpretation=interpretation,
        )

    return per_feature


def _compute_uncertainty_metrics(
    method_results: dict[str, dict], consensus_ranking: list[str]
) -> UncertaintyData:
    """Compute uncertainty and stability metrics."""
    # For now, focus on PFI which has repeat data
    pfi_result = method_results.get("pfi", {})
    has_pfi = bool(pfi_result)

    method_stability = {}
    confidence_intervals: dict[str, dict[str, tuple[float, float]]] = {}
    coefficient_of_variation: dict[str, dict[str, float]] = {}
    rank_stability: dict[str, list[int]] = {}

    if has_pfi:
        feature_names = pfi_result.get("feature_names", [])
        importances_mean = pfi_result.get("importances_mean", [])
        importances_std = pfi_result.get("importances_std", [])

        # Validate length consistency
        _validate_lengths_match(
            ("feature_names", feature_names),
            ("importances_mean", importances_mean),
            ("importances_std", importances_std),
        )

        # Method stability: average CV across features
        cvs = []
        cv_dict = {}
        for feat, mean, std in zip(feature_names, importances_mean, importances_std, strict=True):
            if mean != 0:
                cv = std / abs(mean)
                cvs.append(cv)
                cv_dict[feat] = float(cv)
            else:
                cv_dict[feat] = 0.0

        method_stability["pfi"] = float(1.0 - np.mean(cvs)) if cvs else 1.0
        coefficient_of_variation["pfi"] = cv_dict

        # Confidence intervals (use standard error for CI of the mean)
        n_repeats = pfi_result.get("n_repeats", 1)
        sqrt_n = np.sqrt(max(n_repeats, 1))
        ci_dict = {}
        for feat, mean, std in zip(feature_names, importances_mean, importances_std, strict=True):
            se = std / sqrt_n  # Standard error of the mean
            ci_dict[feat] = (float(mean - 1.96 * se), float(mean + 1.96 * se))
        confidence_intervals["pfi"] = ci_dict

        # Rank stability (if we had bootstrap data, we'd track rank distributions)
        # For now, mark as placeholder
        for feat in consensus_ranking:
            rank_stability[feat] = []  # Placeholder for bootstrap ranks

    return UncertaintyData(
        method_stability=method_stability,
        rank_stability=rank_stability,
        confidence_intervals=confidence_intervals,
        coefficient_of_variation=coefficient_of_variation,
    )


def _build_method_comparison(
    method_agreement: dict[str, float], method_results: dict[str, dict], methods_run: list[str]
) -> MethodComparisonData:
    """Build method comparison metrics."""
    # Build correlation matrix
    len(methods_run)
    correlation_matrix = []

    for method1 in methods_run:
        row = []
        for method2 in methods_run:
            if method1 == method2:
                row.append(1.0)
            else:
                # Find correlation in method_agreement dict
                key1 = f"{method1}_vs_{method2}"
                key2 = f"{method2}_vs_{method1}"
                corr = method_agreement.get(key1, method_agreement.get(key2, 0.0))
                row.append(float(corr))
        correlation_matrix.append(row)

    # Compute rank differences
    method_rankings: dict[str, list[str]] = {}
    for method_name, method_result in method_results.items():
        feature_names = method_result.get("feature_names", [])
        if method_name == "pfi":
            importances = method_result.get("importances_mean", [])
        else:
            importances = method_result.get("importances", [])

        # Validate length consistency
        _validate_lengths_match(
            ("feature_names", feature_names),
            ("importances", importances),
        )

        importances_dict = dict(zip(feature_names, importances, strict=True))
        ranking = sorted(feature_names, key=lambda f: importances_dict[f], reverse=True)
        method_rankings[method_name] = ranking

    rank_differences: dict[tuple[str, str], dict[str, int]] = {}
    for i, method1 in enumerate(methods_run):
        for method2 in methods_run[i + 1 :]:
            diff_dict = {}
            ranking1 = method_rankings.get(method1, [])
            ranking2 = method_rankings.get(method2, [])

            for feat in ranking1:
                if feat in ranking2:
                    rank1 = ranking1.index(feat) + 1
                    rank2 = ranking2.index(feat) + 1
                    diff_dict[feat] = abs(rank1 - rank2)

            rank_differences[(method1, method2)] = diff_dict

    return MethodComparisonData(
        correlation_matrix=correlation_matrix,
        correlation_methods=methods_run,
        rank_differences=rank_differences,
        agreement_summary=method_agreement,
    )


def _generate_feature_interpretation(
    feature_name: str, consensus_rank: int, method_ranks: dict[str, int], agreement_level: str
) -> str:
    """Generate auto-interpretation for a single feature."""
    if agreement_level == "high":
        return (
            f"'{feature_name}' ranks #{consensus_rank} with strong consensus across methods. "
            f"All methods agree on its importance level."
        )
    elif agreement_level == "medium":
        rank_str = ", ".join([f"{m}=#{r}" for m, r in method_ranks.items()])
        return (
            f"'{feature_name}' ranks #{consensus_rank} overall but shows moderate variation "
            f"across methods ({rank_str}). Consider investigating method-specific biases."
        )
    else:
        rank_str = ", ".join([f"{m}=#{r}" for m, r in method_ranks.items()])
        return (
            f"'{feature_name}' ranks #{consensus_rank} but shows significant disagreement "
            f"across methods ({rank_str}). This may indicate interaction effects or "
            f"method-specific artifacts. Further investigation recommended."
        )


def _generate_llm_context(
    summary: dict[str, Any],
    _per_method: dict[str, MethodImportanceData],
    _method_comparison: MethodComparisonData,
    uncertainty: UncertaintyData,
    warnings: list[str],
) -> LLMContextData:
    """Generate auto-narratives and insights for LLM consumption."""
    n_features = summary["n_features"]
    n_methods = summary["n_methods"]
    methods_run = summary["methods_run"]
    top_feature = summary["top_feature"]
    avg_agreement = summary["avg_method_agreement"]
    agreement_level = summary["agreement_level"]

    # Build summary narrative
    summary_narrative = (
        f"This feature importance analysis examined {n_features} features using "
        f"{n_methods} method{'s' if n_methods > 1 else ''} ({', '.join(methods_run)}). "
    )

    if top_feature:
        summary_narrative += (
            f"The consensus ranking identified '{top_feature}' as the most important feature. "
        )

    if n_methods > 1:
        summary_narrative += (
            f"Method agreement is {agreement_level} (average correlation: {avg_agreement:.2f}). "
        )

    # Generate key insights
    key_insights = []

    # Insight 1: Top features
    key_insights.append(
        f"Top consensus feature: '{top_feature}'"
        if top_feature
        else "No clear top feature identified"
    )

    # Insight 2: Method agreement
    if n_methods > 1:
        if agreement_level == "high":
            key_insights.append(
                f"Strong consensus across methods (avg correlation: {avg_agreement:.2f})"
            )
        elif agreement_level == "medium":
            key_insights.append(
                f"Moderate method agreement (avg correlation: {avg_agreement:.2f}) - some variation expected"
            )
        else:
            key_insights.append(
                f"Low method agreement (avg correlation: {avg_agreement:.2f}) - investigate method-specific biases"
            )

    # Insight 3: Stability (if available)
    if uncertainty.get("method_stability"):
        for method, stability in uncertainty["method_stability"].items():
            if stability < 0.7:
                key_insights.append(
                    f"{method.upper()} shows low stability (score: {stability:.2f}) - "
                    "importance estimates have high variance"
                )

    # Generate recommendations
    recommendations = []

    # Rec 1: Based on agreement
    if n_methods > 1 and avg_agreement < 0.6:
        recommendations.append(
            "Investigate features with large rank disagreements between methods. "
            "This may indicate interaction effects or method-specific artifacts."
        )

    # Rec 2: Based on stability
    if uncertainty.get("method_stability") and any(
        s < 0.7 for s in uncertainty["method_stability"].values()
    ):
        recommendations.append(
            "Increase number of repeats or use cross-validation to improve importance stability estimates."
        )

    # Rec 3: General best practice
    recommendations.append(
        "Focus on top consensus features for model interpretability and feature selection."
    )

    # Caveats
    caveats = []
    if warnings:
        caveats.append(f"Analysis generated {len(warnings)} warning(s) - review carefully.")

    if n_methods == 1:
        caveats.append(
            "Only one method used. Consider running multiple methods to validate findings."
        )

    # Determine overall quality
    if n_methods >= 2 and avg_agreement > 0.7 and len(warnings) == 0:
        analysis_quality = "high"
    elif n_methods >= 2 and avg_agreement > 0.5:
        analysis_quality = "medium"
    else:
        analysis_quality = "low"

    return LLMContextData(
        summary_narrative=summary_narrative,
        key_insights=key_insights,
        recommendations=recommendations,
        caveats=caveats,
        analysis_quality=analysis_quality,
    )
