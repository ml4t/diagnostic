"""Feature-outcome relationship analysis: Comprehensive IC diagnostics.

This module provides the main entry point for evaluating feature predictive power,
combining IC analysis, significance testing, monotonicity validation, and decay analysis.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from ml4t.diagnostic.evaluation.metrics.basic import compute_forward_returns
from ml4t.diagnostic.evaluation.metrics.ic_statistics import (
    compute_ic_decay,
    compute_ic_hac_stats,
    compute_ic_summary_stats,
)
from ml4t.diagnostic.evaluation.metrics.information_coefficient import (
    compute_ic_ir,
    compute_ic_series,
    information_coefficient,
)
from ml4t.diagnostic.evaluation.metrics.monotonicity import compute_monotonicity

if TYPE_CHECKING:
    pass


def analyze_feature_outcome_series(
    predictions: np.ndarray | pd.Series | pl.Series,
    outcomes: np.ndarray | pd.Series | pl.Series,
    *,
    method: str = "spearman",
    include_monotonicity: bool = False,
    n_quantiles: int = 5,
) -> dict[str, Any]:
    """Analyze single-series feature-outcome relationship.

    This helper is the canonical metrics-layer engine for one feature aligned
    to one outcome series.
    """
    pred_array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    out_array = np.asarray(outcomes, dtype=np.float64).reshape(-1)

    if len(pred_array) != len(out_array):
        raise ValueError(
            f"Predictions ({len(pred_array)}) and outcomes ({len(out_array)}) must have same length"
        )

    valid_mask = ~(np.isnan(pred_array) | np.isnan(out_array))
    pred_clean = pred_array[valid_mask]
    out_clean = out_array[valid_mask]
    n_observations = int(len(pred_clean))

    if n_observations == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "p_value": np.nan,
            "n_observations": 0,
            "monotonicity_analysis": None,
        }

    if n_observations < 3:
        # Keep contract stable for small samples while signaling weak evidence.
        ic_val = float(information_coefficient(pred_clean, out_clean, method=method))
        return {
            "ic_mean": ic_val,
            "ic_std": float(np.std(pred_clean)),
            "ic_ir": float(ic_val / (np.std(pred_clean) + 1e-10)),
            "p_value": 1.0,
            "n_observations": n_observations,
            "monotonicity_analysis": None,
        }

    from scipy.stats import pearsonr, spearmanr

    method_lower = method.lower()
    if method_lower == "spearman":
        ic_mean, p_value = spearmanr(pred_clean, out_clean)
    elif method_lower == "pearson":
        ic_mean, p_value = pearsonr(pred_clean, out_clean)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'.")

    ic_mean_f = float(ic_mean)
    ic_std_f = float(np.std(pred_clean))
    ic_ir_f = float(ic_mean_f / (ic_std_f + 1e-10))

    monotonicity_analysis: dict[str, Any] | None = None
    if include_monotonicity:
        monotonicity_analysis = compute_monotonicity(
            features=pred_clean,
            outcomes=out_clean,
            n_quantiles=n_quantiles,
            method=method_lower,
        )

    return {
        "ic_mean": ic_mean_f,
        "ic_std": ic_std_f,
        "ic_ir": ic_ir_f,
        "p_value": float(p_value),
        "n_observations": n_observations,
        "monotonicity_analysis": monotonicity_analysis,
    }


def analyze_feature_outcome(
    predictions: pl.DataFrame | pd.DataFrame,
    prices: pl.DataFrame | pd.DataFrame,
    pred_col: str = "prediction",
    price_col: str = "close",
    date_col: str = "date",
    group_col: str | None = None,
    horizons: list[int] | None = None,
    n_quantiles: int = 5,
    method: str = "spearman",
    include_decay: bool = True,
    include_monotonicity: bool = True,
    include_hac: bool = True,
    annualization_factor: float = np.sqrt(252),
) -> dict[str, Any]:
    """Comprehensive feature-outcome relationship analysis (FR-C1-C4).

    This is the main diagnostic function that combines IC analysis, significance
    testing, monotonicity validation, and decay analysis into a single comprehensive
    summary of feature quality.

    Use this function as the primary entry point for evaluating whether a feature
    (prediction/signal) has predictive power for outcomes (returns).

    Parameters
    ----------
    predictions : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with predictions, must have pred_col, date_col, and optionally group_col
    prices : Union[pl.DataFrame, pd.DataFrame]
        DataFrame with prices, must have price_col, date_col, and optionally group_col
    pred_col : str, default "prediction"
        Column name for predictions
    price_col : str, default "close"
        Column name for prices
    date_col : str, default "date"
        Column name for dates
    group_col : str | None, default None
        Column name for grouping (e.g., "symbol" for multi-asset)
    horizons : list[int] | None, default None
        List of forward horizons in days for multi-horizon analysis.
        If None, uses [1, 2, 5, 10, 21] for decay analysis
    n_quantiles : int, default 5
        Number of quantile bins for monotonicity analysis
    method : str, default "spearman"
        Correlation method: "spearman" or "pearson"
    include_decay : bool, default True
        Whether to compute IC decay analysis
    include_monotonicity : bool, default True
        Whether to compute monotonicity analysis
    include_hac : bool, default True
        Whether to compute HAC-adjusted significance
    annualization_factor : float, default sqrt(252)
        Factor to annualize IC-IR (sqrt(periods_per_year))

    Returns
    -------
    dict[str, Any]
        Comprehensive analysis dictionary with:
        - ic_summary: Core IC statistics (mean, std, IR, significance)
        - ic_series: Time series of IC values
        - decay_analysis: IC decay across horizons (if include_decay=True)
        - monotonicity_analysis: Quantile-based monotonicity (if include_monotonicity=True)
        - interpretation: Textual interpretation and guidance
        - metadata: Analysis parameters and timestamps

    Examples
    --------
    >>> # Comprehensive feature analysis
    >>> analysis = analyze_feature_outcome(
    ...     predictions=pred_df,
    ...     prices=price_df,
    ...     group_col="symbol",
    ...     horizons=[1, 2, 5, 10, 21]
    ... )
    >>>
    >>> # Check core statistics
    >>> print(f"Mean IC: {analysis['ic_summary']['mean_ic']:.4f}")
    >>> print(f"IC-IR: {analysis['ic_summary']['ic_ir']:.2f}")
    >>> print(f"P-value: {analysis['ic_summary']['p_value']:.4f}")
    >>> print(f"Significant: {analysis['ic_summary']['is_significant']}")
    Mean IC: 0.0234
    IC-IR: 1.12
    P-value: 0.0327
    Significant: True
    >>>
    >>> # Check decay characteristics
    >>> print(f"Half-life: {analysis['decay_analysis']['half_life']:.1f} days")
    >>> print(f"Optimal horizon: {analysis['decay_analysis']['optimal_horizon']} days")
    Half-life: 8.3 days
    Optimal horizon: 1 days
    >>>
    >>> # Check monotonicity
    >>> print(f"Monotonic: {analysis['monotonicity_analysis']['is_monotonic']}")
    >>> print(f"Direction: {analysis['monotonicity_analysis']['direction']}")
    Monotonic: True
    Direction: increasing
    >>>
    >>> # Read interpretation guidance
    >>> print(analysis['interpretation'])
    FEATURE QUALITY: GOOD
    - Mean IC: 0.0234 (positive predictive power)
    - IC-IR: 1.12 (excellent consistency)
    - Statistical Significance: p < 0.05 (robust)
    - Monotonicity: Increasing (valid predictor)
    - Signal Persistence: Moderate (half-life 8.3 days)
    RECOMMENDATION: Feature shows strong predictive power with good consistency.
    Consider using for short-to-medium term predictions (1-10 days).

    Notes
    -----
    This function is designed to be the primary entry point for feature evaluation,
    combining multiple analyses into a comprehensive assessment. For more focused
    analysis, use individual functions:
    - compute_ic_series(): Time series IC only
    - compute_ic_ir(): Information ratio only
    - compute_ic_decay(): Decay analysis only
    - compute_monotonicity(): Monotonicity only
    - compute_ic_hac_stats(): Significance testing only

    Quality Thresholds:
    - Mean IC: >0.02 is good, >0.05 is excellent
    - IC-IR: >0.5 is good, >1.0 is excellent
    - P-value: <0.05 for significance
    - Monotonicity score: >0.8 for strong monotonicity
    - Half-life: Depends on strategy horizon (align with holding period)
    """
    output_as_pandas = isinstance(predictions, pd.DataFrame)
    predictions_pl = (
        predictions if isinstance(predictions, pl.DataFrame) else pl.from_pandas(predictions)
    )
    prices_pl = prices if isinstance(prices, pl.DataFrame) else pl.from_pandas(prices)

    # 1. Compute forward returns from prices using compute_forward_returns
    prices_with_fwd = compute_forward_returns(
        prices=prices_pl,
        periods=1,  # 1-day forward returns for IC series
        price_col=price_col,
        group_col=group_col,
    )
    prices_with_fwd_pl = (
        prices_with_fwd
        if isinstance(prices_with_fwd, pl.DataFrame)
        else pl.from_pandas(prices_with_fwd)
    )

    # 2. Merge predictions with returns
    merge_cols = [date_col, group_col] if group_col else [date_col]
    merged = predictions_pl.join(prices_with_fwd_pl, on=merge_cols, how="inner").filter(
        pl.col("fwd_ret_1").is_not_null()
    )

    # 3. Compute IC time series (cross-sectional IC per date)
    # For panel data, compute IC by grouping on date and correlating across assets
    if group_col:
        ic_series_pl = compute_ic_series(
            predictions=merged.select([date_col, group_col, pred_col]),
            returns=merged.select([date_col, group_col, "fwd_ret_1"]),
            pred_col=pred_col,
            ret_col="fwd_ret_1",
            date_col=date_col,
            entity_col=group_col,
            method=method,
            min_periods=2,
        )
    else:
        ic_series_pl = compute_ic_series(
            predictions=merged.select([date_col, pred_col]),
            returns=merged.select([date_col, "fwd_ret_1"]),
            pred_col=pred_col,
            ret_col="fwd_ret_1",
            date_col=date_col,
            method=method,
        )
    ic_series: pl.DataFrame | pd.DataFrame = (
        ic_series_pl.to_pandas() if output_as_pandas else ic_series_pl
    )

    # 4. Compute IC-IR (Information Ratio)
    ic_ir_result = compute_ic_ir(
        ic_series=ic_series,
        ic_col="ic",
        annualization_factor=annualization_factor,
        confidence_intervals=True,
    )

    # 5. Compute HAC-adjusted significance (if requested)
    if include_hac:
        hac_stats = compute_ic_hac_stats(ic_series=ic_series, ic_col="ic")
    else:
        # Fallback to canonical non-HAC summary statistics.
        summary = compute_ic_summary_stats(ic_series=ic_series, ic_col="ic")
        n_periods = int(summary["n_periods"])
        hac_se = summary["std_ic"] / np.sqrt(n_periods) if n_periods >= 2 else np.nan
        hac_stats = {
            "mean_ic": summary["mean_ic"],
            "hac_se": float(hac_se),
            "t_stat": summary["t_stat"],
            "p_value": summary["p_value"],
            "n_periods": n_periods,
        }

    # 6. Compute IC decay analysis (if requested)
    decay_analysis = None
    if include_decay:
        decay_analysis = compute_ic_decay(
            predictions=predictions,
            prices=prices,
            horizons=horizons,
            pred_col=pred_col,
            price_col=price_col,
            date_col=date_col,
            group_col=group_col,
            method=method,
            estimate_half_life=True,
        )

    # 7. Compute monotonicity analysis (if requested)
    monotonicity_analysis = None
    if include_monotonicity:
        monotonicity_analysis = compute_monotonicity(
            features=merged[pred_col].to_numpy(),
            outcomes=merged["fwd_ret_1"].to_numpy(),
            n_quantiles=n_quantiles,
            method=method,
        )

    # 8. Build comprehensive summary
    # Extract IC values for std calculation - explicitly convert to float array
    if isinstance(ic_series, pl.DataFrame):
        ic_values_for_std = np.asarray(ic_series["ic"].to_numpy(), dtype=np.float64)
    elif isinstance(ic_series, pd.DataFrame):
        ic_values_for_std = np.asarray(ic_series["ic"].to_numpy(), dtype=np.float64)
    else:
        raise TypeError(f"ic_series must be DataFrame, got {type(ic_series)}")

    ic_summary = {
        "mean_ic": hac_stats["mean_ic"],
        "std_ic": float(np.std(ic_values_for_std, ddof=1)),
        "ic_ir": ic_ir_result["ic_ir"] if isinstance(ic_ir_result, dict) else ic_ir_result,
        "ic_ir_lower_ci": ic_ir_result.get("lower_ci") if isinstance(ic_ir_result, dict) else None,
        "ic_ir_upper_ci": ic_ir_result.get("upper_ci") if isinstance(ic_ir_result, dict) else None,
        "t_stat": hac_stats["t_stat"],
        "p_value": hac_stats["p_value"],
        "is_significant": hac_stats["p_value"] < 0.05,
        "n_periods": hac_stats["n_periods"],
        "fraction_positive": float(np.mean(ic_values_for_std > 0)),
    }

    # 9. Generate interpretation guidance
    interpretation = _generate_interpretation(
        ic_summary=ic_summary,
        decay_analysis=decay_analysis,
        monotonicity_analysis=monotonicity_analysis,
    )

    # 10. Build final result
    result = {
        "ic_summary": ic_summary,
        "ic_series": ic_series,
        "interpretation": interpretation,
        "metadata": {
            "analysis_date": datetime.now(UTC).isoformat(),
            "method": method,
            "n_quantiles": n_quantiles,
            "horizons": horizons or [1, 2, 5, 10, 21],
            "include_decay": include_decay,
            "include_monotonicity": include_monotonicity,
            "include_hac": include_hac,
        },
    }

    if decay_analysis is not None:
        result["decay_analysis"] = decay_analysis

    if monotonicity_analysis is not None:
        result["monotonicity_analysis"] = monotonicity_analysis

    return result


def _generate_interpretation(
    ic_summary: dict[str, Any],
    decay_analysis: dict[str, Any] | None,
    monotonicity_analysis: dict[str, Any] | None,
) -> str:
    """Generate human-readable interpretation of feature-outcome analysis.

    Parameters
    ----------
    ic_summary : dict
        IC summary statistics
    decay_analysis : dict | None
        IC decay analysis results
    monotonicity_analysis : dict | None
        Monotonicity analysis results

    Returns
    -------
    str
        Multi-line interpretation text
    """
    lines = []

    # Determine overall quality
    mean_ic = ic_summary["mean_ic"]
    ic_ir = ic_summary["ic_ir"]
    is_sig = ic_summary["is_significant"]

    if abs(mean_ic) > 0.05 and ic_ir > 1.0 and is_sig:
        quality = "EXCELLENT"
    elif abs(mean_ic) > 0.02 and ic_ir > 0.5 and is_sig:
        quality = "GOOD"
    elif abs(mean_ic) > 0.01 and is_sig:
        quality = "MODERATE"
    else:
        quality = "WEAK"

    lines.append(f"FEATURE QUALITY: {quality}")
    lines.append("")

    # IC statistics
    lines.append(
        f"- Mean IC: {mean_ic:.4f} ({'positive' if mean_ic > 0 else 'negative'} predictive power)"
    )
    lines.append(
        f"- IC-IR: {ic_ir:.2f} ({'excellent' if ic_ir > 1.0 else 'good' if ic_ir > 0.5 else 'moderate'} consistency)"
    )
    lines.append(
        f"- Statistical Significance: p = {ic_summary['p_value']:.4f} ({'robust' if is_sig else 'not significant'})"
    )

    # Monotonicity
    if monotonicity_analysis:
        is_mono = monotonicity_analysis["is_monotonic"]
        direction = monotonicity_analysis["direction"]
        score = monotonicity_analysis["monotonicity_score"]
        lines.append(
            f"- Monotonicity: {direction.replace('_', ' ').title()} (score: {score:.2f}, {'valid' if is_mono or score > 0.8 else 'weak'})"
        )

    # Decay characteristics
    if decay_analysis and decay_analysis.get("half_life"):
        half_life = decay_analysis["half_life"]
        if half_life < 5:
            persistence = "Short-term"
        elif half_life < 20:
            persistence = "Moderate"
        else:
            persistence = "Long-term"
        lines.append(f"- Signal Persistence: {persistence} (half-life {half_life:.1f} days)")

    lines.append("")

    # Recommendation
    if quality in ["EXCELLENT", "GOOD"]:
        if decay_analysis and decay_analysis.get("half_life"):
            hl = decay_analysis["half_life"]
            horizon_rec = (
                f"short-to-medium term predictions (1-{int(hl * 2)} days)"
                if hl < 10
                else f"medium-to-long term predictions ({int(hl)}-{int(hl * 3)} days)"
            )
        else:
            horizon_rec = "predictions aligned with signal strength"

        lines.append(
            f"RECOMMENDATION: Feature shows {quality.lower()} predictive power with {'excellent' if ic_ir > 1 else 'good'} consistency."
        )
        lines.append(f"Consider using for {horizon_rec}.")
    elif quality == "MODERATE":
        lines.append("RECOMMENDATION: Feature shows moderate predictive power.")
        lines.append(
            "Consider combining with other features or transforming (e.g., ranking, winsorization)."
        )
    else:
        lines.append("RECOMMENDATION: Feature shows weak predictive power.")
        lines.append(
            "Investigate data quality, consider feature transformations, or exclude from model."
        )

    return "\n".join(lines)
