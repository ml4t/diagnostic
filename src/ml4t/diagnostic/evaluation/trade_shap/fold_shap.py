"""Fold-aware SHAP value computation for walk-forward CV models.

Computes SHAP values from per-fold boosters and predictions, producing
row-aligned outputs ready for TradeShapPipeline consumption.

Typical usage with book case studies:
    >>> boosters = {0: lgb_fold0, 1: lgb_fold1, ...}
    >>> features_df, shap_values = compute_fold_shap(
    ...     boosters=boosters,
    ...     predictions_df=predictions,
    ...     features_df=features,
    ...     feature_names=feature_cols,
    ... )
    >>> pipeline = TradeShapPipeline(features_df, shap_values, feature_cols)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import shap as _shap
except ImportError:
    _shap = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray


def compute_fold_shap(
    boosters: dict[int, Any],
    predictions_df: pl.DataFrame,
    features_df: pl.DataFrame,
    feature_names: list[str],
    *,
    entity_col: str = "symbol",
    date_col: str = "timestamp",
    explainer_type: str = "auto",
    max_samples_per_fold: int | None = None,
    show_progress: bool = False,
) -> tuple[pl.DataFrame, NDArray[np.floating]]:
    """Compute SHAP values across walk-forward CV folds.

    For each fold_id in predictions_df, joins with features_df, runs
    shap.TreeExplainer on the corresponding booster, and concatenates
    all results in prediction order.

    Parameters
    ----------
    boosters : dict[int, Any]
        Mapping from fold_id to trained model (e.g. LightGBM Booster).
    predictions_df : pl.DataFrame
        Predictions with columns [date_col, entity_col, "fold_id", ...].
    features_df : pl.DataFrame
        Feature matrix with columns [date_col, entity_col] + feature_names.
    feature_names : list[str]
        Feature column names to extract for SHAP computation.
    entity_col : str
        Entity/symbol column name (default: "symbol").
    date_col : str
        Date/timestamp column name (default: "timestamp").
    explainer_type : str
        SHAP explainer type: "auto", "tree", or "linear" (default: "auto").
    max_samples_per_fold : int, optional
        If set, subsample each fold to this many rows (for speed).
    show_progress : bool
        Whether to print progress per fold (default: False).

    Returns
    -------
    tuple[pl.DataFrame, NDArray[np.floating]]
        - aligned_features_df: Polars DataFrame with [date_col, entity_col]
          + feature_names, row-aligned with shap_values.
        - shap_values: numpy array of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If required columns are missing or fold_ids have no booster.
    ImportError
        If shap is not installed.
    """
    import polars as pl

    # Validate columns
    _validate_columns(predictions_df, features_df, feature_names, date_col, entity_col)

    # Validate boosters cover all fold_ids
    fold_ids = sorted(predictions_df["fold_id"].unique().to_list())
    missing_boosters = [fid for fid in fold_ids if fid not in boosters]
    if missing_boosters:
        raise ValueError(
            f"Missing boosters for fold_ids: {missing_boosters}. "
            f"Available: {sorted(boosters.keys())}"
        )

    if _shap is None:
        raise ImportError(
            "shap is required for compute_fold_shap. Install with:\n  pip install shap>=0.41.0"
        )

    all_features_parts: list[pl.DataFrame] = []
    all_shap_parts: list[NDArray[np.floating]] = []

    for fold_id in fold_ids:
        if show_progress:
            print(f"  Fold {fold_id}...", end=" ", flush=True)

        # Filter predictions to this fold
        fold_preds = predictions_df.filter(pl.col("fold_id") == fold_id)

        if fold_preds.height == 0:
            warnings.warn(
                f"Fold {fold_id}: no predictions, skipping.",
                stacklevel=2,
            )
            continue

        # Inner join with features
        join_cols = [date_col, entity_col]
        joined = fold_preds.select(join_cols).join(
            features_df.select(join_cols + feature_names),
            on=join_cols,
            how="inner",
        )

        if joined.height == 0:
            warnings.warn(
                f"Fold {fold_id}: no matching features for {fold_preds.height} predictions, "
                f"skipping.",
                stacklevel=2,
            )
            continue

        # Optional subsampling
        if max_samples_per_fold is not None and joined.height > max_samples_per_fold:
            joined = joined.sample(n=max_samples_per_fold, seed=fold_id)

        # Extract feature matrix
        x_fold = joined.select(feature_names).to_numpy()

        # Create explainer
        booster = boosters[fold_id]
        if explainer_type == "auto" or explainer_type == "tree":
            explainer = _shap.TreeExplainer(booster)
        elif explainer_type == "linear":
            explainer = _shap.LinearExplainer(booster, x_fold)
        else:
            raise ValueError(f"Unknown explainer_type: {explainer_type!r}")

        sv = explainer.shap_values(x_fold)

        # Handle multi-output (classification returns list)
        if isinstance(sv, list):
            # Binary classification: take class 1 (positive class)
            sv = sv[1] if len(sv) == 2 else sv[0]

        sv = np.asarray(sv, dtype=np.float64)

        # Store aligned features (with date/entity cols for downstream alignment)
        aligned_part = joined.select(join_cols + feature_names)
        all_features_parts.append(aligned_part)
        all_shap_parts.append(sv)

        if show_progress:
            print(f"{joined.height} samples")

    if not all_features_parts:
        raise ValueError("No folds produced SHAP values. Check predictions/features alignment.")

    aligned_features_df = pl.concat(all_features_parts)
    shap_values = np.concatenate(all_shap_parts, axis=0)

    return aligned_features_df, shap_values


def _validate_columns(
    predictions_df: pl.DataFrame,
    features_df: pl.DataFrame,
    feature_names: list[str],
    date_col: str,
    entity_col: str,
) -> None:
    """Validate required columns exist in DataFrames."""
    # predictions_df must have [date_col, entity_col, fold_id]
    pred_required = {date_col, entity_col, "fold_id"}
    pred_missing = pred_required - set(predictions_df.columns)
    if pred_missing:
        raise ValueError(f"predictions_df missing columns: {pred_missing}")

    # features_df must have [date_col, entity_col] + feature_names
    feat_required = {date_col, entity_col} | set(feature_names)
    feat_missing = feat_required - set(features_df.columns)
    if feat_missing:
        raise ValueError(f"features_df missing columns: {feat_missing}")
