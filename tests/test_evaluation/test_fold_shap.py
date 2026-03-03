"""Tests for fold-aware SHAP computation.

Uses mocked shap.TreeExplainer to avoid requiring real LightGBM models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def feature_names() -> list[str]:
    return ["momentum_20d", "volatility_10d", "rsi_14", "volume_ratio"]


@pytest.fixture
def predictions_df() -> pl.DataFrame:
    """Predictions from 2 folds."""
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-02-01",
                    "2023-02-02",
                    "2023-02-03",
                ]
            ).str.to_datetime(),
            "symbol": ["AAPL", "MSFT", "AAPL", "AAPL", "MSFT", "AAPL"],
            "fold_id": [0, 0, 0, 1, 1, 1],
            "prediction": [0.5, 0.6, 0.4, 0.55, 0.45, 0.65],
        }
    )


@pytest.fixture
def features_df(feature_names: list[str]) -> pl.DataFrame:
    """Feature matrix covering both folds."""
    np.random.seed(42)
    n = 6
    data: dict = {
        "timestamp": pl.Series(
            [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-02-01",
                "2023-02-02",
                "2023-02-03",
            ]
        ).str.to_datetime(),
        "symbol": ["AAPL", "MSFT", "AAPL", "AAPL", "MSFT", "AAPL"],
    }
    for fname in feature_names:
        data[fname] = np.random.randn(n)
    return pl.DataFrame(data)


@pytest.fixture
def mock_boosters() -> dict[int, MagicMock]:
    """Mock LightGBM boosters for 2 folds."""
    return {0: MagicMock(name="booster_fold0"), 1: MagicMock(name="booster_fold1")}


def _make_mock_explainer(n_features: int):
    """Create a mock TreeExplainer that returns deterministic SHAP values."""

    def _side_effect(booster):
        mock_exp = MagicMock()

        def _shap_values(x):
            np.random.seed(42)
            return np.random.randn(x.shape[0], n_features) * 0.1

        mock_exp.shap_values = _shap_values
        return mock_exp

    return _side_effect


class TestComputeFoldShap:
    """Tests for compute_fold_shap()."""

    @patch("ml4t.diagnostic.evaluation.trade_shap.fold_shap._shap")
    def test_basic_shapes(
        self, mock_shap, predictions_df, features_df, feature_names, mock_boosters
    ):
        """Test output shapes are correct."""
        mock_shap.TreeExplainer = MagicMock(side_effect=_make_mock_explainer(len(feature_names)))

        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        aligned_df, shap_values = compute_fold_shap(
            boosters=mock_boosters,
            predictions_df=predictions_df,
            features_df=features_df,
            feature_names=feature_names,
        )

        assert aligned_df.height == 6
        assert shap_values.shape == (6, len(feature_names))
        assert set(aligned_df.columns) == {"timestamp", "symbol"} | set(feature_names)

    @patch("ml4t.diagnostic.evaluation.trade_shap.fold_shap._shap")
    def test_row_alignment(
        self, mock_shap, predictions_df, features_df, feature_names, mock_boosters
    ):
        """Test that features and SHAP values are row-aligned."""
        mock_shap.TreeExplainer = MagicMock(side_effect=_make_mock_explainer(len(feature_names)))

        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        aligned_df, shap_values = compute_fold_shap(
            boosters=mock_boosters,
            predictions_df=predictions_df,
            features_df=features_df,
            feature_names=feature_names,
        )

        # Rows should match prediction order
        assert aligned_df.height == shap_values.shape[0]

    def test_missing_booster_raises(self, predictions_df, features_df, feature_names):
        """Test that missing booster raises ValueError."""
        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        boosters = {0: MagicMock()}  # Missing fold 1

        with pytest.raises(ValueError, match="Missing boosters"):
            compute_fold_shap(
                boosters=boosters,
                predictions_df=predictions_df,
                features_df=features_df,
                feature_names=feature_names,
            )

    @patch("ml4t.diagnostic.evaluation.trade_shap.fold_shap._shap")
    def test_max_samples_per_fold(
        self, mock_shap, predictions_df, features_df, feature_names, mock_boosters
    ):
        """Test subsampling per fold."""
        mock_shap.TreeExplainer = MagicMock(side_effect=_make_mock_explainer(len(feature_names)))

        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        aligned_df, shap_values = compute_fold_shap(
            boosters=mock_boosters,
            predictions_df=predictions_df,
            features_df=features_df,
            feature_names=feature_names,
            max_samples_per_fold=2,
        )

        # Each fold has 3 rows, but we limit to 2 per fold => 4 total
        assert aligned_df.height == 4
        assert shap_values.shape[0] == 4

    def test_column_validation_predictions(self, features_df, feature_names, mock_boosters):
        """Test validation of predictions_df columns."""
        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        bad_preds = pl.DataFrame({"timestamp": [], "symbol": []})  # Missing fold_id

        with pytest.raises(ValueError, match="missing columns"):
            compute_fold_shap(
                boosters=mock_boosters,
                predictions_df=bad_preds,
                features_df=features_df,
                feature_names=feature_names,
            )

    def test_column_validation_features(self, predictions_df, mock_boosters):
        """Test validation of features_df columns."""
        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        bad_features = pl.DataFrame({"timestamp": [], "symbol": []})  # Missing feature cols

        with pytest.raises(ValueError, match="missing columns"):
            compute_fold_shap(
                boosters=mock_boosters,
                predictions_df=predictions_df,
                features_df=bad_features,
                feature_names=["momentum_20d"],
            )

    @patch("ml4t.diagnostic.evaluation.trade_shap.fold_shap._shap")
    def test_empty_fold_warning(self, mock_shap, features_df, feature_names, mock_boosters):
        """Test warning on empty fold (no matching features)."""
        mock_shap.TreeExplainer = MagicMock(side_effect=_make_mock_explainer(len(feature_names)))

        from ml4t.diagnostic.evaluation.trade_shap.fold_shap import compute_fold_shap

        # Predictions with timestamps that don't match features
        preds = pl.DataFrame(
            {
                "timestamp": pl.Series(["2099-01-01", "2099-01-02"]).str.to_datetime(),
                "symbol": ["AAPL", "MSFT"],
                "fold_id": [0, 0],
            }
        )
        # Add fold 0 booster only
        boosters = {0: MagicMock()}

        with pytest.raises(ValueError, match="No folds produced"):
            compute_fold_shap(
                boosters=boosters,
                predictions_df=preds,
                features_df=features_df,
                feature_names=feature_names,
            )
