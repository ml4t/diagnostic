"""Tests for evaluation/trade_dashboard/normalize.py.

This module tests the data normalization functions for the trade dashboard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation.trade_dashboard.normalize import (
    _build_patterns_df,
    _build_trades_df,
    _extract_returns,
    _normalize_explanation,
    _parse_timestamp,
    _safe_float,
    normalize_result,
)
from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


class TestParseTimestamp:
    """Tests for _parse_timestamp helper function."""

    def test_none_input(self):
        """Test None input returns None."""
        assert _parse_timestamp(None) is None

    def test_datetime_passthrough(self):
        """Test datetime passes through unchanged."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert _parse_timestamp(dt) == dt

    def test_iso_format(self):
        """Test ISO format parsing."""
        result = _parse_timestamp("2024-01-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_iso_format_with_z(self):
        """Test ISO format with Z suffix."""
        result = _parse_timestamp("2024-01-15T10:30:00Z")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_common_datetime_format(self):
        """Test common datetime format."""
        result = _parse_timestamp("2024-01-15 10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_date_only_format(self):
        """Test date-only format."""
        result = _parse_timestamp("2024-01-15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_slash_date_format(self):
        """Test slash date format."""
        result = _parse_timestamp("2024/01/15")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_empty_string(self):
        """Test empty string returns None."""
        assert _parse_timestamp("") is None

    def test_na_string(self):
        """Test N/A string returns None."""
        assert _parse_timestamp("N/A") is None
        assert _parse_timestamp("None") is None

    def test_invalid_format(self):
        """Test invalid format returns None."""
        assert _parse_timestamp("not-a-date") is None
        assert _parse_timestamp("abc123") is None


class TestSafeFloat:
    """Tests for _safe_float helper function."""

    def test_none_input(self):
        """Test None returns None."""
        assert _safe_float(None) is None

    def test_float_passthrough(self):
        """Test float passes through."""
        assert _safe_float(3.14) == 3.14

    def test_int_conversion(self):
        """Test int converted to float."""
        assert _safe_float(42) == 42.0

    def test_string_number(self):
        """Test numeric string converted."""
        assert _safe_float("3.14") == 3.14

    def test_invalid_string(self):
        """Test invalid string returns None."""
        assert _safe_float("abc") is None
        assert _safe_float("N/A") is None

    def test_numpy_types(self):
        """Test numpy types converted."""
        assert _safe_float(np.float64(3.14)) == pytest.approx(3.14)
        assert _safe_float(np.int64(42)) == 42.0


class TestNormalizeExplanation:
    """Tests for _normalize_explanation helper function."""

    def test_basic_explanation(self):
        """Test basic explanation normalization."""
        exp = {
            "trade_id": "T001",
            "timestamp": "2024-01-15T10:30:00",
            "shap_vector": [0.1, -0.2, 0.3],
            "top_features": [("feature_a", 0.3), ("feature_b", -0.2)],
        }

        result = _normalize_explanation(exp)

        assert result["trade_id"] == "T001"
        assert isinstance(result["timestamp"], datetime)
        assert result["shap_vector"] == [0.1, -0.2, 0.3]
        assert result["top_features"] == [("feature_a", 0.3), ("feature_b", -0.2)]
        assert result["trade_metrics"] is None

    def test_with_trade_metrics(self):
        """Test explanation with trade metrics."""
        exp = {
            "trade_id": "T001",
            "trade_metrics": {
                "pnl": -100.5,
                "return_pct": -0.05,
                "entry_time": "2024-01-15T10:30:00",
                "exit_time": "2024-01-15T14:30:00",
                "duration_days": 0.17,
                "entry_price": 100.0,
                "exit_price": 95.0,
                "symbol": "AAPL",
            },
        }

        result = _normalize_explanation(exp)

        assert result["trade_metrics"] is not None
        tm = result["trade_metrics"]
        assert tm["pnl"] == -100.5
        assert tm["return_pct"] == -0.05
        assert isinstance(tm["entry_time"], datetime)
        assert tm["symbol"] == "AAPL"

    def test_missing_fields(self):
        """Test with missing optional fields."""
        exp = {"trade_id": "T001"}

        result = _normalize_explanation(exp)

        assert result["trade_id"] == "T001"
        assert result["timestamp"] is None
        assert result["shap_vector"] == []
        assert result["top_features"] == []


class TestBuildTradesDf:
    """Tests for _build_trades_df helper function."""

    def test_basic_trades(self):
        """Test building trades DataFrame."""
        explanations = [
            {
                "trade_id": "T001",
                "timestamp": datetime(2024, 1, 15, 10, 0, 0),
                "top_features": [("feature_a", 0.5)],
                "trade_metrics": {
                    "pnl": -100.0,
                    "return_pct": -0.05,
                    "entry_time": datetime(2024, 1, 15, 10, 0, 0),
                    "symbol": "AAPL",
                },
            },
            {
                "trade_id": "T002",
                "timestamp": datetime(2024, 1, 16, 11, 0, 0),
                "top_features": [("feature_b", -0.3)],
                "trade_metrics": {
                    "pnl": 50.0,
                    "return_pct": 0.02,
                    "entry_time": datetime(2024, 1, 16, 11, 0, 0),
                    "symbol": "GOOG",
                },
            },
        ]

        df = _build_trades_df(explanations)

        assert len(df) == 2
        assert "trade_id" in df.columns
        assert "pnl" in df.columns
        assert "top_feature" in df.columns
        assert df["trade_id"].tolist() == ["T001", "T002"]

    def test_empty_explanations(self):
        """Test with empty explanations list."""
        df = _build_trades_df([])

        assert df.empty
        assert "trade_id" in df.columns
        assert "pnl" in df.columns

    def test_top_feature_extraction(self):
        """Test top feature extraction."""
        explanations = [
            {
                "trade_id": "T001",
                "top_features": [("feature_a", 0.5), ("feature_b", 0.3)],
            },
        ]

        df = _build_trades_df(explanations)

        assert df["top_feature"].iloc[0] == "feature_a"
        assert df["top_shap_value"].iloc[0] == 0.5

    def test_no_top_features(self):
        """Test with no top features."""
        explanations = [
            {
                "trade_id": "T001",
                "top_features": [],
            },
        ]

        df = _build_trades_df(explanations)

        assert df["top_feature"].iloc[0] is None
        assert df["top_shap_value"].iloc[0] is None


class TestExtractReturns:
    """Tests for _extract_returns helper function."""

    def test_prefer_return_pct(self):
        """Test preference for return_pct over pnl."""
        df = pd.DataFrame(
            {
                "return_pct": [0.05, -0.02, 0.03],
                "pnl": [100, -50, 75],
            }
        )

        returns, label = _extract_returns(df)

        assert label == "return_pct"
        assert len(returns) == 3
        np.testing.assert_array_almost_equal(returns, [0.05, -0.02, 0.03])

    def test_fallback_to_pnl(self):
        """Test fallback to pnl when return_pct unavailable."""
        df = pd.DataFrame(
            {
                "pnl": [100, -50, 75],
            }
        )

        returns, label = _extract_returns(df)

        assert label == "pnl"
        assert len(returns) == 3

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        returns, label = _extract_returns(df)

        assert returns is None
        assert label == "none"

    def test_all_nan_return_pct(self):
        """Test with all NaN return_pct falls back to pnl."""
        df = pd.DataFrame(
            {
                "return_pct": [np.nan, np.nan],
                "pnl": [100, -50],
            }
        )

        returns, label = _extract_returns(df)

        assert label == "pnl"


class TestBuildPatternsDf:
    """Tests for _build_patterns_df helper function."""

    def test_dict_patterns(self):
        """Test building patterns DataFrame from dict patterns."""
        patterns = [
            {
                "cluster_id": 0,
                "n_trades": 10,
                "description": "Morning failures",
                "top_features": [("time_of_day", 0.5)],
                "hypothesis": "Early morning volatility",
                "actions": ["Avoid 9:30-10:00 trades"],
                "confidence": 0.85,
            },
            {
                "cluster_id": 1,
                "n_trades": 5,
                "description": "High volatility failures",
                "top_features": [("volatility", 0.6)],
            },
        ]

        df = _build_patterns_df(patterns)

        assert len(df) == 2
        assert "cluster_id" in df.columns
        assert "n_trades" in df.columns
        assert "description" in df.columns

    def test_empty_patterns(self):
        """Test with empty patterns list."""
        df = _build_patterns_df([])

        assert df.empty
        assert "cluster_id" in df.columns

    def test_object_patterns(self):
        """Test building from object patterns (using getattr)."""

        class MockPattern:
            cluster_id = 0
            n_trades = 5
            description = "Test pattern"
            top_features = [("feature", 0.5)]
            separation_score = 0.9
            distinctiveness = 0.8

        patterns = [MockPattern()]

        df = _build_patterns_df(patterns)

        assert len(df) == 1
        assert df["cluster_id"].iloc[0] == 0
        assert df["n_trades"].iloc[0] == 5


class TestNormalizeResult:
    """Tests for normalize_result main function."""

    def test_basic_dict_result(self):
        """Test normalizing a dict result."""
        result = {
            "n_trades_analyzed": 50,
            "n_trades_explained": 45,
            "n_trades_failed": 5,
            "explanations": [
                {
                    "trade_id": "T001",
                    "trade_metrics": {
                        "pnl": -100.0,
                        "return_pct": -0.05,
                        "entry_time": "2024-01-15T10:30:00",
                    },
                    "top_features": [("feature_a", 0.5)],
                }
            ],
            "failed_trades": [("T002", "Missing data")],
            "error_patterns": [
                {
                    "cluster_id": 0,
                    "n_trades": 10,
                    "description": "Test pattern",
                }
            ],
        }

        bundle = normalize_result(result)

        assert isinstance(bundle, DashboardBundle)
        assert bundle.n_trades_analyzed == 50
        assert bundle.n_trades_explained == 45
        assert bundle.n_trades_failed == 5
        assert len(bundle.explanations) == 1
        assert not bundle.trades_df.empty
        assert not bundle.patterns_df.empty

    def test_empty_result(self):
        """Test normalizing empty result."""
        result: dict[str, Any] = {
            "explanations": [],
            "error_patterns": [],
        }

        bundle = normalize_result(result)

        assert isinstance(bundle, DashboardBundle)
        assert bundle.trades_df.empty
        assert bundle.patterns_df.empty
        assert bundle.returns is None

    def test_chronological_sorting(self):
        """Test that trades are sorted chronologically."""
        result = {
            "explanations": [
                {
                    "trade_id": "T001",
                    "trade_metrics": {"entry_time": "2024-01-20T10:00:00"},
                },
                {
                    "trade_id": "T002",
                    "trade_metrics": {"entry_time": "2024-01-15T10:00:00"},
                },
                {
                    "trade_id": "T003",
                    "trade_metrics": {"entry_time": "2024-01-18T10:00:00"},
                },
            ],
        }

        bundle = normalize_result(result)

        # Should be sorted chronologically
        trade_ids = bundle.trades_df["trade_id"].tolist()
        assert trade_ids == ["T002", "T003", "T001"]  # Sorted by entry_time

    def test_returns_extraction(self):
        """Test that returns are properly extracted."""
        result = {
            "explanations": [
                {"trade_id": "T001", "trade_metrics": {"return_pct": 0.05}},
                {"trade_id": "T002", "trade_metrics": {"return_pct": -0.02}},
            ],
        }

        bundle = normalize_result(result)

        assert bundle.returns is not None
        assert bundle.returns_label == "return_pct"
        assert len(bundle.returns) == 2
