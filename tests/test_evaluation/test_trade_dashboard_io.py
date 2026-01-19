"""Tests for evaluation/trade_dashboard/io.py.

This module tests the file I/O functions for the trade dashboard.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml4t.diagnostic.evaluation.trade_dashboard.io import (
    PickleDisabledError,
    _explanation_to_dict,
    _pattern_to_dict,
    coerce_result_to_dict,
    load_result_from_upload,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


class TestLoadResultFromUpload:
    """Tests for load_result_from_upload function."""

    def test_json_file_loading(self):
        """Test loading a JSON file."""
        data = {"n_trades_analyzed": 50, "explanations": []}
        content = json.dumps(data).encode("utf-8")
        uploaded = MockUploadedFile("result.json", content)

        result = load_result_from_upload(uploaded)

        assert result == data
        assert result["n_trades_analyzed"] == 50

    def test_json_nested_data(self):
        """Test loading JSON with nested data."""
        data = {
            "explanations": [
                {"trade_id": "T001", "trade_metrics": {"pnl": -100.0}},
            ],
            "error_patterns": [
                {"cluster_id": 0, "n_trades": 10},
            ],
        }
        content = json.dumps(data).encode("utf-8")
        uploaded = MockUploadedFile("result.json", content)

        result = load_result_from_upload(uploaded)

        assert len(result["explanations"]) == 1
        assert result["explanations"][0]["trade_id"] == "T001"

    def test_pickle_disabled_by_default(self):
        """Test that pickle files are disabled by default."""
        import pickle

        data = {"test": "data"}
        content = pickle.dumps(data)
        uploaded = MockUploadedFile("result.pkl", content)

        with pytest.raises(PickleDisabledError, match="disabled for security"):
            load_result_from_upload(uploaded)

    def test_pickle_disabled_error_message(self):
        """Test pickle disabled error message content."""
        uploaded = MockUploadedFile("result.pickle", b"dummy")

        with pytest.raises(PickleDisabledError) as exc_info:
            load_result_from_upload(uploaded, allow_pickle=False)

        assert "arbitrary code" in str(exc_info.value)
        assert "JSON format" in str(exc_info.value)

    def test_pickle_enabled_with_flag(self):
        """Test that pickle works when explicitly enabled."""
        import pickle

        data = {"test": "data", "value": 42}
        content = pickle.dumps(data)
        uploaded = MockUploadedFile("result.pkl", content)

        result = load_result_from_upload(uploaded, allow_pickle=True)

        assert result == data
        assert result["value"] == 42

    def test_unsupported_file_format(self):
        """Test error on unsupported file format."""
        uploaded = MockUploadedFile("result.txt", b"some text")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_result_from_upload(uploaded)

    def test_invalid_json(self):
        """Test error on invalid JSON."""
        uploaded = MockUploadedFile("result.json", b"not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_result_from_upload(uploaded)


class TestCoerceResultToDict:
    """Tests for coerce_result_to_dict function."""

    def test_dict_passthrough(self):
        """Test that dict passes through unchanged."""
        data = {"n_trades_analyzed": 50, "explanations": []}

        result = coerce_result_to_dict(data)

        assert result == data
        assert result is data  # Same object

    def test_object_conversion(self):
        """Test converting object to dict."""

        class MockExplanation:
            trade_id = "T001"
            timestamp = "2024-01-15"
            shap_vector = [0.1, -0.2]
            top_features = [("feature_a", 0.5)]
            trade_metrics = None

        class MockPattern:
            cluster_id = 0
            n_trades = 5
            description = "Test"
            top_features = [("feature", 0.3)]
            separation_score = 0.9
            distinctiveness = 0.8
            hypothesis = "Test hypothesis"
            actions = ["Action 1"]
            confidence = 0.85

        class MockResult:
            n_trades_analyzed = 50
            n_trades_explained = 45
            n_trades_failed = 5
            explanations = [MockExplanation()]
            failed_trades = [("T002", "Error")]
            error_patterns = [MockPattern()]

        result = coerce_result_to_dict(MockResult())

        assert result["n_trades_analyzed"] == 50
        assert result["n_trades_explained"] == 45
        assert result["n_trades_failed"] == 5
        assert len(result["explanations"]) == 1
        assert len(result["error_patterns"]) == 1


class TestExplanationToDict:
    """Tests for _explanation_to_dict helper function."""

    def test_dict_passthrough(self):
        """Test that dict passes through."""
        exp = {"trade_id": "T001", "shap_vector": [0.1]}

        result = _explanation_to_dict(exp)

        assert result == exp

    def test_object_conversion(self):
        """Test converting object to dict."""

        class MockExp:
            trade_id = "T001"
            timestamp = "2024-01-15"
            shap_vector = [0.1, -0.2]
            top_features = [("feature_a", 0.5)]
            trade_metrics = None

        result = _explanation_to_dict(MockExp())

        assert result["trade_id"] == "T001"
        assert result["timestamp"] == "2024-01-15"
        assert result["shap_vector"] == [0.1, -0.2]

    def test_with_trade_metrics(self):
        """Test conversion with trade_metrics."""

        class MockMetrics:
            pnl = -100.0
            return_pct = -0.05
            entry_time = "2024-01-15 10:00:00"
            exit_time = "2024-01-15 14:00:00"
            duration_days = 0.17
            entry_price = 100.0
            exit_price = 95.0
            symbol = "AAPL"

        class MockExp:
            trade_id = "T001"
            trade_metrics = MockMetrics()

        result = _explanation_to_dict(MockExp())

        assert "trade_metrics" in result
        assert result["trade_metrics"]["pnl"] == -100.0
        assert result["trade_metrics"]["symbol"] == "AAPL"


class TestPatternToDict:
    """Tests for _pattern_to_dict helper function."""

    def test_dict_passthrough(self):
        """Test that dict passes through."""
        pattern = {"cluster_id": 0, "n_trades": 10}

        result = _pattern_to_dict(pattern)

        assert result == pattern

    def test_object_conversion(self):
        """Test converting object to dict."""

        class MockPattern:
            cluster_id = 0
            n_trades = 5
            description = "Test pattern"
            top_features = [("feature", 0.3)]
            separation_score = 0.9
            distinctiveness = 0.8
            hypothesis = "Test hypothesis"
            actions = ["Action 1", "Action 2"]
            confidence = 0.85

        result = _pattern_to_dict(MockPattern())

        assert result["cluster_id"] == 0
        assert result["n_trades"] == 5
        assert result["description"] == "Test pattern"
        assert result["hypothesis"] == "Test hypothesis"
        assert result["actions"] == ["Action 1", "Action 2"]

    def test_missing_optional_fields(self):
        """Test with missing optional fields."""

        class MockPattern:
            cluster_id = 0
            n_trades = 5

        result = _pattern_to_dict(MockPattern())

        assert result["cluster_id"] == 0
        assert result["n_trades"] == 5
        assert result["description"] == ""  # Default
        assert result["actions"] == []  # Default


class TestPickleDisabledError:
    """Tests for PickleDisabledError exception."""

    def test_is_exception(self):
        """Test that PickleDisabledError is an exception."""
        assert issubclass(PickleDisabledError, Exception)

    def test_can_be_raised(self):
        """Test that it can be raised with message."""
        with pytest.raises(PickleDisabledError, match="test message"):
            raise PickleDisabledError("test message")
