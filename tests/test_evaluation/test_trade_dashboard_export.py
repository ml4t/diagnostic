"""Tests for evaluation/trade_dashboard/export/ modules.

This module tests the CSV and HTML export functions for the trade dashboard.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.evaluation.trade_dashboard.export.csv import (
    export_patterns_csv,
    export_trades_csv,
)
from ml4t.diagnostic.evaluation.trade_dashboard.export.html import export_html_report
from ml4t.diagnostic.evaluation.trade_dashboard.types import (
    DashboardBundle,
    DashboardConfig,
)


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Create sample trades DataFrame."""
    return pd.DataFrame(
        {
            "trade_id": ["T001", "T002", "T003"],
            "symbol": ["AAPL", "GOOG", "MSFT"],
            "entry_time": [
                datetime(2024, 1, 15, 10, 0),
                datetime(2024, 1, 16, 11, 0),
                datetime(2024, 1, 17, 9, 0),
            ],
            "exit_time": [
                datetime(2024, 1, 15, 14, 0),
                datetime(2024, 1, 16, 15, 0),
                datetime(2024, 1, 17, 13, 0),
            ],
            "pnl": [-100.0, 50.0, -75.0],
            "return_pct": [-0.05, 0.02, -0.03],
            "duration_days": [0.17, 0.17, 0.17],
            "entry_price": [100.0, 2000.0, 300.0],
            "exit_price": [95.0, 2040.0, 291.0],
            "top_feature": ["volatility", "momentum", "trend"],
            "top_shap_value": [0.5, -0.3, 0.2],
        }
    )


@pytest.fixture
def sample_patterns_df() -> pd.DataFrame:
    """Create sample patterns DataFrame."""
    return pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "n_trades": [10, 5, 3],
            "description": ["Morning failures", "High vol failures", "News events"],
            "hypothesis": ["Early volatility", "VIX spikes", "Earnings surprise"],
            "confidence": [0.85, 0.72, 0.90],
            "separation_score": [0.9, 0.8, 0.95],
            "distinctiveness": [0.88, 0.75, 0.92],
        }
    )


@pytest.fixture
def sample_bundle(sample_trades_df, sample_patterns_df) -> DashboardBundle:
    """Create sample DashboardBundle."""
    returns = np.array([-0.05, 0.02, -0.03])
    return DashboardBundle(
        trades_df=sample_trades_df,
        returns=returns,
        returns_label="return_pct",
        explanations=[],
        patterns_df=sample_patterns_df,
        n_trades_analyzed=50,
        n_trades_explained=45,
        n_trades_failed=5,
        failed_trades=[("T004", "Missing data")],
        config=DashboardConfig(),
    )


@pytest.fixture
def empty_bundle() -> DashboardBundle:
    """Create empty DashboardBundle."""
    return DashboardBundle(
        trades_df=pd.DataFrame(),
        returns=None,
        returns_label="none",
        explanations=[],
        patterns_df=pd.DataFrame(),
        n_trades_analyzed=0,
        n_trades_explained=0,
        n_trades_failed=0,
        failed_trades=[],
        config=DashboardConfig(),
    )


class TestExportTradesCsv:
    """Tests for export_trades_csv function."""

    def test_basic_export(self, sample_bundle):
        """Test basic CSV export."""
        csv_str = export_trades_csv(sample_bundle)

        assert csv_str  # Not empty
        assert "trade_id" in csv_str
        assert "T001" in csv_str
        assert "T002" in csv_str

    def test_column_order(self, sample_bundle):
        """Test that columns are in expected order."""
        csv_str = export_trades_csv(sample_bundle)
        header = csv_str.split("\n")[0]

        # First column should be trade_id
        assert header.startswith("trade_id")

    def test_csv_parseable(self, sample_bundle):
        """Test that output is valid CSV."""
        csv_str = export_trades_csv(sample_bundle)
        df = pd.read_csv(StringIO(csv_str))

        assert len(df) == 3
        assert "trade_id" in df.columns

    def test_empty_bundle(self, empty_bundle):
        """Test with empty bundle."""
        csv_str = export_trades_csv(empty_bundle)

        assert csv_str == ""

    def test_missing_columns_handled(self):
        """Test with DataFrame missing some columns."""
        trades_df = pd.DataFrame(
            {
                "trade_id": ["T001"],
                "pnl": [-100.0],
                # Missing other columns
            }
        )

        bundle = DashboardBundle(
            trades_df=trades_df,
            returns=None,
            returns_label="none",
            explanations=[],
            patterns_df=pd.DataFrame(),
            n_trades_analyzed=1,
            n_trades_explained=1,
            n_trades_failed=0,
            failed_trades=[],
            config=DashboardConfig(),
        )

        csv_str = export_trades_csv(bundle)

        assert "trade_id" in csv_str
        assert "pnl" in csv_str


class TestExportPatternsCsv:
    """Tests for export_patterns_csv function."""

    def test_basic_export(self, sample_bundle):
        """Test basic patterns CSV export."""
        csv_str = export_patterns_csv(sample_bundle)

        assert csv_str  # Not empty
        assert "cluster_id" in csv_str
        assert "n_trades" in csv_str
        assert "description" in csv_str

    def test_csv_parseable(self, sample_bundle):
        """Test that output is valid CSV."""
        csv_str = export_patterns_csv(sample_bundle)
        df = pd.read_csv(StringIO(csv_str))

        assert len(df) == 3
        assert "cluster_id" in df.columns
        assert "n_trades" in df.columns

    def test_empty_bundle(self, empty_bundle):
        """Test with empty bundle."""
        csv_str = export_patterns_csv(empty_bundle)

        assert csv_str == ""


class TestExportHtmlReport:
    """Tests for export_html_report function."""

    def test_basic_export(self, sample_bundle):
        """Test basic HTML export."""
        html = export_html_report(sample_bundle)

        assert html  # Not empty
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_contains_title(self, sample_bundle):
        """Test that HTML contains title."""
        html = export_html_report(sample_bundle)

        assert "Trade-SHAP Diagnostics Report" in html

    def test_contains_statistics(self, sample_bundle):
        """Test that HTML contains statistical summary."""
        html = export_html_report(sample_bundle)

        assert "Statistical Summary" in html
        assert "Sharpe Ratio" in html
        assert "PSR" in html
        assert "Win Rate" in html

    def test_contains_trades_table(self, sample_bundle):
        """Test that HTML contains trades table."""
        html = export_html_report(sample_bundle)

        assert "Worst Trades" in html
        assert "Trade ID" in html
        assert "T001" in html or "T002" in html

    def test_contains_patterns(self, sample_bundle):
        """Test that HTML contains patterns section."""
        html = export_html_report(sample_bundle)

        assert "Error Patterns" in html
        assert "Pattern" in html

    def test_contains_css(self, sample_bundle):
        """Test that HTML contains embedded CSS."""
        html = export_html_report(sample_bundle)

        assert "<style>" in html
        assert "</style>" in html

    def test_empty_bundle(self, empty_bundle):
        """Test with empty bundle."""
        html = export_html_report(empty_bundle)

        # Should still produce valid HTML structure
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        # But won't have statistical summary
        assert "Statistical Summary" not in html

    def test_handles_nan_values(self):
        """Test that NaN values are handled in display."""
        trades_df = pd.DataFrame(
            {
                "trade_id": ["T001"],
                "symbol": [None],  # NaN symbol
                "pnl": [np.nan],  # NaN pnl
                "return_pct": [np.nan],  # NaN return
                "top_feature": ["volatility"],
            }
        )

        bundle = DashboardBundle(
            trades_df=trades_df,
            returns=np.array([0.01, 0.02]),  # Need valid returns for stats
            returns_label="return_pct",
            explanations=[],
            patterns_df=pd.DataFrame(),
            n_trades_analyzed=1,
            n_trades_explained=1,
            n_trades_failed=0,
            failed_trades=[],
            config=DashboardConfig(),
        )

        html = export_html_report(bundle)

        # Should handle NaN gracefully with N/A
        assert "N/A" in html

    def test_pattern_actions_displayed(self):
        """Test that pattern actions are displayed."""
        patterns_df = pd.DataFrame(
            {
                "cluster_id": [0],
                "n_trades": [10],
                "description": ["Test pattern"],
                "hypothesis": ["Test hypothesis"],
                "actions": [["Action 1", "Action 2"]],
            }
        )

        bundle = DashboardBundle(
            trades_df=pd.DataFrame(),
            returns=None,
            returns_label="none",
            explanations=[],
            patterns_df=patterns_df,
            n_trades_analyzed=10,
            n_trades_explained=10,
            n_trades_failed=0,
            failed_trades=[],
            config=DashboardConfig(),
        )

        html = export_html_report(bundle)

        assert "Error Patterns" in html
        assert "Action" in html

    def test_html_structure_valid(self, sample_bundle):
        """Test that HTML has valid structure."""
        html = export_html_report(sample_bundle)

        # Check basic structure
        assert html.count("<html") == 1
        assert html.count("</html>") == 1
        assert html.count("<head>") == 1
        assert html.count("</head>") == 1
        assert html.count("<body>") == 1
        assert html.count("</body>") == 1

    def test_timestamp_in_report(self, sample_bundle):
        """Test that generation timestamp is included."""
        html = export_html_report(sample_bundle)

        assert "Generated:" in html
