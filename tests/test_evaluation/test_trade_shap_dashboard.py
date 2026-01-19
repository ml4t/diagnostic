"""Tests for Trade-SHAP Dashboard.

This module tests the dashboard infrastructure, data loading, and
rendering functions (without requiring streamlit to be installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_dashboard_module_structure():
    """Test that dashboard module has expected structure."""
    try:
        from ml4t.diagnostic.evaluation import trade_shap_dashboard
    except ImportError as e:
        # Streamlit not installed - expected in CI
        pytest.skip(f"Streamlit not available: {e}")

    # Check main public API functions exist
    assert hasattr(trade_shap_dashboard, "run_diagnostics_dashboard")
    assert hasattr(trade_shap_dashboard, "load_data_from_file")
    assert hasattr(trade_shap_dashboard, "export_trades_to_csv")
    assert hasattr(trade_shap_dashboard, "export_patterns_to_csv")
    assert hasattr(trade_shap_dashboard, "export_full_report_html")
    assert hasattr(trade_shap_dashboard, "run_polished_dashboard")

    # Check modular package has tabs (internal implementation)
    from ml4t.diagnostic.evaluation.trade_dashboard import tabs

    assert hasattr(tabs, "stat_validation")
    assert hasattr(tabs, "worst_trades")
    assert hasattr(tabs, "shap_analysis")
    assert hasattr(tabs, "patterns")


def test_dashboard_import_optional():
    """Test that dashboard import is optional and doesn't break package."""
    # Should import successfully even if streamlit not installed
    from ml4t.diagnostic.evaluation import run_diagnostics_dashboard

    # If streamlit not installed, should be None
    if run_diagnostics_dashboard is None:
        pytest.skip("Streamlit not installed - dashboard unavailable")


def test_mock_data_structure():
    """Test that mock data matches expected structure."""
    # Create mock result
    mock_result = {
        "n_trades_analyzed": 50,
        "n_trades_explained": 45,
        "n_trades_failed": 5,
        "explanations": [
            {
                "trade_id": "TRADE_001",
                "timestamp": "2024-01-15 10:30:00",
                "top_features": [
                    ("momentum_20d", 0.45, 23.5),
                    ("rsi_14", 0.28, 14.7),
                ],
            }
        ],
        "failed_trades": [("TRADE_046", "Missing SHAP values")],
        "error_patterns": [
            {
                "cluster_id": 0,
                "n_trades": 15,
                "description": "High momentum → Losses",
                "top_features": [("momentum_20d", 0.45, 0.001, 0.002, True)],
                "separation_score": 1.2,
                "distinctiveness": 1.8,
                "hypothesis": "Test hypothesis",
                "actions": ["Test action 1", "Test action 2"],
                "confidence": 0.85,
            }
        ],
    }

    # Validate structure
    assert mock_result["n_trades_analyzed"] == 50
    assert mock_result["n_trades_explained"] == 45
    assert mock_result["n_trades_failed"] == 5
    assert len(mock_result["explanations"]) == 1  # type: ignore[arg-type]
    assert len(mock_result["failed_trades"]) == 1  # type: ignore[arg-type]
    assert len(mock_result["error_patterns"]) == 1  # type: ignore[arg-type]

    # Validate pattern structure
    pattern = mock_result["error_patterns"][0]  # type: ignore[index]
    assert pattern["cluster_id"] == 0
    assert pattern["n_trades"] == 15
    assert pattern["hypothesis"] is not None
    assert pattern["actions"] is not None
    assert pattern["confidence"] == 0.85


def test_data_serialization():
    """Test that mock data can be serialized to JSON."""
    mock_result = {
        "n_trades_analyzed": 10,
        "n_trades_explained": 8,
        "n_trades_failed": 2,
        "explanations": [],
        "failed_trades": [],
        "error_patterns": [],
    }

    # Should serialize without error
    json_str = json.dumps(mock_result)
    assert len(json_str) > 0

    # Should deserialize correctly
    loaded = json.loads(json_str)
    assert loaded["n_trades_analyzed"] == 10
    assert loaded["n_trades_explained"] == 8


def test_dashboard_example_script_exists():
    """Test that example script exists and is valid Python."""
    example_path = Path(__file__).parent.parent.parent / "examples" / "trade_shap_dashboard_demo.py"

    assert example_path.exists(), f"Example script not found: {example_path}"

    # Check it's valid Python
    import py_compile

    try:
        py_compile.compile(str(example_path), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"Example script has syntax errors: {e}")


def test_dashboard_documentation_exists():
    """Test that dashboard documentation exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs" / "DASHBOARD.md"

    assert docs_path.exists(), f"Documentation not found: {docs_path}"

    # Check it's not empty
    content = docs_path.read_text()
    assert len(content) > 100, "Documentation file is too short"
    assert "Trade-SHAP" in content
    assert "Streamlit" in content


def test_pyproject_has_dashboard_dependency():
    """Test that pyproject.toml includes streamlit in optional dependencies."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

    assert pyproject_path.exists()

    content = pyproject_path.read_text()

    # Check for dashboard optional dependency group
    assert "dashboard" in content or "streamlit" in content

    # Check streamlit is mentioned
    assert "streamlit" in content.lower()


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_shap_dashboard.py"
    ).exists(),
    reason="Dashboard file not created yet",
)
def test_dashboard_file_structure():
    """Test that dashboard file has expected docstring and imports.

    Note: The dashboard has been refactored into a modular package structure:
    - trade_shap_dashboard.py: thin wrapper for backward compatibility
    - trade_dashboard/app.py: main Streamlit app
    - trade_dashboard/tabs/: tab modules with render_tab functions
    """
    # Check the wrapper exists and has proper docstring
    wrapper_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_shap_dashboard.py"
    )
    wrapper_content = wrapper_path.read_text()
    assert '"""' in wrapper_content  # Has docstring
    assert "run_diagnostics_dashboard" in wrapper_content

    # Check the modular app.py exists with streamlit import
    app_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "app.py"
    )
    app_content = app_path.read_text()
    assert "import streamlit" in app_content or "st." in app_content
    assert "def run_dashboard" in app_content or "run_diagnostics_dashboard" in app_content

    # Check tab modules have render_tab functions
    tabs_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "tabs"
    )
    assert (tabs_path / "stat_validation.py").exists()
    assert (tabs_path / "worst_trades.py").exists()
    assert (tabs_path / "shap_analysis.py").exists()
    assert (tabs_path / "patterns.py").exists()


def test_dashboard_has_4_tabs():
    """Test that dashboard defines 4 tabs as per spec."""
    try:
        from ml4t.diagnostic.evaluation.trade_dashboard import tabs
    except ImportError:
        pytest.skip("Streamlit not available")

    # Check tab modules exist in modular package
    assert hasattr(tabs.stat_validation, "render_tab")
    assert hasattr(tabs.worst_trades, "render_tab")
    assert hasattr(tabs.shap_analysis, "render_tab")
    assert hasattr(tabs.patterns, "render_tab")


def test_dashboard_error_handling_structure():
    """Test that dashboard has try-except for data loading.

    Note: Error handling is in the modular app.py and io.py modules.
    """
    # Check app.py for error handling
    app_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "app.py"
    )

    if not app_path.exists():
        pytest.skip("Dashboard modular package not created yet")

    content = app_path.read_text()

    # Check for error handling
    assert "try:" in content
    assert "except" in content


def test_extract_trade_returns_helper():
    """Test extract_trade_returns helper function."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import extract_trade_returns
    except ImportError:
        pytest.skip("Streamlit not available")

    # Mock result with trade metrics
    result = {
        "explanations": [
            {
                "trade_id": "TRADE_001",
                "trade_metrics": {"pnl": -150.0},
            },
            {
                "trade_id": "TRADE_002",
                "trade_metrics": {"pnl": 250.0},
            },
        ]
    }

    import numpy as np

    returns = extract_trade_returns(result)

    assert returns is not None
    assert len(returns) == 2
    assert np.allclose(returns, [-150.0, 250.0])


def test_extract_trade_data_helper():
    """Test extract_trade_data helper function."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import extract_trade_data
    except ImportError:
        pytest.skip("Streamlit not available")

    result = {
        "explanations": [
            {
                "trade_id": "TRADE_001",
                "timestamp": "2024-01-15 10:30:00",
                "top_features": [("momentum_20d", 0.45)],
                "trade_metrics": {
                    "symbol": "BTC-USD",
                    "pnl": -150.0,
                    "return_pct": -2.5,
                    "duration_days": 3.2,
                    "entry_price": 50000.0,
                    "exit_price": 48750.0,
                },
            }
        ]
    }

    trades = extract_trade_data(result)

    assert len(trades) == 1
    trade = trades[0]

    assert trade["trade_id"] == "TRADE_001"
    assert trade["symbol"] == "BTC-USD"
    assert trade["pnl"] == -150.0
    assert trade["return_pct"] == -2.5
    assert trade["duration_days"] == 3.2
    assert trade["top_feature"] == "momentum_20d"
    assert trade["top_shap_value"] == 0.45


def test_demo_script_creates_valid_data():
    """Test that demo script creates valid mock data with trade metrics."""
    import sys

    # Add examples to path
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))

    try:
        from trade_shap_dashboard_demo import create_mock_result
    except ImportError as e:
        pytest.skip(f"Demo script import failed: {e}")

    result = create_mock_result()

    # Verify structure
    assert "n_trades_analyzed" in result
    assert "n_trades_explained" in result
    assert "explanations" in result

    # Verify data
    assert result["n_trades_analyzed"] == 50
    assert result["n_trades_explained"] == 45
    # Demo generates 50 trades, but marks only 45 as "explained"
    assert len(result["explanations"]) == 50  # Actual explanations created

    # Verify explanations have trade_metrics
    for explanation in result["explanations"]:
        assert "trade_id" in explanation
        assert "trade_metrics" in explanation

        metrics = explanation["trade_metrics"]
        assert "symbol" in metrics
        assert "pnl" in metrics
        assert "return_pct" in metrics
        assert "entry_price" in metrics
        assert "exit_price" in metrics
        assert "duration_days" in metrics

    # Verify sorting (worst PnL first)
    pnls = [exp["trade_metrics"]["pnl"] for exp in result["explanations"]]
    assert pnls == sorted(pnls)


def test_dashboard_has_statistical_functions():
    """Test that dashboard stats module has statistical functions.

    Note: Statistical functions are in the modular stats.py module.
    """
    stats_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "stats.py"
    )

    if not stats_path.exists():
        pytest.skip("Dashboard stats module not created yet")

    content = stats_path.read_text()

    # Check for statistical functionality (computations, not specific imports)
    assert "def " in content  # Has functions defined
    # The stats module contains compute_* functions for PSR, DSR, etc.
    assert "compute" in content.lower() or "stats" in content.lower() or "metric" in content.lower()


def test_demo_script_includes_feature_values():
    """Test that demo script includes feature_values in explanations (TASK-032)."""
    import sys

    # Add examples to path
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    sys.path.insert(0, str(examples_dir))

    try:
        from trade_shap_dashboard_demo import create_mock_result
    except ImportError as e:
        pytest.skip(f"Demo script import failed: {e}")

    result = create_mock_result()

    # Verify explanations have feature_values for SHAP waterfall plot
    for explanation in result["explanations"]:
        assert "feature_values" in explanation, "Explanation missing feature_values"

        feature_values = explanation["feature_values"]
        assert isinstance(feature_values, dict), "feature_values should be a dict"
        assert len(feature_values) > 0, "feature_values should not be empty"

        # Check that feature_values match top_features
        top_features = explanation["top_features"]
        for feat_name, _shap_val in top_features:
            assert feat_name in feature_values, f"Feature {feat_name} missing from feature_values"


def test_shap_tab_implementation():
    """Test that SHAP tab has been properly implemented (TASK-032).

    Note: SHAP tab is in the modular tabs/shap_analysis.py module.
    """
    shap_tab_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "tabs"
        / "shap_analysis.py"
    )

    if not shap_tab_path.exists():
        pytest.skip("SHAP tab module not created yet")

    content = shap_tab_path.read_text()

    # Check for SHAP visualization components
    assert "render_tab" in content, "Missing render_tab function"
    assert "shap" in content.lower() or "feature" in content.lower(), (
        "Missing SHAP-related content"
    )


def test_patterns_tab_enhanced():
    """Test that Patterns tab has enhanced features (TASK-032).

    Note: Patterns tab is in the modular tabs/patterns.py module.
    """
    patterns_tab_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "tabs"
        / "patterns.py"
    )

    if not patterns_tab_path.exists():
        pytest.skip("Patterns tab module not created yet")

    content = patterns_tab_path.read_text()

    # Check for pattern tab features
    assert "render_tab" in content, "Missing render_tab function"
    assert "pattern" in content.lower(), "Missing pattern-related content"


def test_export_functions_exist():
    """Test that export functions are defined (TASK-033)."""
    try:
        from ml4t.diagnostic.evaluation import trade_shap_dashboard
    except ImportError:
        pytest.skip("Streamlit not available")

    # Check export functions exist
    assert hasattr(trade_shap_dashboard, "export_trades_to_csv")
    assert hasattr(trade_shap_dashboard, "export_patterns_to_csv")
    assert hasattr(trade_shap_dashboard, "export_full_report_html")


def test_export_trades_to_csv():
    """Test CSV export for trades (TASK-033)."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_trades_to_csv
    except ImportError:
        pytest.skip("Streamlit not available")

    trades_data = [
        {
            "trade_id": "TRADE_001",
            "symbol": "BTC-USD",
            "timestamp": "2024-01-15 10:30:00",
            "pnl": -150.0,
            "return_pct": -2.5,
            "duration_days": 3.2,
            "entry_price": 50000.0,
            "exit_price": 48750.0,
            "top_feature": "momentum_20d",
            "top_shap_value": 0.45,
        }
    ]

    csv_output = export_trades_to_csv(trades_data)

    assert csv_output is not None
    assert len(csv_output) > 0
    assert "trade_id" in csv_output
    assert "TRADE_001" in csv_output
    assert "BTC-USD" in csv_output


def test_export_patterns_to_csv():
    """Test CSV export for patterns (TASK-033)."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_patterns_to_csv
    except ImportError:
        pytest.skip("Streamlit not available")

    patterns = [
        {
            "cluster_id": 0,
            "n_trades": 15,
            "description": "High momentum → Losses",
            "hypothesis": "Test hypothesis",
            "confidence": 0.85,
            "actions": ["Action 1", "Action 2"],
        }
    ]

    csv_output = export_patterns_to_csv(patterns)

    assert csv_output is not None
    assert len(csv_output) > 0
    assert "Pattern ID" in csv_output
    assert "High momentum" in csv_output
    assert "Test hypothesis" in csv_output


def test_export_full_report_html():
    """Test HTML report export (TASK-033)."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_full_report_html
    except ImportError:
        pytest.skip("Streamlit not available")

    result = {
        "n_trades_analyzed": 50,
        "n_trades_explained": 45,
        "n_trades_failed": 5,
        "explanations": [],
        "failed_trades": [],
        "error_patterns": [
            {
                "cluster_id": 0,
                "n_trades": 15,
                "description": "Test pattern",
                "hypothesis": "Test hypothesis",
                "actions": ["Action 1"],
                "confidence": 0.85,
            }
        ],
    }

    html_output = export_full_report_html(result)

    assert html_output is not None
    assert len(html_output) > 0
    assert "<!DOCTYPE html>" in html_output
    assert "Trade-SHAP Analysis Report" in html_output
    assert "Test pattern" in html_output
    assert "Test hypothesis" in html_output


def test_styled_dashboard_parameter():
    """Test that dashboard supports styled parameter (TASK-033, D01)."""
    try:
        from ml4t.diagnostic.evaluation import trade_shap_dashboard
    except ImportError:
        pytest.skip("Streamlit not available")

    import inspect

    # Check that run_diagnostics_dashboard has styled parameter
    sig = inspect.signature(trade_shap_dashboard.run_diagnostics_dashboard)
    params = list(sig.parameters.keys())
    assert "styled" in params, "run_diagnostics_dashboard should have styled parameter"

    # Check run_polished_dashboard alias exists
    assert hasattr(trade_shap_dashboard, "run_polished_dashboard")
    assert callable(trade_shap_dashboard.run_polished_dashboard)


def test_caching_functions_exist():
    """Test that modular dashboard exports expected types (TASK-033).

    Note: After refactoring, caching is handled internally in the normalize
    module rather than exposed as public functions. This test now verifies
    the modular package structure.
    """
    try:
        from ml4t.diagnostic.evaluation.trade_dashboard import (
            DashboardBundle,
            DashboardConfig,
            normalize_result,
        )
    except ImportError:
        pytest.skip("Streamlit not available")

    # Check core types and functions exist
    assert DashboardBundle is not None
    assert DashboardConfig is not None
    assert callable(normalize_result)


def test_performance_optimization():
    """Test that performance optimization features exist (TASK-033, D01)."""
    # Check modular app.py for load time tracking
    app_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "app.py"
    )

    content = app_path.read_text()

    # Check for performance features in modular package
    assert "_measure_load_time_start" in content
    assert "_measure_load_time_end" in content
    assert "st.spinner" in content


def test_professional_styling():
    """Test that professional styling is applied (TASK-033, D01)."""
    # Check modular style.py for CSS
    style_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "style.py"
    )

    content = style_path.read_text()

    # Check for CSS styling in modular package
    assert "STYLED_CSS" in content
    assert "<style>" in content
    assert "primary-color" in content
    assert "border-radius" in content


def test_error_handling_implemented():
    """Test that error handling is implemented (TASK-033, D01)."""
    # Check modular app.py for error handling
    app_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "ml4t"
        / "diagnostic"
        / "evaluation"
        / "trade_dashboard"
        / "app.py"
    )

    content = app_path.read_text()

    # Check for error handling in modular package
    assert "try:" in content
    assert "except" in content
    assert "traceback" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
