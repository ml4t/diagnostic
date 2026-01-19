"""Streamlit dashboard for Trade-SHAP diagnostics.

This module provides an interactive Streamlit dashboard for visualizing
Trade-SHAP analysis results, including statistical validation, worst trades,
SHAP explanations, and error patterns.

The dashboard is designed for systematic trade debugging and continuous
improvement of ML trading strategies.

Usage:
    # From command line
    streamlit run -m ml4t.diagnostic.evaluation.trade_shap_dashboard

    # Programmatically
    from ml4t.diagnostic.evaluation.trade_shap_dashboard import run_diagnostics_dashboard
    run_diagnostics_dashboard(result)

Example:
    >>> from ml4t.diagnostic.evaluation import TradeShapAnalyzer
    >>> from ml4t.diagnostic.evaluation.trade_shap_dashboard import run_diagnostics_dashboard
    >>>
    >>> # Analyze trades and get results
    >>> analyzer = TradeShapAnalyzer(model, features_df, shap_values)
    >>> result = analyzer.explain_worst_trades(worst_trades)
    >>>
    >>> # Launch interactive dashboard
    >>> run_diagnostics_dashboard(result)

Note:
    This module is a thin wrapper around the modular dashboard package.
    The implementation has been refactored into:
    - ml4t.diagnostic.evaluation.trade_dashboard.app (main orchestrator)
    - ml4t.diagnostic.evaluation.trade_dashboard.tabs (tab modules)
    - ml4t.diagnostic.evaluation.trade_dashboard.stats (statistical computations)
    - ml4t.diagnostic.evaluation.trade_dashboard.export (export functions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Re-export the main entry point for backward compatibility
from ml4t.diagnostic.evaluation.trade_dashboard import run_diagnostics_dashboard

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_shap.models import TradeShapResult

# Import utilities for backward compatibility
from ml4t.diagnostic.evaluation.trade_dashboard.io import (
    PickleDisabledError,
)
from ml4t.diagnostic.evaluation.trade_dashboard.io import (
    load_result_from_upload as load_data_from_file,
)

__all__ = [
    "run_diagnostics_dashboard",
    "run_polished_dashboard",
    "export_full_report_html",
    "export_patterns_to_csv",
    "export_trades_to_csv",
    "load_data_from_file",
    "PickleDisabledError",
    "extract_trade_returns",
    "extract_trade_data",
]


def run_polished_dashboard(
    result: TradeShapResult | dict[str, Any] | None = None,
    title: str = "Trade-SHAP Diagnostics Dashboard",
) -> None:
    """Run dashboard with styled=True. Alias for backward compat."""
    run_diagnostics_dashboard(result=result, title=title, styled=True)


# Backward-compatible export functions that accept raw dicts/lists
def export_trades_to_csv(trades_data: list[dict[str, Any]]) -> str:
    """Export trades to CSV format. Backward-compatible API.

    Parameters
    ----------
    trades_data : list of dict
        List of trade dictionaries.

    Returns
    -------
    str
        CSV formatted string.
    """
    import pandas as pd

    if not trades_data:
        return ""
    return pd.DataFrame(trades_data).to_csv(index=False)


def export_patterns_to_csv(patterns: list[dict[str, Any]]) -> str:
    """Export patterns to CSV format. Backward-compatible API.

    Parameters
    ----------
    patterns : list of dict
        List of pattern dictionaries.

    Returns
    -------
    str
        CSV formatted string with headers Pattern ID, etc.
    """
    import pandas as pd

    if not patterns:
        return ""

    # Transform to expected format
    records = []
    for p in patterns:
        records.append(
            {
                "Pattern ID": p.get("cluster_id", 0),
                "N Trades": p.get("n_trades", 0),
                "Description": p.get("description", ""),
                "Hypothesis": p.get("hypothesis", ""),
                "Confidence": p.get("confidence", ""),
            }
        )
    return pd.DataFrame(records).to_csv(index=False)


def export_full_report_html(result: dict[str, Any]) -> str:
    """Export full HTML report. Backward-compatible API.

    Parameters
    ----------
    result : dict
        Analysis result dictionary.

    Returns
    -------
    str
        HTML report string.
    """
    from datetime import datetime

    patterns = result.get("error_patterns", [])
    n_analyzed = result.get("n_trades_analyzed", 0)
    n_explained = result.get("n_trades_explained", 0)
    n_failed = result.get("n_trades_failed", 0)

    patterns_html = ""
    for p in patterns:
        hypothesis = p.get("hypothesis", "No hypothesis")
        actions = p.get("actions", [])
        actions_html = "".join(f"<li>{a}</li>" for a in actions) if actions else ""

        patterns_html += f"""
        <div class="pattern">
            <h3>Pattern {p.get("cluster_id", "N/A")}: {p.get("n_trades", 0)} trades</h3>
            <p><strong>Description:</strong> {p.get("description", "N/A")}</p>
            <p><strong>Hypothesis:</strong> {hypothesis}</p>
            <ul>{actions_html}</ul>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Trade-SHAP Analysis Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #1f77b4; color: white; padding: 20px; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f0f0f0; padding: 15px; }}
        .pattern {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trade-SHAP Analysis Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    <div class="metrics">
        <div class="metric"><strong>Analyzed:</strong> {n_analyzed}</div>
        <div class="metric"><strong>Explained:</strong> {n_explained}</div>
        <div class="metric"><strong>Failed:</strong> {n_failed}</div>
    </div>
    <h2>Error Patterns</h2>
    {patterns_html}
</body>
</html>"""


def extract_trade_returns(result: dict[str, Any]) -> list[float]:
    """Extract trade PnL values from analysis result.

    Parameters
    ----------
    result : dict
        Analysis result dictionary with "explanations" key.

    Returns
    -------
    list of float
        List of PnL values from each trade.

    Examples
    --------
    >>> result = {"explanations": [{"trade_metrics": {"pnl": 100.0}}]}
    >>> extract_trade_returns(result)
    [100.0]
    """
    explanations = result.get("explanations", [])
    returns = []
    for exp in explanations:
        trade_metrics = exp.get("trade_metrics", {})
        pnl = trade_metrics.get("pnl", 0.0)
        returns.append(pnl)
    return returns


def extract_trade_data(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract trade data for display from analysis result.

    Parameters
    ----------
    result : dict
        Analysis result dictionary with "explanations" key.

    Returns
    -------
    list of dict
        List of trade data dictionaries with keys:
        - trade_id: Trade identifier
        - timestamp: Trade timestamp
        - symbol: Trading symbol
        - pnl: Profit/loss
        - return_pct: Return percentage
        - duration_days: Trade duration
        - entry_price: Entry price
        - exit_price: Exit price
        - top_feature: Most important feature
        - top_shap_value: SHAP value of top feature

    Examples
    --------
    >>> result = {"explanations": [{"trade_id": "T1", "trade_metrics": {"pnl": 100.0}}]}
    >>> data = extract_trade_data(result)
    >>> data[0]["trade_id"]
    'T1'
    """
    explanations = result.get("explanations", [])
    trade_data = []

    for exp in explanations:
        trade_metrics = exp.get("trade_metrics", {})
        top_features = exp.get("top_features", [])

        # Get top feature info
        top_feature = top_features[0][0] if top_features else None
        top_shap_value = top_features[0][1] if top_features else None

        trade_data.append(
            {
                "trade_id": exp.get("trade_id", ""),
                "timestamp": exp.get("timestamp", ""),
                "symbol": trade_metrics.get("symbol", ""),
                "pnl": trade_metrics.get("pnl", 0.0),
                "return_pct": trade_metrics.get("return_pct", 0.0),
                "duration_days": trade_metrics.get("duration_days", 0.0),
                "entry_price": trade_metrics.get("entry_price", 0.0),
                "exit_price": trade_metrics.get("exit_price", 0.0),
                "top_feature": top_feature,
                "top_shap_value": top_shap_value,
            }
        )

    return trade_data


# Allow running as a standalone Streamlit app
if __name__ == "__main__":
    run_diagnostics_dashboard()
