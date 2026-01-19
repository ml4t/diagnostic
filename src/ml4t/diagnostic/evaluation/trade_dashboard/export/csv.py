"""CSV export functions for dashboard data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def export_trades_csv(bundle: DashboardBundle) -> str:
    """Export trades with SHAP values to CSV format.

    Parameters
    ----------
    bundle : DashboardBundle
        Normalized dashboard data.

    Returns
    -------
    str
        CSV formatted string with trade data.
    """
    if bundle.trades_df.empty:
        return ""

    # Select columns for export
    export_columns = [
        "trade_id",
        "symbol",
        "entry_time",
        "exit_time",
        "pnl",
        "return_pct",
        "duration_days",
        "entry_price",
        "exit_price",
        "top_feature",
        "top_shap_value",
    ]

    # Only include columns that exist
    available_columns = [c for c in export_columns if c in bundle.trades_df.columns]

    df = bundle.trades_df[available_columns].copy()

    return df.to_csv(index=False)


def export_patterns_csv(bundle: DashboardBundle) -> str:
    """Export error patterns to CSV format.

    Parameters
    ----------
    bundle : DashboardBundle
        Normalized dashboard data.

    Returns
    -------
    str
        CSV formatted string with pattern data.
    """
    if bundle.patterns_df.empty:
        return ""

    # Select columns for export
    export_columns = [
        "cluster_id",
        "n_trades",
        "description",
        "hypothesis",
        "confidence",
        "separation_score",
        "distinctiveness",
    ]

    # Only include columns that exist
    available_columns = [c for c in export_columns if c in bundle.patterns_df.columns]

    df = bundle.patterns_df[available_columns].copy()

    return df.to_csv(index=False)
