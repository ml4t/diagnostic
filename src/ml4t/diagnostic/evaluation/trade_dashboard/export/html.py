"""HTML report export for dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.trade_dashboard.types import DashboardBundle


def export_html_report(bundle: DashboardBundle) -> str:
    """Generate comprehensive HTML report from dashboard data.

    Parameters
    ----------
    bundle : DashboardBundle
        Normalized dashboard data.

    Returns
    -------
    str
        Complete HTML report as string.
    """
    from ml4t.diagnostic.evaluation.trade_dashboard.stats import (
        compute_return_summary,
        probabilistic_sharpe_ratio,
    )

    # Compute summary if we have returns
    summary_html = ""
    if bundle.returns is not None and len(bundle.returns) > 0:
        summary = compute_return_summary(bundle.returns)
        psr_result = probabilistic_sharpe_ratio(
            observed_sharpe=summary.sharpe,
            benchmark_sharpe=0.0,
            n_samples=summary.n_samples,
            skewness=summary.skewness,
            kurtosis=summary.kurtosis,
            return_components=True,
        )
        psr = psr_result["psr"]

        summary_html = f"""
        <div class="section">
            <h2>Statistical Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Trades Analyzed</div>
                    <div class="metric-value">{bundle.n_trades_analyzed}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{summary.sharpe:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">PSR (vs SR=0)</div>
                    <div class="metric-value">{psr:.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{summary.win_rate:.1%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mean Return</div>
                    <div class="metric-value">{summary.mean:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{summary.std:.4f}</div>
                </div>
            </div>
            <p class="caption">Returns based on: {bundle.returns_label}</p>
        </div>
        """

    # Worst trades table
    trades_html = ""
    if not bundle.trades_df.empty:
        worst_trades = bundle.trades_df.head(10)
        rows = ""
        for _, row in worst_trades.iterrows():
            pnl = row.get("pnl")
            pnl_str = f"${pnl:.2f}" if pnl is not None and not np.isnan(pnl) else "N/A"
            return_pct = row.get("return_pct")
            return_str = (
                f"{return_pct:.2f}%"
                if return_pct is not None and not np.isnan(return_pct)
                else "N/A"
            )

            rows += f"""
            <tr>
                <td>{row.get("trade_id", "N/A")}</td>
                <td>{row.get("symbol", "N/A")}</td>
                <td>{pnl_str}</td>
                <td>{return_str}</td>
                <td>{row.get("top_feature", "N/A")}</td>
            </tr>
            """

        trades_html = f"""
        <div class="section">
            <h2>Worst Trades (Top 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Trade ID</th>
                        <th>Symbol</th>
                        <th>PnL</th>
                        <th>Return %</th>
                        <th>Top Feature</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    # Patterns section
    patterns_html = ""
    if not bundle.patterns_df.empty:
        pattern_cards = ""
        for _, pattern in bundle.patterns_df.iterrows():
            hypothesis = pattern.get("hypothesis") or "No hypothesis generated"
            actions = pattern.get("actions", [])
            actions_list = ""
            if actions:
                for action in actions:
                    actions_list += f"<li>{action}</li>"
                actions_html = f"<ul>{actions_list}</ul>"
            else:
                actions_html = "<p>No actions suggested</p>"

            pattern_cards += f"""
            <div class="pattern-card">
                <h3>Pattern {pattern.get("cluster_id", "N/A")}: {pattern.get("n_trades", 0)} trades</h3>
                <p><strong>Description:</strong> {pattern.get("description", "N/A")}</p>
                <p><strong>Hypothesis:</strong> {hypothesis}</p>
                <p><strong>Actions:</strong></p>
                {actions_html}
            </div>
            """

        patterns_html = f"""
        <div class="section">
            <h2>Error Patterns</h2>
            {pattern_cards}
        </div>
        """

    # Complete HTML document
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade-SHAP Diagnostics Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
        }}
        .header {{
            background: #1f77b4;
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            margin: 0;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1f77b4;
            margin-top: 0;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.5rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f77b4;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .pattern-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
        }}
        .pattern-card h3 {{
            margin-top: 0;
            color: #1f77b4;
        }}
        .caption {{
            font-size: 0.85rem;
            color: #666;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 0.85rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trade-SHAP Diagnostics Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    {summary_html}
    {trades_html}
    {patterns_html}

    <div class="footer">
        <p>Generated by Trade-SHAP Diagnostics Dashboard</p>
    </div>
</body>
</html>
    """

    return html
