"""Report generation for signal analysis.

Internal module for HTML report generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml4t.diagnostic.signal.result import SignalResult


def generate_html(result: SignalResult, path: str) -> None:
    """Generate HTML report from signal analysis results.

    Parameters
    ----------
    result : SignalResult
        Analysis results.
    path : str
        Output file path.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        # Fallback to text-only report
        _generate_text_html(result, path)
        return

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("IC Time Series", "Quantile Returns", "IC Summary", "Spread Summary"),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "table"}],
        ],
    )

    # IC Time Series
    for period_key, ic_vals in result.ic_series.items():
        if ic_vals:
            fig.add_trace(
                go.Scatter(
                    y=ic_vals,
                    mode="lines",
                    name=f"IC {period_key}",
                ),
                row=1,
                col=1,
            )

    # Quantile Returns (first period)
    if result.periods:
        first_period = f"{result.periods[0]}D"
        q_returns = result.quantile_returns.get(first_period, {})
        if q_returns:
            quantiles = sorted(q_returns.keys())
            returns = [q_returns[q] for q in quantiles]
            fig.add_trace(
                go.Bar(
                    x=[f"Q{q}" for q in quantiles],
                    y=returns,
                    name=f"Returns {first_period}",
                ),
                row=1,
                col=2,
            )

    # IC Summary Table
    ic_data = []
    for period in result.periods:
        period_key = f"{period}D"
        ic_data.append(
            [
                period_key,
                f"{result.ic.get(period_key, float('nan')):.4f}",
                f"{result.ic_t_stat.get(period_key, float('nan')):.2f}",
                f"{result.ic_p_value.get(period_key, float('nan')):.4f}",
            ]
        )

    fig.add_trace(
        go.Table(
            header={"values": ["Period", "IC", "t-stat", "p-value"]},
            cells={"values": list(zip(*ic_data)) if ic_data else [[], [], [], []]},
        ),
        row=2,
        col=1,
    )

    # Spread Summary Table
    spread_data = []
    for period in result.periods:
        period_key = f"{period}D"
        spread_data.append(
            [
                period_key,
                f"{result.spread.get(period_key, float('nan')):.4f}",
                f"{result.spread_t_stat.get(period_key, float('nan')):.2f}",
                f"{result.monotonicity.get(period_key, float('nan')):.3f}",
            ]
        )

    fig.add_trace(
        go.Table(
            header={"values": ["Period", "Spread", "t-stat", "Monotonicity"]},
            cells={"values": list(zip(*spread_data)) if spread_data else [[], [], [], []]},
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text=f"Signal Analysis: {result.n_assets} assets, {result.n_dates} dates",
        height=800,
        showlegend=True,
    )

    # Write HTML
    fig.write_html(path, include_plotlyjs=True)


def _generate_text_html(result: SignalResult, path: str) -> None:
    """Generate text-only HTML report (no Plotly)."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Signal Analysis Report</title>
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        table {{ border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        pre {{ background-color: #f5f5f5; padding: 15px; }}
    </style>
</head>
<body>
    <h1>Signal Analysis Report</h1>
    <pre>{result.summary()}</pre>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


__all__ = ["generate_html"]
