"""Portfolio dashboard combining all visualizations.

Creates comprehensive tear sheets with multiple panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization.core import (
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import (
        PortfolioAnalysis,
        PortfolioMetrics,
    )


@dataclass
class PortfolioTearSheet:
    """Container for portfolio tear sheet with all figures and data.

    Provides methods to display in notebooks, export to HTML,
    and access individual figures.

    Attributes:
        metrics: Portfolio performance metrics
        figures: Dictionary of individual Plotly figures
        html_content: Pre-rendered HTML content (if generated)
    """

    metrics: PortfolioMetrics
    figures: dict[str, go.Figure] = field(default_factory=dict)
    html_content: str | None = None
    _analysis: PortfolioAnalysis | None = None

    def show(self) -> None:
        """Display tear sheet in Jupyter notebook."""
        # Display metrics first
        print(self.metrics.summary())
        print()

        # Display each figure
        for name, fig in self.figures.items():
            print(f"\n{'=' * 50}")
            print(f"  {name}")
            print("=" * 50)
            fig.show()

    def save_html(
        self,
        path: str | Path,
        include_plotlyjs: bool | str = "cdn",
        full_html: bool = True,
    ) -> None:
        """Save tear sheet as self-contained HTML file.

        Parameters
        ----------
        path : str or Path
            Output file path
        include_plotlyjs : bool or str, default "cdn"
            How to include Plotly.js:
            - True: Embed full library (~3MB)
            - "cdn": Link to CDN (smaller file, requires internet)
            - False: Don't include (for embedding in larger page)
        full_html : bool, default True
            Include full HTML structure (<!DOCTYPE>, <head>, etc.)
        """
        path = Path(path)

        # Build HTML content
        html_parts = []

        if full_html:
            html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Portfolio Tear Sheet</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .metrics-summary {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-summary pre {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <h1>Portfolio Tear Sheet</h1>
""")

        # Add metrics summary
        html_parts.append(f"""
    <div class="metrics-summary">
        <h2>Performance Summary</h2>
        <pre>{self.metrics.summary()}</pre>
    </div>
""")

        # Add each figure
        for name, fig in self.figures.items():
            fig_html = fig.to_html(
                include_plotlyjs=include_plotlyjs
                if name == list(self.figures.keys())[0]
                else False,
                full_html=False,
            )
            html_parts.append(f"""
    <div class="plot-container">
        <h2>{name}</h2>
        {fig_html}
    </div>
""")

        if full_html:
            html_parts.append("""
</body>
</html>
""")

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(html_parts))

    def get_figures(self) -> dict[str, go.Figure]:
        """Get dictionary of all figures.

        Returns
        -------
        dict[str, go.Figure]
            Named Plotly figures
        """
        return self.figures.copy()

    def to_dict(self) -> dict[str, Any]:
        """Get all underlying data as dictionary.

        Returns
        -------
        dict
            Contains metrics and figure data
        """
        return {
            "metrics": self.metrics.to_dict(),
            "figures": {name: fig.to_dict() for name, fig in self.figures.items()},
        }


def create_portfolio_dashboard(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    include_positions: bool = True,
    include_transactions: bool = True,
    include_benchmark: bool = True,
    height_per_row: int = 300,
    width: int | None = None,
) -> PortfolioTearSheet:
    """Create comprehensive portfolio tear sheet.

    Generates a complete portfolio analysis dashboard with:
    - Performance metrics summary
    - Cumulative returns chart
    - Drawdown underwater curve
    - Rolling Sharpe ratio
    - Rolling volatility
    - Monthly returns heatmap
    - Returns distribution

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object with returns data
    theme : str, optional
        Plot theme ("default", "dark", "print", "presentation")
    include_positions : bool, default True
        Include position analysis (if positions data available)
    include_transactions : bool, default True
        Include transaction analysis (if transactions data available)
    include_benchmark : bool, default True
        Include benchmark comparison (if benchmark available)
    height_per_row : int, default 300
        Height per subplot row
    width : int, optional
        Figure width

    Returns
    -------
    PortfolioTearSheet
        Complete tear sheet with all figures and metrics

    Examples
    --------
    >>> analysis = PortfolioAnalysis(returns=daily_returns)
    >>> tear_sheet = create_portfolio_dashboard(analysis)
    >>> tear_sheet.show()  # Display in notebook
    >>> tear_sheet.save_html("report.html")  # Export to file
    """
    theme = validate_theme(theme)
    get_theme_config(theme)

    # Compute all metrics
    metrics = analysis.compute_summary_stats()

    # Import visualization functions
    from .drawdown_plots import (
        plot_drawdown_periods,
        plot_drawdown_underwater,
    )
    from .returns_plots import (
        plot_annual_returns_bar,
        plot_cumulative_returns,
        plot_monthly_returns_heatmap,
        plot_returns_distribution,
    )
    from .risk_plots import (
        plot_rolling_beta,
        plot_rolling_sharpe,
        plot_rolling_volatility,
    )

    figures = {}

    # 1. Cumulative returns
    figures["Cumulative Returns"] = plot_cumulative_returns(
        analysis,
        theme=theme,
        show_benchmark=include_benchmark and analysis.has_benchmark,
        height=height_per_row,
        width=width,
    )

    # 2. Drawdown underwater
    figures["Drawdown"] = plot_drawdown_underwater(
        analysis,
        theme=theme,
        height=int(height_per_row * 0.8),
        width=width,
    )

    # 3. Rolling metrics
    rolling_result = analysis.compute_rolling_metrics(
        windows=[21, 63, 252],
        metrics=["sharpe", "volatility", "returns"],
    )

    figures["Rolling Sharpe Ratio"] = plot_rolling_sharpe(
        rolling_result=rolling_result,
        windows=[63, 252],
        theme=theme,
        height=height_per_row,
        width=width,
    )

    figures["Rolling Volatility"] = plot_rolling_volatility(
        rolling_result=rolling_result,
        windows=[21, 63, 252],
        theme=theme,
        height=height_per_row,
        width=width,
    )

    # 4. Rolling beta (if benchmark available)
    if include_benchmark and analysis.has_benchmark:
        beta_rolling = analysis.compute_rolling_metrics(
            windows=[126],
            metrics=["beta"],
        )
        figures["Rolling Beta"] = plot_rolling_beta(
            rolling_result=beta_rolling,
            window=126,
            theme=theme,
            height=height_per_row,
            width=width,
        )

    # 5. Annual returns
    figures["Annual Returns"] = plot_annual_returns_bar(
        analysis,
        theme=theme,
        show_benchmark=include_benchmark and analysis.has_benchmark,
        height=height_per_row,
        width=width,
    )

    # 6. Monthly returns heatmap
    figures["Monthly Returns Heatmap"] = plot_monthly_returns_heatmap(
        analysis,
        theme=theme,
        height=int(height_per_row * 1.2),
        width=width,
    )

    # 7. Returns distribution
    figures["Returns Distribution"] = plot_returns_distribution(
        analysis,
        theme=theme,
        height=height_per_row,
        width=width,
    )

    # 8. Top drawdowns
    figures["Top Drawdowns"] = plot_drawdown_periods(
        analysis,
        top_n=5,
        theme=theme,
        height=height_per_row,
        width=width,
    )

    return PortfolioTearSheet(
        metrics=metrics,
        figures=figures,
        _analysis=analysis,
    )


def create_simple_dashboard(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    height: int = 800,
    width: int | None = None,
) -> go.Figure:
    """Create simple 4-panel dashboard as single figure.

    A quick overview with:
    - Cumulative returns (top left)
    - Drawdown (top right)
    - Rolling Sharpe (bottom left)
    - Monthly heatmap (bottom right)

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object
    theme : str, optional
        Plot theme
    height : int, default 800
        Total figure height
    width : int, optional
        Figure width

    Returns
    -------
    go.Figure
        Single combined figure
    """
    theme = validate_theme(theme)
    theme_config = get_theme_config(theme)

    import numpy as np

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Cumulative Returns",
            "Drawdown",
            "Rolling Sharpe (252d)",
            "Monthly Returns",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    dates = analysis.dates.to_list()

    # === Cumulative Returns (1,1) ===
    cum_returns = (1 + analysis.returns).cumprod()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_returns,
            mode="lines",
            name="Strategy",
            line={"color": theme_config["colorway"][0], "width": 2},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if analysis.has_benchmark and analysis.benchmark is not None:
        bench_cum = (1 + analysis.benchmark).cumprod()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bench_cum,
                mode="lines",
                name="Benchmark",
                line={"color": theme_config["colorway"][1], "width": 2, "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # === Drawdown (1,2) ===
    dd_result = analysis.compute_drawdown_analysis()
    underwater = dd_result.underwater_curve.to_numpy()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=underwater,
            mode="lines",
            name="Drawdown",
            line={"color": theme_config["colorway"][1], "width": 1},
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.3)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # === Rolling Sharpe (2,1) ===
    rolling = analysis.compute_rolling_metrics(windows=[252], metrics=["sharpe"])
    sharpe_252 = rolling.sharpe[252].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sharpe_252,
            mode="lines",
            name="Sharpe",
            line={"color": theme_config["colorway"][2], "width": 1.5},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="green", row=2, col=1)

    # === Monthly Returns Heatmap (2,2) ===
    matrix = analysis.get_monthly_returns_matrix()
    years = matrix["year"].to_list()
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    z_values = []
    for row in matrix.iter_rows():
        z_values.append([row[i] if row[i] is not None else np.nan for i in range(1, 13)])

    z_array = np.array(z_values)
    max_abs = np.nanmax(np.abs(z_array)) if not np.all(np.isnan(z_array)) else 0.1

    fig.add_trace(
        go.Heatmap(
            z=z_array,
            x=months,
            y=years,
            colorscale=[
                [0.0, "#d73027"],
                [0.5, "#ffffff"],
                [1.0, "#1a9850"],
            ],
            zmin=-max_abs,
            zmax=max_abs,
            showscale=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title="Portfolio Overview",
        height=height,
        width=width,
        **theme_config["layout"],
    )

    # Format axes
    fig.update_yaxes(tickformat=".0%", row=1, col=2)  # Drawdown
    fig.update_yaxes(autorange="reversed", row=2, col=2)  # Heatmap

    return fig
