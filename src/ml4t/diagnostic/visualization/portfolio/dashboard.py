"""Portfolio dashboard combining all visualizations.

Creates comprehensive tear sheets with multiple panels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from stat import S_IMODE
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ml4t.diagnostic.visualization._colors import COLORS
from ml4t.diagnostic.visualization.core import (
    get_theme_config,
    validate_theme,
)

if TYPE_CHECKING:
    from ml4t.diagnostic.evaluation.portfolio_analysis import (
        HoldingsAnalysisResult,
        PortfolioAnalysis,
        PortfolioMetrics,
    )


@dataclass
class PortfolioTearSheet:
    """Portfolio figures, owned prepared holdings and availability.

    Provides methods to display in notebooks, export to HTML,
    and access individual figures.

    Attributes:
        metrics: Portfolio performance metrics
        figures: Dictionary of individual Plotly figures
        html_content: Pre-rendered HTML content (if generated)
        holdings: Independent prepared snapshot, or None if absent or disabled
        availability: Component statuses and reasons for unavailable/disabled analysis
    """

    metrics: PortfolioMetrics
    figures: dict[str, go.Figure] = field(default_factory=dict)
    html_content: str | None = None
    _analysis: PortfolioAnalysis | None = None
    holdings: HoldingsAnalysisResult | None = None
    availability: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _availability_html(self) -> str:
        if not self.availability:
            return ""
        notes = "".join(
            f"<li><strong>{escape(name.replace('_', ' '))}</strong>: "
            f"{escape(info['status'])}. {escape(info.get('reason') or '')}</li>"
            for name, info in self.availability.items()
            if info.get("reason") or info["status"] != "available"
        )
        return (
            '<section class="metrics-summary"><h2>Analysis availability</h2>'
            "<p>Panels describe the supplied observations. Missing analysis is not zero risk.</p>"
            f"<ul>{notes}</ul></section>"
        )

    def show(self, compact: bool = False) -> None:
        """Display tear sheet in Jupyter notebook or browser.

        In Jupyter: renders the 4-panel overview inline (compact=True) or
        displays all individual figures (compact=False).
        Individual figures use Plotly's configured renderer. Compact mode is
        the existing returns-only overview, not the prepared holdings report.

        Parameters
        ----------
        compact : bool, default False
            If True, show 4-panel overview only. If False, show all figures.
        """
        if compact and self._analysis is not None:
            fig = create_simple_dashboard(self._analysis)
            fig.show()
            return

        # Try Jupyter inline display first
        try:
            from IPython.display import HTML, display

            display(HTML(f"<pre>{escape(self.metrics.summary())}</pre>"))
            if self.availability:
                display(HTML(self._availability_html()))
            for fig in self.figures.values():
                fig.show()
        except ImportError:
            # Terminal fallback: print metrics, show each figure
            print(self.metrics.summary())
            for name, info in self.availability.items():
                if info.get("reason") or info["status"] != "available":
                    print(f"{name.replace('_', ' ')}: {info['status']}. {info.get('reason') or ''}")
            print()
            for name, fig in self.figures.items():
                print(f"\n  {name}")
                fig.show()

    def save_html(
        self,
        path: str | Path,
        include_plotlyjs: bool | str = "cdn",
        full_html: bool = True,
    ) -> None:
        """Save HTML with embedded or linked Plotly.js.

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
            html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Portfolio Tear Sheet</title>
    <style>
        body {{
            font-family: 'DM Sans', 'DejaVu Sans', -apple-system, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: {COLORS["bg_light"]};
            color: {COLORS["blue"]};
        }}
        .metrics-summary {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(10,22,40,0.08);
        }}
        .metrics-summary pre {{
            font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace;
            font-size: 13px;
            line-height: 1.5;
            color: {COLORS["neutral"]};
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(10,22,40,0.08);
            overflow-x: auto;
        }}
        h1 {{
            color: {COLORS["blue"]};
            border-bottom: 2px solid {COLORS["amber"]};
            padding-bottom: 10px;
        }}
        h2 {{
            color: {COLORS["slate"]};
            margin-top: 0;
            font-size: 16px;
        }}
        .figure-note {{ font-size: 13px; line-height: 1.5; }}
    </style>
</head>
<body>
    <h1>Portfolio Tear Sheet</h1>
""")

        # Add metrics summary
        html_parts.append(f"""
    <div class="metrics-summary">
        <h2>Performance Summary</h2>
        <pre>{escape(self.metrics.summary())}</pre>
    </div>
""")

        html_parts.append(self._availability_html())

        # Add each figure
        for name, fig in self.figures.items():
            description = (fig.layout.meta or {}).get("description", "")
            fig = go.Figure(fig).update_layout(title=None)
            fig_html = fig.to_html(
                include_plotlyjs=include_plotlyjs
                if name == list(self.figures.keys())[0]
                else False,
                full_html=False,
            )
            html_parts.append(f"""
    <div class="plot-container">
        <h2>{escape(name)}</h2>
        <p class="figure-note">{escape(description)}</p>
        {fig_html}
    </div>
""")

        if full_html:
            html_parts.append("""
</body>
</html>
""")

        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        created = False
        try:
            with temporary.open("x", encoding="utf-8") as output:
                created = True
                output.write("".join(html_parts))
            if path.exists():
                temporary.chmod(S_IMODE(path.stat().st_mode))
            temporary.replace(path)
            created = False
        finally:
            if created:
                temporary.unlink(missing_ok=True)

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
        from .holdings_plots import _holdings_records

        holdings = (
            None
            if self.holdings is None
            else {
                "positions": _holdings_records(self.holdings.positions),
                "timeline": _holdings_records(self.holdings.timeline),
                "sectors": _holdings_records(self.holdings.sectors),
                "metrics": deepcopy(self.holdings.metrics),
                "coverage": deepcopy(self.holdings.coverage),
                "snapshot_kind": self.holdings.snapshot_kind,
                "top_positions": {
                    side: self.holdings.top_positions(side=side).to_dicts()
                    for side in ["long", "short", "absolute"]
                },
            }
        )
        return {
            "metrics": self.metrics.to_dict(),
            "figures": {name: fig.to_dict() for name, fig in self.figures.items()},
            "holdings": holdings,
            "availability": deepcopy(self.availability),
        }


def create_portfolio_dashboard(
    analysis: PortfolioAnalysis,
    theme: str | None = None,
    include_positions: bool = True,
    include_transactions: bool = True,
    include_benchmark: bool = True,
    height_per_row: int = 300,
    width: int | None = None,
    *,
    holdings: HoldingsAnalysisResult | None = None,
) -> PortfolioTearSheet:
    """Create a return dashboard with optional prepared holdings.

    Includes:
    - Performance metrics summary
    - Cumulative returns chart
    - Drawdown underwater curve
    - Rolling Sharpe ratio
    - Rolling volatility
    - Monthly returns heatmap
    - Returns distribution
    - Prepared holdings panels when supplied and requested

    Parameters
    ----------
    analysis : PortfolioAnalysis
        Portfolio analysis object with returns data
    theme : str, optional
        Plot theme ("default", "dark", "print", "presentation")
    include_positions : bool, default True
        Render explicit holdings, or record why preparation is required
    include_transactions : bool, default True
        Record the request; transaction reporting here remains unavailable
    include_benchmark : bool, default True
        Include benchmark comparison (if benchmark available)
    height_per_row : int, default 300
        Height per subplot row
    width : int, optional
        Figure width
    holdings : HoldingsAnalysisResult, optional
        Explicitly prepared complete holdings on this analysis's exact observation
        keys. Computed analytics are rendered without inferring NAV or positions.
        Requested but absent results have an explicit availability reason.

    Returns
    -------
    PortfolioTearSheet
        Figures, metrics, owned holdings and explicit component availability

    Examples
    --------
    >>> analysis = PortfolioAnalysis(returns=daily_returns)
    >>> tear_sheet = create_portfolio_dashboard(analysis)
    >>> tear_sheet.show()  # Display in notebook
    >>> tear_sheet.save_html("report.html")  # Export to file
    """
    theme = validate_theme(theme)
    get_theme_config(theme)

    analysis.require_dates()
    availability = {
        "positions": {"status": "not_requested", "reason": "Position analysis was not requested."},
        "transactions": {
            "status": "unavailable" if include_transactions else "not_requested",
            "reason": "Prepared transaction reporting is not implemented in this dashboard."
            if include_transactions
            else "Transaction analysis was not requested.",
        },
    }
    prepared = None
    if include_positions:
        if holdings is None:
            availability["positions"] = {
                "status": "unavailable",
                "reason": "Supply holdings=analysis.compute_holdings_analysis(valuations, complete_snapshots=True); NAV and snapshot completeness are not inferred.",
            }
        else:
            from ml4t.diagnostic.evaluation.portfolio_analysis import HoldingsAnalysisResult
            from ml4t.diagnostic.evaluation.portfolio_analysis.inputs import date_index

            if not isinstance(holdings, HoldingsAnalysisResult):
                raise TypeError("holdings must be a prepared HoldingsAnalysisResult")
            actual, expected = date_index(holdings.timeline["date"]), date_index(analysis.dates)
            if (holdings.timeline["date"].dtype == pl.Date) != (analysis.dates.dtype == pl.Date):
                raise ValueError(
                    "holdings observation keys must preserve Date labels versus Datetime values"
                )
            if (
                len(actual) != len(expected)
                or actual.tz != expected.tz
                or not (actual == expected).all()
            ):
                raise ValueError(
                    "holdings must match the exact analysis observation keys and timezone"
                )
            prepared = deepcopy(holdings)
            availability["positions"] = {"status": "available", "reason": None}
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

    if prepared is not None:
        from .holdings_plots import _holdings_figures

        position_figures, position_availability = _holdings_figures(
            prepared, theme, height_per_row, width
        )
        figures.update(position_figures)
        availability.update(position_availability)
    for fig in figures.values():
        fig.update_layout(font_size=14, title_font_size=16, legend_font_size=14)
        fig.update_xaxes(tickfont_size=14, title_font_size=14)
        fig.update_yaxes(tickfont_size=14, title_font_size=14)
        fig.update_annotations(font_size=13)
        for trace in fig.data:
            if hasattr(trace, "colorbar"):
                trace.update(colorbar_tickfont_size=14, colorbar_title_font_size=14)
        if width is None:
            fig.update_layout(width=None, autosize=True)

    return PortfolioTearSheet(
        metrics=metrics,
        figures=figures,
        _analysis=analysis,
        holdings=prepared,
        availability=availability,
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
    analysis.require_dates()
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

    # Update layout — apply theme first, then function-specific overrides
    fig.update_layout(theme_config["layout"])
    fig.update_layout(
        title="Portfolio Overview",
        height=height,
        width=width,
    )

    # Format axes
    fig.update_yaxes(tickformat=".0%", row=1, col=2)  # Drawdown
    fig.update_yaxes(autorange="reversed", row=2, col=2)  # Heatmap

    return fig
