"""Barrier tear sheet containing all barrier analysis results.

This module provides the BarrierTearSheet class that aggregates all barrier
analysis results (hit rates, profit factor, precision/recall, time-to-target)
into a single exportable result object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.barrier_results.hit_rate import HitRateResult
from ml4t.diagnostic.results.barrier_results.precision_recall import PrecisionRecallResult
from ml4t.diagnostic.results.barrier_results.profit_factor import ProfitFactorResult
from ml4t.diagnostic.results.barrier_results.time_to_target import TimeToTargetResult
from ml4t.diagnostic.results.base import BaseResult


class BarrierTearSheet(BaseResult):
    """Complete tear sheet containing all barrier analysis results.

    Aggregates hit rates, profit factor, and visualization data into
    a single exportable result object.

    Examples
    --------
    >>> tear_sheet = barrier_analysis.create_tear_sheet()
    >>> tear_sheet.show()  # Display in Jupyter
    >>> tear_sheet.save_html("barrier_report.html")
    """

    analysis_type: str = Field(default="barrier_tear_sheet", frozen=True)

    # ==========================================================================
    # Component Results
    # ==========================================================================

    hit_rate_result: HitRateResult | None = Field(
        default=None,
        description="Hit rate analysis results",
    )

    profit_factor_result: ProfitFactorResult | None = Field(
        default=None,
        description="Profit factor analysis results",
    )

    precision_recall_result: PrecisionRecallResult | None = Field(
        default=None,
        description="Precision/recall analysis results",
    )

    time_to_target_result: TimeToTargetResult | None = Field(
        default=None,
        description="Time-to-target analysis results",
    )

    # ==========================================================================
    # Metadata
    # ==========================================================================

    signal_name: str = Field(
        default="signal",
        description="Name of the signal analyzed",
    )

    n_assets: int = Field(
        ...,
        description="Number of unique assets",
    )

    n_dates: int = Field(
        ...,
        description="Number of unique dates",
    )

    n_observations: int = Field(
        ...,
        description="Total number of observations analyzed",
    )

    date_range: tuple[str, str] = Field(
        ...,
        description="Date range (start, end) in ISO format",
    )

    # ==========================================================================
    # Figures (stored as JSON for serialization)
    # ==========================================================================

    figures: dict[str, Any] = Field(
        default_factory=dict,
        description="Plotly figures as JSON (for HTML export)",
    )

    def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
        """Get results as Polars DataFrame.

        Parameters
        ----------
        name : str | None
            DataFrame to retrieve - routes to component results

        Returns
        -------
        pl.DataFrame
            Requested DataFrame
        """
        if name is None or name == "summary":
            return self._build_summary_df()

        # Route to component results
        if name.startswith("hit_rate_"):
            if self.hit_rate_result is None:
                raise ValueError("Hit rate analysis not available")
            component_name = name[9:] if name != "hit_rate_result" else None
            return self.hit_rate_result.get_dataframe(component_name)

        if name.startswith("profit_factor_"):
            if self.profit_factor_result is None:
                raise ValueError("Profit factor analysis not available")
            component_name = name[14:] if name != "profit_factor_result" else None
            return self.profit_factor_result.get_dataframe(component_name)

        if name.startswith("precision_recall_"):
            if self.precision_recall_result is None:
                raise ValueError("Precision/recall analysis not available")
            component_name = name[17:] if name != "precision_recall_result" else None
            return self.precision_recall_result.get_dataframe(component_name)

        if name.startswith("time_to_target_"):
            if self.time_to_target_result is None:
                raise ValueError("Time-to-target analysis not available")
            component_name = name[15:] if name != "time_to_target_result" else None
            return self.time_to_target_result.get_dataframe(component_name)

        raise ValueError(
            f"Unknown DataFrame name: {name}. Use 'summary' or prefix with "
            "'hit_rate_', 'profit_factor_', 'precision_recall_', 'time_to_target_'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        available = ["summary"]
        if self.hit_rate_result:
            available.extend(
                [f"hit_rate_{n}" for n in self.hit_rate_result.list_available_dataframes()]
            )
        if self.profit_factor_result:
            available.extend(
                [
                    f"profit_factor_{n}"
                    for n in self.profit_factor_result.list_available_dataframes()
                ]
            )
        if self.precision_recall_result:
            available.extend(
                [
                    f"precision_recall_{n}"
                    for n in self.precision_recall_result.list_available_dataframes()
                ]
            )
        if self.time_to_target_result:
            available.extend(
                [
                    f"time_to_target_{n}"
                    for n in self.time_to_target_result.list_available_dataframes()
                ]
            )
        return available

    def _build_summary_df(self) -> pl.DataFrame:
        """Build summary DataFrame with key metrics."""
        rows: list[dict[str, str]] = [
            {"metric": "signal_name", "value": self.signal_name},
            {"metric": "n_assets", "value": str(self.n_assets)},
            {"metric": "n_dates", "value": str(self.n_dates)},
            {"metric": "n_observations", "value": str(self.n_observations)},
            {"metric": "date_range_start", "value": self.date_range[0]},
            {"metric": "date_range_end", "value": self.date_range[1]},
        ]

        if self.hit_rate_result:
            rows.append(
                {
                    "metric": "overall_hit_rate_tp",
                    "value": f"{self.hit_rate_result.overall_hit_rate_tp:.4f}",
                }
            )
            rows.append(
                {"metric": "chi2_significant", "value": str(self.hit_rate_result.is_significant)}
            )

        if self.profit_factor_result:
            rows.append(
                {
                    "metric": "overall_profit_factor",
                    "value": f"{self.profit_factor_result.overall_profit_factor:.4f}",
                }
            )

        if self.precision_recall_result:
            rows.append(
                {
                    "metric": "baseline_tp_rate",
                    "value": f"{self.precision_recall_result.baseline_tp_rate:.4f}",
                }
            )
            rows.append(
                {
                    "metric": "best_f1_score",
                    "value": f"{self.precision_recall_result.best_f1_score:.4f}",
                }
            )
            rows.append(
                {
                    "metric": "best_f1_quantile",
                    "value": self.precision_recall_result.best_f1_quantile,
                }
            )

        if self.time_to_target_result:
            rows.append(
                {
                    "metric": "overall_mean_bars",
                    "value": f"{self.time_to_target_result.overall_mean_bars:.1f}",
                }
            )
            rows.append(
                {
                    "metric": "overall_mean_bars_tp",
                    "value": f"{self.time_to_target_result.overall_mean_bars_tp:.1f}",
                }
            )

        return pl.DataFrame(rows)

    def summary(self) -> str:
        """Get human-readable summary of complete tear sheet."""
        lines = [
            "=" * 60,
            f"Barrier Analysis Tear Sheet: {self.signal_name}",
            "=" * 60,
            "",
            f"Assets:       {self.n_assets:>10,}",
            f"Dates:        {self.n_dates:>10,}",
            f"Observations: {self.n_observations:>10,}",
            f"Range:        {self.date_range[0]} to {self.date_range[1]}",
            f"Created:      {self.created_at}",
            "",
        ]

        if self.hit_rate_result:
            lines.append("--- Hit Rate Analysis ---")
            lines.append(self.hit_rate_result.summary())
            lines.append("")

        if self.profit_factor_result:
            lines.append("--- Profit Factor Analysis ---")
            lines.append(self.profit_factor_result.summary())
            lines.append("")

        if self.precision_recall_result:
            lines.append("--- Precision/Recall Analysis ---")
            lines.append(self.precision_recall_result.summary())
            lines.append("")

        if self.time_to_target_result:
            lines.append("--- Time-to-Target Analysis ---")
            lines.append(self.time_to_target_result.summary())

        return "\n".join(lines)

    def show(self) -> None:
        """Display tear sheet in Jupyter notebook."""
        try:
            from IPython.display import HTML, display

            display(HTML(f"<h2>Barrier Analysis: {self.signal_name}</h2>"))
            display(
                HTML(
                    f"<p>{self.n_assets} assets, {self.n_dates} dates, {self.n_observations} observations</p>"
                )
            )

            for _name, fig_json in self.figures.items():
                import plotly.io as pio

                fig = pio.from_json(fig_json)
                fig.show()

        except ImportError:
            print("IPython not available. Use save_html() instead.")
            print(self.summary())

    def save_html(
        self,
        path: str | Path,
        include_plotlyjs: str | bool = "cdn",
    ) -> Path:
        """Save tear sheet as self-contained HTML file.

        Parameters
        ----------
        path : str | Path
            Output file path
        include_plotlyjs : str | bool
            How to include plotly.js: 'cdn', 'directory', True (embed), False

        Returns
        -------
        Path
            Path to saved file
        """
        import plotly.io as pio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # NOTE: Plotly.js is included via pio.to_html with include_plotlyjs parameter
        # Do NOT add hardcoded CDN script here - it would duplicate the inclusion
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Barrier Analysis: {self.signal_name}</title>",
            "<style>",
            "body { font-family: -apple-system, system-ui, sans-serif; margin: 40px; }",
            "h1 { color: #2C3E50; }",
            ".summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }",
            ".plot-container { margin-bottom: 40px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Barrier Analysis: {self.signal_name}</h1>",
            "<div class='summary'>",
            f"<p><strong>Assets:</strong> {self.n_assets:,}</p>",
            f"<p><strong>Dates:</strong> {self.n_dates:,}</p>",
            f"<p><strong>Observations:</strong> {self.n_observations:,}</p>",
            f"<p><strong>Range:</strong> {self.date_range[0]} to {self.date_range[1]}</p>",
            f"<p><strong>Generated:</strong> {self.created_at}</p>",
            "</div>",
        ]

        # Add figures
        plotlyjs_included = False
        for name, fig_json in self.figures.items():
            fig = pio.from_json(fig_json)
            fig_html = pio.to_html(
                fig,
                include_plotlyjs=include_plotlyjs if not plotlyjs_included else False,
                full_html=False,
            )
            html_parts.append("<div class='plot-container'>")
            html_parts.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
            html_parts.append(fig_html)
            html_parts.append("</div>")
            plotlyjs_included = True

        html_parts.extend(["</body>", "</html>"])
        path.write_text("\n".join(html_parts))

        return path

    def save_json(self, path: str | Path, exclude_figures: bool = False) -> Path:
        """Export all metrics as structured JSON.

        Parameters
        ----------
        path : str | Path
            Output file path
        exclude_figures : bool, default=False
            If True, exclude figure JSON data to reduce file size

        Returns
        -------
        Path
            Path to saved file
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict(exclude_none=True)

        if exclude_figures:
            data.pop("figures", None)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return path
