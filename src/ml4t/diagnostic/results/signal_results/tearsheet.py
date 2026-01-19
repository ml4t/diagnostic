"""SignalTearSheet class for complete signal analysis results.

This module provides the SignalTearSheet class that aggregates all signal
analysis components (IC, quantile, turnover, IR_tc) into a single exportable
result object with visualization and export capabilities.

References
----------
Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import Field

from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.results.signal_results.ic import SignalICResult
from ml4t.diagnostic.results.signal_results.irtc import IRtcResult
from ml4t.diagnostic.results.signal_results.quantile import QuantileAnalysisResult
from ml4t.diagnostic.results.signal_results.turnover import TurnoverAnalysisResult
from ml4t.diagnostic.results.signal_results.validation import _figure_from_data


class SignalTearSheet(BaseResult):
    """Complete tear sheet containing all signal analysis results.

    Aggregates IC, quantile, turnover, and visualization data into
    a single exportable result object.

    Examples
    --------
    >>> tear_sheet = signal_analysis.create_tear_sheet()
    >>> tear_sheet.show()  # Display in Jupyter
    >>> tear_sheet.save_html("signal_report.html")
    """

    analysis_type: str = Field(default="signal_tear_sheet", frozen=True)

    # ==========================================================================
    # Component Results
    # ==========================================================================

    ic_analysis: SignalICResult | None = Field(
        default=None,
        description="Signal IC analysis results",
    )

    quantile_analysis: QuantileAnalysisResult | None = Field(
        default=None,
        description="Quantile analysis results",
    )

    turnover_analysis: TurnoverAnalysisResult | None = Field(
        default=None,
        description="Turnover analysis results",
    )

    ir_tc_analysis: IRtcResult | None = Field(
        default=None,
        description="IR_tc analysis results",
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
        if name.startswith("ic_"):
            if self.ic_analysis is None:
                raise ValueError("IC analysis not available")
            component_name = name[3:] if name != "ic_analysis" else None
            return self.ic_analysis.get_dataframe(component_name)

        if name.startswith("quantile_"):
            if self.quantile_analysis is None:
                raise ValueError("Quantile analysis not available")
            component_name = name[9:] if name != "quantile_analysis" else None
            return self.quantile_analysis.get_dataframe(component_name)

        if name.startswith("turnover_"):
            if self.turnover_analysis is None:
                raise ValueError("Turnover analysis not available")
            component_name = name[9:] if name != "turnover_analysis" else None
            return self.turnover_analysis.get_dataframe(component_name)

        raise ValueError(
            f"Unknown DataFrame name: {name}. Use 'summary' or prefix with "
            "'ic_', 'quantile_', 'turnover_'"
        )

    def list_available_dataframes(self) -> list[str]:
        """List available DataFrame views."""
        available = ["summary"]
        if self.ic_analysis:
            available.extend([f"ic_{n}" for n in self.ic_analysis.list_available_dataframes()])
        if self.quantile_analysis:
            available.extend(
                [f"quantile_{n}" for n in self.quantile_analysis.list_available_dataframes()]
            )
        if self.turnover_analysis:
            available.extend(
                [f"turnover_{n}" for n in self.turnover_analysis.list_available_dataframes()]
            )
        return available

    def _build_summary_df(self) -> pl.DataFrame:
        """Build summary DataFrame with key metrics."""
        rows = [
            {"metric": "signal_name", "value": self.signal_name},
            {"metric": "n_assets", "value": str(self.n_assets)},
            {"metric": "n_dates", "value": str(self.n_dates)},
            {"metric": "date_range_start", "value": self.date_range[0]},
            {"metric": "date_range_end", "value": self.date_range[1]},
        ]

        if self.ic_analysis:
            for period, ic in self.ic_analysis.ic_mean.items():
                rows.append({"metric": f"ic_mean_{period}", "value": f"{ic:.4f}"})

        return pl.DataFrame(rows)

    def summary(self) -> str:
        """Get human-readable summary of complete tear sheet."""
        lines = [
            "=" * 60,
            f"Signal Analysis Tear Sheet: {self.signal_name}",
            "=" * 60,
            "",
            f"Assets:     {self.n_assets:>10}",
            f"Dates:      {self.n_dates:>10}",
            f"Range:      {self.date_range[0]} to {self.date_range[1]}",
            f"Created:    {self.created_at}",
            "",
        ]

        if self.ic_analysis:
            lines.append("--- IC Analysis ---")
            lines.append(self.ic_analysis.summary())

        if self.quantile_analysis:
            lines.append("--- Quantile Analysis ---")
            lines.append(self.quantile_analysis.summary())

        if self.turnover_analysis:
            lines.append("--- Turnover Analysis ---")
            lines.append(self.turnover_analysis.summary())

        if self.ir_tc_analysis:
            lines.append("--- IR_tc Analysis ---")
            lines.append(self.ir_tc_analysis.summary())

        return "\n".join(lines)

    def show(self) -> None:
        """Display tear sheet in Jupyter notebook.

        Renders all figures inline using IPython display.
        Falls back to printing summary if not in a Jupyter environment.
        """
        try:
            from IPython.display import HTML, display

            # Display summary
            display(HTML(f"<h2>Signal Analysis: {self.signal_name}</h2>"))
            display(HTML(f"<p>{self.n_assets} assets, {self.n_dates} dates</p>"))

            # Display figures
            for _name, fig_json in self.figures.items():
                fig = _figure_from_data(fig_json)
                fig.show()

        except (ImportError, ValueError):
            # ImportError: IPython not installed
            # ValueError: nbformat not installed (required for Plotly mime rendering)
            print("Interactive display not available. Use save_html() instead.")
            print(self.summary())

    def save_html(
        self,
        path: str | Path,
        use_dashboard: bool = True,
        include_plotlyjs: str | bool = "cdn",
        theme: Literal["light", "dark"] = "light",
    ) -> Path:
        """Save tear sheet as self-contained HTML file.

        Parameters
        ----------
        path : str | Path
            Output file path
        use_dashboard : bool, default=True
            If True, use multi-tab SignalDashboard format.
            If False, use simple stacked plot layout.
        include_plotlyjs : str | bool
            How to include plotly.js: 'cdn', 'directory', True (embed), False
        theme : str, default='light'
            Theme for dashboard: 'light' or 'dark' (only used if use_dashboard=True)

        Returns
        -------
        Path
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if use_dashboard:
            # Use multi-tab dashboard format
            from ml4t.diagnostic.visualization.signal.dashboard import SignalDashboard

            dashboard = SignalDashboard(
                title=f"Signal Analysis: {self.signal_name}",
                theme=theme,
            )
            html = dashboard.generate(self)
            path.write_text(html)
        else:
            # Use simple stacked layout (legacy behavior)
            import plotly.io as pio

            # NOTE: Plotly.js is included via pio.to_html with include_plotlyjs parameter
            # Do NOT add hardcoded CDN script here - it would duplicate the inclusion
            html_parts = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                f"<title>Signal Analysis: {self.signal_name}</title>",
                "<style>",
                "body { font-family: -apple-system, system-ui, sans-serif; margin: 40px; }",
                "h1 { color: #2C3E50; }",
                ".summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }",
                ".plot-container { margin-bottom: 40px; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>Signal Analysis: {self.signal_name}</h1>",
                "<div class='summary'>",
                f"<p><strong>Assets:</strong> {self.n_assets}</p>",
                f"<p><strong>Dates:</strong> {self.n_dates}</p>",
                f"<p><strong>Range:</strong> {self.date_range[0]} to {self.date_range[1]}</p>",
                f"<p><strong>Generated:</strong> {self.created_at}</p>",
                "</div>",
            ]

            # Add figures
            plotlyjs_included = False
            for name, fig_json in self.figures.items():
                fig = _figure_from_data(fig_json)
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

        Examples
        --------
        >>> tear_sheet.save_json("signal_metrics.json")
        >>> tear_sheet.save_json("signal_compact.json", exclude_figures=True)
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

    def save_png(
        self,
        output_dir: str | Path,
        figures: list[str] | None = None,
        width: int = 1200,
        height: int = 600,
        scale: float = 2.0,
    ) -> list[Path]:
        """Export figures as PNG images.

        Requires the `kaleido` package for static image export.
        Install with: pip install kaleido

        Parameters
        ----------
        output_dir : str | Path
            Output directory for PNG files
        figures : list[str] | None
            List of figure names to export. If None, exports all figures.
        width : int, default=1200
            Image width in pixels
        height : int, default=600
            Image height in pixels
        scale : float, default=2.0
            Scale factor for resolution (2.0 = 2x resolution)

        Returns
        -------
        list[Path]
            Paths to saved PNG files

        Raises
        ------
        ImportError
            If kaleido is not installed

        Examples
        --------
        >>> paths = tear_sheet.save_png("./images/")
        >>> paths = tear_sheet.save_png("./images/", figures=["ic_time_series"])
        """
        try:
            import plotly.io as pio

            # Check if kaleido is available
            pio.kaleido.scope  # noqa: B018 - Check if kaleido is installed
        except (ImportError, AttributeError) as e:
            raise ImportError(
                "kaleido is required for PNG export. Install with: pip install kaleido"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        figure_names = figures if figures is not None else list(self.figures.keys())

        for name in figure_names:
            if name not in self.figures:
                continue

            fig_json = self.figures[name]
            fig = _figure_from_data(fig_json)

            output_path = output_dir / f"{name}.png"
            fig.write_image(
                str(output_path),
                width=width,
                height=height,
                scale=scale,
            )
            saved_paths.append(output_path)

        return saved_paths

    def to_dashboard(self, theme: Literal["light", "dark"] = "light") -> Any:
        """Convert to SignalDashboard for customization.

        Returns a SignalDashboard instance that can be further customized
        before generating HTML output.

        Parameters
        ----------
        theme : Literal["light", "dark"], default='light'
            Dashboard theme: 'light' or 'dark'

        Returns
        -------
        SignalDashboard
            Dashboard instance ready for customization

        Examples
        --------
        >>> dashboard = tear_sheet.to_dashboard(theme="dark")
        >>> dashboard.title = "Custom Title"
        >>> html = dashboard.generate(tear_sheet)
        """
        from ml4t.diagnostic.visualization.signal.dashboard import SignalDashboard

        return SignalDashboard(
            title=f"Signal Analysis: {self.signal_name}",
            theme=theme,
        )

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Export to dictionary, excluding large figure data by default."""
        data = super().to_dict(exclude_none=exclude_none)
        # Optionally exclude figures to reduce size
        if exclude_none and not self.figures:
            data.pop("figures", None)
        return data
