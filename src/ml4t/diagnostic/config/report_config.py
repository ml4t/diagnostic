"""Reporting configuration (Module E).

This module defines configuration for report generation:
- Output formats (HTML, JSON, PDF)
- HTML report settings (templates, themes, tables)
- Visualization settings (plots, colors, interactivity)
- JSON output structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from ml4t.diagnostic.config.base import BaseConfig
from ml4t.diagnostic.config.validation import (
    DataFrameExportFormat,
    PositiveInt,
    ReportFormat,
    ReportTemplate,
    ReportTheme,
    TableFormat,
)


class OutputFormatConfig(BaseConfig):
    """Configuration for output formats and file management.

    Attributes:
        formats: Output formats to generate
        output_dir: Output directory
        filename_template: Filename template with placeholders
        compress: Create .zip if multiple outputs
        overwrite_existing: Overwrite existing files

    Examples:
        >>> # Default: HTML + JSON
        >>> config = OutputFormatConfig()

        >>> # Custom: Only HTML with custom filename
        >>> config = OutputFormatConfig(
        ...     formats=[ReportFormat.HTML],
        ...     filename_template="report_{strategy_name}_{date}.html"
        ... )
    """

    formats: list[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.HTML, ReportFormat.JSON],
        description="Output formats to generate",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd() / "diagnostic_reports",
        description="Output directory",
    )
    filename_template: str = Field(
        "qeval_report_{date}",
        description="Filename template (placeholders: {date}, {strategy_name}, {timestamp})",
    )
    compress: bool = Field(False, description="Create .zip if multiple outputs")
    overwrite_existing: bool = Field(True, description="Overwrite existing files")

    @field_validator("formats")
    @classmethod
    def check_formats(cls, v: list[ReportFormat]) -> list[ReportFormat]:
        """Ensure at least one format specified."""
        if not v:
            raise ValueError("Must specify at least one output format")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class HTMLConfig(BaseConfig):
    """Configuration for HTML report generation.

    Attributes:
        template: HTML template to use
        theme: Visual theme
        color_scheme: Color scheme for plots
        interactive_plots: Use Plotly (True) or matplotlib (False)
        include_sections: Which module sections to include
        table_format: Table styling
        include_toc: Include table of contents
        include_summary: Include executive summary
        custom_css: Path to custom CSS file

    Examples:
        >>> # Default: Full report with dark theme
        >>> config = HTMLConfig()

        >>> # Custom: Summary report with professional theme
        >>> config = HTMLConfig(
        ...     template=ReportTemplate.SUMMARY,
        ...     theme=ReportTheme.PROFESSIONAL,
        ...     include_sections=["module_a", "module_c"]
        ... )
    """

    template: ReportTemplate = Field(
        ReportTemplate.FULL, description="HTML template: full, summary, or diagnostic"
    )
    theme: ReportTheme = Field(
        ReportTheme.LIGHT, description="Visual theme: light, dark, or professional"
    )
    color_scheme: str = Field(
        "viridis", description="Color scheme for plots (matplotlib/plotly colormap)"
    )
    interactive_plots: bool = Field(
        True, description="Use Plotly (interactive) vs matplotlib (static)"
    )
    include_sections: list[str] = Field(
        default_factory=lambda: [
            "stationarity",
            "acf",
            "volatility",
            "distribution",
            "correlation",
            "ic",
            "sharpe",
        ],
        description="Which sections to include (stationarity, acf, volatility, distribution, correlation, pca, clustering, redundancy, ic, binary_classification, threshold_analysis, ml_diagnostics, sharpe, summary)",
    )
    table_format: TableFormat = Field(
        TableFormat.STYLED, description="Table format: styled, plain, or datatables"
    )
    include_toc: bool = Field(True, description="Include table of contents")
    include_summary: bool = Field(True, description="Include executive summary")
    custom_css: Path | None = Field(None, description="Path to custom CSS file")

    @field_validator("include_sections")
    @classmethod
    def check_sections(cls, v: list[str]) -> list[str]:
        """Validate section names."""
        valid_sections = {
            "stationarity",
            "acf",
            "volatility",
            "distribution",
            "correlation",
            "pca",
            "clustering",
            "redundancy",
            "ic",
            "binary_classification",
            "threshold_analysis",
            "ml_diagnostics",
            "sharpe",
            "summary",
        }
        invalid = set(v) - valid_sections
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}. Valid: {valid_sections}")
        return v

    @field_validator("custom_css")
    @classmethod
    def check_custom_css(cls, v: Path | None) -> Path | None:
        """Validate custom CSS exists if specified."""
        if v is not None and not v.exists():
            raise ValueError(f"Custom CSS file not found: {v}")
        return v


class VisualizationConfig(BaseConfig):
    """Configuration for visualization settings.

    Attributes:
        plot_dpi: DPI for static plots (matplotlib)
        plot_width: Plot width in pixels
        plot_height: Plot height in pixels
        max_features_plot: Maximum features to plot (avoid clutter)
        max_points_plot: Maximum points per plot (subsample if needed)
        correlation_heatmap: Include correlation heatmap
        time_series_plots: Include time series plots
        distribution_plots: Include distribution plots (histograms, QQ)
        scatter_plots: Include scatter plots (IC, etc.)
        save_plots: Save plots as separate files
        plot_format: Plot file format (png, svg, pdf)

    Examples:
        >>> # Default: All plots, moderate resolution
        >>> config = VisualizationConfig()

        >>> # Custom: High-res plots for publication
        >>> config = VisualizationConfig(
        ...     plot_dpi=300,
        ...     plot_format="pdf",
        ...     save_plots=True
        ... )
    """

    plot_dpi: PositiveInt = Field(100, description="DPI for static plots")
    plot_width: PositiveInt = Field(800, description="Plot width in pixels")
    plot_height: PositiveInt = Field(600, description="Plot height in pixels")
    max_features_plot: PositiveInt = Field(50, description="Max features to plot (avoid clutter)")
    max_points_plot: PositiveInt | None = Field(
        10000, description="Max points per plot (subsample if needed, None = no limit)"
    )
    correlation_heatmap: bool = Field(True, description="Include correlation heatmap")
    time_series_plots: bool = Field(True, description="Include time series plots")
    distribution_plots: bool = Field(True, description="Include distribution plots")
    scatter_plots: bool = Field(True, description="Include scatter plots")
    save_plots: bool = Field(False, description="Save plots as separate files")
    plot_format: str = Field("png", description="Plot file format (png, svg, pdf)")

    @field_validator("plot_format")
    @classmethod
    def check_plot_format(cls, v: str) -> str:
        """Validate plot format."""
        valid_formats = {"png", "svg", "pdf", "jpg", "jpeg"}
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid plot format: {v}. Valid: {valid_formats}")
        return v.lower()


class JSONConfig(BaseConfig):
    """Configuration for JSON output.

    Attributes:
        pretty_print: Pretty-print JSON (vs compact)
        include_metadata: Include metadata (timestamp, config, versions)
        export_dataframes: DataFrame serialization format
        include_raw_data: Include raw data (features, returns) in output
        indent: JSON indentation (if pretty_print=True)

    Examples:
        >>> # Default: Pretty JSON with metadata
        >>> config = JSONConfig()

        >>> # Custom: Compact JSON without raw data
        >>> config = JSONConfig(
        ...     pretty_print=False,
        ...     include_raw_data=False
        ... )
    """

    pretty_print: bool = Field(True, description="Pretty-print JSON (vs compact)")
    include_metadata: bool = Field(
        True, description="Include metadata (timestamp, config, versions)"
    )
    export_dataframes: DataFrameExportFormat = Field(
        DataFrameExportFormat.RECORDS, description="DataFrame serialization format"
    )
    include_raw_data: bool = Field(
        False, description="Include raw data (features, returns) in output"
    )
    indent: PositiveInt = Field(2, description="JSON indentation (if pretty_print=True)")


class ReportConfig(BaseConfig):
    """Top-level configuration for reporting (Module E).

    Orchestrates report generation:
    - Output formats (HTML, JSON, PDF)
    - HTML settings (templates, themes, tables)
    - Visualization (plots, colors, interactivity)
    - JSON structure

    Attributes:
        output_format: Output format configuration
        html: HTML report configuration
        visualization: Visualization configuration
        json: JSON output configuration
        lazy_rendering: Don't generate plots until accessed
        cache_plots: Cache generated plots
        parallel_plotting: Generate plots in parallel
        n_jobs: Parallel jobs for plotting

    Examples:
        >>> # Quick start with defaults
        >>> config = ReportConfig()
        >>> reporter = Reporter(config)
        >>> reporter.generate(results, output_name="my_strategy")

        >>> # Load from YAML
        >>> config = ReportConfig.from_yaml("report_config.yaml")

        >>> # Custom configuration
        >>> config = ReportConfig(
        ...     output_format=OutputFormatConfig(
        ...         formats=[ReportFormat.HTML, ReportFormat.PDF]
        ...     ),
        ...     html=HTMLConfig(
        ...         template=ReportTemplate.SUMMARY,
        ...         theme=ReportTheme.PROFESSIONAL
        ...     ),
        ...     visualization=VisualizationConfig(
        ...         plot_dpi=300,
        ...         save_plots=True
        ...     )
        ... )
    """

    output_format: OutputFormatConfig = Field(
        default_factory=OutputFormatConfig, description="Output format configuration"
    )
    html: HTMLConfig = Field(default_factory=HTMLConfig, description="HTML report configuration")
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization configuration"
    )
    json_config: JSONConfig = Field(
        default_factory=JSONConfig, description="JSON output configuration"
    )

    # Performance settings
    lazy_rendering: bool = Field(
        False, description="Don't generate plots until accessed (saves time)"
    )
    cache_plots: bool = Field(True, description="Cache generated plots")
    parallel_plotting: bool = Field(False, description="Generate plots in parallel")
    n_jobs: int = Field(-1, ge=-1, description="Parallel jobs for plotting (-1 = all cores)")

    @classmethod
    def for_quick_report(cls) -> ReportConfig:
        """Preset for quick HTML-only report (minimal plots).

        Returns:
            Config optimized for speed
        """
        return cls(
            output_format=OutputFormatConfig(formats=[ReportFormat.HTML]),
            html=HTMLConfig(
                template=ReportTemplate.SUMMARY,
                interactive_plots=False,  # Faster static plots
            ),
            visualization=VisualizationConfig(
                correlation_heatmap=True,
                time_series_plots=False,
                distribution_plots=False,
                scatter_plots=False,
            ),
            lazy_rendering=True,
        )

    @classmethod
    def for_publication(cls) -> ReportConfig:
        """Preset for publication-quality reports (high-res, all plots).

        Returns:
            Config optimized for publication
        """
        return cls(
            output_format=OutputFormatConfig(
                formats=[ReportFormat.HTML, ReportFormat.PDF],
                compress=True,
            ),
            html=HTMLConfig(
                template=ReportTemplate.FULL,
                theme=ReportTheme.PROFESSIONAL,
                table_format=TableFormat.STYLED,
            ),
            visualization=VisualizationConfig(
                plot_dpi=300,
                plot_format="pdf",
                save_plots=True,
                correlation_heatmap=True,
                time_series_plots=True,
                distribution_plots=True,
                scatter_plots=True,
            ),
            json_config=JSONConfig(pretty_print=True, include_metadata=True),
            cache_plots=True,
            parallel_plotting=True,
        )

    @classmethod
    def for_programmatic_access(cls) -> ReportConfig:
        """Preset for programmatic access (JSON only, no plots).

        Returns:
            Config optimized for API/programmatic use
        """
        return cls(
            output_format=OutputFormatConfig(formats=[ReportFormat.JSON]),
            visualization=VisualizationConfig(
                correlation_heatmap=False,
                time_series_plots=False,
                distribution_plots=False,
                scatter_plots=False,
            ),
            json_config=JSONConfig(
                pretty_print=False,  # Compact for parsing
                include_raw_data=True,  # Include data for downstream processing
                export_dataframes=DataFrameExportFormat.SPLIT,  # Efficient format
            ),
            lazy_rendering=True,
        )
