"""Report generation for feature diagnostic analysis.

This module provides functionality to generate comprehensive reports from
FeatureDiagnostics results in multiple formats:

- **HTML**: Interactive reports with embedded Plotly charts
- **JSON**: Structured data for programmatic access
- **Markdown**: Documentation-friendly format

The reports can be generated for single features or multiple features
in comparative format.

Key Features:
    - Interactive HTML reports with embedded visualizations
    - Customizable templates for branding/styling
    - JSON export with full diagnostic data
    - Markdown reports for documentation
    - Multi-feature comparison reports
    - Standalone files (no external dependencies)

Example:
    >>> from ml4t.diagnostic.evaluation import FeatureDiagnostics, generate_html_report
    >>> import numpy as np
    >>>
    >>> # Run diagnostics
    >>> diagnostics = FeatureDiagnostics()
    >>> data = np.random.randn(1000)
    >>> result = diagnostics.run_diagnostics(data, name="momentum")
    >>>
    >>> # Generate HTML report
    >>> html = generate_html_report(result, include_plots=True)
    >>> with open("diagnostics_report.html", "w") as f:
    ...     f.write(html)
    >>>
    >>> # Generate JSON export
    >>> json_data = generate_json_report(result)
    >>> with open("diagnostics.json", "w") as f:
    ...     f.write(json_data)
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feature_diagnostics import FeatureDiagnosticsResult

__all__ = [
    "generate_html_report",
    "generate_json_report",
    "generate_markdown_report",
    "generate_multi_feature_html_report",
    "save_report",
]


def generate_html_report(
    result: FeatureDiagnosticsResult,
    include_plots: bool = True,
    title: str | None = None,
    template: str | None = None,
) -> str:
    """Generate interactive HTML report for feature diagnostics.

    Creates a standalone HTML file with embedded Plotly charts and
    comprehensive diagnostic results. The report is fully self-contained
    with no external dependencies.

    Args:
        result: FeatureDiagnosticsResult from diagnostic analysis
        include_plots: Whether to embed interactive Plotly charts
        title: Custom report title (default: "Feature Diagnostics: {name}")
        template: Custom HTML template (None = use default)

    Returns:
        Complete HTML document as string

    Example:
        >>> diagnostics = FeatureDiagnostics()
        >>> result = diagnostics.run_diagnostics(data, name="momentum")
        >>> html = generate_html_report(result, include_plots=True)
        >>> with open("report.html", "w") as f:
        ...     f.write(html)
    """
    if template is not None:
        # Use custom template
        return _render_custom_template(result, template, include_plots)

    # Use default template
    return _generate_default_html(result, include_plots, title)


def generate_json_report(
    result: FeatureDiagnosticsResult,
    indent: int = 2,
) -> str:
    """Generate JSON export of diagnostic results.

    Exports all diagnostic data in structured JSON format for programmatic
    access. Includes test statistics, p-values, recommendations, and
    summary information.

    Args:
        result: FeatureDiagnosticsResult from diagnostic analysis
        indent: JSON indentation level (None for compact)

    Returns:
        JSON string with complete diagnostic data

    Example:
        >>> result = diagnostics.run_diagnostics(data, name="momentum")
        >>> json_data = generate_json_report(result)
        >>> with open("diagnostics.json", "w") as f:
        ...     f.write(json_data)
        >>> # Later, load and analyze
        >>> import json
        >>> with open("diagnostics.json") as f:
        ...     data = json.load(f)
        >>> print(data['health_score'])
    """
    # Convert result to dictionary
    data = _result_to_dict(result)

    # Add metadata
    try:
        pkg_version = get_version("ml4t-diagnostic")
    except Exception:
        pkg_version = "unknown"
    data["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "qeval_version": pkg_version,
        "format_version": "1.0",
    }

    return json.dumps(data, indent=indent, default=str)


def generate_markdown_report(
    result: FeatureDiagnosticsResult,
    include_summary_table: bool = True,
    include_recommendations: bool = True,
) -> str:
    """Generate Markdown report for feature diagnostics.

    Creates a documentation-friendly Markdown report with test results,
    summary table, and recommendations. Suitable for version control,
    documentation systems, or inclusion in notebooks.

    Args:
        result: FeatureDiagnosticsResult from diagnostic analysis
        include_summary_table: Whether to include summary DataFrame as table
        include_recommendations: Whether to include recommendation list

    Returns:
        Markdown-formatted report

    Example:
        >>> result = diagnostics.run_diagnostics(data, name="momentum")
        >>> markdown = generate_markdown_report(result)
        >>> with open("diagnostics.md", "w") as f:
        ...     f.write(markdown)
    """
    lines = []

    # Header
    lines.append(f"# Feature Diagnostics: {result.feature_name}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Observations**: {result.n_obs:,}")
    lines.append(f"**Health Score**: {result.health_score:.2f}/1.00")
    lines.append("")

    # Flags (if any)
    if result.flags:
        lines.append("## ⚠️ Flags")
        lines.append("")
        for flag in result.flags:
            lines.append(f"- {flag}")
        lines.append("")

    # Summary table
    if include_summary_table and not result.summary_df.empty:
        lines.append("## Test Summary")
        lines.append("")
        lines.append(result.summary_df.to_markdown(index=False))
        lines.append("")

    # Module-specific results
    if result.stationarity is not None:
        lines.append("## Stationarity Analysis")
        lines.append("")
        lines.append(f"**Consensus**: {result.stationarity.consensus}")
        lines.append("")

        if result.stationarity.adf_result is not None:
            adf = result.stationarity.adf_result
            lines.append(
                f"- **ADF**: statistic={adf.test_statistic:.4f}, p-value={adf.p_value:.4f}"
            )

        if result.stationarity.kpss_result is not None:
            kpss = result.stationarity.kpss_result
            lines.append(
                f"- **KPSS**: statistic={kpss.test_statistic:.4f}, p-value={kpss.p_value:.4f}"
            )

        if result.stationarity.pp_result is not None:
            pp = result.stationarity.pp_result
            lines.append(f"- **PP**: statistic={pp.test_statistic:.4f}, p-value={pp.p_value:.4f}")

        lines.append("")

    if result.autocorrelation is not None:
        lines.append("## Autocorrelation Analysis")
        lines.append("")
        n_sig_acf = len(result.autocorrelation.significant_acf_lags)
        n_sig_pacf = len(result.autocorrelation.significant_pacf_lags)
        lines.append(f"- **Significant ACF lags**: {n_sig_acf}")
        lines.append(f"- **Significant PACF lags**: {n_sig_pacf}")
        lines.append(f"- **Suggested ARIMA order**: {result.autocorrelation.suggested_arima_order}")
        lines.append(
            f"- **White noise**: {'Yes' if result.autocorrelation.is_white_noise else 'No'}"
        )
        lines.append("")

    if result.volatility is not None:
        lines.append("## Volatility Analysis")
        lines.append("")
        has_clustering = "Yes" if result.volatility.has_volatility_clustering else "No"
        lines.append(f"- **Volatility clustering**: {has_clustering}")
        if result.volatility.arch_lm_result is not None:
            arch = result.volatility.arch_lm_result
            lines.append(
                f"- **ARCH-LM**: statistic={arch.test_statistic:.4f}, p-value={arch.p_value:.4f}"
            )
        lines.append("")

    if result.distribution is not None:
        lines.append("## Distribution Analysis")
        lines.append("")
        lines.append(
            f"- **Recommended distribution**: {result.distribution.recommended_distribution}"
        )
        lines.append(f"- **Is normal**: {'Yes' if result.distribution.is_normal else 'No'}")

        if result.distribution.moments_result is not None:
            mom = result.distribution.moments_result
            lines.append(f"- **Mean**: {mom.mean:.6f}")
            lines.append(f"- **Std Dev**: {mom.std:.6f}")
            lines.append(
                f"- **Skewness**: {mom.skewness:.4f} ({'significant' if mom.skewness_significant else 'not significant'})"
            )
            lines.append(
                f"- **Excess Kurtosis**: {mom.excess_kurtosis:.4f} ({'significant' if mom.excess_kurtosis_significant else 'not significant'})"
            )

        if result.distribution.jarque_bera_result is not None:
            jb = result.distribution.jarque_bera_result
            lines.append(
                f"- **Jarque-Bera**: statistic={jb.statistic:.4f}, p-value={jb.p_value:.4f}"
            )

        lines.append("")

    # Recommendations
    if include_recommendations and result.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


def generate_multi_feature_html_report(
    results: list[FeatureDiagnosticsResult],
    include_plots: bool = True,
    title: str = "Multi-Feature Diagnostic Report",
) -> str:
    """Generate comparative HTML report for multiple features.

    Creates a single HTML report comparing diagnostics across multiple
    features. Useful for portfolio-level analysis or comparing alternative
    feature transformations.

    Args:
        results: List of FeatureDiagnosticsResult objects
        include_plots: Whether to embed interactive Plotly charts
        title: Report title

    Returns:
        Complete HTML document as string

    Example:
        >>> results = []
        >>> for name, data in features.items():
        ...     result = diagnostics.run_diagnostics(data, name=name)
        ...     results.append(result)
        >>> html = generate_multi_feature_html_report(results)
        >>> with open("portfolio_diagnostics.html", "w") as f:
        ...     f.write(html)
    """
    if not results:
        raise ValueError("results list cannot be empty")

    # Build comparison table
    import pandas as pd

    comparison_data = []
    for result in results:
        row = {
            "Feature": result.feature_name,
            "N": result.n_obs,
            "Health Score": f"{result.health_score:.2f}",
            "Flags": len(result.flags),
        }

        if result.stationarity is not None:
            row["Stationarity"] = result.stationarity.consensus

        if result.autocorrelation is not None:
            row["Significant ACF Lags"] = len(result.autocorrelation.significant_acf_lags)

        if result.volatility is not None:
            row["Vol Clustering"] = "Yes" if result.volatility.has_volatility_clustering else "No"

        if result.distribution is not None:
            row["Distribution"] = result.distribution.recommended_distribution

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Generate HTML
    html_parts = []

    # Header
    html_parts.append(_html_header(title))

    # Overview section
    html_parts.append("<h2>Overview</h2>")
    html_parts.append(f"<p><strong>Features analyzed:</strong> {len(results)}</p>")
    html_parts.append(
        f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    )

    # Comparison table
    html_parts.append("<h2>Comparison Table</h2>")
    html_parts.append(comparison_df.to_html(index=False, classes="dataframe"))

    # Individual feature sections
    for result in results:
        html_parts.append("<hr>")
        html_parts.append(f"<h2>Feature: {result.feature_name}</h2>")
        html_parts.append(_generate_feature_section_html(result, include_plots))

    # Footer
    html_parts.append(_html_footer())

    return "\n".join(html_parts)


def save_report(
    content: str,
    filepath: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save report to file.

    Args:
        content: Report content (HTML, JSON, or Markdown)
        filepath: Destination file path
        overwrite: Whether to overwrite existing file

    Returns:
        Path to saved file

    Raises:
        FileExistsError: If file exists and overwrite=False

    Example:
        >>> html = generate_html_report(result)
        >>> save_report(html, "diagnostics.html", overwrite=True)
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File {filepath} already exists. Set overwrite=True to replace.")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


# ============================================================================
# Private helper functions
# ============================================================================


def _result_to_dict(result: FeatureDiagnosticsResult) -> dict[str, Any]:
    """Convert FeatureDiagnosticsResult to dictionary.

    Args:
        result: Diagnostic result object

    Returns:
        Dictionary with all diagnostic data
    """
    data: dict[str, Any] = {
        "feature_name": result.feature_name,
        "n_obs": result.n_obs,
        "health_score": result.health_score,
        "flags": result.flags,
        "recommendations": result.recommendations,
    }

    # Summary DataFrame
    if not result.summary_df.empty:
        data["summary"] = result.summary_df.to_dict(orient="records")

    # Module results
    if result.stationarity is not None:
        data["stationarity"] = _stationarity_to_dict(result.stationarity)

    if result.autocorrelation is not None:
        data["autocorrelation"] = _autocorrelation_to_dict(result.autocorrelation)

    if result.volatility is not None:
        data["volatility"] = _volatility_to_dict(result.volatility)

    if result.distribution is not None:
        data["distribution"] = _distribution_to_dict(result.distribution)

    return data


def _stationarity_to_dict(result) -> dict[str, Any]:
    """Convert StationarityAnalysisResult to dict."""
    data = {"consensus": result.consensus}

    if result.adf_result is not None:
        data["adf"] = {
            "test_statistic": result.adf_result.test_statistic,
            "p_value": result.adf_result.p_value,
            "critical_values": result.adf_result.critical_values,
            "is_stationary": result.adf_result.is_stationary,
        }

    if result.kpss_result is not None:
        data["kpss"] = {
            "test_statistic": result.kpss_result.test_statistic,
            "p_value": result.kpss_result.p_value,
            "critical_values": result.kpss_result.critical_values,
            "is_stationary": result.kpss_result.is_stationary,
        }

    if result.pp_result is not None:
        data["pp"] = {
            "test_statistic": result.pp_result.test_statistic,
            "p_value": result.pp_result.p_value,
            "critical_values": result.pp_result.critical_values,
            "is_stationary": result.pp_result.is_stationary,
        }

    return data


def _autocorrelation_to_dict(result) -> dict[str, Any]:
    """Convert AutocorrelationAnalysisResult to dict."""
    return {
        "acf_values": result.acf_result.acf_values.tolist() if result.acf_result else None,
        "pacf_values": result.pacf_result.pacf_values.tolist() if result.pacf_result else None,
        "significant_acf_lags": result.significant_acf_lags,
        "significant_pacf_lags": result.significant_pacf_lags,
        "suggested_arima_order": list(result.suggested_arima_order),
        "is_white_noise": result.is_white_noise,
    }


def _volatility_to_dict(result) -> dict[str, Any]:
    """Convert VolatilityAnalysisResult to dict."""
    data = {
        "has_volatility_clustering": result.has_volatility_clustering,
        "persistence": result.persistence,
    }

    if result.arch_lm_result is not None:
        data["arch_lm"] = {
            "test_statistic": result.arch_lm_result.test_statistic,
            "p_value": result.arch_lm_result.p_value,
            "lags": result.arch_lm_result.lags,
        }

    if result.garch_result is not None:
        data["garch"] = {
            "omega": result.garch_result.omega,
            "alpha": result.garch_result.alpha,
            "beta": result.garch_result.beta,
            "persistence": result.garch_result.persistence,
            "half_life": result.garch_result.half_life,
        }

    return data


def _distribution_to_dict(result) -> dict[str, Any]:
    """Convert DistributionAnalysisResult to dict."""
    data = {
        "is_normal": result.is_normal,
        "recommended_distribution": result.recommended_distribution,
        "recommended_df": result.recommended_df,
    }

    if result.moments_result is not None:
        data["moments"] = {
            "mean": result.moments_result.mean,
            "std": result.moments_result.std,
            "skewness": result.moments_result.skewness,
            "skewness_significant": result.moments_result.skewness_significant,
            "excess_kurtosis": result.moments_result.excess_kurtosis,
            "excess_kurtosis_significant": result.moments_result.excess_kurtosis_significant,
        }

    if result.jarque_bera_result is not None:
        data["jarque_bera"] = {
            "statistic": result.jarque_bera_result.statistic,
            "p_value": result.jarque_bera_result.p_value,
            "is_normal": result.jarque_bera_result.is_normal,
        }

    if result.shapiro_wilk_result is not None:
        data["shapiro_wilk"] = {
            "statistic": result.shapiro_wilk_result.statistic,
            "p_value": result.shapiro_wilk_result.p_value,
            "is_normal": result.shapiro_wilk_result.is_normal,
        }

    if (
        result.tail_analysis_result is not None
        and result.tail_analysis_result.hill_result is not None
    ):
        hill = result.tail_analysis_result.hill_result
        data["tail_analysis"] = {
            "classification": hill.classification,
            "tail_index": hill.tail_index,
            "has_heavy_tails": hill.classification in ["heavy", "very_heavy"],
            "best_fit": result.tail_analysis_result.best_fit,
        }

    return data


def _generate_default_html(
    result: FeatureDiagnosticsResult,
    include_plots: bool,
    title: str | None,
) -> str:
    """Generate HTML report using default template."""
    html_parts = []

    # Header
    report_title = title or f"Feature Diagnostics: {result.feature_name}"
    html_parts.append(_html_header(report_title))

    # Summary section
    html_parts.append("<h2>Summary</h2>")
    html_parts.append(f"<p><strong>Feature:</strong> {result.feature_name}</p>")
    html_parts.append(f"<p><strong>Observations:</strong> {result.n_obs:,}</p>")
    html_parts.append(f"<p><strong>Health Score:</strong> {result.health_score:.2f}/1.00</p>")

    if result.flags:
        html_parts.append("<h3>⚠️ Flags</h3>")
        html_parts.append("<ul>")
        for flag in result.flags:
            html_parts.append(f"<li>{flag}</li>")
        html_parts.append("</ul>")

    # Summary table
    if not result.summary_df.empty:
        html_parts.append("<h3>Test Summary</h3>")
        html_parts.append(result.summary_df.to_html(index=False, classes="dataframe"))

    # Feature section
    html_parts.append(_generate_feature_section_html(result, include_plots))

    # Recommendations
    if result.recommendations:
        html_parts.append("<h2>Recommendations</h2>")
        html_parts.append("<ol>")
        for rec in result.recommendations:
            html_parts.append(f"<li>{rec}</li>")
        html_parts.append("</ol>")

    # Footer
    html_parts.append(_html_footer())

    return "\n".join(html_parts)


def _generate_feature_section_html(
    result: FeatureDiagnosticsResult,
    include_plots: bool,
) -> str:
    """Generate HTML section for a single feature's diagnostics."""
    parts = []

    # Stationarity
    if result.stationarity is not None:
        parts.append("<h3>Stationarity Analysis</h3>")
        parts.append(f"<p><strong>Consensus:</strong> {result.stationarity.consensus}</p>")

        if result.stationarity.adf_result is not None:
            adf = result.stationarity.adf_result
            parts.append(
                f"<p><strong>ADF:</strong> statistic={adf.test_statistic:.4f}, p-value={adf.p_value:.4f}</p>"
            )

        if result.stationarity.kpss_result is not None:
            kpss = result.stationarity.kpss_result
            parts.append(
                f"<p><strong>KPSS:</strong> statistic={kpss.test_statistic:.4f}, p-value={kpss.p_value:.4f}</p>"
            )

        if result.stationarity.pp_result is not None:
            pp = result.stationarity.pp_result
            parts.append(
                f"<p><strong>PP:</strong> statistic={pp.test_statistic:.4f}, p-value={pp.p_value:.4f}</p>"
            )

    # Autocorrelation
    if result.autocorrelation is not None:
        parts.append("<h3>Autocorrelation Analysis</h3>")
        n_sig_acf = len(result.autocorrelation.significant_acf_lags)
        n_sig_pacf = len(result.autocorrelation.significant_pacf_lags)
        parts.append(f"<p><strong>Significant ACF lags:</strong> {n_sig_acf}</p>")
        parts.append(f"<p><strong>Significant PACF lags:</strong> {n_sig_pacf}</p>")
        parts.append(
            f"<p><strong>Suggested ARIMA order:</strong> {result.autocorrelation.suggested_arima_order}</p>"
        )
        parts.append(
            f"<p><strong>White noise:</strong> {'Yes' if result.autocorrelation.is_white_noise else 'No'}</p>"
        )

    # Volatility
    if result.volatility is not None:
        parts.append("<h3>Volatility Analysis</h3>")
        has_clustering = "Yes" if result.volatility.has_volatility_clustering else "No"
        parts.append(f"<p><strong>Volatility clustering:</strong> {has_clustering}</p>")

        if result.volatility.arch_lm_result is not None:
            arch = result.volatility.arch_lm_result
            parts.append(
                f"<p><strong>ARCH-LM:</strong> statistic={arch.test_statistic:.4f}, p-value={arch.p_value:.4f}</p>"
            )

    # Distribution
    if result.distribution is not None:
        parts.append("<h3>Distribution Analysis</h3>")
        parts.append(
            f"<p><strong>Recommended distribution:</strong> {result.distribution.recommended_distribution}</p>"
        )
        parts.append(
            f"<p><strong>Is normal:</strong> {'Yes' if result.distribution.is_normal else 'No'}</p>"
        )

        if result.distribution.moments_result is not None:
            mom = result.distribution.moments_result
            parts.append(f"<p><strong>Mean:</strong> {mom.mean:.6f}</p>")
            parts.append(f"<p><strong>Std Dev:</strong> {mom.std:.6f}</p>")
            parts.append(
                f"<p><strong>Skewness:</strong> {mom.skewness:.4f} "
                f"({'significant' if mom.skewness_significant else 'not significant'})</p>"
            )
            parts.append(
                f"<p><strong>Excess Kurtosis:</strong> {mom.excess_kurtosis:.4f} "
                f"({'significant' if mom.excess_kurtosis_significant else 'not significant'})</p>"
            )

        if result.distribution.jarque_bera_result is not None:
            jb = result.distribution.jarque_bera_result
            parts.append(
                f"<p><strong>Jarque-Bera:</strong> statistic={jb.statistic:.4f}, p-value={jb.p_value:.4f}</p>"
            )

    # Plots (if requested)
    if include_plots:
        parts.append("<h3>Visualizations</h3>")
        parts.append("<p><em>Interactive Plotly charts would be embedded here.</em></p>")
        parts.append(
            "<p><em>Implementation note: Requires creating plots from result data</em></p>"
        )
        # NOTE: Plot embedding planned for future version using diagnostic_plots module

    return "\n".join(parts)


def _html_header(title: str) -> str:
    """Generate HTML header with styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .dataframe {{
            border-collapse: collapse;
            margin: 20px 0;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .dataframe th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .dataframe td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .dataframe tr:hover {{
            background-color: #f1f1f1;
        }}
        ul, ol {{
            line-height: 1.8;
        }}
        p {{
            line-height: 1.6;
        }}
        hr {{
            border: none;
            border-top: 2px solid #95a5a6;
            margin: 40px 0;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""


def _html_footer() -> str:
    """Generate HTML footer."""
    return f"""
    <div class="footer">
        <p>Generated by ML4T Diagnostic {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Interactive diagnostic reports for quantitative trading features</p>
    </div>
</body>
</html>
"""


def _render_custom_template(
    result: FeatureDiagnosticsResult,
    template: str,
    _include_plots: bool,
) -> str:
    """Render custom template with result data.

    Args:
        result: Diagnostic result
        template: Template string with {placeholders}
        include_plots: Whether to include plots

    Returns:
        Rendered HTML
    """
    # Prepare template variables
    template_vars = {
        "feature_name": result.feature_name,
        "n_obs": result.n_obs,
        "health_score": f"{result.health_score:.2f}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add test results
    if result.stationarity is not None:
        template_vars["stationarity_consensus"] = result.stationarity.consensus

    if result.summary_df is not None and not result.summary_df.empty:
        template_vars["summary_table"] = result.summary_df.to_html(index=False, classes="dataframe")

    # Render template
    return template.format(**template_vars)
