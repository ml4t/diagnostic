"""HTML report generation for feature evaluation results.

This module provides functions for generating comprehensive HTML reports that combine
multiple Plotly visualizations with narrative text, analysis summaries, and styling.

All report functions follow the standard API defined in docs/plot_api_standards.md:
- Accept evaluation results from analyze_*() functions
- Generate self-contained HTML files with embedded plots
- Support theme customization and styling
- Provide flexible report templates

Example workflow:
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance, compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import generate_importance_report
    >>>
    >>> # Run evaluations
    >>> importance = analyze_ml_importance(model, X, y)
    >>> interactions = compute_shap_interactions(model, X)
    >>>
    >>> # Generate comprehensive HTML report
    >>> report_path = generate_importance_report(
    ...     importance_results=importance,
    ...     interaction_results=interactions,
    ...     output_file="feature_analysis.html",
    ...     title="Feature Analysis Report",
    ...     theme="dark"
    ... )
    >>> print(f"Report saved to: {report_path}")
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from ml4t.diagnostic.visualization.core import get_theme_config, validate_theme
from ml4t.diagnostic.visualization.feature_plots import (
    plot_importance_bar,
    plot_importance_distribution,
    plot_importance_heatmap,
    plot_importance_summary,
)
from ml4t.diagnostic.visualization.interaction_plots import (
    plot_interaction_bar,
    plot_interaction_heatmap,
    plot_interaction_network,
)

__all__ = [
    "generate_importance_report",
    "generate_interaction_report",
    "generate_combined_report",
    "combine_figures_to_html",
    "export_figures_to_pdf",
]


def combine_figures_to_html(
    figures: list[go.Figure],
    *,
    title: str = "Analysis Report",
    sections: list[dict[str, Any]] | None = None,
    output_file: str | Path | None = None,
    theme: str | None = None,
    include_toc: bool = True,
) -> str:
    """Combine multiple Plotly figures into a single HTML document.

    This is the core function for generating HTML reports. It takes a list of
    Plotly figures and optional narrative sections, and produces a self-contained
    HTML file with embedded visualizations.

    Parameters
    ----------
    figures : list[go.Figure]
        List of Plotly figure objects to include in the report.
        Figures are rendered in the order provided.
    title : str, optional
        Report title displayed at the top. Default is "Analysis Report".
    sections : list[dict[str, Any]] | None, optional
        List of section dictionaries defining report structure. Each section can contain:
        - "title": str - Section heading
        - "text": str - Narrative text (supports HTML and markdown-style formatting)
        - "figure_index": int - Index of figure to include (from figures list)
        If None, figures are rendered sequentially without additional text.
    output_file : str | Path | None, optional
        Path where HTML file should be saved. If None, returns HTML string without saving.
    theme : str | None, optional
        Theme name ("default", "dark", "print", "presentation").
        Affects overall page styling. If None, uses "default".
    include_toc : bool, optional
        Whether to include a table of contents at the top of the report.
        Default is True. TOC is generated from section titles.

    Returns
    -------
    str
        If output_file is None: HTML content as string
        If output_file is provided: Path to saved HTML file

    Raises
    ------
    ValueError
        If figures list is empty
        If section refers to invalid figure_index
    TypeError
        If figures contains non-Figure objects

    Examples
    --------
    Generate report with multiple plots:

    >>> from ml4t.diagnostic.visualization import (
    ...     plot_importance_bar,
    ...     plot_importance_heatmap,
    ...     combine_figures_to_html
    ... )
    >>>
    >>> # Create figures
    >>> fig1 = plot_importance_bar(results, top_n=15)
    >>> fig2 = plot_importance_heatmap(results)
    >>>
    >>> # Define sections with narrative
    >>> sections = [
    ...     {
    ...         "title": "Feature Importance Rankings",
    ...         "text": "Top 15 features ranked by consensus importance across methods.",
    ...         "figure_index": 0
    ...     },
    ...     {
    ...         "title": "Method Agreement Analysis",
    ...         "text": "Correlation matrix showing agreement between importance methods.",
    ...         "figure_index": 1
    ...     }
    ... ]
    >>>
    >>> # Generate HTML report
    >>> html_path = combine_figures_to_html(
    ...     figures=[fig1, fig2],
    ...     title="Feature Importance Analysis",
    ...     sections=sections,
    ...     output_file="report.html",
    ...     theme="dark"
    ... )

    Generate simple report without sections:

    >>> figs = [plot_importance_bar(results), plot_importance_heatmap(results)]
    >>> html = combine_figures_to_html(figs, title="Quick Report")
    >>> print(html[:100])  # Preview HTML

    Notes
    -----
    - HTML is self-contained with embedded Plotly.js from CDN
    - First figure includes full Plotly.js, subsequent figures reuse it
    - CSS styling is embedded in <style> tag
    - Reports are responsive and work on mobile devices
    - File size depends on number of data points in figures
    """
    # Validation
    if not figures:
        raise ValueError("At least one figure is required")

    if not all(isinstance(fig, go.Figure) for fig in figures):
        raise TypeError("All items in figures list must be plotly.graph_objects.Figure instances")

    if sections is not None:
        for i, section in enumerate(sections):
            if "figure_index" in section:
                idx = section["figure_index"]
                if idx < 0 or idx >= len(figures):
                    raise ValueError(
                        f"Section {i} has invalid figure_index {idx}. Must be between 0 and {len(figures) - 1}"
                    )

    # Validate theme
    theme = theme or "default"
    validate_theme(theme)
    theme_config = get_theme_config(theme)

    # Convert figures to HTML divs
    figure_htmls = []
    for i, fig in enumerate(figures):
        # First figure includes Plotly.js from CDN, others don't
        include_plotlyjs = "cdn" if i == 0 else False

        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=include_plotlyjs, div_id=f"plot-{i}"
        )
        figure_htmls.append(fig_html)

    # Build HTML content
    html_content = _build_html_document(
        title=title,
        figure_htmls=figure_htmls,
        sections=sections,
        theme_config=theme_config,
        include_toc=include_toc,
    )

    # Save or return
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        return str(output_path.absolute())
    else:
        return html_content


def export_figures_to_pdf(
    figures: list[go.Figure],
    output_file: str | Path,
    *,
    layout: str = "vertical",
    page_size: tuple[int, int] = (800, 600),
    scale: float = 2.0,
) -> str:
    """Export multiple Plotly figures to a single PDF file.

    Each figure is rendered as a separate page in the PDF. Uses kaleido for
    high-quality vector rendering.

    Parameters
    ----------
    figures : list[go.Figure]
        List of Plotly figure objects to export.
    output_file : str | Path
        Path where PDF file should be saved.
    layout : str, optional
        Layout mode for figures:
        - "vertical": Each figure on its own page (default)
        - "compact": Attempt to fit multiple small figures per page
        Default is "vertical".
    page_size : tuple[int, int], optional
        Page size in pixels (width, height).
        Default is (800, 600) which approximates A4 landscape at 96 DPI.
        Common sizes:
        - (800, 600): A4 landscape-like
        - (600, 800): A4 portrait-like
        - (1200, 900): Larger landscape
    scale : float, optional
        Resolution scale factor for rendering. Higher values produce
        better quality but larger files. Default is 2.0.

    Returns
    -------
    str
        Absolute path to generated PDF file.

    Raises
    ------
    ValueError
        If figures list is empty
    ImportError
        If kaleido is not installed
    TypeError
        If figures contains non-Figure objects

    Examples
    --------
    Export multiple plots to PDF:

    >>> from ml4t.diagnostic.visualization import plot_importance_bar, export_figures_to_pdf
    >>>
    >>> fig1 = plot_importance_bar(results, top_n=15)
    >>> fig2 = plot_importance_heatmap(results)
    >>>
    >>> pdf_path = export_figures_to_pdf(
    ...     figures=[fig1, fig2],
    ...     output_file="analysis.pdf",
    ...     page_size=(800, 600),
    ...     scale=2.0
    ... )

    Export with custom page size:

    >>> pdf_path = export_figures_to_pdf(
    ...     figures=[fig1, fig2, fig3],
    ...     output_file="report.pdf",
    ...     page_size=(1200, 900),  # Larger pages
    ...     scale=3.0  # High resolution
    ... )

    Notes
    -----
    - Requires kaleido package: `pip install kaleido`
    - Each figure is exported as a vector PDF page
    - File size depends on plot complexity and scale factor
    - For print quality, use scale >= 2.0
    - For web sharing, use scale = 1.0 to reduce file size
    """
    # Validation
    if not figures:
        raise ValueError("At least one figure is required")

    if not all(isinstance(fig, go.Figure) for fig in figures):
        raise TypeError("All items in figures list must be plotly.graph_objects.Figure instances")

    # Check kaleido availability
    try:
        import kaleido  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "kaleido is required for PDF export. Install it with: pip install kaleido"
        ) from e

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export strategy depends on layout
    if layout == "vertical":
        # Each figure gets its own page
        return _export_figures_multipage(figures, output_path, page_size, scale)
    elif layout == "compact":
        # Try to fit multiple figures per page (not implemented yet)
        raise NotImplementedError("Compact layout is not yet implemented. Use 'vertical'.")
    else:
        raise ValueError(f"Invalid layout '{layout}'. Must be 'vertical' or 'compact'.")


def generate_importance_report(
    importance_results: dict[str, Any],
    *,
    output_file: str | Path,
    title: str | None = None,
    theme: str | None = None,
    include_sections: list[str] | None = None,
    top_n: int = 20,
    export_pdf: bool = False,
    pdf_page_size: tuple[int, int] = (800, 600),
    pdf_scale: float = 2.0,
) -> str:
    """Generate comprehensive HTML report for feature importance analysis.

    Creates a multi-section report combining:
    - Executive summary with key findings
    - Consensus importance rankings (bar chart)
    - Method agreement analysis (heatmap)
    - Importance score distributions
    - Interpretation and recommendations

    Parameters
    ----------
    importance_results : dict[str, Any]
        Results from analyze_ml_importance() containing:
        - "consensus_ranking": Features ranked by consensus
        - "method_results": Individual method results
        - "method_agreement": Cross-method correlations
        - "top_features_consensus": Features in all top-10s
    output_file : str | Path
        Path where HTML report will be saved.
    title : str | None, optional
        Report title. If None, uses "Feature Importance Analysis Report".
    theme : str | None, optional
        Visual theme ("default", "dark", "print", "presentation").
        If None, uses "default".
    include_sections : list[str] | None, optional
        Which sections to include in report. Options:
        - "summary": Executive summary
        - "rankings": Consensus rankings bar chart
        - "agreement": Method agreement heatmap
        - "distributions": Score distributions
        - "recommendations": Interpretation and next steps
        If None, includes all sections.
    top_n : int, optional
        Number of top features to display in charts. Default is 20.
    export_pdf : bool, optional
        If True, also export the report figures to PDF format.
        Default is False (HTML only).
    pdf_page_size : tuple[int, int], optional
        Page size for PDF export (width, height) in pixels.
        Default is (800, 600). Only used if export_pdf=True.
    pdf_scale : float, optional
        Resolution scale for PDF export. Higher = better quality.
        Default is 2.0. Only used if export_pdf=True.

    Returns
    -------
    str
        Absolute path to generated HTML file.

    Examples
    --------
    Generate full report:

    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import generate_importance_report
    >>>
    >>> results = analyze_ml_importance(model, X, y, methods=["mdi", "pfi", "shap"])
    >>> report_path = generate_importance_report(
    ...     importance_results=results,
    ...     output_file="importance_report.html",
    ...     theme="dark"
    ... )

    Generate minimal report with specific sections:

    >>> report_path = generate_importance_report(
    ...     importance_results=results,
    ...     output_file="quick_report.html",
    ...     include_sections=["summary", "rankings"],
    ...     top_n=10
    ... )
    """
    # Default title
    if title is None:
        title = "Feature Importance Analysis Report"

    # Default sections
    if include_sections is None:
        include_sections = ["summary", "rankings", "agreement", "distributions", "recommendations"]

    # Validate sections
    valid_sections = {"summary", "rankings", "agreement", "distributions", "recommendations"}
    invalid = set(include_sections) - valid_sections
    if invalid:
        raise ValueError(f"Invalid sections: {invalid}. Valid options: {valid_sections}")

    # Generate figures
    figures: list[go.Figure] = []
    sections: list[dict[str, str | int]] = []

    # Add summary section
    if "summary" in include_sections:
        summary_text = _generate_importance_summary_text(importance_results)
        sections.append({"title": "Executive Summary", "text": summary_text})

    # Add consensus rankings
    if "rankings" in include_sections:
        fig_bar = plot_importance_bar(importance_results, top_n=top_n, theme=theme)
        figures.append(fig_bar)
        sections.append(
            {
                "title": "Consensus Feature Rankings",
                "text": (
                    f"The top {top_n} features ranked by consensus across all importance methods. "
                    "Features appearing at the top are consistently identified as important by "
                    "multiple methodologies (MDI, PFI, SHAP)."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Add method agreement
    if "agreement" in include_sections:
        fig_heatmap = plot_importance_heatmap(importance_results, theme=theme)
        figures.append(fig_heatmap)
        sections.append(
            {
                "title": "Method Agreement Analysis",
                "text": (
                    "Spearman correlation matrix showing agreement between different importance "
                    "methods. High correlation (>0.7) indicates methods agree on feature rankings. "
                    "Low correlation (<0.5) suggests method-specific biases or feature interactions."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Add distributions
    if "distributions" in include_sections:
        fig_dist = plot_importance_distribution(importance_results, theme=theme)
        figures.append(fig_dist)
        sections.append(
            {
                "title": "Importance Score Distributions",
                "text": (
                    "Distribution of importance scores from each method. Overlapping distributions "
                    "indicate consensus, while separation suggests method disagreement."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Add recommendations
    if "recommendations" in include_sections:
        rec_text = _generate_importance_recommendations(importance_results)
        sections.append({"title": "Interpretation & Recommendations", "text": rec_text})

    # Generate HTML
    html_path = combine_figures_to_html(
        figures=figures,
        title=title,
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_toc=True,
    )

    # Optionally export to PDF
    if export_pdf and figures:
        pdf_path = Path(output_file).with_suffix(".pdf")
        export_figures_to_pdf(
            figures=figures,
            output_file=pdf_path,
            page_size=pdf_page_size,
            scale=pdf_scale,
        )

    return html_path


def generate_interaction_report(
    interaction_results: dict[str, Any],
    *,
    output_file: str | Path,
    title: str | None = None,
    theme: str | None = None,
    include_sections: list[str] | None = None,
    top_n: int = 20,
    export_pdf: bool = False,
    pdf_page_size: tuple[int, int] = (800, 600),
    pdf_scale: float = 2.0,
) -> str:
    """Generate comprehensive HTML report for feature interaction analysis.

    Creates a multi-section report combining:
    - Top feature pair interactions (bar chart)
    - Full interaction matrix (heatmap)
    - Interaction network graph
    - Interpretation and recommendations

    Parameters
    ----------
    interaction_results : dict[str, Any]
        Results from compute_shap_interactions() or analyze_interactions().
    output_file : str | Path
        Path where HTML report will be saved.
    title : str | None, optional
        Report title. If None, uses "Feature Interaction Analysis Report".
    theme : str | None, optional
        Visual theme. If None, uses "default".
    include_sections : list[str] | None, optional
        Which sections to include. Options:
        - "top_pairs": Top N strongest interactions (bar)
        - "matrix": Full interaction matrix (heatmap)
        - "network": Interactive network graph
        - "recommendations": Interpretation
        If None, includes all sections.
    top_n : int, optional
        Number of top interactions to display. Default is 20.
    export_pdf : bool, optional
        If True, also export the report figures to PDF format.
        Default is False (HTML only).
    pdf_page_size : tuple[int, int], optional
        Page size for PDF export (width, height) in pixels.
        Default is (800, 600). Only used if export_pdf=True.
    pdf_scale : float, optional
        Resolution scale for PDF export. Higher = better quality.
        Default is 2.0. Only used if export_pdf=True.

    Returns
    -------
    str
        Absolute path to generated HTML file.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import generate_interaction_report
    >>>
    >>> interactions = compute_shap_interactions(model, X)
    >>> report_path = generate_interaction_report(
    ...     interaction_results=interactions,
    ...     output_file="interactions.html"
    ... )
    """
    # Default title
    if title is None:
        title = "Feature Interaction Analysis Report"

    # Default sections
    if include_sections is None:
        include_sections = ["top_pairs", "matrix", "network", "recommendations"]

    # Generate figures
    figures: list[go.Figure] = []
    sections: list[dict[str, str | int]] = []

    # Top pairs
    if "top_pairs" in include_sections:
        fig_bar = plot_interaction_bar(interaction_results, top_n=top_n, theme=theme)
        figures.append(fig_bar)
        sections.append(
            {
                "title": f"Top {top_n} Feature Interactions",
                "text": (
                    "Strongest pairwise feature interactions ranked by mean absolute interaction strength. "
                    "High interaction values indicate non-linear or conditional relationships."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Matrix
    if "matrix" in include_sections:
        fig_heatmap = plot_interaction_heatmap(interaction_results, theme=theme)
        figures.append(fig_heatmap)
        sections.append(
            {
                "title": "Interaction Strength Matrix",
                "text": (
                    "Symmetric matrix showing pairwise interaction strengths. "
                    "Darker colors indicate stronger interactions."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Network
    if "network" in include_sections:
        fig_network = plot_interaction_network(interaction_results, theme=theme, top_n=top_n)
        figures.append(fig_network)
        sections.append(
            {
                "title": "Interaction Network Graph",
                "text": (
                    "Network visualization of feature interactions. Node size represents "
                    "feature importance, edge thickness represents interaction strength. "
                    "Isolated nodes have weak interactions."
                ),
                "figure_index": len(figures) - 1,
            }
        )

    # Recommendations
    if "recommendations" in include_sections:
        rec_text = _generate_interaction_recommendations(interaction_results)
        sections.append({"title": "Interpretation & Recommendations", "text": rec_text})

    # Generate HTML
    html_path = combine_figures_to_html(
        figures=figures,
        title=title,
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_toc=True,
    )

    # Optionally export to PDF
    if export_pdf and figures:
        pdf_path = Path(output_file).with_suffix(".pdf")
        export_figures_to_pdf(
            figures=figures,
            output_file=pdf_path,
            page_size=pdf_page_size,
            scale=pdf_scale,
        )

    return html_path


def generate_combined_report(
    importance_results: dict[str, Any],
    interaction_results: dict[str, Any] | None = None,
    *,
    output_file: str | Path,
    title: str | None = None,
    theme: str | None = None,
    top_n: int = 20,
    export_pdf: bool = False,
    pdf_page_size: tuple[int, int] = (800, 600),
    pdf_scale: float = 2.0,
) -> str:
    """Generate comprehensive report combining importance and interaction analysis.

    Creates a unified report with all feature analysis visualizations and interpretations.

    Parameters
    ----------
    importance_results : dict[str, Any]
        Results from analyze_ml_importance().
    interaction_results : dict[str, Any] | None, optional
        Results from compute_shap_interactions(). If None, only importance analysis included.
    output_file : str | Path
        Path where HTML report will be saved.
    title : str | None, optional
        Report title. If None, uses "Complete Feature Analysis Report".
    theme : str | None, optional
        Visual theme. If None, uses "default".
    top_n : int, optional
        Number of top features/interactions to display. Default is 20.
    export_pdf : bool, optional
        If True, also export the report figures to PDF format.
        Default is False (HTML only).
    pdf_page_size : tuple[int, int], optional
        Page size for PDF export (width, height) in pixels.
        Default is (800, 600). Only used if export_pdf=True.
    pdf_scale : float, optional
        Resolution scale for PDF export. Higher = better quality.
        Default is 2.0. Only used if export_pdf=True.

    Returns
    -------
    str
        Absolute path to generated HTML file.

    Examples
    --------
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance, compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import generate_combined_report
    >>>
    >>> importance = analyze_ml_importance(model, X, y)
    >>> interactions = compute_shap_interactions(model, X)
    >>>
    >>> report_path = generate_combined_report(
    ...     importance_results=importance,
    ...     interaction_results=interactions,
    ...     output_file="complete_analysis.html",
    ...     theme="presentation"
    ... )
    """
    # Default title
    if title is None:
        title = "Complete Feature Analysis Report"

    # Generate figures
    figures: list[go.Figure] = []
    sections: list[dict[str, str | int]] = []

    # Overview section
    overview_text = _generate_combined_overview(importance_results, interaction_results)
    sections.append({"title": "Analysis Overview", "text": overview_text})

    # Importance section
    sections.append({"title": "Part 1: Feature Importance Analysis", "text": ""})

    # Summary plot (4-panel importance summary)
    fig_importance_summary = plot_importance_summary(importance_results, top_n=15, theme=theme)
    figures.append(fig_importance_summary)
    sections.append(
        {
            "title": "Importance Summary (Multi-Panel View)",
            "text": (
                "Comprehensive view of feature importance combining consensus rankings, "
                "method agreement, and score distributions in a single multi-panel visualization."
            ),
            "figure_index": len(figures) - 1,
        }
    )

    # Interaction section (if provided)
    if interaction_results is not None:
        sections.append({"title": "Part 2: Feature Interaction Analysis", "text": ""})

        # Network visualization
        fig_network = plot_interaction_network(interaction_results, theme=theme, top_n=top_n)
        figures.append(fig_network)
        sections.append(
            {
                "title": "Interaction Network",
                "text": (
                    "Interactive network showing how features interact. Strong interactions "
                    "may indicate opportunities for feature engineering."
                ),
                "figure_index": len(figures) - 1,
            }
        )

        # Interaction heatmap
        fig_int_heatmap = plot_interaction_heatmap(interaction_results, theme=theme)
        figures.append(fig_int_heatmap)
        sections.append(
            {
                "title": "Interaction Matrix",
                "text": "Complete pairwise interaction strength matrix.",
                "figure_index": len(figures) - 1,
            }
        )

    # Recommendations
    rec_text = _generate_combined_recommendations(importance_results, interaction_results)
    sections.append({"title": "Actionable Recommendations", "text": rec_text})

    # Generate HTML
    html_path = combine_figures_to_html(
        figures=figures,
        title=title,
        sections=sections,
        output_file=output_file,
        theme=theme,
        include_toc=True,
    )

    # Optionally export to PDF
    if export_pdf and figures:
        pdf_path = Path(output_file).with_suffix(".pdf")
        export_figures_to_pdf(
            figures=figures,
            output_file=pdf_path,
            page_size=pdf_page_size,
            scale=pdf_scale,
        )

    return html_path


# ============================================================================
# Private Helper Functions
# ============================================================================


def _build_html_document(
    title: str,
    figure_htmls: list[str],
    sections: list[dict[str, Any]] | None,
    theme_config: dict[str, Any],
    include_toc: bool,
) -> str:
    """Build complete HTML document from components.

    Parameters
    ----------
    title : str
        Document title
    figure_htmls : list[str]
        List of figure HTML div strings
    sections : list[dict] | None
        Section definitions with title, text, figure_index
    theme_config : dict
        Theme configuration from get_theme_config()
    include_toc : bool
        Whether to include table of contents

    Returns
    -------
    str
        Complete HTML document
    """
    # Extract colors from theme
    bg_color = theme_config.get("plot_bgcolor", "#FFFFFF")
    text_color = theme_config.get("font_color", "#1F1F1F")
    grid_color = theme_config.get("gridcolor", "#E5E5E5")

    # Determine if dark theme
    is_dark = "dark" in theme_config.get("template", "").lower() or bg_color in [
        "#1E1E1E",
        "#0E0E0E",
    ]

    # Generate CSS
    css = _generate_css(bg_color, text_color, grid_color, is_dark)

    # Generate TOC if requested
    toc_html = ""
    if include_toc and sections:
        toc_html = _generate_toc(sections)

    # Generate body content
    body_html = _generate_body_content(figure_htmls, sections)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Assemble complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="ML4T Diagnostic Visualization Library">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

{toc_html}

        <main>
{body_html}
        </main>

        <footer>
            <p>Generated by <a href="https://github.com/yourusername/ml4t-diagnostic" target="_blank">ML4T Diagnostic</a> - Quantitative Evaluation Library</p>
        </footer>
    </div>
</body>
</html>"""

    return html


def _generate_css(bg_color: str, text_color: str, grid_color: str, is_dark: bool) -> str:
    """Generate CSS styles for report."""
    # Derive additional colors
    if is_dark:
        header_bg = "#2A2A2A"
        section_bg = "#252525"
        border_color = "#404040"
        link_color = "#6FA8DC"
    else:
        header_bg = "#F5F5F5"
        section_bg = "#FAFAFA"
        border_color = grid_color
        link_color = "#1A73E8"

    css = f"""
        /* Reset and base styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            text-align: center;
            padding: 40px 20px;
            background-color: {header_bg};
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        header h1 {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .timestamp {{
            color: {text_color};
            opacity: 0.7;
            font-size: 0.9em;
        }}

        /* Table of Contents */
        .toc {{
            background-color: {section_bg};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid {border_color};
        }}

        .toc h2 {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}

        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}

        .toc li {{
            margin: 8px 0;
        }}

        .toc a {{
            color: {link_color};
            text-decoration: none;
            transition: opacity 0.2s;
        }}

        .toc a:hover {{
            opacity: 0.7;
        }}

        /* Sections */
        .section {{
            margin-bottom: 50px;
        }}

        .section-title {{
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid {border_color};
        }}

        .section-text {{
            font-size: 1.1em;
            margin-bottom: 20px;
            line-height: 1.8;
        }}

        /* Plot containers */
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: {section_bg};
            border-radius: 8px;
            border: 1px solid {border_color};
        }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 30px 20px;
            margin-top: 50px;
            border-top: 1px solid {border_color};
            opacity: 0.7;
        }}

        footer a {{
            color: {link_color};
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            header h1 {{
                font-size: 2em;
            }}

            .section-title {{
                font-size: 1.5em;
            }}

            .plot-container {{
                padding: 10px;
            }}
        }}

        /* Print styles */
        @media print {{
            body {{
                background-color: white;
                color: black;
            }}

            .container {{
                max-width: none;
            }}

            .plot-container {{
                page-break-inside: avoid;
            }}
        }}
    """

    return css


def _generate_toc(sections: list[dict[str, Any]]) -> str:
    """Generate table of contents HTML."""
    toc_items = []

    for i, section in enumerate(sections):
        section_title = section.get("title", "")
        if section_title:
            # Create anchor-safe ID
            section_id = f"section-{i}"
            toc_items.append(f'            <li><a href="#{section_id}">{section_title}</a></li>')

    toc_html = f"""        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
{chr(10).join(toc_items)}
            </ul>
        </nav>
"""

    return toc_html


def _generate_body_content(figure_htmls: list[str], sections: list[dict[str, Any]] | None) -> str:
    """Generate main body content HTML."""
    if sections is None:
        # Simple case: just render all figures sequentially
        body_parts = []
        for _i, fig_html in enumerate(figure_htmls):
            body_parts.append(f"""            <div class="plot-container">
{fig_html}
            </div>
""")
        return "\n".join(body_parts)

    # Complex case: render sections with associated figures
    body_parts = []

    for i, section in enumerate(sections):
        section_id = f"section-{i}"
        section_title = section.get("title", "")
        section_text = section.get("text", "")
        figure_index = section.get("figure_index")

        # Start section
        section_html = f'            <section class="section" id="{section_id}">\n'

        # Add title if present
        if section_title:
            section_html += f'                <h2 class="section-title">{section_title}</h2>\n'

        # Add text if present (section_text may contain HTML block elements,
        # so we don't wrap in <p> to avoid invalid nesting)
        if section_text:
            section_html += '                <div class="section-text">\n'
            section_html += f"                    {section_text}\n"
            section_html += "                </div>\n"

        # Add figure if specified
        if figure_index is not None and 0 <= figure_index < len(figure_htmls):
            section_html += '                <div class="plot-container">\n'
            section_html += figure_htmls[figure_index]
            section_html += "\n                </div>\n"

        # Close section
        section_html += "            </section>\n"

        body_parts.append(section_html)

    return "\n".join(body_parts)


def _generate_importance_summary_text(results: dict[str, Any]) -> str:
    """Generate executive summary text for importance analysis."""
    consensus_ranking = results.get("consensus_ranking", [])
    top_consensus = results.get("top_features_consensus", [])
    method_agreement = results.get("method_agreement", {})

    # Calculate average agreement
    avg_agreement = (
        sum(method_agreement.values()) / len(method_agreement) if method_agreement else 0.0
    )

    summary = f"""
    <p><strong>Key Findings:</strong></p>
    <ul>
        <li>Analyzed {len(consensus_ranking)} features across multiple importance methods</li>
        <li>Top consensus feature: <strong>{consensus_ranking[0] if consensus_ranking else "N/A"}</strong></li>
        <li>Features with strong consensus: {len(top_consensus)} features appear in all methods' top-10</li>
        <li>Average method agreement: {avg_agreement:.2f} (Spearman correlation)</li>
    </ul>
    """

    return summary.strip()


def _generate_importance_recommendations(_results: dict[str, Any]) -> str:
    """Generate recommendations text for importance analysis."""
    recommendations = """
    <p><strong>Interpretation Guidelines:</strong></p>
    <ul>
        <li><strong>High consensus + high agreement</strong>: Trust the rankings - features are robustly important</li>
        <li><strong>Method disagreement</strong>: Investigate feature-specific biases (MDI vs PFI patterns)</li>
        <li><strong>SHAP divergence</strong>: Indicates interaction effects - consider feature engineering</li>
    </ul>

    <p><strong>Next Steps:</strong></p>
    <ul>
        <li>Focus on top consensus features for model interpretability</li>
        <li>Investigate features with large method disagreement</li>
        <li>Consider removing features with low importance across all methods</li>
        <li>Analyze SHAP interaction effects for top features</li>
    </ul>
    """

    return recommendations.strip()


def _generate_interaction_recommendations(_results: dict[str, Any]) -> str:
    """Generate recommendations text for interaction analysis."""
    recommendations = """
    <p><strong>Interpreting Interactions:</strong></p>
    <ul>
        <li><strong>Strong interactions</strong>: Non-linear or conditional relationships between features</li>
        <li><strong>Network clusters</strong>: Groups of related features that interact strongly</li>
        <li><strong>Isolated features</strong>: Features with weak interactions (may be independent)</li>
    </ul>

    <p><strong>Feature Engineering Opportunities:</strong></p>
    <ul>
        <li>Create explicit interaction terms for top pairs (e.g., feature_A * feature_B)</li>
        <li>Consider non-linear transformations for interacting features</li>
        <li>Investigate domain-specific meanings of top interactions</li>
    </ul>
    """

    return recommendations.strip()


def _generate_combined_overview(
    importance_results: dict[str, Any], interaction_results: dict[str, Any] | None
) -> str:
    """Generate overview text for combined report."""
    n_features = len(importance_results.get("consensus_ranking", []))

    overview = f"""
    <p>This comprehensive report analyzes feature importance and interactions for a machine learning model
    with {n_features} features. The analysis combines multiple methodologies to provide robust insights.</p>

    <p><strong>Report Contents:</strong></p>
    <ul>
        <li><strong>Part 1: Feature Importance</strong> - Which features the model relies on most</li>
    """

    if interaction_results is not None:
        overview += """        <li><strong>Part 2: Feature Interactions</strong> - How features combine and interact</li>
    """

    overview += """    </ul>
    """

    return overview.strip()


def _generate_combined_recommendations(
    _importance_results: dict[str, Any], interaction_results: dict[str, Any] | None
) -> str:
    """Generate combined recommendations."""
    recommendations = """
    <p><strong>Prioritized Action Items:</strong></p>
    <ol>
        <li><strong>Focus on consensus features</strong>: Top features identified by multiple methods are most reliable</li>
        <li><strong>Investigate method disagreements</strong>: Understand why different methods rank features differently</li>
    """

    if interaction_results is not None:
        recommendations += """        <li><strong>Engineer interaction terms</strong>: Create explicit features for strong interactions</li>
        <li><strong>Analyze interaction clusters</strong>: Groups of interacting features may represent domain concepts</li>
    """

    recommendations += """    </ol>

    <p><strong>Model Improvement Strategies:</strong></p>
    <ul>
        <li>Remove low-importance features to reduce overfitting risk</li>
        <li>Add domain knowledge to interpret top features and interactions</li>
        <li>Consider model architecture changes if interactions are prevalent</li>
        <li>Validate findings on out-of-sample data</li>
    </ul>
    """

    return recommendations.strip()


def _export_figures_multipage(
    figures: list[go.Figure],
    output_path: Path,
    page_size: tuple[int, int],
    scale: float,
) -> str:
    """Export multiple figures to a single multi-page PDF.

    Uses kaleido to export each figure to PDF, then combines them using pypdf.

    Parameters
    ----------
    figures : list[go.Figure]
        Figures to export
    output_path : Path
        Output PDF file path
    page_size : tuple[int, int]
        Page dimensions (width, height) in pixels
    scale : float
        Rendering scale factor

    Returns
    -------
    str
        Path to created PDF file
    """
    import tempfile
    from pathlib import Path as TempPath

    # Try to import pypdf for merging
    pdf_writer_class: type
    try:
        from pypdf import PdfWriter as _PypdfWriter

        pdf_writer_class = _PypdfWriter
    except ImportError:
        # Fallback to PyPDF2 if pypdf not available
        try:
            from PyPDF2 import (
                PdfWriter as _Pypdf2Writer,  # type: ignore[import-not-found,unused-ignore]
            )

            pdf_writer_class = _Pypdf2Writer
        except ImportError as e:
            raise ImportError(
                "pypdf or PyPDF2 is required for PDF merging. Install it with: pip install pypdf"
            ) from e

    width, height = page_size

    # Create temporary directory for individual PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdfs = []

        # Export each figure to its own PDF
        for i, fig in enumerate(figures):
            temp_pdf = TempPath(temp_dir) / f"page_{i}.pdf"

            # Update figure layout for PDF export
            fig_copy = go.Figure(fig)  # Make a copy to avoid modifying original
            fig_copy.update_layout(
                width=width,
                height=height,
                margin={"l": 50, "r": 50, "t": 80, "b": 50},  # Add margins for print
            )

            # Export to PDF using kaleido
            fig_copy.write_image(
                str(temp_pdf),
                format="pdf",
                width=width,
                height=height,
                scale=scale,
            )

            temp_pdfs.append(temp_pdf)

        # Merge all PDFs into single file
        writer = pdf_writer_class()
        for pdf_path in temp_pdfs:
            writer.append(str(pdf_path))

        # Write merged PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

    return str(output_path.absolute())
