"""Dashboard generation for ml4t-diagnostic evaluation results.

This module provides HTML dashboard generation with interactive Plotly
visualizations for comprehensive evaluation reports.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from jinja2 import Template

from .themes import apply_theme
from .visualization import (
    plot_feature_distributions,
    plot_ic_heatmap,
    plot_quantile_returns,
    plot_turnover_decay,
)

if TYPE_CHECKING:
    from .framework import EvaluationResult


# HTML template for dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ml4t-diagnostic Evaluation Report - {{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary-color: #3366CC;
            --success-color: #00CC88;
            --danger-color: #FF4444;
            --bg-color: #F8F9FA;
            --text-color: #333333;
            --border-color: #E0E0E0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        h1 {
            color: var(--primary-color);
            margin-bottom: 10px;
        }

        .metadata {
            color: #666;
            font-size: 14px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }

        .metric-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
        }

        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .metric-value.positive {
            color: var(--success-color);
        }

        .metric-value.negative {
            color: var(--danger-color);
        }

        .metric-detail {
            font-size: 12px;
            color: #999;
        }

        .tabs {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }

        .tab-nav {
            display: flex;
            border-bottom: 2px solid var(--border-color);
        }

        .tab-button {
            padding: 15px 30px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 16px;
            color: #666;
            transition: all 0.3s;
            position: relative;
        }

        .tab-button:hover {
            color: var(--primary-color);
            background-color: rgba(51, 102, 204, 0.05);
        }

        .tab-button.active {
            color: var(--primary-color);
            font-weight: bold;
        }

        .tab-button.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background-color: var(--primary-color);
        }

        .tab-content {
            display: none;
            padding: 30px;
        }

        .tab-content.active {
            display: block;
        }

        .plot-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .plot-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: var(--text-color);
        }

        .section {
            margin-bottom: 40px;
        }

        .insights {
            background-color: #FFF3CD;
            border: 1px solid #FFE5A0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .insights h4 {
            color: #856404;
            margin-bottom: 10px;
        }

        .insights ul {
            margin-left: 20px;
            color: #856404;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .tab-button {
                padding: 10px 15px;
                font-size: 14px;
            }

            .summary-grid {
                grid-template-columns: 1fr;
            }
        }

        @media print {
            .tabs {
                display: none;
            }

            .tab-content {
                display: block !important;
                page-break-inside: avoid;
            }

            .plot-container {
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <div class="metadata">
                <p><strong>Tier {{ tier }} Evaluation</strong> |
                   Generated: {{ timestamp }} |
                   Splitter: {{ splitter }}</p>
            </div>
        </header>

        <!-- Summary Metrics -->
        <div class="summary-grid">
            {% for metric_name, metric_data in metrics.items() %}
            <div class="metric-card">
                <h3>{{ metric_name|upper|replace('_', ' ') }}</h3>
                <div class="metric-value {% if metric_data.value > 0 %}positive{% else %}negative{% endif %}">
                    {{ "%.4f"|format(metric_data.value) }}
                </div>
                {% if metric_data.std %}
                <div class="metric-detail">± {{ "%.4f"|format(metric_data.std) }}</div>
                {% endif %}
                {% if metric_data.significant is defined %}
                <div class="metric-detail">
                    {% if metric_data.significant %}✓ Significant{% else %}Not Significant{% endif %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <!-- Insights Section -->
        {% if insights %}
        <div class="insights">
            <h4>Key Insights</h4>
            <ul>
                {% for insight in insights %}
                <li>{{ insight }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <!-- Tabbed Content -->
        <div class="tabs">
            <div class="tab-nav">
                <button class="tab-button active" onclick="showTab(event, 'performance')">Performance</button>
                <button class="tab-button" onclick="showTab(event, 'statistics')">Statistical Tests</button>
                <button class="tab-button" onclick="showTab(event, 'stability')">Stability Analysis</button>
                <button class="tab-button" onclick="showTab(event, 'distributions')">Distributions</button>
                {% if custom_tabs %}
                    {% for tab_name in custom_tabs %}
                    <button class="tab-button" onclick="showTab(event, '{{ tab_name }}')">{{ tab_name|title }}</button>
                    {% endfor %}
                {% endif %}
            </div>

            <!-- Performance Tab -->
            <div id="performance" class="tab-content active">
                <div class="section">
                    {% for plot_id, plot_title in performance_plots.items() %}
                    <div class="plot-container">
                        <div class="plot-title">{{ plot_title }}</div>
                        <div id="{{ plot_id }}"></div>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <!-- Statistical Tests Tab -->
            <div id="statistics" class="tab-content">
                <div class="section">
                    {% if statistical_tests %}
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead>
                            <tr style="border-bottom: 2px solid #ddd;">
                                <th style="padding: 10px; text-align: left;">Test</th>
                                <th style="padding: 10px; text-align: right;">Statistic</th>
                                <th style="padding: 10px; text-align: right;">P-Value</th>
                                <th style="padding: 10px; text-align: center;">Significant</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for test_name, test_data in statistical_tests.items() %}
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px;">{{ test_name|upper|replace('_', ' ') }}</td>
                                <td style="padding: 10px; text-align: right;">{{ "%.4f"|format(test_data.statistic) }}</td>
                                <td style="padding: 10px; text-align: right;">{{ "%.4f"|format(test_data.p_value) }}</td>
                                <td style="padding: 10px; text-align: center;">
                                    {% if test_data.p_value < 0.05 %}
                                    <span style="color: var(--success-color);">✓</span>
                                    {% else %}
                                    <span style="color: var(--danger-color);">✗</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% else %}
                    <p>No statistical tests were performed for this evaluation.</p>
                    {% endif %}
                </div>
            </div>

            <!-- Stability Analysis Tab -->
            <div id="stability" class="tab-content">
                <div class="section">
                    {% for plot_id, plot_title in stability_plots.items() %}
                    <div class="plot-container">
                        <div class="plot-title">{{ plot_title }}</div>
                        <div id="{{ plot_id }}"></div>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <!-- Distributions Tab -->
            <div id="distributions" class="tab-content">
                <div class="section">
                    {% for plot_id, plot_title in distribution_plots.items() %}
                    <div class="plot-container">
                        <div class="plot-title">{{ plot_title }}</div>
                        <div id="{{ plot_id }}"></div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>

    <script>
        // Tab functionality
        function showTab(evt, tabName) {
            var i, tabcontent, tabbuttons;

            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }

            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].classList.remove("active");
            }

            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }

        // Plot data
        {{ plot_data }}
    </script>
</body>
</html>
"""


class DashboardBuilder:
    """Build comprehensive HTML evaluation reports."""

    def __init__(self, result: "EvaluationResult", theme: str = "default"):
        """Initialize dashboard builder.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to visualize
        theme : str, default "default"
            Theme to apply: "default", "dark", "print"
        """
        self.result = result
        self.theme = theme
        self.figures: dict[str, Any] = {}
        self.plot_data: dict[str, Any] = {}

    def add_performance_plots(
        self,
        predictions: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
    ) -> None:
        """Add performance-related plots.

        Parameters
        ----------
        predictions : pd.DataFrame, optional
            Model predictions for IC analysis
        returns : pd.DataFrame, optional
            Returns data for analysis
        """
        # IC Heatmap
        if predictions is not None and returns is not None:
            fig = plot_ic_heatmap(
                predictions,
                returns,
                title="Information Coefficient Term Structure",
            )
            fig = apply_theme(fig, self.theme)
            self.figures["ic_heatmap"] = fig
            self.plot_data["ic_heatmap"] = fig.to_json()

        # Quantile Returns
        if predictions is not None and returns is not None:
            # Use first column if DataFrame
            pred_series = (
                predictions.iloc[:, 0] if isinstance(predictions, pd.DataFrame) else predictions
            )
            ret_series = returns.iloc[:, 0] if isinstance(returns, pd.DataFrame) else returns

            fig = plot_quantile_returns(
                pred_series,
                ret_series,
                title="Returns by Prediction Quantile",
            )
            fig = apply_theme(fig, self.theme)
            self.figures["quantile_returns"] = fig
            self.plot_data["quantile_returns"] = fig.to_json()

    def add_stability_plots(self, factor_values: pd.DataFrame | None = None) -> None:
        """Add stability analysis plots.

        Parameters
        ----------
        factor_values : pd.DataFrame, optional
            Time series of factor values
        """
        if factor_values is not None:
            fig = plot_turnover_decay(
                factor_values,
                title="Factor Turnover and Decay Analysis",
            )
            fig = apply_theme(fig, self.theme)
            self.figures["turnover_decay"] = fig
            self.plot_data["turnover_decay"] = fig.to_json()

    def add_distribution_plots(self, features: pd.DataFrame | None = None) -> None:
        """Add feature distribution plots.

        Parameters
        ----------
        features : pd.DataFrame, optional
            Feature values over time
        """
        if features is not None:
            fig = plot_feature_distributions(
                features,
                title="Feature Distribution Analysis",
            )
            fig = apply_theme(fig, self.theme)
            self.figures["feature_distributions"] = fig
            self.plot_data["feature_distributions"] = fig.to_json()

    def _generate_insights(self) -> list[str]:
        """Generate key insights from evaluation results."""
        insights = []

        # Check IC significance
        if "ic" in self.result.metrics_results:
            ic_mean = self.result.metrics_results["ic"].get("mean", 0)
            if abs(ic_mean) > 0.03:
                insights.append(
                    f"Information Coefficient of {ic_mean:.3f} indicates "
                    f"{'positive' if ic_mean > 0 else 'negative'} predictive relationship",
                )

        # Check Sharpe ratio
        if "sharpe" in self.result.metrics_results:
            sharpe = self.result.metrics_results["sharpe"].get("mean", 0)
            if sharpe > 1:
                insights.append(
                    f"Sharpe ratio of {sharpe:.2f} suggests strong risk-adjusted returns",
                )
            elif sharpe < 0:
                insights.append(
                    f"Negative Sharpe ratio ({sharpe:.2f}) indicates poor performance",
                )

        # Check statistical significance
        if self.result.statistical_tests:
            significant_tests = [
                name
                for name, test in self.result.statistical_tests.items()
                if isinstance(test, dict) and test.get("p_value", 1) < 0.05
            ]
            if significant_tests:
                insights.append(
                    f"Statistically significant results for: {', '.join(significant_tests)}",
                )

        # Check tier-specific insights
        if self.result.tier == 1:
            insights.append(
                "Tier 1 evaluation provides rigorous multiple testing corrections",
            )
        elif self.result.tier == 2:
            insights.append(
                "Tier 2 evaluation includes HAC-adjusted significance tests",
            )
        else:
            insights.append("Tier 3 evaluation provides fast screening metrics")

        return insights

    def generate_html(
        self,
        filename: str,
        title: str | None = None,
        _include_data: bool = True,
    ) -> None:
        """Generate complete HTML report.

        Parameters
        ----------
        filename : str
            Output filename for HTML report
        title : str, optional
            Report title. If None, auto-generated
        include_data : bool, default True
            Whether to include raw data in report
        """
        # Prepare template data
        template_data = {
            "title": title or f"ml4t-diagnostic Evaluation Report - Tier {self.result.tier}",
            "tier": self.result.tier,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "splitter": self.result.splitter_name,
            "metrics": {},
            "statistical_tests": {},
            "insights": self._generate_insights(),
            "performance_plots": {},
            "stability_plots": {},
            "distribution_plots": {},
            "custom_tabs": [],
            "plot_data": "",
        }

        # Process metrics - cast nested dicts for type safety
        metrics_dict = cast(dict[str, Any], template_data["metrics"])
        for metric_name, metric_data in self.result.metrics_results.items():
            if isinstance(metric_data, dict):
                metrics_dict[metric_name] = {
                    "value": metric_data.get("mean", 0),
                    "std": metric_data.get("std"),
                    "significant": metric_data.get("significant"),
                }
            else:
                metrics_dict[metric_name] = {
                    "value": metric_data,
                    "std": None,
                    "significant": None,
                }

        # Process statistical tests
        stats_dict = cast(dict[str, Any], template_data["statistical_tests"])
        for test_name, test_data in self.result.statistical_tests.items():
            if isinstance(test_data, dict) and "error" not in test_data:
                stats_dict[test_name] = {
                    "statistic": test_data.get(
                        "test_statistic",
                        test_data.get("dsr", test_data.get("ic", np.nan)),
                    ),
                    "p_value": test_data.get("p_value", np.nan),
                }

        # Prepare plot references
        perf_plots_dict = cast(dict[str, Any], template_data["performance_plots"])
        for plot_id in ["ic_heatmap", "quantile_returns"]:
            if plot_id in self.plot_data:
                perf_plots_dict[plot_id] = plot_id.replace("_", " ").title()

        stability_plots_dict = cast(dict[str, Any], template_data["stability_plots"])
        for plot_id in ["turnover_decay"]:
            if plot_id in self.plot_data:
                stability_plots_dict[plot_id] = plot_id.replace("_", " ").title()

        dist_plots_dict = cast(dict[str, Any], template_data["distribution_plots"])
        for plot_id in ["feature_distributions"]:
            if plot_id in self.plot_data:
                dist_plots_dict[plot_id] = plot_id.replace("_", " ").title()

        # Generate JavaScript for plots
        plot_js = []
        for plot_id, plot_json in self.plot_data.items():
            plot_js.append(
                f"Plotly.newPlot('{plot_id}', {plot_json}.data, {plot_json}.layout);",
            )
        template_data["plot_data"] = "\n".join(plot_js)

        # Render template
        template = Template(DASHBOARD_TEMPLATE)
        html_content = template.render(**template_data)

        # Save to file
        output_path = Path(filename)
        output_path.write_text(html_content)

    def export_plots(self, output_dir: str, format: str = "png") -> None:
        """Export individual plots as static images.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        format : str, default "png"
            Export format: "png", "svg", "pdf"
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            import kaleido  # noqa: F401 (availability check)
        except ImportError:
            raise ImportError(  # noqa: B904
                "kaleido is required for static export. Install with: pip install kaleido",
            )

        for name, fig in self.figures.items():
            filename = output_path / f"{name}.{format}"
            fig.write_image(str(filename))


def create_evaluation_dashboard(
    result: "EvaluationResult",
    output_file: str,
    predictions: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    theme: str = "default",
    title: str | None = None,
) -> None:
    """Convenience function to create a complete evaluation dashboard.

    Parameters
    ----------
    result : EvaluationResult
        Evaluation results to visualize
    output_file : str
        Output HTML filename
    predictions : pd.DataFrame, optional
        Model predictions for visualizations
    returns : pd.DataFrame, optional
        Returns data for visualizations
    features : pd.DataFrame, optional
        Feature data for distribution analysis
    theme : str, default "default"
        Dashboard theme
    title : str, optional
        Dashboard title

    Examples:
    --------
    >>> create_evaluation_dashboard(
    ...     result,
    ...     "evaluation_report.html",
    ...     predictions=pred_df,
    ...     returns=returns_df
    ... )
    """
    builder = DashboardBuilder(result, theme)

    # Add visualizations based on available data
    if predictions is not None or returns is not None:
        builder.add_performance_plots(predictions, returns)

    if features is not None:
        builder.add_stability_plots(features)
        builder.add_distribution_plots(features)

    # Generate HTML
    builder.generate_html(output_file, title)
