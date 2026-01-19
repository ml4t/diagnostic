"""Feature importance dashboard for comprehensive analysis.

This module provides the FeatureImportanceDashboard class for creating
rich, interactive dashboards exploring feature importance across multiple methods.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_extraction import (
    ImportanceVizData,
    extract_importance_viz_data,
    extract_interaction_viz_data,
)
from .base import BaseDashboard, DashboardSection, get_theme


class FeatureImportanceDashboard(BaseDashboard):
    """Interactive dashboard for comprehensive feature importance analysis.

    Provides rich exploration of feature importance with 4-tab architecture:
    - **Overview Tab**: Consensus ranking, method agreement heatmap, key insights
    - **Method Comparison Tab**: Aligned feature rankings across methods, stability analysis
    - **Feature Details Tab**: Searchable/filterable table with all features and metrics
    - **Interactions Tab**: Network visualization or top pairs table (adaptive based on feature count)

    All visualizations are interactive (zoom, pan, hover for details).
    Real-time search filtering available in Feature Details tab.

    Examples
    --------
    **Basic Usage: Single Method**

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance
    >>> from ml4t.diagnostic.visualization import FeatureImportanceDashboard
    >>>
    >>> # Train model
    >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Analyze importance (single method)
    >>> results = analyze_ml_importance(
    ...     model, X_train, y_train,
    ...     methods=['mdi'],
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Create and save dashboard
    >>> dashboard = FeatureImportanceDashboard(title="MDI Analysis")
    >>> dashboard.save("mdi_dashboard.html", results)

    **Multiple Methods with Permutation Repeats**

    >>> # Run multiple methods for comparison
    >>> results = analyze_ml_importance(
    ...     model, X_train, y_train,
    ...     methods=['mdi', 'pfi'],
    ...     pfi_n_repeats=10,  # Get uncertainty estimates
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Dark theme with custom top N
    >>> dashboard = FeatureImportanceDashboard(
    ...     title="Feature Importance: MDI vs PFI",
    ...     theme="dark",
    ...     n_top_features=15
    ... )
    >>> dashboard.save("multi_method_dashboard.html", results)

    **With Feature Interactions (SHAP)**

    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>>
    >>> # Compute importance
    >>> importance_results = analyze_ml_importance(
    ...     model, X_train, y_train,
    ...     methods=['mdi', 'pfi', 'shap'],
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Compute interactions (adds 3 more visualizations)
    >>> interaction_results = compute_shap_interactions(
    ...     model, X_train,
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Create dashboard with both
    >>> dashboard = FeatureImportanceDashboard(
    ...     title="Full Feature Analysis with Interactions"
    ... )
    >>> html = dashboard.generate(
    ...     analysis_results=importance_results,
    ...     interaction_results=interaction_results
    ... )
    >>> with open("full_dashboard.html", "w") as f:
    ...     f.write(html)

    Notes
    -----
    - Dashboard requires results from `analyze_ml_importance()`
    - Interaction visualizations only appear if `interaction_results` provided
    - PFI distribution plots only shown if `pfi_n_repeats > 1`
    - All Plotly visualizations are interactive (zoom, pan, hover)

    See Also
    --------
    analyze_ml_importance : Compute feature importance across methods
    compute_shap_interactions : Compute pairwise feature interactions
    FeatureInteractionDashboard : Standalone interaction analysis
    """

    # Class constants for visualization thresholds
    INTERACTION_NETWORK_THRESHOLD = 20  # Show network if ≤20 features, else table
    INTERACTION_TOP_EDGES = 20  # Number of top interaction pairs to display
    INTERACTION_MATRIX_SIZE = 15  # Max features for interaction matrix heatmap

    def __init__(
        self,
        title: str = "Feature Importance Analysis",
        theme: Literal["light", "dark"] = "light",
        width: int | None = None,
        height: int | None = None,
        n_top_features: int = 10,
    ):
        """Initialize Feature Importance Dashboard.

        Parameters
        ----------
        title : str, default="Feature Importance Analysis"
            Dashboard title
        theme : {'light', 'dark'}, default='light'
            Visual theme
        width : int, optional
            Dashboard width in pixels
        height : int, optional
            Dashboard height in pixels
        n_top_features : int, default=10
            Number of top features to highlight in visualizations
        """
        super().__init__(title, theme, width, height)
        self.n_top_features = n_top_features

    def generate(
        self,
        analysis_results: dict[str, Any],
        interaction_results: dict[str, Any] | None = None,
        **_kwargs,
    ) -> str:
        """Generate complete dashboard HTML.

        Parameters
        ----------
        analysis_results : dict
            Results from analyze_ml_importance()
        interaction_results : dict, optional
            Results from compute_shap_interactions() to include interaction analysis
        **kwargs
            Additional parameters (currently unused)

        Returns
        -------
        str
            Complete HTML document
        """
        # Extract structured data
        viz_data = extract_importance_viz_data(analysis_results)

        # Create tabbed layout with improved organization
        self._create_tabbed_layout(viz_data, interaction_results)

        # Compose HTML
        return self._compose_html()

    def _create_tabbed_layout(
        self, viz_data: ImportanceVizData, interaction_results: dict[str, Any] | None = None
    ) -> None:
        """Create tabbed dashboard layout with improved organization."""
        # Create tabs
        tabs = [
            ("overview", "Overview"),
            ("methods", "Method Comparison"),
            ("features", "Feature Details"),
        ]

        # Add interactions tab if we have interaction results
        if interaction_results is not None:
            tabs.append(("interactions", "Interactions"))

        # Build tab content
        tab_contents = {
            "overview": self._create_overview_tab(viz_data),
            "methods": self._create_method_comparison_tab(viz_data),
            "features": self._create_feature_details_tab(viz_data),
        }

        if interaction_results is not None:
            tab_contents["interactions"] = self._create_interactions_tab(
                viz_data, interaction_results
            )

        # Build tab navigation buttons
        tab_buttons = "".join(
            [
                f'<button class="tab-button{" active" if i == 0 else ""}" '
                f"onclick=\"switchTab(event, '{tab_id}')\">{tab_name}</button>"
                for i, (tab_id, tab_name) in enumerate(tabs)
            ]
        )

        # Build tab content divs
        tab_divs = "".join(
            [
                f'<div id="{tab_id}" class="tab-content{" active" if i == 0 else ""}">{tab_contents[tab_id]}</div>'
                for i, (tab_id, _) in enumerate(tabs)
            ]
        )

        # Compose complete tabbed layout
        html_content = f"""
        <div class="tab-navigation">
            {tab_buttons}
        </div>
        {tab_divs}
        """

        # Create single section with tabbed content
        section = DashboardSection(
            title="Feature Importance Analysis", description="", content=html_content
        )

        self.sections.append(section)

    def _create_overview_tab(self, viz_data: ImportanceVizData) -> str:
        """Create Overview tab with executive summary."""
        summary = viz_data["summary"]
        per_feature = viz_data["per_feature"]
        per_method = viz_data["per_method"]
        method_comparison = viz_data["method_comparison"]
        llm_context = viz_data["llm_context"]

        methods = list(per_method.keys())
        method_names_display = ", ".join([m.upper() for m in methods])

        # Metric cards
        html = ["<h2>Overview</h2>"]
        html.append(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Features Analyzed</div>
                <div class="metric-value">{summary["n_features"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Methods Used</div>
                <div class="metric-value">{summary["n_methods"]}</div>
                <div class="metric-sublabel">({method_names_display})</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Top Feature</div>
                <div class="metric-value">{summary["top_feature"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label" title="Average rank correlation between methods (1.0 = perfect agreement, 0.0 = no agreement)">
                    Method Agreement
                    <span style="font-size: 0.8em; color: #666; cursor: help;">ⓘ</span>
                </div>
                <div class="metric-value">{summary["avg_method_agreement"]:.2f}</div>
                <div class="metric-sublabel">
                    {"High" if summary["avg_method_agreement"] > 0.7 else "Medium" if summary["avg_method_agreement"] > 0.4 else "Low"} Agreement
                </div>
            </div>
        </div>
        """)

        # Consensus ranking chart
        consensus_ranking = summary["consensus_ranking"][: self.n_top_features]
        consensus_scores = [
            per_feature[feat]["consensus_score"] * 100 for feat in consensus_ranking
        ]

        fig_consensus = go.Figure()
        fig_consensus.add_trace(
            go.Bar(
                y=consensus_ranking,
                x=consensus_scores,
                orientation="h",
                marker={"color": consensus_scores, "colorscale": "Blues", "showscale": False},
                text=[f"{score:.2f}%" for score in consensus_scores],
                textposition="auto",
            )
        )
        fig_consensus.update_layout(
            title=f"Top {self.n_top_features} Features (Consensus Ranking)",
            xaxis_title="Consensus Importance (%)",
            yaxis_title="Feature",
            yaxis={"autorange": "reversed"},
            template=get_theme(self.theme)["template"],
            height=max(400, len(consensus_ranking) * 40),
            margin={"l": 150, "r": 50, "t": 80, "b": 80},
        )
        html.append(fig_consensus.to_html(include_plotlyjs=False, div_id="plot-consensus"))

        # Insights panel
        html.append("""
        <div class="insights-panel">
            <h3>Key Insights</h3>
            <ul>
        """)
        for insight in llm_context["key_insights"]:
            html.append(f"<li>{insight}</li>")
        html.append("</ul></div>")

        # Method agreement heatmap (if multiple methods)
        if len(methods) > 1:
            corr_matrix = method_comparison["correlation_matrix"]
            corr_methods = method_comparison["correlation_methods"]

            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix,
                    x=[m.upper() for m in corr_methods],
                    y=[m.upper() for m in corr_methods],
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=[[f"{val:.2f}" for val in row] for row in corr_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    colorbar={"title": "Correlation"},
                )
            )
            fig_corr.update_layout(
                title="Method Agreement (Rank Correlation)",
                template=get_theme(self.theme)["template"],
                height=400,
                margin={"l": 100, "r": 100, "t": 100, "b": 80},
            )
            html.append(fig_corr.to_html(include_plotlyjs=False, div_id="plot-method-corr"))

        return "".join(html)

    def _create_method_comparison_tab(self, viz_data: ImportanceVizData) -> str:
        """Create Method Comparison tab with ALIGNED features across methods."""
        summary = viz_data["summary"]
        per_method = viz_data["per_method"]
        uncertainty = viz_data["uncertainty"]

        methods = list(per_method.keys())
        n_methods = len(methods)

        html = ["<h2>Method Comparison</h2>"]

        # Explanation section
        html.append("""
        <div class="section-description">
            <p><strong>What you're seeing:</strong> The same top features (consensus-ranked) shown across all methods. This reveals where methods agree and disagree.</p>
            <p><strong>Look for:</strong></p>
            <ul>
                <li>Similar bar heights across methods → strong agreement on importance</li>
                <li>Very different heights → methods disagree (investigate further)</li>
                <li>Large error bars → unstable importance estimates</li>
            </ul>
        </div>
        """)

        if n_methods > 1:
            # Get consensus top features (SAME features shown across all methods)
            consensus_features = summary["consensus_ranking"][: self.n_top_features]

            # Side-by-side method comparison with ALIGNED features
            fig_methods = make_subplots(
                rows=1,
                cols=n_methods,
                subplot_titles=[m.upper() for m in methods],
                shared_yaxes=True,
            )

            for col_idx, method_name in enumerate(methods, start=1):
                method_data = per_method[method_name]

                # Get importance values for consensus features (in consensus order)
                importances = [
                    method_data["importances"].get(feat, 0) * 100 for feat in consensus_features
                ]

                # Get error bars if available
                error_y = None
                if method_data["std"] is not None:
                    error_std = [
                        method_data["std"].get(feat, 0) * 100 for feat in consensus_features
                    ]
                    error_y = {"type": "data", "array": error_std, "visible": True}

                fig_methods.add_trace(
                    go.Bar(
                        y=consensus_features,  # SAME features for all methods
                        x=importances,
                        orientation="h",
                        error_x=error_y,
                        text=[f"{imp:.2f}%" for imp in importances],
                        textposition="auto",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )
                fig_methods.update_xaxes(title_text="Importance (%)", row=1, col=col_idx)

            fig_methods.update_yaxes(title_text="Feature", autorange="reversed", row=1, col=1)
            fig_methods.update_layout(
                title=f"Top {self.n_top_features} Features: How Each Method Ranks Them",
                template=get_theme(self.theme)["template"],
                height=max(500, self.n_top_features * 40),
                margin={"l": 150, "r": 50, "t": 100, "b": 80},
            )
            html.append(fig_methods.to_html(include_plotlyjs=False, div_id="plot-methods"))

        # Add stability analysis if available
        has_uncertainty = bool(uncertainty.get("coefficient_of_variation"))
        if has_uncertainty:
            html.append('<h3 style="margin-top: 30px;">Stability Analysis</h3>')

            consensus_ranking = summary["consensus_ranking"][: self.n_top_features]

            # CV plot
            cv_data = uncertainty.get("coefficient_of_variation", {}).get("pfi", {})
            if cv_data:
                cv_values = [cv_data.get(feat, 0) for feat in consensus_ranking]
                fig_cv = go.Figure()
                fig_cv.add_trace(
                    go.Bar(
                        y=consensus_ranking,
                        x=cv_values,
                        orientation="h",
                        marker={
                            "color": cv_values,
                            "colorscale": "Reds",
                            "showscale": False,
                            "reversescale": True,
                        },
                        text=[f"{cv:.2f}" for cv in cv_values],
                        textposition="auto",
                    )
                )
                fig_cv.update_layout(
                    title="Feature Stability (Coefficient of Variation - Lower is Better)",
                    xaxis_title="Coefficient of Variation",
                    yaxis_title="Feature",
                    yaxis={"autorange": "reversed"},
                    template=get_theme(self.theme)["template"],
                    height=max(400, len(consensus_ranking) * 40),
                    margin={"l": 150, "r": 50, "t": 80, "b": 80},
                )
                html.append(fig_cv.to_html(include_plotlyjs=False, div_id="plot-cv"))

        return "".join(html)

    def _create_feature_details_tab(self, viz_data: ImportanceVizData) -> str:
        """Create Feature Details tab with searchable, filterable table."""
        summary = viz_data["summary"]
        per_feature = viz_data["per_feature"]
        per_method = viz_data["per_method"]

        methods = list(per_method.keys())

        html = ["<h2>Feature Details</h2>"]

        # Search box
        html.append("""
        <div style="margin: 20px 0;">
            <input type="text" id="feature-search" placeholder="Type to filter features..."
                   style="width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px;">
        </div>
        """)

        # Build detailed feature table
        table_rows = []
        for rank, feature_name in enumerate(summary["consensus_ranking"], start=1):
            feat_data = per_feature[feature_name]

            # Determine if this row should be highlighted for low agreement
            agreement_class = " low-agreement" if feat_data["agreement_level"] == "low" else ""

            # Build row
            row_cells = [
                f'<td style="font-weight: 600;">{rank}</td>',
                f'<td style="font-weight: 500;">{feature_name}</td>',
                f"<td>{feat_data['consensus_score'] * 100:.2f}%</td>",
            ]

            # Add separate column for each method's rank
            for m in methods:
                method_rank = feat_data["method_ranks"].get(m, None)
                if method_rank is not None:
                    row_cells.append(f'<td style="text-align: center;">{method_rank}</td>')
                else:
                    row_cells.append('<td style="text-align: center; opacity: 0.3;">-</td>')

            # Add agreement and stability
            row_cells.extend(
                [
                    f'<td><span class="badge badge-{feat_data["agreement_level"]}">{feat_data["agreement_level"]}</span></td>',
                    f"<td>{feat_data['stability_score']:.3f}</td>",
                ]
            )

            table_rows.append(f'<tr class="feature-row{agreement_class}">{"".join(row_cells)}</tr>')

        # Build table header
        method_header_cols = "".join(
            [
                f'<th title="{m.upper()} rank (lower is better)" style="text-align: center;">{m.upper()}<br/>Rank</th>'
                for m in methods
            ]
        )

        html.append(f"""
        <div style="overflow-x: auto;">
            <table class="feature-table" id="feature-importance-table">
                <thead>
                    <tr>
                        <th title="Consensus rank across all methods">Consensus<br/>Rank</th>
                        <th>Feature</th>
                        <th title="Average importance across all methods (normalized to %)">Consensus<br/>Score (%)</th>
                        {method_header_cols}
                        <th title="How well methods agree on this feature's importance (high/medium/low)">Agreement</th>
                        <th title="Consistency of rank across resampling (1.0 = perfectly stable)">Stability</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        <p style="font-size: 0.85em; opacity: 0.7; margin-top: 10px;">
            💡 <strong>Tip:</strong> Click column headers to sort the table. Use the search box above to filter features.
            Rows highlighted in orange indicate low method agreement.
        </p>

        <script>
        // Real-time search filtering
        document.getElementById('feature-search').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const table = document.getElementById('feature-importance-table');

            // Guard against missing table
            if (!table) return;

            const tbody = table.getElementsByTagName('tbody')[0];
            if (!tbody) return;

            const rows = tbody.getElementsByTagName('tr');

            for (let row of rows) {{
                // Guard against malformed rows
                if (!row.cells || row.cells.length < 2) continue;

                const featureName = row.cells[1].textContent.toLowerCase();
                row.style.display = featureName.includes(searchTerm) ? '' : 'none';
            }}
        }});
        </script>
        """)

        return "".join(html)

    def _create_interactions_tab(
        self, viz_data: ImportanceVizData, interaction_results: dict[str, Any]
    ) -> str:
        """Create Interactions tab with adaptive display based on feature count."""
        summary = viz_data["summary"]
        n_features = summary["n_features"]

        html = ["<h2>Feature Interactions</h2>"]
        html.append("""
        <p style="margin: 20px 0; font-style: italic; opacity: 0.8;">
        SHAP interaction values show how feature contributions change based on other features.
        Strong interactions suggest non-linear relationships and feature dependencies.
        </p>
        """)

        # Extract interaction viz data
        interaction_viz_data = extract_interaction_viz_data(
            interaction_results,
            importance_results={"consensus_ranking": summary["consensus_ranking"]},
        )

        inter_summary = interaction_viz_data["summary"]
        interaction_viz_data["per_feature"]
        network_data = interaction_viz_data["network_graph"]

        # Top interaction info
        strongest_pair = inter_summary["strongest_pair"]
        strongest_value = inter_summary["strongest_interaction"]

        html.append(f"""
        <div class="insights-panel">
            <h3>Top Interaction</h3>
            <p><strong>{strongest_pair[0]}</strong> ↔ <strong>{strongest_pair[1]}</strong>: {strongest_value:.4f}</p>
            <p style="margin-top: 10px;">Most interactive feature: <strong>{inter_summary["most_interactive_feature"]}</strong>
            (total strength: {inter_summary["max_total_interaction"]:.4f})</p>
        </div>
        """)

        # Adaptive display based on feature count
        edges = network_data["edges"][: self.INTERACTION_TOP_EDGES]

        if n_features > self.INTERACTION_NETWORK_THRESHOLD:
            # Too many features - show top pairs table instead of network
            html.append(f"""
            <div class="section-description">
                <p><strong>Note:</strong> With {n_features} features, a network visualization would be unreadable.
                Showing top 10 interaction pairs instead.</p>
            </div>
            """)

            # Top interaction pairs table
            html.append("""
            <h3 style="margin-top: 30px;">Top 10 Interaction Pairs</h3>
            <div style="overflow-x: auto;">
                <table class="feature-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature A</th>
                            <th>Feature B</th>
                            <th>Interaction Strength</th>
                        </tr>
                    </thead>
                    <tbody>
            """)

            for i, edge in enumerate(edges[:10], start=1):
                html.append(f"""
                    <tr>
                        <td>{i}</td>
                        <td><strong>{edge["source"]}</strong></td>
                        <td><strong>{edge["target"]}</strong></td>
                        <td>{edge["abs_weight"]:.4f}</td>
                    </tr>
                """)

            html.append("</tbody></table></div>")

            # Bar chart of top pairs
            pair_labels = [f"{edge['source']} ↔ {edge['target']}" for edge in edges[:10]]
            pair_strengths = [edge["abs_weight"] for edge in edges[:10]]

            fig_pairs = go.Figure()
            fig_pairs.add_trace(
                go.Bar(
                    y=pair_labels,
                    x=pair_strengths,
                    orientation="h",
                    marker={"color": pair_strengths, "colorscale": "Viridis", "showscale": False},
                    text=[f"{s:.4f}" for s in pair_strengths],
                    textposition="auto",
                )
            )
            fig_pairs.update_layout(
                title="Top 10 Feature Interaction Pairs",
                xaxis_title="Interaction Strength",
                yaxis_title="Feature Pair",
                yaxis={"autorange": "reversed"},
                template=get_theme(self.theme)["template"],
                height=500,
            )
            html.append(fig_pairs.to_html(include_plotlyjs=False, div_id="plot-top-pairs"))

        else:
            # Few features - show network + matrix
            nodes = network_data["nodes"]

            if edges:
                # Create network visualization
                n_nodes = len(nodes)
                angles = [2 * np.pi * i / n_nodes for i in range(n_nodes)]
                node_x = [np.cos(angle) for angle in angles]
                node_y = [np.sin(angle) for angle in angles]

                node_positions = {
                    node["id"]: (node_x[i], node_y[i]) for i, node in enumerate(nodes)
                }
                node_importances = {node["id"]: node["importance"] for node in nodes}

                # Create edge traces
                edge_traces = []
                for edge in edges:
                    x0, y0 = node_positions[edge["source"]]
                    x1, y1 = node_positions[edge["target"]]

                    edge_traces.append(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode="lines",
                            line={
                                "width": edge["abs_weight"] * 10,
                                "color": "rgba(125,125,125,0.3)",
                            },
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

                # Create node trace
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    marker={
                        "size": [node_importances.get(node["id"], 0.1) * 100 for node in nodes],
                        "color": [node["total_interaction"] for node in nodes],
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Total<br>Interaction"},
                        "line": {"width": 2, "color": "white"},
                    },
                    text=[node["label"] for node in nodes],
                    textposition="top center",
                    textfont={"size": 10},
                    hovertemplate="<b>%{text}</b><br>Total Interaction: %{marker.color:.3f}<extra></extra>",
                    showlegend=False,
                )

                fig_network = go.Figure(data=edge_traces + [node_trace])
                fig_network.update_layout(
                    title=f"Feature Interaction Network (Top {len(edges)} Strongest Interactions)",
                    template=get_theme(self.theme)["template"],
                    height=600,
                    xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                    yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                    hovermode="closest",
                )
                html.append(fig_network.to_html(include_plotlyjs=False, div_id="plot-network"))

            # Interaction matrix heatmap
            matrix_data = interaction_viz_data["interaction_matrix"]
            matrix = matrix_data["matrix"]

            # Show up to INTERACTION_MATRIX_SIZE features for readability
            n_features_display = min(self.INTERACTION_MATRIX_SIZE, len(matrix_data["features"]))
            features_for_matrix = matrix_data["features"][:n_features_display]

            # Extract the subset of the matrix we need
            matrix_subset = [
                [matrix[i][j] for j in range(n_features_display)] for i in range(n_features_display)
            ]

            fig_matrix = go.Figure(
                data=go.Heatmap(
                    z=matrix_subset,
                    x=features_for_matrix,
                    y=features_for_matrix,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar={"title": "Interaction<br>Strength"},
                )
            )

            fig_matrix.update_layout(
                title=f"Interaction Matrix Heatmap (Top {len(features_for_matrix)} Features)",
                template=get_theme(self.theme)["template"],
                height=600,
                xaxis={"tickangle": -45},
            )
            html.append(fig_matrix.to_html(include_plotlyjs=False, div_id="plot-matrix"))

        return "".join(html)

    def _compose_html(self) -> str:
        """Compose final HTML document."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{self.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {self._get_base_styles()}
        </head>
        <body>
            {self._build_header()}
            {self._build_navigation()}
            <div class="dashboard-container">
                {self._build_sections()}
            </div>
            {self._get_base_scripts()}
        </body>
        </html>
        """


__all__ = ["FeatureImportanceDashboard"]
