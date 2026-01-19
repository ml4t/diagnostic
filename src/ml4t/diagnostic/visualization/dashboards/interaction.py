"""Feature interaction dashboard for deep-dive analysis.

This module provides the FeatureInteractionDashboard class for creating
standalone dashboards focused on feature interaction analysis.

Note: Currently a scaffold implementation. For working interaction
visualizations, use FeatureImportanceDashboard with interaction_results.
"""

from __future__ import annotations

from typing import Any, Literal

from ..data_extraction import InteractionVizData, extract_interaction_viz_data
from .base import BaseDashboard, DashboardSection


class FeatureInteractionDashboard(BaseDashboard):
    """Standalone dashboard for deep-dive feature interaction analysis.

    **⚠️ NOTE**: Currently a scaffold implementation (network, matrix, per-feature
    sections are placeholders). For working interaction visualizations, use
    ``FeatureImportanceDashboard`` with ``interaction_results`` parameter, which
    provides network graph, interaction matrix, and top pairs table.

    **Planned Features** (when fully implemented):
    - **Network View**: Interactive force-directed graph with filtering/drill-down
    - **Matrix View**: Large heatmap with clustering and sorting controls
    - **Per-Feature**: Individual feature interaction profiles with charts
    - **Cluster Analysis**: Automatic grouping of interacting feature sets
    - **Insights**: Auto-generated interpretation and recommendations

    This dashboard is intended for scenarios where you want a dedicated,
    interaction-only report (no importance analysis). When implemented, it will
    provide more advanced interaction-specific features than the integrated view
    in FeatureImportanceDashboard.

    Examples
    --------
    **Current Usage (Scaffolding)**

    >>> from ml4t.diagnostic.evaluation import compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import FeatureInteractionDashboard
    >>>
    >>> # Compute interactions
    >>> results = compute_shap_interactions(model, X_train,
    ...                                     feature_names=X_train.columns)
    >>>
    >>> # Create dashboard (generates scaffold with insights only)
    >>> dashboard = FeatureInteractionDashboard(
    ...     title="Feature Interaction Deep-Dive",
    ...     theme="dark",
    ...     min_interaction_strength=0.1
    ... )
    >>> dashboard.save("interaction_dashboard.html", results)
    >>> # ⚠️ Network/matrix/per-feature sections will show placeholders

    **Recommended Alternative (Fully Implemented)**

    >>> from ml4t.diagnostic.evaluation import analyze_ml_importance, compute_shap_interactions
    >>> from ml4t.diagnostic.visualization import FeatureImportanceDashboard
    >>>
    >>> # Run importance analysis
    >>> importance_results = analyze_ml_importance(
    ...     model, X_train, y_train,
    ...     methods=['mdi', 'pfi'],
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Compute interactions
    >>> interaction_results = compute_shap_interactions(
    ...     model, X_train,
    ...     feature_names=X_train.columns
    ... )
    >>>
    >>> # Use FeatureImportanceDashboard with interactions
    >>> dashboard = FeatureImportanceDashboard(
    ...     title="Complete Analysis (Importance + Interactions)"
    ... )
    >>> dashboard.save(
    ...     "full_dashboard.html",
    ...     analysis_results=importance_results,
    ...     interaction_results=interaction_results  # Adds 3 interaction sections
    ... )
    >>> # ✅ Includes: network graph, interaction matrix, top pairs table

    Notes
    -----
    - Currently only insights section is implemented
    - Network/matrix/per-feature sections show placeholder messages
    - For working interaction visualizations, use ``FeatureImportanceDashboard``
    - Future implementation will add: force-directed layout, clustering,
      filtering, drill-down, comparative analysis across datasets

    See Also
    --------
    FeatureImportanceDashboard : Includes working interaction visualizations
    compute_shap_interactions : Compute pairwise feature interactions
    extract_interaction_viz_data : Extract structured data for custom dashboards
    """

    def __init__(
        self,
        title: str = "Feature Interaction Analysis",
        theme: Literal["light", "dark"] = "light",
        width: int | None = None,
        height: int | None = None,
        min_interaction_strength: float = 0.1,
    ):
        """Initialize Feature Interaction Dashboard.

        Parameters
        ----------
        title : str, default="Feature Interaction Analysis"
            Dashboard title
        theme : {'light', 'dark'}, default='light'
            Visual theme
        width : int, optional
            Dashboard width in pixels
        height : int, optional
            Dashboard height in pixels
        min_interaction_strength : float, default=0.1
            Minimum interaction strength to display in network view
        """
        super().__init__(title, theme, width, height)
        self.min_interaction_strength = min_interaction_strength

    def generate(
        self,
        analysis_results: dict[str, Any],
        importance_results: dict[str, Any] | None = None,
        **_kwargs,
    ) -> str:
        """Generate complete dashboard HTML.

        Parameters
        ----------
        analysis_results : dict
            Results from compute_shap_interactions()
        importance_results : dict, optional
            Results from analyze_ml_importance() for cross-referencing
        **kwargs
            Additional parameters (currently unused)

        Returns
        -------
        str
            Complete HTML document
        """
        # Extract structured data
        viz_data = extract_interaction_viz_data(
            analysis_results, importance_results=importance_results
        )

        # Create sections
        self._create_network_section(viz_data)
        self._create_matrix_section(viz_data)
        self._create_per_feature_section(viz_data)
        self._create_insights_section(viz_data)

        # Compose HTML
        return self._compose_html()

    def _create_network_section(self, _viz_data: InteractionVizData) -> None:
        """Create network graph section."""
        # Placeholder: Network visualization not implemented in v1.0
        section = DashboardSection(
            title="Network View",
            description="Interactive network graph showing feature interactions",
            content="<p>Network section - to be implemented</p>",
        )
        self.sections.append(section)

    def _create_matrix_section(self, _viz_data: InteractionVizData) -> None:
        """Create interaction matrix heatmap section."""
        # Placeholder: Matrix heatmap not implemented in v1.0
        section = DashboardSection(
            title="Matrix View",
            description="Heatmap of all pairwise feature interactions",
            content="<p>Matrix section - to be implemented</p>",
        )
        self.sections.append(section)

    def _create_per_feature_section(self, _viz_data: InteractionVizData) -> None:
        """Create per-feature drill-down section."""
        # Placeholder: Per-feature views not implemented in v1.0
        section = DashboardSection(
            title="Per-Feature Analysis",
            description="Detailed interaction analysis for each feature",
            content="<p>Per-feature section - to be implemented</p>",
        )
        self.sections.append(section)

    def _create_insights_section(self, viz_data: InteractionVizData) -> None:
        """Create insights and recommendations section."""
        llm_context = viz_data["llm_context"]

        insights_html = f"""
        <div class="insights-panel">
            <h3>Summary</h3>
            <p>{llm_context["summary_narrative"]}</p>
        </div>

        <div class="insights-panel">
            <h3>Key Insights</h3>
            <ul>
                {"".join(f"<li>{insight}</li>" for insight in llm_context["key_insights"])}
            </ul>
        </div>

        <div class="insights-panel">
            <h3>Recommendations</h3>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in llm_context["recommendations"])}
            </ul>
        </div>

        {
            f'''
        <div class="insights-panel">
            <h3>Important Notes</h3>
            <ul>
                {"".join(f"<li>{caveat}</li>" for caveat in llm_context["caveats"])}
            </ul>
        </div>
        '''
            if llm_context["caveats"]
            else ""
        }
        """

        section = DashboardSection(
            title="Insights & Recommendations",
            description="Auto-generated interpretation and actionable recommendations",
            content=insights_html,
        )
        self.sections.append(section)

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


__all__ = ["FeatureInteractionDashboard"]
