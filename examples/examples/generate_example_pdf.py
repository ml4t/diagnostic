"""Example: Generate PDF reports from feature importance analysis.

This script demonstrates how to export visualization reports to PDF format.
"""

import plotly.graph_objects as go

from ml4t.diagnostic.visualization import export_figures_to_pdf

# Create sample figures
fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"],
        y=[0.35, 0.28, 0.22, 0.10, 0.05],
        marker=dict(
            color=[0.35, 0.28, 0.22, 0.10, 0.05],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
    )
)
fig1.update_layout(
    title="Feature Importance Rankings",
    xaxis_title="Features",
    yaxis_title="Importance Score",
    height=500,
    width=800,
)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=["MDI", "PFI", "SHAP"],
        y=[0.85, 0.92, 0.78],
        mode="markers+lines",
        marker=dict(size=12, color="royalblue"),
        line=dict(width=2, color="royalblue"),
    )
)
fig2.update_layout(
    title="Method Agreement (Correlation)",
    xaxis_title="Method Comparison",
    yaxis_title="Spearman Correlation",
    height=400,
    width=800,
    yaxis=dict(range=[0, 1]),
)

# Export to PDF
print("Generating example PDF...")
pdf_path = export_figures_to_pdf(
    figures=[fig1, fig2],
    output_file="example_feature_importance.pdf",
    page_size=(800, 600),  # A4-like landscape
    scale=2.0,  # High quality
)

print(f"PDF generated successfully: {pdf_path}")
print("Open the PDF to view the visualizations.")
