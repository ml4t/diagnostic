"""Dashboard styling constants.

CSS styling for the Trade SHAP diagnostics dashboard.
"""

from __future__ import annotations

# Professional CSS styling for styled mode
STYLED_CSS = """
<style>
/* Professional theme colors */
:root {
    --primary-color: #1f77b4;
    --success-color: #51CF66;
    --warning-color: #FF9800;
    --error-color: #FF6B6B;
}

/* Headers with professional styling */
h1 {
    color: var(--primary-color);
    font-weight: 600;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e0e0e0;
}

h2 {
    font-weight: 600;
    margin-top: 1.5rem;
}

/* Enhanced metrics */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 600;
}

/* Polished containers */
.stExpander {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Professional buttons */
.stButton>button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Download buttons styling */
.stDownloadButton>button {
    background-color: var(--primary-color) !important;
    color: white !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    font-weight: 500;
}

/* Progress bars */
.stProgress > div > div {
    background-color: var(--success-color);
}
</style>
"""
