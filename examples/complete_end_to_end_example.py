"""Complete End-to-End Example: From Model Training to PDF Reports.

ML4T Evaluation provides rigorous statistical evaluation and visualization for ML models,
with a focus on feature analysis and preventing overfitting through proper
statistical testing.

This example demonstrates the complete workflow:
1. Train a model (using sklearn as an example)
2. Analyze feature importance using multiple methods (MDI, PFI, SHAP)
3. Compute feature interactions (SHAP-based)
4. Generate interactive HTML visualizations
5. Export high-quality PDF reports for stakeholders

Why use ML4T Evaluation?
--------------
- Multi-method importance analysis catches method-specific biases
- SHAP interactions reveal feature synergies often missed by univariate importance
- Interactive HTML reports for exploration, PDFs for presentations
- Designed for financial ML but applicable to any domain

What you'll learn:
-----------------
- How to run importance analysis with consensus ranking
- How to compute and interpret feature interactions
- How to generate publication-quality visualizations
- How to export to PDF for sharing with non-technical stakeholders

Expected runtime: ~10-20 seconds (depending on your machine)
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ML4T Evaluation imports
from ml4t.diagnostic.evaluation import analyze_ml_importance, compute_shap_interactions
from ml4t.diagnostic.visualization import (
    generate_combined_report,
    generate_importance_report,
)

# ============================================================================
# Step 1: Create Dataset and Train Model
# ============================================================================
# In practice, you would load your own dataset here. This example uses
# synthetic data to demonstrate the workflow without external dependencies.

print("=" * 70)
print("STEP 1: Training Model")
print("=" * 70)

# Create synthetic dataset with known structure:
# - 8 informative features (truly predictive)
# - 3 redundant features (linear combinations of informative features)
# - 4 noise features (random, no predictive power)
# This structure lets us validate that importance methods correctly identify
# the informative features and downweight noise.
X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    n_repeated=0,
    n_classes=2,
    class_sep=1.0,  # Moderate difficulty
    random_state=42,
)

# Feature names help interpret results
# In practice, use your actual feature names (e.g., "price_momentum", "volume_ratio")
feature_names = [
    f"informative_{i}" if i < 8 else f"redundant_{i - 8}" if i < 11 else f"noise_{i - 11}"
    for i in range(15)
]

# Split data - use test set for feature importance to avoid overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest
# ML4T Evaluation works with any sklearn-compatible model (RandomForest, XGBoost, LightGBM, etc.)
print(f"Training Random Forest on {len(X_train)} samples...")
model = RandomForestClassifier(
    n_estimators=100,  # More trees = more stable importance
    max_depth=10,  # Limit depth to prevent overfitting
    min_samples_leaf=5,  # Regularization
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate model quality
# Feature importance is only meaningful if the model actually works!
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"✅ Model trained! Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

if test_score < 0.6:
    print("⚠️  Warning: Low test accuracy. Feature importance may not be meaningful.")

# ============================================================================
# Step 2: Analyze Feature Importance
# ============================================================================
# ML4T Evaluation supports three importance methods:
#
# - MDI (Mean Decrease Impurity): Fast, built into tree models, but biased
#   toward high-cardinality features. Good for quick exploration.
#
# - PFI (Permutation Feature Importance): Unbiased, model-agnostic, but slower.
#   Measures impact on model performance when feature is shuffled.
#
# - SHAP (SHapley Additive exPlanations): Theoretically sound (game theory),
#   provides local + global importance. Slowest but most rigorous.
#
# Using multiple methods provides consensus and catches method-specific biases.

print("\n" + "=" * 70)
print("STEP 2: Analyzing Feature Importance (MDI, PFI)")
print("=" * 70)

importance_results = analyze_ml_importance(
    model,
    X_test,  # Use held-out test set to avoid overfitting
    y_test,
    feature_names=feature_names,
    methods=["mdi", "pfi"],  # Add "shap" for full rigor (slower)
    n_repeats=10,  # PFI repeats for statistical stability
    random_state=42,
)

# Results include:
# - Individual method results (importances, rankings)
# - Consensus ranking (Borda count across methods)
# - Method agreement (Spearman correlation between rankings)
# - Warnings (e.g., low model performance, high variance)

print("\n✅ Importance analysis complete!")
print(f"   Methods run: {', '.join(importance_results['methods_run'])}")
print(f"   Top 5 features: {', '.join(importance_results['consensus_ranking'][:5])}")
print(
    f"   Method agreement (correlation): {importance_results['method_agreement']['mdi_vs_pfi']:.3f}"
)

# Method agreement > 0.7 is good, > 0.8 is excellent
# Low agreement suggests method-specific biases or unstable importance
if importance_results["method_agreement"]["mdi_vs_pfi"] < 0.7:
    print("   ⚠️  Low method agreement - consider adding SHAP for tie-breaking")

if importance_results["warnings"]:
    print("\n⚠️  Warnings:")
    for warning in importance_results["warnings"]:
        print(f"   - {warning}")

# ============================================================================
# Step 3: Analyze Feature Interactions (Optional but Powerful!)
# ============================================================================
# Feature importance tells you which features matter in isolation, but
# interactions reveal synergies: features that work together to produce
# stronger effects than either feature alone.
#
# Example: In trading, "momentum" and "volatility" might interact:
# - High momentum + low volatility = strong signal
# - High momentum + high volatility = noise
#
# SHAP interactions decompose predictions into pairwise feature effects.
# Computation is O(n_features²), so we use a subsample for large datasets.

print("\n" + "=" * 70)
print("STEP 3: Computing SHAP Feature Interactions")
print("=" * 70)

# Use smaller sample for speed
# For production: use full test set if n_features < 20, otherwise sample 500-1000
interaction_results = compute_shap_interactions(
    model,
    X_test[:200],  # Subsample for speed (full: X_test)
    feature_names=feature_names,
    max_samples=200,  # Safety limit (will error if input > max_samples)
)

print("\n✅ Interaction analysis complete!")
print(f"   Computation time: {interaction_results['computation_time']:.2f}s")
print("   Top 3 interactions:")
for feat_i, feat_j, strength in interaction_results["top_interactions"][:3]:
    print(f"      {feat_i} × {feat_j}: {strength:.4f}")

# Interaction strength interpretation:
# - > 0.1: Strong interaction (investigate further)
# - 0.05-0.1: Moderate interaction
# - < 0.05: Weak interaction (may be noise)

# ============================================================================
# Step 4: Generate Reports (HTML + PDF)
# ============================================================================
# ML4T Evaluation can generate three types of reports:
#
# 1. Importance Report: Feature rankings, method comparison, consensus
# 2. Interaction Report: Heatmap, network graph, top interactions
# 3. Combined Report: Both importance and interactions in one document
#
# Each report comes in two formats:
# - HTML: Interactive, zoomable, hover tooltips (open in browser)
# - PDF: Static, high-quality, shareable (for presentations/emails)
#
# Available themes: "default", "dark", "presentation" (large fonts)

print("\n" + "=" * 70)
print("STEP 4: Generating Reports")
print("=" * 70)

# Option A: Importance-only report
# Best for: Quick analysis, presentations focusing on feature selection
print("\n📊 Generating importance report (HTML + PDF)...")
importance_html = generate_importance_report(
    importance_results=importance_results,
    output_file="importance_report.html",
    title="Feature Importance Analysis",
    theme="dark",  # Options: "default", "dark", "presentation"
    export_pdf=True,  # Also creates importance_report.pdf
    pdf_scale=2.0,  # Higher = better quality but larger file (1.0-3.0)
)
print(f"   ✅ Saved: {importance_html}")
print("   ✅ Saved: importance_report.pdf")

# Option B: Interaction-only report
# Best for: Deep-dive into feature synergies
#
# ⚠️  KNOWN ISSUE: Network plot has a theme layout conflict bug
# Workaround: Use theme="default" or skip network plot
# See: tests/test_visualization/test_report_generation.py (TASK-171)
#
# Uncomment this when the bug is fixed:
# print("\n📊 Generating interaction report (HTML + PDF)...")
# interaction_html = generate_interaction_report(
#     interaction_results=interaction_results,
#     output_file="interaction_report.html",
#     title="Feature Interaction Analysis",
#     theme="default",  # Use "default" to avoid network plot bug
#     export_pdf=True,
#     pdf_scale=2.0,
# )
# print(f"   ✅ Saved: {interaction_html}")
# print(f"   ✅ Saved: interaction_report.pdf")

# Option C: Combined comprehensive report
# Best for: Complete analysis, sharing with stakeholders
print("\n📊 Generating combined report (HTML + PDF)...")
combined_html = generate_combined_report(
    importance_results=importance_results,
    interaction_results=None,  # Set to interaction_results when bug fixed
    output_file="complete_analysis.html",
    title="Complete Feature Analysis Report",
    theme="presentation",  # Large fonts perfect for presentations!
    export_pdf=True,
    pdf_scale=2.5,  # Very high quality for stakeholder review
)
print(f"   ✅ Saved: {combined_html}")
print("   ✅ Saved: complete_analysis.pdf")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("✅ COMPLETE END-TO-END WORKFLOW SUCCESSFUL!")
print("=" * 70)

print("""
Generated Files:
  - importance_report.html & importance_report.pdf
  - complete_analysis.html & complete_analysis.pdf

Next Steps:
  1. Open HTML files in your browser for interactive exploration
     - Hover over bars to see exact values
     - Zoom in on plots by clicking and dragging
     - Toggle methods on/off by clicking legend items

  2. Share PDF files with stakeholders
     - High-quality vector graphics (zoom without pixelation)
     - Professional presentation format
     - No software dependencies required

  3. Use insights to improve your model:
     - Remove low-importance features (reduce overfitting)
     - Engineer new features based on strong interactions
     - Investigate why methods disagree (data quality issues?)
     - Validate results on different time periods/datasets

  4. Customize for your workflow:
     - Add SHAP to methods list for more rigor
     - Try different themes ("dark" for presentations)
     - Adjust pdf_scale for file size vs quality tradeoff
     - Use interaction_results in combined report (when bug fixed)

Key Insights from This Analysis:
""")

print(f"  Top 3 features: {', '.join(importance_results['consensus_ranking'][:3])}")
print(f"  Method agreement: {importance_results['method_agreement']['mdi_vs_pfi']:.1%}")

if interaction_results["top_interactions"]:
    feat_i, feat_j, strength = interaction_results["top_interactions"][0]
    print(f"  Strongest interaction: {feat_i} × {feat_j} (strength: {strength:.4f})")

print("""
For More Information:
  - Documentation: /docs (when published)
  - Example projects: /examples
  - API reference: See docstrings in ml4t.diagnostic.evaluation and ml4t.diagnostic.visualization
  - Feature requests: Open an issue on GitHub
""")

print("=" * 70)
