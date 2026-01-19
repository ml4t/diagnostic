"""Example usage of report generation for feature diagnostics.

This script demonstrates how to generate comprehensive diagnostic reports in
multiple formats (HTML, JSON, Markdown) from FeatureDiagnostics results.

Examples:
    Run this script to generate example reports:

        $ python examples/report_generation_example.py

    Reports will be saved to examples/output/ directory.
"""

from pathlib import Path

import numpy as np

from ml4t.diagnostic.evaluation import (
    FeatureDiagnostics,
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
    generate_multi_feature_html_report,
    save_report,
)


def example_1_basic_reports():
    """Example 1: Generate all three report formats."""
    print("=" * 70)
    print("Example 1: Basic Report Generation (HTML, JSON, Markdown)")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(500)

    # Run diagnostics
    print("\nRunning diagnostics...")
    diagnostics = FeatureDiagnostics()
    result = diagnostics.run_diagnostics(data, name="test_signal")
    print(f"✓ Health Score: {result.health_score:.2f}/1.00")

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate HTML
    print("\nGenerating HTML report...")
    html = generate_html_report(result)
    html_path = save_report(html, output_dir / "report.html", overwrite=True)
    print(f"✓ HTML: {html_path} ({len(html):,} bytes)")

    # Generate JSON
    print("\nGenerating JSON report...")
    json_data = generate_json_report(result)
    json_path = save_report(json_data, output_dir / "report.json", overwrite=True)
    print(f"✓ JSON: {json_path} ({len(json_data):,} bytes)")

    # Generate Markdown
    print("\nGenerating Markdown report...")
    markdown = generate_markdown_report(result)
    md_path = save_report(markdown, output_dir / "report.md", overwrite=True)
    print(f"✓ Markdown: {md_path} ({len(markdown):,} bytes)")

    print("\n✓ Example 1 complete\n")


def example_2_multi_feature():
    """Example 2: Compare multiple features in one report."""
    print("=" * 70)
    print("Example 2: Multi-Feature Comparison")
    print("=" * 70)

    # Generate different types of signals
    np.random.seed(42)
    diagnostics = FeatureDiagnostics()
    results = []

    print("\nAnalyzing multiple signals...")

    # White noise
    data1 = np.random.randn(500)
    result1 = diagnostics.run_diagnostics(data1, name="white_noise")
    results.append(result1)
    print(f"✓ white_noise: score={result1.health_score:.2f}")

    # Trending
    data2 = np.cumsum(np.random.randn(500)) * 0.01
    result2 = diagnostics.run_diagnostics(data2, name="trending")
    results.append(result2)
    print(f"✓ trending: score={result2.health_score:.2f}")

    # Heavy tails
    data3 = np.random.standard_t(df=3, size=500) * 0.02
    result3 = diagnostics.run_diagnostics(data3, name="heavy_tails")
    results.append(result3)
    print(f"✓ heavy_tails: score={result3.health_score:.2f}")

    # Generate comparison report
    print("\nGenerating comparison report...")
    html = generate_multi_feature_html_report(results, title="Portfolio Diagnostics")

    output_dir = Path(__file__).parent / "output"
    html_path = save_report(html, output_dir / "comparison.html", overwrite=True)
    print(f"✓ Comparison HTML: {html_path} ({len(html):,} bytes)")

    print("\n✓ Example 2 complete\n")


def main():
    """Run examples."""
    print("\nFeature Diagnostics Report Generation")
    print("=" * 70)

    try:
        example_1_basic_reports()
        example_2_multi_feature()

        print("=" * 70)
        print("All examples complete!")
        print("Check the examples/output/ directory for generated reports.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


if __name__ == "__main__":
    main()
