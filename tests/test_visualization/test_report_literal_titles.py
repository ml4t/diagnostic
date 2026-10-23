"""Plain report titles retain their literal text in the rendered document."""

import shutil

import plotly.graph_objects as go
import pytest

from ml4t.diagnostic.visualization.report_generation import combine_figures_to_html


def test_literal_report_titles_survive_browser_parsing(tmp_path):
    from choreographer.browsers.chromium import Chromium

    playwright = pytest.importorskip("playwright.sync_api")
    chrome = Chromium.find_browser(skip_local=False) or shutil.which("chromium")
    title = 'Research <b>alpha</b> & "beta"'
    section_title = "Portfolio <small>long & short</small>"
    destination = tmp_path / "report.html"
    figure = go.Figure(go.Scatter(x=[1, 2], y=[0.01, 0.02]))
    original = figure.to_json()
    combine_figures_to_html(
        [figure],
        title=title,
        sections=[
            {
                "title": section_title,
                "text": "<p>Trusted section text</p>",
                "figure_index": 0,
            }
        ],
        output_file=destination,
    )
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            executable_path=chrome or runtime.chromium.executable_path,
            args=["--no-sandbox"],
        )
        try:
            context = browser.new_context(offline=True)
            page = context.new_page()
            page.goto(destination.as_uri(), wait_until="load")
            assert page.title() == title
            assert page.locator("h1").inner_text() == title
            assert page.locator(".section-title").inner_text() == section_title
            assert page.locator(".toc a").inner_text() == section_title
            assert page.locator(".section-text p").inner_text() == "Trusted section text"
        finally:
            browser.close()
    assert figure.to_json() == original
