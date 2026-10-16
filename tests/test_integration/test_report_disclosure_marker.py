"""Real report disclosures must render the intended marker, not a Python escape."""

import shutil

import pytest

from ml4t.diagnostic.integration import generate_tearsheet_from_result

from .test_report_holding_units import _actual_result


@pytest.mark.parametrize("width", [1280, 375])
def test_actual_report_disclosure_renders_triangle_and_opens_real_rebalance_rows(width, tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
    if chrome is None:
        pytest.skip("Installed Chrome/Chromium required for actual report checks")
    playwright = pytest.importorskip("playwright.sync_api")
    result = _actual_result()
    path = tmp_path / "report.html"
    path.write_text(generate_tearsheet_from_result(result, include_statistical=False))
    errors, requests = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
        try:
            page = browser.new_page(viewport={"width": width, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: requests.append(request.url))
            page.goto(path.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            page.get_by_role("button", name="Trading", exact=True).click()
            summary = page.locator(".section-detail-summary").filter(has_text="Rebalance Events")
            summary.wait_for(state="visible")
            measured = summary.evaluate("""e=>({
                content:getComputedStyle(e,'::before').content,
                font:getComputedStyle(e).fontFamily,
                markerFont:getComputedStyle(e,'::before').fontFamily,
                open:e.parentElement.open,
                overflow:document.documentElement.scrollWidth>innerWidth,
                width:innerWidth
            })""")
            assert measured["width"] == width and not measured["overflow"]
            assert not measured["open"] and measured["font"] == measured["markerFont"]
            assert measured["content"] == '"▸"'
            summary.click()
            opened = summary.evaluate("""e=>({
                open:e.parentElement.open,
                rows:e.parentElement.querySelectorAll('tbody tr').length,
                visible:e.parentElement.querySelector('table').getBoundingClientRect().height>0
            })""")
            assert opened == {"open": True, "rows": 2, "visible": True}
        finally:
            browser.close()
    assert not errors and not requests
