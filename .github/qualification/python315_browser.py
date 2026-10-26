"""Qualify previously omitted report browsers against real native and installed code."""

import hashlib
import json
import os
import platform
import shutil
import sys
import sysconfig
import xml.etree.ElementTree as ET
from importlib.metadata import distributions
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from playwright.sync_api import sync_playwright

import ml4t.diagnostic as diagnostic

mode = sys.argv[1]
assert mode in {"native", "whl", "tar.gz"}
assert sys.version_info[:2] == (3, 15) and not sysconfig.get_config_var("Py_GIL_DISABLED")
assert diagnostic.__version__ == "0.3.0.dev0"
manifest = json.loads(Path("artifacts/canonical/manifest.json").read_text())
assert manifest["source_revision"] == os.environ["GITHUB_SHA"]
for name, digest in manifest["archives"].items():
    assert hashlib.sha256((Path("artifacts/canonical") / name).read_bytes()).hexdigest() == digest
package = Path(diagnostic.__file__).resolve()
assert package.is_relative_to(Path.cwd()) == (mode == "native")
if mode != "native":
    assert all(
        hashlib.sha256((package.parents[2] / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["modules"].items()
    )
unsupported = {
    d.metadata["Name"]: d.metadata["Requires-Python"]
    for d in distributions()
    if d.metadata.get("Requires-Python")
    and not SpecifierSet(d.metadata["Requires-Python"]).contains(platform.python_version())
}
assert not unsupported, unsupported
chrome = shutil.which("google-chrome") or shutil.which("chromium")
assert chrome is not None and Path(chrome).is_file()
assert hashlib.sha256(Path(chrome).read_bytes()).hexdigest() == os.environ["ML4T_BROWSER_SHA256"]
with sync_playwright() as runtime:
    browser = runtime.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
    try:
        browser_version = browser.version
        assert browser_version and browser_version.split(".")[0].isdigit()
    finally:
        browser.close()
required = json.loads(Path(".github/qualification/python315-browser-cases.json").read_text())
expected = set(required["required_cases"])
assert len(expected) == 127
pdf_paths = [
    "tests/test_contract/test_pdf_collection_lifecycle.py",
    "tests/test_visualization/test_pdf_report_lifecycle.py",
    "tests/test_visualization/test_pdf_process_ownership.py",
    "tests/test_visualization/test_report_generation.py",
]
output = Path("artifacts/python315-browser")
output.mkdir(parents=True, exist_ok=True)
report = output / (mode + ".xml")
identity = {
    "mode": mode,
    "qualifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "package_versions": {d.metadata["Name"]: d.version for d in distributions()},
    "python": sys.version,
    "module": str(package),
    "unsupported": unsupported,
    "manifest": manifest,
    "browser": chrome,
    "browser_version": browser_version,
    "browser_sha256": os.environ["ML4T_BROWSER_SHA256"],
    "required_browser_cases": sorted(expected),
}
(output / (mode + "-identity.json")).write_text(json.dumps(identity, indent=2))
code = pytest.main(
    [
        "--noconftest",
        "--rootdir",
        str(Path.cwd()),
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-q",
        "-s",
        "--timeout",
        "120",
        "--basetemp",
        str(output / mode),
        "--junitxml",
        str(report),
        *sorted(expected),
        *pdf_paths,
    ]
)
if code:
    raise SystemExit(code)
cases = list(ET.parse(report).iter("testcase"))
assert len(cases) == 227
actual = {c.get("classname").replace(".", "/") + ".py::" + c.get("name"): c for c in cases}
assert expected <= actual.keys(), sorted(expected - actual.keys())
for name in expected:
    assert not any(actual[name].find(tag) is not None for tag in ("skipped", "failure", "error")), (
        name
    )
for case in cases:
    skipped = case.find("skipped")
    assert skipped is None or not any(
        word in skipped.get("message", "").lower() for word in ("chrome", "chromium", "playwright")
    ), (case.get("name"), skipped.get("message"))
pdf_modules = [module.removesuffix(".py").replace("/", ".") for module in pdf_paths]
pdf_cases = [
    case
    for case in cases
    if any(case.attrib["classname"].startswith(module) for module in pdf_modules)
]
assert len(pdf_cases) == 100
for case in pdf_cases:
    skipped = case.find("skipped")
    if skipped is not None:
        message = skipped.get("message", "")
        if sys.platform == "win32":
            assert "POSIX" in message, (case.get("name"), message)
        else:
            assert "Windows" in message, (case.get("name"), message)
print("Qualified all127 formerly omitted real browser cases and complete100-case PDF modules", mode)
