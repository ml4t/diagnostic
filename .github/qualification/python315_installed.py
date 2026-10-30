"""Run full and serial browser contracts against the common installed candidate."""

import hashlib
import json
import os
import platform
import sys
import sysconfig
from importlib.metadata import distributions
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

import ml4t.diagnostic as diagnostic

root = Path("artifacts/python315")
manifest = json.loads((root / "dist/manifest.json").read_text())
assert manifest["source_revision"] == os.environ["GITHUB_SHA"]
for name, digest in manifest["archives"].items():
    assert hashlib.sha256((root / "dist" / name).read_bytes()).hexdigest() == digest
package = Path(diagnostic.__file__).resolve()
assert not package.is_relative_to(Path.cwd())
assert diagnostic.__version__ == "0.3.0.dev0"
assert sys.version_info[:2] == (3, 15) and sys.version_info.releaselevel == "final"
assert not sysconfig.get_config_var("Py_GIL_DISABLED")
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
identity = {
    "python": sys.version,
    "module": str(package),
    "manifest": manifest,
    "unsupported": unsupported,
    "packages": {d.metadata["Name"]: d.version for d in distributions()},
}
(root / "installed-identity.json").write_text(json.dumps(identity, indent=2))
assert not unsupported, unsupported
core = pytest.main(
    ["tests/", "-q", "-n", "2", "--timeout", "120", "--junitxml", str(root / "installed.xml")]
)
browser = pytest.main(
    [
        "-o",
        "addopts=",
        "-v",
        "-s",
        "-n",
        "0",
        "--timeout",
        "120",
        "--junitxml",
        str(root / "installed-browser.xml"),
        "tests/test_contract/test_pdf_collection_lifecycle.py",
        "tests/test_visualization/test_pdf_report_lifecycle.py",
        "tests/test_visualization/test_pdf_process_ownership.py",
        "tests/test_visualization/test_report_generation.py",
        "tests/test_visualization/test_report_literal_titles.py",
    ]
)
raise SystemExit(core or browser)
