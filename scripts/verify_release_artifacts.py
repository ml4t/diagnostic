"""Validate required contents and metadata in built release artifacts."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

from packaging.specifiers import SpecifierSet

EXPECTED_DESCRIPTION = (
    "Signal diagnostics, statistical validation, and backtest evaluation for quantitative "
    "trading workflows."
)
EXPECTED_URLS = {
    "Homepage": "https://www.ml4trading.io/docs/diagnostic/",
    "Documentation": "https://www.ml4trading.io/docs/diagnostic/",
    "Repository": "https://github.com/ml4t/diagnostic",
    "Issues": "https://github.com/ml4t/diagnostic/issues",
    "Changelog": "https://github.com/ml4t/diagnostic/releases",
}


def metadata_failures(metadata: Message, *, expected_version: str) -> list[str]:
    """Return canonical metadata mismatches for a parsed email message."""
    get = metadata.get
    get_all = metadata.get_all
    failures = []
    expected_scalars = {
        "Name": "ml4t-diagnostic",
        "Version": expected_version,
        "Summary": EXPECTED_DESCRIPTION,
        "Author-email": "Stefan Jansen <stefan@applied-ai.com>",
        "Maintainer-email": "Stefan Jansen <pm@ml4trading.io>",
        "License-Expression": "MIT",
    }
    for field, expected in expected_scalars.items():
        if get(field) != expected:
            failures.append(f"{field} is {get(field)!r}, expected {expected!r}")
    if SpecifierSet(str(get("Requires-Python", ""))) != SpecifierSet(">=3.12,<3.15"):
        failures.append("Requires-Python differs from >=3.12,<3.15")
    project_urls = {}
    for value in get_all("Project-URL", []):
        label, separator, url = value.partition(", ")
        if separator:
            project_urls[label] = url
    if project_urls != EXPECTED_URLS:
        failures.append("Project-URL metadata differs from the canonical URLs")
    keywords = {value.strip() for value in str(get("Keywords", "")).split(",") if value.strip()}
    if not {"finance", "quantitative-finance", "algorithmic-trading"} <= keywords:
        failures.append("Keywords omit a required ecosystem keyword")
    if not any(str(name).endswith("LICENSE") for name in get_all("License-File", [])):
        failures.append("License-File metadata does not identify LICENSE")
    return failures


def main() -> None:
    """Check one sdist and wheel under the requested distribution directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    sdists = list(args.dist.glob("*.tar.gz"))
    wheels = list(args.dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise RuntimeError("expected exactly one source distribution and one wheel")

    with tarfile.open(sdists[0], "r:gz") as archive:
        names = archive.getnames()
    required_suffixes = (
        "/README.md",
        "/LICENSE",
        "/pyproject.toml",
        "/docs/getting-started/quickstart.md",
        "/examples/volatility_example.py",
        "/scripts/run_readme_quickstart.py",
        "/tests/test_documentation_examples.py",
    )
    missing = [suffix for suffix in required_suffixes if not any(n.endswith(suffix) for n in names)]
    if missing:
        raise RuntimeError(f"source distribution is missing required files: {missing}")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    failures = metadata_failures(metadata, expected_version=args.expected_version)
    if failures:
        raise RuntimeError(f"wheel metadata validation failed: {failures}")
    requirements = metadata.get_all("Requires-Dist", [])
    requirement_names = {
        match.group(0).lower().replace("_", "-")
        for requirement in requirements
        if (match := re.match(r"[A-Za-z0-9_.-]+", requirement)) is not None
    }
    if "pyarrow" not in requirement_names:
        raise RuntimeError("wheel metadata does not declare the required pyarrow dependency")
    if any(name.startswith("ml4t/diagnostic/artifacts/") for name in wheel_names):
        raise RuntimeError("wheel contains the removed artifact adapter")
    if any(name.endswith("/AGENTS.md") for name in wheel_names):
        raise RuntimeError("wheel contains repository agent orientation")


if __name__ == "__main__":
    main()
