"""Validate required contents and metadata in built release artifacts."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def main() -> None:
    """Check one sdist and wheel under the requested distribution directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()

    sdists = list(args.dist.glob("*.tar.gz"))
    wheels = list(args.dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise RuntimeError("expected exactly one source distribution and one wheel")

    with tarfile.open(sdists[0], "r:gz") as archive:
        names = archive.getnames()
    required_suffixes = (
        "/README.md",
        "/docs/getting-started/quickstart.md",
        "/examples/volatility_example.py",
        "/tests/test_documentation_examples.py",
    )
    missing = [suffix for suffix in required_suffixes if not any(n.endswith(suffix) for n in names)]
    if missing:
        raise RuntimeError(f"source distribution is missing required files: {missing}")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
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


if __name__ == "__main__":
    main()
