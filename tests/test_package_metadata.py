"""Distribution metadata contracts."""

import tomllib
from pathlib import Path


def test_supported_python_range_matches_qualified_matrix() -> None:
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["requires-python"] == ">=3.12,<3.15"
    assert "Development Status :: 5 - Production/Stable" in project["classifiers"]
    assert "Programming Language :: Python :: 3.15" not in project["classifiers"]
    assert "pydantic>=2.13.4,<3" in project["dependencies"]
