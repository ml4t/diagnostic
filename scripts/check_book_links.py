"""Verify every pinned book file link against the selected companion Git tree."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).parents[1]
BOOK_REPO = "stefan-jansen/machine-learning-for-trading"
BOOK_COMMIT = "2d6e8f95eeccaee66906245606471f570b5807e5"
MARKDOWN_LINK = re.compile(
    r"\]\((https://github\.com/stefan-jansen/machine-learning-for-trading/[^)\s]+)\)"
)


def documentation_paths() -> list[Path]:
    """Include the README and every public Markdown documentation page."""
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def book_file_links(paths: list[Path]) -> set[str]:
    """Reject mutable, mixed-revision, and unsupported companion URLs."""
    files: set[str] = set()
    for page in paths:
        content = page.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(content):
            url = urlsplit(match.group(1))
            if url.query or url.fragment:
                raise ValueError(f"Book link has a query or fragment in {page}: {url.geturl()}")
            suffix = url.path.removeprefix(f"/{BOOK_REPO}/")
            if suffix == f"tree/{BOOK_COMMIT}":
                continue
            if not suffix.startswith("blob/"):
                raise ValueError(f"Book link is not a pinned file in {page}: {url.geturl()}")
            _, revision, path = suffix.split("/", 2)
            if revision != BOOK_COMMIT:
                raise ValueError(f"Book link uses {revision}, expected {BOOK_COMMIT}: {page}")
            files.add(unquote(path))
    if not files:
        raise ValueError("No pinned book file links found")
    return files


def github_tree(ref: str, *, recursive: bool = True) -> dict:
    """Read one public Git tree using the GitHub CLI's configured credentials."""
    suffix = "?recursive=1" if recursive else ""
    completed = subprocess.run(
        ["gh", "api", f"repos/{BOOK_REPO}/git/trees/{ref}{suffix}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(f"GitHub tree lookup failed for {ref}: {completed.stderr.strip()}")
    tree = json.loads(completed.stdout)
    if tree.get("truncated"):
        raise ValueError(f"GitHub tree response was truncated for {ref}")
    return tree


def verify_book_files(files: set[str]) -> None:
    """Compare linked files with the immutable book revision, not today's main."""
    completed = subprocess.run(
        ["gh", "api", f"repos/{BOOK_REPO}/git/commits/{BOOK_COMMIT}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode:
        raise RuntimeError(f"Book commit lookup failed: {completed.stderr.strip()}")
    root_tree = github_tree(json.loads(completed.stdout)["tree"]["sha"], recursive=False)
    directories = {
        entry["path"]: entry["sha"] for entry in root_tree["tree"] if entry["type"] == "tree"
    }
    grouped: dict[str, set[str]] = {}
    for path in files:
        directory, separator, relative = path.partition("/")
        if not separator or directory not in directories:
            raise ValueError(f"Book path has no checked top-level directory: {path}")
        grouped.setdefault(directory, set()).add(relative)
    missing = []
    for directory, paths in grouped.items():
        subtree = github_tree(directories[directory])
        available = {entry["path"] for entry in subtree["tree"] if entry["type"] == "blob"}
        missing.extend(f"{directory}/{path}" for path in sorted(paths - available))
    if missing:
        raise ValueError(f"Book files missing at {BOOK_COMMIT}: {missing}")


def main() -> None:
    """Check linked file existence and enforce one recorded book revision."""
    files = book_file_links(documentation_paths())
    verify_book_files(files)
    print(f"Checked {len(files)} book files at {BOOK_COMMIT}")


if __name__ == "__main__":
    main()
