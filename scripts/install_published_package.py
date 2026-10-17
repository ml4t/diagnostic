"""Install the exact published version without changing release artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def install_published_package(
    python: Path,
    version: str,
    *,
    index_url: str = "https://pypi.org/simple",
    attempts: int = 12,
    delay: float = 10.0,
    timeout: float = 120.0,
) -> int:
    """Return the actual installer status for an exact-version verification install."""
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--no-cache",
        "--index-url",
        index_url,
        f"ml4t-diagnostic=={version}",
    ]
    try:
        return subprocess.run(command, check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print("Published package installation timed out; verification failed.", flush=True)
        return 124


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    return install_published_package(
        args.python,
        args.version,
        index_url=args.index_url,
        attempts=args.attempts,
        delay=args.delay,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
