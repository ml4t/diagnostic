"""Install the exact published version without changing release artifacts."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import time
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
    """Retry install-index propagation, never publication or artifact creation."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("version must be an exact stable version, such as 0.1.9")
    if attempts < 1 or not math.isfinite(delay) or delay < 0:
        raise ValueError("attempts must be positive and delay finite and nonnegative")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
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
    for attempt in range(1, attempts + 1):
        print(f"Published package install attempt {attempt}/{attempts}", flush=True)
        try:
            result = subprocess.run(command, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Published package installation timed out; verification failed.", flush=True)
            return 124
        if result.returncode == 0:
            return 0
        if attempt < attempts:
            time.sleep(delay)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        return install_published_package(
            args.python,
            args.version,
            index_url=args.index_url,
            attempts=args.attempts,
            delay=args.delay,
            timeout=args.timeout,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
