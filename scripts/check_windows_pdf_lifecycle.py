"""Qualify real Windows renderer-crash cleanup in independent test processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This qualification must execute on actual Windows")
    if args.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    results = []
    for attempt in range(1, args.repetitions + 1):
        log = args.output / f"attempt-{attempt:02}.log"
        report = args.output / f"attempt-{attempt:02}.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_visualization/test_pdf_process_ownership.py",
            "-k",
            "exit7 or render_error or close_error or close_command_hang or cancel or profile_sharing or slow_shutdown or completed_page or previous_blank_page",
            "-o",
            "addopts=",
            "-q",
            "-rA",
            "--timeout",
            "120",
            "--junitxml",
            str(report),
            "--basetemp",
            str(args.output / f"attempt-{attempt:02}-tmp"),
        ]
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
            timed_out = False
            try:
                process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                timed_out = True
                children = psutil.Process(process.pid).children(recursive=True)
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait(timeout=5)
                psutil.wait_procs(children, timeout=5)
        cases = list(ET.parse(report).iter("testcase")) if report.exists() else []
        passed = (
            process.returncode == 0
            and not timed_out
            and len(cases) == 10
            and not any(
                child.tag in ("failure", "error", "skipped") for case in cases for child in case
            )
        )
        row = {
            "attempt": attempt,
            "command": command,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "collected_cases": len(cases),
            "passed": passed,
            "elapsed_seconds": time.monotonic() - started,
            "log": log.name,
            "junit": report.name,
        }
        results.append(row)
        (args.output / "results.json").write_text(
            json.dumps(
                {
                    "platform": sys.platform,
                    "packages": {
                        name: version(name)
                        for name in [
                            "ml4t-diagnostic",
                            "plotly",
                            "kaleido",
                            "choreographer",
                            "pypdf",
                            "psutil",
                        ]
                    },
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)
    if not all(row["passed"] for row in results):
        raise SystemExit("At least one real crash-cleanup attempt failed; no retry is discarded")


if __name__ == "__main__":
    main()
