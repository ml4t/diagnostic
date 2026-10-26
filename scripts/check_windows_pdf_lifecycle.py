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
from typing import Any

import psutil

_MODULE = "tests/test_visualization/test_pdf_process_ownership.py"
_SELECTION = (
    "exit7 or render_error or close_error or close_command_hang or cancel or "
    "profile_sharing or slow_shutdown or completed_page or previous_blank_page or "
    "context_burst or unresponsive_startup or pipe_close_evidence or readonly_browser_profile or "
    "profile_cleanup_deadline or unresponsive_browser_after_closed_channel or close_hang or "
    "late_job_members or lost_worker_exit or early_pipe_closure or live_worker or worker_owned or unknown_worker_exit or loaded_renderer_document or bounded_shutdown_fault"
)


def _collect_cases(root: Path, output: Path) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            _MODULE,
            "-k",
            _SELECTION,
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (output / "collection.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    cases = [line.strip() for line in result.stdout.splitlines() if line.startswith(_MODULE + "::")]
    if result.returncode or len(cases) != 29 or len(set(cases)) != 29:
        raise SystemExit("The real Windows lifecycle corpus must collect all 29 distinct cases")
    return cases


def _run_case(root: Path, node: str, output: Path) -> dict[str, Any]:
    output.mkdir()
    log, report = output / "case.log", output / "case.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        node,
        "-o",
        "addopts=",
        "-vv",
        "-rA",
        "--timeout",
        "120",
        "--junitxml",
        str(report),
        "--basetemp",
        str(output / "tmp"),
    ]
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        owner = psutil.Process(process.pid)
        timed_out = False
        try:
            process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timed_out = True
            children = owner.children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait(timeout=5)
            psutil.wait_procs(children, timeout=5)
    report_error = None
    try:
        cases = list(ET.parse(report).iter("testcase")) if report.exists() else []
    except ET.ParseError as error:
        cases, report_error = [], str(error)
    return {
        "node": node,
        "command": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "collected_cases": len(cases),
        "passed": process.returncode == 0
        and not timed_out
        and len(cases) == 1
        and not any(
            child.tag in ("failure", "error", "skipped") for case in cases for child in case
        ),
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log.relative_to(output.parent.parent)),
        "junit": str(report.relative_to(output.parent.parent)),
        "report_error": report_error,
    }


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
    args.output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    nodes = _collect_cases(root, args.output)
    results = []
    for attempt in range(1, args.repetitions + 1):
        log = args.output / f"attempt-{attempt:02}.log"
        report = args.output / f"attempt-{attempt:02}.xml"
        started = time.monotonic()
        directory = args.output / f"attempt-{attempt:02}"
        directory.mkdir()
        checks = []
        document = ET.Element("testsuites")
        with log.open("w", encoding="utf-8") as stream:
            for number, node in enumerate(nodes, 1):
                print(json.dumps({"attempt": attempt, "case": number, "node": node}), flush=True)
                check = _run_case(root, node, directory / f"case-{number:02}")
                checks.append(check)
                stream.write(node + "\n" + (args.output / check["log"]).read_text(encoding="utf-8"))
                stream.flush()
                result_path = args.output / check["junit"]
                if result_path.exists() and check["report_error"] is None:
                    tree = ET.parse(result_path).getroot()
                    document.extend(list(tree) if tree.tag == "testsuites" else [tree])
        ET.ElementTree(document).write(report, encoding="utf-8", xml_declaration=True)
        row = {
            "attempt": attempt,
            "cases": checks,
            "exit_code": 0 if all(check["passed"] for check in checks) else 1,
            "timed_out": any(check["timed_out"] for check in checks),
            "collected_cases": sum(check["collected_cases"] for check in checks),
            "passed": len(checks) == len(nodes) and all(check["passed"] for check in checks),
            "elapsed_seconds": time.monotonic() - started,
            "log": log.name,
            "junit": report.name,
        }
        results.append(row)
        (args.output / "results.json").write_text(
            json.dumps(
                {
                    "platform": sys.platform,
                    "python": sys.version,
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
