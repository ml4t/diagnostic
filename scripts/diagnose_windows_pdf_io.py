"""Retain the pending native syscall in the actual bounded browser-close workflow."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def validate_observer(output: Path) -> None:
    trace = runpy.run_path("tests/test_visualization/_windows_pdf_trace.py")
    ready = output / "reader.json"
    program = """
import _winapi,json,os,sys,threading
from pathlib import Path
ready=Path(sys.argv[1])
temporary=ready.with_suffix('.new')
temporary.write_text(json.dumps({
    'pid':os.getpid(),'thread':threading.get_native_id(),
    'handle':_winapi.GetStdHandle(-10)}))
temporary.replace(ready)
assert os.read(0,1)==b'x'
"""
    process = subprocess.Popen([sys.executable, "-c", program, str(ready)], stdin=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 8
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        identity = json.loads(ready.read_text())
        while True:
            rows = trace["_pending_io"](process.pid)
            selected = [row for row in rows if row["tid"] == identity["thread"]]
            if selected and selected[0].get("io_pending"):
                break
            assert time.monotonic() < deadline, rows
            time.sleep(0.01)
        call = selected[0]["last_syscall"]
        result = {"identity": identity, "threads": rows}
        (output / "observer-validation.json").write_text(json.dumps(result, indent=2) + "\n")
        assert call["status"] == 0, result
        assert call["first_argument"] == identity["handle"], result
        assert call["matching_exports"] == ["NtReadFile"], result
        assert process.stdin is not None
        process.stdin.write(b"x")
        process.stdin.close()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()


def main() -> None:
    assert sys.platform == "win32"
    output = Path("artifacts/native-io-diagnosis").resolve()
    output.mkdir(parents=True, exist_ok=False)
    validate_observer(output)
    runner = runpy.run_path("scripts/check_windows_pdf_lifecycle.py")
    root = Path.cwd()
    node = (
        "tests/test_visualization/test_pdf_process_ownership.py::"
        "test_bounded_shutdown_fault_is_not_reapplied_by_the_actual_pipe_error_callback"
    )
    rows = []
    for attempt in range(1, 13):
        directory = output / f"attempt-{attempt:02}"
        directory.mkdir()
        result = runner["_run_case"](root, node, directory / "case-01")
        rows.append(result)
        document = {
            "source_revision": os.environ.get("GITHUB_SHA"),
            "python": sys.version,
            "results": rows,
            "scope": "Diagnostic only; all outcomes retained. Stop at first failure to inspect its actual syscall/handle. Passing outcomes do not explain the retained main failure.",
        }
        (output / "results.json").write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise SystemExit("Actual bounded-close failure retained; inspect native evidence")


if __name__ == "__main__":
    main()
