"""Retain the pending native syscall in the actual bounded browser-close workflow."""

from __future__ import annotations

import ast
import hashlib
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
            rows = trace["_pending_io"](identity["pid"])
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
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            import psutil

            children = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait(timeout=5)
            psutil.wait_procs(children, timeout=5)


def main() -> None:
    assert sys.platform == "win32"
    output = Path("artifacts/native-io-diagnosis").resolve()
    output.mkdir(parents=True, exist_ok=False)
    validate_observer(output)
    module_path = "tests/test_visualization/test_pdf_process_ownership.py"
    current = runpy.run_path(module_path)
    test = current["test_late_job_members_handshake_survives_real_file_sharing_denial"]
    revision = "3db007cccc5ccb2ea1e766dbfd7402271dc14166"
    original = subprocess.check_output(["git", "show", f"{revision}:{module_path}"], text=True)
    base = next(
        ast.literal_eval(statement.value)
        for statement in ast.parse(original).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PROBE" for target in statement.targets
        )
    )
    repaired = test.__globals__["_PROBE"]
    results = []
    for variant, probe in [("base", base), ("fixed", repaired)]:
        directory = output / f"readiness-{variant}"
        directory.mkdir()
        test.__globals__["_PROBE"] = probe
        error = None
        try:
            test(directory)
        except AssertionError as failure:
            error = str(failure)
        finally:
            test.__globals__["_PROBE"] = repaired
        denial = json.loads((directory / "report.processes.ready-sharing.json").read_text())
        results.append(
            {
                "variant": variant,
                "probe_sha256": hashlib.sha256(probe.encode()).hexdigest(),
                "actual_denials": denial,
                "error": error,
            }
        )
        (output / "readiness-regression.json").write_text(
            json.dumps(
                {
                    "base_source": revision,
                    "fixed_source": os.environ.get("GITHUB_SHA"),
                    "results": results,
                    "scope": "Actual failed-base test producer versus repaired fixture, same public PDF runtime; separate from the unresolved native cleanup recurrence.",
                },
                indent=2,
            )
            + "\n"
        )
        assert denial["denials"] and all(
            row["errno"] == 13 and row["native_read_failed"] and row["native_error"] == 32
            for row in denial["denials"]
        ), denial
        if variant == "base":
            assert error and "Permission denied" in error and "debugger-ready.json" in error, error
        else:
            assert error is None, error
    sys.argv = [
        "check_windows_pdf_lifecycle.py",
        "--repetitions",
        "1",
        "--output",
        str(output / "original-order"),
    ]
    runpy.run_path("scripts/check_windows_pdf_lifecycle.py", run_name="__main__")


if __name__ == "__main__":
    main()
