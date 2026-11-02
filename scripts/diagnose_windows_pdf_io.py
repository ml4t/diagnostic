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
    focused_kernel_trace(output)


def focused_kernel_trace(output: Path) -> None:
    runner = runpy.run_path("scripts/check_windows_pdf_lifecycle.py")
    nodes = [
        "tests/test_visualization/test_pdf_process_ownership.py::" + name
        for name in [
            "test_real_early_pipe_closure_is_still_rejected_when_owned_cleanup_succeeds",
            "test_bounded_shutdown_fault_is_not_reapplied_by_the_actual_pipe_error_callback",
        ]
    ]
    commands = []

    def record(command: list[str], timeout: int = 60) -> None:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        commands.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        (output / "kernel-trace-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert result.returncode == 0, commands[-1]

    record(["wpr", "-profiles"])
    buffers = output / "kernel-buffers"
    buffers.mkdir()
    profile_path = str(Path("scripts/windows_pdf_io.wprp").resolve())
    record(["wpr", "-profiles", profile_path])
    profile = profile_path + "!OwnedPdfIO"
    record(["wpr", "-start", profile, "-filemode", "-recordtempto", str(buffers)])
    results = []
    try:
        directory = output / "focused-close"
        directory.mkdir()
        for attempt in range(1, 25):
            check = runner["_run_case"](
                Path.cwd(), nodes[(attempt - 1) % len(nodes)], directory / f"case-{attempt:02}"
            )
            results.append(check)
            (output / "focused-results.json").write_text(
                json.dumps(
                    {
                        "source_revision": os.environ.get("GITHUB_SHA"),
                        "python": sys.version,
                        "results": results,
                        "scope": "Diagnostic observation only; stop on first failure. Alternate the two retained early-close workflows; passing attempts do not clear retained main failure or qualify a runtime repair.",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(check), flush=True)
            trace_bytes = sum(path.stat().st_size for path in buffers.rglob("*") if path.is_file())
            (output / "kernel-storage.json").write_text(
                json.dumps(
                    {
                        "temporary_bytes": trace_bytes,
                        "stop_budget_bytes": 512 * 1024 * 1024,
                        "scope": "Check storage after each bounded case; stop on first failure or budget. Raw trace is retained.",
                    },
                    indent=2,
                )
                + "\n"
            )
            if not check["passed"] or trace_bytes >= 512 * 1024 * 1024:
                break
    finally:
        trace = str(output / "owned-close.etl")
        record(
            ["wpr", "-stop", trace, "Actual owned Chrome bounded close", "-skipPdbGen"], timeout=120
        )
    record(
        [
            "tracerpt",
            trace,
            "-o",
            str(output / "owned-close.csv"),
            "-of",
            "CSV",
            "-summary",
            str(output / "owned-close-summary.txt"),
            "-y",
        ],
        timeout=120,
    )
    assert results and all(check["passed"] for check in results), (
        "Native failure retained with kernel trace"
    )


if __name__ == "__main__":
    main()
