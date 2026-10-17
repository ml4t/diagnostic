"""Own the PDF worker and its browser independently of Kaleido's close path."""

import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

_BOOTSTRAP = (
    "import sys,runpy; "
    "assert sys.stdin.buffer.read(1)==b'1'; "
    "runpy.run_module('ml4t.diagnostic.visualization._pdf_worker',run_name='__main__')"
)
_RENDER_TIMEOUT = 90
_CLOSE_TIMEOUT = 15
_logger = logging.getLogger(__name__)


class _WindowsJob:
    """A non-breakaway job owns descendants before the worker can start Chrome."""

    def __init__(self, pid: int) -> None:
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("active_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "QueryInformationJobObject": (
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p],
                wintypes.BOOL,
            ),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = args
            function.restype = result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._check(
                self.api.SetInformationJobObject(
                    self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
            process = self.api.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                self._check(self.api.AssignProcessToJobObject(self.handle, process))
            finally:
                self.api.CloseHandle(process)
        except BaseException:
            self.api.CloseHandle(self.handle)
            raise

    @staticmethod
    def _check(ok: int) -> None:
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self) -> None:
        from ctypes import wintypes

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("page_faults", wintypes.DWORD),
                ("total", wintypes.DWORD),
                ("active", wintypes.DWORD),
                ("terminated", wintypes.DWORD),
            ]

        try:
            self._check(self.api.TerminateJobObject(self.handle, 120))
            deadline = time.monotonic() + 5
            while True:
                accounting = Accounting()
                self._check(
                    self.api.QueryInformationJobObject(
                        self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
                    )
                )
                if accounting.active == 0:
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "PDF renderer job still has active processes after termination"
                    )
                time.sleep(0.05)
        finally:
            self.api.CloseHandle(self.handle)


class _OwnedProcesses:
    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self.root = psutil.Process(process.pid)
        self.observed: dict[int, psutil.Process] = {process.pid: self.root}
        self.job = _WindowsJob(process.pid) if sys.platform == "win32" else None

    def observe(self) -> None:
        try:
            for child in self.root.children(recursive=True):
                self.observed[child.pid] = child
        except psutil.NoSuchProcess:
            pass

    def terminate(self) -> None:
        self.observe()
        if self.job is not None:
            self.job.terminate()
        else:
            # Chrome starts a new session of its own. Kill observed child-led groups too.
            for process in reversed(list(self.observed.values())):
                try:
                    if not process.is_running():
                        continue
                    group = os.getpgid(process.pid)
                    if group == process.pid:
                        os.killpg(group, signal.SIGKILL)
                    else:
                        process.kill()
                except (psutil.NoSuchProcess, ProcessLookupError):
                    pass
            _, alive = psutil.wait_procs(list(self.observed.values()), timeout=5)
            running = []
            for process in alive:
                try:
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        running.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
            if running:
                raise RuntimeError(f"PDF renderer processes still running after cleanup: {running}")
        self.process.wait(timeout=5)


def render_pdf_pages(figures: list[Any], paths: list[Path], **options: Any) -> None:
    """Return only after all pages render and the owned worker tree is terminal."""
    import plotly.io as pio
    from plotly.utils import PlotlyJSONEncoder

    request = {
        "defaults": {
            name: getattr(pio.defaults, name, None) for name in ("plotlyjs", "mathjax", "headers")
        },
        "pages": [
            {
                "fig": figure.to_plotly_json(),
                "path": str(path),
                "opts": {"format": "pdf", **options},
                "topojson": getattr(pio.defaults, "topojson", None),
            }
            for figure, path in zip(figures, paths, strict=True)
        ],
    }
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile(mode="w+b") as output:
        request_path = Path(directory) / "request.json"
        status_path = Path(directory) / "status.json"
        request_path.write_text(json.dumps(request, cls=PlotlyJSONEncoder), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", _BOOTSTRAP, str(request_path), str(status_path)],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=output,
            start_new_session=sys.platform != "win32",
        )
        owner = None
        status: dict[str, Any] = {}
        failure = None
        try:
            owner = _OwnedProcesses(process)
            assert process.stdin is not None
            process.stdin.write(b"1")
            process.stdin.close()
            previous = None
            deadline = time.monotonic() + _RENDER_TIMEOUT
            while True:
                owner.observe()
                if status_path.exists():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                progress = (status.get("phase"), status.get("completed"))
                if progress != previous:
                    deadline = time.monotonic() + (
                        _CLOSE_TIMEOUT if status.get("phase") == "closing" else _RENDER_TIMEOUT
                    )
                    previous = progress
                if process.poll() is not None:
                    # Re-read after exit so a final atomic status cannot race the poll.
                    if status_path.exists():
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                    if process.returncode != 0 or status.get("phase") not in (
                        "closed",
                        "close_failed",
                    ):
                        failure = f"PDF renderer exited unexpectedly with code {process.returncode}"
                    break
                phase = status.get("phase")
                if phase in ("closed", "close_failed", "render_failed"):
                    if phase == "closed":
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        if process.returncode not in (0, None):
                            failure = (
                                f"PDF renderer exited unexpectedly with code {process.returncode}"
                            )
                    break
                if time.monotonic() >= deadline:
                    if phase != "closing":
                        failure = "PDF renderer exceeded its progress deadline"
                    break
                time.sleep(0.05)
            if status.get("completed") != len(paths) or status.get("phase") == "render_failed":
                failure = failure or f"PDF rendering failed: {status.get('error', status)}"
        finally:
            try:
                if owner is not None:
                    owner.terminate()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
        if failure:
            output.seek(0)
            raise RuntimeError(f"{failure}\n{output.read().decode('utf-8', errors='replace')}")
        if status.get("phase") != "closed":
            _logger.warning("Terminated owned PDF renderer after browser close failure: %s", status)
