"""Own the PDF worker and its browser independently of Kaleido's close path."""

import ctypes
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil

_BOOTSTRAP = """
import asyncio,os,sys,runpy,threading,traceback
assert sys.stdin.buffer.read(1)==b'1'
diagnostic_stop=threading.Event()
def snapshot_threads():
    while not diagnostic_stop.wait(20):
        lines=['PDF worker thread snapshot\\n']
        loops=set()
        for identifier,frame in sys._current_frames().items():
            lines.append(f'Thread {identifier}:\\n')
            lines.extend(traceback.format_stack(frame))
            cursor=frame
            while cursor is not None:
                candidate=cursor.f_locals.get('self')
                if isinstance(candidate,asyncio.AbstractEventLoop): loops.add(candidate)
                cursor=cursor.f_back
        for loop in loops:
            try:
                for task in asyncio.all_tasks(loop):
                    lines.append(f'Async task {task.get_name()}:\\n')
                    awaiting=task.get_coro()
                    while awaiting is not None:
                        frame=getattr(awaiting,'cr_frame',getattr(awaiting,'gi_frame',None))
                        if frame is not None: lines.extend(traceback.format_stack(frame,limit=1))
                        awaiting=getattr(awaiting,'cr_await',getattr(awaiting,'gi_yieldfrom',None))
            except Exception as error:
                lines.append(f'Async task snapshot failed: {error!r}\\n')
        os.write(2,''.join(lines).encode('utf-8',errors='backslashreplace'))
watcher=threading.Thread(target=snapshot_threads,daemon=True)
watcher.start()
try:
    worker=sys.argv.pop(1)
    runpy.run_path(worker,run_name='__main__')
finally:
    diagnostic_stop.set()
    watcher.join(timeout=1)
"""
_RENDER_TIMEOUT = 90
_CLOSE_TIMEOUT = 15
_STATUS_READ_TIMEOUT = 2.0
_logger = logging.getLogger(__name__)


def _remove_readonly_file(function: Callable[..., Any], path: str, error: BaseException) -> None:
    if not (
        sys.platform == "win32"
        and function is os.unlink
        and isinstance(error, PermissionError)
        and getattr(error, "winerror", None) == 5
    ):
        raise error
    try:
        metadata = os.lstat(path)
        if not (
            stat.S_ISREG(metadata.st_mode)
            and getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY
        ):
            raise error
        os.chmod(path, metadata.st_mode | stat.S_IWRITE)
        function(path)
    except FileNotFoundError:
        return


@contextmanager
def _renderer_directory():
    directory = Path(tempfile.mkdtemp())
    primary = None
    try:
        yield directory
    except BaseException as error:
        primary = error
        raise
    finally:
        deadline = time.monotonic() + 2
        while True:
            try:
                shutil.rmtree(directory, onexc=_remove_readonly_file)
                break
            except OSError as error:
                if (
                    isinstance(error, PermissionError)
                    and sys.platform == "win32"
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                    continue
                if primary is None:
                    raise
                primary.add_note(f"PDF profile cleanup failed; retained {directory}: {error!r}")
                break


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
            "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            "IsProcessInJob": (
                [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)],
                wintypes.BOOL,
            ),
            "GetExitCodeProcess": (
                [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
                wintypes.BOOL,
            ),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes = args
            function.restype = result
        self.worker_pids = {pid}
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

    def terminate(self, processes: list[psutil.Process], *, worker_exited: bool) -> None:
        from ctypes import wintypes

        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("page_faults", wintypes.DWORD),
                ("total", wintypes.DWORD),
                ("active", wintypes.DWORD),
                ("terminated", wintypes.DWORD),
            ]

        handles = []
        try:
            started = time.monotonic()
            deadline = started + 5
            # Kernel job membership remains authoritative while the worker is alive.
            capacity = 64
            while True:

                class ProcessIds(ctypes.Structure):
                    _fields_ = [
                        ("assigned", wintypes.DWORD),
                        ("listed", wintypes.DWORD),
                        ("pids", ctypes.c_size_t * capacity),
                    ]

                members = ProcessIds()
                complete = self.api.QueryInformationJobObject(
                    self.handle, 3, ctypes.byref(members), ctypes.sizeof(members), None
                )
                if complete and members.listed >= members.assigned:
                    break
                error = ctypes.get_last_error()
                if not complete and error != 234:
                    raise ctypes.WinError(error)
                if time.monotonic() >= deadline:
                    raise RuntimeError("PDF job membership did not stabilize before cleanup")
                capacity = max(capacity * 2, members.assigned)
            running_pids = {process.pid for process in processes if process.is_running()}
            processes = list(processes)
            for pid in members.pids[: members.listed]:
                if pid not in running_pids:
                    try:
                        processes.append(psutil.Process(pid))
                    except psutil.NoSuchProcess:
                        pass
            for process in processes:
                if not process.is_running():
                    continue
                handle = self.api.OpenProcess(
                    0x00101001, False, process.pid
                )  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION | TERMINATE
                can_terminate = bool(handle)
                if not handle and ctypes.get_last_error() == 5:
                    # Keep a waitable identity handle when termination access is denied.
                    handle = self.api.OpenProcess(0x00101000, False, process.pid)
                if handle:
                    member = wintypes.BOOL()
                    membership = (
                        bool(member.value)
                        if self.api.IsProcessInJob(handle, self.handle, ctypes.byref(member))
                        else None
                    )
                    identity = {}
                    try:
                        identity["name"] = process.name()
                        identity["role"] = next(
                            (arg for arg in process.cmdline() if arg.startswith("--type=")),
                            "main",
                        )
                    except psutil.Error:
                        pass
                    handles.append((process.pid, handle, membership, identity, can_terminate))
                elif ctypes.get_last_error() != 87:  # PID already gone
                    raise ctypes.WinError(ctypes.get_last_error())
            job_terminated = False

            def request_termination(item):
                _, handle, membership, _, can_terminate = item
                state = self.api.WaitForSingleObject(handle, 0)
                if state == 258 and membership and can_terminate:
                    if not self.api.TerminateProcess(handle, 120):
                        error = ctypes.get_last_error()
                        # A repeated termination request can be denied during exit.
                        # Only the following process-handle wait establishes completion.
                        if error != 5 and self.api.WaitForSingleObject(handle, 0) != 0:
                            raise ctypes.WinError(error)

            if not worker_exited:
                # Native exit can depend on another owned process being stopped first.
                for item in reversed(handles):
                    if item[0] not in self.worker_pids:
                        request_termination(item)
            # A descendant can depend on resources retained by the worker itself.
            # Stop the complete owned job before waiting on any member's native exit.
            self._check(self.api.TerminateJobObject(self.handle, 120))
            job_terminated = True
            for pid, handle, membership, identity, can_terminate in handles:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                result = self.api.WaitForSingleObject(handle, remaining_ms)
                if result != 0:
                    exit_code = wintypes.DWORD()
                    observed_exit = (
                        exit_code.value
                        if self.api.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                        else None
                    )
                    raise RuntimeError(
                        f"Owned PDF process {pid} not terminal "
                        f"{'after' if job_terminated else 'before'} job termination: "
                        f"wait result {result}; identity={identity}, "
                        f"member_before_termination={membership}, exit_code={observed_exit}, "
                        f"termination_access={can_terminate}, "
                        f"cleanup_elapsed={time.monotonic() - started:.6f}, "
                        f"wait_budget_ms={remaining_ms}"
                    )
            while True:
                accounting = Accounting()
                self._check(
                    self.api.QueryInformationJobObject(
                        self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
                    )
                )
                if accounting.active == 0:
                    job_zero_elapsed = time.monotonic() - started
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "PDF renderer job still has active processes after termination"
                    )
                time.sleep(0.05)
            # Accounting can reach zero before process teardown releases file handles.
            for pid, handle, membership, identity, _ in handles:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                result = self.api.WaitForSingleObject(handle, remaining_ms)
                if result != 0:
                    exit_code = wintypes.DWORD()
                    observed_exit = (
                        exit_code.value
                        if self.api.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                        else None
                    )
                    raise RuntimeError(
                        f"Owned PDF process {pid} not terminal: wait result {result}; "
                        f"identity={identity}, "
                        f"member_before_termination={membership}, exit_code={observed_exit}, "
                        f"job_active={accounting.active}, job_zero_elapsed={job_zero_elapsed:.6f}, "
                        f"cleanup_elapsed={time.monotonic() - started:.6f}, wait_budget_ms={remaining_ms}"
                    )
        finally:
            for _, handle, _, _, _ in handles:
                self.api.CloseHandle(handle)
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

    def observe_worker(self, pid: int) -> None:
        if self.job is None or pid in self.job.worker_pids:
            return
        try:
            worker = psutil.Process(pid)
            lineage = [worker, *worker.parents()]
            root = next((index for index, item in enumerate(lineage) if item == self.root), None)
            if root is None:
                raise RuntimeError("PDF interpreter is not a child of its owned launcher")
            for item in lineage[: root + 1]:
                self.observed[item.pid] = item
                self.job.worker_pids.add(item.pid)
        except psutil.NoSuchProcess:
            # Final progress can be read after the interpreter has already exited.
            pass

    def terminate(self) -> None:
        self.observe()
        if self.job is not None:
            self.job.terminate(
                list(self.observed.values()), worker_exited=self.process.poll() is not None
            )
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


def _read_status(path: Path) -> dict[str, Any]:
    """Read one published record, with bounded Windows sharing-error recovery.

    Publication is already atomic, but a scanner can temporarily deny readers.
    Persistent denial remains the original PermissionError. Failed attempts do
    not advance the sequence or reset the renderer's progress deadline.
    """
    deadline = time.monotonic() + _STATUS_READ_TIMEOUT
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if sys.platform != "win32" or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def render_pdf_pages(figures: list[Any], paths: list[Path], **options: Any) -> None:
    """Return only after all pages render and the owned worker tree is terminal."""
    import plotly.io as pio
    from plotly.utils import PlotlyJSONEncoder

    def resource(value: Any) -> Any:
        return os.fspath(value) if isinstance(value, os.PathLike) else value

    request = {
        "defaults": {
            name: resource(value)
            for name in ("plotlyjs", "mathjax", "headers")
            if (value := getattr(pio.defaults, name, None)) is not None
        },
        "pages": [
            {
                "fig": figure.to_plotly_json(),
                "path": str(path),
                "opts": {
                    "format": "pdf",
                    **{
                        name: value or getattr(pio.defaults, f"default_{name}")
                        for name, value in options.items()
                    },
                },
                "topojson": resource(getattr(pio.defaults, "topojson", None)),
            }
            for figure, path in zip(figures, paths, strict=True)
        ],
    }
    with _renderer_directory() as directory, tempfile.TemporaryFile(mode="w+b") as output:
        request_path = Path(directory) / "request.json"
        status_path = Path(directory) / "status.json"
        request_path.write_text(json.dumps(request, cls=PlotlyJSONEncoder), encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _BOOTSTRAP,
                str(Path(__file__).with_name("_pdf_worker.py")),
                str(request_path),
                str(status_path),
            ],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=output,
            start_new_session=sys.platform != "win32",
        )
        owner = None
        status: dict[str, Any] = {}
        failure = None
        primary_error = None
        cleanup_error = None
        sequence = 0

        def read_progress() -> None:
            nonlocal status, sequence
            while True:
                next_path = status_path.with_name(f"{status_path.stem}-{sequence + 1:06d}.json")
                if not next_path.exists():
                    return
                status = _read_status(next_path)
                if owner is not None and (worker_pid := status.get("worker_pid")) is not None:
                    owner.observe_worker(worker_pid)
                sequence += 1

        try:
            owner = _OwnedProcesses(process)
            assert process.stdin is not None
            process.stdin.write(b"1")
            process.stdin.close()
            previous = None
            deadline = time.monotonic() + _RENDER_TIMEOUT
            while True:
                owner.observe()
                read_progress()
                progress = (status.get("phase"), status.get("completed"))
                if progress != previous:
                    deadline = time.monotonic() + (
                        _CLOSE_TIMEOUT if status.get("phase") == "closing" else _RENDER_TIMEOUT
                    )
                    previous = progress
                if process.poll() is not None:
                    # Re-read after exit so a final atomic status cannot race the poll.
                    read_progress()
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
                        failure = (
                            "PDF renderer exceeded its progress deadline: "
                            f"phase={phase!r}, completed={status.get('completed', 0)}, "
                            f"expected={len(paths)}, worker_pid={process.pid}"
                        )
                    break
                time.sleep(0.05)
            if status.get("completed") != len(paths) or status.get("phase") == "render_failed":
                failure = failure or f"PDF rendering failed: {status.get('error', status)}"
            if failure:
                raise RuntimeError(failure)
        except BaseException as error:
            primary_error = error
            if owner is not None and process.poll() is None:
                try:
                    request_path.with_suffix(".cancel").touch()
                    process.wait(timeout=_CLOSE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass
                except Exception as cancel_error:
                    primary_error.add_note(f"PDF graceful cancellation failed: {cancel_error!r}")
        finally:
            try:
                if owner is not None:
                    owner.terminate()
            except BaseException as error:
                cleanup_error = error
            finally:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    else:
                        cleanup_error.add_note(f"PDF worker reap failed: {error!r}")
                finally:
                    if process.stdin is not None and not process.stdin.closed:
                        process.stdin.close()
        if failure:
            output.seek(0)
            notes = getattr(primary_error, "__notes__", ())
            primary_error = RuntimeError(
                f"{failure}\n{output.read().decode('utf-8', errors='replace')}"
            )
            for note in notes:
                primary_error.add_note(note)
        if primary_error is not None:
            if cleanup_error is not None:
                primary_error.add_note(f"PDF process cleanup failed: {cleanup_error!r}")
            raise primary_error from cleanup_error
        if cleanup_error is not None:
            output.seek(0)
            cleanup_error.add_note(
                f"PDF worker final status: {status!r}\n"
                f"{output.read().decode('utf-8', errors='replace')}"
            )
            raise cleanup_error
        if status.get("phase") != "closed":
            _logger.warning("Terminated owned PDF renderer after browser close failure: %s", status)
