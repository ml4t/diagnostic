"""Test whether suspended pending pipe I/O alone delays owned native process exit."""

from __future__ import annotations

import ctypes
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path


def peer(mode: str, directory: Path) -> None:
    with (directory / "owned-profile-file").open("wb") as profile:
        profile.write(b"Native peer holds this file until kernel teardown")
        profile.flush()
        ready = directory / "peer.json"
        temporary = ready.with_suffix(".new")
        temporary.write_text(json.dumps({"pid": os.getpid()}))
        temporary.replace(ready)
        if mode == "read":
            os.read(0, 1)
        else:
            os.write(1, b"x" * (1024 * 1024))
        raise AssertionError("The parent must stop the genuinely blocked native peer")


def worker(mode: str, directory: Path) -> None:
    assert sys.stdin.buffer.read(1) == b"1"
    child = subprocess.Popen(
        [sys.executable, __file__, "peer", mode, str(directory)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert child.stdin is not None and child.stdout is not None
    sys.stdin.buffer.read(1)
    raise AssertionError("The owned job must terminate the worker without releasing its pipes")


def debug_port(api, pid: int) -> int:
    from ctypes import wintypes

    query = ctypes.WinDLL("ntdll").NtQueryInformationProcess
    query.argtypes = [
        wintypes.HANDLE,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_void_p,
    ]
    query.restype = wintypes.LONG
    handle = api.OpenProcess(0x0400, False, pid)
    assert handle, ctypes.WinError(ctypes.get_last_error())
    try:
        value = ctypes.c_size_t()
        assert query(handle, 7, ctypes.byref(value), ctypes.sizeof(value), None) == 0
        return value.value
    finally:
        api.CloseHandle(handle)


def run_case(mode: str, suspended: bool, directory: Path) -> dict:
    import psutil

    from ml4t.diagnostic.visualization._pdf_process import _WindowsJob

    directory.mkdir(parents=True)
    trace = runpy.run_path(str(Path("tests/test_visualization/_windows_pdf_trace.py")))
    pending_io = trace["_pending_io"]
    observe_suspended = runpy.run_path("tests/test_visualization/_pdf_suspension.py")[
        "observe_suspended"
    ]
    row = {"mode": mode, "suspended": suspended, "events": [], "passed": False}
    neighbor = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"])
    root = subprocess.Popen(
        [sys.executable, __file__, "worker", mode, str(directory.resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    job = None
    peer_process = None
    try:
        job = _WindowsJob(root.pid)
        api = job.api
        assert root.stdin is not None
        root.stdin.write(b"1")
        root.stdin.flush()
        ready = directory / "peer.json"
        deadline = time.monotonic() + 8
        while not ready.exists():
            assert time.monotonic() < deadline and root.poll() is None
            time.sleep(0.01)
        pid = json.loads(ready.read_text())["pid"]
        peer_process = psutil.Process(pid)
        while True:
            io = pending_io(pid)
            if any(thread.get("io_pending") for thread in io):
                break
            assert time.monotonic() < deadline, io
            time.sleep(0.01)
        row.update(peer_pid=pid, peer_created=peer_process.create_time(), pending_before=io)
        assert debug_port(api, pid) == 0, "An external debugger would confound this mechanism"
        if suspended:
            peer_process.suspend()
            io = pending_io(pid)
            row["pending_after_suspend"] = io
            assert any(thread.get("io_pending") for thread in io), io
            row["suspension"] = observe_suspended(peer_process)
            assert all(count == 1 for count in row["suspension"]["thread_suspend_counts"].values())
        from ctypes import wintypes

        member = wintypes.BOOL()
        handle = api.OpenProcess(0x00101000, False, neighbor.pid)
        assert handle
        try:
            assert api.IsProcessInJob(handle, job.handle, ctypes.byref(member))
            assert not member.value, "Neighbor must remain outside the owned job"
        finally:
            api.CloseHandle(handle)
        try:
            (directory / "owned-profile-file").unlink()
        except PermissionError as error:
            assert error.winerror == 32
            row["profile_locked_before"] = error.winerror
        else:
            raise AssertionError("Real peer file ownership was not established")

        class ObservedApi:
            def __getattr__(self, name):
                if name not in ("TerminateProcess", "TerminateJobObject", "WaitForSingleObject"):
                    return getattr(api, name)

                def call(*args):
                    started = time.monotonic()
                    result = getattr(api, name)(*args)
                    row["events"].append(
                        {
                            "call": name,
                            "result": int(result),
                            "started": started,
                            "elapsed": time.monotonic() - started,
                            "wait_ms": args[1] if name == "WaitForSingleObject" else None,
                        }
                    )
                    return result

                return call

        job.api = ObservedApi()
        started = time.monotonic()
        try:
            job.terminate([psutil.Process(root.pid), peer_process], worker_exited=False)
        finally:
            row["cleanup_elapsed"] = time.monotonic() - started
            job = None  # Production terminate closes its job handle even on failure.
        assert root.wait(timeout=5) == 120
        assert neighbor.poll() is None
        (directory / "owned-profile-file").unlink()
        row.update(passed=True, profile_removed=True, neighbor_alive=True, debug_port=0)
    except Exception as error:
        row["error"] = repr(error)
    finally:
        if job is not None:
            job.api.TerminateJobObject(job.handle, 120)
            job.api.CloseHandle(job.handle)
        if root.poll() is None:
            root.kill()
        try:
            root.wait(timeout=5)
        except subprocess.TimeoutExpired:
            row["worker_still_alive"] = True
        if root.stdin is not None:
            root.stdin.close()
        neighbor.kill()
        neighbor.wait(timeout=5)
        (directory / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("peer", "worker"):
        {"peer": peer, "worker": worker}[sys.argv[1]](sys.argv[2], Path(sys.argv[3]))
        return
    assert sys.platform == "win32", "This diagnostic requires actual Windows kernel behavior"
    output = Path("artifacts/native-pipe-mechanism")
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(mode, suspended, output / f"{mode}-suspended-{int(suspended)}")
        for mode in ("read", "write")
        for suspended in (False, True)
    ]
    result = {
        "python": sys.version,
        "source_revision": os.environ.get("GITHUB_SHA"),
        "cases": cases,
        "scope": "Native pipe/job mechanism only; not an attribution of historical Chrome failures",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(case["passed"] for case in cases) else 1)


if __name__ == "__main__":
    main()
