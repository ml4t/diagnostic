"""Hold a real Windows native exit event for renderer regressions."""

import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path


class ExceptionRecord(ctypes.Structure):
    _fields_ = [
        ("code", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("record", ctypes.c_void_p),
        ("address", ctypes.c_void_p),
        ("parameter_count", wintypes.DWORD),
        ("parameters", ctypes.c_size_t * 15),
    ]


class ExceptionInfo(ctypes.Structure):
    _fields_ = [("record", ExceptionRecord), ("first_chance", wintypes.DWORD)]


class CreateProcessInfo(ctypes.Structure):
    _fields_ = [
        ("file", wintypes.HANDLE),
        ("process", wintypes.HANDLE),
        ("thread", wintypes.HANDLE),
        ("base", ctypes.c_void_p),
        ("debug_offset", wintypes.DWORD),
        ("debug_size", wintypes.DWORD),
        ("thread_local_base", ctypes.c_void_p),
        ("start_address", ctypes.c_void_p),
        ("image_name", ctypes.c_void_p),
        ("unicode", wintypes.WORD),
    ]


class LoadDllInfo(ctypes.Structure):
    _fields_ = [
        ("file", wintypes.HANDLE),
        ("base", ctypes.c_void_p),
        ("debug_offset", wintypes.DWORD),
        ("debug_size", wintypes.DWORD),
        ("image_name", ctypes.c_void_p),
        ("unicode", wintypes.WORD),
    ]


class DebugInfo(ctypes.Union):
    _fields_ = [
        ("exception", ExceptionInfo),
        ("created", CreateProcessInfo),
        ("loaded", LoadDllInfo),
        ("exit_code", wintypes.DWORD),
    ]


class DebugEvent(ctypes.Structure):
    _fields_ = [
        ("code", wintypes.DWORD),
        ("pid", wintypes.DWORD),
        ("tid", wintypes.DWORD),
        ("info", DebugInfo),
    ]


def hold_exit(pid: int, ready: Path, blocked: Path, release_delay: float | None = None) -> None:
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "DebugActiveProcess": ([wintypes.DWORD], wintypes.BOOL),
        "WaitForDebugEvent": ([ctypes.POINTER(DebugEvent), wintypes.DWORD], wintypes.BOOL),
        "ContinueDebugEvent": ([wintypes.DWORD] * 3, wintypes.BOOL),
        "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        "GetExitCodeProcess": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result

    def check(ok: int) -> None:
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def record(path: Path, values: dict) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(values))
        temporary.replace(path)

    try:
        check(api.DebugActiveProcess(pid))
        process_handle = api.OpenProcess(0x00101000, False, pid)
        check(process_handle)
        while True:
            event = DebugEvent()
            if not api.WaitForDebugEvent(ctypes.byref(event), 1000):
                error = ctypes.get_last_error()
                if error == 121:
                    continue
                raise ctypes.WinError(error)
            status = 0x00010002
            if event.code == 3:
                if event.info.created.file:
                    check(api.CloseHandle(event.info.created.file))
                record(ready, {"pid": pid, "debugger": os.getpid()})
            elif event.code == 6 and event.info.loaded.file:
                check(api.CloseHandle(event.info.loaded.file))
            elif event.code == 1:
                if event.info.exception.record.code != 0x80000003:
                    status = 0x80010001
            elif event.code == 5:
                exit_code = wintypes.DWORD()
                check(api.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)))
                record(
                    blocked,
                    {
                        "pid": pid,
                        "event_exit_code": event.info.exit_code,
                        "process_exit_code": exit_code.value,
                        "wait_result": api.WaitForSingleObject(process_handle, 0),
                    },
                )
                if release_delay is None:
                    # Keep kernel teardown pending while the late job member remains alive.
                    while True:
                        time.sleep(0.1)
                held_since = time.monotonic()
                time.sleep(release_delay)
                record(
                    blocked.with_suffix(".released.json"),
                    {"held_seconds": time.monotonic() - held_since},
                )
                check(api.ContinueDebugEvent(event.pid, event.tid, status))
                terminal = api.WaitForSingleObject(process_handle, 2000)
                check(api.CloseHandle(process_handle))
                if terminal != 0:
                    raise RuntimeError("Released browser exit did not become terminal")
                return
            check(api.ContinueDebugEvent(event.pid, event.tid, status))
    except BaseException as error:
        record(ready, {"error": repr(error), "pid": pid})
        raise


def main() -> None:
    pid, ready, blocked = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    assert sys.stdin.buffer.read(1) == b"1", (
        "Debugger must receive the ownership handshake before attaching"
    )
    hold_exit(pid, ready, blocked, float(sys.argv[4]) if len(sys.argv) > 4 else None)


if __name__ == "__main__":
    main()
