"""Verify the real suspension fault without assuming every thread has one wait reason."""

import ctypes
import os
from ctypes import wintypes
from typing import Any

import psutil


def observe_suspended(process: psutil.Process) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "pid": process.pid,
        "created": process.create_time(),
        "status": process.status(),
    }
    if os.name != "nt":
        assert observation["status"] == psutil.STATUS_STOPPED, observation
        return observation

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.GetProcessIdOfThread.argtypes = [wintypes.HANDLE]
    api.GetProcessIdOfThread.restype = wintypes.DWORD
    for name in ("SuspendThread", "ResumeThread"):
        getattr(api, name).argtypes = [wintypes.HANDLE]
        getattr(api, name).restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    counts = {}
    threads = {thread.id for thread in process.threads()}
    assert threads, "The suspension control has no live threads"
    for thread in sorted(threads):
        handle = api.OpenThread(0x0802, False, thread)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            assert api.GetProcessIdOfThread(handle) == process.pid, (process.pid, thread)
            previous = api.SuspendThread(handle)
            if previous == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            # Read the documented previous count and immediately restore our increment.
            restored = api.ResumeThread(handle)
            if restored == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            assert restored == previous + 1, (thread, previous, restored)
            assert previous > 0, f"Thread {thread} was not suspended before observation"
            counts[str(thread)] = previous
        finally:
            if not api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
    assert {thread.id for thread in process.threads()} == threads
    observation["thread_suspend_counts"] = counts
    return observation
