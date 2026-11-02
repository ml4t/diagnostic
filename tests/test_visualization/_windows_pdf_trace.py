"""Opt-in native observations for the real PDF cleanup probe, without changing its verdict."""

import ctypes
import json
import os
import struct
import time
from ctypes import wintypes
from pathlib import Path

import psutil


def _threads(pids):
    # SystemProcessInformation and SYSTEM_THREAD_INFORMATION, Windows x64 ABI.
    # Validate the returned process and thread identities before retaining observations.
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
    query.argtypes = [wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p]
    query.restype = wintypes.LONG
    size = 1024 * 1024
    while True:
        buffer = ctypes.create_string_buffer(size)
        required = wintypes.ULONG()
        status = query(5, buffer, size, ctypes.byref(required))
        if status == 0:
            break
        if status & 0xFFFFFFFF != 0xC0000004:
            raise OSError(f"NtQuerySystemInformation returned {status & 0xFFFFFFFF:#x}")
        size = max(size * 2, required.value)
        assert size <= 64 * 1024 * 1024
    data = buffer.raw
    result = {}
    offset = 0
    while True:
        following, count = struct.unpack_from("<II", data, offset)
        pid = struct.unpack_from("<Q", data, offset + 80)[0]
        if pid in pids:
            rows = []
            for index in range(count):
                start = offset + 256 + index * 80
                process, thread = struct.unpack_from("<QQ", data, start + 40)
                switches, state, reason = struct.unpack_from("<III", data, start + 64)
                assert process == pid, (pid, process, thread)
                rows.append(
                    {"tid": thread, "state": state, "wait_reason": reason, "switches": switches}
                )
            result[pid] = rows
        if not following:
            return result
        offset += following


def _record(path, event, pids, **extra):
    try:
        states = _threads(set(pids))
        processes = []
        for pid in pids:
            row = {"pid": pid, "native_threads": states.get(pid, [])}
            try:
                process = psutil.Process(pid)
                row.update(
                    created=process.create_time(),
                    name=process.name(),
                    status=process.status(),
                    threads=[thread._asdict() for thread in process.threads()],
                )
            except psutil.Error as error:
                row["error"] = repr(error)
            processes.append(row)
        row = {"event": event, "time": time.monotonic(), "processes": processes, **extra}
    except Exception as error:
        row = {"event": event, "time": time.monotonic(), "trace_error": repr(error), **extra}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def _pending_io(pid):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.GetThreadIOPendingFlag.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    api.GetThreadIOPendingFlag.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    rows = []
    for thread in _threads({pid}).get(pid, []):
        access = 0x0048 if os.environ.get("ML4T_PDF_NATIVE_HANDLE_TRACE") == "1" else 0x0040
        handle = api.OpenThread(access, False, thread["tid"])
        row = dict(thread)
        if not handle:
            row["open_error"] = ctypes.get_last_error()
        else:
            try:
                pending = wintypes.BOOL()
                if api.GetThreadIOPendingFlag(handle, ctypes.byref(pending)):
                    row["io_pending"] = bool(pending.value)
                else:
                    row["query_error"] = ctypes.get_last_error()
                if os.environ.get("ML4T_PDF_NATIVE_HANDLE_TRACE") == "1":
                    row["last_syscall"] = _last_syscall(handle)
            finally:
                api.CloseHandle(handle)
        rows.append(row)
    return rows


def _last_syscall(thread):
    # Diagnostic-only Windows x64 layout from winsiderss/phnt ntpsapi.h.
    class LastCall(ctypes.Structure):
        _fields_ = [
            ("first_argument", ctypes.c_void_p),
            ("number", wintypes.USHORT),
            ("wait_time", ctypes.c_uint64),
        ]

    ntdll = ctypes.WinDLL("ntdll")
    query = ntdll.NtQueryInformationThread
    query.argtypes = [
        wintypes.HANDLE,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_void_p,
    ]
    query.restype = wintypes.LONG
    value = LastCall()
    status = query(thread, 21, ctypes.byref(value), ctypes.sizeof(value), None)
    result = {
        "status": status,
        "first_argument": value.first_argument,
        "number": value.number,
        "wait_time": value.wait_time,
        "matching_exports": [],
    }
    for name in (
        "NtReadFile",
        "NtWriteFile",
        "NtFsControlFile",
        "NtDeviceIoControlFile",
        "NtClose",
        "NtSetInformationFile",
        "NtFlushBuffersFile",
        "NtWaitForSingleObject",
        "NtTerminateProcess",
    ):
        stub = ctypes.string_at(ctypes.cast(getattr(ntdll, name), ctypes.c_void_p), 8)
        if stub[:4] == bytes.fromhex("4c8bd1b8"):
            number = int.from_bytes(stub[4:8], "little")
            if number == value.number:
                result["matching_exports"].append(name)
    return result


def install_worker(observed):
    if os.environ.get("ML4T_PDF_NATIVE_HANDLE_TRACE") == "1":
        import _winapi

        path = Path(observed).with_suffix(".native-handles.json")
        actual = _winapi.CreateProcess

        def create(*args, **kwargs):
            if "--remote-debugging-pipe" not in str(args[1]):
                return actual(*args, **kwargs)
            startup = args[8]
            row = {
                "worker_pid": os.getpid(),
                "command": args[1],
                "inherit_handles": args[4],
                "stdin": startup.hStdInput,
                "stdout": startup.hStdOutput,
                "stderr": startup.hStdError,
            }
            path.write_text(json.dumps(row))
            result = actual(*args, **kwargs)
            row.update(browser_pid=result[2], browser_main_thread=result[3])
            path.write_text(json.dumps(row))
            return result

        _winapi.CreateProcess = create
    if os.environ.get("ML4T_PDF_NATIVE_TRACE") == "timeout":
        return
    observed = Path(observed)
    path = observed.with_suffix(".native-worker.jsonl")
    for name in ("suspend", "resume"):
        original = getattr(psutil.Process, name)

        def call(self, _original=original, _name=name):
            _record(path, "before_" + _name, [self.pid])
            try:
                return _original(self)
            finally:
                _record(path, "after_" + _name, [self.pid])

        setattr(psutil.Process, name, call)


def install_parent(module, observed):
    path = observed.with_suffix(".native-parent.jsonl")
    actual = module._WindowsJob.terminate
    timeout_only = os.environ.get("ML4T_PDF_NATIVE_TRACE") == "timeout"

    def terminate(self, processes, *, worker_exited):
        pids = {process.pid for process in processes}
        if not timeout_only:
            _record(path, "before_owned_cleanup", sorted(pids), worker_exited=worker_exited)
        native = self.api
        native.GetProcessId.argtypes = [wintypes.HANDLE]
        native.GetProcessId.restype = wintypes.DWORD
        query = ctypes.WinDLL("ntdll").NtQueryInformationProcess
        query.argtypes = [
            wintypes.HANDLE,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
        ]
        query.restype = wintypes.LONG

        class ObservedApi:
            def __getattr__(self, name):
                if name == "TerminateJobObject":
                    return self.terminate_job
                if name == "WaitForSingleObject":
                    return self.wait
                if (
                    name == "TerminateProcess"
                    and os.environ.get("ML4T_PDF_NATIVE_HANDLE_TRACE") == "1"
                ):
                    return self.terminate_process
                return getattr(native, name)

            def terminate_process(self, handle, code):
                pid = native.GetProcessId(handle)
                try:
                    launch = json.loads(observed.with_suffix(".native-handles.json").read_text())
                    if pid == launch["browser_pid"]:
                        _record(
                            path, "before_browser_termination", [pid], thread_io=_pending_io(pid)
                        )
                except Exception as error:
                    _record(
                        path, "before_browser_termination_observer_error", [pid], error=repr(error)
                    )
                return native.TerminateProcess(handle, code)

            def terminate_job(self, handle, code):
                if not timeout_only:
                    _record(path, "before_job_termination", sorted(pids))
                return native.TerminateJobObject(handle, code)

            def wait(self, handle, milliseconds):
                result = native.WaitForSingleObject(handle, milliseconds)
                if milliseconds and result == 258:
                    pid = native.GetProcessId(handle)
                    debug_port = ctypes.c_size_t()
                    debug_handle = native.OpenProcess(0x0400, False, pid)
                    if debug_handle:
                        try:
                            debug_status = query(
                                debug_handle,
                                7,
                                ctypes.byref(debug_port),
                                ctypes.sizeof(debug_port),
                                None,
                            )
                        finally:
                            native.CloseHandle(debug_handle)
                    else:
                        debug_status = {"open_error": ctypes.get_last_error()}
                    try:
                        pending = _pending_io(pid)
                    except Exception as error:
                        pending = {"query_error": repr(error)}
                    _record(
                        path,
                        "native_wait_timeout",
                        [pid],
                        wait_ms=milliseconds,
                        debug_query_status=debug_status,
                        debug_port=debug_port.value,
                        thread_io=pending,
                    )
                return result

        self.api = ObservedApi()
        try:
            return actual(self, processes, worker_exited=worker_exited)
        finally:
            self.api = native
            if not timeout_only:
                _record(path, "after_owned_cleanup", sorted(pids))

    module._WindowsJob.terminate = terminate
    helper = Path(__file__).resolve()
    module._BOOTSTRAP = (
        f"import runpy; runpy.run_path({str(helper)!r})['install_worker']({str(observed)!r})\n"
        + module._BOOTSTRAP
    )


def instrument(code):
    assert os.name == "nt"
    anchor = "neighbor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(180)'])"
    assert code.count(anchor) == 1
    setup = (
        f"\nimport runpy; runpy.run_path({str(Path(__file__).resolve())!r})"
        "['install_parent'](_pdf_process,observed)\n"
    )
    return code.replace(anchor, setup + anchor)
