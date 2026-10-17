"""Render pages in one browser, with progress independent of browser shutdown."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import kaleido


async def _render(request_path: Path, status_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # The parent removes both the Chromium profile and Kaleido page after OS termination,
    # even when the worker cannot complete its own close cleanup.
    tempfile.tempdir = str(request_path.parent)
    completed = 0
    sequence = 0

    def status(phase: str, **details: object) -> None:
        nonlocal sequence
        sequence += 1
        destination = status_path.with_name(f"{status_path.stem}-{sequence:06d}.json")
        temporary = destination.with_suffix(".new")
        temporary.write_text(
            json.dumps({"phase": phase, "completed": completed, **details}), encoding="utf-8"
        )
        temporary.replace(destination)

    renderer = kaleido.Kaleido(n=1, timeout=90, **request["defaults"])
    try:
        await renderer.open()
        status("rendering", browser_pid=renderer.subprocess.pid)
        for spec in request["pages"]:
            await renderer.write_fig_from_object([spec], cancel_on_error=True)
            completed += 1
            status("rendering", browser_pid=renderer.subprocess.pid)
    except Exception as error:
        status("render_failed", error=f"{type(error).__name__}: {error}")
        raise
    status("closing", browser_pid=renderer.subprocess.pid)
    try:
        await renderer.close()
    except Exception as error:
        # The parent still owns the OS process tree and must verify its termination.
        status("close_failed", error=f"{type(error).__name__}: {error}")
    else:
        status("closed")


if __name__ == "__main__":
    asyncio.run(_render(Path(sys.argv[1]), Path(sys.argv[2])))
