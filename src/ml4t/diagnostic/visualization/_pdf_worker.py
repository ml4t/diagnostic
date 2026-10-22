"""Render pages in one browser, with progress independent of browser shutdown."""

import asyncio
import json
import sys
import tempfile
from contextlib import suppress
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

    async def pages() -> None:
        nonlocal completed
        await renderer.open()
        status("rendering", browser_pid=renderer.subprocess.pid)
        for spec in request["pages"]:
            await renderer.write_fig_from_object([spec], cancel_on_error=True)
            completed += 1
            status("rendering", browser_pid=renderer.subprocess.pid)

    rendering = asyncio.create_task(pages())

    async def watch_cancellation() -> None:
        while not request_path.with_suffix(".cancel").exists():
            await asyncio.sleep(0.05)
        rendering.cancel()

    cancellation = asyncio.create_task(watch_cancellation())
    render_error = None
    try:
        await rendering
    except asyncio.CancelledError:
        render_error = RuntimeError("PDF rendering cancelled")
    except Exception as error:
        render_error = error
    finally:
        cancellation.cancel()
        with suppress(asyncio.CancelledError):
            await cancellation
    details = (
        {"error": f"{type(render_error).__name__}: {render_error}"}
        if render_error is not None
        else {}
    )
    browser = getattr(renderer, "subprocess", None)
    status("closing", browser_pid=getattr(browser, "pid", None), **details)
    close_error = None
    try:
        await asyncio.wait_for(renderer.close(), timeout=10)
    except Exception as error:
        close_error = error

        # A failed vendor close can leave a healthy browser owning its profile.
        # Request normal browser shutdown before the parent resorts to OS termination.
        async def close_browser() -> None:
            await renderer.send_command("Browser.close")
            while renderer.subprocess.poll() is None:
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(close_browser(), timeout=3)
        except Exception:
            pass
    if render_error is not None:
        if close_error is not None:
            details["close_error"] = f"{type(close_error).__name__}: {close_error}"
        status("render_failed", **details)
        raise render_error
    if close_error is not None:
        # The parent still owns the OS process tree and must verify its termination.
        status("close_failed", error=f"{type(close_error).__name__}: {close_error}")
    else:
        status("closed")


if __name__ == "__main__":
    asyncio.run(_render(Path(sys.argv[1]), Path(sys.argv[2])))
