"""Render pages in one browser, with progress independent of browser shutdown."""

import asyncio
import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

import kaleido
from choreographer import Tab
from kaleido._kaleido_tab import _KaleidoTab


class _PageEventsKaleido(kaleido.Kaleido):
    async def _conform_tabs(self, tabs: Iterable[Tab] | None = None) -> None:
        if tabs is None:
            # A startup target can stop answering Page.enable before any navigation.
            previous = list(self.tabs)
            selected = [await self.create_tab(url="about:blank", window=True)]
            for target_id in previous:
                await self.close_tab(target_id)
        else:
            selected = list(tabs)
        # Page.enable does not replay a load event missed during fast navigation.
        for tab in selected:
            await tab.send_command("Page.enable")
        await super()._conform_tabs(selected)

    async def _get_kaleido_tab(self) -> _KaleidoTab:
        wrapped = await super()._get_kaleido_tab()
        await self._renderer_document_ready(wrapped)
        return wrapped

    async def _renderer_document_ready(self, wrapped: _KaleidoTab) -> None:
        target = wrapped.tab
        expression = (
            f"location.href === new URL({json.dumps(self._index)}).href && "
            "document.readyState === 'complete' && "
            "typeof Plotly !== 'undefined' && typeof Plotly.newPlot === 'function' && "
            "typeof kaleido_scopes !== 'undefined' && "
            "typeof kaleido_scopes.plotly === 'function'"
        )
        observation = (
            "JSON.stringify({"
            f"ready: Boolean({expression}), "
            "href: location.href, "
            f"expected_href: new URL({json.dumps(self._index)}).href, "
            "document_state: document.readyState, plotly_type: typeof Plotly, "
            "plotly_function_type: typeof Plotly === 'undefined' ? 'undefined' : "
            "typeof Plotly.newPlot, scopes_type: typeof kaleido_scopes, "
            "scopes_function_type: typeof kaleido_scopes === 'undefined' ? 'undefined' : "
            "typeof kaleido_scopes.plotly})"
        )
        while True:
            response = await target.send_command(
                "Runtime.evaluate", params={"expression": observation, "returnByValue": True}
            )
            if isinstance(response, dict):
                value = response.get("result", {}).get("result", {}).get("value")
                if isinstance(value, str):
                    self._document_readiness = json.loads(value)
                    if self._document_readiness["ready"]:
                        break
            await asyncio.sleep(0.05)

        frame = await target.send_command("Page.getFrameTree")
        if not isinstance(frame, dict) or "error" in frame:
            raise RuntimeError(f"PDF renderer frame query failed: {frame!r}")
        frame_id = frame["result"]["frameTree"]["frame"]["id"]
        await target.send_command("Runtime.disable")
        # A one-shot future drops later contexts emitted before Runtime.enable returns.
        contexts = asyncio.Queue()
        event = "Runtime.executionContextCreated"

        async def context_created(message):
            contexts.put_nowait(message)

        target.subscribe(event, context_created)
        try:
            await target.send_command("Runtime.enable")
            while True:
                current = (await contexts.get())["params"]["context"]
                auxiliary = current.get("auxData", {})
                if auxiliary.get("isDefault") and auxiliary.get("frameId") == frame_id:
                    response = await target.send_command(
                        "Runtime.evaluate",
                        params={
                            "expression": expression,
                            "contextId": current["id"],
                            "returnByValue": True,
                        },
                    )
                    if (
                        isinstance(response, dict)
                        and response.get("result", {}).get("result", {}).get("value") is True
                    ):
                        wrapped._current_js_id = current["id"]
                        return
        finally:
            target.unsubscribe(event)


async def _render(request_path: Path, status_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    browser_logger = logging.getLogger("browser_proc")
    browser_logger.setLevel(logging.DEBUG)
    browser_output = logging.StreamHandler()
    browser_output.setFormatter(logging.Formatter("PDF browser: %(message)s"))
    browser_logger.addHandler(browser_output)
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
            json.dumps(
                {"phase": phase, "completed": completed, "worker_pid": os.getpid(), **details}
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)

    status("starting")
    renderer = _PageEventsKaleido(n=1, timeout=90, **request["defaults"])

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
    readiness = getattr(renderer, "_document_readiness", None)
    if render_error is not None and readiness is not None:
        render_error.add_note(f"PDF document readiness: {json.dumps(readiness)}")
    details = (
        {"error": f"{type(render_error).__name__}: {render_error}"}
        if render_error is not None
        else {}
    )
    browser = getattr(renderer, "subprocess", None)
    status("closing", browser_pid=getattr(browser, "pid", None), **details)
    close_deadline = asyncio.get_running_loop().time() + 13
    close_error = None
    try:
        await asyncio.wait_for(renderer.close(), timeout=10)
    except Exception as error:
        close_error = error

        # A failed vendor close can leave a healthy browser owning its profile.
        # Request normal browser shutdown before the parent resorts to OS termination.
        try:
            try:
                await asyncio.wait_for(
                    renderer.send_command("Browser.close"),
                    # Reserve the rest of the same deadline for pipe closure and process exit.
                    timeout=max(0, close_deadline - asyncio.get_running_loop().time()) / 2,
                )
            finally:
                renderer._channel.close()
        except Exception:
            pass
        while (
            browser is not None
            and browser.poll() is None
            and asyncio.get_running_loop().time() < close_deadline
        ):
            await asyncio.sleep(0.05)
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
