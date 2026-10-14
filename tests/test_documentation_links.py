"""Exercise external documentation checks through the CLI and a real HTTP endpoint."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_documentation_links.py"


def _run_check(
    tmp_path: Path, statuses: tuple[int, ...]
) -> tuple[subprocess.CompletedProcess[str], list[int]]:
    responses: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = statuses[min(len(responses), len(statuses) - 1)]
            responses.append(status)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        script = scripts / SCRIPT.name
        shutil.copy2(SCRIPT, script)
        url = f"http://127.0.0.1:{server.server_port}/guide"
        (tmp_path / "README.md").write_text(f"[Guide]({url})\n", encoding="utf-8")
        site = tmp_path / "site"
        site.mkdir()
        (site / "index.html").write_text(
            f'<article><a href="{url}">Guide</a></article>', encoding="utf-8"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--site", str(site), "--external"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return completed, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_checker_recovers_from_temporary_503(tmp_path: Path) -> None:
    completed, responses = _run_check(tmp_path, (503, 200))

    assert completed.returncode == 0, completed.stderr
    assert responses == [503, 200]


def test_checker_does_not_retry_missing_external_page(tmp_path: Path) -> None:
    completed, responses = _run_check(tmp_path, (404, 200))

    assert completed.returncode != 0
    assert "HTTP Error 404" in completed.stderr
    assert responses == [404]


def test_checker_fails_after_bounded_service_unavailable_retries(tmp_path: Path) -> None:
    completed, responses = _run_check(tmp_path, (503,))

    assert completed.returncode != 0
    assert "HTTP Error 503" in completed.stderr
    assert responses == [503, 503, 503]
