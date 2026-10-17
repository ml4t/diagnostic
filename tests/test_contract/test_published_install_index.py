"""Exercise release installation against actual wheels and a changing simple index."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/install_published_package.py"


@pytest.fixture(scope="module")
def produced_wheels(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-index-wheels")
    project = root / "project"
    module = project / "release_index_fixture"
    module.mkdir(parents=True)
    wheels = {}
    for version in ("1.2.2", "1.2.3"):
        (project / "pyproject.toml").write_text(
            '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
            f'[project]\nname="ml4t-diagnostic"\nversion="{version}"\n'
            '[tool.hatch.build.targets.wheel]\npackages=["release_index_fixture"]\n'
        )
        (module / "__init__.py").write_text(f'BUILD_VERSION = "{version}"\n')
        built = subprocess.run(
            ["uv", "build", "--offline", "--wheel", "--out-dir", str(root / version)],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert built.returncode == 0, built.stderr
        (wheel,) = (root / version).glob("*.whl")
        wheels[version] = (wheel.name, wheel.read_bytes())
    return wheels


@contextmanager
def _index(wheels, scenario):
    state = {"index_requests": 0}
    payloads = dict(wheels.values())
    release_stalled_request = threading.Event()

    class Index(BaseHTTPRequestHandler):
        def do_HEAD(self):
            body = payloads.get(self.path.removeprefix("/packages/"))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()

        def do_GET(self):
            if self.path == "/pypi/ml4t-diagnostic/1.2.3/json":
                name, data = wheels["1.2.3"]
                body = json.dumps(
                    {
                        "info": {"version": "1.2.3"},
                        "urls": [
                            {
                                "filename": name,
                                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                            }
                        ],
                    }
                ).encode()
                mime = "application/json"
            elif self.path == "/simple/ml4t-diagnostic/":
                state["index_requests"] += 1
                if scenario == "stall":
                    release_stalled_request.wait(timeout=15)
                versions = (
                    ["1.2.2"]
                    if scenario == "missing" or state["index_requests"] == 1
                    else ["1.2.2", "1.2.3"]
                )
                body = (
                    "<!doctype html>"
                    + "".join(
                        f'<a href="/packages/{wheels[v][0]}#sha256='
                        f'{hashlib.sha256(wheels[v][1]).hexdigest()}">{wheels[v][0]}</a>'
                        for v in versions
                    )
                ).encode()
                mime = "text/html"
            elif self.path.startswith("/packages/"):
                body = payloads.get(self.path.removeprefix("/packages/"))
                if body is None:
                    self.send_error(404)
                    return
                if scenario == "corrupt" and "1.2.3" in self.path:
                    body = b"corrupted artifact bytes"
                mime = "application/octet-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                if scenario != "stall":
                    raise

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Index)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        release_stalled_request.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _python(tmp_path):
    environment = tmp_path / "venv"
    created = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert created.returncode == 0, created.stderr
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _install(python, origin, attempts, *, timeout=120):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--python",
            str(python),
            "--version",
            "1.2.3",
            "--index-url",
            f"{origin}/simple",
            "--attempts",
            str(attempts),
            "--delay",
            "0",
            "--timeout",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.parametrize("old_installed", [False, True], ids=["fresh", "older-version"])
def test_install_retries_real_index_visibility_even_after_json_is_available(
    produced_wheels, tmp_path, old_installed
):
    python = _python(tmp_path)
    if old_installed:
        name, data = produced_wheels["1.2.2"]
        wheel = tmp_path / name
        wheel.write_bytes(data)
        prior = subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-cache", str(wheel)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert prior.returncode == 0, prior.stderr
    with _index(produced_wheels, "delayed") as (origin, state):
        with urllib.request.urlopen(
            f"{origin}/pypi/ml4t-diagnostic/1.2.3/json", timeout=5
        ) as response:
            assert json.load(response)["info"]["version"] == "1.2.3"
        result = _install(python, origin, attempts=3)
        assert result.returncode == 0, result.stderr
        assert state["index_requests"] == 2
    smoke = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; "
            "from release_index_fixture import BUILD_VERSION; "
            "assert version('ml4t-diagnostic') == BUILD_VERSION == '1.2.3'",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert smoke.returncode == 0, smoke.stderr


@pytest.mark.parametrize("scenario", ["missing", "corrupt"])
def test_install_still_fails_for_missing_or_corrupt_release(produced_wheels, tmp_path, scenario):
    python = _python(tmp_path)
    with _index(produced_wheels, scenario) as (origin, state):
        result = _install(python, origin, attempts=3)
        assert result.returncode != 0
        assert state["index_requests"] == 3
        if scenario == "corrupt":
            assert "invalid" in result.stderr.lower() or "hash" in result.stderr.lower()
        else:
            assert "no version" in result.stderr
    absent = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata; importlib.metadata.version('ml4t-diagnostic')",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert absent.returncode != 0, "Failed installation must not claim the target is installed"


def test_stalled_real_installer_is_terminated_without_starting_overlapping_attempts(
    produced_wheels, tmp_path
):
    python = _python(tmp_path)
    with _index(produced_wheels, "stall") as (origin, state):
        result = _install(python, origin, attempts=3, timeout=2)
        assert state["index_requests"] == 1
        assert result.returncode == 124
        assert "timed out" in result.stdout
        assert "attempt 2/" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--attempts", "0"],
        ["--delay", "nan"],
        ["--timeout", "inf"],
        ["--timeout", "0"],
        ["--version", ">=1.2.3"],
    ],
    ids=["no-attempts", "nan-delay", "infinite-timeout", "zero-timeout", "version-range"],
)
def test_invalid_retry_bounds_or_unpinned_version_fail_before_installer(
    produced_wheels, tmp_path, arguments
):
    python = _python(tmp_path)
    with _index(produced_wheels, "delayed") as (origin, state):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--python",
                str(python),
                "--version",
                "1.2.3",
                "--index-url",
                f"{origin}/simple",
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 2
        assert state["index_requests"] == 0
