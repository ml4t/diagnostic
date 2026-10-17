"""Actual multipage PDF export must share and close the modern renderer."""

import json
import os
import subprocess
import sys
import tempfile
import time

import pytest


def test_pandas_timestamp_return_coordinates_produce_a_real_dated_pdf(tmp_path, caplog):
    caplog.set_level("DEBUG", logger="choreographer")
    pytest.importorskip("kaleido")
    go = pytest.importorskip("plotly.graph_objects")
    pypdf = pytest.importorskip("pypdf")
    import pandas as pd

    from ml4t.diagnostic.visualization import export_figures_to_pdf

    figure = go.Figure(
        go.Scatter(
            x=[pd.Timestamp("2025-07-03", tz="UTC"), pd.Timestamp("2025-07-07", tz="UTC")],
            y=[0.01, -0.02],
        )
    )
    figure.update_layout(title="Observed dated returns")
    original = figure.to_json()
    output = tmp_path / "dated.pdf"
    export_figures_to_pdf([figure], output, scale=1)
    reader = pypdf.PdfReader(output)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "Observed dated returns" in text and "2025" in text
    assert "Jul 3" in text and "Jul 7" in text
    assert figure.to_json() == original


@pytest.mark.parametrize(
    ("scale", "observe_renderer"),
    [
        pytest.param(1.0, True, id="scale1-observed"),
        pytest.param(2.0, True, id="scale2-observed"),
        pytest.param(1.0, False, id="scale1-public-first"),
        pytest.param(2.0, False, id="scale2-public-first"),
        pytest.param(1.0, False, id="scale1-public-second"),
        pytest.param(2.0, False, id="scale2-public-second"),
        pytest.param(1.0, False, id="scale1-public-third"),
        pytest.param(2.0, False, id="scale2-public-third"),
    ],
)
def test_multipage_pdf_preserves_pages_and_returns_from_real_public_export(
    scale, observe_renderer, tmp_path
):
    _assert_multipage_pdf_export(scale, observe_renderer, tmp_path)


def _assert_multipage_pdf_export(scale, observe_renderer, tmp_path):
    kaleido = pytest.importorskip("kaleido")
    pio = pytest.importorskip("plotly.io")
    pytest.importorskip("pypdf")
    if not callable(getattr(pio, "write_images", None)) or not callable(
        getattr(kaleido, "write_fig_sync", None)
    ):
        pytest.skip("Modern batch lifecycle requires Plotly write_images and Kaleido1")
    code = """
import atexit,hashlib,json,os,sys,threading,time,traceback
diagnostic_stop=threading.Event()
def snapshot_threads():
    while not diagnostic_stop.wait(20):
        lines=['PDF exporter thread snapshot'+chr(10)]
        for identifier,frame in sys._current_frames().items():
            lines.append(f'Thread {identifier}:'+chr(10))
            lines.extend(traceback.format_stack(frame))
        os.write(2,''.join(lines).encode('utf-8',errors='backslashreplace'))
diagnostic_thread=threading.Thread(target=snapshot_threads,daemon=True)
diagnostic_thread.start()
def stop_diagnostics():
    diagnostic_stop.set()
    diagnostic_thread.join(timeout=1)
atexit.register(stop_diagnostics)
started=time.monotonic()
def stage(name,**details):
    print(json.dumps({'stage':name,'elapsed_seconds':time.monotonic()-started,**details}),file=sys.stderr,flush=True)
stage('imports_start')
from importlib.metadata import version
from pathlib import Path
import kaleido
import plotly.graph_objects as go
from pypdf import PdfReader
from ml4t import diagnostic
from ml4t.diagnostic.visualization import export_figures_to_pdf
observe=sys.argv[3]=='1'
if observe:
    from ml4t.diagnostic.visualization import _pdf_process
    observation_path=Path(sys.argv[1]).with_suffix('.renderer.json')
    observer='''
import json,kaleido,psutil
from pathlib import Path
original_open=kaleido.Kaleido.open
async def observe_open(self,*args,**kwargs):
    result=await original_open(self,*args,**kwargs)
    path=Path(OBSERVATION_PATH)
    observations=json.loads(path.read_text()) if path.exists() else []
    observations.append({'pid':self.subprocess.pid,'created':psutil.Process(self.subprocess.pid).create_time()})
    path.write_text(json.dumps(observations))
    return result
kaleido.Kaleido.open=observe_open
'''.replace('OBSERVATION_PATH',repr(str(observation_path)))
    _pdf_process._BOOTSTRAP=observer+chr(10)+_pdf_process._BOOTSTRAP
stage('imports_complete',versions={name:version(name) for name in ['plotly','kaleido','choreographer','pypdf']})
figures=[go.Figure(go.Bar(x=['A','B'],y=[1,-2])),
         go.Figure(go.Scatter(x=[1,2,3],y=[.85,.92,.78])),
         go.Figure(go.Histogram(x=[0.,.1,-.05,.02]))]
titles=['Holdings values','Method agreement','Observed returns']
for figure,title in zip(figures,titles):
    figure.update_layout(title=title)
before=[figure.to_json() for figure in figures]
stage('public_export_start')
path=export_figures_to_pdf(figures,sys.argv[1],page_size=(800,600),scale=float(sys.argv[2]))
stage('public_export_return')
reader=PdfReader(path)
stage('pdf_reader_ready')
texts=[page.extract_text() for page in reader.pages]
stage('pdf_text_extracted')
assert len(texts)==3 and all(title in text for title,text in zip(titles,texts))
assert [figure.to_json() for figure in figures]==before
if observe:
    import psutil
    observations=json.loads(observation_path.read_text())
    assert len(observations)==1,observations
    terminal=[]
    for observed in observations:
        try:
            actual=psutil.Process(observed['pid'])
            if actual.create_time()!=observed['created'] or actual.status()==psutil.STATUS_ZOMBIE:
                terminal.append(True)
            else:
                try:
                    actual.wait(timeout=5)
                    terminal.append(True)
                except psutil.TimeoutExpired:
                    terminal.append(False)
        except psutil.NoSuchProcess:
            terminal.append(True)
    assert all(terminal),observations
print(json.dumps({'module':diagnostic.__file__,
    'versions':{name:version(name) for name in ['ml4t-diagnostic','plotly','kaleido','choreographer','pypdf']},
    'renderer_observed':observe,'renderer_instances':len(observations) if observe else 0,
    'renderer_terminal':terminal if observe else [],
    'page_boxes':[[float(value) for value in page.mediabox] for page in reader.pages],
    'page_text':texts,'pdf_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    'original_figures_unchanged':True}))
sys.stdout.flush()
stage('final_process_state',pid=os.getpid(),threads=[
    {'name':thread.name,'daemon':thread.daemon,'ident':thread.ident}
    for thread in threading.enumerate()])
atexit.register(lambda:stage('python_atexit_reached'))
stage('assertions_complete')
"""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(tmp_path / "report.pdf"),
                str(scale),
                "1" if observe_renderer else "0",
            ],
            stdout=output,
            stderr=None,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            returncode_at_timeout = process.poll()
            process.kill()
            process.wait(timeout=5)
            output.seek(0)
            pytest.fail(
                "Actual PDF export subprocess exceeded 120 seconds; "
                f"returncode before cleanup={returncode_at_timeout}:\n{output.read()}"
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        output.seek(0)
        stdout = output.read()
    assert process.returncode == 0, stdout
    result = json.loads(stdout)
    if observe_renderer:
        assert result["renderer_instances"] == 1, result
        assert all(result["renderer_terminal"]), result
    assert result["page_boxes"] == [[0, 0, 600 * scale, 450 * scale]] * 3
    print(json.dumps(result))


def test_real_pdf_process_exit_does_not_wait_for_unrelated_inherited_stdout(tmp_path, monkeypatch):
    original = subprocess.Popen
    release = tmp_path / "release"
    started = tmp_path / "descendant-started"
    finished = tmp_path / "descendant-finished"
    descendant_code = (
        "import time\nfrom pathlib import Path\n"
        f"release=Path({str(release)!r})\n"
        "deadline=time.monotonic()+30\n"
        "while not release.exists() and time.monotonic()<deadline: time.sleep(0.05)\n"
        f"Path({str(finished)!r}).touch()\n"
    )

    class InheritedStdout(original):
        def __init__(self, args, *other, **kwargs):
            if isinstance(args, list) and len(args) > 2 and "assertions_complete" in str(args[2]):
                args = list(args)
                injection = (
                    "import subprocess\n"
                    f"Path({str(started)!r}).touch()\n"
                    f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}])\n"
                    "sys.stdout.flush()"
                )
                args[2] = args[2].replace("sys.stdout.flush()", injection)
            super().__init__(args, *other, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", InheritedStdout)
    try:
        _assert_multipage_pdf_export(1, True, tmp_path)
        assert started.exists()
        assert not finished.exists(), (
            "The probe waited for stdout EOF after the exporter had exited"
        )
    finally:
        release.touch()
        if started.exists():
            deadline = time.monotonic() + 5
            while not finished.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert finished.exists(), "Controlled stdout owner did not terminate"
