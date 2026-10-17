"""Real rendered pages survive close failures, but not incomplete or crashed rendering."""

import json
import subprocess
import sys
import tempfile

import pytest

_PROBE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
import kaleido
import plotly.graph_objects as go
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from ml4t.diagnostic.visualization import export_figures_to_pdf

mode=sys.argv[1]
output=Path(sys.argv[2])
observed=output.with_suffix('.processes.json')
patch=r"""
import asyncio,json,os,psutil,kaleido
from pathlib import Path
from pypdf import PdfReader
mode=MODE
observed=Path(OBSERVED)
original_open=kaleido.Kaleido.open
original_write=kaleido.Kaleido.write_fig_from_object
def snapshot(self):
    processes=[psutil.Process(self.subprocess.pid)]
    processes+=processes[0].children(recursive=True)
    rows=json.loads(observed.read_text()) if observed.exists() else {}
    for process in processes:
        try: rows[str(process.pid)]=process.create_time()
        except psutil.NoSuchProcess: pass
    observed.write_text(json.dumps(rows))
    profiles=[str(arg).split('=',1)[1] for arg in self.subprocess.args if str(arg).startswith('--user-data-dir=')]
    assert profiles, 'No actual browser profile observed'
    observed.with_suffix('.profiles.json').write_text(json.dumps(profiles))
async def opened(self,*args,**kwargs):
    result=await original_open(self,*args,**kwargs)
    snapshot(self)
    return result
async def written(self,specs,*args,**kwargs):
    result=await original_write(self,specs,*args,**kwargs)
    snapshot(self)
    for spec in specs:
        page=PdfReader(spec['path'])
        assert len(page.pages)==1 and 'Owned PDF page' in page.pages[0].extract_text()
    if mode=='render_error': raise RuntimeError('Actual renderer failed after first page')
    if mode=='invalid_page': Path(specs[0]['path']).write_bytes(b'invalid PDF')
    return result
async def closed(self,*args,**kwargs):
    snapshot(self)
    if mode=='exit7': os._exit(7)
    if mode=='close_hang': await asyncio.sleep(3600)
    raise RuntimeError("Couldn't close or kill browser subprocess")
kaleido.Kaleido.open=opened
kaleido.Kaleido.write_fig_from_object=written
kaleido.Kaleido.close=closed
""".replace('MODE',repr(mode)).replace('OBSERVED',repr(str(observed)))
exec(patch)
try:
    from ml4t.diagnostic.visualization import _pdf_process
except ImportError:
    pass
else:
    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP

neighbor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(180)'])
success=False
try:
    figures=[go.Figure(go.Scatter(x=[1,2,3],y=[1,3,2])) for _ in range(2)]
    for i,figure in enumerate(figures): figure.update_layout(title=f'Owned PDF page {i+1}')
    originals=[figure.to_json() for figure in figures]
    try:
        result=export_figures_to_pdf(figures,output,page_size=(600,450),scale=1)
    except Exception as error:
        if mode in ('close_error','close_hang'): raise
        if mode=='exit7': assert 'code 7' in str(error),repr(error)
        if mode=='render_error': assert 'Actual renderer failed after first page' in str(error),repr(error)
        if mode=='invalid_page': assert isinstance(error,PdfReadError),repr(error)
        assert not output.exists(), 'Failed rendering must not publish a partial report'
    else:
        assert mode in ('close_error','close_hang'), 'Rendering defect was swallowed'
        pages=PdfReader(result).pages
        assert len(pages)==2
        assert all(f'Owned PDF page {i+1}' in page.extract_text() for i,page in enumerate(pages))
        assert all(abs(float(page.mediabox.width)-450)<0.1 for page in pages)
        assert all(abs(float(page.mediabox.height)-337.5)<1 for page in pages)
    assert originals==[figure.to_json() for figure in figures]
    assert neighbor.poll() is None, 'Unrelated process was killed by renderer cleanup'
    rows=json.loads(observed.read_text())
    assert rows, 'No actual browser processes observed'
    processes=[]
    for pid,created in rows.items():
        try:
            process=psutil.Process(int(pid))
            if process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:
                processes.append(process)
        except psutil.NoSuchProcess: pass
    gone,alive=psutil.wait_procs(processes,timeout=5)
    running=[]
    for process in alive:
        try:
            if process.status()!=psutil.STATUS_ZOMBIE: running.append(process.pid)
        except psutil.NoSuchProcess: pass
    assert not running, running
    try: psutil.Process(neighbor.pid).wait(timeout=0)
    except psutil.TimeoutExpired: pass
    else: raise AssertionError('Process-wait control accepted a live unrelated child')
    profiles=json.loads(observed.with_suffix('.profiles.json').read_text())
    assert all(not Path(profile).exists() for profile in profiles), profiles
    print(json.dumps({'mode':mode,'observed_browser_processes':len(rows),'running':running,
                      'waited_exit_codes':[process.returncode for process in gone],
                      'neighbor_alive':True,'input_figures_unchanged':True}),flush=True)
    success=True
finally:
    # A deliberately broken base provider must not leak its real browser in the red run.
    descendants=psutil.Process().children(recursive=True)
    for process in reversed(descendants):
        try: process.kill()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(descendants,timeout=5)
    neighbor.wait(timeout=5)
    if not success:
        import traceback
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)
'''


@pytest.mark.parametrize(
    "mode", ["close_error", "close_hang", "render_error", "invalid_page", "exit7"]
)
def test_real_pdf_cleanup_owns_browser_without_hiding_render_failures(mode, tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    psutil = pytest.importorskip("psutil")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE, mode, str(tmp_path / "report.pdf")],
            stdout=output,
            stderr=output,
        )
        try:
            process.wait(timeout=120)
        finally:
            if process.poll() is None:
                owned = psutil.Process(process.pid).children(recursive=True)
                for child in reversed(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait(timeout=5)
                psutil.wait_procs(owned, timeout=5)
        output.seek(0)
        text = output.read()
    assert process.returncode == 0, text
    result = json.loads(text.splitlines()[-1])
    assert result["mode"] == mode and result["running"] == [] and result["neighbor_alive"]


def test_local_plotlyjs_path_still_renders_a_real_pdf_in_the_owned_worker(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    from pathlib import Path

    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import export_figures_to_pdf

    bundle = Path(plotly.__file__).parent / "package_data/plotly.min.js"
    assert bundle.is_file()
    monkeypatch.setattr(pio.defaults, "plotlyjs", bundle)
    figure = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 3, 2]))
    figure.update_layout(title="Local Plotly source")
    original = figure.to_json()
    path = export_figures_to_pdf([figure], tmp_path / "local.pdf", scale=1)
    pages = pypdf.PdfReader(path).pages
    assert len(pages) == 1 and "Local Plotly source" in pages[0].extract_text()
    assert figure.to_json() == original
