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
import plotly.io as pio
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from ml4t.diagnostic.visualization import export_figures_to_pdf

# Plain-text cleanup controls must reach their intended fault without a vendor CDN.
pio.defaults.mathjax=False
mode=sys.argv[1]
output=Path(sys.argv[2])
startup_delay=float(sys.argv[3])
observed=output.with_suffix('.processes.json')
patch=r"""
import asyncio,json,os,psutil,kaleido
from pathlib import Path
from pypdf import PdfReader
mode=MODE
startup_delay=STARTUP_DELAY
observed=Path(OBSERVED)
original_open=kaleido.Kaleido.open
original_write=kaleido.Kaleido.write_fig_from_object
original_close=kaleido.Kaleido.close
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
    await asyncio.sleep(startup_delay)
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
    if mode=='render_hang':
        observed.with_suffix('.fault.json').write_text(json.dumps({'stage':'after_first_valid_page'}))
        await asyncio.sleep(3600)
    if mode=='invalid_page': Path(specs[0]['path']).write_bytes(b'invalid PDF')
    return result
async def closed(self,*args,**kwargs):
    snapshot(self)
    if mode.startswith('resource_error'):
        marker=observed.with_suffix('.close.json')
        marker.write_text(json.dumps({'entered':True,'finished':False}))
        if mode=='resource_error_close_hang': await asyncio.sleep(3600)
        result=await original_close(self,*args,**kwargs)
        marker.write_text(json.dumps({'entered':True,'finished':True}))
        return result
    if mode in ('shared_status','permanent_status'):
        return await original_close(self,*args,**kwargs)
    if mode=='exit7': os._exit(7)
    if mode=='close_hang': await asyncio.sleep(3600)
    raise RuntimeError("Couldn't close or kill browser subprocess")
kaleido.Kaleido.open=opened
kaleido.Kaleido.write_fig_from_object=written
kaleido.Kaleido.close=closed
""".replace('MODE',repr(mode)).replace('STARTUP_DELAY',repr(startup_delay)).replace('OBSERVED',repr(str(observed)))
exec(patch)
read_handles=[]
try:
    from ml4t.diagnostic.visualization import _pdf_process
except ImportError:
    pass
else:
    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP
    if mode=='render_hang':
        original_observe=_pdf_process._OwnedProcesses.observe
        def arm_timeout_after_browser_open(self):
            original_observe(self)
            if observed.exists(): _pdf_process._RENDER_TIMEOUT=15
        _pdf_process._OwnedProcesses.observe=arm_timeout_after_browser_open
        _pdf_process._BOOTSTRAP=_pdf_process._BOOTSTRAP.replace('dump_traceback_later(20,','dump_traceback_later(1,')
        _pdf_process._BOOTSTRAP=_pdf_process._BOOTSTRAP.replace('.wait(20)','.wait(1)')
    if mode=='held_status':
        original_observe=_pdf_process._OwnedProcesses.observe
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def hold_first_progress(self):
            original_observe(self)
            if not read_handles:
                base=Path(self.process.args[-1])
                for candidate in [base,base.with_name(base.stem+'-000001.json')]:
                    if candidate.exists():
                        read_handles.append(candidate.open('rb'))
                        break
        def terminate_then_release(self):
            try: return original_terminate(self)
            finally:
                for handle in read_handles: handle.close()
        _pdf_process._OwnedProcesses.observe=hold_first_progress
        _pdf_process._OwnedProcesses.terminate=terminate_then_release
    if mode in ('shared_status','permanent_status'):
        import ctypes,threading
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                 ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        api.CreateFileW.restype=wintypes.HANDLE
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        api.CloseHandle.restype=wintypes.BOOL
        original_read=Path.read_text
        original_terminate=_pdf_process._OwnedProcesses.terminate
        sharing_started=[]
        sharing_handles=[]
        sharing_timers=[]
        sharing_mutex=threading.Lock()
        actual_denials=[]
        def release_sharing():
            with sharing_mutex:
                handles=sharing_handles[:]
                sharing_handles.clear()
            for handle in handles: assert api.CloseHandle(handle)
        def read_with_real_share_lock(self,*args,**kwargs):
            if self.name.startswith('status-') and self.suffix=='.json' and not sharing_started:
                # Real Windows handle permits writes/deletion but denies other readers,
                # as a scanner can do after the actual worker's atomic publication.
                handle=api.CreateFileW(str(self),0x80000000,0x00000006,None,3,0x80,None)
                if handle==ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                sharing_started.append(str(self))
                sharing_handles.append(handle)
                if mode=='shared_status':
                    timer=threading.Timer(.2,release_sharing)
                    timer.daemon=True
                    sharing_timers.append(timer)
                    timer.start()
            try:
                return original_read(self,*args,**kwargs)
            except PermissionError as error:
                if str(self) in sharing_started:
                    actual_denials.append({'errno':error.errno,'winerror':getattr(error,'winerror',None)})
                    observed.with_suffix('.fault.json').write_text(json.dumps({'mode':mode,'denials':actual_denials}))
                raise
        def terminate_and_release_sharing(self):
            try: return original_terminate(self)
            finally:
                release_sharing()
                for timer in sharing_timers: timer.join(timeout=1)
        Path.read_text=read_with_real_share_lock
        _pdf_process._OwnedProcesses.terminate=terminate_and_release_sharing

neighbor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(180)'])
server=None
requests=[]
success=False
try:
    figures=[go.Figure(go.Scatter(x=[1,2,3],y=[1,3,2])) for _ in range(2)]
    if mode.startswith('resource_error'):
        import threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        import plotly.io as pio
        class MissingTopojson(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_error(404,'Deliberately unavailable map resource')
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),MissingTopojson)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        pio.defaults.topojson=f'http://127.0.0.1:{server.server_port}/'
        figures[1]=go.Figure(go.Choropleth(locations=['USA'],z=[1]))
    for i,figure in enumerate(figures): figure.update_layout(title=f'Owned PDF page {i+1}')
    originals=[figure.to_json() for figure in figures]
    try:
        result=export_figures_to_pdf(figures,output,page_size=(600,450),scale=1)
    except Exception as error:
        if mode in ('close_error','close_hang','held_status','shared_status'): raise
        if mode=='permanent_status': assert isinstance(error,PermissionError),repr(error)
        if mode=='exit7': assert 'code 7' in str(error),repr(error)
        if mode=='render_error': assert 'Actual renderer failed after first page' in str(error),repr(error)
        if mode.startswith('resource_error'):
            assert requests and any('world' in path for path in requests),requests
            assert 'topojson' in str(error).lower(),repr(error)
            marker=observed.with_suffix('.close.json')
            assert marker.exists(), 'A real rendering error skipped graceful browser shutdown'
            assert json.loads(marker.read_text())=={'entered':True,'finished':mode=='resource_error'}
        if mode=='render_hang':
            assert json.loads(observed.with_suffix('.fault.json').read_text())=={'stage':'after_first_valid_page'}
            assert "phase='rendering', completed=0, expected=2" in str(error),repr(error)
            assert 'File "' in str(error) and 'Thread' in str(error),repr(error)
        if mode=='invalid_page': assert isinstance(error,PdfReadError),repr(error)
        assert not output.exists(), 'Failed rendering must not publish a partial report'
    else:
        assert mode in ('close_error','close_hang','held_status','shared_status'), 'Rendering defect was swallowed'
        pages=PdfReader(result).pages
        assert len(pages)==2
        assert all(f'Owned PDF page {i+1}' in page.extract_text() for i,page in enumerate(pages))
        assert all(abs(float(page.mediabox.width)-450)<0.1 for page in pages)
        assert all(abs(float(page.mediabox.height)-337.5)<1 for page in pages)
    assert originals==[figure.to_json() for figure in figures]
    if mode=='held_status': assert len(read_handles)==1 and all(handle.closed for handle in read_handles)
    if mode in ('shared_status','permanent_status'):
        assert sharing_started and actual_denials, 'No real Windows sharing denial was exercised'
        assert not sharing_handles and all(not timer.is_alive() for timer in sharing_timers)
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
    print(json.dumps({'mode':mode,'startup_delay':startup_delay,'observed_browser_processes':len(rows),'running':running,
                      'waited_exit_codes':[process.returncode for process in gone],
                      'neighbor_alive':True,'input_figures_unchanged':True}),flush=True)
    success=True
finally:
    if server is not None:
        server.shutdown()
        server.server_close()
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


def _run_pdf_probe(code, arguments):
    psutil = pytest.importorskip("psutil")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-c", code, *arguments], stdout=output, stderr=output
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
    return json.loads(text.splitlines()[-1])


@pytest.mark.parametrize(
    "mode,startup_delay",
    [
        (mode, 0)
        for mode in [
            "close_error",
            "close_hang",
            "render_error",
            "render_hang",
            "invalid_page",
            "exit7",
            "held_status",
            "resource_error",
            "resource_error_close_hang",
        ]
    ]
    + [pytest.param("render_hang", 16, id="render_hang_slow_open")]
    + [
        pytest.param(
            mode,
            0,
            id=mode,
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="Actual Windows file-sharing denial is required"
            ),
        )
        for mode in ("shared_status", "permanent_status")
    ],
)
def test_real_pdf_cleanup_owns_browser_without_hiding_render_failures(
    mode, startup_delay, tmp_path
):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    result = _run_pdf_probe(_PROBE, [mode, str(tmp_path / "report.pdf"), str(startup_delay)])
    assert result["mode"] == mode and result["running"] == [] and result["neighbor_alive"]
    assert result["startup_delay"] == startup_delay


def test_owned_worker_preserves_packaged_and_supported_custom_plotlyjs(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    from pathlib import Path

    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import export_figures_to_pdf

    bundle = Path(plotly.__file__).parent / "package_data/plotly.min.js"
    assert bundle.is_file()
    custom_path_supported = hasattr(pio.defaults, "plotlyjs")
    if custom_path_supported:
        monkeypatch.setattr(pio.defaults, "plotlyjs", bundle)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    figure = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 3, 2]))
    figure.update_layout(title="Local Plotly source")
    original = figure.to_json()
    path = export_figures_to_pdf([figure], tmp_path / "local.pdf", scale=1)
    pages = pypdf.PdfReader(path).pages
    assert len(pages) == 1 and "Local Plotly source" in pages[0].extract_text()
    assert figure.to_json() == original
    print(
        json.dumps(
            {"custom_plotlyjs_supported": custom_path_supported, "packaged_bundle": str(bundle)}
        )
    )


@pytest.fixture
def unavailable_mathjax_default():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []

    class UnavailableDefault(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), UnavailableDefault) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/unavailable-mathjax.js"
        try:
            yield url, requests
        finally:
            server.shutdown()
            thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("mathjax", [False, None, True])
def test_explicit_mathjax_disable_does_not_fetch_default_script(
    mathjax, tmp_path, monkeypatch, unavailable_mathjax_default
):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    url, requests = unavailable_mathjax_default
    # Redirect only the real renderer's default resource, not its loading logic.
    bootstrap = (
        "import kaleido._page_generator as pages\n"
        + f"pages.DEFAULT_MATHJAX={url!r}\n"
        + _pdf_process._BOOTSTRAP
    )
    monkeypatch.setattr(_pdf_process, "_BOOTSTRAP", bootstrap)
    monkeypatch.setattr(pio.defaults, "mathjax", mathjax)
    figure = go.Figure(go.Bar(x=["A", "B"], y=[1, 3]))
    figure.update_layout(title="Explicit MathJax configuration")
    original = figure.to_json()
    path = export_figures_to_pdf([figure], tmp_path / "mathjax.pdf", scale=1)
    pages = pypdf.PdfReader(path).pages
    assert len(pages) == 1
    assert "Explicit MathJax configuration" in pages[0].extract_text()
    assert figure.to_json() == original
    if mathjax is False:
        assert requests == [], "Explicit False must disable the default script request"
    else:
        assert requests and set(requests) == {"/unavailable-mathjax.js"}


def test_plain_text_cleanup_fault_reaches_topojson_failure_without_vendor_mathjax(
    tmp_path, unavailable_mathjax_default
):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    url, requests = unavailable_mathjax_default
    resource = "import kaleido._page_generator as pages\n" + f"pages.DEFAULT_MATHJAX={url!r}\n"
    bootstrap = (
        "from ml4t.diagnostic.visualization import _pdf_process\n"
        + f"_pdf_process._BOOTSTRAP={resource!r}+_pdf_process._BOOTSTRAP\n"
    )
    observed = _run_pdf_probe(
        bootstrap + _PROBE, ["resource_error_close_hang", str(tmp_path / "failure.pdf"), "0"]
    )
    assert observed["running"] == [] and observed["neighbor_alive"]
    assert requests == [], "Plain-text cleanup controls must not depend on the vendor CDN"


def test_pdf_worker_renders_without_initializing_the_analytical_package(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    imports_path = tmp_path / "worker-imports.jsonl"
    observer = f"""
import json,sys
from pathlib import Path
observed_imports=Path({str(imports_path)!r})
observed_imports.touch()
def observe_import(event,args):
    if event=='import' and str(args[0]).startswith('ml4t.diagnostic'):
        with observed_imports.open('a',encoding='utf-8') as output:
            output.write(json.dumps(str(args[0]))+chr(10))
sys.addaudithook(observe_import)
"""
    monkeypatch.setattr(_pdf_process, "_BOOTSTRAP", observer + "\n" + _pdf_process._BOOTSTRAP)
    figure = go.Figure(go.Bar(x=["A", "B"], y=[1, -2]))
    figure.update_layout(title="Standalone renderer")
    original = figure.to_json()
    output = export_figures_to_pdf([figure], tmp_path / "standalone.pdf", scale=1)
    pages = pypdf.PdfReader(output).pages
    assert len(pages) == 1 and "Standalone renderer" in pages[0].extract_text()
    assert figure.to_json() == original
    assert imports_path.is_file()
    assert imports_path.read_text(encoding="utf-8").splitlines() == []
