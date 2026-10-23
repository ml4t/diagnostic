"""Real rendered pages survive close failures, but not incomplete or crashed rendering."""

import json
import os
import shutil
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
fault_delay=float(sys.argv[4]) if len(sys.argv)>4 else .1
observed=output.with_suffix('.processes.json')
observed.with_suffix('.channel.bin').write_bytes(bytes(5))
patch=r"""
import asyncio,json,mmap,os,psutil,kaleido,sys,time
from pathlib import Path
from pypdf import PdfReader
mode=MODE
startup_delay=STARTUP_DELAY
fault_delay=FAULT_DELAY
observed=Path(OBSERVED)
pipe_proof_file=observed.with_suffix('.channel.bin').open('r+b')
pipe_proof=mmap.mmap(pipe_proof_file.fileno(),5,access=mmap.ACCESS_WRITE)
original_open=kaleido.Kaleido.open
original_write=kaleido.Kaleido.write_fig_from_object
original_close=kaleido.Kaleido.close
original_command=kaleido.Kaleido.send_command
failed_channel=None
from choreographer.channels import Pipe
original_pipe_init=Pipe.__init__
original_os_close=os.close
owned_pipe_descriptors={}
def record_pipe_init(self):
    original_pipe_init(self)
    pipe_proof[4]+=1
    for slot,name in enumerate(('_write_to_browser','_write_from_browser','_read_from_browser','_read_to_browser')):
        owned_pipe_descriptors[getattr(self,name)]=slot
def record_actual_descriptor_close(fd):
    result=original_os_close(fd)
    slot=owned_pipe_descriptors.pop(fd,None)
    if slot is not None: pipe_proof[slot]=1
    return result
Pipe.__init__=record_pipe_init
os.close=record_actual_descriptor_close
if mode in ('close_error','close_command_hang','close_command_hang_slow_shutdown','render_hang','pipe_number_reuse'):
    if os.name!='nt':
        from choreographer.browsers.chromium import Chromium
        original_cli=Chromium.get_cli
        def direct_browser_cli(self):
            cli=original_cli(self)
            # Match Windows' direct browser, without the Unix wrapper's synthetic bye.
            direct="import os,sys;os.dup2(0,3);os.dup2(1,4);os.set_inheritable(3,True);os.set_inheritable(4,True);os.execv(sys.argv[1],sys.argv[1:])"
            return [cli[0],'-c',direct,*cli[2:]]
        Chromium.get_cli=direct_browser_cli
async def command(self,command,*args,**kwargs):
    if command=='Browser.close' and mode=='close_command_hang_slow_shutdown':
        browser=psutil.Process(self.subprocess.pid)
        browser.suspend()
        print('PDF close control: browser suspended',time.monotonic(),file=sys.stderr,flush=True)
        original_write_json=self._channel.write_json
        sent=[]
        def record_real_close_request(message):
            result=original_write_json(message)
            if message.get('method')=='Browser.close': sent.append(True)
            return result
        self._channel.write_json=record_real_close_request
        pending=asyncio.create_task(original_command(self,command,*args,**kwargs))
        try:
            while not sent: await asyncio.sleep(.01)
            print('PDF close control: Browser.close transmitted',time.monotonic(),file=sys.stderr,flush=True)
            observed.with_suffix('.graceful.json').write_text(json.dumps({'browser_close_sent':True}))
            await asyncio.sleep(3600)
        finally:
            pending.cancel()
            try: await pending
            except (Exception,asyncio.CancelledError): pass
    result=await original_command(self,command,*args,**kwargs)
    if command=='Browser.close':
        observed.with_suffix('.graceful.json').write_text(json.dumps({'browser_close':True}))
        if mode=='close_command_hang': await asyncio.sleep(3600)
    return result
kaleido.Kaleido.send_command=command
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
if mode=='unresponsive_startup':
    from choreographer import Browser
    original_populate=Browser.populate_targets
    startup_armed=False
    async def populate(self,*args,**kwargs):
        global startup_armed
        result=await original_populate(self,*args,**kwargs)
        if not startup_armed and self.tabs:
            startup_armed=True
            target=next(iter(self.tabs.values()))
            snapshot(self)
            await target.send_command('Runtime.enable')
            await target.send_command('Runtime.addBinding',params={'name':'diagnosticStartupBlocked'})
            entered=asyncio.Queue()
            async def observed_entry(message):
                if message.get('params',{}).get('name')=='diagnosticStartupBlocked':
                    entered.put_nowait(message)
            target.subscribe('Runtime.bindingCalled',observed_entry)
            evaluate=asyncio.create_task(target.send_command('Runtime.evaluate',params={
                'expression':f"setTimeout(()=>{{diagnosticStartupBlocked('entered');while(true){{}}}},{int(fault_delay*1000)});true",
            }))
            try:
                await entered.get()
            finally:
                target.unsubscribe('Runtime.bindingCalled')
                evaluate.cancel()
                try: await evaluate
                except (Exception,asyncio.CancelledError): pass
            probe=asyncio.create_task(target.send_command('Page.enable'))
            done,_=await asyncio.wait([probe],timeout=.5)
            assert not done,'Initial renderer did not actually stop answering Page.enable'
            probe.cancel()
            try: await probe
            except asyncio.CancelledError: pass
            observed.with_suffix('.fault.json').write_text(json.dumps({'stage':'initial_page_command_unresponsive'}))
        return result
    Browser.populate_targets=populate
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
    if mode in ('render_error','cleanup_error'): raise RuntimeError('Actual renderer failed after first page')
    if mode in ('render_hang','cancel'):
        observed.with_suffix('.fault.json').write_text(json.dumps({'stage':'after_first_valid_page'}))
        await asyncio.sleep(3600)
    if mode=='invalid_page': Path(specs[0]['path']).write_bytes(b'invalid PDF')
    return result
async def closed(self,*args,**kwargs):
    global failed_channel
    if mode=='termination_error':
        result=await original_close(self,*args,**kwargs)
        print('PDF close control: actual browser close returned',file=sys.stderr,flush=True)
        return result
    if mode in ('close_error','close_command_hang','close_command_hang_slow_shutdown','render_hang','pipe_number_reuse'):
        failed_channel=self._channel
    snapshot(self)
    if mode=='pipe_number_reuse':
        await original_close(self,*args,**kwargs)
        old_numbers={getattr(failed_channel,name) for name in ('_write_to_browser','_write_from_browser','_read_from_browser','_read_to_browser')}
        retained=[]
        while not old_numbers.intersection(retained):
            retained.append(os.open(observed.with_suffix('.after-close.log'),os.O_WRONLY|os.O_CREAT,0o600))
            assert len(retained)<256,'No actual pipe descriptor number was reused'
        observed.with_suffix('.reused.json').write_text(json.dumps({'reused':sorted(old_numbers.intersection(retained))}))
    if mode=='close_command_hang_slow_shutdown':
        import threading
        original_pipe_close=self._channel.close
        pipe_closed=[]
        def close_then_allow_bounded_browser_exit():
            if pipe_closed: return original_pipe_close()
            pipe_closed.append(True)
            browser=psutil.Process(self.subprocess.pid)
            print('PDF close control: entering pipe close',time.monotonic(),file=sys.stderr,flush=True)
            original_pipe_close()
            print('PDF close control: pipe close returned',time.monotonic(),file=sys.stderr,flush=True)
            def resume():
                try:
                    browser.resume()
                    print('PDF close control: browser resumed',time.monotonic(),file=sys.stderr,flush=True)
                except psutil.NoSuchProcess: pass
            threading.Timer(1,resume).start()
            print('PDF close control: resume timer started',time.monotonic(),file=sys.stderr,flush=True)
        self._channel.close=close_then_allow_bounded_browser_exit
    if mode=='slow_shutdown':
        import threading
        process=psutil.Process(self.subprocess.pid)
        process.suspend()
        def resume():
            try: process.resume()
            except psutil.NoSuchProcess: pass
        timer=threading.Timer(4,resume)
        timer.daemon=True
        timer.start()
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
""".replace('MODE',repr(mode)).replace('STARTUP_DELAY',repr(startup_delay)).replace('FAULT_DELAY',repr(fault_delay)).replace('OBSERVED',repr(str(observed)))
exec(patch)
read_handles=[]
try:
    from ml4t.diagnostic.visualization import _pdf_process
except ImportError:
    pass
else:
    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP
    if mode=='unresponsive_startup': _pdf_process._RENDER_TIMEOUT=30
    if mode in ('slow_shutdown','close_command_hang_slow_shutdown'):
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def verify_graceful_before_escalation(self):
            browser_pid=int(next(iter(json.loads(observed.read_text()))))
            try:
                browser=psutil.Process(browser_pid)
                running=browser.is_running() and browser.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess: running=False
            original_terminate(self)
            assert not running, 'OS escalation interrupted a bounded graceful browser shutdown'
        _pdf_process._OwnedProcesses.terminate=verify_graceful_before_escalation
    if mode=='cancel':
        import _thread
        original_observe=_pdf_process._OwnedProcesses.observe
        interrupted=[]
        def cancel_after_valid_page(self):
            original_observe(self)
            if observed.with_suffix('.fault.json').exists() and not interrupted:
                interrupted.append(True)
                _thread.interrupt_main()
        _pdf_process._OwnedProcesses.observe=cancel_after_valid_page
    if mode in ('cleanup_error','termination_error'):
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def terminate_then_fail(self):
            original_terminate(self)
            raise RuntimeError('Owned termination verification failed')
        _pdf_process._OwnedProcesses.terminate=terminate_then_fail
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
    if mode in ('shared_status','permanent_status','profile_sharing','profile_delete_pending'):
        import ctypes,threading
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                 ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        api.CreateFileW.restype=wintypes.HANDLE
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        api.CloseHandle.restype=wintypes.BOOL
        api.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,
                                                   ctypes.c_void_p,wintypes.DWORD]
        api.GetFileInformationByHandleEx.restype=wintypes.BOOL
        class FileStandardInfo(ctypes.Structure):
            _fields_=[('allocation_size',ctypes.c_int64),('end_of_file',ctypes.c_int64),
                      ('links',wintypes.DWORD),('delete_pending',ctypes.c_ubyte),
                      ('directory',ctypes.c_ubyte)]
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
        if mode in ('profile_sharing','profile_delete_pending'):
            original_rmtree=_pdf_process.shutil.rmtree
            profile_directory=None
            cleanup_denials=[]
            pending_states=[]
            def observe_actual_cleanup(directory,*args,**kwargs):
                try: return original_rmtree(directory,*args,**kwargs)
                except OSError as error:
                    if Path(directory)==profile_directory and sharing_handles:
                        cleanup_denials.append(getattr(error,'winerror',None))
                        if mode=='profile_delete_pending':
                            info=FileStandardInfo()
                            assert api.GetFileInformationByHandleEx(sharing_handles[0],1,
                                ctypes.byref(info),ctypes.sizeof(info)),ctypes.get_last_error()
                            pending_states.append(bool(info.delete_pending))
                            assert info.directory and info.delete_pending,'No actual pending directory deletion'
                        observed.with_suffix('.fault.json').write_text(json.dumps({
                            'denials':actual_denials,
                            'cleanup_denials':cleanup_denials,
                            'pending_states':pending_states,
                            'directory':str(profile_directory),
                        }))
                        if not sharing_timers:
                            timer=threading.Timer(.2,release_sharing)
                            timer.daemon=True
                            sharing_timers.append(timer)
                            timer.start()
                    raise
            _pdf_process.shutil.rmtree=observe_actual_cleanup
            def terminate_then_lock_profile(self):
                global profile_directory
                original_terminate(self)
                profile_directory=Path(self.process.args[-2]).parent
                profile=Path(json.loads(observed.with_suffix('.profiles.json').read_text())[0])
                candidate=profile if mode=='profile_delete_pending' else next(
                    path for path in profile.rglob('*') if path.is_file())
                sharing=7 if mode=='profile_delete_pending' else 3
                flags=0x02000000 if mode=='profile_delete_pending' else 0x80
                handle=api.CreateFileW(str(candidate),0x80000000,sharing,None,3,flags,None)
                if handle==ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                sharing_handles.append(handle)
                if mode=='profile_delete_pending': return
                try: candidate.unlink()
                except PermissionError as error:
                    assert error.winerror==32
                    actual_denials.append(error.winerror)
                    observed.with_suffix('.fault.json').write_text(json.dumps({
                        'denials':actual_denials,
                        'directory':str(Path(self.process.args[-2]).parent),
                    }))
                else: raise AssertionError('Profile lock did not prevent deletion')
            _pdf_process._OwnedProcesses.terminate=terminate_then_lock_profile
        else:
            Path.read_text=read_with_real_share_lock
            _pdf_process._OwnedProcesses.terminate=terminate_and_release_sharing

neighbor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(180)'])
server=None
requests=[]
success=False
expected_render_error=None
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
    except BaseException as error:
        expected_render_error={'type':type(error).__name__,'message':str(error),
                               'notes':list(getattr(error,'__notes__',[]))}
        observed.with_suffix('.error.json').write_text(json.dumps(expected_render_error))
        if mode in ('close_error','close_hang','held_status','shared_status','profile_sharing','profile_delete_pending','slow_shutdown','unresponsive_startup','close_command_hang_slow_shutdown','pipe_number_reuse'): raise
        if mode=='permanent_status': assert isinstance(error,PermissionError),repr(error)
        if mode=='cancel': assert isinstance(error,KeyboardInterrupt),repr(error)
        if mode=='exit7': assert 'code 7' in str(error),repr(error)
        if mode in ('render_error','cleanup_error'):
            assert 'Actual renderer failed after first page' in str(error),repr(error)
            if mode=='cleanup_error':
                assert any('Owned termination verification failed' in note for note in error.__notes__)
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
            assert 'Async task' in str(error) and 'in written' in str(error),repr(error)
        if mode=='invalid_page': assert isinstance(error,PdfReadError),repr(error)
        if mode=='termination_error':
            assert 'Owned termination verification failed' in str(error),repr(error)
            notes='\n'.join(getattr(error,'__notes__',[]))
            assert 'actual browser close returned' in notes,notes
            assert "'phase': 'closed'" in notes and "'completed': 2" in notes,notes
        assert not output.exists(), 'Failed rendering must not publish a partial report'
    else:
        assert mode in ('close_error','close_hang','close_command_hang','held_status','shared_status','profile_sharing','profile_delete_pending','slow_shutdown','unresponsive_startup','close_command_hang_slow_shutdown','pipe_number_reuse'), 'Rendering defect was swallowed'
        pages=PdfReader(result).pages
        assert len(pages)==2
        assert all(f'Owned PDF page {i+1}' in page.extract_text() for i,page in enumerate(pages))
        assert all(abs(float(page.mediabox.width)-450)<0.1 for page in pages)
        assert all(abs(float(page.mediabox.height)-337.5)<1 for page in pages)
        if mode=='close_command_hang_slow_shutdown':
            assert json.loads(observed.with_suffix('.graceful.json').read_text())=={'browser_close_sent':True}
    assert originals==[figure.to_json() for figure in figures]
    if mode=='unresponsive_startup':
        assert json.loads(observed.with_suffix('.fault.json').read_text())=={'stage':'initial_page_command_unresponsive'}
    if mode=='held_status': assert len(read_handles)==1 and all(handle.closed for handle in read_handles)
    if mode in ('shared_status','permanent_status'):
        assert sharing_started and actual_denials, 'No real Windows sharing denial was exercised'
        assert not sharing_handles and all(not timer.is_alive() for timer in sharing_timers)
    if mode in ('profile_sharing','profile_delete_pending'):
        assert not sharing_handles
        if mode=='profile_sharing':
            assert actual_denials==[32]
            assert 32 in cleanup_denials,'No actual directory cleanup encountered the sharing denial'
        else:
            assert 145 in cleanup_denials,'No actual directory-not-empty error observed'
            assert pending_states and all(pending_states)
        for timer in sharing_timers: timer.join(timeout=1)
        assert all(not timer.is_alive() for timer in sharing_timers)
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
    assert all(not Path(profile).exists() for profile in profiles), {
        'profiles':profiles,'expected_render_error':expected_render_error}
    if mode in ('close_error','close_command_hang','render_error','render_hang','cancel'):
        assert observed.with_suffix('.graceful.json').exists(), 'Close failure skipped graceful browser shutdown'
    if mode in ('close_error','close_command_hang','render_hang','pipe_number_reuse'):
        assert pipe_proof[4]==1,'The resource control requires one observed renderer pipe'
        assert list(pipe_proof[:4])==[1,1,1,1], 'Graceful fallback retained owned browser pipe descriptors'
    print(json.dumps({'mode':mode,'startup_delay':startup_delay,'observed_browser_processes':len(rows),'running':running,
                      'observed_pipe_channels':pipe_proof[4],'successful_owned_pipe_closes':list(pipe_proof[:4]),
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
    if mode=='profile_delete_pending':
        release_sharing()
        for timer in sharing_timers: timer.join(timeout=1)
        if profile_directory.exists(): original_rmtree(profile_directory)
    pipe_proof.close()
    pipe_proof_file.close()
    if not success:
        import traceback
        print(json.dumps({'expected_render_error':expected_render_error}),flush=True)
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


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual Windows delete-pending directory required"
)
@pytest.mark.parametrize("release_delay", [0.2, 3], ids=["released", "past_deadline"])
def test_delete_pending_browser_profile_cleanup_obeys_existing_deadline(tmp_path, release_delay):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    probe = _PROBE.replace(
        "threading.Timer(.2,release_sharing)", f"threading.Timer({release_delay},release_sharing)"
    )
    if release_delay == 3:
        target.write_bytes(b"Existing report must survive failed profile cleanup")
        with pytest.raises(AssertionError) as failure:
            _run_pdf_probe(probe, ["profile_delete_pending", str(target), "0"])
        assert "WinError 145" in str(failure.value)
        assert target.read_bytes() == b"Existing report must survive failed profile cleanup"
    else:
        result = _run_pdf_probe(probe, ["profile_delete_pending", str(target), "0"])
        assert result["running"] == [] and result["neighbor_alive"]


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="Actual unprivileged POSIX directory denial required",
)
def test_crashed_worker_profile_failure_retains_original_error_and_cleanup_notes(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    original = "_pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    denial = r"""
    blocked_profile_roots=[]
    original_termination=_pdf_process._OwnedProcesses.terminate
    def deny_profile_after_actual_termination(self):
        original_termination(self)
        directory=Path(self.process.args[-2]).parent
        blocked_profile_roots.append(directory)
        directory.chmod(0)
        try: list(directory.iterdir())
        except PermissionError: pass
        else: raise AssertionError('Actual directory access was not denied')
    _pdf_process._OwnedProcesses.terminate=deny_profile_after_actual_termination
"""
    probe = _PROBE.replace(original, original + denial)
    probe = probe.replace(
        "    profiles=json.loads(observed.with_suffix('.profiles.json').read_text())",
        "    assert len(blocked_profile_roots)==1\n"
        "    for directory in blocked_profile_roots: directory.chmod(0o700)\n"
        "    profiles=json.loads(observed.with_suffix('.profiles.json').read_text())",
    )
    probe = probe.replace(
        "    neighbor.wait(timeout=5)",
        "    neighbor.wait(timeout=5)\n"
        "    for directory in blocked_profile_roots:\n"
        "        directory.chmod(0o700)\n"
        "        _pdf_process.shutil.rmtree(directory)",
    )
    with pytest.raises(AssertionError) as failure:
        _run_pdf_probe(probe, ["exit7", str(tmp_path / "report.pdf"), "0"])
    assert "PDF profile cleanup failed" in str(failure.value)
    assert "code 7" in str(failure.value)


@pytest.mark.parametrize(
    "mode,startup_delay",
    [
        (mode, 0)
        for mode in [
            "close_error",
            "close_command_hang",
            "close_command_hang_slow_shutdown",
            "pipe_number_reuse",
            "close_hang",
            "slow_shutdown",
            "unresponsive_startup",
            "render_error",
            "cleanup_error",
            "cancel",
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
        for mode in ("shared_status", "permanent_status", "profile_sharing")
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


def test_browser_fault_setup_waits_for_actual_script_entry_instead_of_elapsed_sleep(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    result = _run_pdf_probe(
        _PROBE, ["unresponsive_startup", str(tmp_path / "report.pdf"), "0", "2"]
    )
    assert result["observed_browser_processes"] > 0
    assert result["running"] == [] and result["neighbor_alive"]


def test_pipe_close_evidence_survives_interrupted_terminal_status_notification(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    delayed_notification = r"""
import time
original_notification_write=Path.write_text
def delay_terminal_notification(self,text,*args,**kwargs):
    if self.name.startswith('status-') and self.suffix=='.new':
        if json.loads(text).get('phase') in ('close_failed','render_failed'):
            observed.with_suffix('.terminal-stall').write_bytes(b'1')
            time.sleep(20)
    return original_notification_write(self,text,*args,**kwargs)
Path.write_text=delay_terminal_notification
"""
    original = "_pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    assert original in _PROBE
    delayed = _PROBE.replace(
        original,
        "_pdf_process._BOOTSTRAP=patch+chr(10)+"
        + repr(delayed_notification)
        + "+chr(10)+_pdf_process._BOOTSTRAP",
    )
    try:
        result = _run_pdf_probe(delayed, ["close_command_hang", str(tmp_path / "report.pdf"), "0"])
    finally:
        assert (tmp_path / "report.processes.terminal-stall").read_bytes() == b"1"
    assert result["running"] == [] and result["neighbor_alive"]


def test_completed_pdf_termination_failure_preserves_worker_shutdown_diagnostics(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    result = _run_pdf_probe(_PROBE, ["termination_error", str(tmp_path / "report.pdf"), "0"])
    assert result["running"] == [] and result["neighbor_alive"]


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows sharing denial required")
def test_real_profile_sharing_fails_without_directory_cleanup_recovery(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    baseline = (
        "import tempfile\nfrom ml4t.diagnostic.visualization import _pdf_process\n"
        "_pdf_process._renderer_directory=tempfile.TemporaryDirectory\n"
    )
    with pytest.raises(AssertionError, match="PermissionError.*|WinError 32"):
        _run_pdf_probe(baseline + _PROBE, ["profile_sharing", str(tmp_path / "report.pdf"), "0"])
    from pathlib import Path

    fault = json.loads((tmp_path / "report.processes.fault.json").read_text())
    assert fault["denials"] == [32]
    assert 32 in fault["cleanup_denials"], (
        "The faulty directory cleanup never encountered the sharing denial"
    )
    directory = Path(fault["directory"])
    if directory.exists():
        shutil.rmtree(directory)


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


def test_completed_page_cannot_lose_its_load_event_before_renderer_initialization(
    tmp_path, monkeypatch
):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    marker = tmp_path / "actual-navigation.json"
    patch = r"""
import asyncio,json,choreographer
from pathlib import Path
marker=Path(MARKER)
original_command=choreographer.Tab.send_command
navigated=set()
async def command(self,method,*args,**kwargs):
    if method=='Page.enable' and self.target_id in navigated:
        for attempt in range(200):
            result=await original_command(self,'Runtime.evaluate',params={'expression':'document.readyState','returnByValue':True})
            if result.get('result',{}).get('result',{}).get('value')=='complete':
                marker.write_text(json.dumps({'ready_before_page_enable':True}))
                break
            await asyncio.sleep(.05)
        else: raise RuntimeError('Actual Chrome document did not finish loading')
    result=await original_command(self,method,*args,**kwargs)
    if method=='Page.navigate': navigated.add(self.target_id)
    return result
choreographer.Tab.send_command=command
""".replace("MARKER", repr(str(marker)))
    monkeypatch.setattr(_pdf_process, "_BOOTSTRAP", patch + "\n" + _pdf_process._BOOTSTRAP)
    monkeypatch.setattr(_pdf_process, "_RENDER_TIMEOUT", 30)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    figure = go.Figure(go.Scatter(x=[1, 2], y=[1, 3]))
    figure.update_layout(title="Actual completed document")
    original = figure.to_json()
    output = export_figures_to_pdf([figure], tmp_path / "completed-document.pdf", scale=1)
    assert json.loads(marker.read_text())["ready_before_page_enable"]
    pages = pypdf.PdfReader(output).pages
    assert len(pages) == 1 and "Actual completed document" in pages[0].extract_text()
    assert figure.to_json() == original


def test_previous_blank_page_cannot_mark_unloaded_renderer_scripts_as_ready(tmp_path, monkeypatch):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    bundle = (Path(plotly.__file__).parent / "package_data/plotly.min.js").read_bytes()
    bundle = b"setTimeout(function(){\n" + bundle + b"\n}, 750);"
    requested = []

    class DelayedBundle(BaseHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            time.sleep(1)
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(bundle)))
            self.end_headers()
            try:
                self.wfile.write(bundle)
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    marker = tmp_path / "actual-previous-load.json"
    patch = r"""
import asyncio,json,choreographer
from pathlib import Path
marker=Path(MARKER)
original_command=choreographer.Tab.send_command
first_navigation=True
async def command(self,method,*args,**kwargs):
    global first_navigation
    if method=='Page.navigate' and first_navigation:
        first_navigation=False
        loaded=self.subscribe_once('Page.loadEventFired')
        await original_command(self,'Page.navigate',params={'url':'data:text/html,<title>previous-document</title>'})
        await asyncio.wait_for(loaded,10)
        marker.write_text(json.dumps({'previous_document_load_delivered':True}))
    return await original_command(self,method,*args,**kwargs)
choreographer.Tab.send_command=command
""".replace("MARKER", repr(str(marker)))
    monkeypatch.setattr(_pdf_process, "_BOOTSTRAP", patch + "\n" + _pdf_process._BOOTSTRAP)
    monkeypatch.setattr(_pdf_process, "_RENDER_TIMEOUT", 30)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    figure = go.Figure(go.Bar(x=["A", "B"], y=[1, 3]))
    figure.update_layout(title="Renderer document is ready")
    original = figure.to_json()
    with ThreadingHTTPServer(("127.0.0.1", 0), DelayedBundle) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        delayed_bundle = f"http://127.0.0.1:{server.server_port}/plotly.min.js"
        renderer_patch = r"""
import kaleido
original_init=kaleido.Kaleido.__init__
def init(self,*args,**kwargs):
    kwargs['plotlyjs']=DELAYED_BUNDLE
    original_init(self,*args,**kwargs)
kaleido.Kaleido.__init__=init
""".replace("DELAYED_BUNDLE", repr(delayed_bundle))
        monkeypatch.setattr(
            _pdf_process, "_BOOTSTRAP", renderer_patch + "\n" + _pdf_process._BOOTSTRAP
        )
        try:
            output = export_figures_to_pdf(
                [figure, figure], tmp_path / "ready-document.pdf", scale=1
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert json.loads(marker.read_text())["previous_document_load_delivered"]
    assert requested and set(requested) == {"/plotly.min.js"}
    pages = pypdf.PdfReader(output).pages
    assert len(pages) == 2
    assert all("Renderer document is ready" in page.extract_text() for page in pages)
    assert figure.to_json() == original


def test_browser_startup_failure_retains_the_actual_process_stderr(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    import plotly.graph_objects as go

    from ml4t.diagnostic.visualization import export_figures_to_pdf

    monkeypatch.setenv("BROWSER_PATH", sys.executable)
    output = tmp_path / "report.pdf"
    output.write_bytes(b"previous verified report")
    with pytest.raises(RuntimeError) as failed:
        export_figures_to_pdf([go.Figure(go.Bar(x=[1, 2], y=[1, 3]))], output, scale=1)
    assert "PDF browser:" in str(failed.value)
    assert "unknown option" in str(failed.value).lower()
    assert "--disable-gpu" in str(failed.value)
    assert output.read_bytes() == b"previous verified report"
    assert not getattr(failed.value, "__notes__", [])


def test_readiness_timeout_identifies_the_loaded_document_and_missing_plotly(tmp_path, monkeypatch):
    import re

    pytest.importorskip("kaleido")
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    wrong_bundle = tmp_path / "not-plotly.js"
    wrong_bundle.write_text("window.nonPlotlyAssetLoaded = true;", encoding="utf-8")
    monkeypatch.setattr(pio.defaults, "plotlyjs", wrong_bundle.as_uri(), raising=False)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    monkeypatch.setattr(_pdf_process, "_RENDER_TIMEOUT", 30)
    output = tmp_path / "report.pdf"
    output.write_bytes(b"previous verified report")
    with pytest.raises(RuntimeError) as failed:
        export_figures_to_pdf([go.Figure(go.Bar(x=[1, 2], y=[1, 3]))], output, scale=1)
    assert "phase='rendering', completed=0, expected=1" in str(failed.value)
    assert output.read_bytes() == b"previous verified report"
    assert not getattr(failed.value, "__notes__", [])
    observations = re.findall(r"PDF document readiness: (\{[^\r\n]+\})", str(failed.value))
    assert observations, "The timeout discarded the actual browser readiness failure"
    observed = json.loads(observations[-1])
    assert observed["ready"] is False and observed["document_state"] == "complete"
    assert observed["href"] == observed["expected_href"]
    assert observed["plotly_type"] == "undefined"


def test_context_burst_cannot_drop_ready_main_world_after_a_child_frame(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    patch = r"""
import choreographer
original_command=choreographer.Tab.send_command
armed=False
async def command(self,method,*args,**kwargs):
    global armed
    if method=='Runtime.disable':
        result=await original_command(self,method,*args,**kwargs)
        await original_command(self,'Runtime.enable')
        armed=True
        return result
    if method=='Runtime.enable' and armed:
        armed=False
        child=self.subscribe_once('Runtime.executionContextCreated')
        await original_command(self,'Runtime.evaluate',params={'expression':"document.body.appendChild(document.createElement('iframe'))"})
        await child
        await original_command(self,'Runtime.disable')
    return await original_command(self,method,*args,**kwargs)
choreographer.Tab.send_command=command
"""
    monkeypatch.setattr(_pdf_process, "_BOOTSTRAP", patch + "\n" + _pdf_process._BOOTSTRAP)
    monkeypatch.setattr(_pdf_process, "_RENDER_TIMEOUT", 30)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    figure = go.Figure(go.Bar(x=["A", "B"], y=[1, 3]))
    figure.update_layout(title="Main rendering context survives a burst")
    original = figure.to_json()
    output = export_figures_to_pdf([figure], tmp_path / "context-burst.pdf", scale=1)
    pages = pypdf.PdfReader(output).pages
    assert len(pages) == 1
    assert "Main rendering context survives a burst" in pages[0].extract_text()
    assert figure.to_json() == original


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
