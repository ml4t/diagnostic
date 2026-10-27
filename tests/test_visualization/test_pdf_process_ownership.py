"""Real rendered pages survive close failures, but not incomplete or crashed rendering."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
import asyncio,json,mmap,os,psutil,kaleido,subprocess,sys,time
from pathlib import Path
from pypdf import PdfReader
mode=MODE
startup_delay=STARTUP_DELAY
fault_delay=FAULT_DELAY
observed=Path(OBSERVED)
debugger_path=DEBUGGER_PATH
pipe_proof_file=observed.with_suffix('.channel.bin').open('r+b')
pipe_proof=mmap.mmap(pipe_proof_file.fileno(),5,access=mmap.ACCESS_WRITE)
original_open=kaleido.Kaleido.open
original_write=kaleido.Kaleido.write_fig_from_object
original_close=kaleido.Kaleido.close
original_command=kaleido.Kaleido.send_command
failed_channel=None
live_dependency_armed=False
slow_shutdown_armed=False
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
if mode in ('close_error','close_command_hang','close_command_hang_slow_shutdown','render_hang','pipe_number_reuse','slow_shutdown'):
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
    if command=='Browser.close' and mode=='acknowledged_native_exit':
        original_request_write=self._channel.write_json
        def record_actual_close_request(message):
            result=original_request_write(message)
            if message.get('method')=='Browser.close':
                observed.with_suffix('.graceful-request.json').write_text(json.dumps({'browser_close_sent':True}))
            return result
        self._channel.write_json=record_actual_close_request
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
    if command=='Browser.close' and mode=='acknowledged_native_exit':
        await asyncio.sleep(3600)
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
            # The arbitrary initial target may already be unresponsive before injection.
            previous=list(self.tabs)
            target=await self.create_tab(url='about:blank',window=True)
            for target_id in previous:
                await self.close_tab(target_id)
            observed.with_suffix('.setup.json').write_text(json.dumps({
                'target_id':target.target_id,'replaced_target_ids':previous}))
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
    global failed_channel,slow_shutdown_armed
    if mode=='termination_error':
        result=await original_close(self,*args,**kwargs)
        print('PDF close control: actual browser close returned',file=sys.stderr,flush=True)
        return result
    if mode in ('close_error','close_command_hang','close_command_hang_slow_shutdown','render_hang','pipe_number_reuse'):
        failed_channel=self._channel
    if mode!='acknowledged_native_exit': snapshot(self)
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
    if mode=='slow_shutdown' and not slow_shutdown_armed:
        slow_shutdown_armed=True
        import threading
        process=psutil.Process(self.subprocess.pid)
        process.suspend()
        resume_lock=threading.Lock()
        resumed=[]
        pipe_closed=[]
        original_pipe_close=self._channel.close
        original_write_json=self._channel.write_json
        close_sent=[]
        def record_actual_close_request(message):
            result=original_write_json(message)
            if message.get('method')=='Browser.close': close_sent.append(True)
            return result
        self._channel.write_json=record_actual_close_request
        def observe_pipe_close():
            with resume_lock:
                if not pipe_closed:
                    pipe_closed.append(True)
                    observed.with_suffix('.slow-close.json').write_text(json.dumps({
                        'resumed_before_pipe_close':bool(resumed),
                        'browser_close_sent':bool(close_sent),
                    }))
            return original_pipe_close()
        self._channel.close=observe_pipe_close
        def resume():
            with resume_lock:
                try: process.resume()
                except psutil.NoSuchProcess: pass
                else: resumed.append(True)
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
    if mode=='job_dependency':
        browser=psutil.Process(self.subprocess.pid)
        browser.suspend()
        observed.with_suffix('.browser-held.json').write_text(json.dumps({
            'pid':browser.pid,'suspend_returned':True}))
    if mode in ('exit7','profile_readonly','job_dependency'): os._exit(7)
    if mode in ('live_worker_dependency','worker_exit_dependency'):
        global live_dependency_armed
        if live_dependency_armed: await asyncio.sleep(3600)
        live_dependency_armed=True
        browser=psutil.Process(self.subprocess.pid)
        browser.suspend()
        ready=observed.with_suffix('.debugger-ready.json')
        blocked=observed.with_suffix('.kernel-exit.json')
        if mode=='worker_exit_dependency':
            import runpy,threading
            native=runpy.run_path(debugger_path)
            thread=threading.Thread(target=native['hold_exit'],args=(browser.pid,ready,blocked),daemon=True)
            thread.start()
            helper=None
        else:
            helper=subprocess.Popen(
                [sys.executable,debugger_path,str(browser.pid),str(ready),str(blocked)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            helper.stdin.write(b'1')
            helper.stdin.close()
        deadline=time.monotonic()+8
        while not ready.exists() and time.monotonic()<deadline:
            await asyncio.sleep(.01)
        assert ready.exists(), 'Live worker debugger never attached'
        attached=json.loads(ready.read_text())
        assert 'error' not in attached, attached
        if helper is None:
            assert attached['debugger']==os.getpid()
            observed.with_suffix('.worker-debugger.json').write_text(json.dumps({
                'worker_pid':os.getpid(),'worker_created':psutil.Process().create_time(),
                'browser_pid':browser.pid,'debugger_pid':attached['debugger']}))
        else:
            observed.with_suffix('.live-helper.json').write_text(json.dumps({
                'worker_pid':os.getpid(),'browser_pid':browser.pid,'helper_pid':helper.pid,
                'helper_created':psutil.Process(helper.pid).create_time()}))
        await asyncio.sleep(3600)
    if mode=='acknowledged_native_exit': await asyncio.sleep(3600)
    if mode=='close_hang': await asyncio.sleep(3600)
    raise RuntimeError("Couldn't close or kill browser subprocess")
kaleido.Kaleido.open=opened
kaleido.Kaleido.write_fig_from_object=written
kaleido.Kaleido.close=closed
""".replace('MODE',repr(mode)).replace('STARTUP_DELAY',repr(startup_delay)).replace('FAULT_DELAY',repr(fault_delay)).replace('OBSERVED',repr(str(observed))).replace('DEBUGGER_PATH',repr(sys.argv[5] if len(sys.argv)>5 else ''))
exec(patch)
read_handles=[]
try:
    from ml4t.diagnostic.visualization import _pdf_process
except ImportError:
    pass
else:
    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP
    if mode=='acknowledged_native_exit':
        import ctypes
        from ctypes import wintypes
        native=[]
        external=[]
        actual_observe=_pdf_process._OwnedProcesses.observe
        actual_terminate=_pdf_process._OwnedProcesses.terminate
        def retain_actual_browser_handle(self):
            actual_observe(self)
            if native or not observed.exists(): return
            try: recorded=json.loads(observed.read_text())
            except (json.JSONDecodeError,PermissionError): return
            pid=int(next(iter(recorded)))
            assert psutil.Process(pid).create_time()==recorded[str(pid)]
            handle=self.job.api.OpenProcess(0x00101000,False,pid)
            if not handle: raise ctypes.WinError(ctypes.get_last_error())
            member=wintypes.BOOL()
            self.job._check(self.job.api.IsProcessInJob(handle,self.job.handle,ctypes.byref(member)))
            assert member.value, 'The actual browser must belong to the renderer job'
            native.append((pid,handle))
            ready=observed.with_suffix('.ack-debugger-ready.json')
            blocked=observed.with_suffix('.ack-kernel-exit.json')
            debugger=subprocess.Popen(
                [sys.executable,sys.argv[5],str(pid),str(ready),str(blocked),'30'],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            external.append(debugger)
            witness=self.job.api.OpenProcess(0x00101000,False,debugger.pid)
            if not witness: raise ctypes.WinError(ctypes.get_last_error())
            try:
                outside=wintypes.BOOL()
                self.job._check(self.job.api.IsProcessInJob(witness,self.job.handle,ctypes.byref(outside)))
                assert not outside.value, 'Native I/O witness must remain outside renderer ownership'
            finally: self.job.api.CloseHandle(witness)
            observed.with_suffix('.ack-control.json').write_text(json.dumps({
                'browser_pid':pid,'debugger_pid':debugger.pid,'outside_renderer_job':True,
                'release_delay':30}))
            debugger.stdin.write(b'1')
            debugger.stdin.close()
        def record_native_exit_around_forced_cleanup(self):
            assert native, 'No actual browser handle retained'
            pid,handle=native[0]
            returned=False
            try:
                result=actual_terminate(self)
                returned=True
                return result
            finally:
                code=wintypes.DWORD()
                self.job._check(self.job.api.GetExitCodeProcess(handle,ctypes.byref(code)))
                observed.with_suffix('.ack-after-cleanup.json').write_text(json.dumps({
                    'pid':pid,'native_wait_result':self.job.api.WaitForSingleObject(handle,0),
                    'exit_code':code.value,'cleanup_returned':returned}))
                self.job.api.CloseHandle(handle)
        _pdf_process._OwnedProcesses.observe=retain_actual_browser_handle
        _pdf_process._OwnedProcesses.terminate=record_native_exit_around_forced_cleanup
    if mode=='job_dependency':
        import ctypes
        from ctypes import wintypes
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def add_unobserved_job_dependency(self):
            self.observe()
            assert self.process.poll()==7, 'The actual worker must have exited before job cleanup'
            browser_pid=int(next(iter(json.loads(observed.read_text()))))
            assert browser_pid in self.observed
            held=json.loads(observed.with_suffix('.browser-held.json').read_text())
            assert held=={'pid':browser_pid,'suspend_returned':True}
            ready=observed.with_suffix('.debugger-ready.json')
            blocked=observed.with_suffix('.kernel-exit.json')
            debugger=subprocess.Popen(
                [sys.executable,sys.argv[5],str(browser_pid),str(ready),str(blocked)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            handle=self.job.api.OpenProcess(0x1101,False,debugger.pid)
            if not handle: raise ctypes.WinError(ctypes.get_last_error())
            try:
                self.job._check(self.job.api.AssignProcessToJobObject(self.job.handle,handle))
                member=wintypes.BOOL()
                self.job._check(self.job.api.IsProcessInJob(handle,self.job.handle,ctypes.byref(member)))
                assert member.value
            finally: self.job.api.CloseHandle(handle)
            debugger.stdin.write(b'1')
            debugger.stdin.close()
            try:
                deadline=time.monotonic()+10
                while not ready.exists() and debugger.poll() is None and time.monotonic()<deadline:
                    time.sleep(.01)
                assert ready.exists(), 'Owned debugger never attached'
                attachment=json.loads(ready.read_text())
                assert 'error' not in attachment,attachment
                earlier=self.observed.get(debugger.pid)
                assert earlier is None or not earlier.is_running(), 'Control must reproduce a late job member'
                observed.with_suffix('.late-job.json').write_text(json.dumps({
                    'worker_exit':self.process.returncode,'browser_pid':browser_pid,
                    'debugger_pid':debugger.pid,'job_member':True,'in_snapshot':False}))
                result=original_terminate(self)
                assert debugger.poll() is not None, 'Cleanup returned before the late job member was terminal'
                return result
            finally:
                debugger.wait(timeout=5)
        _pdf_process._OwnedProcesses.terminate=add_unobserved_job_dependency
    if mode=='worker_exit_dependency':
        import ctypes
        from ctypes import wintypes
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def observe_worker_owned_exit(self):
            self.observe()
            assert self.process.poll() is None, 'The worker dependency must still be alive'
            recorded=json.loads(observed.with_suffix('.worker-debugger.json').read_text())
            assert recorded['worker_pid']==recorded['debugger_pid']
            worker=psutil.Process(recorded['worker_pid'])
            assert worker.create_time()==recorded['worker_created']
            assert worker.pid in self.job.worker_pids
            handles=[]
            try:
                for pid in [recorded['worker_pid'],recorded['browser_pid']]:
                    handle=self.job.api.OpenProcess(0x00101000,False,pid)
                    if not handle: raise ctypes.WinError(ctypes.get_last_error())
                    handles.append(handle)
                    member=wintypes.BOOL()
                    self.job._check(self.job.api.IsProcessInJob(handle,self.job.handle,ctypes.byref(member)))
                    assert member.value
                    assert self.job.api.WaitForSingleObject(handle,0)==258
                observed.with_suffix('.worker-job.json').write_text(json.dumps({
                    **recorded,'worker_alive':True,'debugger_inside_worker':True,
                    'worker_and_browser_job_members':True}))
                result=original_terminate(self)
                waits=[self.job.api.WaitForSingleObject(handle,0) for handle in handles]
                assert waits==[0,0], 'Cleanup returned before native worker/browser teardown'
                observed.with_suffix('.worker-terminal.json').write_text(json.dumps({
                    'native_wait_results':waits,'cleanup_returned':True}))
                return result
            finally:
                for handle in handles: self.job.api.CloseHandle(handle)
        _pdf_process._OwnedProcesses.terminate=observe_worker_owned_exit
    if mode=='live_worker_dependency':
        import ctypes
        from ctypes import wintypes
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def require_order_independent_live_cleanup(self):
            self.observe()
            assert self.process.poll() is None, 'The real worker must still be alive'
            live=json.loads(observed.with_suffix('.live-helper.json').read_text())
            worker=psutil.Process(live['worker_pid'])
            helper=psutil.Process(live['helper_pid'])
            lineage=[worker,*worker.parents()]
            identities=[]
            for candidate in [self.root,worker,helper,psutil.Process(live['browser_pid'])]:
                native=self.job.api.OpenProcess(0x00101000,False,candidate.pid)
                if not native: raise ctypes.WinError(ctypes.get_last_error())
                try:
                    membership=wintypes.BOOL()
                    self.job._check(self.job.api.IsProcessInJob(native,self.job.handle,ctypes.byref(membership)))
                    identities.append({'pid':candidate.pid,'ppid':candidate.ppid(),
                        'exe':candidate.exe(),'cmdline':candidate.cmdline(),
                        'job_member':bool(membership.value)})
                finally: self.job.api.CloseHandle(native)
            observed.with_suffix('.live-lineage.json').write_text(json.dumps({
                'popen_pid':self.process.pid,'worker_ancestors':[item.pid for item in lineage],
                'identities':identities}))
            assert self.process.pid in [item.pid for item in lineage]
            assert helper.ppid()==worker.pid
            assert helper.create_time()==live['helper_created']
            assert helper.pid in self.observed
            handle=self.job.api.OpenProcess(0x00101000,False,helper.pid)
            if not handle: raise ctypes.WinError(ctypes.get_last_error())
            try:
                member=wintypes.BOOL()
                self.job._check(self.job.api.IsProcessInJob(handle,self.job.handle,ctypes.byref(member)))
                assert member.value
                browser=self.observed.pop(live['browser_pid'])
                # A descendant snapshot has no dependency-order guarantee.
                self.observed[browser.pid]=browser
                observed.with_suffix('.live-job.json').write_text(json.dumps({
                    **live,'worker_alive':True,'helper_is_worker_child':True,
                    'helper_is_job_member':True,'observation_order':list(self.observed),
                    'protected_worker_pids':list(getattr(self.job,'worker_pids',()))}))
                result=original_terminate(self)
                terminal=self.job.api.WaitForSingleObject(handle,0)
                assert terminal==0, 'Cleanup returned before the owned helper was terminal'
                observed.with_suffix('.live-terminal.json').write_text(json.dumps({
                    'helper_wait_result':terminal,'cleanup_returned':True}))
                return result
            finally:
                self.job.api.CloseHandle(handle)
        _pdf_process._OwnedProcesses.terminate=require_order_independent_live_cleanup
    if mode=='unresponsive_startup': _pdf_process._RENDER_TIMEOUT=30
    if mode=='slow_shutdown':
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def record_graceful_before_escalation(self):
            browser_pid=int(next(iter(json.loads(observed.read_text()))))
            try:
                browser=psutil.Process(browser_pid)
                running=browser.is_running() and browser.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess: running=False
            result=original_terminate(self)
            observed.with_suffix('.slow-termination.json').write_text(json.dumps({
                'browser_live_before_owned_cleanup':running,
                'owned_cleanup_returned':True,
            }))
            return result
        _pdf_process._OwnedProcesses.terminate=record_graceful_before_escalation
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
    if mode=='profile_readonly':
        import stat
        original_terminate=_pdf_process._OwnedProcesses.terminate
        def mark_real_profile_readonly_then_terminate(self):
            profile=Path(json.loads(observed.with_suffix('.profiles.json').read_text())[0])
            profile.mkdir(exist_ok=True)
            candidate=profile/'diagnostic-readonly.txt'
            candidate.write_bytes(b'Retained read-only renderer artifact')
            candidate.chmod(stat.S_IREAD)
            attributes=candidate.stat().st_file_attributes
            assert attributes & stat.FILE_ATTRIBUTE_READONLY
            try: candidate.unlink()
            except PermissionError as error:
                assert error.winerror==5
                observed.with_suffix('.readonly.json').write_text(json.dumps({
                    'path':str(candidate),'attributes':attributes,'winerror':error.winerror,
                    'directory':str(Path(self.process.args[-2]).parent)}))
            else: raise AssertionError('Actual read-only profile file allowed deletion')
            return original_terminate(self)
        _pdf_process._OwnedProcesses.terminate=mark_real_profile_readonly_then_terminate
    if mode in ('shared_status','permanent_status','profile_sharing'):
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
        if mode=='profile_sharing':
            original_rmtree=_pdf_process.shutil.rmtree
            profile_directory=None
            cleanup_denials=[]
            def observe_actual_cleanup(directory,*args,**kwargs):
                try: return original_rmtree(directory,*args,**kwargs)
                except OSError as error:
                    if Path(directory)==profile_directory and sharing_handles:
                        cleanup_denials.append(getattr(error,'winerror',None))
                        observed.with_suffix('.fault.json').write_text(json.dumps({
                            'denials':actual_denials,
                            'cleanup_denials':cleanup_denials,
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
                candidate=next(path for path in profile.rglob('*') if path.is_file())
                handle=api.CreateFileW(str(candidate),0x80000000,0x00000003,None,3,0x80,None)
                if handle==ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                sharing_handles.append(handle)
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
        if mode in ('acknowledged_native_exit','close_error','close_hang','held_status','shared_status','profile_sharing','slow_shutdown','unresponsive_startup','close_command_hang_slow_shutdown','pipe_number_reuse','live_worker_dependency','worker_exit_dependency'): raise
        if mode=='permanent_status': assert isinstance(error,PermissionError),repr(error)
        if mode=='cancel': assert isinstance(error,KeyboardInterrupt),repr(error)
        if mode in ('exit7','profile_readonly','job_dependency'): assert 'code 7' in str(error),repr(error)
        if mode=='job_dependency':
            assert observed.with_suffix('.late-job.json').exists(),expected_render_error
            assert not any('PDF process cleanup failed' in note or 'PDF profile cleanup failed' in note
                           for note in expected_render_error['notes']),expected_render_error
        if mode=='profile_readonly':
            assert observed.with_suffix('.readonly.json').exists(),expected_render_error
            assert not any('PDF process cleanup failed' in note or 'PDF profile cleanup failed' in note
                           for note in expected_render_error['notes']),expected_render_error
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
        assert mode in ('acknowledged_native_exit','close_error','close_hang','close_command_hang','held_status','shared_status','profile_sharing','slow_shutdown','unresponsive_startup','close_command_hang_slow_shutdown','pipe_number_reuse','live_worker_dependency','worker_exit_dependency'), 'Rendering defect was swallowed'
        pages=PdfReader(result).pages
        assert len(pages)==2
        assert all(f'Owned PDF page {i+1}' in page.extract_text() for i,page in enumerate(pages))
        assert all(abs(float(page.mediabox.width)-450)<0.1 for page in pages)
        assert all(abs(float(page.mediabox.height)-337.5)<1 for page in pages)
        if mode=='close_command_hang_slow_shutdown':
            assert json.loads(observed.with_suffix('.graceful.json').read_text())=={'browser_close_sent':True}
    if mode=='slow_shutdown':
        observation=json.loads(observed.with_suffix('.slow-close.json').read_text())
        assert observation['resumed_before_pipe_close'],'Recovering browser lost its command channel before resume'
        assert observation['browser_close_sent'],'Graceful fallback did not transmit Browser.close'
        cleanup=json.loads(observed.with_suffix('.slow-termination.json').read_text())
        assert cleanup['owned_cleanup_returned'],'Owned process cleanup did not finish'
    if mode=='acknowledged_native_exit':
        assert len(external)==1
        assert external[0].wait(timeout=5)==0, 'Renderer cleanup killed the external native witness'
    assert originals==[figure.to_json() for figure in figures]
    if mode=='unresponsive_startup':
        assert json.loads(observed.with_suffix('.fault.json').read_text())=={'stage':'initial_page_command_unresponsive'}
    if mode=='held_status': assert len(read_handles)==1 and all(handle.closed for handle in read_handles)
    if mode in ('shared_status','permanent_status'):
        assert sharing_started and actual_denials, 'No real Windows sharing denial was exercised'
        assert not sharing_handles and all(not timer.is_alive() for timer in sharing_timers)
    if mode=='profile_sharing':
        assert actual_denials==[32] and not sharing_handles
        assert 32 in cleanup_denials,'No actual directory cleanup encountered the sharing denial'
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
    assert not running, {'reason':'Owned browser processes remain after export','running':running}
    try: psutil.Process(neighbor.pid).wait(timeout=0)
    except psutil.TimeoutExpired: pass
    else: raise AssertionError('Process-wait control accepted a live unrelated child')
    profiles=json.loads(observed.with_suffix('.profiles.json').read_text())
    assert all(not Path(profile).exists() for profile in profiles), {
        'profiles':profiles,'expected_render_error':expected_render_error}
    if mode in ('close_error','close_command_hang','render_error','render_hang','cancel'):
        assert observed.with_suffix('.graceful.json').exists(), 'Close failure skipped graceful browser shutdown'
    if mode in ('close_error','close_command_hang','render_hang','pipe_number_reuse','live_worker_dependency','worker_exit_dependency'):
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
    if mode=='profile_readonly' and observed.with_suffix('.readonly.json').exists():
        import stat
        readonly=json.loads(observed.with_suffix('.readonly.json').read_text())
        candidate=Path(readonly['path'])
        if candidate.exists(): candidate.chmod(stat.S_IWRITE)
        directory=Path(readonly['directory'])
        if directory.exists(): _pdf_process.shutil.rmtree(directory)
    if mode=='profile_sharing':
        release_sharing()
        for timer in sharing_timers: timer.join(timeout=1)
        if profile_directory is not None and profile_directory.exists():
            original_rmtree(profile_directory)
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
    with (
        tempfile.TemporaryDirectory(prefix="ml4t-pdf-probe-") as directory,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output,
    ):
        script = Path(directory) / "probe.py"
        script.write_text(code, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(script), *arguments], stdout=output, stderr=output
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


def _defer_native_job_termination(probe, *, omit_membership=False, serial=False, lost_exit=False):
    marker = "    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    fault = r"""
    import inspect,textwrap
    code=textwrap.dedent(inspect.getsource(_pdf_process._WindowsJob.terminate))
    stop='        self._check(self.api.TerminateJobObject(self.handle, 120))'+chr(10)+'        job_terminated = True'
    accounting='        while True:'+chr(10)+'            accounting = Accounting()'
    assert code.count(stop)==code.count(accounting)==1
    code=code.replace(stop,'',1).replace(accounting,stop+chr(10)+accounting,1)
    loop='        for pid, handle, membership, identity, can_terminate in handles:'
    assert code.count(loop)==1
    old_loop=chr(10).join([
        '        ordered = list(reversed(handles))',
        '        ordered.sort(key=lambda item: item[0] in self.worker_pids)',
        '        for item in ordered:',
        '            pid, handle, membership, identity, can_terminate = item',
        '            request_termination(item)',
    ])
    code=code.replace(loop,old_loop,1)
    if OMIT_MEMBERSHIP:
        begin=code.index('        capacity = 64'+chr(10))
        end=code.index('        for process in processes:'+chr(10),begin)
        code=code[:begin]+'        if worker_exited:'+chr(10)+textwrap.indent(code[begin:end],'    ')+code[end:]
    if SERIAL:
        intended='for item in reversed(handles):'
        assert code.count(intended)==1
        code=code.replace(intended,'for item in ():')
    namespace={}
    exec(compile(code,_pdf_process.__file__,'exec'),vars(_pdf_process),namespace)
    wrong_order=namespace['terminate']
    def wait_before_owned_job_stop(self,processes,*,worker_exited):
        return wrong_order(self,processes,worker_exited=False if LOST_EXIT else worker_exited)
    _pdf_process._WindowsJob.terminate=wait_before_owned_job_stop
"""
    fault = fault.replace("OMIT_MEMBERSHIP", repr(omit_membership))
    fault = fault.replace("SERIAL", repr(serial)).replace("LOST_EXIT", repr(lost_exit))
    return probe.replace(marker, marker + "\n" + fault)


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows read-only profile required")
def test_readonly_browser_profile_is_removed_after_actual_worker_crash(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    result = _run_pdf_probe(_PROBE, ["profile_readonly", str(target), "0"])
    observation = json.loads(target.with_suffix(".processes.readonly.json").read_text())
    assert observation["winerror"] == 5 and observation["attributes"] & 1
    assert result["running"] == [] and result["neighbor_alive"]


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows job and debug events required")
def test_crashed_worker_cleanup_stops_late_job_members_before_process_waits(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    result = _run_pdf_probe(_PROBE, ["job_dependency", str(target), "0", "0.1", debugger])
    observation = json.loads(target.with_suffix(".processes.late-job.json").read_text())
    assert observation["worker_exit"] == 7 and observation["job_member"]
    assert not observation["in_snapshot"]
    assert result["running"] == [] and result["neighbor_alive"]


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows job and debug events required")
def test_lost_worker_exit_evidence_leaves_a_real_kernel_exit_pending(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    probe = _defer_native_job_termination(_PROBE, omit_membership=True, lost_exit=True)
    with pytest.raises(AssertionError) as failure:
        _run_pdf_probe(probe, ["job_dependency", str(target), "0", "0.1", debugger])
    assert "code 7" in str(failure.value)
    assert "not terminal before job termination" in str(failure.value)
    blocked = json.loads(target.with_suffix(".processes.kernel-exit.json").read_text())
    assert blocked["event_exit_code"] == blocked["process_exit_code"] == 120
    assert blocked["wait_result"] == 258


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows sharing denial required")
def test_profile_cleanup_deadline_preserves_existing_report_with_real_sharing_denial(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    previous = b"Existing report must survive failed profile cleanup"
    target.write_bytes(previous)
    probe = _PROBE.replace(
        "threading.Timer(.2,release_sharing)", "threading.Timer(3,release_sharing)"
    )
    with pytest.raises(AssertionError) as failure:
        _run_pdf_probe(probe, ["profile_sharing", str(target), "0"])
    assert "WinError 32" in str(failure.value)
    error = json.loads(target.with_suffix(".processes.error.json").read_text())
    fault = json.loads(target.with_suffix(".processes.fault.json").read_text())
    assert error["type"] == "PermissionError" and "WinError 32" in error["message"]
    assert fault["denials"] == [32] and 32 in fault["cleanup_denials"]
    assert target.read_bytes() == previous


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


@pytest.mark.skipif(sys.platform != "win32", reason="Actual inherited Windows pipe I/O required")
def test_browser_pipe_shutdown_does_not_change_a_pending_inherited_read(tmp_path):
    import ctypes
    import threading
    import time
    import types
    from ctypes import wintypes

    pytest.importorskip("kaleido")
    from choreographer.channels import Pipe

    from ml4t.diagnostic.visualization._pdf_worker import _PageEventsKaleido

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.GetThreadIOPendingFlag.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    api.GetThreadIOPendingFlag.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL

    def exercise(directory, *, restore_vendor_mode_change=False):
        directory.mkdir()
        renderer = _PageEventsKaleido(n=1, timeout=90)
        channel = renderer._channel
        if restore_vendor_mode_change:
            channel._unblock_fd = types.MethodType(Pipe._unblock_fd, channel)
        renderer._browser_impl.pre_open()
        cli = renderer._browser_impl.get_cli()
        handles = next(arg for arg in cli if arg.startswith("--remote-debugging-io-pipes="))
        inherited_read = handles.split("=", 1)[1].split(",", 1)[0]
        ready = directory / "peer.json"
        code = """
import _winapi,ctypes,json,sys,threading,time
from ctypes import wintypes
from pathlib import Path
api=ctypes.WinDLL('kernel32',use_last_error=True)
api.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
api.OpenThread.restype=wintypes.HANDLE
api.GetThreadIOPendingFlag.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
api.GetThreadIOPendingFlag.restype=wintypes.BOOL
api.CloseHandle.argtypes=[wintypes.HANDLE]
tid=threading.get_native_id()
def notify_actual_read():
    handle=api.OpenThread(0x0040,False,tid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    try:
        pending=wintypes.BOOL()
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if not api.GetThreadIOPendingFlag(handle,ctypes.byref(pending)):
                raise ctypes.WinError(ctypes.get_last_error())
            if pending.value:
                ready=Path(sys.argv[2])
                temporary=ready.with_suffix('.new')
                temporary.write_text(json.dumps({'tid':tid,'actual_read_pending':True}))
                temporary.replace(ready)
                return
            time.sleep(.01)
        raise RuntimeError('Native peer read never became pending')
    finally:
        api.CloseHandle(handle)
threading.Thread(target=notify_actual_read,daemon=True).start()
try:
    value,error=_winapi.ReadFile(int(sys.argv[1]),1)
except OSError as error:
    result={'read':'','error':error.winerror}
else:
    result={'read':value.hex(),'error':error}
print(json.dumps(result),flush=True)
"""
        peer = subprocess.Popen(
            [sys._base_executable, "-I", "-c", code, inherited_read, str(ready)],
            close_fds=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        thread_handle = None
        closer = None
        errors = []
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "Actual inherited pipe peer never started"
            tid = json.loads(ready.read_text())["tid"]
            thread_handle = api.OpenThread(0x0040, False, tid)
            if not thread_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            pending = wintypes.BOOL()
            while time.monotonic() < deadline:
                if not api.GetThreadIOPendingFlag(thread_handle, ctypes.byref(pending)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if pending.value:
                    break
                time.sleep(0.01)
            assert pending.value and peer.poll() is None, (
                "Peer did not enter actual pending ReadFile"
            )
            channel.open()

            def close_actual_channel():
                try:
                    channel.close()
                except BaseException as error:
                    errors.append(repr(error))

            closer = threading.Thread(target=close_actual_channel, daemon=True)
            closer.start()
            closer.join(timeout=1)
            blocked = closer.is_alive()
            if blocked:
                if not api.GetThreadIOPendingFlag(thread_handle, ctypes.byref(pending)):
                    raise ctypes.WinError(ctypes.get_last_error())
                assert pending.value, "Shutdown delay was not the real pending peer read"
                os.write(channel._write_to_browser, b"x")
            closer.join(timeout=5)
            assert not closer.is_alive() and not errors, errors
            output, stderr = peer.communicate(timeout=5)
            assert peer.returncode == 0, stderr
            result = {
                "blocked_until_peer_input": blocked,
                "initial_native_io_pending": True,
                "peer_result": json.loads(output),
                "restored_vendor_mode": restore_vendor_mode_change,
            }
            (directory / "result.json").write_text(json.dumps(result))
            return result
        finally:
            if peer.poll() is None:
                peer.kill()
            peer.wait(timeout=5)
            if closer is not None:
                closer.join(timeout=5)
            if thread_handle:
                api.CloseHandle(thread_handle)
            channel.close()
            renderer._browser_impl.clean()

    actual = exercise(tmp_path / "actual")
    assert not actual["blocked_until_peer_input"], (
        "Inherited browser pipe mode change blocked shutdown while its actual peer read was pending"
    )
    assert actual["peer_result"] == {"read": "", "error": 109}
    restored = exercise(tmp_path / "restored", restore_vendor_mode_change=True)
    assert restored["blocked_until_peer_input"]
    assert restored["peer_result"] == {"read": "78", "error": 0}


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual Windows acknowledged native exit required"
)
def test_external_native_exit_hold_cannot_bypass_kernel_cleanup_budget(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    previous = b"Existing report must survive externally blocked native exit"
    target.write_bytes(previous)
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    with pytest.raises(AssertionError, match="not terminal after job termination.*wait result 258"):
        _run_pdf_probe(_PROBE, ["acknowledged_native_exit", str(target), "0", ".1", debugger])
    held = json.loads(target.with_suffix(".processes.ack-kernel-exit.json").read_text())
    terminal = json.loads(target.with_suffix(".processes.ack-after-cleanup.json").read_text())
    request = json.loads(target.with_suffix(".processes.graceful-request.json").read_text())
    control = json.loads(target.with_suffix(".processes.ack-control.json").read_text())
    attachment = json.loads(target.with_suffix(".processes.ack-debugger-ready.json").read_text())
    assert held["wait_result"] == terminal["native_wait_result"] == 258
    assert held["pid"] == terminal["pid"] == control["browser_pid"]
    assert not terminal["cleanup_returned"]
    assert request == {"browser_close_sent": True}
    with pytest.raises(FileNotFoundError):
        target.with_suffix(".processes.graceful.json").read_text()
    assert control["outside_renderer_job"] and control["release_delay"] == 30
    assert "error" not in attachment, attachment
    assert target.read_bytes() == previous


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual worker-owned Windows debug event required"
)
def test_worker_owned_native_exit_dependency_stops_before_process_waits(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    result = _run_pdf_probe(_PROBE, ["worker_exit_dependency", str(target), "0", ".1", debugger])
    owned = json.loads(target.with_suffix(".processes.worker-job.json").read_text())
    terminal = json.loads(target.with_suffix(".processes.worker-terminal.json").read_text())
    assert owned["worker_alive"] and owned["debugger_inside_worker"]
    assert owned["worker_and_browser_job_members"]
    assert terminal == {"native_wait_results": [0, 0], "cleanup_returned": True}
    assert result["running"] == [] and result["neighbor_alive"]
    assert result["successful_owned_pipe_closes"] == [1, 1, 1, 1]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual worker-owned Windows debug event required"
)
def test_waiting_before_job_stop_retains_worker_owned_native_exit(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    previous = b"Existing report must survive worker-dependent cleanup failure"
    target.write_bytes(previous)
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    probe = _defer_native_job_termination(_PROBE)
    with pytest.raises(AssertionError) as failure:
        _run_pdf_probe(probe, ["worker_exit_dependency", str(target), "0", ".1", debugger])
    assert "not terminal before job termination" in str(failure.value)
    blocked = json.loads(target.with_suffix(".processes.kernel-exit.json").read_text())
    assert blocked["event_exit_code"] == blocked["process_exit_code"] == 120
    assert blocked["wait_result"] == 258
    owned = json.loads(target.with_suffix(".processes.worker-job.json").read_text())
    assert owned["worker_alive"] and owned["debugger_inside_worker"]
    assert owned["worker_and_browser_job_members"]
    assert target.read_bytes() == previous


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows job and debug events required")
def test_unknown_worker_exit_still_stops_all_native_job_members(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    marker = "    _pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    replacement = r"""
    actual_stop=_pdf_process._WindowsJob.terminate
    def absent_worker_exit_metadata(self,processes,*,worker_exited):
        return actual_stop(self,processes,worker_exited=False)
    _pdf_process._WindowsJob.terminate=absent_worker_exit_metadata
"""
    probe = _PROBE.replace(marker, marker + "\n" + replacement)
    result = _run_pdf_probe(probe, ["job_dependency", str(target), "0", ".1", debugger])
    error = json.loads(target.with_suffix(".processes.error.json").read_text())
    assert "code 7" in error["message"]
    assert not any("cleanup failed" in note for note in error["notes"])
    assert result["running"] == [] and result["neighbor_alive"]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual live Windows worker and debug events required"
)
def test_live_worker_stops_all_observed_dependencies_before_process_waits(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    result = _run_pdf_probe(_PROBE, ["live_worker_dependency", str(target), "0", ".1", debugger])
    observed = json.loads(target.with_suffix(".processes.live-job.json").read_text())
    terminal = json.loads(target.with_suffix(".processes.live-terminal.json").read_text())
    assert observed["worker_alive"] and observed["helper_is_worker_child"]
    assert observed["helper_is_job_member"]
    assert observed["worker_pid"] in observed["protected_worker_pids"]
    assert observed["observation_order"].index(observed["helper_pid"]) < observed[
        "observation_order"
    ].index(observed["browser_pid"])
    assert terminal == {"helper_wait_result": 0, "cleanup_returned": True}
    assert result["running"] == [] and result["neighbor_alive"]
    assert result["successful_owned_pipe_closes"] == [1, 1, 1, 1]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual live Windows worker and debug events required"
)
def test_serial_live_worker_termination_leaves_a_real_kernel_exit_pending(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    target = tmp_path / "report.pdf"
    previous = b"Existing report must survive failed native cleanup"
    target.write_bytes(previous)
    debugger = str(Path(__file__).with_name("_windows_pdf_debugger.py"))
    probe = _defer_native_job_termination(_PROBE, serial=True)
    with pytest.raises(AssertionError) as failure:
        _run_pdf_probe(probe, ["live_worker_dependency", str(target), "0", ".1", debugger])
    assert "not terminal before job termination" in str(failure.value)
    blocked = json.loads(target.with_suffix(".processes.kernel-exit.json").read_text())
    assert blocked["event_exit_code"] == blocked["process_exit_code"] == 120
    assert blocked["wait_result"] == 258
    observed = json.loads(target.with_suffix(".processes.live-job.json").read_text())
    assert observed["worker_alive"] and observed["helper_is_job_member"]
    assert target.read_bytes() == previous


def test_real_early_pipe_closure_is_still_rejected_when_owned_cleanup_succeeds(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    fault = r"""
    original_worker=Path(_pdf_process.__file__).with_name('_pdf_worker.py')
    code=original_worker.read_text(encoding='utf-8')
    intended='timeout=max(0, close_deadline - asyncio.get_running_loop().time()) / 2,'
    assert code.count(intended)==1
    code=code.replace(intended,'timeout=0.05,')
    bad_worker=Path(__file__).with_name('early_pipe_worker.py')
    bad_worker.write_text(code,encoding='utf-8')
    _pdf_process._BOOTSTRAP=_pdf_process._BOOTSTRAP.replace(
        'worker=sys.argv.pop(1)', 'sys.argv.pop(1);worker='+repr(str(bad_worker)))
"""
    anchor = "_pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    probe = _PROBE.replace(anchor, anchor + fault)
    with pytest.raises(
        AssertionError, match="Recovering browser lost its command channel before resume"
    ):
        _run_pdf_probe(probe, ["slow_shutdown", str(tmp_path / "report.pdf"), "0"])
    observation = json.loads((tmp_path / "report.processes.slow-close.json").read_text())
    assert observation["browser_close_sent"]
    assert not observation["resumed_before_pipe_close"]
    cleanup = json.loads((tmp_path / "report.processes.slow-termination.json").read_text())
    assert cleanup["owned_cleanup_returned"]


def test_unresponsive_browser_after_closed_channel_requires_verified_os_cleanup(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    psutil = pytest.importorskip("psutil")
    target = tmp_path / "report.pdf"
    probe = _PROBE.replace(
        "threading.Timer(1,resume).start()",
        "observed.with_suffix('.unresponsive-peer.json').write_text(json.dumps({"
        "'pid':browser.pid,'created':browser.create_time(),'status':browser.status()}))",
    ).replace(
        "print('PDF close control: resume timer started',time.monotonic(),file=sys.stderr,flush=True)",
        "print('PDF close control: browser stays suspended after channel close',file=sys.stderr,flush=True)",
    )
    original = "_pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    observation = r"""
    actual_termination_for_unresponsive_control=_pdf_process._OwnedProcesses.terminate
    def record_unresponsive_before_cleanup(self):
        browser_pid=int(next(iter(json.loads(observed.read_text()))))
        browser=psutil.Process(browser_pid)
        observed.with_suffix('.os-cleanup.json').write_text(json.dumps({
            'pid':browser.pid,'created':browser.create_time(),'status':browser.status()}))
        return actual_termination_for_unresponsive_control(self)
    _pdf_process._OwnedProcesses.terminate=record_unresponsive_before_cleanup
"""
    probe = probe.replace(original, original + observation)
    result = _run_pdf_probe(probe, ["close_command_hang_slow_shutdown", str(target), "0"])
    fault = json.loads(target.with_suffix(".processes.unresponsive-peer.json").read_text())
    cleanup = json.loads(target.with_suffix(".processes.os-cleanup.json").read_text())
    assert fault == cleanup
    assert fault["status"] == psutil.STATUS_STOPPED
    assert result["running"] == [] and result["neighbor_alive"]


def test_bounded_shutdown_fault_is_not_reapplied_by_the_actual_pipe_error_callback(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    begin = _PROBE.index("async def closed(self,*args,**kwargs):")
    end = _PROBE.index("kaleido.Kaleido.open=opened", begin)
    callback = _PROBE[begin:end].replace("mode=='slow_shutdown'", "mode=='close_error'")
    entry = "    if mode=='termination_error':"
    record = r"""
    trace_path=observed.with_suffix('.bounded-shutdown.json')
    trace=json.loads(trace_path.read_text()) if trace_path.exists() else {'entries':0,'suspensions':[]}
    trace['entries']+=1
    trace_path.write_text(json.dumps(trace))
"""
    assert callback.count(entry) == 1
    callback = callback.replace(entry, record + entry)
    pause = "        process.suspend()"
    observe_pause = r"""
        trace=json.loads(trace_path.read_text())
        trace['suspensions'].append({'pid':process.pid,'created':process.create_time()})
        trace_path.write_text(json.dumps(trace))
"""
    assert callback.count(pause) == 1
    callback = callback.replace(pause, pause + observe_pause)
    sent = "            if message.get('method')=='Browser.close': close_sent.append(True)"
    record_sent = r"""
            if message.get('method')=='Browser.close':
                close_sent.append(True)
                trace=json.loads(trace_path.read_text())
                trace['browser_close_sent']=True
                trace_path.write_text(json.dumps(trace))
                if len(close_sent)==1:
                    threading.Timer(.1,self._channel.close).start()
"""
    assert callback.count(sent) == 1
    callback = callback.replace(sent, record_sent)
    probe = _PROBE[:begin] + callback + _PROBE[end:]
    acknowledgement = "assert observed.with_suffix('.graceful.json').exists(), 'Close failure skipped graceful browser shutdown'"
    request = "assert json.loads(observed.with_suffix('.bounded-shutdown.json').read_text())['browser_close_sent'], 'EOF control never transmitted the real shutdown request'"
    assert probe.count(acknowledgement) == 1
    probe = probe.replace(acknowledgement, request)
    target = tmp_path / "report.pdf"
    error = None
    try:
        result = _run_pdf_probe(probe, ["close_error", str(target), "0"])
    except AssertionError as exc:
        error = exc
    trace = json.loads(target.with_suffix(".processes.bounded-shutdown.json").read_text())
    assert trace["entries"] >= 2, "The actual closed channel did not reenter vendor shutdown"
    assert len(trace["suspensions"]) == 1, trace
    assert trace["browser_close_sent"]
    assert error is None, str(error)
    assert result["running"] == [] and result["neighbor_alive"]
    assert result["successful_owned_pipe_closes"] == [1, 1, 1, 1]


def test_browser_fault_setup_waits_for_actual_script_entry_instead_of_elapsed_sleep(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    result = _run_pdf_probe(
        _PROBE, ["unresponsive_startup", str(tmp_path / "report.pdf"), "0", "2"]
    )
    assert result["observed_browser_processes"] > 0
    assert result["running"] == [] and result["neighbor_alive"]


def test_browser_fault_setup_can_replace_an_already_blocked_initial_target(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    blocked_initial = r"""
        if not startup_armed and self.tabs:
            initial=next(iter(self.tabs.values()))
            await initial.send_command('Runtime.enable')
            await initial.send_command('Runtime.addBinding',params={'name':'initialAlreadyBlocked'})
            entered=asyncio.Queue()
            async def initial_entry(message):
                if message.get('params',{}).get('name')=='initialAlreadyBlocked':
                    entered.put_nowait(message)
            initial.subscribe('Runtime.bindingCalled',initial_entry)
            evaluation=asyncio.create_task(initial.send_command('Runtime.evaluate',params={
                'expression':"initialAlreadyBlocked('entered');while(true){}",
            }))
            try: await entered.get()
            finally:
                initial.unsubscribe('Runtime.bindingCalled')
                evaluation.cancel()
                try: await evaluation
                except (Exception,asyncio.CancelledError): pass
            observed.with_suffix('.initial-blocked.json').write_text(json.dumps({
                'target_id':initial.target_id,'actual_script_entered':True}))
"""
    anchor = "        result=await original_populate(self,*args,**kwargs)"
    assert _PROBE.count(anchor) == 1
    probe = _PROBE.replace(anchor, anchor + blocked_initial)
    target = tmp_path / "report.pdf"
    result = _run_pdf_probe(probe, ["unresponsive_startup", str(target), "0"])
    initial = json.loads(target.with_suffix(".processes.initial-blocked.json").read_text())
    setup = json.loads(target.with_suffix(".processes.setup.json").read_text())
    assert initial["actual_script_entered"]
    assert initial["target_id"] in setup["replaced_target_ids"]
    assert initial["target_id"] != setup["target_id"]
    assert result["running"] == [] and result["neighbor_alive"]


def test_exporter_reusing_the_blocked_initial_target_still_fails(tmp_path):
    pytest.importorskip("kaleido")
    pytest.importorskip("pypdf")
    fault = r"""
    original_worker=Path(_pdf_process.__file__).with_name('_pdf_worker.py')
    code=original_worker.read_text(encoding='utf-8')
    intended='''            previous = list(self.tabs)
            selected = [await self.create_tab(url="about:blank", window=True)]
            for target_id in previous:
                await self.close_tab(target_id)'''
    assert code.count(intended)==1
    code=code.replace(intended,'            selected = list(self.tabs.values())')
    bad_worker=Path(__file__).with_name('blocked_initial_worker.py')
    bad_worker.write_text(code,encoding='utf-8')
    _pdf_process._BOOTSTRAP=_pdf_process._BOOTSTRAP.replace(
        'worker=sys.argv.pop(1)', 'sys.argv.pop(1);worker='+repr(str(bad_worker)))
"""
    anchor = "_pdf_process._BOOTSTRAP=patch+chr(10)+_pdf_process._BOOTSTRAP"
    probe = _PROBE.replace(anchor, anchor + fault)
    target = tmp_path / "report.pdf"
    with pytest.raises(AssertionError, match="exceeded its progress deadline"):
        _run_pdf_probe(probe, ["unresponsive_startup", str(target), "0"])
    observation = json.loads(target.with_suffix(".processes.fault.json").read_text())
    assert observation == {"stage": "initial_page_command_unresponsive"}
    assert not target.exists()


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


def test_loaded_renderer_document_can_finish_without_a_second_load_event(tmp_path, monkeypatch):
    pytest.importorskip("kaleido")
    pypdf = pytest.importorskip("pypdf")
    import plotly.graph_objects as go
    import plotly.io as pio

    from ml4t.diagnostic.visualization import _pdf_process, export_figures_to_pdf

    marker = tmp_path / "actual-same-document-navigation.json"
    patch = r"""
import asyncio,json,choreographer,kaleido
from pathlib import Path
marker=Path(MARKER)
original_command=choreographer.Tab.send_command
original_populate=kaleido.Kaleido.populate_targets
async def populate(self):
    self._index=str(self._index)+'#diagnostic-ready'
    return await original_populate(self)
kaleido.Kaleido.populate_targets=populate
original_create=choreographer.Browser.create_tab
async def create(self,*args,**kwargs):
    tab=await original_create(self,*args,**kwargs)
    marker.write_text(json.dumps({'stage':'target_created'}))
    await original_command(tab,'Page.enable')
    marker.write_text(json.dumps({'stage':'page_enabled'}))
    preload_url=str(self._index).split('#',1)[0]
    expression=("JSON.stringify({href:location.href,expected_href:new URL("+json.dumps(preload_url)+").href,"
        "state:document.readyState,plotly:typeof Plotly!=='undefined'&&typeof Plotly.newPlot==='function',"
        "scopes:typeof kaleido_scopes!=='undefined'&&typeof kaleido_scopes.plotly==='function'})")
    async def preload_ready():
        navigation=None
        while True:
            result=await original_command(tab,'Runtime.evaluate',params={'expression':expression,'returnByValue':True})
            value=result.get('result',{}).get('result',{}).get('value')
            if isinstance(value,str):
                observed=json.loads(value)
                marker.write_text(json.dumps({'stage':'preload_readiness','observation':observed,'response':navigation}))
                if observed['href']==observed['expected_href'] and observed['state']=='complete' and observed['plotly'] and observed['scopes']:
                    return observed,navigation
            if navigation is None:
                navigation=await original_command(tab,'Page.navigate',params={'url':preload_url})
                assert 'result' in navigation and 'error' not in navigation and not navigation['result'].get('errorText'),navigation
                marker.write_text(json.dumps({'stage':'preload_navigation_sent','response':navigation}))
            await asyncio.sleep(.05)
    observed,navigation=await asyncio.wait_for(preload_ready(),10)
    marker.write_text(json.dumps({'initial_document_loaded':True,'preload':observed,
        'preload_response':navigation,'preload_navigation_sent':navigation is not None}))
    return tab
choreographer.Browser.create_tab=create
async def command(self,method,*args,**kwargs):
    response=await original_command(self,method,*args,**kwargs)
    if method=='Page.navigate' and '#diagnostic-ready' in kwargs['params']['url']:
        observation=json.loads(marker.read_text())
        observation.update({'same_document_navigation':'loaderId' not in response.get('result',{}),
            'response':response})
        marker.write_text(json.dumps(observation))
    return response
choreographer.Tab.send_command=command
""".replace("MARKER", repr(str(marker)))
    monkeypatch.setattr(
        _pdf_process,
        "_BOOTSTRAP",
        patch
        + "\n"
        + _pdf_process._BOOTSTRAP.replace("diagnostic_stop.wait(20)", "diagnostic_stop.wait(5)"),
    )
    monkeypatch.setattr(_pdf_process, "_RENDER_TIMEOUT", 15)
    monkeypatch.setattr(pio.defaults, "mathjax", False)
    figure = go.Figure(go.Scatter(x=[1, 2], y=[1, 3]))
    figure.update_layout(title="Loaded renderer without another load event")
    original = figure.to_json()
    output = export_figures_to_pdf([figure], tmp_path / "same-document.pdf", scale=1)
    observation = json.loads(marker.read_text())
    assert observation["initial_document_loaded"] and observation["same_document_navigation"]
    pages = pypdf.PdfReader(output).pages
    assert (
        len(pages) == 1 and "Loaded renderer without another load event" in pages[0].extract_text()
    )
    assert figure.to_json() == original


def test_loaded_renderer_document_setup_does_not_require_a_page_event(tmp_path, monkeypatch):
    from ml4t.diagnostic.visualization import _pdf_process

    control = tmp_path / "actual-preload-event-control.json"
    prefix = r"""
import asyncio,json,choreographer
from pathlib import Path
_preload_control_path=Path(CONTROL)
_preload_original_command=choreographer.Tab.send_command
_preload_original_create=choreographer.Browser.create_tab
_preload_control={}
_preload_load_events=[]
_preload_target_urls={}
async def _preload_create_at_renderer(self,*args,**kwargs):
    url=str(self._index).split('#',1)[0]
    kwargs['url']=url
    tab=await _preload_original_create(self,*args,**kwargs)
    _preload_target_urls[tab.target_id]=url
    return tab
choreographer.Browser.create_tab=_preload_create_at_renderer
async def _preload_load_event(message):
    _preload_load_events.append(message)
async def _preload_without_page_events(self,method,*args,**kwargs):
    if method=='Page.enable' and _preload_control and 'load_events_before_renderer_enable' not in _preload_control:
        _preload_control['load_events_before_renderer_enable']=list(_preload_load_events)
        self.unsubscribe('Page.loadEventFired')
        _preload_control_path.write_text(json.dumps(_preload_control))
    result=await _preload_original_command(self,method,*args,**kwargs)
    if method=='Page.enable' and not _preload_control and self.target_id in _preload_target_urls:
        url=_preload_target_urls[self.target_id]
        expression=("JSON.stringify({href:location.href,expected_href:new URL("+json.dumps(url)+").href,"
            "state:document.readyState,plotly:typeof Plotly!=='undefined'&&typeof Plotly.newPlot==='function',"
            "scopes:typeof kaleido_scopes!=='undefined'&&typeof kaleido_scopes.plotly==='function'})")
        async def actual_document_ready():
            while True:
                response=await _preload_original_command(self,'Runtime.evaluate',params={'expression':expression,'returnByValue':True})
                value=response.get('result',{}).get('result',{}).get('value')
                if isinstance(value,str):
                    ready=json.loads(value)
                    _preload_control_path.write_text(json.dumps({'stage':'preparing_control','observation':ready}))
                    if ready['href']==ready['expected_href'] and ready['state']=='complete' and ready['plotly'] and ready['scopes']:
                        return ready
                await asyncio.sleep(.05)
        ready=await asyncio.wait_for(actual_document_ready(),10)
        identity='prepared-renderer-'+self.target_id
        assigned=await _preload_original_command(self,'Runtime.evaluate',params={
            'expression':'window.__diagnostic_preload_identity='+json.dumps(identity),'returnByValue':True})
        assert assigned['result']['result']['value']==identity,assigned
        disabled=await _preload_original_command(self,'Page.disable')
        assert 'result' in disabled and 'error' not in disabled,disabled
        _preload_control.update({'actual_page_disable_response':disabled,
            'actual_page_ready_before_disable':ready,'prepared_document_identity':identity,
            'preload_navigations_before_renderer_enable':[]})
        self.subscribe('Page.loadEventFired',_preload_load_event)
        _preload_control_path.write_text(json.dumps(_preload_control))
    if method=='Page.navigate' and _preload_control:
        if 'load_events_before_renderer_enable' not in _preload_control:
            _preload_control['preload_navigations_before_renderer_enable'].append(result)
        if '#diagnostic-ready' in kwargs['params']['url']:
            identity=await _preload_original_command(self,'Runtime.evaluate',params={
                'expression':'window.__diagnostic_preload_identity','returnByValue':True})
            _preload_control['prepared_document_identity_at_fragment_navigation']=identity['result']['result'].get('value')
        _preload_control_path.write_text(json.dumps(_preload_control))
    return result
choreographer.Tab.send_command=_preload_without_page_events
""".replace("CONTROL", repr(str(control)))

    class WithoutPreloadPageEvents:
        def setattr(self, target, name, value):
            if target is _pdf_process and name == "_BOOTSTRAP":
                value = prefix + "\n" + value
            monkeypatch.setattr(target, name, value)

    test_loaded_renderer_document_can_finish_without_a_second_load_event(
        tmp_path, WithoutPreloadPageEvents()
    )
    observed = json.loads(control.read_text())
    assert "result" in observed["actual_page_disable_response"]
    assert observed["load_events_before_renderer_enable"] == []
    prepared = observed["actual_page_ready_before_disable"]
    assert prepared["href"] == prepared["expected_href"]
    assert prepared["state"] == "complete" and prepared["plotly"] and prepared["scopes"]
    assert (
        observed["prepared_document_identity_at_fragment_navigation"]
        == observed["prepared_document_identity"]
    ), "Preload replaced the already prepared renderer document"
    assert observed["preload_navigations_before_renderer_enable"] == []
    document = json.loads((tmp_path / "actual-same-document-navigation.json").read_text())
    assert not document["preload_navigation_sent"]
    assert document["preload"]["href"] == document["preload"]["expected_href"]
    assert document["preload"]["state"] == "complete"
    assert document["preload"]["plotly"] and document["preload"]["scopes"]
    assert document["same_document_navigation"]


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
    assert "phase='starting', completed=0, expected=1" in str(failed.value)
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
