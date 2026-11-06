"""Real local process/HTTP shutdown checks; no models or GPU are loaded."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'visualization'))
from detect_and_view import ProtocolClient
spec=importlib.util.spec_from_file_location('validation_shutdown_lab',ROOT/'visualization/validation_lab/server.py')
lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)


class ValidationShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.worker=self.root/'fake_worker.py'
        self.worker.write_text('''import json,sys,threading
for line in sys.stdin:
    req=json.loads(line)
    if req['cmd']=='hang':threading.Event().wait()
    if req['cmd']=='crash':sys.exit(3)
    print(json.dumps({'ok':True,'request_id':req['request_id']}),flush=True)
    if req['cmd']=='release':break
''',encoding='utf-8')

    def client(self):
        client=ProtocolClient(Path(sys.executable),self.worker,self.root/'worker.log')
        self.addCleanup(lambda:client.close(force=True))
        return client

    def wait_pending(self,client,count):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with client.pending_lock:
                if len(client.pending)>=count:return
            time.sleep(.01)
        self.fail('requests did not enter protocol wait')

    def test_force_close_wakes_all_pending_requests_and_is_idempotent(self):
        client=self.client()
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(client.request,{'cmd':'hang'},900) for _ in range(2)]
            self.wait_pending(client,2)
            client.close(force=True)
            for future in futures:
                with self.assertRaises(RuntimeError):future.result(timeout=3)
        client.close(force=True)
        self.assertIsNotNone(client.process.poll())
        self.assertFalse(client.reader.is_alive())
        with self.assertRaises(RuntimeError):client.request({'cmd':'ping'},900)

    def test_unexpected_worker_exit_wakes_waiter_without_900_second_timeout(self):
        client=self.client()
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(client.request,{'cmd':'crash'},900)
            with self.assertRaisesRegex(RuntimeError,'exited'):future.result(timeout=5)

    def test_normal_close_still_releases_protocol_worker(self):
        client=self.client()
        self.assertTrue(client.request({'cmd':'ping'},5)['ok'])
        client.close()
        self.assertEqual(client.process.returncode,0)

    @unittest.skipUnless(os.name=='nt','Windows pythonw integration')
    def test_pythonw_ui_uses_hidden_python_protocol_worker(self):
        pythonw=Path(sys.executable).with_name('pythonw.exe')
        client=ProtocolClient(pythonw,self.worker,self.root/'pythonw-worker.log')
        self.addCleanup(lambda:client.close(force=True))
        self.assertEqual(Path(client.process.args[0]).name.casefold(),'python.exe')
        self.assertTrue(client.request({'cmd':'ping'},5)['ok'])

    def test_http_stop_interrupts_active_job_before_executor_join(self):
        server=lab.Server(0,self.root/'active-server')
        client=self.client();server.runner.client=client
        def serve():
            with server:server.serve_forever(poll_interval=.05)
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        job=server.pool.submit(client.request,{'cmd':'hang'},900)
        self.wait_pending(client,1)
        base=f'http://127.0.0.1:{server.server_port}'
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(base+'/api/shutdown',data=b'{}',headers={'Origin':'http://example.com'}))
        self.assertEqual(error.exception.code,403)
        with urlopen(Request(base+'/api/shutdown',data=b'{}')) as response:
            self.assertTrue(json.load(response)['stopping'])
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        with self.assertRaises(RuntimeError):job.result(timeout=1)
        self.assertIsNotNone(client.process.poll())
        with self.assertRaisesRegex(RuntimeError,'停止'):
            server.runner._ensure({'model_path':{}},lambda _:None)

    def test_command_line_reuses_running_service_and_stop_exits_background_python(self):
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
        python=Path(sys.executable)
        background=python.with_name('pythonw.exe')
        if not background.is_file():background=python
        command=['-B',str(ROOT/'visualization/validation_lab/server.py'),'--port',str(port),'--output-dir',str(self.root/'cli')]
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        log=(self.root/'service.log').open('w',encoding='utf-8');self.addCleanup(log.close)
        process=subprocess.Popen([str(background),*command],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
        def cleanup():
            if process.poll() is None:process.kill()
            process.wait(timeout=5)
        self.addCleanup(cleanup)
        deadline=time.monotonic()+20
        while not lab.service_running(port):
            if process.poll() is not None:self.fail('service exited during startup')
            if time.monotonic()>deadline:self.fail('service did not start')
            time.sleep(.05)
        duplicate=subprocess.run([str(python),*command],cwd=ROOT,capture_output=True,timeout=20,creationflags=flags)
        self.assertEqual(duplicate.returncode,0)
        self.assertEqual(len(list((self.root/'cli').glob('session-*'))),1)
        self.assertIsNone(process.poll())
        stopped=subprocess.run([str(python),*command,'--stop'],cwd=ROOT,capture_output=True,timeout=20,creationflags=flags)
        self.assertEqual(stopped.returncode,0)
        self.assertEqual(process.wait(timeout=10),0)

    def test_stop_does_not_contact_foreign_service(self):
        with patch.object(lab,'service_running',return_value=False), \
             patch.object(lab,'urlopen',side_effect=AssertionError('must not stop foreign service')), \
             patch.object(sys,'argv',['server.py','--stop','--port','8770']):
            lab.main()

    def test_simultaneous_start_rechecks_existing_service_after_bind_race(self):
        with patch.object(lab,'service_running',side_effect=[False,True]), \
             patch.object(lab,'Server',side_effect=OSError('address in use')), \
             patch.object(lab.webbrowser,'open') as browser, \
             patch.object(sys,'argv',['server.py','--open','--port','8770']):
            lab.main()
        browser.assert_called_once_with('http://127.0.0.1:8770')


if __name__=='__main__':unittest.main()
