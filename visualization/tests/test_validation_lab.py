import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
import numpy as np
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('validation_lab', ROOT/'visualization/validation_lab/server.py')
lab = importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.env_overrides=kwargs.get('env_overrides',{})
        self.process=SimpleNamespace(pid=123,poll=lambda:None)
        self.timeline=[];self.log=io.StringIO();self.requests=[];self.closed=False
        self.condition=threading.Condition();self.pieces={};self.released=set();self.fail_piece=None
    def request(self, q, timeout, *, on_sent=None):
        with self.condition:
            self.requests.append(q)
            request_id='fake-'+str(len(self.requests))
            self.timeline.append({'cmd':q['cmd'],'piece_id':q.get('piece_id'),'duration_ms':1,'request_id':request_id})
            if on_sent is not None:on_sent(request_id,time.perf_counter())
            board=q.get('board_id');cmd=q['cmd']
            response=dict(ok=True,cmd=cmd,board_id=board,timings={})
            if cmd=='init_algorithm':response['threading']={'main_cpu_ids':list(range(8)),'main_reserved_cpu_ids':list(range(8))}
            if cmd=='detect_2d_piece':
                pid=q['piece_id'];response.update(piece_id=pid,results_2d={'is_ok':True})
                if pid==self.fail_piece:return dict(response,ok=False,error='synthetic piece error',message='chip aspect ratio 1.9072 does not match configured range')
                self.pieces.setdefault(board,set()).add(pid);self.condition.notify_all()
            if cmd=='detect_3d_board':
                self.condition.wait_for(lambda:len(self.pieces.get(board,set()))==2 or board in self.released,2)
                if board in self.released:return dict(response,ok=False,error='board released')
                response['results_board']={'is_ok':False,'pieces':[]}
            if cmd=='detect_3d_piece':
                pid=q['piece_id'];matched=pid in self.pieces.get(board,set())
                self.pieces.get(board,set()).discard(pid)
                response['results_piece']=dict(piece_id=pid,is_ok=matched,
                    skip_3d_reason=None if matched else 'missing_2d_result')
            if cmd=='release_board':
                self.released.add(board);self.pieces.pop(board,None);self.condition.notify_all()
            return response
    def close(self, *, force=False):self.closed=True


def init_request(*args):
    return dict(cmd='init_algorithm', config_text=json.dumps({
        'common_3d': {'parameters': {'input': {'piece_count': 2}}},
        'height_measurement_3d': {'description': 'synthetic height settings', 'thresholds': {}, 'parameters': {}},
        'segmentation_3d': {'description': 'synthetic windows', 'thresholds': {}, 'parameters': {
            'mode': 'fixed', 'expected_piece_count': 2, 'fixed_rois': [[0,0,4,2],[0,2,4,4]]}},
    }))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.rows=[]
        for kind,name,piece in [('2d','2_1.png',1),('2d','2_2.png',2),('3d','2.png',None)]:
            path=self.root/name
            Image.fromarray(np.full((4,4),33000,np.uint16)).save(path) if kind=='3d' else Image.new('RGB',(4,4)).save(path)
            self.rows.append(dict(kind=kind,name=name,path=str(path),board_id='02',piece_id=piece))

    def test_filename_guess_and_board_normalization(self):
        self.assertEqual(lab.guess_numbers('2_06.png','2d'),dict(board_id='2',piece_id=6))
        self.assertEqual(lab.guess_numbers('2.png','3d')['board_id'],'2')
        self.assertIsNone(lab.guess_numbers('custom.png','2d')['piece_id'])
        self.assertEqual(lab.board_number('002'),'2');self.assertEqual(lab.board_number(0),'0')

    def test_validation_accepts_partial_2d_and_rejects_retired_modes(self):
        board, images, depth = lab.board_inputs(self.rows[:1], "2", "2d", 2)
        self.assertEqual(board, "2")
        self.assertEqual([image["piece_id"] for image in images], [1])
        self.assertIsNone(depth)
        for mode in ("joint", "manual_board_3d"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "未知验证方式"):
                lab.board_inputs(self.rows, "2", mode, 2)
        with self.assertRaisesRegex(ValueError, "重复"):
            lab.board_inputs(self.rows + [self.rows[0]], "2", "2d", 2)

    def test_manual_numbering_and_board_isolation(self):
        rows=[*self.rows,dict(self.rows[0],board_id='3',piece_id=2)]
        self.assertEqual(len(lab.board_inputs(rows,'2','2d',2)[1]),2)
        rows[0]=dict(rows[0],piece_id=3)
        with self.assertRaisesRegex(ValueError,'片号'):lab.board_inputs(rows,'2','2d',2)

    def make_runner(self, fail=None):
        clients=[]
        def factory(*args, **kwargs):
            c=FakeClient(**kwargs);c.fail_piece=fail;clients.append(c);return c
        def mapper(stack,path,kind,piece_id=None,**kwargs):
            return dict(shm_name=path.name,**({'piece_id':piece_id,'image_size':3} if kind=='2d' else
                {'data_size':32,'data_format':'uint16','image_width':4,'image_height':4}))
        def reporter(doc,path,**kwargs):path.write_text('<html>report</html>',encoding='utf-8')
        return lab.Runner(self.root,factory,init_request,mapper,reporter),clients

    def test_runner_sends_explicit_ids_and_reuses_initialized_models(self):
        runner,clients=self.make_runner();self.addCleanup(runner.close)
        for n in range(2):
            folder=self.root/('run'+str(n));folder.mkdir()
            result=runner.run(self.rows,'2','2d',2,folder,lambda x:None)
            # New logical board after each cleanup for the fake's release tombstone.
            clients[0].released.clear()
            self.assertTrue(result['ok']);self.assertTrue(result['product_is_ok'])
            self.assertEqual(result['model_reused'],n==1)
            self.assertTrue((folder/'report.html').exists());self.assertTrue((folder/'timings.csv').exists())
            saved=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(list(range(8)),saved['benchmark']['threading']['main_reserved_cpu_ids'])
        self.assertEqual(len(clients),1)
        self.assertEqual(sum(q['cmd']=='init_algorithm' for q in clients[0].requests),1)
        requests=[q for q in clients[0].requests if q['cmd']=='detect_2d_piece']
        self.assertTrue(all(q['board_id']=='2' for q in requests))
        self.assertEqual([q['piece_id'] for q in requests],[1,2,1,2])

    def test_partial_2d_and_failure_release_board(self):
        for fail,rows in [(None,self.rows[:1]),(1,self.rows)]:
            runner,clients=self.make_runner(fail);self.addCleanup(runner.close)
            folder=self.root/('case'+str(fail));folder.mkdir()
            result=runner.run(rows,'2','2d',2,folder,lambda x:None)
            self.assertEqual(result['ok'],fail is None)
            self.assertTrue(any(q['cmd']=='release_board' for q in clients[0].requests))
            self.assertFalse(any(q['cmd'].startswith('detect_3d') for q in clients[0].requests))
            if fail is not None:
                self.assertIn('synthetic piece error',result['error'])
                self.assertIn('1.9072',result['error'])
                saved=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                self.assertNotEqual(saved['response']['message'],'board cache released')

    def test_manual_piece_rejects_invalid_depth_without_submitting_detection(self):
        Image.new("RGB", (4, 4)).save(self.rows[2]["path"])
        runner, clients = self.make_runner()
        self.addCleanup(runner.stop)
        folder = self.root / "bad-depth"
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, "请先执行"):
            runner.run_manual(self.rows, "2", "manual_piece_3d", 1, 2, folder, lambda _: None)
        runner.start_pitch("2", self.rows[2])
        with self.assertRaisesRegex(ValueError, "16位"):
            runner.run_manual(self.rows, "2", "manual_piece_3d", 1, 2, folder, lambda _: None)
        self.assertFalse(any(q["cmd"].startswith("detect_") for q in clients[0].requests))

    def test_manual_clicks_keep_pairing_and_send_one_protocol_request(self):
        runner,clients=self.make_runner();self.addCleanup(runner.stop)
        pixels=np.arange(16,dtype=np.uint16).reshape(4,4)+33000
        Image.fromarray(pixels).save(self.rows[2]['path'])

        def click(n,mode,piece=None):
            folder=self.root/str(n);folder.mkdir()
            return runner.run_manual(self.rows,'2',mode,piece,2,folder,lambda _:None)

        two_d=click(1,'manual_2d',2)
        self.assertEqual(two_d['ready_piece_ids'],[2])
        runner.start_pitch("2", self.rows[2])
        single=click(2,'manual_piece_3d',2)
        self.assertTrue(single['product_is_ok'])
        self.assertEqual(single['ready_piece_ids'],[])
        with Image.open(self.root/'2/piece-depth.png') as image:
            np.testing.assert_array_equal(np.asarray(image),pixels[2:])
        missing=click(3,'manual_piece_3d',1)
        self.assertEqual(missing['skip_3d_reason'],'missing_2d_result')
        click(4,'manual_2d',1)
        click(5,'manual_2d',2)
        self.assertTrue(click(6,'manual_piece_3d',1)['ok'])
        self.assertEqual(len(clients),1)
        detect=[(q['cmd'],q.get('piece_id')) for q in clients[0].requests if q['cmd'].startswith('detect_')]
        self.assertEqual(detect,[('detect_2d_piece',2),('detect_3d_piece',2),('detect_3d_piece',1),
            ('detect_2d_piece',1),('detect_2d_piece',2),('detect_3d_piece',1)])

    def test_returned_detection_is_kept_if_the_html_report_cannot_be_written(self):
        runner, _ = self.make_runner()
        self.addCleanup(runner.stop)
        folder = self.root/"report-write-error"
        folder.mkdir()
        with patch.object(runner, "reporter", side_effect=OSError("report file unavailable")):
            result = runner.run_manual(self.rows, "2", "manual_2d", 1, 2, folder, lambda _: None)
        self.assertTrue(result["ok"])
        self.assertTrue(result["product_is_ok"])
        self.assertIn("report file unavailable", result["artifact_errors"]["report.html"])
        saved = json.loads((folder/"result.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["response"]["results_2d"]["is_ok"])
        self.assertTrue((folder/"timings.csv").is_file())

    def test_manual_affinity_uses_local_process_pools_when_core_budget_is_too_large(self):
        cfg={'common_2d':{'parameters':{'threading':{'cpu_process_affinity':{
            'enabled':True,'physical_cores':2}}}},'common_3d':{'parameters':{'threading':{
            'main_reserved_physical_cores':2,'registration_affinity':{'enabled':True,'physical_cores':2},
            'joint_affinity':{'enabled':True,'physical_cores':4}}}}}
        cores=[{'logical_cpus':[i],'efficiency_class':0} for i in range(10)]
        with patch.object(lab.os,'name','nt'), patch('algorithm.execution.cpu_partition.physical_cores',return_value=cores):
            with patch('algorithm.execution.cpu_partition.current_cpus',return_value=list(range(10))):
                self.assertIsNone(lab.fit_manual_affinity(cfg))
            with patch('algorithm.execution.cpu_partition.current_cpus',return_value=list(range(6))):
                self.assertIn('需要 10 个',lab.fit_manual_affinity(cfg))
        self.assertFalse(cfg['common_2d']['parameters']['threading']['cpu_process_affinity']['enabled'])
        three_d=cfg['common_3d']['parameters']['threading']
        self.assertFalse(three_d['registration_affinity']['enabled'])
        self.assertFalse(three_d['joint_affinity']['enabled'])
        self.assertEqual(three_d['main_reserved_physical_cores'],0)

    def test_two_d_mode_does_not_mutate_or_require_depth_calibration(self):
        runner,clients=self.make_runner();self.addCleanup(runner.close)
        cfg=json.loads(init_request()['config_text'])
        cfg['smartray_pitch_3d']={'description':'test','thresholds':{},'parameters':{'enabled_3d':True,'scan_pitch_mm':0}}
        cfg['common_3d']['parameters']['threading']={'registration_affinity':{'enabled':True,'physical_cores':8},'joint_affinity':{'enabled':True,'physical_cores':8}}
        runner.init_builder=lambda *args:dict(cmd='init_algorithm',config_text=json.dumps(cfg))
        folder=self.root/'two-only';folder.mkdir()
        result=runner.run(self.rows[:1],'2','2d',2,folder,lambda x:None)
        self.assertTrue(result['ok']);self.assertTrue(cfg['smartray_pitch_3d']['parameters']['enabled_3d'])
        request=next(q for q in clients[0].requests if q['cmd']=='init_algorithm')
        self.assertNotIn('smartray_pitch_3d',json.loads(request['config_text']))
        effective=json.loads(request['config_text'])
        self.assertFalse(effective['common_3d']['parameters']['threading']['registration_affinity']['enabled'])
        self.assertFalse(effective['common_3d']['parameters']['threading']['joint_affinity']['enabled'])
        self.assertTrue(effective['common_2d']['parameters']['threading']['cpu_process_affinity']['enabled'])
        self.assertTrue(cfg['common_3d']['parameters']['threading']['registration_affinity']['enabled'])
        self.assertTrue(cfg['common_3d']['parameters']['threading']['joint_affinity']['enabled'])

    def test_http_uploads_defaults_and_rejects_bad_pairing_before_model_load(self):
        server=lab.Server(0,self.root/'output');threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        for p in ('/','/app.js','/pairing.js','/style.css','/api/history'):
            with urlopen(base+p) as r:self.assertEqual(r.status,200)
        q=Request(base+'/api/upload?kind=2d&name=2_6.png',data=b'fixture',headers={'Content-Type':'application/octet-stream'})
        with urlopen(q) as r:row=json.load(r)
        self.assertEqual(row['name'],'2_6.png');self.assertEqual(row['piece_id'],6)
        with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/api/history',headers={'Origin':'http://example.com'}))
        self.assertEqual(error.exception.code,403)
        q=Request(base+'/api/run',data=json.dumps({'files':[row],'board_id':'2','mode':'joint'}).encode(),headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as error:urlopen(q)
        self.assertEqual(error.exception.code,400);self.assertIsNone(server.runner.client)
        key='a'*32;folder=server.folder/key;folder.mkdir();(folder/'result.json').write_text('{"ok":true}')
        with urlopen(base+'/runs/'+key+'/result.json') as r:self.assertEqual(json.load(r),{'ok':True})

    def test_product_selection_binds_pitch_and_detection_and_records_version(self):
        project = self.root / 'project'
        config_dir = project / 'config'
        source = ROOT / 'config/products/normal'
        product_dir = config_dir / 'products' / 'A100'
        product_dir.mkdir(parents=True)
        default = dict(config_2d=str(source / '2D/config.json'), config_3d=str(source / '3D/config.json'),
                       config_inspection=str(source / 'inspection.json'))
        normal_dir = config_dir / 'products' / 'normal'
        normal_dir.mkdir()
        (normal_dir / 'config.json').write_text(json.dumps(default), encoding='utf-8')
        three = json.loads((source / '3D/config.json').read_text(encoding='utf-8'))
        three['common_3d']['parameters']['input']['piece_count'] = 2
        three['segmentation_3d']['parameters'].update(expected_piece_count=2, fixed_rois=[[0, 0, 4, 2], [0, 2, 4, 4]])
        three['smartray_pitch_3d']['parameters'].update(enabled_3d=False, angle_deg=11.0)
        calibration = product_dir / 'calibration'
        calibration.mkdir()
        profile_path = calibration / 'affine_initializer.json'
        profile = json.loads((source / '3D/calibration/affine_initializer.json').read_text(encoding='utf-8'))
        profile['product_fixture'] = 'A100'
        profile_path.write_text(json.dumps(profile), encoding='utf-8')
        (product_dir / 'measure.json').write_text(json.dumps(three), encoding='utf-8')
        (product_dir / 'config.json').write_text(json.dumps(dict(default, config_3d='measure.json')), encoding='utf-8')

        server = lab.Server(0, self.root / 'product-api')
        server.runner.stop()
        server.runner, clients = self.make_runner()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'

        def post(route, **body):
            request = Request(base + route, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(request) as response:
                return json.load(response)

        with patch.object(lab, 'ROOT', project):
            with urlopen(base + '/api/defaults') as response:
                options = json.load(response)
            self.assertEqual([product['id'] for product in options['products']], ['A100', 'normal'])
            self.assertEqual(options['product']['id'], 'normal')
            profile_path.unlink()
            with self.assertRaises(HTTPError):
                post('/api/product', product_id='A100')
            self.assertEqual(server.runner.product.id, 'normal')
            profile_path.write_text(json.dumps(profile), encoding='utf-8')
            selected = post('/api/product', product_id='A100')
            self.assertEqual(selected['piece_count'], 2)
            self.assertEqual(selected['pitch']['angle_deg'], 11)
            version = selected['product']['version']
            scope = dict(product_id='A100', product_version=version)
            post('/api/pitch/start', board_id='2', files=self.rows, **scope)
            init_config = json.loads(server.runner.product.request['config_text'])
            self.assertEqual(init_config['dynamic_affine_3d']['parameters']['calibration']['profile'], profile)
            job = post('/api/run', board_id='2', mode='manual_2d', piece_id=1, files=self.rows, **scope)
            result_file = server.folder / job['id'] / 'result.json'
            deadline = time.monotonic() + 5
            while server.jobs[job['id']]['status'] == 'running' and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(server.jobs[job['id']]['status'], 'done', server.jobs[job['id']])
            self.assertEqual(json.loads(result_file.read_text(encoding='utf-8'))['product'], selected['product'])
            group = post('/api/workflows', run_ids=[job['id']], outcome='done')
            self.assertIn('A100', group['title'])

            post('/api/product', product_id='normal')
            self.assertTrue(clients[0].closed)
            with self.assertRaises(HTTPError):
                post('/api/pitch/start', board_id='2', files=self.rows, **scope)

    def test_http_manual_action_preserves_exact_piece_and_dispatches_without_batch(self):
        server=lab.Server(0,self.root/'manual-api')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        body=dict(files=self.rows,board_id='02',mode='manual_2d',piece_id=2)
        def post(data):
            return urlopen(Request(base+'/api/run',data=json.dumps(data).encode(),
                headers={'Content-Type':'application/json'}))
        done=threading.Event()
        def manual(*args, on_sent, on_finished):
            on_sent(10.0)
            on_finished(10.6, True)
            done.set()
            return {'ok':True,'piece_id':2}
        with patch.object(server.runner,'run_manual',side_effect=manual) as exact, patch.object(server.runner,'run') as batch:
            with self.assertRaises(HTTPError) as invalid:
                post({**body,'piece_id':True})
            self.assertEqual(invalid.exception.code,400)
            with post(body) as response:
                self.assertEqual(response.status,202)
                accepted=json.load(response)
                self.assertTrue(accepted['submitted'])
                job_id=accepted['id']
            self.assertTrue(done.wait(2))
            args=exact.call_args.args
            self.assertEqual((args[1],args[2],args[3]),('2','manual_2d',2))
            self.assertEqual([row['name'] for row in args[0]],['2_2.png'])
            batch.assert_not_called()
        with urlopen(base+'/api/jobs/'+job_id) as response:
            job = json.load(response)
            self.assertEqual(job['piece_id'], 2)
            self.assertEqual(job['request_sent_at'], 10.0)
            self.assertEqual(job['request_finished_at'], 10.6)
            self.assertEqual(accepted['submitted_at'], job['submitted_at'])
            self.assertAlmostEqual(job['received_at'] - job['submitted_at'], .6, places=5)

    def test_http_submits_same_piece_3d_while_2d_result_is_pending(self):
        release_2d = threading.Event()

        class DelayedClient(FakeClient):
            def request(self, q, timeout, *, on_sent=None):
                reply = super().request(q, timeout, on_sent=on_sent)
                if q['cmd'] == 'detect_2d_piece' and not release_2d.wait(5):
                    raise TimeoutError("test did not release 2D")
                return reply

        server = lab.Server(0, self.root/'overlapping-api')
        runner, _ = self.make_runner()
        runner.client_factory = DelayedClient
        server.runner = runner
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(release_2d.set)
        base = f'http://127.0.0.1:{server.server_port}'

        def submit(mode):
            payload = dict(files=self.rows, board_id='02', mode=mode, piece_id=2)
            request = Request(base+'/api/run', data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=3) as response:
                accepted = json.load(response)
            self.assertTrue(accepted['submitted'], accepted)
            return accepted['id']

        def completed(job_id):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with urlopen(base+'/api/jobs/'+job_id, timeout=3) as response:
                    item = json.load(response)
                if item['status'] != 'running':
                    self.assertEqual('done', item['status'], item)
                    return json.loads((server.folder/job_id/'result.json').read_text(encoding='utf-8'))
                time.sleep(.01)
            self.fail("job did not finish")

        with patch.object(lab, 'defaults', return_value=dict(piece_count=2)), \
             patch.object(runner, '_manual_piece_depth', return_value=(Path(self.rows[2]['path']), {})):
            two_d = submit('manual_2d')
            three_d = submit('manual_piece_3d')
            try:
                with urlopen(base+'/api/jobs/'+two_d, timeout=3) as response:
                    self.assertEqual('running', json.load(response)['status'])
                result_3d = completed(three_d)
                self.assertTrue(result_3d['response']['results_piece']['is_ok'])
            finally:
                release_2d.set()
            result_2d = completed(two_d)
        self.assertEqual(['detect_2d_piece'], [row['cmd'] for row in result_2d['timeline']])
        self.assertEqual(['detect_3d_piece'], [row['cmd'] for row in result_3d['timeline']])
        self.assertNotIn('2', runner._manual_boards)

    def test_pitch_dispatches_first_piece_while_second_is_still_correcting(self):
        from validation_lab import pitch_stream

        allow_second = threading.Event()
        second_started = threading.Event()
        first_3d_sent = threading.Event()
        prepared = []

        class GatedPitch:
            def __init__(self, settings):
                self.settings = settings

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def prepare(self, pixels):
                prepared.append(pixels.copy())
                if len(prepared) == 2:
                    second_started.set()
                    if not allow_second.wait(5):
                        raise TimeoutError("second piece was not released")
                return pixels.copy(), {"applied": False}

        class TrackedClient(FakeClient):
            def request(self, q, timeout, *, on_sent=None):
                reply = super().request(q, timeout, on_sent=on_sent)
                if q["cmd"] == "detect_3d_piece" and q["piece_id"] == 1:
                    first_3d_sent.set()
                return reply

        server = lab.Server(0, self.root/"pitch-api")
        runner, _ = self.make_runner()
        runner.client_factory = TrackedClient
        server.runner = runner
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(allow_second.set)
        base = f"http://127.0.0.1:{server.server_port}"

        def submit(mode, piece_id, **options):
            payload = dict(files=self.rows, board_id="2", mode=mode, piece_id=piece_id, **options)
            request = Request(base+"/api/run", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                message = json.load(response)
            self.assertNotEqual(message.get("submitted"), False, message)
            return message["id"]

        def start_pitch():
            request = Request(base+"/api/pitch/start",
                              data=json.dumps(dict(files=self.rows, board_id="2")).encode())
            with urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 202)

        def completed(key):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with urlopen(base+"/api/jobs/"+key, timeout=3) as response:
                    job = json.load(response)
                if job["status"] != "running":
                    self.assertEqual(job["status"], "done", job)
                    return job
                time.sleep(.01)
            self.fail("request did not finish")

        def mapping(stack, pixels, piece_id, metadata):
            return dict(piece_id=piece_id, shm_name=f"depth-{piece_id}", pitch_correction=metadata)

        with patch.object(lab, "defaults", return_value=dict(piece_count=2)), \
             patch.object(lab, "mapped_depth_piece", side_effect=mapping), \
             patch.object(pitch_stream, "DepthPreprocessor", side_effect=GatedPitch) as correction:
            first, second = (submit(lab.PITCH_3D, pid) for pid in (1, 2))
            try:
                self.assertFalse(second_started.wait(.1), "arming 3D started Pitch")
                correction.assert_not_called()
                start_pitch()
                self.assertTrue(second_started.wait(3))
                self.assertFalse(first_3d_sent.is_set())
                first_2d, second_2d = (submit("manual_2d", pid) for pid in (1, 2))
                self.assertTrue(first_3d_sent.wait(3), "3D waited for the whole board")
                job = completed(first)
                self.assertLessEqual(job["pitch_ready_at"], job["submitted_at"])
                self.assertLessEqual(job["submitted_at"], job["received_at"])
                with urlopen(base+"/api/jobs/"+second, timeout=3) as response:
                    self.assertNotIn("submitted_at", json.load(response))
            finally:
                allow_second.set()
            for key in (second, first_2d, second_2d):
                completed(key)
            self.assertEqual(len(prepared), 2)
            correction.assert_called_once()
            waiting = submit(lab.PITCH_3D, 1, wait_for_pitch_start=True)
            completed(submit("manual_2d", 1))
            with urlopen(base+"/api/jobs/"+waiting, timeout=3) as response:
                self.assertNotIn("pitch_ready_at", json.load(response))
            finish = Request(base+"/api/pitch/finish",
                             data=json.dumps(dict(run_ids=[waiting], stopped=False)).encode())
            with urlopen(finish, timeout=3) as response:
                self.assertEqual(response.status, 200)
            with urlopen(base+"/api/jobs/"+waiting, timeout=3) as response:
                unsent = json.load(response)
            self.assertEqual(unsent["status"], "failed")
            self.assertNotIn("submitted_at", unsent)
            self.assertIn("尚未启动 Pitch", unsent["error"])

    def test_http_workflow_publishes_one_history_entry_and_combined_report(self):
        server=lab.Server(0,self.root/'workflow-api')
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        ids=['1'*32,'2'*32]
        for key, mode, result, (sent, finished) in zip(ids, ('manual_2d', 'manual_piece_3d'), (
            {'results_2d':{'is_ok':True}}, {'results_piece':{'piece_id':1,'is_ok':True}}), ((10.0, 11.5), (10.6, 11.0))):
            folder=server.folder/key;folder.mkdir()
            envelope=dict(board_id='2',response=dict(ok=True,board_id='2',piece_id=1,**result))
            (folder/'result.json').write_text(json.dumps(envelope),encoding='utf-8')
            server.jobs[key] = dict(status='done', board_id='2', piece_id=1, mode=mode,
                                    request_sent_at=sent, request_finished_at=finished)
        body=json.dumps({'run_ids':ids,'outcome':'done'}).encode()
        request=Request(base+'/api/workflows',data=body, headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            self.assertEqual(response.status,201)
            group=json.load(response)
        self.assertEqual(group['runs'],2)
        self.assertEqual(group['total_wall_ms'], 1500.0)
        with urlopen(base+'/api/history') as response:
            self.assertEqual(json.load(response),[group])
        with urlopen(base+'/runs/'+group['id']+'/report.html') as response:
            html=response.read().decode()
        self.assertIn('workflow_stages',html)
        self.assertIn('流程 1',html)
        with urlopen(base+'/runs/'+group['id']+'/result.json') as response:
            document = json.load(response)
            self.assertEqual([r['id'] for r in document['runs']], ids)
            self.assertEqual(document['total_wall_ms'], 1500.0)
        with urlopen(base+'/runs/'+group['id']+'/timings.csv') as response:
            self.assertIn(',workflow,,total_wall_ms,1500.0', response.read().decode('utf-8-sig'))

if __name__=='__main__':unittest.main()
