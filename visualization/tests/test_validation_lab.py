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
            self.requests.append(q);self.timeline.append({'cmd':q['cmd'],'piece_id':q.get('piece_id'),'duration_ms':1})
            if on_sent is not None:on_sent('fake-'+str(len(self.requests)),time.perf_counter())
            board=q.get('board_id');cmd=q['cmd']
            response=dict(ok=True,cmd=cmd,board_id=board,timings={})
            if cmd=='detect_2d_piece':
                pid=q['piece_id'];response.update(piece_id=pid,results_2d={'is_ok':True})
                if pid==self.fail_piece:return dict(response,ok=False,error='synthetic piece error',message='chip aspect ratio 1.9072 does not match configured range')
                self.pieces.setdefault(board,set()).add(pid);self.condition.notify_all()
            if cmd=='detect_3d_board':
                self.condition.wait_for(lambda:len(self.pieces.get(board,set()))==2 or board in self.released,2)
                if board in self.released:return dict(response,ok=False,error='board released')
                response['results_board']={'is_ok':False,'pieces':[]}
            if cmd=='release_board':
                self.released.add(board);self.pieces.pop(board,None);self.condition.notify_all()
            return response
    def close(self, *, force=False):self.closed=True


def init_request(*args):
    return dict(cmd='init_algorithm',config_text=json.dumps({'common_3d':{'parameters':{'input':{'piece_count':2}}},'height_measurement_3d':{'parameters':{}}}))


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

    def test_validation_requires_exact_board_inputs(self):
        board,images,depth=lab.board_inputs(self.rows,'2','joint',2)
        self.assertEqual(board,'2');self.assertEqual([x['piece_id'] for x in images],[1,2])
        self.assertEqual(depth['name'],'2.png')
        for bad in (self.rows[:1],self.rows[:2],self.rows+[self.rows[0]],self.rows+[self.rows[2]]):
            with self.assertRaises(ValueError):lab.board_inputs(bad,'2','joint',2)
        one=lab.board_inputs(self.rows[:1],'2','2d',2)
        self.assertEqual(len(one[1]),1);self.assertIsNone(one[2])

    def test_manual_numbering_and_board_isolation(self):
        rows=[*self.rows,dict(self.rows[0],board_id='3',piece_id=2)]
        self.assertEqual(len(lab.board_inputs(rows,'2','joint',2)[1]),2)
        rows[0]=dict(rows[0],piece_id=3)
        with self.assertRaisesRegex(ValueError,'片号'):lab.board_inputs(rows,'2','joint',2)

    def make_runner(self, fail=None):
        clients=[]
        def factory(*args, **kwargs):
            c=FakeClient(**kwargs);c.fail_piece=fail;clients.append(c);return c
        def mapper(stack,path,kind,piece_id=None):
            return dict(shm_name=path.name,**({'piece_id':piece_id,'image_size':3} if kind=='2d' else {'data_size':3}))
        def reporter(doc,path,**kwargs):path.write_text('<html>report</html>',encoding='utf-8')
        return lab.Runner(self.root,factory,init_request,mapper,reporter),clients

    def test_runner_sends_explicit_ids_and_reuses_initialized_models(self):
        runner,clients=self.make_runner();self.addCleanup(runner.close)
        for n in range(2):
            folder=self.root/('run'+str(n));folder.mkdir()
            result=runner.run(self.rows,'2','joint',2,folder,lambda x:None)
            # New logical board after each cleanup for the fake's release tombstone.
            clients[0].released.clear()
            self.assertTrue(result['ok']);self.assertFalse(result['product_is_ok'])
            self.assertEqual(result['model_reused'],n==1)
            self.assertTrue((folder/'report.html').exists());self.assertTrue((folder/'timings.csv').exists())
        self.assertEqual(len(clients),1)
        self.assertEqual(sum(q['cmd']=='init_algorithm' for q in clients[0].requests),1)
        requests=[q for q in clients[0].requests if q['cmd']=='detect_2d_piece']
        self.assertTrue(all(q['board_id']=='2' for q in requests))
        self.assertEqual([q['piece_id'] for q in requests],[1,2,1,2])

    def test_partial_2d_and_failure_release_board(self):
        for mode,fail,rows in [('2d',None,self.rows[:1]),('joint',1,self.rows)]:
            runner,clients=self.make_runner(fail);self.addCleanup(runner.close)
            folder=self.root/('case'+mode);folder.mkdir()
            result=runner.run(rows,'2',mode,2,folder,lambda x:None)
            self.assertEqual(result['ok'],fail is None)
            self.assertTrue(any(q['cmd']=='release_board' for q in clients[0].requests))
            if mode=='2d':self.assertFalse(any(q['cmd']=='detect_3d_board' for q in clients[0].requests))
            else:
                self.assertIn('synthetic piece error',result['error'])
                self.assertIn('1.9072',result['error'])
                saved=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                self.assertNotEqual(saved['response']['message'],'board cache released')

    def test_bad_depth_is_rejected_before_models_are_started(self):
        Image.new('RGB',(4,4)).save(self.rows[2]['path'])
        runner,clients=self.make_runner();self.addCleanup(runner.close)
        folder=self.root/'bad-depth';folder.mkdir()
        result=runner.run(self.rows,'2','joint',2,folder,lambda x:None)
        self.assertFalse(result['ok']);self.assertIn('16位',result['error']);self.assertEqual(clients,[])
        self.assertTrue((folder/'result.json').exists())

    def test_two_d_mode_does_not_mutate_or_require_depth_calibration(self):
        runner,clients=self.make_runner();self.addCleanup(runner.close)
        cfg=json.loads(init_request()['config_text'])
        cfg['smartray_pitch_3d']={'description':'test','parameters':{'enabled_3d':True,'scan_pitch_mm':0}}
        cfg['common_3d']['parameters']['threading']={'registration_affinity':{'enabled':True,'physical_cores':4}}
        runner.init_builder=lambda *args:dict(cmd='init_algorithm',config_text=json.dumps(cfg))
        folder=self.root/'two-only';folder.mkdir()
        result=runner.run(self.rows[:1],'2','2d',2,folder,lambda x:None)
        self.assertTrue(result['ok']);self.assertTrue(cfg['smartray_pitch_3d']['parameters']['enabled_3d'])
        request=next(q for q in clients[0].requests if q['cmd']=='init_algorithm')
        self.assertFalse(json.loads(request['config_text'])['smartray_pitch_3d']['parameters']['enabled_3d'])
        effective=json.loads(request['config_text'])
        self.assertFalse(effective['common_3d']['parameters']['threading']['registration_affinity']['enabled'])
        self.assertTrue(effective['common_2d']['parameters']['threading']['cpu_process_affinity']['enabled'])
        self.assertTrue(cfg['common_3d']['parameters']['threading']['registration_affinity']['enabled'])

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


if __name__=='__main__':unittest.main()
