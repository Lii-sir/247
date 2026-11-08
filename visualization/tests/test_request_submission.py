"""Paced input submission using tiny synthetic images and model-free protocol stubs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_validation_lab as fixtures
FakeClient, init_request, lab = fixtures.FakeClient, fixtures.init_request, fixtures.lab
from detect_and_view import ProtocolClient


class PacedSubmissionTests(unittest.TestCase):
    setUp = fixtures.ValidationTests.setUp
    make_runner = fixtures.ValidationTests.make_runner
    def test_paced_sorted_2d_then_depth_and_actual_gaps(self):
        runner, clients = self.make_runner()
        self.addCleanup(runner.close)
        folder = self.root / "paced"
        folder.mkdir()
        result = runner.run(list(reversed(self.rows)), '2', 'joint', 2, folder, lambda _: None,
                            submission_mode='paced', request_interval_ms=30)
        self.assertTrue(result['ok'], result)
        requests = [q for q in clients[0].requests if q['cmd'].startswith('detect_')]
        self.assertEqual([('detect_2d_piece', 1), ('detect_2d_piece', 2), ('detect_3d_board', None)],
                         [(q['cmd'], q.get('piece_id')) for q in requests])
        document = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        submission = document['submission']
        self.assertFalse(submission['aborted'])
        self.assertEqual(30, submission['request_interval_ms'])
        self.assertEqual([1, 2, 3], [q['order'] for q in submission['requests']])
        for request in submission['requests'][1:]:
            self.assertGreaterEqual(request['gap_ms'], 29.9)

    def test_zero_interval_keeps_order_and_partial_2d_has_no_depth(self):
        runner, clients = self.make_runner()
        self.addCleanup(runner.close)
        folder = self.root/'zero'
        folder.mkdir()
        result = runner.run(list(reversed(self.rows[:2])), '2', '2d', 2, folder, lambda _: None,
                            submission_mode='paced', request_interval_ms=0)
        self.assertTrue(result['ok'])
        requests = [q for q in clients[0].requests if q['cmd'].startswith('detect_')]
        self.assertEqual([1, 2], [q['piece_id'] for q in requests])
        self.assertTrue(all(q['cmd'] == 'detect_2d_piece' for q in requests))

    def test_error_stops_subsequent_submission_and_preserves_original_failure(self):
        runner, clients = self.make_runner(fail=1)
        self.addCleanup(runner.close)
        folder = self.root/'failed-paced'
        folder.mkdir()
        result = runner.run(self.rows, '2', 'joint', 2, folder, lambda _: None,
                            submission_mode='paced', request_interval_ms=100)
        self.assertFalse(result['ok'])
        self.assertIn('synthetic piece error', result['error'])
        requests = [q for q in clients[0].requests if q['cmd'].startswith('detect_')]
        self.assertEqual([('detect_2d_piece', 1)], [(q['cmd'], q.get('piece_id')) for q in requests])
        document = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        self.assertTrue(document['submission']['aborted'])
        self.assertTrue(any(q['cmd'] == 'release_board' for q in clients[0].requests))

    def test_invalid_intervals_rejected_before_client_creation(self):
        for value in (-1, True, None, '100', float('nan'), float('inf'), 3600001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lab.submission_options('paced', value)
        for mode in (None, [], 'serial'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                lab.submission_options(mode, 0)

    def test_cleanup_transport_error_does_not_replace_known_piece_failure(self):
        class Client(FakeClient):
            def request(self, q, timeout, **kwargs):
                result = super().request(q, timeout, **kwargs)
                if q['cmd'] == 'detect_3d_board' and not result['ok']:
                    raise RuntimeError('cleanup transport closed')
                return result
        runner, _ = self.make_runner()
        client = Client()
        client.fail_piece = 1
        runner.client_factory = lambda *args, **kwargs: client
        self.addCleanup(runner.close)
        folder = self.root/'transport-failure'
        folder.mkdir()
        result = runner.run(self.rows, '2', 'joint', 2, folder, lambda _: None)
        self.assertFalse(result['ok'])
        self.assertIn('synthetic piece error', result['error'])
        self.assertNotIn('cleanup transport closed', result['error'])

    def test_stop_interrupts_long_interval_and_extends_only_child_cache_timeout(self):
        environment = patch.dict(os.environ, {'BRIDGE_BOARD_TIMEOUT_SECONDS':'90'})
        environment.start()
        self.addCleanup(environment.stop)
        sent, clients = threading.Event(), []
        class Client(FakeClient):
            def request(self, q, timeout, **kwargs):
                result = super().request(q, timeout, **kwargs)
                if q['cmd'] == 'detect_2d_piece':
                    sent.set()
                return result
        runner, _ = self.make_runner()
        def factory(*args, **kwargs):
            client = Client(**kwargs)
            clients.append(client)
            return client
        runner.client_factory = factory
        self.addCleanup(runner.close)
        folder = self.root/'stop-paced'
        folder.mkdir()
        original = os.environ.get('BRIDGE_BOARD_TIMEOUT_SECONDS')
        pool = ThreadPoolExecutor(1)
        future = pool.submit(runner.run, self.rows, '2', 'joint', 2, folder, lambda _: None,
                             submission_mode='paced', request_interval_ms=3600000)
        try:
            self.assertTrue(sent.wait(3))
            runner.stop()
            result = future.result(3)
            self.assertFalse(result['ok'])
            requests = [q for q in clients[0].requests if q['cmd'].startswith('detect_')]
            self.assertEqual(1, len(requests))
            self.assertGreaterEqual(float(clients[0].env_overrides['BRIDGE_BOARD_TIMEOUT_SECONDS']), 7230)
            self.assertEqual(original, os.environ.get('BRIDGE_BOARD_TIMEOUT_SECONDS'))
        finally:
            runner.stop()
            pool.shutdown(wait=True)

    def test_cadence_changes_reuse_models_until_cache_timeout_override_changes(self):
        runner, clients = self.make_runner()
        self.addCleanup(runner.close)
        self.assertFalse(runner._ensure(init_request(), lambda _: None)['model_reused'])
        self.assertTrue(runner._ensure(init_request(), lambda _: None)['model_reused'])
        changed = runner._ensure(init_request(), lambda _: None, board_timeout_seconds=180)
        self.assertFalse(changed['model_reused'])
        self.assertEqual('180', clients[-1].env_overrides['BRIDGE_BOARD_TIMEOUT_SECONDS'])
        self.assertTrue(runner._ensure(init_request(), lambda _: None, board_timeout_seconds=180)['model_reused'])

    def test_http_validation_and_options_are_forwarded_without_starting_models(self):
        server = lab.Server(0, self.root/'http')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        server.runner.run = Mock(return_value=dict(ok=True, product_is_ok=True, board_id='2', pieces=2))
        url = f'http://127.0.0.1:{server.server_port}/api/run'
        for options in ({'submission_mode': 'bad'}, {'submission_mode':'paced', 'request_interval_ms': -1},
                        {'submission_mode':'paced', 'request_interval_ms': float('nan')}):
            request = dict(files=self.rows, board_id='2', mode='joint', **options)
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url, data=json.dumps(request).encode()))
            self.assertEqual(400, error.exception.code)
        server.runner.run.assert_not_called()
        self.assertEqual({}, server.jobs)
        with patch.object(lab, 'defaults', return_value=dict(piece_count=2)):
            request = dict(files=self.rows, board_id='2', mode='joint', submission_mode='paced', request_interval_ms=12.5)
            with urlopen(Request(url, data=json.dumps(request).encode())) as response:
                key = json.load(response)['id']
        deadline = time.monotonic()+3
        while server.jobs[key]['status'] == 'running' and time.monotonic() < deadline:
            threading.Event().wait(.01)
        self.assertEqual('done', server.jobs[key]['status'])
        self.assertEqual(dict(submission_mode='paced', request_interval_ms=12.5), server.runner.run.call_args.kwargs)


class RealProtocolSubmissionTests(unittest.TestCase):
    def test_transport_sends_depth_before_waiting_for_2d_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root/'worker.py'
            worker.write_text('''import json,sys,os
pending=[]
def emit(q, **fields):
    print(json.dumps(dict(ok=True,request_id=q['request_id'],cmd=q['cmd'],**fields)),flush=True)
for line in sys.stdin:
    q=json.loads(line);cmd=q['cmd']
    if cmd=='detect_2d_piece':
        pending.append(q)
        continue
    if cmd=='detect_3d_board':
        emit(q,results_board=dict(is_ok=len(pending)==2,pieces=[]))
        for image in pending:emit(image,piece_id=image['piece_id'],results_2d=dict(is_ok=True))
        pending=[]
    else:emit(q)
    if cmd=='release':break
''', encoding='utf-8')
            from PIL import Image
            import numpy as np
            rows = []
            for kind, name, piece in [('2d','2_1.png',1),('2d','2_2.png',2),('3d','2.png',None)]:
                path = root/name
                (Image.fromarray(np.full((4,4),33000,np.uint16)) if kind=='3d' else Image.new('RGB',(4,4))).save(path)
                rows.append(dict(kind=kind, name=name, path=str(path), board_id='2', piece_id=piece))
            def mapper(stack, path, kind, piece_id=None):
                return dict(shm_name=path.name, **({'piece_id':piece_id,'image_size':3} if kind=='2d' else {'data_size':3}))
            def factory(python, ignored_server, log, **kwargs):
                return ProtocolClient(python, worker, log, **kwargs)
            runner = lab.Runner(root, factory, init_request, mapper,
                                lambda doc, path, **kwargs: path.write_text('<html>synthetic report</html>'))
            output = root/'result'
            output.mkdir()
            pool = ThreadPoolExecutor(1)
            future = pool.submit(runner.run, rows, '2', 'joint', 2, output, lambda _: None,
                                 submission_mode='paced', request_interval_ms=20)
            try:
                result = future.result(8)
                self.assertTrue(result['ok'], result)
                document = json.loads((output/'result.json').read_text(encoding='utf-8'))
                sends = document['submission']['requests']
                self.assertEqual([1, 2, None], [q['piece_id'] for q in sends])
                self.assertEqual(3, len({q['request_id'] for q in sends}))
                for request in sends[1:]:
                    self.assertGreaterEqual(request['gap_ms'], 19.9)
                timeline = document['timeline']
                depth = next(q for q in timeline if q['cmd']=='detect_3d_board')
                two_d = [q for q in timeline if q['cmd']=='detect_2d_piece']
                self.assertTrue(all(q['received_ms'] >= depth['sent_ms'] for q in two_d))
            finally:
                runner.close(force=True)
                pool.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
