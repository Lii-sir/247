"""Event-gated capacity and board critical-path checks, not hardware benchmarks."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
from algorithm.board_pipeline import BoardPipeline
from algorithm.config import load_runtime_config
from pipeline_fakes import FakeDetector
from test_early_joint_pipeline import GatedDetector


class StageGates(FakeDetector):
    def __init__(self):
        super().__init__(count=10,delay=0)
        self.register_gate=threading.Event()
        self.measure_gate=threading.Event()
        self.started=threading.Condition()
        self.registering=set();self.measuring=set()

    def register_prepared_piece(self,pid,*args):
        with self.started:
            self.registering.add(pid);self.started.notify_all()
        if not self.register_gate.wait(5):raise TimeoutError('registration gate')
        return dict(is_valid=True)

    def measure_registered_piece(self,pid,entry,depth,piece,registration,correction=None):
        with self.started:
            self.measuring.add(pid);self.started.notify_all()
        if not self.measure_gate.wait(5):raise TimeoutError('measurement gate')
        return super().measure_registered_piece(pid,entry,depth,piece,registration,correction)


class BoardQueueTests(unittest.TestCase):
    def exercise(self,registration_workers,measurement_workers,early):
        detector=StageGates()
        with ExitStack() as stack:
            cpu=stack.enter_context(ThreadPoolExecutor(4))
            gpu=stack.enter_context(ThreadPoolExecutor(1))
            registration=stack.enter_context(ThreadPoolExecutor(registration_workers))
            measurement=stack.enter_context(ThreadPoolExecutor(measurement_workers))
            pipe=BoardPipeline(detector,cpu,gpu,registration_pool=registration,
                               joint_pool=measurement,early_joint_workers=early)
            try:
                for pid in range(1,11):
                    self.assertTrue(pipe.submit_2d(dict(board_id='A',piece_id=pid)).result(3)['ok'])
                board=pipe.submit_3d(dict(board_id='A'))
                with detector.started:
                    self.assertTrue(detector.started.wait_for(lambda:len(detector.registering)==registration_workers,3))
                    # Every started task is blocked; further pieces must remain
                    # queued when only four execution slots are available.
                    self.assertEqual(registration_workers,len(detector.registering))
                detector.register_gate.set()
                with detector.started:
                    self.assertTrue(detector.started.wait_for(lambda:len(detector.measuring)==measurement_workers,3))
                    self.assertEqual(measurement_workers,len(detector.measuring))
                self.assertFalse(board.done())
                detector.measure_gate.set()
                response=board.result(5)
                self.assertTrue(response['ok'],response)
                return response
            finally:
                detector.register_gate.set();detector.measure_gate.set();pipe.close()

    def test_previous_four_slot_configuration_really_queues_ten_pieces(self):
        response=self.exercise(4,4,8)
        diagnostic=response['results_board']['pipeline_diagnostics']
        self.assertEqual(4,diagnostic['registration_capacity'])
        self.assertEqual(4,diagnostic['measurement_capacity'])
        self.assertEqual(10,len(response['results_board']['pieces']))

    def test_project_configuration_admits_ten_pieces_to_each_stage_at_once(self):
        path=ROOT/'config/config.json'
        config=load_runtime_config(path.read_text(encoding='utf-8'),str(path))
        options=config['common_3d']['parameters']['threading']
        response=self.exercise(options['registration_workers'],options['joint_workers'],options['early_joint_workers'])
        diagnostic=response['results_board']['pipeline_diagnostics']
        self.assertEqual(10,diagnostic['registration_capacity'])
        self.assertEqual(10,diagnostic['measurement_capacity'])
        self.assertEqual(0.,response['timings']['board_wait_2d_after_3d_ms'])
        for piece in response['results_board']['pieces']:
            timeline=piece['timeline']
            self.assertLessEqual(timeline['inputs_ready_ms'],timeline['registration_admitted_ms'])
            self.assertLessEqual(timeline['registration_admitted_ms'],timeline['measurement_admitted_ms'])
            self.assertLessEqual(timeline['measurement_admitted_ms'],timeline['joint_completed_ms'])
            self.assertLess(timeline['diagnostics_2d_ready_ms'],0)

    def test_summary_identifies_waiting_for_late_two_d_diagnostics(self):
        measured=threading.Event()
        class ObservedPipeline(BoardPipeline):
            def _measured(self,*args,**kwargs):
                super()._measured(*args,**kwargs)
                measured.set()
        detector=GatedDetector()
        with ThreadPoolExecutor(2) as cpu,ThreadPoolExecutor(1) as gpu,ThreadPoolExecutor(1) as post:
            pipe=ObservedPipeline(detector,cpu,gpu,postprocess_pool=post)
            try:
                board=pipe.submit_3d(dict(board_id='A'))
                piece=pipe.submit_2d(dict(board_id='A',piece_id=1,diagnostics_ng=True))
                self.assertTrue(measured.wait(3))
                self.assertFalse(board.done())
                detector.diagnostics_gate.set()
                self.assertTrue(piece.result(3)['ok'])
                result=board.result(3)
                diagnostic=result['results_board']['pipeline_diagnostics']
                self.assertEqual('2d_diagnostics',diagnostic['completion_gate'])
                self.assertGreater(diagnostic['wait_2d_after_3d_ms'],0)
                self.assertFalse(result['results_board']['is_ok'])
            finally:
                detector.diagnostics_gate.set();pipe.close()

    def test_summary_identifies_last_three_d_piece_instead_of_first_piece(self):
        nine_done=threading.Event();last_gate=threading.Event()
        class Detector(FakeDetector):
            def measure_registered_piece(self,pid,*args):
                if pid==10 and not last_gate.wait(5):raise TimeoutError('last piece')
                return super().measure_registered_piece(pid,*args)
        class ObservedPipeline(BoardPipeline):
            def _measured(self,state,*args,**kwargs):
                super()._measured(state,*args,**kwargs)
                with self._lock:
                    if len(state.results)==9:nine_done.set()
        with ThreadPoolExecutor(4) as cpu,ThreadPoolExecutor(1) as gpu,ThreadPoolExecutor(10) as joint:
            pipe=ObservedPipeline(Detector(delay=0),cpu,gpu,joint_pool=joint,early_joint_workers=10)
            try:
                for pid in range(1,11):self.assertTrue(pipe.submit_2d(dict(board_id='A',piece_id=pid)).result(3)['ok'])
                board=pipe.submit_3d(dict(board_id='A'))
                self.assertTrue(nine_done.wait(3));self.assertFalse(board.done())
                last_gate.set()
                result=board.result(3)
                diagnostic=result['results_board']['pipeline_diagnostics']
                self.assertEqual(10,diagnostic['last_3d_piece_id'])
                self.assertEqual('3d_results',diagnostic['completion_gate'])
                self.assertAlmostEqual(diagnostic['all_3d_results_ready_ms'],
                    result['results_board']['pieces'][-1]['timeline']['joint_completed_ms'])
                json.dumps(result,allow_nan=False)
            finally:
                last_gate.set();pipe.close()


if __name__=='__main__':unittest.main()
