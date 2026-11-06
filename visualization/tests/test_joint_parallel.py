"""CPU synthetic regressions for production instance concurrency; no models."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from algorithm.detect_board import DetectBoard
from algorithm.instance_tasks import InstanceTasks
from algorithm.timing import record_duration, timing_context, timed_call
from algorithm.config import load_runtime_config


class InstanceParallelTests(unittest.TestCase):
    def setUp(self):
        self.detector = DetectBoard(lambda message: None)
        self.addCleanup(self.detector._instance_tasks.close)

    def test_tasks_overlap_keep_order_and_reuse_pool(self):
        pool = self.detector._instance_tasks
        pool.configure(2)
        executor = pool._pool
        barrier = threading.Barrier(2)

        def task(value):
            barrier.wait(timeout=3)
            return value, threading.current_thread().name

        for _ in range(2):
            result = pool.map(task, [2, 1])
            self.assertEqual([v for v, _ in result], [2, 1])
            self.assertTrue(all(name.startswith('instance') for _, name in result))
            self.assertIs(pool._pool, executor)

    def test_parallel_mask_mapping_equals_serial_without_input_mutation(self):
        instances = []
        for index in range(4):
            mask = np.zeros((64, 90), np.uint8)
            mask[5+index:20+index, 10+index:35+index] = 255
            mask.flags.writeable = False
            instances.append(dict(instance_id=index+1, class_name='wire', mask=mask))
        registration = dict(scale=.5, tx=3., ty=2.,
            homography_2d_to_3d=[[.5,.01,3.],[-.01,.51,2.],[0.,0.,1.]])
        before = copy.deepcopy(registration)
        expected = self.detector._align_instances(instances, (60,70), registration, (10,20,100,84))
        self.detector._instance_tasks.configure(3)
        actual = self.detector._align_instances(instances, (60,70), registration, (10,20,100,84))
        self.assertEqual(registration, before)
        for a, b in zip(actual, expected):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if isinstance(a[key], np.ndarray):
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    self.assertEqual(a[key], b[key])
        self.assertEqual(self.detector._align_instances([], (60,70), registration), [])

    def test_child_timing_isolated_between_concurrent_pieces(self):
        self.detector._instance_tasks.configure(4)
        barrier = threading.Barrier(4)

        def piece(pid):
            totals = {}
            def child(value):
                barrier.wait(timeout=3)
                record_duration('synthetic_ms', value)
                return pid, value
            with timing_context(totals, phase='joint', piece_id=pid):
                values = self.detector._instance_tasks.map(child, [pid,pid], timing_stage='work_ms')
            return values, totals

        with ThreadPoolExecutor(max_workers=2) as parents:
            results = list(parents.map(piece, [1,2]))
        for pid, (values, totals) in enumerate(results, 1):
            self.assertEqual(values, [(pid,pid),(pid,pid)])
            self.assertEqual(totals['synthetic_ms'], 2*pid)
            self.assertGreaterEqual(totals['work_ms'], 0)

    def test_error_waits_for_running_siblings_and_retains_timings(self):
        pool = self.detector._instance_tasks
        pool.configure(2)
        barrier = threading.Barrier(2)
        finished = threading.Event()

        def child(value):
            barrier.wait(timeout=3)
            record_duration('child_ms', 1)
            if value == 0:
                raise ValueError('synthetic failure')
            finished.set()
            return value

        totals = {}
        with timing_context(totals, phase='joint'):
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                pool.map(child, [0,1])
        self.assertTrue(finished.is_set())
        self.assertEqual(totals['child_ms'], 2)
        self.assertEqual(pool.map(lambda value: value+1, [1,2]), [2,3])

    def test_configuration_serial_mode_and_release(self):
        config = load_runtime_config((ROOT/'config/config.json').read_text(encoding='utf-8-sig'),
                                     str(ROOT/'config/config.json'))
        config['smartray_pitch_3d']['parameters']['enabled_3d'] = False
        config['common_3d']['parameters']['threading']['instance_workers'] = 3
        self.detector._load_config(json.dumps(config))
        self.assertEqual(self.detector._instance_tasks.workers, 3)
        executor = self.detector._instance_tasks._pool
        self.detector._load_config(json.dumps(config))
        self.assertIs(self.detector._instance_tasks._pool, executor)
        for invalid in (0,-1,True,2.5,'2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.detector._instance_tasks.configure(invalid)
        self.detector._instance_tasks.configure(1)
        self.assertTrue(executor._shutdown)
        name = threading.current_thread().name
        self.assertEqual(self.detector._instance_tasks.map(lambda _: threading.current_thread().name,[1,2]),[name,name])
        self.detector._instance_tasks.configure(2)
        with patch('algorithm.detect_board.torch.cuda.is_available', return_value=False):
            self.detector.release()
        self.assertIsNone(self.detector._instance_tasks._pool)
        self.detector._load_config(json.dumps(config))
        self.assertEqual(self.detector._instance_tasks.workers,3)


if __name__ == '__main__':
    unittest.main()
