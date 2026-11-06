"""CPU synthetic regressions for production instance concurrency; no models."""
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from algorithm.detect_board import DetectBoard
from algorithm.instance_tasks import InstanceTasks
from algorithm.timing import record_duration, timing_context, timed_call
from algorithm.config import load_runtime_config
from algorithm import detect_board as board
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures.wire_reference import LegacyWireReference


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


class WireParallelTests(unittest.TestCase):
    def setUp(self):
        self.detector = DetectBoard(lambda _: None)
        self.reference = LegacyWireReference(lambda _: None)
        self.addCleanup(self.detector._instance_tasks.close)
        self.detector._instance_tasks.configure(4)
        for detector in (self.detector, self.reference):
            detector.min_component_area = 4
            detector.height_config = replace(detector.height_config,
                plane_min_valid_points=4, chip_min_valid_points=4)

    @staticmethod
    def fixture():
        depth = np.full((140, 180), 33000, np.uint16)
        masks = []
        for index in range(4):
            mask = np.zeros((120,160), np.uint8)
            cv2.line(mask, (0, 15+index*23), (150, 25+index*23), 255, 7)
            masks.append(dict(instance_id=index+1, class_name='thin' if index%2 else 'wire',
                              _mask_2d_array=mask, _mask_3d_array=np.zeros_like(depth,np.uint8)))
        return depth, masks

    def test_local_wire_path_equals_frozen_full_frame_reference(self):
        rng = np.random.default_rng(194)
        for case in range(28):
            depth, items = self.fixture()
            # Covers missing ROI, cleaning rejection, border objects and ROI
            # clipping after cleaning. Fractional affine/crop offsets matter.
            if case%4 == 0:
                items[0]['_mask_2d_array'][60:75, 20:95] = 255
            if case%5 == 0:
                items[2]['_mask_2d_array'][:] = 0
            depth[rng.random(depth.shape)<.03] = 0
            roi = [(10, 10, 140, 100)] if case%3 else [(140,0,159,120),(0,20,50,80)]
            a = float(rng.uniform(.7,1.1))
            matrix = [[a,float(rng.uniform(-.015,.015)),float(rng.uniform(-5,5))],
                      [float(rng.uniform(-.015,.015)),a,float(rng.uniform(-3,3))],[0,0,1]]
            registration = dict(scale=a,tx=matrix[0][2],ty=matrix[1][2],homography_2d_to_3d=matrix)
            radius = case%3
            for detector in (self.detector,self.reference):
                detector.height_config = replace(detector.height_config,
                    wire_measurement_erode_px=radius,thin_measurement_erode_px=radius)
            args=(registration,dict(coefficients=[.02,-.01,32768.]),roi,(7,9,167,129))
            expected=self.reference._measure_wire_regions(depth,copy.deepcopy(items),*args)
            for workers in (1,4):
                self.detector._instance_tasks.configure(workers)
                before=copy.deepcopy(items)
                actual=self.detector._measure_wire_regions(depth,items,*args)
                with self.subTest(case=case,workers=workers):self.assertEqual(actual,expected)
                for item,original in zip(items,before):
                    np.testing.assert_array_equal(item['_mask_2d_array'],original['_mask_2d_array'])

    def test_wire_tasks_overlap_preserve_skipped_instance_order(self):
        depth,items=self.fixture()
        barrier=threading.Barrier(4)
        def task(depth,wire,*args):
            barrier.wait(timeout=3)
            return None if wire['instance_id']==2 else {'instance_id':wire['instance_id']}
        with patch.object(self.detector,'_measure_wire_region',side_effect=task):
            result=self.detector._measure_wire_regions(depth,items,{},None)
        self.assertEqual(result,[{'instance_id':1,'region_id':1},{'instance_id':3,'region_id':2},{'instance_id':4,'region_id':3}])

    def test_thin_split_once_per_piece_and_reused_for_endpoints(self):
        depth,items=self.fixture()
        original=copy.deepcopy(items)
        registration=dict(scale=1.,tx=0.,ty=0.,is_valid=True)
        with patch.object(board,'split_thin_mask',wraps=board.split_thin_mask) as split:
            actual=self.detector._measure_piece(depth,items,registration,[(0,0,160,120)],(0,0,160,120))
        self.assertEqual(split.call_count,2)
        expected=self.reference._measure_piece(depth,original,registration,[(0,0,160,120)],(0,0,160,120))
        self.assertEqual(actual,expected)
        for item in items:
            if item['class_name']=='thin':
                self.assertIn('_thin_wire_mask_2d',item)
                self.assertEqual(len(item['_thin_solder_masks_2d']),2)

    def test_cleaning_and_roi_allocation_are_local(self):
        mask=np.zeros((1000,1500),np.uint8);mask[400:410,700:800]=255
        item=dict(instance_id=1,class_name='wire',_mask_2d_array=mask,bending_2d={'is_ok':True})
        with patch.object(board._geometry,'clean_mask',wraps=board._geometry.clean_mask) as clean, \
             patch.object(self.detector,'_mask_in_wire_rois',wraps=self.detector._mask_in_wire_rois) as clip:
            self.detector._measure_wire_regions(np.full((1000,1500),33000,np.uint16),[item],
                dict(scale=1,tx=0,ty=0),dict(coefficients=[0,0,32768]),[(0,0,1500,1000)],(0,0,1500,1000))
        self.assertLess(clean.call_args.args[0].size,mask.size//20)
        self.assertLess(clip.call_args.args[0].size,mask.size//20)


if __name__ == '__main__':
    unittest.main()
