"""Synthetic mapping/height process tests; no real affinity or camera experiments."""
import copy
from dataclasses import replace
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
sys.path.insert(0, str(ROOT/'tests'))
import cv2
import numpy as np
from algorithm.height_measurement import HeightConfig
from algorithm.joint_process import JointMeasurementProcesses
from algorithm.registration_process import SharedBoardInputs
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector


def fixture():
    shape = (180, 240)
    local = np.full(shape, 33000, np.uint16)
    items = []
    for pid, kind in ((1,'thin'),(2,'bond'),(3,'chip'),(4,'chip'),(5,'wire'),(6,'bond')):
        mask = np.zeros(shape, np.uint8)
        if pid == 1: cv2.line(mask, (30,120), (175,85), 255, 7)
        elif pid == 2: mask[80:90,150:166] = 255
        elif pid == 3:
            mask[30:140,100:220] = 255
            mask[70:100,140:180] = 0
        elif pid == 4: mask[65:110,135:185] = 255
        elif pid == 5: cv2.line(mask, (30,155), (210,160), 255, 7)
        else: mask[20:30,20:40] = 255
        if kind == 'chip': local[mask > 0] = 33500
        elif kind in {'bond','thin'}: local[mask > 0] = 33900
        items.append(dict(instance_id=pid, class_id=pid, class_name=kind, confidence=.9,
                          mask=mask, rectangle=None, rectangle_ratio=1., is_ok=True,
                          bending_2d=dict(is_valid=True,is_ok=True) if kind in {'wire','thin'} else None))
    depth = np.pad(local, ((11,0),(0,0)))
    entry = dict(instances=items, image_shape=[240,180], crop_box=(0,0,240,180), circle=(30,70,10),
                 roi_boxes=[(0,0,240,180)], wire_roi_checks=[dict(is_ok=True)],
                 silver_overflow_2d=dict(status='ok',is_ok=True),
                 silver_continuity_2d=dict(status='ok',is_ok=True))
    config = replace(HeightConfig(), plane_min_valid_points=4, chip_min_valid_points=4,
                     platform_roi_offsets=((-2.,-3.,-1.,0.),))
    registration = dict(scale=1.,tx=0.,ty=0.,is_valid=True,affine_valid=True,
                        homography_2d_to_3d=np.eye(3).tolist())
    return depth, entry, config, registration


class JointProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = JointMeasurementProcesses(1, dict(enabled=False), instance_workers=1)

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def setUp(self):
        self.inputs = SharedBoardInputs()
        self.pool = ThreadPoolExecutor(1)
        self.addCleanup(self.inputs.close)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.depth, self.entry, self.config, self.registration = fixture()

    def submit(self, **overrides):
        params = dict(preparation_pool=self.pool, inputs=self.inputs, piece_id=1, entry=self.entry,
                      depth=self.depth, piece=dict(piece_id=1,x1=10,x2=220,y1=11,y2=191),
                      registration=self.registration, correction=dict(applied=True), height_config=self.config,
                      min_component_area=4, crop_roi=(0,0,240,180), wire_rois=[], timings={},
                      metadata=dict(board_id='A',piece_id=1,request_id='joint'), is_current=lambda: True)
        params.update(overrides)
        return self.service.submit_piece(**params)

    def test_complete_result_matches_direct_and_preserves_input_masks(self):
        from algorithm.detect_board import DetectBoard
        detector = DetectBoard(lambda _: None)
        self.addCleanup(detector.release)
        detector.height_config, detector.min_component_area = self.config, 4
        expected = detector.measure_registered_piece(1, copy.deepcopy(self.entry), self.depth,
            dict(y1=11,y2=191), self.registration, dict(applied=True))
        before = [item['mask'].copy() for item in self.entry['instances']]
        actual = self.submit().result(20)
        worker = actual.pop('joint_measurement_worker')
        actual = detector.finalize_joint_piece(actual, self.entry)
        self.assertEqual(expected, actual)
        self.assertEqual('joint_measurement', worker['role'])
        self.assertEqual(1, worker['instance_workers'])
        self.assertFalse(worker['torch_loaded'])
        self.assertNotEqual(os.getpid(), worker['pid'])
        self.assertEqual([1,5], [item['instance_id'] for item in actual['wire_measurements']])
        self.assertEqual([(1,'start'),(1,'end'),(2,'bond'),(6,'bond')],
                         [(item['source_instance_id'], item['endpoint']) for item in actual['solder_measurements']])
        for mask, item in zip(before, self.entry['instances']):
            np.testing.assert_array_equal(mask, item['mask'])
        self.assertTrue(all('height_3d' not in item for item in self.entry['instances']))

    def test_depth_is_reused_from_registration_owner_and_worker_is_persistent(self):
        dummy = np.zeros((3,3), np.uint8)
        existing = self.inputs.prepare(1, dummy, dummy, self.depth)['depth']
        first = self.submit().result(20)
        second = self.submit().result(20)
        self.assertEqual(existing, self.inputs._blocks['depth'][1])
        self.assertEqual(9, len(self.inputs._blocks))  # depth+gray+coarse mask+six source masks
        self.assertEqual(first['joint_measurement_worker']['pid'], second['joint_measurement_worker']['pid'])
        self.assertEqual(first['wire_measurements'], second['wire_measurements'])

    def test_invalid_rows_and_released_board_fail_and_release_lease(self):
        with self.assertRaisesRegex(RuntimeError, 'row bounds'):
            self.submit(piece=dict(y1=0,y2=999)).result(20)
        self.assertEqual(0, self.inputs._leases)
        with self.assertRaisesRegex(RuntimeError, 'released'):
            self.submit(is_current=lambda: False).result(20)
        self.assertEqual(0, self.inputs._leases)
        self.assertIn('wire_measurements', self.submit().result(20))

    def test_late_2d_ng_is_merged_in_parent_without_changing_height(self):
        from algorithm.detect_board import DetectBoard
        relaxed = replace(self.config,max_wire_platform_height_mm=10.,max_solder_platform_height_mm=10.,max_solder_chip_height_mm=10.)
        raw = self.submit(height_config=relaxed).result(20)
        self.assertTrue(raw['is_ok'])
        before = copy.deepcopy(raw['wire_measurements'])
        entry = {**self.entry, 'silver_overflow_2d':dict(status='overflow',is_ok=False)}
        merged = DetectBoard.finalize_joint_piece(raw, entry)
        self.assertFalse(merged['is_ok'])
        self.assertEqual(before, merged['wire_measurements'])

    def test_formal_pipeline_never_calls_parent_mapping_or_height(self):
        from algorithm.detect_board import DetectBoard
        from algorithm.board_pipeline import BoardPipeline
        service, source, depth, height, matrix = self.service, self.entry, self.depth, self.config, self.registration
        class Detector(FakeDetector):
            submit_joint_measurement = DetectBoard.submit_joint_measurement
            finalize_joint_piece = staticmethod(DetectBoard.finalize_joint_piece)

            def __init__(self):
                super().__init__(count=1,delay=0)
                self.joint_measurement_process = service
                self.log = lambda _:None
                self.height_config = replace(height,max_wire_platform_height_mm=10.,max_solder_platform_height_mm=10.,max_solder_chip_height_mm=10.)
                self.min_component_area, self.crop_roi, self.wire_rois = 4, (0,0,240,180), []
                self.segmentation_config = import_module('algorithm.3D.segment').SegmentationConfig(
                    expected_piece_count=1,mode='fixed',fixed_rois=((0,11,240,191),))
                self._align_instances = Mock(side_effect=AssertionError('parent mapping executed'))
                self._measure_piece = Mock(side_effect=AssertionError('parent height executed'))

            def _read_3d(self,req):return depth
            def register_prepared_piece(self,*args):return matrix
            def prepare_joint_piece(self,prepared):return {**source,'piece_id':1,'board_id':prepared['board_id']}
            def finish_2d_piece(self,prepared,*,publish_cache,on_joint_ready=None):
                response, entry = super().finish_2d_piece(prepared,publish_cache=publish_cache)
                entry.update(silver_overflow_2d=dict(status='overflow',is_ok=False))
                return response, entry

        detector = Detector()
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu, ThreadPoolExecutor(1) as joint:
            pipe = BoardPipeline(detector,cpu,gpu,joint_pool=joint)
            board = pipe.submit_3d(dict(board_id='A'))
            owner = pipe._boards['A'].registration_inputs
            self.assertTrue(pipe.submit_2d(dict(board_id='A',piece_id=1)).result(10)['ok'])
            result = board.result(20)
            self.assertTrue(result['ok'],result)
            piece = result['results_board']['pieces'][0]
            self.assertNotEqual(os.getpid(),piece['joint_measurement_worker']['pid'])
            self.assertFalse(piece['is_ok'])
            self.assertEqual('overflow',piece['silver_overflow_2d']['status'])
            self.assertEqual(11,piece['depth_start_row'])
            self.assertEqual({},owner._blocks)
            self.assertEqual(0,owner._leases)
            detector._align_instances.assert_not_called()
            detector._measure_piece.assert_not_called()
            self.assertGreaterEqual(piece['timings']['joint_measurement_shared_memory_ms'],0)
            pipe.close()


class JointProcessStructureTests(unittest.TestCase):
    def test_model_free_import_does_not_load_torch(self):
        code = "import sys;sys.path.insert(0,'python');from algorithm.joint_geometry import JointGeometry;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(0,result.returncode,result.stderr)

    def test_32_core_partition_leaves_eight_main_cores_and_restores_in_reverse(self):
        from algorithm.cpu_partition import choose_partition
        server = AlgorithmServer()
        self.addCleanup(server.close)
        server.board = FakeDetector()
        cores = [dict(logical_cpus=[2*i,2*i+1], efficiency_class=0) for i in range(32)]
        available, starts, stops = list(range(64)), [], []
        def factory(role):
            def create(count, affinity, **kwargs):
                original = list(available)
                plan = choose_partition(cores, available, affinity['physical_cores'])
                available[:] = plan['main_cpus']
                starts.append(role)
                service = Mock()
                service.info = dict(backend='process',workers=[dict(pid=role,cpu_ids=plan['registration_cpus'])])
                def close():
                    stops.append(role)
                    available[:] = original
                service.close.side_effect = close
                return service
            return create
        from algorithm.config import load_runtime_config
        path = ROOT/'config/config.json'
        settings = server._thread_settings(load_runtime_config(path.read_text(encoding='utf-8'), str(path)))
        with patch('algorithm.registration_process.RegistrationProcesses',side_effect=factory('registration')), \
                patch('algorithm.cpu_2d_process.Cpu2DProcesses',side_effect=factory('cpu_2d')), \
                patch('algorithm.joint_process.JointMeasurementProcesses',side_effect=factory('joint_measurement')), \
                patch('algorithm.cpu_partition.current_cpus',side_effect=lambda:list(available)):
            server._configure_cpu_processes(settings)
        groups = [set(server.threading_info[role+'_execution']['workers'][0]['cpu_ids'])
                  for role in ('registration','cpu_2d','joint_measurement')]
        groups.append(set(server.threading_info['main_cpu_ids']))
        self.assertEqual([16]*4, [len(group) for group in groups])
        for i, group in enumerate(groups):
            self.assertTrue(all(group.isdisjoint(other) for other in groups[i+1:]))
        server._stop_cpu_processes()
        self.assertEqual(list(reversed(starts)),stops)
        self.assertEqual(list(range(64)),available)

    def test_joint_options_validate_without_starting_processes(self):
        server = AlgorithmServer.__new__(AlgorithmServer)
        for fields in (dict(joint_affinity=None),dict(joint_affinity=dict(enabled=1)),
                       dict(joint_workers=3,joint_affinity=dict(enabled=True,physical_cores=2)),
                       dict(instance_workers=0)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                server._thread_settings(dict(common_3d=dict(parameters=dict(threading={
                    'worker_budget':50,'cpu_workers':2,'joint_workers':2,**fields}))))

    def test_formal_joint_submit_has_no_parent_execution_fallback(self):
        from algorithm.detect_board import DetectBoard
        detector = DetectBoard(lambda _:None)
        self.addCleanup(detector.release)
        with self.assertRaisesRegex(RuntimeError,'measurement processes must be initialized'):
            detector.submit_joint_measurement(None,None,1,{},None,{},None,{}, {},{},lambda:True)

    def test_third_pool_failure_closes_earlier_pools_in_reverse(self):
        server = AlgorithmServer()
        self.addCleanup(server.close)
        server.board = FakeDetector()
        order = []
        def created(role):
            instance = Mock()
            instance.info = dict(backend='process')
            instance.close.side_effect = lambda:order.append(role)
            return instance
        with patch('algorithm.registration_process.RegistrationProcesses',return_value=created('registration')), \
                patch('algorithm.cpu_2d_process.Cpu2DProcesses',return_value=created('cpu_2d')), \
                patch('algorithm.joint_process.JointMeasurementProcesses',side_effect=RuntimeError('joint startup failed')):
            with self.assertRaisesRegex(RuntimeError,'joint startup failed'):
                server._configure_cpu_processes(server._thread_settings({}))
        self.assertEqual(['cpu_2d','registration'],order)
        self.assertIsNone(server.joint_measurement_process)
        self.assertFalse(server.threading_info['joint_measurement_execution']['ready'])


if __name__=='__main__':unittest.main()
