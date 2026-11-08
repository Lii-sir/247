"""Synthetic 2D CPU process parity and dispatch tests; never bind real CPU cores."""
import copy
from importlib import import_module
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
import cv2
import numpy as np
from algorithm.cpu_2d_process import Cpu2DProcesses
from algorithm.cpu_2d_worker import compute
from algorithm.timing import timing_context
from algorithm.board_pipeline import BoardPipeline
from pipeline_fakes import FakeDetector
from transport.bridge_server import AlgorithmServer
silver = import_module("algorithm.2D.silver_inspection")


def fixture():
    shape = (120, 180)
    chip = np.zeros(shape, np.uint8)
    chip[30:60, 35:70] = 255
    silver_mask = cv2.dilate(chip, np.ones((17, 17), np.uint8))
    wire = np.zeros_like(chip)
    wire[:, 85:89] = 255
    masks = [chip, silver_mask, wire]
    instances = [dict(instance_id=i+1, class_id=i, class_name=name, confidence=.9, mask=mask,
                      rectangle_ratio=1., rectangle=None, is_ok=True,
                      bending_2d={"is_ok": True} if name == "wire" else None)
                 for i, (name, mask) in enumerate(zip(("chip", "silver", "wire"), masks))]
    settings = dict(overflow=dict(boundary=dict(mode="roi_inset", insets=dict(left=5, top=5, right=5, bottom=5)),
                                  tolerance_px=0, min_area_px=1),
                    continuity=dict(min_chip_area_px=5, outward_length_px=8, sector_count=36))
    crop_box = (10, 20, 190, 140)
    boundary = silver.prepare_boundary(settings, crop_box=crop_box, circle=None, image_shape=shape)
    prepared = dict(joint_entry=dict(instances=instances), crop=np.zeros((*shape, 3), np.uint8),
                    crop_box=crop_box, circle=None, silver_boundary=boundary)
    return prepared, settings


class Cpu2DProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = Cpu2DProcesses(1, dict(enabled=False))

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def submit(self, prepared, settings, **overrides):
        params = dict(preparation_pool=self.pool, prepared=prepared, settings=settings,
                      base_dir=None, timings={}, metadata=dict(board_id="A", piece_id=1, request_id="test"),
                      is_current=lambda: True)
        params.update(overrides)
        return self.service.submit_piece(**params)

    def setUp(self):
        self.pool = ThreadPoolExecutor(1)
        self.addCleanup(self.pool.shutdown, wait=True)

    def test_silver_and_geometry_match_direct_functions_and_preserve_masks(self):
        from algorithm.detect_board import DetectBoard
        prepared, settings = fixture()
        instances = prepared["joint_entry"]["instances"]
        before = [item["mask"].copy() for item in instances]
        expected = silver.inspect_masks(instances, prepared["crop"].shape[:2], settings,
            crop_box=prepared["crop_box"], circle=None, prepared_boundary=prepared["silver_boundary"])
        expected = silver.offset_geometry(expected, prepared["crop_box"][:2])
        detector = DetectBoard(lambda _: None)
        self.addCleanup(detector.release)
        public = [detector._public_2d_instance(item, prepared["crop_box"]) for item in instances]
        result = self.submit(prepared, settings).result(20)
        self.assertEqual(expected, result["diagnostics"])
        self.assertEqual(public, result["public_instances"])
        self.assertEqual("cpu_2d", result["cpu_2d_worker"]["role"])
        self.assertFalse(result["cpu_2d_worker"]["torch_loaded"])
        self.assertEqual(1, result["cpu_2d_worker"]["opencv_threads"])
        self.assertEqual([1, 2, 3], [item["instance_id"] for item in result["public_instances"]])
        for image, original in zip(before, instances):
            np.testing.assert_array_equal(image, original["mask"])

    def test_two_calls_reuse_process_and_keep_timing_boundaries(self):
        prepared, settings = fixture()
        first = self.submit(prepared, settings).result(20)
        timings = {"silver_prepare_phase_ms": 7.}
        second = self.submit(prepared, settings, timings=timings).result(20)
        self.assertEqual(first["cpu_2d_worker"]["pid"], second["cpu_2d_worker"]["pid"])
        self.assertEqual(first["diagnostics"], second["diagnostics"])
        self.assertEqual(timings["silver_total_ms"], timings["silver_mask_checks_ms"]+7.)
        for name in ("cpu_2d_phase_ms", "cpu_2d_queue_ms", "cpu_2d_shared_memory_ms", "cpu_2d_result_transfer_ms"):
            self.assertGreaterEqual(timings[name], 0)

    def test_unrelated_parent_resources_are_not_serialized(self):
        prepared, settings = fixture()
        prepared["joint_entry"]["instances"][0]["unused_model_or_lock"] = threading.Lock()
        prepared["joint_entry"]["instances"][0]["_private_cache"] = threading.Lock()
        result = self.submit(prepared, settings).result(20)
        self.assertEqual(3, len(result["public_instances"]))
        self.assertFalse(result["cpu_2d_worker"]["torch_loaded"])

    def test_no_instances_or_disabled_silver_return_same_empty_geometry(self):
        prepared, _ = fixture()
        prepared["joint_entry"]["instances"] = []
        prepared["silver_boundary"] = silver.PreparedBoundary(None)
        result = self.submit(prepared, {}).result(20)
        self.assertEqual([], result["public_instances"])
        self.assertFalse(result["diagnostics"]["enabled"])

    def test_missing_boundary_and_released_board_fail_without_hanging(self):
        prepared, settings = fixture()
        del prepared["silver_boundary"]
        with self.assertRaisesRegex(RuntimeError, "boundary preparation"):
            self.submit(prepared, settings).result(20)
        prepared, settings = fixture()
        with self.assertRaisesRegex(RuntimeError, "released"):
            self.submit(prepared, settings, is_current=lambda: False).result(20)

    def test_boundary_failure_and_bool_mask_preserve_existing_status(self):
        prepared, settings = fixture()
        prepared["silver_boundary"] = silver.PreparedBoundary(None, "synthetic missing calibration", True)
        prepared["joint_entry"]["instances"][0]["mask"] = prepared["joint_entry"]["instances"][0]["mask"] > 0
        expected = compute(prepared["joint_entry"]["instances"], prepared["crop"].shape[:2], settings,
                           prepared["crop_box"], None, prepared["silver_boundary"], None)
        actual = self.submit(prepared, settings).result(20)
        self.assertEqual(expected["diagnostics"], actual["diagnostics"])
        self.assertEqual(expected["public_instances"], actual["public_instances"])

    def test_parent_finalize_reuses_worker_output_and_matches_full_direct_response(self):
        from algorithm.detect_board import DetectBoard
        detector = DetectBoard(lambda _: None)
        self.addCleanup(detector.release)
        prepared, settings = fixture()
        detector.silver_config = settings
        prepared.update(start=time.perf_counter(), board_id="A", piece_id=1,
            image=np.zeros((140, 190, 3), np.uint8), roi_boxes=[],
            read_decode_ms=0., circle_crop_ms=0., inference_ms=0.)
        prepared["joint_entry"].update(piece_id=1, wire_measurements=[], wire_roi_checks=[],
            chip_rotation_2d=dict(enabled=False), chip_anomaly_2d=dict(enabled=False))
        expected, _ = detector.finish_2d_piece(copy.deepcopy(prepared), publish_cache=False)
        result = self.submit(prepared, settings).result(20)
        prepared["cpu_2d_result"] = result
        with patch.object(detector, "_public_2d_instances", side_effect=AssertionError("shared instance pool used")), \
                patch.object(silver, "inspect_masks", side_effect=AssertionError("silver recomputed in parent")):
            actual, entry = detector.finish_2d_piece(prepared, publish_cache=False)
        for response in (expected, actual):
            response.pop("timings")
            response.pop("elapsed_ms")
        self.assertEqual(expected, actual)
        self.assertEqual(result["diagnostics"]["silver_overflow"], entry["silver_overflow_2d"])

    def test_real_pipeline_handoff_returns_2d_worker_and_late_diagnostics(self):
        from algorithm.detect_board import DetectBoard
        original, settings = fixture()
        service = self.service
        class Detector(FakeDetector):
            submit_2d_postprocess = DetectBoard.submit_2d_postprocess
            finish_2d_piece = DetectBoard.finish_2d_piece
            _silver_config_base_dir = DetectBoard._silver_config_base_dir
            finalize_joint_piece = staticmethod(DetectBoard.finalize_joint_piece)
            elapsed_ms = staticmethod(DetectBoard.elapsed_ms)

            def __init__(self):
                super().__init__(count=1, delay=0)
                self.cpu_2d_process = service
                self.silver_config = settings
                self.config_path = None
                self.log = lambda message: None

            def prepare_2d_piece(self, req):
                prepared = copy.deepcopy(original)
                prepared.update(req, start=time.perf_counter(), image=np.zeros((140, 190, 3), np.uint8),
                    roi_boxes=[], read_decode_ms=0., circle_crop_ms=0., inference_ms=0.)
                prepared["joint_entry"].update(piece_id=1, wire_measurements=[], wire_roi_checks=[],
                    chip_rotation_2d=dict(enabled=False), chip_anomaly_2d=dict(enabled=False))
                return prepared

            def prepare_joint_piece(self, prepared):
                return {**prepared["joint_entry"], "board_id": prepared["board_id"]}

            def prepare_silver_boundary(self, prepared):
                return prepared["silver_boundary"]

        detector = Detector()
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu, ThreadPoolExecutor(1) as post:
            pipe = BoardPipeline(detector, cpu, gpu, postprocess_pool=post)
            board = pipe.submit_3d(dict(board_id="A"))
            response = pipe.submit_2d(dict(board_id="A", piece_id=1)).result(20)
            self.assertTrue(response["ok"], response)
            self.assertEqual("cpu_2d", response["cpu_2d_worker"]["role"])
            timings = response["timings"]
            self.assertEqual(timings["cpu_postprocess_phase_ms"], timings["cpu_2d_phase_ms"]+timings["cpu_postprocess_finalize_phase_ms"])
            self.assertGreaterEqual(timings["cpu_postprocess_pipeline_ms"], timings["cpu_postprocess_phase_ms"])
            final = board.result(20)
            self.assertTrue(final["ok"], final)
            self.assertEqual(response["results_2d"]["is_ok"], final["results_board"]["pieces"][0]["is_ok"])
            self.assertEqual(1, detector.closed_inputs)
            pipe.close()


class CpuPoolPartitionTests(unittest.TestCase):
    def test_disjoint_group_planning_and_reverse_restore_use_mock_topology(self):
        for reg_enabled, cpu_enabled in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(reg=reg_enabled, cpu=cpu_enabled):
                server = AlgorithmServer()
                server.board = FakeDetector()
                available, starts, stops = list(range(12)), [], []
                def factory(role):
                    def create(count, affinity, **kwargs):
                        original = list(available)
                        assigned = original[:count] if affinity["enabled"] else original
                        if affinity["enabled"]:
                            available[:] = original[count:]
                        starts.append(role)
                        service = Mock()
                        service.info = dict(backend="process", workers=[dict(pid=role, cpu_ids=assigned)],
                                            affinity=dict(registration_cpus=assigned) if affinity["enabled"] else None)
                        def close():
                            stops.append(role)
                            if affinity["enabled"]:
                                available[:] = original
                        service.close.side_effect = close
                        return service
                    return create
                settings = server._thread_settings(dict(common_3d=dict(parameters=dict(threading=dict(
                    worker_budget=30, cpu_workers=2, registration_workers=2, joint_workers=2,
                    registration_affinity=dict(enabled=reg_enabled, physical_cores=2)))),
                    common_2d=dict(parameters=dict(threading=dict(cpu_process_workers=2,
                    cpu_process_affinity=dict(enabled=cpu_enabled, physical_cores=2))))))
                try:
                    with patch("algorithm.registration_process.RegistrationProcesses", side_effect=factory("registration")), \
                            patch("algorithm.cpu_2d_process.Cpu2DProcesses", side_effect=factory("cpu_2d")), \
                            patch("algorithm.joint_process.JointMeasurementProcesses", side_effect=factory("joint_measurement")), \
                            patch("algorithm.cpu_partition.current_cpus", side_effect=lambda: list(available)):
                        server._configure_cpu_processes(settings)
                    reg = server.threading_info["registration_execution"]["workers"][0]["cpu_ids"]
                    cpu = server.threading_info["cpu_2d_execution"]["workers"][0]["cpu_ids"]
                    if reg_enabled or cpu_enabled:
                        self.assertTrue(set(reg).isdisjoint(cpu))
                    server._stop_cpu_processes()
                    self.assertEqual(list(reversed(starts)), stops)
                    self.assertEqual(list(range(12)), available)
                finally:
                    server.close()

    def test_invalid_2d_process_configuration_is_rejected(self):
        server = AlgorithmServer.__new__(AlgorithmServer)
        for fields in (dict(cpu_process_workers=0), dict(cpu_process_affinity=None),
                       dict(cpu_process_affinity=dict(enabled=1)),
                       dict(cpu_process_workers=3, cpu_process_affinity=dict(enabled=True, physical_cores=2))):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                server._thread_settings(dict(common_2d=dict(parameters=dict(threading=fields))))

    def test_second_pool_failure_closes_first_and_marks_both_not_ready(self):
        server = AlgorithmServer()
        self.addCleanup(server.close)
        server.board = FakeDetector()
        service = Mock()
        service.info = dict(backend="process")
        with patch("algorithm.registration_process.RegistrationProcesses", return_value=service), \
                patch("algorithm.cpu_2d_process.Cpu2DProcesses", side_effect=RuntimeError("2D startup failed")):
            with self.assertRaisesRegex(RuntimeError, "2D startup failed"):
                server._configure_cpu_processes(server._thread_settings({}))
        service.close.assert_called_once()
        self.assertIsNone(server.registration_process)
        self.assertIsNone(server.cpu_2d_process)
        self.assertFalse(server.threading_info["registration_execution"]["ready"])
        self.assertFalse(server.threading_info["cpu_2d_execution"]["ready"])


class PostprocessSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector(count=1)
        self.cpu, self.gpu, self.post = ThreadPoolExecutor(2), ThreadPoolExecutor(1), ThreadPoolExecutor(1)
        self.pipe = BoardPipeline(self.detector, self.cpu, self.gpu, postprocess_pool=self.post)
        self.pending, self.started = Future(), threading.Event()
        def submit(*args):
            self.started.set()
            return self.pending
        self.detector.submit_2d_postprocess = submit

    def tearDown(self):
        if not self.pending.done():
            self.pending.set_result(None)
        self.pipe.close()
        for pool in (self.cpu, self.gpu, self.post):
            pool.shutdown(wait=True)

    def test_cpu_process_wait_does_not_occupy_parent_pool_or_block_early_3d(self):
        board = self.pipe.submit_3d(dict(board_id="A"))
        piece = self.pipe.submit_2d(dict(board_id="A", piece_id=1))
        self.assertTrue(self.started.wait(2))
        self.assertEqual("free", self.post.submit(lambda: "free").result(1))
        self.assertTrue(self.detector.measured.wait(2))
        self.assertFalse(piece.done())
        self.assertFalse(board.done())
        self.pending.set_result(None)
        self.assertTrue(piece.result(2)["ok"])
        self.assertTrue(board.result(2)["ok"])

    def test_process_failure_releases_image_and_fails_board(self):
        board = self.pipe.submit_3d(dict(board_id="A"))
        piece = self.pipe.submit_2d(dict(board_id="A", piece_id=1))
        self.assertTrue(self.started.wait(2))
        self.assertTrue(self.detector.measured.wait(2))
        self.pending.set_exception(RuntimeError("2D CPU failed"))
        self.assertFalse(piece.result(2)["ok"])
        self.assertFalse(board.result(2)["ok"])
        self.assertEqual(1, self.detector.closed_inputs)


if __name__ == "__main__":
    unittest.main()
