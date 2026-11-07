"""Event-gated regression of the registration -> measurement handoff (no models)."""
import os
from pathlib import Path
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from algorithm.board_pipeline import BoardPipeline
from pipeline_fakes import FakeDetector
from transport.bridge_server import AlgorithmServer
from test_split_stage_pools import config


class RegistrationSettingsTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.server = AlgorithmServer()
        self.addCleanup(self.server.close)
        self.server.board = FakeDetector()

    def test_new_pool_is_independent_injected_and_budgeted(self):
        pipeline = self.server.get_pipeline()
        settings = self.server._thread_settings(config(budget=18, registration_workers=3))
        self.server._configure_thread_pools(settings, source="test")
        self.assertFalse(settings["registration_pool_shared"])
        self.assertIs(pipeline.registration_pool, self.server.registration_pool)
        self.assertIsNot(pipeline.registration_pool, pipeline.joint_pool)
        self.assertEqual(3, pipeline.registration_pool._max_workers)
        self.assertTrue(pipeline.registration_pool.submit(lambda: threading.current_thread().name)
                        .result(2).startswith("bridge-registration"))
        with self.assertRaisesRegex(ValueError, "effective_total=18"):
            self.server._thread_settings(config(budget=17, registration_workers=3))

    def test_missing_key_aliases_joint_and_environment_can_override(self):
        settings = self.server._thread_settings(config())
        self.server._configure_thread_pools(settings, source="test")
        self.assertTrue(settings["registration_pool_shared"])
        self.assertIs(self.server.registration_pool, self.server.joint_pool)
        with patch.dict(os.environ, {"BRIDGE_REGISTRATION_WORKERS": "2"}):
            settings = self.server._thread_settings(config(budget=17))
            self.assertEqual(2, settings["registration_workers"])
            explicit = self.server._thread_settings(config(budget=18, registration_workers=3))
            self.assertEqual(3, explicit["registration_workers"])

    def test_invalid_registration_counts_rejected(self):
        for value in (0, -1, None, True, 1.5, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "registration_workers"):
                self.server._thread_settings(config(budget=100, registration_workers=value))

    def test_independent_early_window_is_not_limited_by_measurement_workers(self):
        settings = self.server._thread_settings(config(budget=18, registration_workers=3,
                                                       joint_workers=2, early_joint_workers=4))
        self.assertEqual(4, settings["early_joint_workers"])
        self.assertEqual(2, settings["joint_workers"])

    def test_registration_pool_creation_failure_keeps_previous_executors(self):
        previous = self.server.cpu_pool
        created = []
        def create(**values):
            if values["thread_name_prefix"] == "bridge-registration":
                raise RuntimeError("registration creation failed")
            pool = ThreadPoolExecutor(**values)
            created.append(pool)
            return pool
        with patch("transport.bridge_server.ThreadPoolExecutor", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "registration creation failed"):
                self.server._configure_thread_pools(self.server._thread_settings(
                    config(budget=18, registration_workers=3)), source="test")
        self.assertTrue(all(pool._shutdown for pool in created))
        self.assertIs(previous, self.server.cpu_pool)
        self.assertEqual(1, previous.submit(lambda: 1).result(2))

    def test_reconfigure_and_close_drain_every_unique_pool_once(self):
        self.server._configure_thread_pools(self.server._thread_settings(
            config(budget=18, registration_workers=3)), source="test")
        old = {getattr(self.server, key+"_pool") for key in
               ("cpu", "gpu", "silver", "postprocess", "registration", "joint")}
        for pool in old:
            pool.shutdown = Mock(wraps=pool.shutdown)
        self.server._configure_thread_pools(self.server._thread_settings(
            config(budget=19, registration_workers=4)), source="test")
        for pool in old:
            pool.shutdown.assert_called_once_with(wait=True)
        final = self.server.registration_pool
        final.shutdown = Mock(wraps=final.shutdown)
        self.server.close()
        self.server.close()
        final.shutdown.assert_called_once_with(wait=True)

    def test_default_project_enables_four_registration_two_measurement(self):
        from algorithm.config import load_runtime_config
        path = ROOT / "config/config.json"
        settings = self.server._thread_settings(load_runtime_config(path.read_text(encoding="utf-8"), str(path)))
        self.assertEqual(4, settings["registration_workers"])
        self.assertEqual(2, settings["joint_workers"])
        self.assertEqual(2, settings["early_joint_workers"])


class RegistrationSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector(count=4, delay=0)
        self.cpu = ThreadPoolExecutor(2)
        self.gpu = ThreadPoolExecutor(1)
        self.post = ThreadPoolExecutor(1)
        self.registration = ThreadPoolExecutor(1, thread_name_prefix="test-registration")
        self.measurement = ThreadPoolExecutor(1, thread_name_prefix="test-measurement")
        self.pipeline = BoardPipeline(self.detector, self.cpu, self.gpu, postprocess_pool=self.post,
                                      registration_pool=self.registration, joint_pool=self.measurement)
        self.gates = []

    def gate(self):
        event = threading.Event()
        self.gates.append(event)
        return event

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        self.pipeline.close()
        for pool in (self.cpu, self.gpu, self.post, self.registration, self.measurement):
            pool.shutdown(wait=True)

    def prepared(self, board="A", **values):
        for pid in range(1, self.detector.expected_piece_count+1):
            response = self.pipeline.submit_2d(dict(board_id=board, piece_id=pid, marker=f"{board}:{pid}", **values)).result(3)
            self.assertTrue(response["ok"], response)

    def test_registration_continues_while_measurement_is_blocked_and_preserves_matrix(self):
        self.prepared()
        gate, all_registered, measured = self.gate(), threading.Event(), threading.Event()
        matrices, seen, threads = {}, [], []
        def register(pid, *args):
            threads.append(threading.current_thread().name)
            result = dict(is_valid=True, matrix=[[1, 0, pid], [0, 1, 0]])
            matrices[pid] = result
            if len(matrices) == 4:
                all_registered.set()
            return result
        original = self.detector.measure_registered_piece
        def measure(pid, entry, depth, piece, matrix, correction):
            self.assertIs(matrices[pid], matrix)
            self.assertTrue(threading.current_thread().name.startswith("test-measurement"))
            seen.append(pid)
            measured.set()
            if not gate.wait(3):
                raise TimeoutError("measurement gate")
            return {**original(pid, entry, depth, piece, matrix, correction), "registration": matrix}
        self.detector.register_prepared_piece = register
        self.detector.measure_registered_piece = measure
        board = self.pipeline.submit_3d(dict(board_id="A"))
        try:
            self.assertTrue(measured.wait(2))
            self.assertTrue(all_registered.wait(2), "registration worker waited for measurement")
            self.assertEqual([1], seen)
            self.assertFalse(board.done())
        finally:
            gate.set()
        result = board.result(3)
        self.assertTrue(result["ok"], result)
        self.assertTrue(all(name.startswith("test-registration") for name in threads))
        pieces = result["results_board"]["pieces"]
        self.assertEqual([1, 2, 3, 4], [piece["piece_id"] for piece in pieces])
        for piece in pieces:
            timings = piece["timings"]
            self.assertEqual(timings["joint_phase_ms"], timings["registration_phase_ms"]+timings["joint_measurement_phase_ms"])
            self.assertEqual(timings["joint_queue_ms"], timings["registration_queue_ms"])
            self.assertGreaterEqual(timings["joint_pipeline_ms"], timings["joint_phase_ms"])
            self.assertGreaterEqual(timings["joint_measurement_queue_ms"], 0)

    def test_each_stage_enforces_its_own_active_task_limit(self):
        self.prepared()
        self.registration.shutdown(wait=True)
        self.measurement.shutdown(wait=True)
        self.registration = ThreadPoolExecutor(2)
        self.measurement = ThreadPoolExecutor(2)
        self.pipeline.registration_pool = self.registration
        self.pipeline.joint_pool = self.measurement
        registration_gate, measurement_gate = self.gate(), self.gate()
        two_registering, two_measuring, all_registered = (threading.Event() for _ in range(3))
        lock = threading.Lock()
        counts = dict(registration=0, measurement=0, peak_registration=0, peak_measurement=0, registered=0)
        def register(pid, *args):
            with lock:
                counts["registration"] += 1
                counts["peak_registration"] = max(counts["peak_registration"], counts["registration"])
                if counts["registration"] == 2:
                    two_registering.set()
            try:
                if not registration_gate.wait(3):
                    raise TimeoutError("registration concurrency gate")
                with lock:
                    counts["registered"] += 1
                    if counts["registered"] == 4:
                        all_registered.set()
                return dict(is_valid=True, piece=pid)
            finally:
                with lock:
                    counts["registration"] -= 1
        def measure(pid, entry, depth, piece, matrix, correction):
            self.assertEqual(pid, matrix["piece"])
            with lock:
                counts["measurement"] += 1
                counts["peak_measurement"] = max(counts["peak_measurement"], counts["measurement"])
                if counts["measurement"] == 2:
                    two_measuring.set()
            try:
                if not measurement_gate.wait(3):
                    raise TimeoutError("measurement concurrency gate")
                return dict(piece_id=pid, registration=matrix, is_ok=True)
            finally:
                with lock:
                    counts["measurement"] -= 1
        self.detector.register_prepared_piece = register
        self.detector.measure_registered_piece = measure
        board = self.pipeline.submit_3d(dict(board_id="A"))
        try:
            self.assertTrue(two_registering.wait(2))
            registration_gate.set()
            self.assertTrue(two_measuring.wait(2))
            self.assertTrue(all_registered.wait(2))
            self.assertFalse(board.done())
        finally:
            registration_gate.set()
            measurement_gate.set()
        self.assertTrue(board.result(3)["ok"])
        self.assertEqual(2, counts["peak_registration"])
        self.assertEqual(2, counts["peak_measurement"])

    def test_registration_failure_never_submits_measurement_and_board_can_retry(self):
        self.prepared()
        original = self.detector.register_prepared_piece
        self.detector.register_prepared_piece = Mock(side_effect=ValueError("bad registration"))
        self.detector.measure_registered_piece = Mock(wraps=self.detector.measure_registered_piece)
        result = self.pipeline.submit_3d(dict(board_id="A")).result(3)
        self.assertEqual("joint_detection_failed", result["error"])
        self.assertIn("bad registration", result["message"])
        self.detector.measure_registered_piece.assert_not_called()
        self.detector.register_prepared_piece = original
        self.prepared()
        self.assertTrue(self.pipeline.submit_3d(dict(board_id="A")).result(3)["ok"])

    def test_release_during_registration_does_not_submit_old_result(self):
        self.prepared()
        gate, started, drained = self.gate(), threading.Event(), threading.Event()
        def register(*args):
            started.set()
            if not gate.wait(3):
                raise TimeoutError("registration gate")
            drained.set()
            return dict(is_valid=True)
        self.detector.register_prepared_piece = register
        self.detector.measure_registered_piece = Mock(wraps=self.detector.measure_registered_piece)
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(started.wait(2))
        self.pipeline.release_board("A")
        self.assertEqual("board_released", board.result(2)["error"])
        self.prepared()  # Same board ID, new generation must not receive the old matrix.
        gate.set()
        self.assertTrue(drained.wait(2))
        self.registration.submit(lambda: None).result(2)
        self.detector.measure_registered_piece.assert_not_called()
        self.assertEqual({}, self.pipeline._boards["A"].results)

    def test_queued_measurements_skip_released_board(self):
        self.prepared()
        gate, occupied = self.gate(), threading.Event()
        def occupy():
            occupied.set()
            gate.wait(3)
        self.measurement.submit(occupy)
        self.assertTrue(occupied.wait(2))
        all_registered = threading.Event()
        calls = []
        # Observe callback submissions rather than relying on sleeps/queue sizes.
        submit = self.measurement.submit
        def record(*args, **kwargs):
            result = submit(*args, **kwargs)
            calls.append(result)
            if len(calls) == 4:
                all_registered.set()
            return result
        self.measurement.submit = record
        self.detector.measure_registered_piece = Mock(wraps=self.detector.measure_registered_piece)
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(all_registered.wait(2))
        self.pipeline.release_board("A")
        gate.set()
        self.assertEqual("board_released", board.result(2)["error"])
        for future in calls:
            with self.assertRaisesRegex(RuntimeError, "released"):
                future.result(2)
        self.detector.measure_registered_piece.assert_not_called()

    def test_submission_failure_completes_board(self):
        self.prepared()
        with patch.object(self.measurement, "submit", side_effect=RuntimeError("pool unavailable")):
            result = self.pipeline.submit_3d(dict(board_id="A")).result(3)
        self.assertEqual("joint_detection_failed", result["error"])
        self.assertIn("pool unavailable", result["message"])

    def test_measurement_failure_completes_board(self):
        self.prepared()
        self.detector.measure_registered_piece = Mock(side_effect=ValueError("bad measurement"))
        result = self.pipeline.submit_3d(dict(board_id="A")).result(3)
        self.assertEqual("joint_detection_failed", result["error"])
        self.assertIn("bad measurement", result["message"])

    def test_released_board_work_drains_before_model_reload(self):
        self.prepared()
        gate, started, drained = self.gate(), threading.Event(), threading.Event()
        original = self.detector.measure_registered_piece
        def measure(*args):
            started.set()
            if not gate.wait(3):
                raise TimeoutError("reload drain gate")
            return original(*args)
        self.detector.measure_registered_piece = measure
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(started.wait(2))
        self.pipeline.release_board("A")
        self.assertEqual("board_released", board.result(2)["error"])
        thread = threading.Thread(target=lambda: (self.pipeline.drain_joint_jobs(), drained.set()))
        thread.start()
        try:
            self.assertFalse(drained.wait(.05))
        finally:
            gate.set()
            thread.join(3)
        self.assertTrue(drained.is_set())

    def test_registration_submission_failure_completes_board(self):
        self.prepared()
        with patch.object(self.registration, "submit", side_effect=RuntimeError("registration pool unavailable")):
            result = self.pipeline.submit_3d(dict(board_id="A")).result(3)
        self.assertEqual("joint_detection_failed", result["error"])
        self.assertIn("registration pool unavailable", result["message"])


class DetectorStageEquivalenceTests(unittest.TestCase):
    def test_wrapper_and_split_have_identical_results_and_full_width_depth(self):
        import numpy as np
        from algorithm import detect_board as module
        detector = module.DetectBoard(lambda _: None)
        self.addCleanup(detector.release)
        gray = np.zeros((80, 100), np.uint8)
        depth = np.zeros((160, 100), np.uint16)
        piece = dict(piece_id=1, x1=10, x2=90, y1=20, y2=100)
        entry = dict(gray_image=gray, registration_mask=gray, instances=[], image_shape=[100, 80],
                     crop_box=[0, 0, 100, 80], roi_boxes=[])
        matrix = dict(is_valid=True, affine_valid=True, scale=1, tx=0, ty=0,
                      homography_2d_to_3d=np.eye(3).tolist())
        def measure(local, *args, **kwargs):
            self.assertEqual((80, 100), local.shape)
            self.assertTrue(np.shares_memory(local, depth))
            return dict(is_ok=True, wire_measurements=[], solder_measurements=[], reference_planes={})
        detector._align_instances = Mock(return_value=[])
        detector._measure_piece = Mock(side_effect=measure)
        with patch.object(module, "platform_mask_3d", return_value=gray), \
                patch.object(module, "register_platform_masks", return_value=matrix), \
                patch.object(module, "refine_affine_registration", return_value=matrix) as refine:
            direct = detector.measure_prepared_piece(1, entry, depth, piece)
            self.assertEqual(1, refine.call_count)
            fitted = detector.register_prepared_piece(1, entry, depth, piece)
            self.assertIs(fitted, matrix)
            split = detector.measure_registered_piece(1, entry, depth, piece, fitted)
            self.assertEqual(2, refine.call_count, "measurement unexpectedly reran registration")
        self.assertEqual(direct, split)


if __name__ == "__main__":
    unittest.main()
