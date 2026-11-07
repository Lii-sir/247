"""Synthetic-only shared-input, persistent process, affinity and lifecycle tests."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
from algorithm.cpu_partition import choose_partition, parse_cores
from algorithm.registration import RegistrationConfig
from algorithm.registration_process import RegistrationProcesses, SharedBoardInputs
from registration_process_tasks import hold


def fixture():
    mask = (np.random.default_rng(2).random((96, 128)) > .5).astype(np.uint8)*255
    depth = (mask > 0).astype(np.uint16)*33000
    config = RegistrationConfig.from_dict(dict(scale=1, roi_3d=[2, 2, 30, 40],
        search_2d_scaled=[0, 0, 128, 96], morphology_kernel_size=1))
    values = dict(score_mode="full", preview_scale=1, morphology_kernel_size=1,
                  rois=[[0, 0, 128, 96]], maxiter=1, maxfev=12)
    return dict(gray_image=mask, registration_mask=mask), depth, config, values


class CpuPartitionTests(unittest.TestCase):
    def test_configuration_rejects_invalid_affinity_and_too_many_processes(self):
        from transport.bridge_server import AlgorithmServer
        server = AlgorithmServer.__new__(AlgorithmServer)
        for affinity in (None, [], dict(enabled=1), dict(enabled=True, physical_cores=0),
                         dict(enabled=True, physical_cores=1)):
            config = dict(common_3d=dict(parameters=dict(threading=dict(worker_budget=30,
                cpu_workers=2, joint_workers=2, registration_workers=2, registration_affinity=affinity))))
            with self.subTest(affinity=affinity), self.assertRaises(ValueError):
                server._thread_settings(config)

    def test_topology_keeps_smt_siblings_and_prefers_performance_cores(self):
        def record(mask, efficiency):
            return struct.pack("<II", 0, 48)+bytes([1, efficiency])+bytes(20)+struct.pack("<H", 1)+struct.pack("<QH", mask, 0)+bytes(6)
        cores = parse_cores(record(3, 0)+record(12, 2)+record(16, 0), 8)
        selected = choose_partition(cores, range(5), 1)
        self.assertEqual([2, 3], selected["registration_cpus"])
        self.assertEqual([0, 1, 4], selected["main_cpus"])
        self.assertTrue(set(selected["main_cpus"]).isdisjoint(selected["registration_cpus"]))
        with self.assertRaises(ValueError):
            choose_partition(cores, range(5), 3)

    def test_malformed_and_multigroup_topology_are_rejected(self):
        for raw in (b"", b"123", struct.pack("<II", 0, 0), struct.pack("<II", 0, 80)+bytes(40)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_cores(raw, 8)
        other_group = struct.pack("<II", 0, 48)+bytes(22)+struct.pack("<H", 1)+struct.pack("<QH", 1, 1)+bytes(6)
        with self.assertRaisesRegex(RuntimeError, "group 0"):
            parse_cores(other_group, 8)


class SharedInputTests(unittest.TestCase):
    def test_close_waits_for_lease_and_reuses_exact_board_buffers(self):
        entry, depth, _, _ = fixture()
        inputs = SharedBoardInputs()
        inputs.acquire()
        first = inputs.prepare(1, entry["gray_image"], entry["registration_mask"], depth)
        second = inputs.prepare(2, entry["gray_image"], entry["registration_mask"], depth)
        self.assertEqual(first["depth"], second["depth"])
        self.assertEqual(5, len(inputs._blocks))
        inputs.close()
        handle = SharedMemory(name=first["depth"]["name"])
        handle.close()
        with self.assertRaisesRegex(RuntimeError, "released"):
            inputs.acquire()
        inputs.release()
        for descriptor in (*first.values(), *second.values()):
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=descriptor["name"])
        inputs.close()

    def test_partial_preparation_failure_still_cleans_created_depth(self):
        inputs = SharedBoardInputs()
        inputs.acquire()
        with self.assertRaises(ValueError):
            inputs.prepare(1, None, None, np.zeros((8, 8), np.uint16))
        name = inputs._blocks["depth"][0].name
        inputs.close()
        inputs.release()
        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name)


class ProcessRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RegistrationProcesses(2, dict(enabled=False))

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def setUp(self):
        self.inputs = SharedBoardInputs()
        self.pool = ThreadPoolExecutor(1)
        self.addCleanup(self.pool.shutdown, wait=True)
        self.addCleanup(self.inputs.close)
        self.entry, self.depth, self.config, self.values = fixture()

    def submit(self, **overrides):
        params = dict(preparation_pool=self.pool, inputs=self.inputs, piece_id=1, entry=self.entry,
                      depth=self.depth, piece=dict(y1=0, y2=96), correction={},
                      registration_config=self.config, affine_config=self.values, timings={},
                      metadata=dict(board_id="A", piece_id=1, request_id="test"), is_current=lambda: True)
        params.update(overrides)
        return self.service.submit(**params)

    def test_persistent_workers_report_no_models_and_reuse_inputs(self):
        first = self.submit().result(20)
        second = self.submit().result(20)
        self.assertEqual(first["matrix_2d_to_3d"], second["matrix_2d_to_3d"])
        self.assertEqual(3, len(self.inputs._blocks))
        self.assertEqual(2, len({worker["pid"] for worker in self.service.info["workers"]}))
        self.assertFalse(first["registration_worker"]["torch_loaded"])
        self.assertEqual(1, first["registration_worker"]["opencv_threads"])

    def test_worker_error_releases_lease_and_pool_remains_usable(self):
        with self.assertRaisesRegex(RuntimeError, "does not fit"):
            self.submit(registration_config=RegistrationConfig()).result(20)
        self.assertEqual(0, self.inputs._leases)
        self.assertIn("matrix_2d_to_3d", self.submit().result(20))

    def test_preparation_and_submission_failures_complete_future(self):
        with self.assertRaises(KeyError):
            self.submit(entry={}).result(10)
        self.assertEqual(0, self.inputs._leases)
        with patch.object(self.pool, "submit", side_effect=RuntimeError("preparation stopped")):
            with self.assertRaisesRegex(RuntimeError, "preparation stopped"):
                self.submit().result(10)
        self.assertEqual(0, self.inputs._leases)

    def test_released_board_is_not_dispatched(self):
        with self.assertRaisesRegex(RuntimeError, "released"):
            self.submit(is_current=lambda: False).result(10)
        self.assertEqual({}, self.inputs._blocks)
        self.assertEqual(0, self.inputs._leases)

    def test_cancelled_public_future_keeps_real_worker_input_lease(self):
        flag = SharedMemory(create=True, size=3)
        flag.buf[:] = b"\0\0\0"
        holders = [self.service._executor.submit(hold, flag.name, i) for i in (0, 1)]
        try:
            deadline = time.monotonic()+10
            while not all(flag.buf[:2]):
                if time.monotonic() > deadline:
                    self.fail("process gate did not start")
                time.sleep(.01)
            future = self.submit()
            self.pool.submit(lambda: None).result(10)
            name = self.inputs._blocks["depth"][0].name
            self.assertTrue(future.cancel())
            self.inputs.close()
            self.assertEqual(1, self.inputs._leases)
            handle = SharedMemory(name=name)
            handle.close()
            pending = list(self.service._pending)
            flag.buf[2] = 1
            for holder in holders:
                holder.result(10)
            self.assertFalse(wait(pending, timeout=10).not_done)
            self.assertEqual(0, self.inputs._leases)
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=name)
        finally:
            flag.buf[2] = 1
            wait(holders, timeout=15)
            flag.close()
            flag.unlink()


class WindowsProcessProbeTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows affinity integration")
    def test_hidden_spawn_numeric_parity_and_real_affinity_restore(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tests/registration_process_probe.py"), "--affinity"],
                                capture_output=True, text=True, timeout=45,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(0, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["affinity_before"], report["affinity_after"])
        self.assertEqual(4, report["pieces"])
        self.assertFalse(report["torch_loaded"])


class ProcessFailureTests(unittest.TestCase):
    def test_broken_worker_fails_next_request_and_releases_lease(self):
        from concurrent.futures.process import BrokenProcessPool
        service = RegistrationProcesses(1, dict(enabled=False))
        self.addCleanup(service.close)
        with self.assertRaises(BrokenProcessPool):
            service._executor.submit(os._exit, 7).result(15)
        inputs = SharedBoardInputs()
        self.addCleanup(inputs.close)
        entry, depth, config, values = fixture()
        with ThreadPoolExecutor(1) as preparation:
            future = service.submit(preparation, inputs, 1, entry, depth, dict(y1=0, y2=96), {},
                config, values, {}, dict(board_id="broken", piece_id=1, request_id=None), lambda: True)
            with self.assertRaises(RuntimeError):
                future.result(15)
        self.assertEqual(0, inputs._leases)
        names = [handle.name for handle, _, _ in inputs._blocks.values()]
        inputs.close()
        for name in names:
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=name)

    @unittest.skipUnless(os.name == "nt", "Windows initializer rejection")
    def test_initializer_failure_restores_launcher_and_reports_error(self):
        from multiprocessing.spawn import get_executable
        from algorithm.cpu_partition import current_cpus
        original, executable, launcher = current_cpus(), get_executable(), os.environ.get("__PYVENV_LAUNCHER__")
        invalid = dict(original_cpus=original, main_cpus=original,
                       registration_cores=[dict(logical_cpus=[999], efficiency_class=0)],
                       registration_cpus=[999])
        with patch("algorithm.cpu_partition.choose_partition", return_value=invalid):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                RegistrationProcesses(1, dict(enabled=True, physical_cores=1), startup_timeout=10)
        self.assertEqual(original, current_cpus())
        self.assertEqual(executable, get_executable())
        self.assertEqual(launcher, os.environ.get("__PYVENV_LAUNCHER__"))


class ProcessPipelineIntegrationTests(unittest.TestCase):
    def test_initialization_failure_releases_model_without_fallback(self):
        from pipeline_fakes import FakeDetector
        from transport.bridge_server import AlgorithmServer
        server = AlgorithmServer()
        self.addCleanup(server.close)
        detector = FakeDetector()
        detector.handle_init = Mock(return_value=dict(ok=True))
        detector.release = Mock()
        server.board = detector
        with patch.object(server, "_configure_registration_process", side_effect=RuntimeError("worker init failed")):
            response = server.handle_init_algorithm(dict(stage="board", config_text="{}"))
        self.assertFalse(response["ok"])
        self.assertIn("worker init failed", response["error"])
        detector.release.assert_called_once()

    def test_bridge_initialization_and_two_complete_boards_use_same_processes(self):
        from algorithm.detect_board import DetectBoard
        from importlib import import_module
        from pipeline_fakes import FakeDetector
        from transport.bridge_server import AlgorithmServer
        from algorithm.registration_worker import register_arrays
        entry, local, registration_config, affine = fixture()
        expected = register_arrays(entry["gray_image"], entry["registration_mask"], local,
                                   registration_config, affine, {})
        class Detector(FakeDetector):
            submit_registration = DetectBoard.submit_registration

            def __init__(self):
                super().__init__(count=2, delay=0)
                self.log = lambda text: None
                self.registration_config = registration_config
                self.dynamic_affine_config = affine
                self.segmentation_config = import_module("algorithm.3D.segment").SegmentationConfig(
                    expected_piece_count=2, mode="fixed", fixed_rois=((0, 0, 128, 96), (0, 96, 128, 192)))

            def handle_init(self, req):
                return dict(ok=True, stage="board")

            def _read_3d(self, req):
                return np.tile(local, (2, 1))

            def prepare_joint_piece(self, prepared):
                return {**prepared, **entry}

            def measure_registered_piece(self, pid, source, depth, piece, matrix, correction):
                return {**self.measure_prepared_piece(pid, source, depth, piece, correction), "registration": matrix}

        with patch.dict(os.environ, {}, clear=True):
            server = AlgorithmServer()
        self.addCleanup(server.close)
        server.board = Detector()
        config = dict(common_2d=dict(parameters=dict(threading=dict(gpu_workers=1, postprocess_workers=1))),
                      common_3d=dict(parameters=dict(threading=dict(worker_budget=7, cpu_workers=2,
                          registration_workers=2, joint_workers=1, early_joint_workers=2))))
        response = server.handle_init_algorithm(dict(stage="board", config_text=json.dumps(config)))
        self.assertTrue(response["ok"], response)
        info = response["threading"]["registration_execution"]
        self.assertTrue(info["ready"])
        self.assertEqual("process", info["backend"])
        pids = {worker["pid"] for worker in info["workers"]}
        pipe = server.get_pipeline()
        for board_id in ("A", "B"):
            replies = [pipe.submit_2d(dict(board_id=board_id, piece_id=pid, marker=f"{board_id}:{pid}")) for pid in (1, 2)]
            self.assertTrue(all(reply.result(10)["ok"] for reply in replies))
            resources = pipe._boards[board_id].registration_inputs
            result = pipe.submit_3d(dict(board_id=board_id)).result(20)
            self.assertTrue(result["ok"], result)
            for piece in result["results_board"]["pieces"]:
                matrix = piece["registration"]
                self.assertIn(matrix["registration_worker"]["pid"], pids)
                self.assertEqual(expected["matrix_2d_to_3d"], matrix["matrix_2d_to_3d"])
                self.assertEqual(expected["affine_score_after"], matrix["affine_score_after"])
                self.assertGreaterEqual(piece["timings"]["registration_shared_memory_ms"], 0)
                self.assertGreaterEqual(piece["timings"]["registration_result_transfer_ms"], 0)
            self.assertTrue(resources._closed)
            self.assertEqual(0, resources._leases)
            self.assertEqual({}, resources._blocks)

    def test_formal_submit_requires_initialized_process_service(self):
        from algorithm.detect_board import DetectBoard
        detector = DetectBoard(lambda _: None)
        self.addCleanup(detector.release)
        with self.assertRaisesRegex(RuntimeError, "processes must be initialized"):
            detector.submit_registration(None, None, 1, {}, None, {}, {}, {}, {}, lambda: True)


if __name__ == "__main__":
    unittest.main()
