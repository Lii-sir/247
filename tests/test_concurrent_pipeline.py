import io
import json
import mmap
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from algorithm.board_pipeline import BoardPipeline
from transport.bridge_server import AlgorithmServer, configure_native_threads
from transport.shared_memory import SharedImage
from pipeline_fakes import FakeDetector


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cpu = ThreadPoolExecutor(max_workers=12)
        self.postprocess = ThreadPoolExecutor(max_workers=2)
        self.gpu = ThreadPoolExecutor(max_workers=1)
        self.detector = FakeDetector()
        self.pipeline = BoardPipeline(self.detector, self.cpu, self.gpu, timeout_seconds=1,
                                      postprocess_pool=self.postprocess)

    def tearDown(self):
        self.detector.gpu_gate.set()
        self.pipeline.close()
        self.gpu.shutdown(wait=True)
        self.postprocess.shutdown(wait=True)
        self.cpu.shutdown(wait=True)

    def pieces(self, board="A", **kwargs):
        return [self.pipeline.submit_2d(dict(board_id=board, piece_id=pid,
                                            marker=f"{board}:{pid}", **kwargs))
                for pid in range(1, 11)]

    def test_3d_first_waits_without_worker_and_pairs_before_ten(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(self.detector.depth_started.wait(1))
        self.assertFalse(board.done())
        first = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="A:1"))
        self.assertTrue(first.result(2)["ok"])
        self.assertTrue(self.detector.measured.wait(1))
        self.assertFalse(board.done())
        rest = [self.pipeline.submit_2d(dict(board_id="A", piece_id=pid, marker=f"A:{pid}"))
                for pid in range(2, 11)]
        self.assertTrue(all(f.result(3)["ok"] for f in rest))
        result = board.result(3)
        self.assertTrue(result["ok"])
        self.assertEqual([p["marker"] for p in result["results_board"]["pieces"]],
                         [f"A:{pid}" for pid in range(1, 11)])
        self.assertEqual(len(self.pipeline._boards), 0)

    def test_2d_first_and_gpu_has_one_owner(self):
        requests = self.pieces()
        self.assertTrue(all(f.result(3)["ok"] for f in requests))
        self.assertTrue(self.pipeline.submit_3d(dict(board_id="A")).result(3)["ok"])
        self.assertEqual(self.detector.max_gpu_active, 1)
        self.assertGreater(self.detector.max_cpu_active, 1)
        self.assertEqual(self.detector.closed_inputs, 10)

    def test_depth_preprocessing_runs_while_gpu_is_busy(self):
        self.detector.gpu_gate.clear()
        requests = self.pieces()
        self.assertTrue(self.detector.gpu_started.wait(1))
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(self.detector.depth_started.wait(1))
        self.assertFalse(board.done())
        self.detector.gpu_gate.set()
        self.assertTrue(all(f.result(3)["ok"] for f in requests))
        self.assertTrue(board.result(3)["ok"])

    def test_heavy_postprocessing_is_bounded_without_blocking_gpu_or_depth(self):
        gate = threading.Event()
        occupied = threading.Event()
        all_inferred = threading.Event()
        counters_lock = threading.Lock()
        active = peak = inferred = 0
        original_finish = self.detector.finish_2d_piece
        original_infer = self.detector.infer_2d_piece

        def finish(prepared, **kwargs):
            nonlocal active, peak
            with counters_lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    occupied.set()
            try:
                if not gate.wait(3):
                    raise TimeoutError("test postprocess gate not opened")
                return original_finish(prepared, **kwargs)
            finally:
                with counters_lock:
                    active -= 1

        def infer(prepared):
            nonlocal inferred
            original_infer(prepared)
            with counters_lock:
                inferred += 1
                if inferred == 10:
                    all_inferred.set()

        with patch.object(self.detector, "finish_2d_piece", side_effect=finish), \
                patch.object(self.detector, "infer_2d_piece", side_effect=infer):
            try:
                requests = self.pieces()
                self.assertTrue(occupied.wait(1))
                # GPU work and depth preparation continue while both heavy
                # workers are busy; excess work stays in the executor queue.
                self.assertTrue(all_inferred.wait(1))
                board = self.pipeline.submit_3d(dict(board_id="A"))
                self.assertTrue(self.detector.depth_started.wait(1))
                self.assertFalse(board.done())
            finally:
                gate.set()
            self.assertTrue(all(f.result(3)["ok"] for f in requests))
            self.assertTrue(board.result(3)["ok"])
        self.assertEqual(peak, 2)
        self.assertEqual(self.detector.max_gpu_active, 1)
        self.assertEqual(self.detector.closed_inputs, 10)

    def test_postprocessing_failure_releases_input_and_allows_retry(self):
        with patch.object(self.detector, "finish_2d_piece", side_effect=ValueError("postprocess failed")):
            response = self.pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(2)
        self.assertFalse(response["ok"])
        self.assertEqual(self.detector.closed_inputs, 1)
        response = self.pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(2)
        self.assertTrue(response["ok"])
        self.assertEqual(self.detector.closed_inputs, 2)

    def test_cross_board_results_never_mix(self):
        boards = [self.pipeline.submit_3d(dict(board_id=name)) for name in ("A", "B")]
        requests = self.pieces("A") + self.pieces("B")
        self.assertTrue(all(f.result(3)["ok"] for f in requests))
        for name, future in zip(("A", "B"), boards):
            self.assertTrue(all(p["board_id"] == name for p in future.result(3)["results_board"]["pieces"]))

    def test_duplicate_and_invalid_piece_are_rejected(self):
        self.detector.gpu_gate.clear()
        first = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        self.assertFalse(self.pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(1)["ok"])
        self.assertFalse(self.pipeline.submit_2d(dict(board_id="A", piece_id=11)).result(1)["ok"])
        self.detector.gpu_gate.set()
        self.assertTrue(first.result(2)["ok"])

    def test_timeout_reports_missing_and_releases(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        # Expiry intentionally ignores active decoding; wait until it is idle.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with self.pipeline._lock:
                if not self.pipeline._boards["A"].depth_inflight:
                    break
            time.sleep(0.001)
        self.assertFalse(self.pipeline._boards["A"].depth_inflight)
        with self.pipeline._lock:
            self.pipeline._boards["A"].touched -= 2
        self.pipeline.expire_pending()
        result = board.result(1)
        self.assertEqual(result["error"], "board_timeout")
        self.assertEqual(result["missing_piece_ids"], list(range(1, 11)))
        self.assertFalse(self.pipeline._boards)

    def test_release_completes_pending_3d(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertTrue(self.pipeline.release_board("A")["ok"])
        self.assertEqual(board.result(1)["error"], "board_released")

    def test_failed_2d_can_be_retried(self):
        failed = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, fail_2d=True)).result(2)
        self.assertFalse(failed["ok"])
        self.assertTrue(self.pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(2)["ok"])

    def test_gpu_failure_releases_input(self):
        self.assertFalse(self.pipeline.submit_2d(dict(board_id="A", piece_id=1, fail_gpu=True)).result(2)["ok"])
        self.assertEqual(self.detector.closed_inputs, 1)

    def test_detailed_queue_and_phase_timings_are_returned(self):
        self.detector.gpu_gate.clear()
        first = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, request_id="first"))
        self.assertTrue(self.detector.gpu_started.wait(1))
        second = self.pipeline.submit_2d(dict(board_id="A", piece_id=2, request_id="second"))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and len(self.detector.prepared_ids) < 2:
            time.sleep(0.001)
        time.sleep(0.025)
        self.detector.gpu_gate.set()
        self.assertTrue(first.result(2)["ok"])
        response = second.result(2)
        self.assertGreater(response["timings"]["gpu_queue_ms"], 10)
        for key in ("cpu_prepare_queue_ms", "cpu_prepare_phase_ms", "gpu_phase_ms",
                    "cpu_postprocess_queue_ms", "cpu_postprocess_phase_ms", "request_pipeline_ms"):
            self.assertGreaterEqual(response["timings"][key], 0)

    def test_3d_failure_and_joint_failure_finish_board(self):
        failed = self.pipeline.submit_3d(dict(board_id="A", fail_3d=True)).result(2)
        self.assertEqual(failed["error"], "3d_board_failed")
        board = self.pipeline.submit_3d(dict(board_id="B"))
        self.pipeline.submit_2d(dict(board_id="B", piece_id=1, fail_joint=True)).result(2)
        self.assertEqual(board.result(2)["error"], "joint_detection_failed")


class ThreadBudgetTests(unittest.TestCase):
    def test_native_defaults_respect_explicit_library_settings(self):
        with patch.dict(os.environ, {"BRIDGE_NATIVE_THREADS": "2", "OMP_NUM_THREADS": "3"}, clear=True):
            configure_native_threads()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "2")

    def test_default_native_threads_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_native_threads()
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
        for value in ("0", "-1", "not-an-int"):
            with self.subTest(value=value), patch.dict(os.environ, {"BRIDGE_NATIVE_THREADS": value}):
                with self.assertRaises(ValueError):
                    configure_native_threads()

    def test_thread_budget_and_minimum_shared_pool(self):
        for workers in (2, 3, 13):
            with self.subTest(workers=workers), patch.dict(os.environ, {
                "BRIDGE_MAX_WORKERS": str(workers),
            }, clear=True):
                server = AlgorithmServer()
                try:
                    info = server.handle_init({})["threading"]
                    total = info["gpu_workers"] + info["cpu_workers"]
                    if not info["postprocess_pool_shared"]:
                        total += info["postprocess_workers"]
                    self.assertEqual(total, workers)
                    self.assertEqual(info["gpu_workers"], 1)
                    self.assertIs(server.postprocess_pool is server.cpu_pool, workers == 2)
                finally:
                    server.close()

    def test_small_budgets_complete_the_entire_pipeline(self):
        for workers in (2, 3):
            with self.subTest(workers=workers), patch.dict(os.environ, {
                "BRIDGE_MAX_WORKERS": str(workers),
            }, clear=True):
                server = AlgorithmServer()
                server.board = FakeDetector()
                try:
                    pipeline = server.get_pipeline()
                    board = pipeline.submit_3d(dict(board_id="small"))
                    requests = [pipeline.submit_2d(dict(board_id="small", piece_id=pid))
                                for pid in range(1, 11)]
                    self.assertTrue(all(f.result(3)["ok"] for f in requests))
                    self.assertTrue(board.result(3)["ok"])
                    self.assertEqual(server.board.closed_inputs, 10)
                finally:
                    server.close()

    def test_configurable_postprocessing_count_and_invalid_budget(self):
        with patch.dict(os.environ, {"BRIDGE_MAX_WORKERS": "13", "BRIDGE_POSTPROCESS_WORKERS": "4"}):
            server = AlgorithmServer()
            try:
                self.assertEqual(server.threading_info["cpu_workers"], 8)
                self.assertEqual(server.threading_info["postprocess_workers"], 4)
                server.board = FakeDetector()
                self.assertIs(server.get_pipeline().postprocess_pool, server.postprocess_pool)
            finally:
                server.close()
        for value in ("0", "-1", "12", "bad"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "BRIDGE_MAX_WORKERS": "13", "BRIDGE_POSTPROCESS_WORKERS": value,
            }):
                with self.assertRaises(ValueError):
                    AlgorithmServer()


class ProtocolTests(unittest.TestCase):
    def test_out_of_order_replies_keep_request_ids(self):
        server = AlgorithmServer()
        server.board = FakeDetector()
        stream = io.StringIO()
        try:
            server._accept(dict(cmd="detect_3d_board", board_id="A", request_id="depth"), stream)
            for pid in range(1, 11):
                server._accept(dict(cmd="detect_2d_piece", board_id="A", piece_id=pid,
                                    request_id=f"image-{pid}", marker=f"marker-{pid}"), stream)
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                with server._pending_lock:
                    if not server._pending:
                        break
                time.sleep(0.01)
            lines = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(len(lines), 11)
            self.assertNotEqual(lines[0]["request_id"], "depth")
            self.assertEqual({line["request_id"] for line in lines},
                             {"depth"} | {f"image-{pid}" for pid in range(1, 11)})
            for line in lines:
                if line["request_id"] != "depth":
                    self.assertEqual(line["marker"], f"marker-{line['piece_id']}")
        finally:
            server.close()


@unittest.skipUnless(os.name == "nt", "Windows named mappings are required")
class SharedMemoryTests(unittest.TestCase):
    def test_readonly_shallow_view_with_stride_and_offset(self):
        import uuid
        name = "bridge-test-" + uuid.uuid4().hex
        with mmap.mmap(-1, 64, tagname=name) as producer:
            producer[8:14] = bytes([1, 2, 3, 4, 5, 6])
            producer[16:22] = bytes([7, 8, 9, 10, 11, 12])
            owner = SharedImage(dict(shm_name=name, image_size=16, image_offset=8,
                                     image_width=2, image_height=2, image_channels=3, image_stride=8))
            self.assertEqual(owner.image[1, 1].tolist(), [10, 11, 12])
            producer[8] = 42
            self.assertEqual(int(owner.image[0, 0, 0]), 42)  # No pixel copy occurred.
            self.assertFalse(owner.image.flags.writeable)
            owner.close()

    def test_invalid_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            SharedImage(dict(shm_name="unused", image_size=2, image_width=2,
                             image_height=2, image_channels=3))


if __name__ == "__main__":
    unittest.main()
