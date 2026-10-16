"""Independent stage-pool configuration and deterministic scheduling regressions."""
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
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector
import test_early_joint_pipeline as early


def config(budget=15, **overrides):
    two = dict(gpu_workers=3, postprocess_workers=2, silver_workers=4)
    three = dict(worker_budget=budget, cpu_workers=2, joint_workers=4)
    for key, value in overrides.items():
        (two if key in two else three)[key] = value
    return {"common_2d": {"parameters": {"threading": two}},
            "common_3d": {"parameters": {"threading": three}}}


class SplitPoolSettingsTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.server = AlgorithmServer()
        self.addCleanup(self.server.close)
        self.server.board = FakeDetector()

    def test_explicit_config_builds_and_injects_five_distinct_pools(self):
        pipeline = self.server.get_pipeline()
        settings = self.server._thread_settings(config(early_joint_workers=3))
        self.server._configure_thread_pools(settings, source="test")
        pools = []
        for stage in ("cpu", "gpu", "postprocess", "silver", "joint"):
            pool = getattr(self.server, stage + "_pool")
            pools.append(pool)
            self.assertIs(getattr(pipeline, stage + "_pool"), pool)
            self.assertEqual(pool._max_workers, settings[stage + "_workers"])
            self.assertTrue(pool.submit(lambda: threading.current_thread().name).result(2)
                            .startswith("bridge-" + stage))
        self.assertEqual(len({id(pool) for pool in pools}), 5)
        self.assertEqual(pipeline.early_joint_workers, 3)

    def test_missing_keys_preserve_old_sharing(self):
        cfg = config(budget=14, gpu_workers=2, postprocess_workers=5, cpu_workers=7)
        del cfg["common_2d"]["parameters"]["threading"]["silver_workers"]
        del cfg["common_3d"]["parameters"]["threading"]["joint_workers"]
        settings = self.server._thread_settings(cfg)
        self.server._configure_thread_pools(settings, source="test")
        self.assertIs(self.server.silver_pool, self.server.postprocess_pool)
        self.assertIs(self.server.joint_pool, self.server.cpu_pool)
        self.assertEqual(settings["early_joint_workers"], 7)

    def test_independent_pools_are_included_in_budget(self):
        with self.assertRaisesRegex(ValueError, "effective_total=15"):
            self.server._thread_settings(config(budget=14))

    def test_invalid_new_counts_are_rejected(self):
        for key in ("silver_workers", "joint_workers", "early_joint_workers"):
            for value in (0, -1, True, 1.5, "bad", None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self.server._thread_settings(config(**{key: value}))

    def test_early_limit_is_bounded_by_joint_not_preparation_pool(self):
        settings = self.server._thread_settings(config(early_joint_workers=3))
        self.assertGreater(settings["early_joint_workers"], settings["cpu_workers"])
        with self.assertRaisesRegex(ValueError, "cannot exceed joint_workers"):
            self.server._thread_settings(config(early_joint_workers=5))

    def test_environment_fallback_reserves_new_pools(self):
        with patch.dict(os.environ, {"BRIDGE_MAX_WORKERS": "13",
                                    "BRIDGE_SILVER_WORKERS": "2",
                                    "BRIDGE_JOINT_WORKERS": "3"}):
            settings = self.server._thread_settings({})
        self.assertEqual(settings["silver_workers"], 2)
        self.assertEqual(settings["joint_workers"], 3)
        self.assertEqual(settings["cpu_workers"], 2)
        self.assertEqual(sum(settings[k + "_workers"] for k in
                             ("cpu", "gpu", "postprocess", "silver", "joint")), 13)

    def test_small_budget_reserves_independent_pools(self):
        with patch.dict(os.environ, {"BRIDGE_MAX_WORKERS": "5",
                                    "BRIDGE_SILVER_WORKERS": "1",
                                    "BRIDGE_JOINT_WORKERS": "1"}):
            settings = self.server._thread_settings({})
        self.assertTrue(all(settings[k + "_workers"] == 1 for k in
                            ("cpu", "gpu", "postprocess", "silver", "joint")))
        self.assertFalse(settings["postprocess_pool_shared"])

    def test_json_takes_precedence_over_environment(self):
        with patch.dict(os.environ, {"BRIDGE_SILVER_WORKERS": "99",
                                    "BRIDGE_JOINT_WORKERS": "99"}):
            settings = self.server._thread_settings(config())
        self.assertEqual(settings["silver_workers"], 4)
        self.assertEqual(settings["joint_workers"], 4)

    def test_checked_in_config_enables_split_pools(self):
        from algorithm.config import load_runtime_config
        manifest = ROOT / 'config' / 'config.json'
        current = load_runtime_config(manifest.read_text(encoding="utf-8-sig"), str(manifest))
        settings = self.server._thread_settings(current)
        self.assertFalse(settings["silver_pool_shared"])
        self.assertFalse(settings["joint_pool_shared"])
        self.assertEqual(settings["silver_workers"],
                         current["common_2d"]["parameters"]["threading"]["silver_workers"])
        self.assertEqual(settings["joint_workers"],
                         current["common_3d"]["parameters"]["threading"]["joint_workers"])

    def test_early_limit_only_update_preserves_executors(self):
        pipeline = self.server.get_pipeline()
        self.server._configure_thread_pools(self.server._thread_settings(config()), source="test")
        original = self.server.joint_pool
        self.server._configure_thread_pools(
            self.server._thread_settings(config(early_joint_workers=1)), source="test")
        self.assertIs(self.server.joint_pool, original)
        self.assertEqual(pipeline.early_joint_workers, 1)

    def test_aliases_are_shutdown_only_once(self):
        pools = {self.server.cpu_pool, self.server.gpu_pool, self.server.postprocess_pool}
        for pool in pools:
            pool.shutdown = Mock(wraps=pool.shutdown)
        self.server.close()
        for pool in pools:
            pool.shutdown.assert_called_once_with(wait=True)

    def test_reconfigure_and_close_shutdown_unique_pools_once(self):
        old = set((self.server.cpu_pool, self.server.gpu_pool, self.server.postprocess_pool))
        for pool in old:
            pool.shutdown = Mock(wraps=pool.shutdown)
        self.server._configure_thread_pools(self.server._thread_settings(config()), source="test")
        for pool in old:
            pool.shutdown.assert_called_once_with(wait=True)
        new = {getattr(self.server, stage + "_pool") for stage in
               ("cpu", "gpu", "postprocess", "silver", "joint")}
        for pool in new:
            pool.shutdown = Mock(wraps=pool.shutdown)
        self.server.close()
        self.server.close()
        for pool in new:
            pool.shutdown.assert_called_once_with(wait=True)

    def test_partial_pool_creation_failure_preserves_old_pools(self):
        old = self.server.cpu_pool
        created = []
        def create(**kwargs):
            if len(created) == 3:
                raise RuntimeError("synthetic executor creation failure")
            pool = ThreadPoolExecutor(**kwargs)
            created.append(pool)
            return pool
        with patch("transport.bridge_server.ThreadPoolExecutor", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                self.server._configure_thread_pools(self.server._thread_settings(config()), source="test")
        self.assertIs(self.server.cpu_pool, old)
        self.assertEqual(old.submit(lambda: 1).result(2), 1)
        self.assertTrue(all(pool._shutdown for pool in created))

    def test_busy_pipeline_rejects_reconfiguration(self):
        pipeline = self.server.get_pipeline()
        self.assertTrue(pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(2)["ok"])
        with self.assertRaisesRegex(RuntimeError, "idle"):
            self.server._configure_thread_pools(self.server._thread_settings(config()), source="test")
        pipeline.release_board("A")


class SplitPoolSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector(count=2)
        self.cpu = ThreadPoolExecutor(1, thread_name_prefix="test-cpu")
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="test-gpu")
        self.post = ThreadPoolExecutor(1, thread_name_prefix="test-post")
        self.silver = ThreadPoolExecutor(2, thread_name_prefix="test-silver")
        self.joint = ThreadPoolExecutor(1, thread_name_prefix="test-joint")
        self.pipeline = BoardPipeline(self.detector, self.cpu, self.gpu,
            postprocess_pool=self.post, silver_pool=self.silver, joint_pool=self.joint)
        self.gate = threading.Event()

    def tearDown(self):
        self.gate.set()
        self.pipeline.close()
        for pool in (self.gpu, self.silver, self.post, self.joint, self.cpu):
            pool.shutdown(wait=True)

    def test_blocked_silver_does_not_occupy_postprocess(self):
        started = threading.Event()
        def boundary(prepared):
            if prepared["piece_id"] == 1:
                started.set()
                if not self.gate.wait(5):
                    raise TimeoutError("silver gate")
        self.detector.prepare_silver_boundary = boundary
        first = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        self.assertTrue(started.wait(2))
        second = self.pipeline.submit_2d(dict(board_id="A", piece_id=2))
        try:
            self.assertTrue(second.result(2)["ok"])
            self.assertFalse(first.done())
            self.assertEqual(self.detector.closed_inputs, 1)
        finally:
            self.gate.set()
        self.assertTrue(first.result(2)["ok"])
        self.assertEqual(self.detector.closed_inputs, 2)

    def test_blocked_joint_leaves_preparation_available(self):
        started = threading.Event()
        measure = self.detector.measure_prepared_piece
        def blocked(*args):
            started.set()
            if not self.gate.wait(5):
                raise TimeoutError("joint gate")
            return measure(*args)
        self.detector.measure_prepared_piece = blocked
        board = self.pipeline.submit_3d(dict(board_id="A"))
        first = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="A:1"))
        try:
            self.assertTrue(started.wait(2))
            second = self.pipeline.submit_2d(dict(board_id="A", piece_id=2, marker="A:2"))
            self.assertTrue(second.result(2)["ok"])
            self.assertTrue(first.result(2)["ok"])
            self.assertFalse(board.done())
        finally:
            self.gate.set()
        result = board.result(3)
        self.assertTrue(result["ok"])
        self.assertEqual([p["marker"] for p in result["results_board"]["pieces"]], ["A:1", "A:2"])
        self.assertEqual(self.detector.closed_inputs, 2)


class SplitPoolEarlyJointTests(early.EarlyJointTests):
    """Repeat lifetime/failure/final-NG join tests using truly independent pools."""
    def setUp(self):
        super().setUp()
        self.silver = ThreadPoolExecutor(2)
        self.joint = ThreadPoolExecutor(2)
        self.pipeline.silver_pool = self.silver
        self.pipeline.joint_pool = self.joint

    def tearDown(self):
        self.detector.diagnostics_gate.set()
        self.silver.shutdown(wait=True)
        self.joint.shutdown(wait=True)
        super().tearDown()


if __name__ == "__main__":
    unittest.main()

