"""Prove repeated batches share one initialization and keep board/timing data isolated."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

VIEWER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIEWER))
spec = importlib.util.spec_from_file_location("repeat_detection_client", VIEWER / "detect_and_view.py")
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)


class FakeClient:
    def __init__(self, fail_piece=False, fail_init=False):
        self.process = SimpleNamespace(pid=12345)
        self.timeline = []
        self.requests = []
        self.started = time.perf_counter()
        self.condition = threading.Condition()
        self.pieces = {}
        self.fail_piece = fail_piece
        self.fail_init = fail_init
        self.close_count = 0

    def request(self, request, timeout):
        sent = (time.perf_counter() - self.started) * 1000
        command = request["cmd"]
        board = request.get("board_id") or request.get("batch_id")
        with self.condition:
            self.requests.append(dict(request))
            response = {"ok": True, "cmd": command, "board_id": board, "elapsed_ms": 1}
            if command == "init_algorithm" and self.fail_init:
                response = {"ok": False, "error": "model unavailable"}
            if command == "detect_2d_piece":
                pid = request["piece_id"]
                self.pieces.setdefault(board, set()).add(pid)
                self.condition.notify_all()
                response.update(piece_id=pid, results_2d={"is_ok": False})
                if self.fail_piece and pid == 3:
                    response.update(ok=False, error="piece failed")
            elif command == "detect_3d_board":
                if not self.condition.wait_for(lambda: len(self.pieces.get(board, set())) == 10, timeout=3):
                    raise RuntimeError("missing piece in test")
                response["results_board"] = {"is_ok": False, "pieces": []}
            elif command == "detect_board":
                response["results_board"] = {"is_ok": False, "pieces": []}
            received = (time.perf_counter() - self.started) * 1000
            self.timeline.append({"cmd": command, "stage": request.get("stage"),
                "piece_id": request.get("piece_id"), "sent_ms": sent,
                "received_ms": received, "duration_ms": received - sent,
                "ok": response["ok"]})
            return response

    def close(self):
        self.close_count += 1
        self.request({"cmd": "release"}, 10)


class RepeatDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.depth = self.root / "depth.png"
        self.depth.write_bytes(b"fake image")
        self.paths = [self.root / f"{i:03d}.png" for i in range(1, 11)]
        self.mappings = []
        self.closed_mappings = []

    def tearDown(self):
        self.temp.cleanup()

    def run_client(self, repeat=2, mode="stream", parallel=True, **failure):
        self.fake = FakeClient(**failure)
        output = self.root / "output"

        def mapping(stack, path, kind, piece_id=None):
            self.mappings.append((path, kind, piece_id))
            stack.callback(self.closed_mappings.append, path)
            if kind == "2d":
                return {"piece_id": piece_id, "shm_name": str(path), "image_size": 1}
            return {"shm_name": str(path), "data_size": 1}

        def render_report(document, path, **kwargs):
            self.assertEqual(self.fake.close_count, 1, "Render only after all inference and model release")
            return path

        argv = ["detect_and_view.py", "--repeat", str(repeat), "--mode", mode,
                "--depth", str(self.depth), "--output-dir", str(output), "--board-id", "test"]
        if parallel:
            argv.append("--parallel")
        with patch.object(sys, "argv", argv), \
                patch.object(client_module, "image_files", return_value=self.paths), \
                patch.object(client_module, "algorithm_request", return_value={"cmd": "init_algorithm", "stage": "board"}), \
                patch.object(client_module, "ProtocolClient", return_value=self.fake) as factory, \
                patch.object(client_module, "mapped_file", side_effect=mapping), \
                patch.object(client_module, "render_report", side_effect=render_report) as render, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = client_module.main()
        self.factory_count = factory.call_count
        self.render_count = render.call_count
        self.output = output
        return code

    def test_two_parallel_batches_init_once_same_pid_different_boards(self):
        # Product NG must not prevent running the next batch.
        self.assertEqual(self.run_client(), 0)
        commands = [request["cmd"] for request in self.fake.requests]
        self.assertEqual(self.factory_count, 1)
        self.assertEqual(commands.count("init"), 1)
        self.assertEqual(commands.count("init_algorithm"), 1)
        self.assertEqual(commands.count("detect_2d_piece"), 20)
        self.assertEqual(commands.count("detect_3d_board"), 2)
        self.assertEqual(commands[-1], "release")
        self.assertEqual(self.fake.close_count, 1)
        self.assertEqual(self.render_count, 2)
        first = json.loads((self.output / "batch-01/result.json").read_text(encoding="utf-8"))
        second = json.loads((self.output / "batch-02/result.json").read_text(encoding="utf-8"))
        self.assertEqual(first["board_id"], "test-run-01")
        self.assertEqual(second["board_id"], "test-run-02")
        self.assertEqual(first["images_2d"], second["images_2d"])
        self.assertEqual(first["image_3d"], second["image_3d"])
        self.assertEqual(len(first["timeline"]), 11)
        self.assertEqual(len(second["timeline"]), 11)
        self.assertLessEqual(max(x["received_ms"] for x in first["timeline"]),
                             min(x["sent_ms"] for x in second["timeline"]))
        self.assertGreater(first["benchmark"]["startup_charged_ms"], 0)
        self.assertEqual(second["benchmark"]["startup_charged_ms"], 0)
        self.assertFalse(first["benchmark"]["model_reused"])
        self.assertTrue(second["benchmark"]["model_reused"])
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["success"])
        self.assertEqual(summary["startup"]["initialization_count"], 1)
        self.assertEqual({x["process_pid"] for x in summary["batches"]}, {12345})
        self.assertEqual(len(self.mappings), 22)
        self.assertEqual(len(self.closed_mappings), 22)

    def test_one_batch_keeps_existing_output_paths(self):
        self.assertEqual(self.run_client(repeat=1), 0)
        self.assertTrue((self.output / "result.json").is_file())
        self.assertTrue((self.output / "timings.csv").is_file())
        self.assertFalse((self.output / "batch-01").exists())
        result = json.loads((self.output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["board_id"], "test")

    def test_sequential_stream_reuses_models_too(self):
        self.assertEqual(self.run_client(parallel=False), 0)
        self.assertEqual(sum(x["cmd"] == "init_algorithm" for x in self.fake.requests), 1)
        self.assertEqual(len(self.mappings), 22)

    def test_board_mode_reuses_models(self):
        self.assertEqual(self.run_client(mode="board", parallel=False), 0)
        self.assertEqual(sum(x["cmd"] == "detect_board" for x in self.fake.requests), 2)
        self.assertEqual(sum(x["cmd"] == "init_algorithm" for x in self.fake.requests), 1)

    def test_request_failure_stops_repeats_and_preserves_first_report(self):
        self.assertEqual(self.run_client(fail_piece=True), 1)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["success"])
        self.assertEqual(summary["completed_batches"], 1)
        self.assertFalse((self.output / "batch-02").exists())
        self.assertEqual(sum(x["cmd"] == "release_board" for x in self.fake.requests), 1)
        self.assertEqual(self.fake.close_count, 1)

    def test_init_failure_does_not_submit_detection(self):
        self.assertEqual(self.run_client(fail_init=True), 1)
        self.assertFalse(any(x["cmd"].startswith("detect") for x in self.fake.requests))
        self.assertTrue((self.output / "result.json").is_file())
        self.assertEqual(self.fake.close_count, 1)

    def test_repeat_must_be_positive(self):
        with self.assertRaises(SystemExit) as exc:
            self.run_client(repeat=0)
        self.assertEqual(exc.exception.code, 2)

    def test_csv_exports_2d_board_and_joint_metrics_without_adding_them(self):
        envelope = {"board_id": "A", "responses_2d": [{"piece_id": 1,
            "timings": {"gpu_queue_ms": 123.4, "flag": True}}],
            "response": {"timings": {"depth_segmentation_ms": 50}, "results_board": {
                "pieces": [{"piece_id": 1, "timings": {"affine_registration_ms": 600}}]}}}
        path = self.root / "timings.csv"
        client_module.save_timing_csv(envelope, path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["phase"] for row in rows}, {"2d", "board", "joint_3d"})
        self.assertEqual(float(rows[0]["elapsed_ms"]), 123.4)


if __name__ == "__main__":
    unittest.main()
