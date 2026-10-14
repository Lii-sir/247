import contextlib
import io
import json
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
from algorithm.timing import current_timings, measure, note, timing_context, timed_call


class TimingTests(unittest.TestCase):
    def records(self, lines):
        return [json.loads(line.removeprefix("[TIMING] ")) for line in lines]

    def test_logs_are_correlated_and_never_written_to_stdout(self):
        lines, timings = [], {}
        stdout = io.StringIO()
        with patch.dict(os.environ, {"BRIDGE_TIMING_LOG": "1"}), contextlib.redirect_stdout(stdout):
            with timing_context(timings, phase="gpu", board_id="A", piece_id=3,
                                request_id="r3", log=lines.append):
                self.assertEqual(timed_call("predict_ms", lambda: 42), 42)
        self.assertEqual(stdout.getvalue(), "")
        self.assertGreaterEqual(timings["predict_ms"], 0)
        records = self.records(lines)
        self.assertEqual([r["event"] for r in records], ["begin", "end"])
        self.assertTrue(all((r["board_id"], r["piece_id"], r["request_id"]) == ("A", 3, "r3") for r in records))
        self.assertEqual(current_timings(), {})

    def test_failure_keeps_elapsed_and_restores_context(self):
        timings, lines = {}, []
        with patch.dict(os.environ, {"BRIDGE_TIMING_LOG": "1"}):
            with self.assertRaises(ValueError):
                with timing_context(timings, phase="cpu", log=lines.append):
                    with measure("failed_ms"):
                        raise ValueError("bad input")
        self.assertIn("failed_ms", timings)
        self.assertEqual(self.records(lines)[-1]["error"], "ValueError")
        self.assertEqual(current_timings(), {})

    def test_contexts_are_isolated_between_workers(self):
        gate = threading.Barrier(2)
        def worker(board):
            lines, values = [], {}
            with timing_context(values, phase="cpu", board_id=board, log=lines.append):
                gate.wait(timeout=2)
                timed_call("work_ms", time.sleep, 0.002)
                note("done")
            return board, values, lines
        with patch.dict(os.environ, {"BRIDGE_TIMING_LOG": "1"}), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, ["A", "B"]))
        for board, values, lines in results:
            self.assertIn("work_ms", values)
            self.assertTrue(all(record["board_id"] == board for record in self.records(lines)))

    def test_logging_can_be_disabled_without_losing_json_timings(self):
        values, lines = {}, []
        with patch.dict(os.environ, {"BRIDGE_TIMING_LOG": "0"}):
            with timing_context(values, phase="cpu", log=lines.append):
                timed_call("work_ms", lambda: None)
        self.assertEqual(lines, [])
        self.assertIn("work_ms", values)

    def test_bad_log_sink_cannot_change_result(self):
        def bad_sink(message):
            raise OSError("log file unavailable")
        with timing_context({}, phase="cpu", log=bad_sink):
            self.assertEqual(timed_call("work_ms", lambda: 5), 5)

    def test_gpu_phase_splits_sync_predict_and_mask_extraction(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from algorithm.detect_board import DetectBoard
        detector = DetectBoard(lambda message: None)
        detector.model = SimpleNamespace(device=torch.device("cpu"))
        detector.chip_anomaly_inspector = SimpleNamespace(inspect=lambda *args, **kw: {"status": "disabled"})
        prepared = dict(crop=np.zeros((10, 10, 3), np.uint8), image=np.zeros((10, 10, 3), np.uint8), crop_box=(0, 0, 10, 10))
        timings = {}
        with patch.object(detector, "_synchronize_model"), \
                patch.object(detector, "_predict_batch", return_value=[SimpleNamespace(
                    speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})]), \
                patch.object(detector, "_extract_instances", return_value=[]), \
                timing_context(timings, phase="gpu"):
            detector.infer_2d_piece(prepared)
        for key in ("cuda_before_yolo_ms", "cuda_after_yolo_ms", "yolo_predict_wall_ms",
                    "mask_extract_to_cpu_ms", "efficientad_total_ms", "model_lock_wait_ms"):
            self.assertIn(key, timings)
        self.assertEqual(timings["yolo_reported_inference_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
