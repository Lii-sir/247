"""Event-gated tests for overlapping 3D computation with unfinished 2D diagnostics."""
import sys
from pathlib import Path
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from algorithm.application.board.board_pipeline import BoardPipeline
from pipeline_fakes import FakeDetector


class GatedDetector(FakeDetector):
    def __init__(self):
        super().__init__(count=1)
        self.diagnostics_started = threading.Event()
        self.diagnostics_gate = threading.Event()

    def finish_2d_piece(self, prepared, *, publish_cache, on_joint_ready=None):
        if on_joint_ready is not None:
            on_joint_ready(dict(prepared))
        self.diagnostics_started.set()
        if not self.diagnostics_gate.wait(3):
            raise TimeoutError("test diagnostics gate not opened")
        if prepared.get("fail_diagnostics"):
            raise ValueError("diagnostics failed after joint publication")
        response, entry = super().finish_2d_piece(prepared, publish_cache=publish_cache)
        entry["diagnostics_ok"] = not prepared.get("diagnostics_ng", False)
        return response, entry

    def finalize_joint_piece(self, piece, entry):
        return {**piece, "is_ok": piece["is_ok"] and entry.get("diagnostics_ok", True),
                "diagnostics_ok": entry.get("diagnostics_ok", True)}


class EarlyJointTests(unittest.TestCase):
    def setUp(self):
        self.detector = GatedDetector()
        self.cpu = ThreadPoolExecutor(max_workers=2)
        self.gpu = ThreadPoolExecutor(max_workers=1)
        self.post = ThreadPoolExecutor(max_workers=1)
        self.pipeline = BoardPipeline(self.detector, self.cpu, self.gpu,
                                      postprocess_pool=self.post)

    def tearDown(self):
        self.detector.diagnostics_gate.set()
        self.pipeline.close()
        self.gpu.shutdown(wait=True)
        self.post.shutdown(wait=True)
        self.cpu.shutdown(wait=True)

    def test_3d_runs_before_diagnostics_but_board_waits_for_final_ng(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, diagnostics_ng=True))
        self.assertTrue(self.detector.diagnostics_started.wait(1))
        try:
            self.assertTrue(self.detector.measured.wait(1), "3D waited for heavy 2D diagnostics")
            self.assertFalse(piece.done())
            self.assertFalse(board.done(), "board returned without final 2D diagnostics")
        finally:
            self.detector.diagnostics_gate.set()
        self.assertTrue(piece.result(2)["ok"])
        result = board.result(2)
        self.assertTrue(result["ok"])
        self.assertFalse(result["results_board"]["is_ok"])
        self.assertFalse(result["results_board"]["pieces"][0]["diagnostics_ok"])
        self.assertEqual(self.detector.closed_inputs, 1)

    def test_3d_arriving_during_diagnostics_uses_early_input(self):
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        self.assertTrue(self.detector.diagnostics_started.wait(1))
        board = self.pipeline.submit_3d(dict(board_id="A"))
        try:
            self.assertTrue(self.detector.measured.wait(1))
            self.assertFalse(board.done())
        finally:
            self.detector.diagnostics_gate.set()
        self.assertTrue(piece.result(2)["ok"])
        self.assertTrue(board.result(2)["ok"])

    def test_diagnostics_exception_after_measurement_fails_board(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, fail_diagnostics=True))
        try:
            self.assertTrue(self.detector.diagnostics_started.wait(1))
            self.assertTrue(self.detector.measured.wait(1))
        finally:
            self.detector.diagnostics_gate.set()
        self.assertFalse(piece.result(2)["ok"])
        self.assertFalse(board.result(2)["ok"])
        self.assertFalse(self.pipeline.has_pending)
        self.assertEqual(self.detector.closed_inputs, 1)

    def test_release_during_both_branches_does_not_resurrect_board(self):
        board = self.pipeline.submit_3d(dict(board_id="A"))
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        self.assertTrue(self.detector.diagnostics_started.wait(1))
        self.pipeline.release_board("A")
        self.detector.diagnostics_gate.set()
        self.assertEqual(board.result(2)["error"], "board_released")
        self.assertFalse(piece.result(2)["ok"])
        self.assertFalse(self.pipeline.has_pending)

    def test_failure_before_3d_can_retry_without_stale_early_input(self):
        self.detector.diagnostics_gate.set()
        failed = self.pipeline.submit_2d(dict(board_id="A", piece_id=1,
                                            fail_diagnostics=True, marker="old"))
        self.assertFalse(failed.result(2)["ok"])
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="new"))
        self.assertTrue(piece.result(2)["ok"])
        board = self.pipeline.submit_3d(dict(board_id="A"))
        self.assertEqual(board.result(2)["results_board"]["pieces"][0]["marker"], "new")

    def test_completed_piece_replacement_cannot_pair_old_data_with_depth(self):
        self.detector.diagnostics_gate.set()
        old = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="old"))
        self.assertTrue(old.result(2)["ok"])
        self.detector.gpu_gate.clear()
        self.detector.gpu_started.clear()
        replacement = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="new"))
        self.assertTrue(self.detector.gpu_started.wait(1))
        board = self.pipeline.submit_3d(dict(board_id="A"))
        try:
            self.assertTrue(self.detector.depth_started.wait(1))
            self.assertFalse(self.detector.measured.wait(0.05))
        finally:
            self.detector.gpu_gate.set()
        self.assertTrue(replacement.result(2)["ok"])
        self.assertEqual(board.result(2)["results_board"]["pieces"][0]["marker"], "new")

    def test_boundary_work_overlaps_gpu_and_keeps_source_until_both_finish(self):
        boundary_started = threading.Event()
        boundary_gate = threading.Event()

        def boundary(prepared):
            boundary_started.set()
            if not boundary_gate.wait(3):
                raise TimeoutError("boundary gate not opened")
            return None

        self.detector.prepare_silver_boundary = boundary
        self.detector.diagnostics_gate.set()
        board = self.pipeline.submit_3d(dict(board_id="A"))
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        try:
            self.assertTrue(boundary_started.wait(1))
            self.assertTrue(self.detector.gpu_started.wait(1), "GPU waited for boundary preparation")
            self.assertTrue(self.detector.measured.wait(1), "3D waited for boundary preparation")
            self.assertEqual(self.detector.closed_inputs, 0)
            self.assertFalse(piece.done())
            self.assertFalse(board.done())
        finally:
            boundary_gate.set()
        self.assertTrue(piece.result(2)["ok"])
        self.assertTrue(board.result(2)["ok"])
        self.assertEqual(self.detector.closed_inputs, 1)

    def test_gpu_failure_waits_for_boundary_reader_before_releasing_source(self):
        boundary_started = threading.Event()
        boundary_gate = threading.Event()

        def boundary(prepared):
            boundary_started.set()
            if not boundary_gate.wait(3):
                raise TimeoutError("boundary gate not opened")
            return None

        self.detector.prepare_silver_boundary = boundary
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, fail_gpu=True))
        try:
            self.assertTrue(boundary_started.wait(1))
            self.assertTrue(self.detector.gpu_started.wait(1))
            self.assertFalse(piece.done())
            self.assertEqual(self.detector.closed_inputs, 0)
        finally:
            boundary_gate.set()
        self.assertFalse(piece.result(2)["ok"])
        self.assertEqual(self.detector.closed_inputs, 1)


class DiagnosticJoinTests(unittest.TestCase):
    def test_real_join_applies_late_ng_and_preserves_height_coordinates(self):
        from algorithm.application.board.detect_board import DetectBoard
        piece = {"is_ok": True, "wire_measurements": [{"height_3d": {
            "point": {"x": 5, "y": 1010}, "point_coordinate_space": "board_3d"}}]}
        silver = {"status": "overflow", "is_ok": False}
        result = DetectBoard.finalize_joint_piece(piece, {"silver_overflow_2d": silver})
        self.assertFalse(result["is_ok"])
        self.assertIs(result["silver_overflow_2d"], silver)
        self.assertEqual(result["wire_measurements"][0]["height_3d"]["point"]["y"], 1010)
        self.assertTrue(piece["is_ok"], "joining mutated the in-flight result")
        self.assertFalse(DetectBoard.finalize_joint_piece(
            {"is_ok": False}, {"silver_overflow_2d": {"status": "ok", "is_ok": True}})["is_ok"])

    def test_disabled_diagnostics_do_not_fail_joint_result(self):
        from algorithm.application.board.detect_board import DetectBoard
        result = DetectBoard.finalize_joint_piece({"is_ok": True}, {
            "silver_overflow_2d": {"status": "disabled", "is_ok": None}})
        self.assertTrue(result["is_ok"])


class DetectionPolicyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from algorithm.application.board import detect_board
        from algorithm.application.defect_checks import detection_order

        self.module = detect_board
        self.detector = detector = detect_board.DetectBoard(lambda _: None)
        self.order = detection_order(["chip_rotation_2d", "silver_overflow_2d"])
        image = np.zeros((16, 16, 3), np.uint8)
        detector.expected_piece_count = 1
        detector.segmentation_config = detect_board.SegmentationConfig(
            expected_piece_count=1, mode="fixed", fixed_rois=((0, 0, 16, 16),))
        detector.model = object()
        detector.chip_anomaly_inspector = Mock()
        detector._read_2d = Mock(return_value=image)
        detector._read_3d = Mock(return_value=np.full((16, 16), 33000, np.uint16))
        detector._stream_crop = Mock(return_value=(image, (0, 0, 16, 16), None, []))
        self.prediction = SimpleNamespace(
            masks=SimpleNamespace(data=torch.ones(1, 16, 16)),
            boxes=SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([0.9])),
            names={0: "chip"},
        )
        detector._predict_batch = Mock(return_value=[self.prediction])
        detector.submit_registration = Mock(side_effect=AssertionError("3D registration must be skipped"))
        detector.submit_joint_measurement = Mock(side_effect=AssertionError("3D heights must be skipped"))

    def run_stream(self):
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            pipeline = BoardPipeline(self.detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            board = pipeline.submit_3d({"board_id": "A"})
            piece = pipeline.submit_2d({"board_id": "A", "piece_id": 1}).result(3)
            self.assertTrue(piece["ok"], piece)
            response = board.result(3)
            self.assertTrue(response["ok"], response)
            self.assertFalse(pipeline.has_pending)
            return piece["results_2d"], response["results_board"]

    def assert_3d_skipped(self, result, reason):
        self.assertFalse(result["is_ok"])
        piece, = result["pieces"]
        self.assertEqual(piece["skip_3d_reason"], reason)
        self.assertEqual(piece["registration"], {"is_valid": False, "status": "skipped"})
        self.assertEqual(piece["wire_measurements"], [])
        self.detector.submit_registration.assert_not_called()
        self.detector.submit_joint_measurement.assert_not_called()
        return piece

    def test_stream_order_skips_unreached_checks_and_3d(self):
        detector = self.detector
        detector.defect_order = self.order
        detector.chip_alignment_config = {}
        detector.chip_anomaly_config = {"configured": True}
        detector.silver_config = {"overflow": {}, "continuity": {}}
        with patch.object(self.module._chip_alignment, "measure_chip_alignment",
                          return_value={"enabled": True, "status": "ng", "is_ok": False}) as rotation, \
             patch.object(detector, "_check_wires") as wires, \
             patch.object(detector, "prepare_silver_boundary") as boundary, \
             patch.object(self.module, "silver_check") as silver:
            two_d, board = self.run_stream()
        self.assertFalse(two_d["is_ok"])
        self.assertEqual(two_d["first_ng_check"], "chip_rotation_2d")
        for name in set(self.order) - {"chip_rotation_2d"}:
            self.assertEqual(two_d[name]["status"], "skipped")
            self.assertIsNone(two_d[name]["is_ok"])
        rotation.assert_called_once()
        wires.assert_not_called()
        boundary.assert_not_called()
        silver.assert_not_called()
        detector.chip_anomaly_inspector.inspect.assert_not_called()
        piece = self.assert_3d_skipped(board, "2d_ng")
        self.assertEqual(piece["first_ng_check"], "chip_rotation_2d")

    def test_no_yolo_masks_skips_3d_with_default_flow(self):
        self.prediction.masks = None
        self.detector.chip_anomaly_inspector.inspect.return_value = {"enabled": False}
        self.detector.submit_2d_postprocess = Mock(side_effect=lambda pool, *args: pool.submit(lambda: None))
        two_d, board = self.run_stream()
        self.assertFalse(two_d["is_ok"])
        self.assertEqual(two_d["instances"], [])
        self.assert_3d_skipped(board, "no_2d_result")

    def test_single_piece_with_empty_2d_never_opens_the_depth_buffer(self):
        self.prediction.masks = None
        self.detector.chip_anomaly_inspector.inspect.return_value = {"enabled": False}
        self.detector.submit_2d_postprocess = Mock(side_effect=lambda pool, *args: pool.submit(lambda: None))
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            pipeline = BoardPipeline(self.detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            response = pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(3)
            self.assertTrue(response["ok"])
            self.assertEqual(response["results_2d"]["instances"], [])
            result = pipeline.submit_3d_piece(dict(
                board_id="A", piece_id=1, shm_name="unopened_depth",
                data_size=512, image_width=16, image_height=16)).result(3)
        self.assertTrue(result["ok"])
        self.assertFalse(result["results_piece"]["is_ok"])
        self.assertEqual(result["results_piece"]["skip_3d_reason"], "no_2d_result")
        self.detector._read_3d.assert_not_called()
        self.detector.submit_registration.assert_not_called()

    def test_priority_continues_after_pass_and_skips_later_checks(self):
        from algorithm.application.defect_checks import detection_order, evaluate_in_order
        rotation, silver, *later = self.order
        checks = {name: Mock(return_value={"is_ok": True}) for name in self.order}
        checks[silver].return_value = {"status": "uncertain", "is_ok": None}
        results, first_ng = evaluate_in_order(self.order, checks)
        self.assertEqual(first_ng, silver)
        checks[rotation].assert_called_once()
        checks[silver].assert_called_once()
        for name in later:
            checks[name].assert_not_called()
            self.assertEqual(results[name]["status"], "skipped")
        for invalid in ([rotation, rotation], ["unknown"], "wire_bending_2d"):
            with self.assertRaises(ValueError):
                detection_order(invalid)

    def test_batch_empty_masks_skips_synchronous_registration(self):
        self.prediction.masks = None
        with patch.object(self.module, "register_platform_masks") as register:
            response = self.detector.handle_detect({"images_2d": [{"piece_id": 1}], "image_3d": {}})
        self.assertTrue(response["ok"], response)
        self.assert_3d_skipped(response["results_board"], "no_2d_result")
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
