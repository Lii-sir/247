"""Event-gated tests for overlapping 3D computation with unfinished 2D diagnostics."""
import sys
from pathlib import Path
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from algorithm.application.board.board_pipeline import BoardPipeline
from algorithm.runtime.defect_plan import WIRE_HEIGHT, defect_orders, detection_order, evaluate_in_order
from pipeline_fakes import FakeDetector, depth_piece_request


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

    def test_3d_waits_for_complete_diagnostics_and_merges_2d_ng(self):
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, diagnostics_ng=True))
        self.assertTrue(self.detector.diagnostics_started.wait(1))
        depth = self.pipeline.submit_3d_piece(depth_piece_request("A"))
        self.assertFalse(self.detector.depth_started.is_set())
        self.assertFalse(depth.done())
        self.detector.diagnostics_gate.set()
        self.assertTrue(piece.result(2)["ok"])
        result = depth.result(2)
        self.assertTrue(result["ok"])
        self.assertFalse(result["results_piece"]["is_ok"])
        self.assertFalse(result["results_piece"]["diagnostics_ok"])
        self.assertEqual(self.detector.closed_inputs, 1)

    def test_diagnostics_exception_skips_waiting_depth(self):
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, fail_diagnostics=True))
        self.assertTrue(self.detector.diagnostics_started.wait(1))
        depth = self.pipeline.submit_3d_piece(depth_piece_request("A"))
        self.detector.diagnostics_gate.set()
        self.assertFalse(piece.result(2)["ok"])
        self.assertEqual(depth.result(2)["results_piece"]["skip_3d_reason"], "missing_2d_result")
        self.assertFalse(self.detector.depth_started.is_set())
        self.assertFalse(self.pipeline.has_pending)

    def test_completed_piece_replacement_cannot_pair_old_data_with_depth(self):
        self.detector.diagnostics_gate.set()
        self.assertTrue(self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="old")).result(2)["ok"])
        self.detector.gpu_gate.clear()
        self.detector.gpu_started.clear()
        replacement = self.pipeline.submit_2d(dict(board_id="A", piece_id=1, marker="new"))
        self.assertTrue(self.detector.gpu_started.wait(1))
        depth = self.pipeline.submit_3d_piece(depth_piece_request("A"))
        try:
            self.assertFalse(self.detector.depth_started.is_set())
        finally:
            self.detector.gpu_gate.set()
        self.assertTrue(replacement.result(2)["ok"])
        self.assertEqual(depth.result(2)["results_piece"]["marker"], "new")

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
        piece = self.pipeline.submit_2d(dict(board_id="A", piece_id=1))
        try:
            self.assertTrue(boundary_started.wait(1))
            self.assertTrue(self.detector.gpu_started.wait(1), "GPU waited for boundary preparation")
            self.assertFalse(piece.done())
            self.assertEqual(self.detector.closed_inputs, 0)
        finally:
            boundary_gate.set()
        self.assertTrue(piece.result(2)["ok"])
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

        self.module = detect_board
        self.detector = detector = detect_board.DetectBoard(lambda _: None)
        self.order = detection_order(["chip_rotation_2d", "silver_overflow_2d"])
        image = np.zeros((16, 16, 3), np.uint8)
        detector.expected_piece_count = 1
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
        detector.submit_2d_postprocess = Mock(side_effect=lambda pool, *args: pool.submit(lambda: None))

    def run_stream(self):
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            pipeline = BoardPipeline(self.detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            piece = pipeline.submit_2d({"board_id": "A", "piece_id": 1}).result(3)
            self.assertTrue(piece["ok"], piece)
            response = pipeline.submit_3d_piece(depth_piece_request("A")).result(3)
            self.assertTrue(response["ok"], response)
            self.assertFalse(pipeline.has_pending)
            return piece["results_2d"], response["results_piece"]

    def assert_3d_skipped(self, result, reason):
        self.assertFalse(result["is_ok"])
        piece = result
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

    def test_invalid_chip_geometry_does_not_fail_the_2d_request_or_appearance_verdict(self):
        import torch

        detector = self.detector
        detector.min_component_area = 1
        self.prediction.masks.data = torch.zeros((1, 16, 16), dtype=torch.uint8)
        self.prediction.masks.data[0, 3, 3] = 1
        self.prediction.masks.data[0, 9:14, 9] = 1
        detector.chip_anomaly_inspector = self.module._chip_anomaly.ChipAnomalyInspector({
            "device": "cpu", "aspect_ratio_models": [{"range": [.9, 1.1], "model": "chip"}],
        }, {})
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu, \
             patch.object(self.module._chip_alignment, "measure_chip_alignment", return_value={"enabled": False}):
            pipeline = BoardPipeline(detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            reply = pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(3)
        self.assertTrue(reply["ok"], reply)
        self.assertTrue(reply["results_2d"]["is_ok"])
        appearance = reply["results_2d"]["chip_anomaly_2d"]
        self.assertEqual("skipped", appearance["status"])
        self.assertEqual(2, len(appearance["skipped_chips"]))

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

    def test_selected_sequence_preserves_order_within_each_stage(self):
        rotation = "chip_rotation_2d"
        self.assertEqual(defect_orders({}), (None, None))
        self.assertEqual(defect_orders({"defect_sequence": []}), ((), ()))
        self.assertEqual(defect_orders({"defect_sequence": [WIRE_HEIGHT]}), ((), (WIRE_HEIGHT,)))
        self.assertEqual(defect_orders({"defect_sequence": [rotation]}), ((rotation,), ()))
        self.assertEqual(defect_orders({"defect_sequence": [WIRE_HEIGHT, rotation]}), ((rotation,), (WIRE_HEIGHT,)))
        self.assertEqual(defect_orders({"defect_sequence": None}), (None, None))
        for sequence in ("invalid", [rotation, rotation], ["unknown"]):
            with self.subTest(sequence=sequence), self.assertRaisesRegex(ValueError, "defect_sequence"):
                defect_orders({"defect_sequence": sequence})

    def test_selected_2d_checks_stop_single_piece_3d_without_reading_depth(self):
        detector = self.detector
        rotation, silver = "chip_rotation_2d", "silver_overflow_2d"
        detector.defect_order, detector.height_order = defect_orders(
            {"defect_sequence": [rotation, silver, WIRE_HEIGHT]})
        detector.skip_3d_on_2d_ng = True
        detector.chip_alignment_config = {}
        detector.silver_config = {"overflow": {}}
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu, \
             patch.object(self.module._chip_alignment, "measure_chip_alignment",
                          return_value={"enabled": True, "status": "ng", "is_ok": False}), \
             patch.object(self.module._silver_inspection, "inspect_masks", return_value=dict(
                 enabled=True, silver_overflow=dict(status="ok", is_ok=True),
                 disconnection=dict(status="disabled", is_ok=None))) as check_silver, \
             patch.object(detector, "_check_wires") as check_wires:
            pipeline = BoardPipeline(detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            response = pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(3)
            self.assertTrue(response["ok"], response)
            two_d = response["results_2d"]
            self.assertEqual(two_d["first_ng_check"], rotation)
            self.assertEqual(two_d[silver]["status"], "ok")
            self.assertEqual(two_d["wire_bending_2d"]["status"], "disabled")
            result = pipeline.submit_3d_piece(dict(
                board_id="A", piece_id=1, shm_name="unopened_depth",
                data_size=512, image_width=16, image_height=16)).result(3)
            self.assertFalse(pipeline.has_pending)
        self.assertTrue(result["ok"], result)
        piece = result["results_piece"]
        self.assertEqual(piece["skip_3d_reason"], "2d_ng")
        self.assertEqual(piece["defect_checks_3d"][WIRE_HEIGHT]["status"], "skipped")
        self.assertFalse(piece["is_ok"])
        check_silver.assert_called_once()
        check_wires.assert_not_called()
        detector._read_3d.assert_not_called()
        detector.submit_registration.assert_not_called()

    def test_gate_only_controls_3d_for_the_matching_piece(self):
        detector = self.detector
        detector.expected_piece_count = 2
        detector.skip_3d_on_2d_ng = True
        detector.defect_order, detector.height_order = defect_orders(
            {"defect_sequence": ["silver_overflow_2d", WIRE_HEIGHT]})

        def silver(pool, inputs, prepared, timings, metadata, is_current):
            passed = prepared["piece_id"] == 2
            overflow = dict(status="ok" if passed else "overflow", is_ok=passed)
            return pool.submit(lambda: dict(
                diagnostics=dict(enabled=True, silver_overflow=overflow,
                                 disconnection=dict(status="disabled", is_ok=None)),
                public_instances=[], mask_postprocess_ms=0., cpu_2d_worker={}))

        def joint(state, piece, entry, depth, correction, inputs, timings, request_id):
            timings.update(registration_queue_ms=0., registration_phase_ms=0., joint_measurement_phase_ms=0.)
            completed = Future()
            completed.set_result(dict(piece_id=piece["piece_id"], registration=dict(is_valid=True), is_ok=True))
            return completed

        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu:
            pipeline = BoardPipeline(detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            with patch.object(detector, "submit_2d_postprocess", side_effect=silver), \
                 patch.object(pipeline, "_submit_joint", side_effect=joint) as measured:
                first, second = (pipeline.submit_2d(dict(board_id="A", piece_id=pid)) for pid in (1, 2))
                self.assertFalse(first.result(3)["results_2d"]["is_ok"])
                self.assertTrue(second.result(3)["results_2d"]["is_ok"])
                rejected, passed = (pipeline.submit_3d_piece(depth_piece_request("A", pid)).result(3)["results_piece"]
                                    for pid in (1, 2))
                self.assertEqual(rejected["skip_3d_reason"], "2d_ng")
                self.assertTrue(passed["is_ok"])
                measured.assert_called_once()
                self.assertEqual(measured.call_args.args[1]["piece_id"], 2)

                detector.skip_3d_on_2d_ng = False
                self.assertFalse(pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(3)["results_2d"]["is_ok"])
                result = pipeline.submit_3d_piece(depth_piece_request("A")).result(3)["results_piece"]
                self.assertEqual(result["first_ng_check"], "silver_overflow_2d")
                self.assertFalse(result["is_ok"])
                self.assertEqual(measured.call_count, 2)


    def test_only_2d_checks_pass_the_board_without_3d_registration(self):
        rotation = "chip_rotation_2d"
        self.detector.defect_order, self.detector.height_order = defect_orders(
            {"defect_sequence": [rotation]})
        self.detector.chip_alignment_config = {}
        with patch.object(self.module._chip_alignment, "measure_chip_alignment",
                          return_value={"enabled": True, "status": "ok", "is_ok": True}):
            two_d, board = self.run_stream()
        self.assertTrue(two_d["is_ok"])
        self.assertTrue(board["is_ok"])
        piece = board
        self.assertEqual(piece["skip_3d_reason"], "no_3d_checks")
        self.assertTrue(all(check["status"] == "disabled" for check in piece["defect_checks_3d"].values()))
        self.detector.submit_registration.assert_not_called()
        self.detector.submit_joint_measurement.assert_not_called()

    def test_only_3d_checks_keep_segmentation_without_any_2d_defect_checks(self):
        self.detector.defect_order, self.detector.height_order = defect_orders(
            {"defect_sequence": [WIRE_HEIGHT]})
        with ThreadPoolExecutor(2) as cpu, ThreadPoolExecutor(1) as gpu, \
             patch.object(self.detector, "_check_wires") as check_wires, \
             patch.object(self.module._chip_alignment, "measure_chip_alignment") as check_rotation:
            pipeline = BoardPipeline(self.detector, cpu, gpu)
            self.addCleanup(pipeline.close)
            response = pipeline.submit_2d(dict(board_id="A", piece_id=1)).result(3)
        self.assertTrue(response["ok"], response)
        two_d = response["results_2d"]
        self.assertTrue(two_d["is_ok"])
        self.assertTrue(two_d["instances"])
        self.assertEqual(two_d["wire_bending_2d"]["status"], "disabled")
        self.assertEqual(two_d["chip_rotation_2d"]["status"], "disabled")
        check_wires.assert_not_called()
        check_rotation.assert_not_called()
        self.detector.chip_anomaly_inspector.inspect.assert_not_called()

    def test_independent_inspection_policy_loaded_for_joint_initialization(self):
        import json
        from algorithm.runtime.config import load_runtime_config

        manifest = ROOT / "config" / "config.json"
        config = load_runtime_config(manifest.read_text(encoding="utf-8-sig"), str(manifest))
        self.assertIn("inspection", config)
        self.assertNotIn("skip_3d_on_2d_ng", config["common_2d"]["parameters"])
        config["inspection"] = {"defect_sequence": [WIRE_HEIGHT], "skip_3d_on_2d_ng": True}
        self.detector.config_path = str(manifest)
        self.detector._load_config(json.dumps(config))
        self.assertEqual(self.detector.defect_order, ())
        self.assertEqual(self.detector.height_order, (WIRE_HEIGHT,))
        self.assertTrue(self.detector.skip_3d_on_2d_ng)


if __name__ == "__main__":
    unittest.main()
