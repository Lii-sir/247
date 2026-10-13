"""Coordinate contract tests; no camera, checkpoints or GPU required."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from algorithm.height_measurement import height_points_to_board


class HeightPointTests(unittest.TestCase):
    def piece(self, start=1000):
        shared = {"is_valid": True, "is_ok": True, "height_mm": 1.3,
                  "height_diff_mm": 0.4, "point": {"x": 75, "y": 31}}
        return {"piece_id": 2, "depth_start_row": start, "depth_end_row": start + 200,
                "wire_measurements": [{"height_3d": shared}],
                "solder_measurements": [{"height_3d": shared}],
                "masks": [{"height_3d": shared, "mask_3d_local": {"bounding_box": [70, 25, 80, 40]}}],
                "reference_planes": {"wire_platform": {"coefficients": [1, 2, 3]}}}

    def test_original_board_coordinates_for_every_output_location(self):
        result = height_points_to_board(self.piece())
        for field in ("wire_measurements", "solder_measurements", "masks"):
            height = result[field][0]["height_3d"]
            self.assertEqual(height["point"], {"x": 75, "y": 1031})
            self.assertEqual(height["point_coordinate_space"], "board_3d")
            self.assertEqual(height["height_diff_mm"], 0.4)

    def test_shared_height_references_are_not_offset_twice_or_mutated(self):
        original = self.piece()
        snapshot = copy.deepcopy(original)
        result = height_points_to_board(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(result["masks"][0]["height_3d"]["point"]["y"], 1031)
        self.assertEqual(result["wire_measurements"][0]["height_3d"]["point"]["y"], 1031)

    def test_conversion_is_idempotent(self):
        once = height_points_to_board(self.piece())
        self.assertEqual(height_points_to_board(once), once)

    def test_zero_start_row(self):
        result = height_points_to_board(self.piece(0))
        self.assertEqual(result["wire_measurements"][0]["height_3d"]["point"], {"x": 75, "y": 31})

    def test_planes_masks_and_registration_stay_local(self):
        original = self.piece()
        original["registration"] = {"ty": 10, "matrix_2d_to_3d": [[1, 0, 4], [0, 1, 10]]}
        result = height_points_to_board(original)
        self.assertEqual(result["reference_planes"], original["reference_planes"])
        self.assertEqual(result["registration"], original["registration"])
        self.assertEqual(result["masks"][0]["mask_3d_local"], original["masks"][0]["mask_3d_local"])

    def test_invalid_and_absent_points_stay_absent(self):
        original = self.piece()
        original["wire_measurements"] = [{"height_3d": {"is_valid": False, "point": None}}]
        original["solder_measurements"] = [{"height_3d": None}]
        original["masks"] = [{"height_3d": None}]
        result = height_points_to_board(original)
        self.assertIsNone(result["wire_measurements"][0]["height_3d"]["point"])
        self.assertIsNone(result["masks"][0]["height_3d"])

    def test_existing_board_point_is_not_shifted(self):
        original = self.piece()
        original["wire_measurements"][0]["height_3d"]["point_coordinate_space"] = "board_3d"
        self.assertEqual(height_points_to_board(original)["wire_measurements"][0]["height_3d"]["point"]["y"], 31)

    def detector(self):
        from algorithm import detect_board as module
        detector = module.DetectBoard(lambda text: None)
        detector._align_instance = Mock(side_effect=lambda item, *args: item)
        measurement = self.piece()
        detector._measure_piece = Mock(return_value={"is_ok": True,
            "wire_measurements": measurement["wire_measurements"],
            "solder_measurements": measurement["solder_measurements"],
            "reference_planes": measurement["reference_planes"]})
        return detector, module, measurement

    def patches(self, module):
        from contextlib import ExitStack
        stack = ExitStack()
        for name in ("platform_mask_2d", "platform_mask_3d", "register_platform_masks",
                     "refine_affine_registration", "estimate_perspective"):
            stack.enter_context(patch.object(module, name, return_value={"is_valid": True}))
        return stack

    def test_stream_and_concurrent_piece_output_is_global(self):
        detector, module, measurement = self.detector()
        entry = {"registration_mask": np.zeros((8, 8), np.uint8),
                 "gray_image": np.zeros((8, 8), np.uint8), "instances": measurement["masks"],
                 "image_shape": [8, 8], "crop_box": (0, 0, 8, 8), "roi_boxes": []}
        with self.patches(module):
            result = detector.measure_prepared_piece(2, entry, np.zeros((1400, 100), np.uint16),
                                                     {"y1": 1000, "y2": 1200})
        self.assertEqual(result["wire_measurements"][0]["height_3d"]["point"]["y"], 1031)
        self.assertEqual(result["masks"][0]["height_3d"]["point"]["y"], 1031)

    def test_batch_output_is_global(self):
        detector, module, measurement = self.detector()
        detector.expected_piece_count = 1
        detector.model = object()
        detector._synchronize_model = Mock()
        detector._validate_descriptors = Mock(return_value=[{"piece_id": 1}])
        detector._read_2d = Mock(return_value=np.zeros((8, 8, 3), np.uint8))
        detector._read_3d = Mock(return_value=np.zeros((1400, 100), np.uint16))
        detector._stream_crop = Mock(return_value=(np.zeros((8, 8, 3), np.uint8), (0, 0, 8, 8), (2, 2, 1), []))
        detector._predict_batch = Mock(return_value=[object()])
        detector._extract_instances = Mock(return_value=measurement["masks"])
        with self.patches(module), patch.object(module, "segment_depth_image", return_value=[{"y1": 1000, "y2": 1200}]):
            response = detector.handle_detect({"image_3d": {}, "images_2d": [], "batch_id": "A"})
        self.assertTrue(response["ok"], response)
        result = response["results_board"]["pieces"][0]
        self.assertEqual(result["solder_measurements"][0]["height_3d"]["point"], {"x": 75, "y": 1031})
        self.assertEqual(result["height_point_coordinate_space"], "board_3d")


if __name__ == "__main__":
    unittest.main()
