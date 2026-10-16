"""Exact differential checks for local 2D masks and shared grayscale."""
import importlib
from pathlib import Path
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))
from algorithm.detect_board import DetectBoard, _mask_geometry
from reference_silver_continuity import measure_continuity as reference_continuity
silver = importlib.import_module("algorithm.2D.silver_inspection")
anomaly = importlib.import_module("algorithm.2D.chip_anomaly")


class Local2DTests(unittest.TestCase):
    def test_continuity_matches_frozen_full_frame_reference(self):
        rng = np.random.default_rng(20261016)
        shape = (140, 180)
        for case in range(30):
            chip = np.zeros(shape, np.uint8)
            for x, y, w, h in [(0, 0, 20, 25), (40, 35, 30, 25),
                                (85, 40, 20, 20), (160, 120, 20, 20)]:
                chip[y:y + h, x:x + w] = 255
            chip[42:46, 49:53] = 0
            instances = [dict(class_name="chip", mask=chip),
                         dict(class_name="silver", mask=(rng.random(shape) < case / 30).astype(np.uint8)),
                         dict(class_name="wire", mask=(rng.random(shape) < 0.15).astype(np.uint8))]
            config = silver.ContinuitySettings(
                min_chip_area_px=20, outward_length_px=1 + case % 25,
                sector_count=[4, 72, 720][case % 3], occlusion_dilation_px=case % 3,
                min_valid_sector_px=1 + case % 4, min_sector_coverage=case / 30)
            with self.subTest(case=case):
                try:
                    expected = reference_continuity(instances, shape, config)
                except ValueError as expected_error:
                    with self.assertRaises(ValueError) as actual_error:
                        silver.measure_continuity(instances, shape, config)
                    self.assertEqual(str(actual_error.exception), str(expected_error))
                    continue
                actual = silver.measure_continuity(instances, shape, config)
                self.assertEqual(actual.summary(), expected.summary())
                for name in ["chip_mask", "ring_mask", "silver_mask", "occlusion_mask", "missing_mask"]:
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_continuity_preserves_error_messages(self):
        shape = (40, 60)
        mask = np.zeros(shape, np.uint8)
        mask[10:20, 10:20] = 1
        cases = [[], [dict(class_name="chip", mask=mask)],
                 [dict(class_name="chip", mask=mask[:10])]]
        for instances in cases:
            with self.assertRaises(ValueError) as expected:
                reference_continuity(instances, shape, silver.ContinuitySettings())
            with self.assertRaises(ValueError) as actual:
                silver.measure_continuity(instances, shape, silver.ContinuitySettings())
            self.assertEqual(str(actual.exception), str(expected.exception))

    def test_parallel_consumers_convert_target_grayscale_only_once(self):
        image = np.random.default_rng(3).integers(0, 256, (300, 500, 3), np.uint8)
        prepared = dict(image=image, gray_lock=threading.Lock())
        converter = cv2.cvtColor
        with patch("algorithm.detect_board.cv2.cvtColor", wraps=converter) as convert:
            with ThreadPoolExecutor(max_workers=8) as pool:
                grays = list(pool.map(lambda _: DetectBoard._prepared_gray_image(prepared), range(20)))
        self.assertEqual(convert.call_count, 1)
        self.assertTrue(all(gray is grays[0] for gray in grays))
        np.testing.assert_array_equal(grays[0], converter(image, cv2.COLOR_BGR2GRAY))

    def test_chip_contours_and_geometry_preserve_offsets_and_source(self):
        for x, y in [(0, 0), (20, 30), (100, 60)]:
            mask = np.zeros((80, 140), np.uint8)
            mask[y:y + 20, x:x + 30] = 128
            mask[y + 5:y + 10, x + 5:x + 10] = 0
            original = mask.copy()
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            expected = max((c for c in contours if len(c) >= 3), key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            expected += [200, -30]
            for view in [mask, mask[::-1, ::-1]]:
                if view is mask:
                    np.testing.assert_array_equal(anomaly._chip_contour_points(view, (200, -30)), expected)
                self.assertEqual(_mask_geometry(view, (200, -30), 1.0),
                                 _mask_geometry(view.astype(bool), (200, -30), 1.0))
            np.testing.assert_array_equal(mask, original)


if __name__ == "__main__":
    unittest.main()
