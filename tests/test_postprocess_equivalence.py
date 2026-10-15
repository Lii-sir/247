"""Pixel/geometry contracts for the CPU postprocessing optimizations."""
import copy
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
silver = importlib.import_module("algorithm.2D.silver_inspection")
geometry = importlib.import_module("algorithm.2D.defect_geometry")
from algorithm.detect_board import DetectBoard, _mask_geometry


def reference_geometry(mask, offset, epsilon):
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    area = int(np.count_nonzero(binary))
    if not area:
        return {"area_px": 0, "bounding_box": [], "contours": []}
    ys, xs = np.nonzero(binary)
    ox, oy = map(int, offset)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([[int(x) + ox, int(y) + oy] for x, y in points])
    return {"area_px": area,
            "bounding_box": [int(xs.min()) + ox, int(ys.min()) + oy,
                             int(xs.max()) + 1 + ox, int(ys.max()) + 1 + oy],
            "contours": polygons}


class PostprocessEquivalenceTests(unittest.TestCase):
    def masks(self):
        shape = (100, 140)
        yield np.zeros(shape, np.uint8)
        yield np.ones(shape, np.uint8)
        for x, y in ((0, 0), (0, 30), (70, 0), (120, 80), (60, 40)):
            mask = np.zeros(shape, np.uint8)
            mask[y:y + 20, x:x + 20] = 1
            mask[y + 6:y + 9, x + 6:x + 9] = 0
            yield mask
        for gap in (2, 6, 12, 30):
            mask = np.zeros(shape, np.uint8)
            mask[25:45, 20:40] = 1
            mask[25:45, 40 + gap:60 + gap] = 1
            yield mask
        random = np.random.default_rng(123)
        for _ in range(12):
            mask = np.zeros(shape, np.uint8)
            for _ in range(5):
                a, b = random.integers([0, 0], [140, 100], (2, 2))
                cv2.line(mask, tuple(a), tuple(b), 1, int(random.integers(2, 10)))
            yield mask

    def test_cropped_morphology_preserves_full_frame_components(self):
        for index, mask in enumerate(self.masks()):
            for kernel in (1, 3, 5, 7):
                with self.subTest(mask=index, kernel=kernel):
                    original = mask.copy()
                    expected = geometry.clean_mask(mask, min_area=5, kernel_size=kernel)
                    local, (x, y) = geometry.clean_mask_crop(mask, min_area=5, kernel_size=kernel)
                    restored = np.zeros_like(mask, dtype=bool)
                    restored[y:y + local.shape[0], x:x + local.shape[1]] = local
                    np.testing.assert_array_equal(restored, expected)
                    np.testing.assert_array_equal(mask, original)

    def test_wire_results_and_roi_assignment_match_full_frame_processing(self):
        detector = DetectBoard(lambda _: None)
        detector.min_component_area = 5
        roi_boxes = [(0, 0, 75, 99), (45, 0, 139, 99)]

        def clean_full(mask, min_area, kernel_size):
            return geometry.clean_mask(mask, min_area, kernel_size), (0, 0)

        for index, mask in enumerate(self.masks()):
            with self.subTest(mask=index):
                base = [dict(instance_id=1, class_name="wire", mask=mask.copy()),
                        dict(instance_id=2, class_name="thin", mask=mask.copy())]
                expected_instances = copy.deepcopy(base)
                with patch.object(geometry, "clean_mask_crop", side_effect=clean_full):
                    expected = detector._measure_2d_wire_instances(
                        expected_instances, roi_boxes, (200, 300, 340, 400))
                actual_instances = copy.deepcopy(base)
                actual = detector._measure_2d_wire_instances(
                    actual_instances, roi_boxes, (200, 300, 340, 400))
                self.assertEqual(actual, expected)
                for actual_instance, expected_instance in zip(actual_instances, expected_instances):
                    for key in ("rectangle", "rectangle_ratio", "bending_2d", "is_ok"):
                        self.assertEqual(actual_instance.get(key), expected_instance.get(key))

    def test_geometry_preserves_area_bounds_contour_order_and_offsets(self):
        for index, mask in enumerate(self.masks()):
            for epsilon in (0.5, 1.0, 2.0):
                with self.subTest(mask=index, epsilon=epsilon):
                    for view in (mask, mask[::-1, ::-1]):
                        self.assertEqual(_mask_geometry(view, (200, -30), epsilon),
                                         reference_geometry(view, (200, -30), epsilon))

    def test_sector_counts_and_missing_pixels_match_individual_sector_scans(self):
        random = np.random.default_rng(94)
        for sector_count in (4, 72, 720):
            for box in ((20, 15, 45, 40), (0, 0, 20, 25), (40, 35, 65, 60)):
                for coverage in (0.0, 0.3, 1.0):
                    with self.subTest(sectors=sector_count, box=box, coverage=coverage):
                        shape = (60, 65)
                        chip = np.zeros(shape, bool)
                        x1, y1, x2, y2 = box
                        chip[y1:y2, x1:x2] = True
                        chip[20:28, 43:48] = True  # Exclude other chip pixels from the ring.
                        component = np.zeros(shape, bool)
                        component[y1:y2, x1:x2] = True
                        silver_mask = random.random(shape) < coverage
                        occlusion = random.random(shape) < 0.15
                        settings = silver.ContinuitySettings(
                            sector_count=sector_count, outward_length_px=8,
                            min_valid_sector_px=3, min_sector_coverage=0.7,
                        )
                        result = silver._measure_chip_component(
                            component, chip, silver_mask, occlusion, settings, 1)
                        ring = np.zeros(shape, bool)
                        ring[max(0, y1-8):min(shape[0], y2+8),
                             max(0, x1-8):min(shape[1], x2+8)] = True
                        ring[y1:y2, x1:x2] = False
                        ring &= ~chip
                        yy, xx = np.indices(shape, dtype=np.float64)
                        center = ((x1+x2)/2, (y1+y2)/2)
                        angles = (np.arctan2(yy-center[1], xx-center[0]) + 2*np.pi) % (2*np.pi)
                        indices = np.floor(angles/(2*np.pi)*sector_count).astype(np.int32)
                        missing = np.zeros(shape, bool)
                        totals = {"covered": 0, "missing": 0, "valid": 0}
                        for index, actual in enumerate(result["sectors"]):
                            sector = ring & (indices == index)
                            count = int(sector.sum())
                            covered = sector & (silver_mask | occlusion)
                            covered_count = int(covered.sum())
                            for key, expected in (("ring_px", count), ("valid_px", count),
                                                  ("silver_px", int((sector & silver_mask).sum())),
                                                  ("covered_px", covered_count),
                                                  ("occluded_px", int((sector & occlusion).sum()))):
                                self.assertEqual(actual[key], expected)
                            expected_coverage = covered_count/count if count else None
                            self.assertEqual(actual["coverage"], expected_coverage)
                            if count < settings.min_valid_sector_px:
                                status = "insufficient"
                            else:
                                status = "covered" if (covered_count >= settings.min_sector_silver_px
                                    and expected_coverage >= settings.min_sector_coverage) else "missing"
                                totals["valid"] += 1
                                totals[status] += 1
                                if status == "missing":
                                    missing |= sector & ~covered
                            self.assertEqual(actual["status"], status)
                        np.testing.assert_array_equal(result["ring_mask"], ring)
                        np.testing.assert_array_equal(result["missing_mask"], missing)
                        self.assertEqual(result["valid_sector_count"], totals["valid"])
                        self.assertEqual(result["covered_sector_count"], totals["covered"])
                        self.assertEqual(result["missing_sector_count"], totals["missing"])


if __name__ == "__main__":
    unittest.main()
