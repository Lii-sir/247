"""Compare localized height operations against the original full-image formulas."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from algorithm import height_measurement as height


def reference_local_plane(depth, target_mask, exclude_mask, config):
    target = (np.asarray(target_mask) > 0).astype(np.uint8)
    if not np.any(target) or config.platform_local_ring_px <= 0:
        return None
    outer = cv2.dilate(target, height._ellipse_kernel(config.platform_local_ring_px)) > 0
    inner = cv2.dilate(target, height._ellipse_kernel(max(1, config.semantic_exclusion_dilation_px))) > 0
    ring = outer & ~inner & (exclude_mask == 0) & height._valid_depth(depth, config)
    ys, xs = np.nonzero(ring)
    if len(xs) < config.plane_min_valid_points:
        return None
    try:
        return height._fit_plane_points(depth, xs, ys, config, "local_wire_platform")
    except ValueError:
        return None


def reference_measure(depth, mask, plane, config, reference, max_height):
    valid = (mask > 0) & height._valid_depth(depth, config)
    ys, xs = np.nonzero(valid)
    result = dict(reference=reference, is_valid=False, is_ok=False,
                  height_mm=None, height_diff_mm=None, point=None)
    if not len(xs) or plane is None:
        return result
    values = depth[ys, xs].astype(np.float64)
    index = int(np.argmin(np.abs(values - np.percentile(values, config.height_percentile))))
    x, y, value = int(xs[index]), int(ys[index]), float(values[index])
    absolute = (value - config.gray_offset) * config.vertical_resolution_mm
    base = (height._evaluate_plane(plane, x, y) - config.gray_offset) * config.vertical_resolution_mm
    return dict(reference=reference, is_valid=True,
                is_ok=None if max_height is None else bool(absolute - base < max_height),
                height_mm=float(absolute), height_diff_mm=float(absolute - base), point=dict(x=x, y=y))


class LocalHeightTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(391)
        yy, xx = np.indices((180, 260))
        self.depth = (33000 + xx + yy * 2 + self.rng.integers(-4, 5, xx.shape)).astype(np.uint16)
        self.depth[self.rng.random(xx.shape) < 0.05] = 0
        self.depth[self.rng.random(xx.shape) < 0.05] = 65535
        self.excluded = (self.rng.random(xx.shape) < 0.08).astype(np.uint8) * 255
        self.config = height.HeightConfig(plane_min_valid_points=10, chip_min_valid_points=10,
                                          platform_local_ring_px=12)

    def test_local_planes_match_full_frame_at_edges_and_for_disjoint_targets(self):
        for radius, inner in [(12, 3), (2, 8), (0, 3)]:
            config = replace(self.config, platform_local_ring_px=radius,
                             semantic_exclusion_dilation_px=inner)
            for x, y in [(0, 0), (250, 0), (0, 170), (250, 170), (110, 60)]:
                mask = np.zeros(self.depth.shape, np.uint8)
                mask[y:y + 10, x:x + 10] = 255
                for disjoint in [False, True]:
                    if disjoint:
                        mask[40:45, 30:35] = 255
                    with self.subTest(radius=radius, inner=inner, x=x, y=y, disjoint=disjoint):
                        self.assertEqual(height.fit_local_platform_plane(
                            self.depth, mask, self.excluded, config), reference_local_plane(
                            self.depth, mask, self.excluded, config))

    def test_local_plane_empty_invalid_excluded_and_wrong_shape(self):
        mask = np.zeros(self.depth.shape, np.uint8)
        self.assertIsNone(height.fit_local_platform_plane(self.depth, mask, self.excluded, self.config))
        mask[50:70, 100:120] = 255
        for depth, excluded in [(self.depth * 0, self.excluded),
                                 (self.depth, np.ones_like(self.excluded))]:
            self.assertEqual(height.fit_local_platform_plane(depth, mask, excluded, self.config),
                             reference_local_plane(depth, mask, excluded, self.config))
        with self.assertRaises(ValueError):
            height.fit_local_platform_plane(self.depth, mask[:10], self.excluded, self.config)
        with self.assertRaises(ValueError):
            height.fit_local_platform_plane(self.depth, mask, self.excluded[:10], self.config)

    def test_sparse_height_preserves_point_order_percentile_and_invalid_result(self):
        plane = dict(coefficients=[1.0, 2.0, 33000.0])
        for density in [0, 0.001, 0.1, 1.0]:
            mask = (self.rng.random(self.depth.shape) < density).astype(np.int16)
            mask[0, 0] = -1  # Negative values are not foreground.
            for percentile in [1, 50, 95, 100]:
                config = replace(self.config, height_percentile=percentile)
                for current_plane in [plane, None]:
                    for limit in [0.0, 0.8, None]:
                        with self.subTest(density=density, percentile=percentile,
                                          plane=current_plane, limit=limit):
                            self.assertEqual(height.measure_mask_height(
                                self.depth, mask, current_plane, config, "platform", limit),
                                reference_measure(self.depth, mask, current_plane, config,
                                                  "platform", limit))

    def test_chip_matching_reuses_regions_and_preserves_smallest_enclosing_chip(self):
        from algorithm import detect_board
        large = np.zeros((100, 140), np.uint8)
        large[10:90, 10:130] = 255
        large[30:70, 40:100] = 0  # The external-region contract fills holes.
        small = np.zeros_like(large)
        small[35:65, 45:95] = 255
        chips = [{"instance_id": 1, "_mask_2d_array": large},
                 {"instance_id": 2, "_mask_2d_array": small}]
        with patch.object(detect_board, "_filled_external_region",
                          wraps=detect_board._filled_external_region) as filled:
            for _ in range(3):
                for x, y, expected in [(50, 50, 2), (20, 20, 1), (0, 0, None)]:
                    bond = np.zeros_like(large)
                    bond[y, x] = 255
                    chip, inside = detect_board.DetectBoard._matching_chip(
                        {"_mask_2d_array": bond}, chips)
                    self.assertEqual(chip["instance_id"] if chip else None, expected)
                    self.assertEqual(inside, expected is not None)
            self.assertEqual(filled.call_count, len(chips))


if __name__ == "__main__":
    unittest.main()
