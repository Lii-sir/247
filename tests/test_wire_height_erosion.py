"""Configurable per-class 3D wire edge removal and height selection contracts."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from algorithm import height_measurement as height
from algorithm import detect_board as board
from algorithm.registration import align_crop_mask_to_depth


class WireHeightErosionTests(unittest.TestCase):
    def test_config_defaults_and_independent_parameters(self):
        defaults = height.HeightConfig.from_dict({})
        self.assertEqual((defaults.wire_measurement_erode_px, defaults.thin_measurement_erode_px), (0, 0))
        configured = height.HeightConfig.from_dict(dict(wire_measurement_erode_px=3, thin_measurement_erode_px=1))
        self.assertEqual((configured.wire_measurement_erode_px, configured.thin_measurement_erode_px), (3, 1))
        for name in ('wire_measurement_erode_px', 'thin_measurement_erode_px'):
            for value in (-1, .5, 2.0, True, '2', None):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    height.HeightConfig.from_dict({name: value})

    def test_disabled_is_zero_copy_and_does_not_scan_or_construct_kernel(self):
        mask = np.ones((10, 10), np.uint8)
        with patch.object(cv2, 'boundingRect', side_effect=AssertionError('unexpected scan')):
            self.assertIs(height.erode_wire_mask_in_place(mask, 0), mask)

    def test_local_erosion_matches_full_frame_zero_border_at_all_edges(self):
        masks = [np.zeros((100, 140), np.uint8), np.ones((100, 140), np.uint8)*255]
        for x, y in ((0, 0), (0, 40), (110, 0), (110, 80), (60, 40)):
            mask = np.zeros((100, 140), np.uint8)
            mask[y:y+20, x:x+30] = 255
            mask[y+8:y+11, x+9:x+14] = 0
            masks.append(mask)
        rng = np.random.default_rng(29)
        for _ in range(8):
            mask = np.zeros((100, 140), np.uint8)
            for _ in range(4):
                points = rng.integers([0, 0], [140, 100], (2, 2))
                cv2.line(mask, tuple(points[0]), tuple(points[1]), 255, int(rng.integers(3, 16)))
            masks.append(mask)
        for index, mask in enumerate(masks):
            for radius in (1, 2, 4, 10):
                with self.subTest(mask=index, radius=radius):
                    expected = cv2.erode(mask, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    actual = mask.copy()
                    self.assertIs(height.erode_wire_mask_in_place(actual, radius), actual)
                    np.testing.assert_array_equal(actual, expected)

    def test_empty_or_overlarge_radius_does_not_allocate_a_large_kernel(self):
        for original in (np.zeros((30, 40), np.uint8), np.ones((30, 40), np.uint8)):
            with patch.object(height, '_ellipse_kernel', side_effect=AssertionError('oversized kernel')):
                actual = height.erode_wire_mask_in_place(original.copy(), 10**8)
                self.assertFalse(np.any(actual))

    def test_kernel_is_cached(self):
        self.assertIs(height._ellipse_kernel(2), height._ellipse_kernel(2))

    def test_local_height_enumeration_matches_original_order_for_uint8_and_views(self):
        rng = np.random.default_rng(400)
        depth = rng.integers(32770, 34000, (100, 140), dtype=np.uint16)
        plane = dict(coefficients=[.01, .02, 32768.])
        for density in (0, .001, .1, 1):
            source = (rng.random(depth.shape) < density).astype(np.uint8)*255
            for mask in (source, source[::-1, ::-1]):
                for percentile in (1, 50, 95, 100):
                    config = height.HeightConfig(height_percentile=percentile)
                    self.assertEqual(height.measure_mask_height(depth, mask, plane, config, 'wire', .8),
                                     height.measure_mask_height(depth, mask.astype(np.int16), plane, config, 'wire', .8))

    def fixture(self):
        depth = np.full((160, 180), 33000, np.uint16)
        items = []
        for instance_id, name, y in ((1, 'wire', 30), (2, 'thin', 80)):
            mask = np.zeros(depth.shape, np.uint8)
            mask[y:y+20, 20:130] = 255
            depth[y:y+2, 20:130] = 35000
            depth[y+18:y+20, 20:130] = 35000
            items.append(dict(instance_id=instance_id, class_name=name,
                              _mask_2d_array=mask, _mask_3d_array=mask.copy(),
                              bending_2d=dict(is_valid=True, is_ok=True)))
        return depth, items

    def detector(self, wire_radius, thin_radius):
        detector = board.DetectBoard(lambda _: None)
        detector.min_component_area = 1
        detector.height_config = replace(detector.height_config,
            wire_measurement_erode_px=wire_radius, thin_measurement_erode_px=thin_radius,
            plane_min_valid_points=10, chip_min_valid_points=10)
        return detector

    def measure(self, detector, depth, items, registration=None):
        h, w = depth.shape
        return detector._measure_wire_regions(depth, items,
            registration or dict(scale=1, tx=0, ty=0), dict(coefficients=[0., 0., 32768.]),
            [(0, 0, w, h)], (0, 0, w, h))

    def test_wire_and_thin_parameters_are_independent_and_remove_reflective_points(self):
        depth, items = self.fixture()
        baseline = self.measure(self.detector(0, 0), depth, copy.deepcopy(items))
        self.assertTrue(all(not item['height_3d']['is_ok'] for item in baseline))
        for radii, expected in (((2, 0), [True, False]), ((0, 2), [False, True]), ((2, 2), [True, True])):
            results = self.measure(self.detector(*radii), depth, copy.deepcopy(items))
            self.assertEqual([item['height_3d']['is_ok'] for item in results], expected)
            for index, item in enumerate(results):
                if radii[index]:
                    self.assertAlmostEqual(item['height_3d']['height_diff_mm'], (33000-32768)*.00105)
                    point = item['height_3d']['point']
                    self.assertEqual(int(depth[point['y'], point['x']]), 33000)
                    self.assertLess(item['mask_3d_local']['area_px'], baseline[index]['mask_3d_local']['area_px'])
                else:
                    self.assertEqual(item, baseline[index])
        for item in items:
            np.testing.assert_array_equal(item['_mask_2d_array'], item['_mask_3d_array'])

    def test_erosion_is_after_warp_and_in_depth_pixels(self):
        depth, items = self.fixture()
        detector = self.detector(2, 0)
        registration = dict(scale=2, tx=3, ty=5)
        depth = np.full((330, 370), 33000, np.uint16)
        with patch.object(board, 'measure_mask_height', wraps=height.measure_mask_height) as measured:
            self.measure(detector, depth, [items[0]], registration)
        actual = measured.call_args.args[1]
        cleaned = board._geometry.clean_mask(items[0]['_mask_2d_array'], min_area=1, kernel_size=5)
        aligned = align_crop_mask_to_depth(cleaned, (0, 0, 370, 330), depth.shape, registration)
        expected = cv2.erode(aligned, height._ellipse_kernel(2), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_array_equal(actual, expected)
        # Complete semantic masks were not changed by the measurement.
        np.testing.assert_array_equal(items[0]['_mask_2d_array'], items[0]['_mask_3d_array'])

    def test_erased_or_invalid_interior_is_invalid_without_edge_fallback(self):
        depth, items = self.fixture()
        for config, modified_depth in ((self.detector(20, 20), depth),
                                       (self.detector(2, 2), depth.copy())):
            if config.height_config.wire_measurement_erode_px == 2:
                modified_depth[32:48, :] = 0
                modified_depth[82:98, :] = 65535
            results = self.measure(config, modified_depth, copy.deepcopy(items))
            for item in results:
                result = item['height_3d']
                self.assertFalse(result['is_valid'])
                self.assertFalse(result['is_ok'])
                self.assertIsNone(result['point'])
                self.assertIsNone(result['height_diff_mm'])

    def test_platform_chip_planes_and_solder_results_do_not_change(self):
        depth, items = self.fixture()
        chip = np.zeros(depth.shape, np.uint8); chip[120:148, 20:100] = 255
        bond = np.zeros(depth.shape, np.uint8); bond[128:138, 45:60] = 255
        for instance_id, name, mask in ((3, 'chip', chip), (4, 'bond', bond)):
            items.append(dict(instance_id=instance_id, class_name=name,
                              _mask_2d_array=mask, _mask_3d_array=mask.copy()))
        outputs = []
        for radius in (0, 2):
            detector = self.detector(radius, radius)
            detector.height_config = replace(detector.height_config, platform_roi_offsets=((0., 0., 1., 1.),))
            with patch.object(board, 'map_platform_rois', return_value=(np.ones_like(depth, np.uint8), [], [])):
                outputs.append(detector._measure_piece(depth, copy.deepcopy(items),
                    dict(scale=1, tx=0, ty=0, is_valid=True, perspective_valid=True),
                    [(0, 0, 180, 160)], (0, 0, 180, 160), (10, 10, 10)))
        self.assertEqual(outputs[0]['reference_planes'], outputs[1]['reference_planes'])
        self.assertEqual(outputs[0]['solder_measurements'], outputs[1]['solder_measurements'])
        self.assertNotEqual(outputs[0]['wire_measurements'], outputs[1]['wire_measurements'])


if __name__ == '__main__':
    unittest.main()
