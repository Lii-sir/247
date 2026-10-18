"""ROI-derived silver boundary, fast distances, and template-free integration."""
import copy
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
silver = importlib.import_module('algorithm.2D.silver_inspection')
from algorithm.detect_board import DetectBoard


def config(**overrides):
    raw = dict(mode='roi_inset', inset_unit='px', insets=dict(left=10, top=7, right=20, bottom=13))
    raw.update(overrides)
    return dict(boundary=raw, silver_class='silver', tolerance_px=0, min_area_px=1)


class SilverRoiInsetTests(unittest.TestCase):
    def build(self, values=None, crop=(100, 200, 200, 280), shape=(80, 100), circle=None):
        return silver.build_boundary(values or config(), crop_box=crop, circle=circle, image_shape=shape)

    def test_pixel_insets_are_independent_and_use_existing_crop(self):
        boundary = self.build(circle=(20, 40, 560))
        self.assertEqual(boundary.points, ((10., 7.), (80., 7.), (80., 67.), (10., 67.)))
        self.assertEqual(boundary.mode, 'roi_inset')
        self.assertEqual(boundary.registration, dict(method='roi_inset', status='ok', is_valid=True, fallback_used=False))
        self.assertEqual(boundary.points, self.build(circle=(30, 40, 700)).points)
        self.assertEqual(boundary.points, self.build(config(insets=dict(left=10, top=7, right=20, bottom=13)),
                         crop=(110, 215, 210, 295)).points)

    def test_default_unit_is_pixels_and_zero_insets_allow_entire_half_open_roi(self):
        values = config(insets=dict(left=0, top=0, right=0, bottom=0))
        del values['boundary']['inset_unit']
        boundary = self.build(values)
        self.assertEqual(boundary.points, ((0., 0.), (100., 0.), (100., 80.), (0., 80.)))
        boundary.validate_image((80, 100))
        self.assertTrue(np.all(boundary.signed_distance([[0, 0], [99, 79]]) <= 0))
        result = silver.measure_overflow(np.ones((80, 100), bool), boundary, silver.OverflowSettings())
        self.assertEqual(result.defect_area_px, 0)

    def test_fractional_pixels_and_optional_radius_unit(self):
        pixel = self.build(config(insets=dict(left=10.5, top=7.5, right=20.5, bottom=13.5)))
        self.assertEqual(pixel.points[0], (10.5, 7.5))
        ratio = self.build(config(inset_unit='radius_ratio', insets=dict(left=.1, top=.07, right=.2, bottom=.13)), circle=(100, 100, 100))
        np.testing.assert_allclose(ratio.points, self.build().points, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'valid detected circle'):
            self.build(config(inset_unit='radius_ratio'))

    def test_invalid_insets_and_roi_fail_instead_of_passing_or_falling_back(self):
        for key in ('left', 'top', 'right', 'bottom'):
            for value in (-1, float('nan'), float('inf'), True, '2', None):
                values = config(); values['boundary']['insets'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.build(values)
        bad = [config(inset_unit='mm', fallback_to_template_points=True), config(insets={}), config(insets=None),
               config(insets=dict(left=80, top=0, right=20, bottom=0)),
               config(insets=dict(left=0, top=40, right=0, bottom=40))]
        for values in bad:
            prepared = silver.prepare_boundary(dict(overflow=values), crop_box=(100, 200, 200, 280),
                                               circle=None, image_shape=(80, 100))
            result = silver.inspect_masks([], (80, 100), dict(overflow=values),
                                          prepared_boundary=prepared)['silver_overflow']
            self.assertIsNone(result['is_ok'])
            self.assertEqual(result['status'], 'uncertain')
            self.assertEqual(result['registration']['method'], 'roi_inset')
            self.assertEqual(result['registration']['status'], 'failed')
            self.assertFalse(result['registration'].get('fallback_used', False))
        for crop in ((0, 0, 0, 80), (0, 0, 100, 79), (0, 0, float('nan'), 80)):
            with self.assertRaises(ValueError):
                self.build(crop=crop)

    def test_roi_mode_has_no_template_io_sift_or_gray_conversion(self):
        values = config(calibration_path='missing.json', template_path='missing.png')
        detector = DetectBoard(lambda _: None)
        detector.silver_config = dict(overflow=values)
        prepared = dict(crop_box=(100, 200, 200, 280), circle=None,
                        crop=np.zeros((80, 100, 3), np.uint8), image=object())
        with patch.object(detector, '_prepared_gray_image', side_effect=AssertionError('gray conversion')), \
             patch.object(silver, '_get_template', side_effect=AssertionError('template processing')), \
             patch.object(silver, '_registered_template_points', side_effect=AssertionError('template registration')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('template IO')):
            result = detector.prepare_silver_boundary(prepared)
        self.assertIsNone(result.error)
        self.assertEqual(result.boundary.points, self.build().points)

    def test_analytic_distance_matches_polygon_at_sides_corners_and_interior(self):
        rng = np.random.default_rng(191)
        boundary = self.build()
        polygon = silver.Boundary(boundary.points)
        points = np.concatenate((rng.uniform([-40, -30], [150, 120], (5000, 2)),
                                 np.array([[10, 7], [80, 67], [9, 6], [81, 68], [10, 40], [40, 67], [40, 40]])))
        np.testing.assert_allclose(boundary.signed_distance(points), polygon.signed_distance(points), atol=1e-10)
        self.assertAlmostEqual(boundary.signed_distance([[9, 6]])[0], np.sqrt(2))

    def test_overflow_decision_regions_and_masks_match_polygon(self):
        rng = np.random.default_rng(240)
        boundary = self.build()
        polygon = silver.Boundary(boundary.points)
        for density in (0.0, .01, .5, 1.0):
            mask = rng.random((80, 100)) < density
            for tolerance in (0, .5, 1, 2.5, 20):
                for area in (1, 5, 100):
                    settings = silver.OverflowSettings(tolerance_px=tolerance, min_area_px=area)
                    expected = silver.measure_overflow(mask, polygon, settings)
                    actual = silver.measure_overflow(mask, boundary, settings)
                    self.assertEqual(actual.defect_area_px, expected.defect_area_px)
                    self.assertEqual(actual.regions, expected.regions)
                    np.testing.assert_array_equal(actual.defect_mask, expected.defect_mask)
                    np.testing.assert_array_equal(actual.outside_mask, expected.outside_mask)

    def test_public_geometry_adds_crop_offset_once_and_prepared_boundary_is_reused(self):
        values = dict(overflow=config())
        prepared = silver.prepare_boundary(values, crop_box=(100, 200, 200, 280), circle=None, image_shape=(80, 100))
        mask = np.zeros((80, 100), np.uint8); mask[8:12, 11:16] = 1
        instances = [dict(class_name='silver', mask=mask)]
        with patch.object(silver, 'build_boundary', side_effect=AssertionError('boundary recomputed')):
            result = silver.inspect_masks(instances, (80, 100), values, prepared_boundary=prepared)
        public = silver.offset_geometry(result, (100, 200))['silver_overflow']
        self.assertTrue(public['is_ok'])
        self.assertEqual(public['boundary']['mode'], 'roi_inset')
        self.assertEqual(public['boundary']['coordinate_system'], 'image')
        self.assertEqual(public['boundary']['points'], [[110., 207.], [180., 207.], [180., 267.], [110., 267.]])
        self.assertTrue(public['boundary']['is_within_crop'])
        self.assertEqual(public['registration'], prepared.boundary.registration)

    def test_no_silver_is_unknown_and_outside_is_ng(self):
        kwargs = dict(crop_box=(100, 200, 200, 280), circle=None)
        settings = dict(overflow=config())
        no_silver = silver.inspect_masks([], (80, 100), settings, **kwargs)['silver_overflow']
        self.assertEqual(no_silver['status'], 'no_silver')
        self.assertIsNone(no_silver['is_ok'])
        mask = np.zeros((80, 100), np.uint8); mask[0, 0] = 1
        outside = silver.inspect_masks([dict(class_name='silver', mask=mask)], (80, 100), settings, **kwargs)['silver_overflow']
        self.assertFalse(outside['is_ok'])
        self.assertEqual(outside['measurement']['defect_area_px'], 1)


if __name__ == '__main__':
    unittest.main()
