"""Solder-only erosion, alias safety, reference stability and coordinate contracts."""
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


class SolderHeightErosionTests(unittest.TestCase):
    def test_independent_config_defaults_and_strict_validation(self):
        config = height.HeightConfig.from_dict({})
        self.assertEqual((config.bond_measurement_erode_px, config.thin_solder_measurement_erode_px), (0, 0))
        config = height.HeightConfig.from_dict(dict(bond_measurement_erode_px=2, thin_solder_measurement_erode_px=3))
        self.assertEqual((config.bond_measurement_erode_px, config.thin_solder_measurement_erode_px), (2, 3))
        for name in ('bond_measurement_erode_px', 'thin_solder_measurement_erode_px'):
            for value in (-1, .5, 2.0, '1', True, None):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    height.HeightConfig.from_dict({name: value})

    def test_disabled_has_no_copy_scan_or_erosion(self):
        mask = np.ones((30, 40), np.uint8)
        with patch.object(cv2, 'boundingRect', side_effect=AssertionError('unexpected scan')):
            actual, offset = height.erode_measurement_mask_crop(mask, 0)
        self.assertIs(actual, mask)
        self.assertEqual(offset, (0, 0))

    def test_local_erosion_matches_full_image_and_never_mutates_shared_mask(self):
        masks = [np.zeros((110, 160), np.uint8), np.ones((110, 160), np.uint8)*255]
        for x, y in ((0, 0), (0, 50), (130, 0), (130, 90), (60, 50)):
            mask = np.zeros((110, 160), np.uint8)
            mask[y:y+20, x:x+30] = 255
            mask[y+8:y+11, x+8:x+11] = 0
            masks.append(mask)
        random = np.random.default_rng(78)
        for _ in range(8):
            mask = np.zeros((110, 160), np.uint8)
            for _ in range(4):
                points = random.integers([0, 0], [160, 110], (2, 2))
                cv2.line(mask, tuple(points[0]), tuple(points[1]), 255, int(random.integers(2, 20)))
            masks.append(mask)
        for mask in masks:
            for radius in (1, 2, 4, 10):
                original = mask.copy()
                local, (x, y) = height.erode_measurement_mask_crop(mask, radius)
                full = np.zeros_like(mask)
                full[y:y+local.shape[0], x:x+local.shape[1]] = local
                expected = cv2.erode(original, height._ellipse_kernel(radius), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                np.testing.assert_array_equal(full, expected)
                np.testing.assert_array_equal(mask, original)
                self.assertFalse(np.shares_memory(mask, local))

    def test_allocation_is_local_and_extreme_radius_does_not_build_huge_kernel(self):
        mask = np.zeros((1600, 2200), np.uint8); mask[400:420, 1000:1040] = 255
        local, offset = height.erode_measurement_mask_crop(mask, 1)
        self.assertEqual(local.shape, (20, 40))
        self.assertEqual(offset, (1000, 400))
        with patch.object(height, '_ellipse_kernel', side_effect=AssertionError('oversized kernel')):
            local, _ = height.erode_measurement_mask_crop(mask, 10**8)
        self.assertFalse(np.any(local))

    def fixture(self):
        depth = np.full((120, 180), 33000, np.uint16)
        mask = np.zeros_like(depth, np.uint8); mask[40:60, 70:110] = 255
        edge = mask - cv2.erode(mask, height._ellipse_kernel(2), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        depth[edge > 0] = 35000
        plane = dict(coefficients=[.2, -.1, 32768.])
        return depth, mask, plane

    def detector(self, bond_radius=0, solder_radius=0):
        result = board.DetectBoard(lambda _: None)
        result.height_config = replace(result.height_config,
            bond_measurement_erode_px=bond_radius, thin_solder_measurement_erode_px=solder_radius)
        return result

    def solder(self, detector, depth, mask, plane, endpoint='bond', chips=None):
        return detector._measure_solder(depth, mask, mask,
            dict(instance_id=5, class_name='bond' if endpoint=='bond' else 'thin'), endpoint,
            chips or [], {9: plane}, plane, np.zeros_like(mask), (0, 0, 180, 120))

    def test_local_sampling_offsets_match_full_mask_on_sloped_plane_and_board_conversion(self):
        depth, mask, plane = self.fixture()
        local, offset = height.erode_measurement_mask_crop(mask, 2)
        full = cv2.erode(mask, height._ellipse_kernel(2), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        expected = height.measure_mask_height(depth, full, plane, height.HeightConfig(), 'chip', .8)
        actual = height.measure_mask_height(depth, local, plane, height.HeightConfig(), 'chip', .8, mask_offset=offset)
        self.assertEqual(actual, expected)
        converted = height.height_points_to_board(dict(depth_start_row=500, solder_measurements=[dict(height_3d=actual)]))
        point = converted['solder_measurements'][0]['height_3d']['point']
        self.assertEqual(point['x'], actual['point']['x'])
        self.assertEqual(point['y'], actual['point']['y']+500)

    def test_bond_reflective_edges_removed_and_local_plane_uses_original_region(self):
        depth, mask, plane = self.fixture()
        original = mask.copy()
        with patch.object(board, 'fit_local_platform_plane', return_value=plane) as fit:
            baseline = self.solder(self.detector(), depth, mask, plane)
            result = self.solder(self.detector(bond_radius=2), depth, mask, plane)
        self.assertEqual(fit.call_count, 2)
        self.assertTrue(all(call.args[1] is mask for call in fit.call_args_list))
        np.testing.assert_array_equal(mask, original)
        self.assertFalse(baseline['height_3d']['is_ok'])
        self.assertTrue(result['height_3d']['is_ok'])
        self.assertEqual(result['reference_plane'], baseline['reference_plane'])
        full = cv2.erode(mask, height._ellipse_kernel(2), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        self.assertEqual(result['mask_3d_local'], board._mask_geometry(full, (0, 0), 1.0))
        self.assertEqual(result['height_3d'], height.measure_mask_height(depth, full, plane,
                         self.detector().height_config, 'local_wire_platform', .8))

    def test_chip_reference_and_membership_do_not_change(self):
        depth, mask, plane = self.fixture()
        chip = np.zeros_like(mask); chip[20:90, 40:150] = 255
        chips = [dict(instance_id=9, _mask_2d_array=chip)]
        with patch.object(board, 'fit_local_platform_plane', side_effect=AssertionError('chip must use chip plane')):
            baseline = self.solder(self.detector(), depth, mask, plane, chips=chips)
            eroded = self.solder(self.detector(bond_radius=2), depth, mask, plane, chips=chips)
        for key in ('location', 'reference_plane', 'reference_instance_id', 'chip_inside'):
            self.assertEqual(baseline[key], eroded[key])
        self.assertEqual(eroded['height_3d']['reference'], 'chip')

    def test_bond_and_both_thin_endpoints_use_independent_parameters(self):
        depth, mask, plane = self.fixture()
        with patch.object(board, 'fit_local_platform_plane', return_value=plane):
            for endpoint in ('bond', 'start', 'end'):
                baseline = self.solder(self.detector(), depth, mask, plane, endpoint)
                correct = self.detector(bond_radius=2) if endpoint=='bond' else self.detector(solder_radius=2)
                unrelated = self.detector(solder_radius=2) if endpoint=='bond' else self.detector(bond_radius=2)
                self.assertEqual(self.solder(unrelated, depth, mask, plane, endpoint), baseline)
                self.assertTrue(self.solder(correct, depth, mask, plane, endpoint)['height_3d']['is_ok'])

    def test_empty_or_invalid_interior_and_missing_reference_have_no_edge_fallback(self):
        depth, mask, plane = self.fixture()
        for radius, data, reference in ((30, depth, plane), (2, depth.copy(), plane), (2, depth, None)):
            if radius==2 and reference is not None:
                data[42:58, 72:108] = 65535
            with patch.object(board, 'fit_local_platform_plane', return_value=reference):
                result = self.solder(self.detector(bond_radius=radius), data, mask, reference)
            self.assertFalse(result['height_3d']['is_valid'])
            self.assertFalse(result['height_3d']['is_ok'])
            self.assertIsNone(result['height_3d']['point'])
            self.assertIsNone(result['height_3d']['height_mm'])
            if radius==30: self.assertEqual(result['mask_3d_local']['area_px'], 0)

    def piece(self, **parameters):
        detector = self.detector()
        detector.min_component_area = 1
        detector.height_config = replace(detector.height_config, plane_min_valid_points=10,
            chip_min_valid_points=10, platform_roi_offsets=((0., 0., 1., 1.),), **parameters)
        depth = np.full((160, 200), 33000, np.uint16)
        items = []
        for index, name, (x1,y1,x2,y2) in ((1,'thin',(30,40,170,60)), (2,'bond',(80,100,110,125)), (3,'chip',(10,20,80,80))):
            mask = np.zeros_like(depth,np.uint8);mask[y1:y2,x1:x2]=255
            items.append(dict(instance_id=index,class_name=name,_mask_2d_array=mask,_mask_3d_array=mask,
                              bending_2d=dict(is_valid=True,is_ok=True)))
        with patch.object(board,'map_platform_rois',return_value=(np.ones_like(depth,np.uint8),[],[])):
            result = detector._measure_piece(depth,items,dict(scale=1,tx=0,ty=0,is_valid=True,perspective_valid=True),
                [(0,0,200,160)],(0,0,200,160),(10,10,10))
        return result

    def test_thin_middle_endpoints_and_reference_planes_are_independent(self):
        baseline = self.piece()
        middle_only = self.piece(thin_measurement_erode_px=2)
        solder_only = self.piece(bond_measurement_erode_px=1, thin_solder_measurement_erode_px=2)
        self.assertEqual(baseline['solder_measurements'], middle_only['solder_measurements'])
        self.assertEqual(baseline['wire_measurements'], solder_only['wire_measurements'])
        self.assertEqual(baseline['reference_planes'], solder_only['reference_planes'])
        self.assertNotEqual(baseline['solder_measurements'], solder_only['solder_measurements'])
        for original, modified in zip(baseline['solder_measurements'], solder_only['solder_measurements']):
            self.assertEqual(original['mask_2d'], modified['mask_2d'])
            self.assertEqual(original['reference_plane'], modified['reference_plane'])
            self.assertEqual(original['location'], modified['location'])
            self.assertLess(modified['mask_3d_local']['area_px'], original['mask_3d_local']['area_px'])


if __name__ == '__main__':
    unittest.main()
