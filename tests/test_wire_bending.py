"""Contracts for ROI edge intersections and per-instance coverage."""
import copy
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
geometry = importlib.import_module("algorithm.2D.defect_geometry")
presence = importlib.import_module("algorithm.2D.wire_roi_presence")
from algorithm.detect_board import DetectBoard
from algorithm.detect_2D import Detect2D


class WireBendingTests(unittest.TestCase):
    roi = (20, 20, 101, 81)

    def straight(self):
        mask = np.zeros((110, 130), np.uint8)
        mask[45:56, 5:120] = 1
        return mask

    def curved(self):
        mask = np.zeros((110, 130), np.uint8)
        cv2.polylines(mask, [np.array([[5, 50], [20, 50], [60, 28],
                                     [100, 50], [120, 50]], np.int32)], False, 1, 9)
        return mask

    def test_straight_exact_intersections_and_denominator(self):
        result = geometry.measure_wire_bending(self.straight(), [self.roi])
        self.assertTrue(result["is_valid"])
        self.assertTrue(result["is_ok"])
        self.assertEqual(result["mask_ratio"], 1)
        self.assertEqual(result["roi_mask_pixel_count"], 81 * 11)
        self.assertEqual(result["covered_mask_pixel_count"], 81 * 11)
        self.assertEqual(set(map(tuple, result["rectangle"])),
                         {(20, 45), (20, 55), (100, 45), (100, 55)})

    def test_bow_uses_instance_pixels_not_polygon_area(self):
        mask = self.curved()
        result = geometry.measure_wire_bending(mask, [self.roi])
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["is_ok"])
        self.assertGreater(result["mask_ratio"], 0)
        self.assertLess(result["mask_ratio"], .75)
        polygon = np.zeros_like(mask)
        cv2.fillPoly(polygon, [np.array(result["rectangle"], np.int32)], 1)
        expected_total = int(np.count_nonzero(mask[20:81, 20:101]))
        expected_covered = int(np.count_nonzero((mask & polygon)[20:81, 20:101]))
        self.assertEqual(result["roi_mask_pixel_count"], expected_total)
        self.assertEqual(result["covered_mask_pixel_count"], expected_covered)
        self.assertEqual(result["mask_ratio"], expected_covered / expected_total)
        self.assertNotEqual(result["mask_ratio"], expected_covered / np.count_nonzero(polygon))

    def test_holes_in_polygon_do_not_penalize_instance_coverage(self):
        mask = self.straight()
        mask[47:54, 35:85] = 0
        result = geometry.measure_wire_bending(mask, [self.roi])
        self.assertEqual(result["mask_ratio"], 1)

    def test_vertical_diagonal_and_tapered_shapes(self):
        for vertices in ([[[45, 5], [55, 5], [55, 100], [45, 100]]],
                         [[[5, 25], [5, 35], [120, 65], [120, 55]]],
                         [[[5, 42], [5, 58], [120, 53], [120, 47]]]):
            mask = np.zeros((110, 130), np.uint8)
            cv2.fillPoly(mask, [np.array(vertices[0], np.int32)], 1)
            result = geometry.measure_wire_bending(mask, [self.roi])
            self.assertTrue(result["is_valid"], result)
            self.assertGreater(result["mask_ratio"], .95)
            self.assertTrue(cv2.isContourConvex(np.array(result["rectangle"], np.int32)))

    def test_boundary_run_wrapping_top_left_corner(self):
        mask = np.zeros((110, 130), np.uint8)
        cv2.line(mask, (5, 5), (110, 100), 1, 11)
        result = geometry.measure_wire_bending(mask, [self.roi])
        self.assertTrue(result["is_valid"], result)
        self.assertGreater(result["mask_ratio"], .9)

    def test_invalid_intersections(self):
        contained = np.zeros((110, 130), np.uint8)
        contained[45:56, 35:85] = 1
        one_end = self.straight()
        one_end[:, 70:] = 0
        multiple = self.straight()
        multiple[65:71, 5:120] = 1
        full = np.ones((110, 130), np.uint8)
        tangent = np.zeros((110, 130), np.uint8)
        tangent[50, 5:120] = 1
        for mask, status in ((contained, "insufficient_intersections"),
                             (one_end, "insufficient_intersections"),
                             (multiple, "ambiguous_intersections"),
                             (full, "ambiguous_intersections"),
                             (tangent, "degenerate_shape")):
            with self.subTest(status=status):
                result = geometry.measure_wire_bending(mask, [self.roi])
                self.assertFalse(result["is_valid"])
                self.assertFalse(result["is_ok"])
                self.assertEqual(result["rectangle"], [])
                self.assertEqual(result["status"], status)

    def test_empty_outside_and_degenerate_roi(self):
        for mask, rois, status in ((np.zeros((110, 130)), [self.roi], "empty_mask"),
                                   (self.straight(), [], "outside_rois"),
                                   (self.straight(), [(0, 0, 5, 5)], "outside_rois"),
                                   (self.straight(), [(20, 20, 21, 80)], "degenerate_shape")):
            result = geometry.measure_wire_bending(mask, rois)
            self.assertEqual(result["status"], status)
            self.assertFalse(result["is_ok"])

    def test_local_mask_and_public_offsets_preserve_results(self):
        mask = self.curved()
        full = geometry.measure_wire_bending(mask, [self.roi], output_offset=(200, 300))
        local = geometry.measure_wire_bending(mask[15:90, 3:125], [self.roi],
                                             mask_offset=(3, 15), output_offset=(200, 300))
        self.assertEqual(full, local)
        local_contained = geometry.measure_wire_bending(mask[25:75, 30:90], [self.roi],
                                                       mask_offset=(30, 25))
        self.assertEqual(local_contained["status"], "insufficient_intersections")

    def test_threshold_is_inclusive(self):
        result = geometry.measure_wire_bending(self.curved(), [self.roi], 0)
        self.assertTrue(geometry.measure_wire_bending(
            self.curved(), [self.roi], result["mask_ratio"])["is_ok"])
        self.assertFalse(geometry.measure_wire_bending(
            self.curved(), [self.roi], result["mask_ratio"] + .001)["is_ok"])

    def detectors(self):
        board = DetectBoard(lambda _: None)
        board.min_component_area = 5
        legacy = Detect2D(lambda _: None)
        legacy.mask_config = dict(min_component_area=5, rectangle_ratio_threshold=.75)
        return board._measure_2d_wire_instances, legacy._measure_2d_wire_regions

    def test_both_entry_points_and_roi_aggregation(self):
        for measure in self.detectors():
            instances = [dict(instance_id=1, class_name="wire", mask=self.straight()),
                         dict(instance_id=2, class_name="thin", mask=self.curved())]
            measure(instances, [self.roi], (200, 300, 330, 410))
            checks = presence.check_wire_roi_presence(instances, [self.roi], (200, 300))
            self.assertFalse(checks[0]["is_ok"])
            self.assertEqual(checks[0]["status"], "bending_out_of_range")
            self.assertEqual(checks[0]["instance_ids"], [1, 2])
            self.assertEqual(checks[0]["mask_pixel_count"], sum(
                i["bending_2d"]["roi_mask_pixel_count"] for i in instances))
            for item in instances:
                self.assertTrue(item["bending_2d"]["is_valid"])
                self.assertTrue(all(x >= 200 and y >= 300 for x, y in item["rectangle"]))

    def test_entry_points_agree_on_valid_and_invalid_measurements(self):
        board_measure, legacy_measure = self.detectors()
        for mask in (self.straight(), self.curved(), np.zeros((110, 130), np.uint8)):
            a = [dict(instance_id=1, class_name="wire", mask=mask)]
            b = copy.deepcopy(a)
            board_measure(a, [self.roi], (200, 300, 330, 410))
            legacy_measure(b, [self.roi], (200, 300, 330, 410))
            self.assertEqual(a[0]["bending_2d"], b[0]["bending_2d"])

    def test_roi_assignment_reuses_cleaned_instance_not_raw_mask(self):
        for measure in self.detectors():
            mask = np.pad(self.straight(), ((0, 150), (0, 270)))
            # Raw noise outweighs the wire but isolated flecks disappear in
            # morphology. Raw-mask assignment would select the wrong ROI.
            mask[3:257:6, 143:397:6] = 1
            self.assertGreater(np.count_nonzero(mask[:, 140:400]), 81 * 11)
            rois = [self.roi, (140, 0, 400, 260)]
            instances = [dict(instance_id=1, class_name="wire", mask=mask)]
            measure(instances, rois, (0, 0, 130, 110))
            checks = presence.check_wire_roi_presence(instances, rois)
            self.assertEqual(instances[0]["bending_2d"]["roi_id"], 1)
            self.assertTrue(checks[0]["is_ok"])
            self.assertEqual(checks[1]["status"], "wire_missing")

    @unittest.skipUnless(torch.cuda.is_available(), "prediction conversion requires CUDA")
    def test_legacy_prediction_conversion_retains_full_wire_and_failures(self):
        detector = Detect2D(lambda _: None)
        detector.mask_config = dict(min_component_area=5, rectangle_ratio_threshold=.75,
                                    crop_padding=5)
        narrow = np.zeros((110, 130), np.uint8)
        narrow[65, 5:120] = 1
        prediction = SimpleNamespace(
            masks=SimpleNamespace(data=torch.as_tensor(
                np.stack([self.straight(), narrow]), device="cuda")),
            boxes=SimpleNamespace(xyxy=torch.zeros((2, 4), device="cuda"),
                                  conf=torch.ones(2, device="cuda"),
                                  cls=torch.zeros(2, device="cuda")), names={0: "wire"})
        instances = detector._convert_predictions(
            [prediction], np.zeros((110, 130, 3), np.uint8), (0, 0, 130, 110), [self.roi])
        self.assertEqual(len(instances), 2)
        np.testing.assert_array_equal(instances[0]["mask"] > 0, self.straight() > 0)
        detector._measure_2d_wire_regions(instances, [self.roi], (0, 0, 130, 110))
        self.assertTrue(instances[0]["bending_2d"]["is_ok"])
        self.assertEqual(instances[1]["bending_2d"]["status"], "empty_mask")

    def test_joint_height_reuses_bending_or_measures_original_thin_instance(self):
        from algorithm import detect_board as module
        detector = DetectBoard(lambda _: None)
        detector.min_component_area = 5
        mask = self.straight()
        expected = detector._measure_2d_wire_instances(
            [dict(instance_id=1, class_name="thin", mask=mask)],
            [self.roi], (200, 300, 330, 410))[0]["bending_2d"]
        central = mask.copy()
        central[:, :40] = 0
        central[:, 85:] = 0
        for cached in (False, True):
            wire = dict(instance_id=1, class_name="thin", _mask_2d_array=mask)
            if cached:
                wire["bending_2d"] = expected
            with patch.object(module, "split_thin_mask", return_value=(central, [])), \
                 patch.object(module, "align_crop_mask_to_depth", return_value=mask), \
                 patch.object(module, "measure_mask_height", return_value={"is_ok": True}), \
                 patch.object(geometry, "measure_wire_bending", wraps=geometry.measure_wire_bending) as measure:
                results = detector._measure_wire_regions(
                    np.zeros_like(mask), [wire], {}, None, [self.roi], (200, 300, 330, 410))
            self.assertEqual(results[0]["bending_2d"], expected)
            self.assertEqual(measure.call_count, 0 if cached else 1)

    def test_cleaning_failure_is_unavailable_and_empty_roi_is_ng(self):
        for measure in self.detectors():
            mask = np.zeros((110, 130), np.uint8)
            mask[50, 20:101] = 1
            instances = [dict(instance_id=1, class_name="wire", mask=mask)]
            measure(instances, [self.roi], (0, 0, 130, 110))
            check = presence.check_wire_roi_presence(instances, [self.roi])[0]
            self.assertEqual(check["status"], "bending_unavailable")
            self.assertFalse(check["is_ok"])
            self.assertEqual(presence.check_wire_roi_presence([], [self.roi])[0]["status"],
                             "wire_missing")


if __name__ == "__main__":
    unittest.main()
