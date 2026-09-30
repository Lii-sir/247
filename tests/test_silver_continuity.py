import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2 as cv
import numpy as np

from part_segmentation.image_io import write_image
from part_segmentation.models import Segment, SegmentationResult, SegmentationSettings
from silver_inspection.continuity import ContinuitySettings, measure_continuity
from silver_inspection.continuity_export import export_continuity_batch
from silver_inspection.continuity_pipeline import SilverContinuityInspector, evaluate_continuity
from silver_inspection.continuity_visualization import render_continuity_comparison, render_continuity_overlay


def make_segments(gap_sector=None, occluded_sector=None):
    shape = (140, 140)
    chip = np.zeros(shape, bool)
    chip[45:95, 45:95] = True
    settings = ContinuitySettings(outward_length_px=8, sector_count=36,
                                  min_sector_silver_px=2, min_sector_coverage=0.001)
    ring = np.zeros(shape, dtype=bool)
    ring[37:103, 37:103] = True
    ring[45:95, 45:95] = False
    yy, xx = np.indices(shape, dtype=float)
    angle = (np.arctan2(yy - 70, xx - 70) + 2 * np.pi) % (2 * np.pi)
    silver = ring.copy()
    if gap_sector is not None:
        sector = (angle >= 2 * np.pi * gap_sector / 36) & (angle < 2 * np.pi * (gap_sector + 1) / 36)
        silver[ring & sector] = False
    thin = np.zeros(shape, bool)
    if occluded_sector is not None:
        sector = (angle >= 2 * np.pi * occluded_sector / 36) & (angle < 2 * np.pi * (occluded_sector + 1) / 36)
        thin[ring & sector] = True
        silver[ring & sector] = False
    segments = (
        Segment(0, "chip", .99, (45, 45, 95, 95), chip),
        Segment(1, "silver", .99, (0, 0, 140, 140), silver),
        Segment(2, "thin", .99, (0, 0, 140, 140), thin),
    )
    return segments, settings


class ContinuityGeometryTests(unittest.TestCase):
    def test_full_ring_is_ok(self):
        segments, settings = make_segments()
        measurement = measure_continuity(segments, (140, 140, 3), settings)
        self.assertFalse(measurement.is_disconnected)
        self.assertEqual(measurement.missing_sector_count, 0)
        self.assertEqual(measurement.valid_sector_count, 36)
        self.assertEqual(measurement.silver_area_px, measurement.ring_area_px)
        self.assertEqual(measurement.chips[0]["bbox_xyxy"], [45, 45, 95, 95])
        self.assertEqual(measurement.chips[0]["outer_bbox_xyxy"], [37, 37, 103, 103])
        self.assertEqual(measurement.ring_area_px, 66 * 66 - 50 * 50)

    def test_one_sector_gap_is_disconnected(self):
        segments, settings = make_segments(gap_sector=0)
        measurement = measure_continuity(segments, (140, 140, 3), settings)
        self.assertTrue(measurement.is_disconnected)
        self.assertGreaterEqual(measurement.missing_sector_count, 1)
        self.assertTrue(measurement.missing_mask.any())

    def test_thin_occlusion_is_ignored(self):
        segments, settings = make_segments(occluded_sector=4)
        measurement = measure_continuity(segments, (140, 140, 3), settings)
        self.assertFalse(measurement.is_disconnected)
        self.assertGreater(measurement.occluded_area_px, 0)
        # Partial thin/bond overlap removes only those pixels; the remaining
        # visible pixels in the sector are still valid evidence.
        self.assertGreaterEqual(measurement.ignored_sector_count, 1)
        self.assertEqual(measurement.silver_area_px + measurement.occluded_area_px, measurement.ring_area_px)
        self.assertFalse(np.any(measurement.missing_mask & measurement.occlusion_mask))

    def test_dilation_and_threshold_validation(self):
        for kwargs in ({"outward_length_px": 0}, {"sector_count": 3}, {"min_sector_silver_px": 0},
                       {"min_sector_coverage": 1.1}, {"occlusion_dilation_px": -1}, {"min_chip_area_px": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContinuitySettings(**kwargs)
        segments, settings = make_segments()
        with self.assertRaises(ValueError):
            measure_continuity(segments, (10, 10, 3), settings)


class ContinuityPipelineTests(unittest.TestCase):
    def _result(self, segments):
        image = np.full((140, 140, 3), 100, np.uint8)
        return SegmentationResult(Path("target.png"), image, segments, 1)

    def test_evaluation_statuses_are_explicit(self):
        segments, settings = make_segments()
        result = evaluate_continuity(self._result(segments), settings)
        self.assertEqual(result.status, "ok")
        self.assertIs(result.is_defect, False)
        broken, _ = make_segments(gap_sector=0)
        result = evaluate_continuity(self._result(broken), settings)
        self.assertEqual(result.status, "disconnected")
        self.assertIs(result.is_defect, True)
        no_chip = tuple(segment for segment in segments if segment.class_name != "chip")
        result = evaluate_continuity(self._result(no_chip), settings)
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.is_defect)
        json.dumps(result.summary(), allow_nan=False)

    def test_required_model_classes_are_validated(self):
        engine = SimpleNamespace(class_names=("bond", "wire", "chip", "thin"))
        with self.assertRaisesRegex(ValueError, "silver"):
            SilverContinuityInspector("best.pt", segmenter=engine)

    def test_images_and_overlays_are_not_mutated(self):
        segments, settings = make_segments(gap_sector=0)
        result = evaluate_continuity(self._result(segments), settings)
        original = result.segmentation.image.copy()
        self.assertEqual(render_continuity_overlay(result).shape, original.shape)
        self.assertEqual(render_continuity_comparison(result).shape, (204, 280, 3))
        np.testing.assert_array_equal(result.segmentation.image, original)

    def test_mocked_inspector_and_batch_export(self):
        segments, settings = make_segments()
        image = np.full((140, 140, 3), 100, np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input", root / "output"
            for name in ("same.png", "nested/same.png"):
                write_image(source / name, image)
            result = evaluate_continuity(SegmentationResult(Path("target.png"), image, segments, 1), settings)
            with patch("silver_inspection.continuity_export.SilverContinuityInspector") as factory:
                factory.return_value.inspect.return_value = result
                status = export_continuity_batch(root / "weights.pt", source, output, settings,
                                                 SegmentationSettings(), recursive=True)
            self.assertEqual(status, 0)
            self.assertTrue((output / "same.png.continuity.png").is_file())
            self.assertTrue((output / "nested/same.png.comparison.jpg").is_file())
            report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"ok": 2})
            with self.assertRaises(ValueError):
                export_continuity_batch(root / "weights.pt", source, source / "output", settings,
                                        SegmentationSettings())


if __name__ == "__main__":
    unittest.main()
