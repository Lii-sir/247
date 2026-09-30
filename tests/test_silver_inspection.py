"""Deterministic geometry and integration tests; no trained silver model required."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2 as cv
import numpy as np

from part_segmentation.image_io import read_image, write_image
from part_segmentation.models import Segment, SegmentationResult, SegmentationSettings
from point_matcher.core import MappedPoint, MatchResult, transform_points
from point_matcher.tests.test_core import textured_image
from silver_inspection.calibration import Calibration, load_calibration, save_calibration
from silver_inspection.export import export_batch
from silver_inspection.geometry import Boundary, OverflowSettings, measure_overflow
from silver_inspection.pipeline import SilverInspector, evaluate
from silver_inspection.visualization import render_comparison, render_overlay


def fixture(mask=None, points=((10, 10), (30, 10), (30, 30), (10, 30)), mode="polygon", name="silver"):
    if mask is None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[15:26, 15:36] = True
    image = np.full((*mask.shape, 3), 120, dtype=np.uint8)
    segment = Segment(0, name, 0.95, (0, 0, mask.shape[1], mask.shape[0]), mask)
    segmentation = SegmentationResult(Path("target.png"), image, (segment,), 1.0)
    boundary = Boundary(points, mode)
    match = MatchResult("target.png", "ok", "matched",
                        [MappedPoint(i + 1, x, y, x, y, True) for i, (x, y) in enumerate(points)])
    return segmentation, match, boundary


class GeometryTests(unittest.TestCase):
    def test_exact_mask_area_and_maximum_distance(self):
        segmentation, _, boundary = fixture()
        m = measure_overflow(segmentation.segments[0].mask, boundary, OverflowSettings())
        self.assertEqual(m.silver_area_px, 231)
        self.assertEqual(m.outside_area_px, 55)
        self.assertEqual(m.defect_area_px, 55)
        self.assertEqual(m.max_outside_distance_px, 5)
        self.assertEqual(m.regions[0]["box_xyxy"], [31, 15, 36, 26])

    def test_inside_and_boundary_pixels_do_not_overflow(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:31, 10:31] = True
        _, _, boundary = fixture(mask)
        self.assertEqual(measure_overflow(mask, boundary, OverflowSettings()).defect_area_px, 0)

    def test_tolerance_uses_target_pixels_and_strict_distance(self):
        segmentation, _, boundary = fixture()
        m = measure_overflow(segmentation.segments[0].mask, boundary, OverflowSettings(tolerance_px=2))
        self.assertEqual(m.outside_area_px, 55)
        self.assertEqual(m.candidate_area_px, 33)
        self.assertEqual(m.defect_area_px, 33)
        self.assertEqual(m.max_outside_distance_px, 5)

    def test_component_area_filter_and_eight_connectivity(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20, 34] = mask[21, 35] = mask[5, 5] = True
        _, _, boundary = fixture(mask)
        m = measure_overflow(mask, boundary, OverflowSettings(min_area_px=2))
        self.assertEqual(m.candidate_area_px, 3)
        self.assertEqual(m.defect_area_px, 2)
        self.assertEqual(len(m.regions), 1)

    def test_polygon_winding_concavity_and_fractional_edges(self):
        points = [(10.5, 10), (30.5, 10), (30.5, 20), (20, 20), (20, 30), (10.5, 30)]
        queries = [(11, 15), (10, 15), (25, 25), (15, 25), (20, 25)]
        expected = [-0.5, 0.5, 5, -4.5, 0]
        np.testing.assert_allclose(Boundary(points).signed_distance(queries), expected)
        np.testing.assert_allclose(Boundary(points[::-1]).signed_distance(queries), expected)

    def test_line_infinite_extension_inside_point_and_reversed_endpoints(self):
        queries = [(15, 5), (20, 5), (24, 45)]
        for points in ([(20, 10), (20, 30), (10, 20)], [(20, 30), (20, 10), (10, 20)]):
            np.testing.assert_allclose(Boundary(points, "line").signed_distance(queries), [-5, 0, 4])
        np.testing.assert_allclose(Boundary([(20, 10), (20, 30), (30, 20)], "line").signed_distance(queries), [5, 0, -4])

    def test_signed_distance_matches_opencv_reference(self):
        points = [(2, 2), (40, 5), (25, 20), (38, 40), (5, 35)]
        rng = np.random.default_rng(1)
        queries = rng.uniform(0, 50, (100, 2))
        contour = np.array(points, np.float32)
        expected = [-cv.pointPolygonTest(contour, tuple(map(float, p)), True) for p in queries]
        np.testing.assert_allclose(Boundary(points).signed_distance(queries), expected, atol=1e-5)

    def test_invalid_boundaries(self):
        cases = [([], "polygon"), ([(0, 0), (1, 1)], "polygon"),
                 ([(0, 0), (5, 5), (0, 5), (5, 0)], "polygon"),
                 ([(0, 0), (5, 0), (5, 5), (0, 0)], "polygon"),
                 ([(0, 0), (5, 0), (2, 0), (2, 5), (0, 5)], "polygon"),
                 ([(0, 0), (2, 2), (4, 4)], "line"),
                 ([(0, 0), (2, 2), (float("nan"), 4)], "polygon"),
                 ([(0, 0), (5, 0), (5, 5)], "unknown")]
        for points, mode in cases:
            with self.subTest(points=points, mode=mode), self.assertRaises(ValueError):
                Boundary(points, mode)

    def test_settings_validation_and_out_of_frame_boundary(self):
        for kwargs in ({"silver_class": " "}, {"tolerance_px": -1}, {"tolerance_px": float("nan")},
                       {"min_area_px": 0}, {"min_area_px": 1.2}, {"min_area_px": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OverflowSettings(**kwargs)
        with self.assertRaises(ValueError):
            measure_overflow(np.zeros((10, 10), bool), Boundary([(0, 0), (20, 0), (0, 20)]), OverflowSettings())


class PipelineTests(unittest.TestCase):
    def test_union_deduplicates_masks_and_preserves_holes(self):
        segmentation, match, boundary = fixture()
        mask = segmentation.segments[0].mask
        mask[20, 34] = False
        segmentation = SegmentationResult(segmentation.image_path, segmentation.image, segmentation.segments * 2, 1)
        before = segmentation.image.copy()
        result = evaluate(segmentation, match, boundary, OverflowSettings())
        self.assertEqual(result.status, "overflow")
        self.assertIs(result.is_defect, True)
        self.assertEqual(result.measurement.defect_area_px, 54)
        self.assertEqual(result.measurement.silver_area_px, 230)
        self.assertFalse(result.measurement.silver_mask[20, 34])
        overlay = render_overlay(result)
        self.assertEqual(overlay.shape, before.shape)
        np.testing.assert_array_equal(segmentation.image, before)
        self.assertEqual(render_comparison(result).shape, (114, 100, 3))
        json.dumps(result.summary(), allow_nan=False)

    def test_no_silver_wrong_class_empty_mask_never_pass(self):
        segmentation, match, boundary = fixture(name="bond")
        for segments in (segmentation.segments, (),
                         (Segment(0, "silver", 0.9, (0, 0, 1, 1), np.zeros((50, 50), bool)),)):
            sample = SegmentationResult(segmentation.image_path, segmentation.image, segments, 1)
            result = evaluate(sample, match, boundary, OverflowSettings())
            self.assertEqual(result.status, "no_silver")
            self.assertIsNone(result.is_defect)

    def test_failed_partial_and_incomplete_matching_never_pass(self):
        segmentation, match, boundary = fixture()
        for status in ("error", "partial", "outside"):
            match.status = status
            result = evaluate(segmentation, match, boundary, OverflowSettings())
            self.assertEqual(result.status, "uncertain")
            self.assertIsNone(result.is_defect)
        match.status = "ok"
        match.points.pop()
        self.assertIsNone(evaluate(segmentation, match, boundary, OverflowSettings()).is_defect)

    def test_mask_mismatch_cannot_pass(self):
        segmentation, match, boundary = fixture()
        wrong = Segment(0, "silver", 0.9, (0, 0, 5, 5), np.ones((5, 5), bool))
        segmentation = SegmentationResult(segmentation.image_path, segmentation.image, (wrong,), 1)
        result = evaluate(segmentation, match, boundary, OverflowSettings())
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.is_defect)

    def test_explicit_class_selection_and_accepted_tolerance(self):
        segmentation, match, boundary = fixture(name="bond")
        result = evaluate(segmentation, match, boundary, OverflowSettings("bond", tolerance_px=5))
        self.assertEqual(result.status, "ok")
        self.assertIs(result.is_defect, False)

    def test_model_missing_silver_rejected_before_matching(self):
        engine = SimpleNamespace(class_names=("bond", "wire", "chip", "thin"))
        calibration = Calibration(Path("missing.png"), fixture()[2])
        with self.assertRaisesRegex(ValueError, "bond, wire, chip, thin"):
            SilverInspector("best.pt", calibration, segmenter=engine)

    def test_real_sift_perspective_with_controlled_segmentation(self):
        cv.setNumThreads(2)
        with tempfile.TemporaryDirectory() as tmp:
            template = textured_image()
            path = Path(tmp) / "template.png"
            write_image(path, template)
            boundary = Boundary([(70, 80), (500, 80), (500, 350), (70, 350)])
            matrix = np.array([[1.05, .09, 95], [-.05, 1.08, 80], [.00014, -.00008, 1]])
            target = cv.warpPerspective(template, matrix, (880, 680))
            mask = np.zeros(target.shape[:2], np.uint8)
            center = tuple(np.rint(transform_points([(250, 200)], matrix)[0]).astype(int))
            cv.circle(mask, center, 20, 1, -1)
            segmentation = SegmentationResult(Path(tmp) / "target.png", target,
                                              (Segment(0, "silver", .9, (0, 0, 880, 680), mask.astype(bool)),), 1)
            engine = Mock(class_names=("silver",))
            engine.predict.return_value = segmentation
            inspector = SilverInspector("best.pt", Calibration(path, boundary), segmenter=engine)
            result = inspector.inspect(segmentation.image_path)
            self.assertEqual(result.status, "ok")
            np.testing.assert_allclose(result.boundary.points, transform_points(boundary.points, matrix), atol=1.5)
            engine.predict.return_value = SegmentationResult(segmentation.image_path, np.zeros_like(target), segmentation.segments, 1)
            self.assertEqual(inspector.inspect(segmentation.image_path).status, "uncertain")


class CalibrationAndExportTests(unittest.TestCase):
    def test_round_trip_relative_template_and_existing_match_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = Calibration(root / "模板.png", fixture()[2])
            path = root / "边界.json"
            save_calibration(path, calibration)
            self.assertEqual(load_calibration(path), calibration)
            data = calibration.summary()
            data.pop("boundary_mode")
            data["template_path"] = "模板.png"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_calibration(path), calibration)
            with self.assertRaises(ValueError):
                save_calibration(path, calibration, [path])
            with self.assertRaises(ValueError):
                save_calibration(calibration.template_path, calibration)

    def test_batch_continues_errors_retains_relative_paths_and_records_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input", root / "output"
            segmentation, match, boundary = fixture()
            for name in ("same.png", "same.bmp", "nested/same.png", "broken.png"):
                write_image(source / name, segmentation.image)
            calibration = Calibration(root / "template.png", boundary)
            def inspect(path):
                if path.name == "broken.png":
                    raise ValueError("broken")
                result = SegmentationResult(path, segmentation.image, segmentation.segments, 1)
                return evaluate(result, match, boundary, OverflowSettings())
            with patch("silver_inspection.export.SilverInspector") as factory:
                factory.return_value.inspect.side_effect = inspect
                from point_matcher.core import MatchSettings
                factory.return_value.match_settings = MatchSettings()
                status = export_batch(root / "weights.pt", source, output, calibration,
                                      OverflowSettings(), SegmentationSettings(), recursive=True)
            self.assertEqual(status, 1)
            for name in ("same.png", "same.bmp", "nested/same.png"):
                self.assertTrue((output / f"{name}.comparison.jpg").is_file())
                self.assertTrue((output / f"{name}.overflow.png").is_file())
            data = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(data["counts"], {"error": 1, "overflow": 3})
            self.assertEqual(data["overflow_settings"]["silver_class"], "silver")
            with self.assertRaises(ValueError):
                export_batch(root / "weights.pt", source, source / "output", calibration,
                             OverflowSettings(), SegmentationSettings())
            with self.assertRaises(ValueError):
                export_batch(root / "weights.pt", source, output, calibration, OverflowSettings(),
                             SegmentationSettings(), calibration_path=output / "summary.json")

    def test_cli_validates_continuity_settings(self):
        from silver_inspection.__main__ import main
        for args, exit_code in ((["--outward-px", "0"], 1), (["--sectors", "3"], 1),
                                (["--min-coverage", "2"], 1)):
            with self.subTest(args=args), self.assertRaises(SystemExit) as exc:
                main(args)
                self.assertEqual(exc.exception.code, exit_code)

    def test_main_routes_without_changing_default_entry(self):
        from main import main
        with patch("silver_inspection.__main__.main", return_value=17) as run:
            self.assertEqual(main(["--silver", "--help"]), 17)
            run.assert_called_once_with(["--help"])
        with patch("part_segmentation.__main__.main", return_value=19) as run:
            self.assertEqual(main(["--help"]), 19)
            run.assert_called_once_with(["--help"])


if __name__ == "__main__":
    unittest.main()
