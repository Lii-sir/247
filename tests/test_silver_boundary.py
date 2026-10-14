"""Test actual SIFT mapping, boundary decisions and calibration installation on CPU."""
import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "visualization"))
silver = importlib.import_module("algorithm.2D.silver_inspection")
calibrator = importlib.import_module("calibrate_silver_boundary")
from algorithm.timing import timing_context


class SilverBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        random = np.random.default_rng(41)
        self.template = random.integers(0, 255, (360, 480, 3), dtype=np.uint8)
        self.template = cv2.GaussianBlur(self.template, (3, 3), 0.7)
        for i in range(18):
            x, y = int(random.integers(25, 450)), int(random.integers(25, 330))
            cv2.circle(self.template, (x, y), int(random.integers(5, 14)), (255, 255, 255), 2)
            cv2.putText(self.template, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        self.path = self.root / "template.png"
        cv2.imencode(".png", self.template)[1].tofile(self.path)
        self.points = [[80, 90], [340, 90], [340, 270], [80, 270]]
        self.calibration = dict(
            mode="polygon", template_path="template.png", template_points=self.points,
            template_size=[480, 360], template_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            min_matches=11, min_inliers=8, ratio_threshold=0.7, min_inlier_ratio=0.25,
            max_image_side=480, max_features=2000, ransac_threshold=3)
        self.json_path = self.root / "boundary.json"
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")
        self.config = dict(boundary=dict(mode="polygon", calibration_path="boundary.json"),
                           silver_class="silver", tolerance_px=0, min_area_px=1)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, image=None, crop=(0, 0, 480, 360), shape=(360, 480), config=None):
        return silver.build_boundary(config or self.config, crop_box=crop, circle=None,
                                     image_shape=shape, image=self.template if image is None else image,
                                     base_dir=self.root)

    def test_real_sift_translation_and_crop_coordinates(self):
        transform = np.array([[1, 0, 12], [0, 1, 8], [0, 0, 1]], np.float64)
        target = cv2.warpPerspective(self.template, transform, (480, 360))
        boundary = self.build(target, crop=(40, 30, 480, 360), shape=(330, 440))
        expected = np.asarray(self.points) + [12 - 40, 8 - 30]
        np.testing.assert_allclose(boundary.points, expected, atol=1.5)
        registration = boundary.registration
        self.assertEqual(registration["method"], "sift_ransac")
        self.assertTrue(registration["is_valid"])
        self.assertGreaterEqual(registration["inliers"], 8)
        self.assertLess(registration["reprojection_error_px"], 1.5)

    def test_uncalibrated_does_not_pass_or_use_chip_expansion(self):
        self.calibration["template_points"] = []
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")
        mask = np.ones((360, 480), dtype=np.uint8)
        result = silver.inspect_masks([dict(class_name="silver", mask=mask), dict(class_name="chip", mask=mask)],
            mask.shape, dict(overflow=self.config), image=self.template, base_dir=self.root)["silver_overflow"]
        self.assertEqual(result["status"], "not_configured")
        self.assertIsNone(result["is_ok"])
        self.assertFalse(result["registration"]["is_valid"])
        self.assertNotIn("boundary", result)

    def test_missing_calibration_has_explicit_status(self):
        self.json_path.unlink()
        result = silver.inspect_masks([], (360, 480), dict(overflow=self.config),
                                      image=self.template, base_dir=self.root)["silver_overflow"]
        self.assertEqual(result["status"], "not_configured")
        self.assertIsNone(result["is_ok"])

    def test_featureless_target_is_uncertain(self):
        result = silver.inspect_masks([], (360, 480), dict(overflow=self.config),
                                      image=np.zeros_like(self.template), base_dir=self.root)["silver_overflow"]
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["is_ok"])
        self.assertFalse(result["registration"]["is_valid"])

    def enable_fallback(self, **overrides):
        self.calibration.update(fallback_to_template_points=True, **overrides)
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")

    def test_featureless_fallback_keeps_points_and_crop_offset(self):
        self.enable_fallback()
        boundary = self.build(np.zeros_like(self.template),
                              crop=(40, 30, 480, 360), shape=(330, 440))
        np.testing.assert_allclose(boundary.points, np.asarray(self.points) - [40, 30])
        self.assertEqual(boundary.registration["status"], "fallback")
        self.assertFalse(boundary.registration["is_valid"])
        self.assertTrue(boundary.registration["fallback_used"])
        np.testing.assert_equal(boundary.registration["homography_template_to_image"], np.eye(3))
        self.assertIn("insufficient features", boundary.registration["message"])
        public = silver.offset_geometry(dict(boundary=dict(points=boundary.points),
                    registration=boundary.registration), (40, 30))
        np.testing.assert_allclose(public["boundary"]["points"], self.points)
        self.assertTrue(public["registration"]["fallback_used"])

    def test_insufficient_matches_and_unreliable_registration_fall_back(self):
        for overrides, reason in [({"min_matches": 100000}, "matches are insufficient"),
                                  ({"min_inliers": 100000}, "registration is unreliable")]:
            with self.subTest(overrides=overrides):
                self.enable_fallback(min_matches=11, min_inliers=8)
                self.enable_fallback(**overrides)
                boundary = self.build()
                np.testing.assert_allclose(boundary.points, self.points)
                self.assertTrue(boundary.registration["fallback_used"])
                self.assertIn(reason, boundary.registration["message"])
                self.assertGreater(boundary.registration["matches"], 0)
                if "min_inliers" in overrides:
                    self.assertGreater(boundary.registration["inliers"], 0)
                    self.assertIn("inlier_ratio=", boundary.registration["message"])

    def test_fallback_continues_actual_overflow_measurement(self):
        self.enable_fallback()
        for outside in (False, True):
            with self.subTest(outside=outside):
                mask = np.zeros((360, 480), dtype=np.uint8)
                mask[120:130, 150:160] = 1
                if outside:
                    mask[120:130, 360:370] = 1
                result = silver.inspect_masks([dict(class_name="silver", mask=mask)],
                    mask.shape, dict(overflow=self.config),
                    image=np.zeros_like(self.template), base_dir=self.root)["silver_overflow"]
                self.assertEqual(result["status"], "overflow" if outside else "ok")
                self.assertEqual(result["is_ok"], not outside)
                self.assertEqual(result["measurement"]["defect_area_px"], 100 if outside else 0)
                self.assertTrue(result["registration"]["fallback_used"])

    def test_fallback_still_checks_template_and_boundary_integrity(self):
        self.enable_fallback(template_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "changed after calibration"):
            self.build(np.zeros_like(self.template))
        self.enable_fallback(template_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        boundary = self.build(np.zeros_like(self.template), crop=(300, 250, 480, 360), shape=(110, 180))
        np.testing.assert_allclose(boundary.points, np.asarray(self.points) - [300, 250])
        self.assertTrue(boundary.registration["fallback_used"])

    def test_shared_calibration_maps_each_target_independently(self):
        self.enable_fallback()
        original = self.json_path.read_bytes()
        with patch.object(silver, "_extract_sift_features",
                          wraps=silver._extract_sift_features) as extract, \
                patch.object(silver, "_decode_image", wraps=silver._decode_image) as decode:
            for dx, dy in [(12, 8), (-15, 10)]:
                target = cv2.warpPerspective(self.template, np.array(
                    [[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64), (480, 360))
                boundary = self.build(target)
                np.testing.assert_allclose(boundary.points,
                                           np.asarray(self.points) + [dx, dy], atol=1.5)
                self.assertFalse(boundary.registration["fallback_used"])
                self.assertTrue(boundary.registration["is_valid"])
        # One template extraction, plus a fresh extraction for each target.
        self.assertEqual(extract.call_count, 3)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(self.json_path.read_bytes(), original)

    def test_template_feature_cache_is_keyed_by_sift_parameters(self):
        with patch.object(silver, "_extract_sift_features",
                          wraps=silver._extract_sift_features) as extract:
            first_template = silver._get_template(self.path, self.calibration)
            first = silver._get_template_features(first_template)
            cached_template = silver._get_template(self.path, self.calibration)
            cached = silver._get_template_features(cached_template)
            resized_template = silver._get_template(
                self.path, {**self.calibration, "max_image_side": 240}
            )
            resized = silver._get_template_features(resized_template)
            fewer_template = silver._get_template(
                self.path, {**self.calibration, "max_features": 500}
            )
            silver._get_template_features(fewer_template)
        self.assertIs(first_template, cached_template)
        self.assertIs(first[0], cached[0])
        self.assertIsNot(first_template, resized_template)
        self.assertIsNot(first[0], resized[0])
        self.assertIsNot(first_template, fewer_template)
        self.assertEqual(extract.call_count, 3)

    def test_concurrent_template_cache_miss_builds_features_once(self):
        original_extract = silver._extract_sift_features

        def delayed_extract(gray, max_side, max_features):
            time.sleep(0.05)
            return original_extract(gray, max_side, max_features)

        with patch.object(silver, "_extract_sift_features", side_effect=delayed_extract) as extract:
            def load_features(_):
                template = silver._get_template(self.path, self.calibration)
                return silver._get_template_features(template)

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(
                    load_features,
                    range(8),
                ))
        self.assertEqual(extract.call_count, 1)
        self.assertTrue(all(features[0] is results[0][0] for features in results))
        self.assertFalse(results[0][0].flags.writeable)
        self.assertFalse(results[0][1].flags.writeable)

    def test_template_feature_errors_still_use_fallback_and_can_retry(self):
        self.enable_fallback()
        for error in (ValueError("template SIFT failed"), cv2.error("template SIFT failed")):
            with self.subTest(error=type(error).__name__):
                with patch.object(silver, "_extract_sift_features", side_effect=error):
                    boundary = self.build()
                self.assertTrue(boundary.registration["fallback_used"])
                self.assertIn("template SIFT failed", boundary.registration["message"])
                self.assertIsNone(silver._get_template(self.path, self.calibration).features)
        boundary = self.build()
        self.assertTrue(boundary.registration["is_valid"])
        self.assertFalse(boundary.registration["fallback_used"])

    def test_warm_cache_checks_modified_bytes_even_if_size_and_timestamp_match(self):
        # BMP pixels can change without changing the encoded file size.
        encoded = cv2.imencode(".bmp", self.template)[1].tobytes()
        self.path.write_bytes(encoded)
        self.enable_fallback(template_sha256=hashlib.sha256(encoded).hexdigest())
        self.build()
        original_stat = self.path.stat()
        self.path.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
        os.utime(self.path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertEqual(self.path.stat().st_size, original_stat.st_size)
        self.assertEqual(self.path.stat().st_mtime_ns, original_stat.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, "changed after calibration"):
            self.build(np.zeros_like(self.template))

    def test_warm_cache_never_falls_back_for_invalid_calibration(self):
        self.enable_fallback()
        self.build()
        self.enable_fallback(template_size=[100, 100])
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            self.build(np.zeros_like(self.template))
        self.enable_fallback(template_size=[480, 360], template_points=[
            [80, 90], [340, 270], [340, 90], [80, 270]
        ])
        with self.assertRaisesRegex(ValueError, "silver boundary"):
            self.build(np.zeros_like(self.template))

    def test_warm_cache_timings_only_recompute_target_features(self):
        cold, warm = {}, {}
        with timing_context(cold, phase="test"):
            self.build()
        with timing_context(warm, phase="test"):
            self.build()
        for stage in ("silver_template_decode_ms", "silver_template_gray_ms",
                      "silver_template_sift_ms"):
            self.assertIn(stage, cold)
            self.assertNotIn(stage, warm)
        self.assertIn("silver_target_sift_ms", warm)
        self.assertIn("silver_template_hash_ms", warm)

    def test_template_cache_is_bounded_and_can_rebuild_evicted_features(self):
        with patch.object(silver, "_TEMPLATE_FEATURE_CACHE_MAX_ENTRIES", 2):
            first = silver._get_template(self.path, self.calibration)
            silver._get_template_features(first)
            for features in (300, 400):
                silver._get_template(self.path, {**self.calibration, "max_features": features})
            self.assertLessEqual(len(silver._template_feature_cache), 2)
            rebuilt = silver._get_template(self.path, self.calibration)
            self.assertIsNot(first, rebuilt)
            np.testing.assert_array_equal(silver._get_template_features(first)[0],
                                          silver._get_template_features(rebuilt)[0])

    def test_cached_template_still_checks_hash_and_dimensions(self):
        silver._get_template(self.path, self.calibration)
        with self.assertRaisesRegex(ValueError, "changed after calibration"):
            silver._get_template(
                self.path, {**self.calibration, "template_sha256": "wrong"}
            )
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            silver._get_template(
                self.path, {**self.calibration, "template_size": [100, 100]}
            )

    def test_template_content_change_uses_a_new_cache_entry(self):
        calibration = {key: value for key, value in self.calibration.items()
                       if key != "template_sha256"}
        first = silver._get_template(self.path, calibration)
        changed = self.template.copy()
        cv2.circle(changed, (240, 180), 24, (0, 0, 255), -1)
        ok, encoded = cv2.imencode(".png", changed)
        self.assertTrue(ok)
        self.path.write_bytes(encoded.tobytes())
        second = silver._get_template(self.path, calibration)
        self.assertIsNot(first, second)
        self.assertNotEqual(first.sha256, second.sha256)

    def test_template_hash_and_size_are_checked(self):
        self.calibration["template_sha256"] = "bad-hash"
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after calibration"):
            self.build()
        self.calibration["template_size"] = [100, 100]
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            self.build()

    def test_boundary_validation_rejects_crossed_polygon(self):
        self.calibration["template_points"] = [[80, 90], [340, 270], [340, 90], [80, 270]]
        self.json_path.write_text(json.dumps(self.calibration), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.build()

    def test_out_of_crop_boundary_is_allowed_and_not_clipped(self):
        boundary = self.build(crop=(300, 250, 480, 360), shape=(110, 180))
        np.testing.assert_allclose(boundary.points, np.asarray(self.points) - [300, 250], atol=1)
        self.assertTrue(boundary.registration["is_valid"])

    def test_out_of_crop_polygon_still_measures_observed_overflow(self):
        boundary = self.build(crop=(300, 250, 480, 360), shape=(110, 180))
        mask = np.zeros((110, 180), dtype=bool)
        mask[5:15, 10:20] = True  # Within the full polygon, even though vertices are outside ROI.
        mask[5:15, 50:60] = True  # Outside the right boundary at local X=40.
        measurement = silver.measure_overflow(mask, boundary, silver.OverflowSettings())
        self.assertEqual(measurement.silver_area_px, 200)
        self.assertEqual(measurement.defect_area_px, 100)

    def test_overflow_pixels_contours_and_tolerance(self):
        boundary = silver.Boundary(((5, 5), (20, 5), (20, 20), (5, 20)))
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:12, 8:12] = True
        mask[8:12, 22:26] = True
        measured = silver.measure_overflow(mask, boundary, silver.OverflowSettings(tolerance_px=0))
        self.assertEqual(measured.defect_area_px, 16)
        self.assertTrue(measured.summary()["defect_contours"])
        tolerant = silver.measure_overflow(mask, boundary, silver.OverflowSettings(tolerance_px=5))
        self.assertEqual(tolerant.defect_area_px, 0)

    def test_offset_translates_only_geometry_once(self):
        registration = {"homography_template_to_image": [[1, 0, 12], [0, 1, 8], [0, 0, 1]],
                        "template_size": [480, 360], "matches": 53, "inliers": 50}
        diagnostics = dict(boundary=dict(mode="polygon", coordinate_system="crop", points=[[1, 2], [8, 2], [8, 9]]),
                           registration=registration,
                           measurement=dict(regions=[dict(box_xyxy=[10, 20, 30, 40])],
                                            defect_contours=[[[1, 1], [3, 1], [3, 3]]]))
        result = silver.offset_geometry(diagnostics, (100, 200))
        self.assertEqual(result["boundary"]["points"][0], [101, 202])
        self.assertEqual(result["boundary"]["coordinate_system"], "image")
        self.assertEqual(result["measurement"]["regions"][0]["box_xyxy"], [110, 220, 130, 240])
        self.assertEqual(result["measurement"]["defect_contours"][0][0], [101, 201])
        self.assertEqual(result["registration"], registration)
        self.assertEqual(diagnostics["boundary"]["points"][0], [1, 2])

    def test_installer_keeps_thresholds_backs_up_and_resolves_template(self):
        self.enable_fallback()
        config_file = self.root / "config.json"
        config = dict(silver_overflow_2d=dict(parameters=dict(
            boundary=dict(mode="chip_expansion"), tolerance_px=2.5, min_area_px=5)),
            unrelated_feature={"threshold": 9})
        config_file.write_text(json.dumps(config), encoding="utf-8")
        old_file = self.root / "calibration/silver_boundary.json"
        old_file.parent.mkdir()
        old_file.write_text('{"mode":"polygon","template_points":[]}', encoding="utf-8")
        target = calibrator.apply_calibration(self.calibration, self.path, config_file)
        self.assertTrue(target.is_file())
        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["template_points"], self.points)
        self.assertTrue(saved["fallback_to_template_points"])
        self.assertTrue((target.parent / saved["template_path"]).is_file())
        after = json.loads(config_file.read_text(encoding="utf-8"))
        self.assertEqual(after["unrelated_feature"], config["unrelated_feature"])
        self.assertEqual(after["silver_overflow_2d"]["parameters"]["tolerance_px"], 2.5)
        self.assertTrue(list(self.root.glob("config.json.*.bak")))
        self.assertTrue(list(target.parent.glob("silver_boundary.json.*.bak")))
        boundary = self.build(config=after["silver_overflow_2d"]["parameters"])
        np.testing.assert_allclose(boundary.points, self.points, atol=1)

    def test_installer_rejects_different_image_without_writes(self):
        config_file = self.root / "config.json"
        original = '{"silver_overflow_2d":{"parameters":{"boundary":{"mode":"chip_expansion"}}}}'
        config_file.write_text(original, encoding="utf-8")
        self.calibration["template_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "模板内容"):
            calibrator.apply_calibration(self.calibration, self.path, config_file)
        self.assertEqual(config_file.read_text(encoding="utf-8"), original)
        self.assertFalse((self.root / "calibration").exists())


if __name__ == "__main__":
    unittest.main()
