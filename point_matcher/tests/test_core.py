from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from point_matcher.core import (
    MatchError, MatchResult, MatchSettings, TemplateMatcher, annotate_image,
    collect_images, export_csv, export_json, read_image, transform_points, write_image,
)


def textured_image(seed=7):
    rng = np.random.default_rng(seed)
    image = np.full((420, 601, 3), 236, np.uint8)
    for _ in range(180):
        x, y = int(rng.integers(15, 585)), int(rng.integers(15, 405))
        color = tuple(int(value) for value in rng.integers(0, 180, 3))
        cv.circle(image, (x, y), int(rng.integers(3, 15)), color, -1, cv.LINE_AA)
    for index in range(5):
        cv.putText(image, f"MATCH {seed}-{index} ABCD", (20, 55 + 78 * index), cv.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv.LINE_AA)
    return image


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(2)
        cls.template = textured_image()
        cls.points = [(50.25, 60.75), (299.5, 210.0), (530, 360)]
        cls.matrix = np.array([[1.05, 0.09, 95], [-0.05, 1.08, 80], [0.00014, -0.00008, 1]], np.float64)
        cls.target = cv.warpPerspective(cls.template, cls.matrix, (880, 680), borderValue=(245, 245, 245))

    def test_perspective_point_transfer(self):
        result = TemplateMatcher(self.template, self.points).match(self.target)
        actual = np.array([(point.target_x, point.target_y) for point in result.points])
        expected = transform_points(self.points, self.matrix)
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.inliers, 8)
        np.testing.assert_allclose(actual, expected, atol=1.5)

    def test_downscaling_preserves_original_pixel_coordinates(self):
        result = TemplateMatcher(self.template, self.points, MatchSettings(max_image_side=400)).match(self.target)
        actual = np.array([(point.target_x, point.target_y) for point in result.points])
        np.testing.assert_allclose(actual, transform_points(self.points, self.matrix), atol=2.0)

    def test_one_arbitrary_point_is_enough(self):
        result = TemplateMatcher(self.template, [(200, 200)]).match(self.template)
        self.assertEqual(len(result.points), 1)
        self.assertAlmostEqual(result.points[0].target_x, 200, places=3)

    def test_outside_points_are_not_clipped(self):
        matrix = np.array([[1, 0, -120], [0, 1, 35], [0, 0, 1]], np.float64)
        target = cv.warpPerspective(self.template, matrix, (650, 500))
        result = TemplateMatcher(self.template, [(20, 200), (400, 200)]).match(target)
        self.assertEqual(result.status, "partial")
        self.assertFalse(result.points[0].inside_image)
        self.assertLess(result.points[0].target_x, 0)
        self.assertTrue(result.points[1].inside_image)
        all_out = TemplateMatcher(self.template, [(20, 200)]).match(target)
        self.assertEqual(all_out.status, "outside")

    def test_empty_and_invalid_points(self):
        for points in ([], [(-1, 3)], [(601, 10)], [(2, 420)], [(float("nan"), 2)]):
            with self.subTest(points=points), self.assertRaises(MatchError):
                TemplateMatcher(self.template, points)

    def test_textureless_images_fail_cleanly(self):
        blank = np.full_like(self.template, 255)
        with self.assertRaises(MatchError):
            TemplateMatcher(blank, self.points)
        with self.assertRaises(MatchError):
            TemplateMatcher(self.template, self.points).match(blank)

    def test_invalid_homography_does_not_crash(self):
        matcher = TemplateMatcher(self.template, self.points)
        with patch("point_matcher.core.cv.findHomography", return_value=(None, None)):
            with self.assertRaisesRegex(MatchError, "透视变换"):
                matcher.match(self.template)

    def test_knn_pair_with_only_one_neighbor_is_ignored(self):
        matcher = TemplateMatcher(self.template, self.points)
        with patch("point_matcher.core.cv.FlannBasedMatcher") as flann:
            flann.return_value.knnMatch.return_value = [[cv.DMatch(0, 0, 0.0)]]
            with self.assertRaisesRegex(MatchError, "匹配点不足"):
                matcher.match(self.template)

    def test_singular_projection_rejected(self):
        matrix = np.eye(3)
        matrix[2] = 0
        with self.assertRaises(MatchError):
            transform_points(self.points, matrix)

    def test_unicode_paths_recursive_scan_and_broken_file(self):
        with tempfile.TemporaryDirectory(prefix="点匹配测试_") as directory:
            root = Path(directory)
            nested = root / "子目录"
            nested.mkdir()
            write_image(root / "模板.PNG", self.template)
            write_image(nested / "目标.png", self.target)
            (root / "损坏.jpg").write_bytes(b"not an image")
            (root / "说明.txt").write_text("ignored")
            self.assertEqual(len(collect_images(root)), 2)
            self.assertEqual(len(collect_images(root, recursive=True)), 3)
            np.testing.assert_array_equal(read_image(root / "模板.PNG"), self.template)
            matcher = TemplateMatcher(self.template, self.points)
            self.assertEqual(matcher.match_path(root / "损坏.jpg").status, "error")
            self.assertEqual(matcher.match_path(root / "不存在.png").status, "error")
            self.assertEqual(matcher.match_path(nested / "目标.png").status, "ok")

    def test_export_keeps_failed_images_and_homography(self):
        success = TemplateMatcher(self.template, self.points).match(self.target, "目标.png")
        failure = MatchResult("损坏.png", "error", "无法读取")
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "结果.csv"
            json_path = Path(directory) / "结果.json"
            export_csv(csv_path, [success, failure], self.points)
            export_json(json_path, "模板.png", self.points, [success, failure])
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(self.points) * 2)
            self.assertEqual(rows[-1]["target_x"], "")
            self.assertEqual(rows[-1]["status"], "error")
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["template_path"], "模板.png")
            self.assertEqual(len(data["results"][0]["homography"]), 3)

    def test_annotation_does_not_modify_original(self):
        result = TemplateMatcher(self.template, self.points).match(self.target)
        original = self.target.copy()
        annotated = annotate_image(self.target, result)
        np.testing.assert_array_equal(original, self.target)
        self.assertFalse(np.array_equal(annotated, original))

    def test_invalid_settings(self):
        for kwargs in ({"ratio_threshold": 1}, {"min_matches": 3}, {"ransac_threshold": float("nan")}, {"max_image_side": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MatchSettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
