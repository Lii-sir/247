"""单应性方向、异常配对、真实 SIFT 链路、掩膜和结果持久化回归。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2 as cv
import numpy as np

from part_segmentation.image_io import read_image, write_image
from part_segmentation.models import Segment, SegmentationResult
from point_matcher.tests.test_core import textured_image
from segmentation_transfer.calibration_io import load_calibration, save_calibration
from segmentation_transfer.export import export_result
from segmentation_transfer.geometry import (
    compose_image_mapping, fit_template_mapping, normalize_homography, project_points,
    validate_domain, warp_binary_mask,
)
from segmentation_transfer.models import Calibration, MappingSettings
from segmentation_transfer.pipeline import TransferPipeline, transfer_instances


POINTS = ((50, 50), (500, 50), (510, 340), (50, 350), (220, 160), (380, 250), (150, 280), (450, 120))


def source_result(path=Path("a1.png"), image=None):
    if image is None:
        image = np.full((100, 150, 3), 128, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[20:80, 30:120] = True
    mask[40:60, 50:90] = False
    return SegmentationResult(path, image, (Segment(2, "chip", 0.91, (30, 20, 120, 80), mask),), 0)


class GeometryTests(unittest.TestCase):
    def test_crop_offset_chain_against_exact_pixel_placement(self):
        # A 是 B 从 (400, 250) 起的裁剪；A1/B1 又分别发生不同平移。
        # 独立手算：(x,y) -> (x-10,y-20) -> (+400,+250) -> (+30,+40)。
        ha = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=float)
        hab = np.array([[1, 0, 400], [0, 1, 250], [0, 0, 1]], dtype=float)
        hb = np.array([[1, 0, 30], [0, 1, 40], [0, 0, 1]], dtype=float)
        actual_h = compose_image_mapping(ha, hab, hb)
        np.testing.assert_allclose(actual_h, [[1, 0, 420], [0, 1, 270], [0, 0, 1]], atol=1e-10)
        source = source_result()
        actual = transfer_instances(source, actual_h, (600, 900))[0]
        expected = np.zeros((600, 900), dtype=bool)
        # 不调用 warp/project 辅助函数生成预期掩膜，直接放到已知像素区域。
        expected[270:370, 420:570] = source.segments[0].mask
        np.testing.assert_array_equal(actual.mask, expected)
        self.assertEqual(actual.box, (450, 290, 540, 350))
        self.assertEqual(actual.status, "ok")

    def test_anisotropic_stretch_and_rotation_against_hand_calculation(self):
        ha = np.array([[2, 0, 10], [0, 3, 15], [0, 0, 1]], dtype=float)
        hab = np.array([[1.5, 0, 100], [0, .5, 80], [0, 0, 1]], dtype=float)
        hb = np.array([[0, -1, 800], [1, 0, 40], [0, 0, 1]], dtype=float)
        a1 = np.array([[10, 15], [90, 165], [210, 315]], dtype=float)
        expected = np.column_stack((722.5 - a1[:, 1] / 6, .75 * a1[:, 0] + 132.5))
        for scales in ((1, 1, 1), (-.25, 8, .002)):
            with self.subTest(scales=scales):
                matrix = compose_image_mapping(ha * scales[0], hab * scales[1], hb * scales[2])
                actual = cv.perspectiveTransform(a1.reshape(-1, 1, 2), matrix).reshape(-1, 2)
                np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_projective_mask_against_independent_inverse_sampling(self):
        # 直接从目标像素反算源坐标作最近邻采样，独立检查 warpPerspective 的方向。
        source = source_result()
        matrix = np.array([[1.07, .06, 12.3], [-.04, .93, 18.7], [.0004, -.0002, 1]])
        height, width = 170, 230
        y, x = np.indices((height, width))
        homogeneous = np.stack((x.ravel(), y.ravel(), np.ones(x.size)))
        source_coordinates = np.linalg.inv(matrix) @ homogeneous
        sx, sy = np.rint(source_coordinates[:2] / source_coordinates[2]).astype(int)
        valid = (sx >= 0) & (sx < source.image.shape[1]) & (sy >= 0) & (sy < source.image.shape[0])
        expected = np.zeros(x.size, dtype=bool)
        expected[valid] = source.segments[0].mask[sy[valid], sx[valid]]
        actual = warp_binary_mask(source.segments[0].mask, matrix, (height, width))
        np.testing.assert_array_equal(actual, expected.reshape(height, width))

    def test_composition_order_and_inverse(self):
        ha = np.array([[1.1, .04, 28], [.03, .9, 15], [.0001, -.0001, 1]])
        hab = np.array([[.8, -.08, 10], [.05, 1.05, 8], [-.0001, .0002, 1]])
        hb = np.array([[1.02, .07, 90], [-.03, .98, 55], [.0002, .0001, 1]])
        points_a1 = project_points(POINTS, ha)
        actual = project_points(points_a1, compose_image_mapping(ha, hab, hb))
        expected = project_points(project_points(POINTS, hab), hb)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_fit_and_outlier(self):
        h = np.array([[1, .04, 12], [.02, .95, 10], [.0001, 0, 1]])
        target = project_points(POINTS, h)
        target[-1] = [570, 380]
        fit = fit_template_mapping(POINTS, target, (420, 601), (420, 601))
        self.assertEqual(int(fit.inliers.sum()), 7)
        self.assertFalse(fit.inliers[-1])
        np.testing.assert_allclose(project_points(POINTS[:-1], fit.matrix), target[:-1], atol=.001)

    def test_four_points(self):
        fit = fit_template_mapping(POINTS[:4], POINTS[:4], (420, 601), (420, 601))
        np.testing.assert_allclose(fit.matrix, np.eye(3), atol=1e-8)

    def test_invalid_correspondences(self):
        bad = [POINTS[:3], [(10, 10)] * 4, [(10, 10), (20, 20), (30, 30), (40, 40)],
               [(-1, 10), *POINTS[1:4]], [(float("nan"), 10), *POINTS[1:4]],
               [(602, 10), *POINTS[1:4]], [(5, 5), (5, 6), (6, 5), (6, 6)]]
        for points in bad:
            with self.subTest(points=points), self.assertRaises(ValueError):
                fit_template_mapping(points, points, (420, 601), (420, 601))
        with self.assertRaisesRegex(ValueError, "相等"):
            fit_template_mapping(POINTS, POINTS[:4], (420, 601), (420, 601))

    def test_inconsistent_pairing_rejected(self):
        rng = np.random.default_rng(18)
        a = rng.uniform([10, 10], [590, 410], (20, 2))
        b = rng.uniform([10, 10], [590, 410], (20, 2))
        with self.assertRaisesRegex(ValueError, "不可靠"):
            fit_template_mapping(a, b, (420, 601), (420, 601), MappingSettings(min_inlier_ratio=.8))

    def test_invalid_matrices_and_horizon(self):
        for matrix in (np.zeros((3, 3)), np.ones((3, 3)), np.eye(2), np.eye(3) * np.nan):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                normalize_homography(matrix)
        horizon = np.array([[1, 0, 0], [0, 1, 0], [-.02, 0, 1]])
        with self.assertRaisesRegex(ValueError, "无穷远"):
            validate_domain(horizon, (100, 100))
        with self.assertRaises(ValueError):
            project_points([[50, 20]], horizon)
        np.testing.assert_allclose(normalize_homography(-3 * np.eye(3)), np.eye(3))

    def test_mask_preserves_hole_and_direction(self):
        source = source_result()
        h = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float)
        original = source.segments[0].mask.copy()
        result = transfer_instances(source, h, (120, 180, 3))[0]
        self.assertEqual(result.box, (40, 25, 130, 85))
        self.assertEqual(result.area, source.segments[0].area)
        self.assertTrue(result.mask[30, 45])
        self.assertFalse(result.mask[50, 70])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.source_confidence, .91)
        np.testing.assert_array_equal(original, source.segments[0].mask)

    def test_clip_empty_and_stable_ids(self):
        source = source_result()
        source = replace(source, segments=(source.segments[0], source.segments[0]))
        h = np.array([[1, 0, 90], [0, 1, 0], [0, 0, 1]], dtype=float)
        items = transfer_instances(source, h, source.image.shape)
        self.assertEqual([i.source_id for i in items], [1, 2])
        self.assertEqual(items[0].status, "clipped")
        self.assertLess(items[0].source_coverage, 1)
        h[0, 2] = 500
        empty = transfer_instances(source, h, source.image.shape)[0]
        self.assertEqual(empty.status, "empty")
        self.assertIsNone(empty.box)
        self.assertEqual(empty.area, 0)
        self.assertEqual(empty.source_coverage, 0)

    def test_empty_source_and_bad_masks(self):
        source = replace(source_result(), segments=())
        self.assertEqual(transfer_instances(source, np.eye(3), (100, 150)), ())
        with self.assertRaises(ValueError):
            warp_binary_mask(np.ones((5, 5), dtype=np.uint8), np.eye(3), (10, 10))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "中文输入"
        self.a = textured_image(7)
        self.b = textured_image(19)  # A/B 不同纹理，不能直接拿 A 去匹配 B。
        self.hab = np.array([[.9, .03, 10], [.02, .9, 12], [.0001, 0, 1]])
        self.ha = np.array([[1.03, .03, 35], [-.02, .98, 40], [.00008, -.00003, 1]])
        self.hb = np.array([[.97, -.03, 60], [.02, 1.04, 50], [-.00004, .00007, 1]])
        self.a1 = cv.warpPerspective(self.a, self.ha, (760, 560))
        self.b1 = cv.warpPerspective(self.b, self.hb, (800, 580))
        for name, image in (("A.png", self.a), ("B.png", self.b), ("A1.png", self.a1), ("B1.png", self.b1)):
            write_image(self.inputs / name, image)
        self.calibration = Calibration(self.inputs / "A.png", self.inputs / "B.png", POINTS,
                                       tuple(map(tuple, project_points(POINTS, self.hab))))
        self.segmenter = Mock()
        mask = np.zeros(self.a1.shape[:2], dtype=bool)
        mask[150:320, 160:430] = True
        mask[210:260, 250:330] = False
        self.source = SegmentationResult(self.inputs / "A1.png", self.a1,
                                         (Segment(0, "bond", .95, (160, 150, 430, 320), mask),), 0)
        self.segmenter.predict.return_value = self.source

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self):
        return TransferPipeline(self.calibration).run(self.inputs / "A1.png", self.inputs / "B1.png", self.segmenter)

    def test_real_sift_two_templates_and_single_segmentation(self):
        result = self.run_pipeline()
        self.segmenter.predict.assert_called_once()
        self.assertEqual(self.segmenter.predict.call_args.args[0], self.inputs / "A1.png")
        # 预期值从已知生成矩阵独立计算，不调用被测组合函数。
        expected_h = self.hb @ self.hab @ np.linalg.inv(self.ha)
        probes = np.array([[100, 100], [250, 180], [450, 300]])
        np.testing.assert_allclose(project_points(probes, result.matrix_a1_to_b1),
                                   project_points(probes, expected_h), atol=1.5)
        expected_mask = warp_binary_mask(self.source.segments[0].mask, expected_h, self.b1.shape)
        actual = result.instances[0].mask
        iou = np.count_nonzero(actual & expected_mask) / np.count_nonzero(actual | expected_mask)
        self.assertGreater(iou, .97)
        self.assertLess(float(result.point_errors_b1.max()), .01)

    def test_identical_inputs_reduce_to_calibration_same_paths_and_copies(self):
        for copies in (False, True):
            with self.subTest(copies=copies):
                a_path = self.inputs / ("A1.png" if copies else "A.png")
                b_path = self.inputs / ("B1.png" if copies else "B.png")
                write_image(a_path, self.a)
                write_image(b_path, self.b)
                self.segmenter.predict.return_value = source_result(a_path, self.a)
                with patch("segmentation_transfer.registration.TemplateMatcher") as matcher:
                    pipeline = TransferPipeline(self.calibration)
                    result = pipeline.run(a_path, b_path, self.segmenter)
                    matcher.assert_not_called()
                np.testing.assert_array_equal(result.match_a.homography, np.eye(3))
                np.testing.assert_array_equal(result.match_b.homography, np.eye(3))
                np.testing.assert_allclose(result.matrix_a1_to_b1, result.fit.matrix, rtol=0, atol=1e-12)
                # 未参与标定的点由独立 OpenCV 投影检查。
                probes = np.array([[[90., 80.]], [[320., 210.]], [[470., 310.]]])
                np.testing.assert_allclose(cv.perspectiveTransform(probes, result.matrix_a1_to_b1),
                                           cv.perspectiveTransform(probes, self.hab), atol=1e-4)
                if copies:
                    output = export_result(self.root / "identity-result", result)
                    document = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
                    self.assertEqual(document["match_a"]["method"], "identity")
                    self.assertEqual(document["match_b"]["inliers"], 0)

    def test_one_identity_leg_and_one_real_feature_leg(self):
        for identity_a in (True, False):
            with self.subTest(identity_a=identity_a):
                a_path = self.inputs / ("A.png" if identity_a else "A1.png")
                b_path = self.inputs / ("B1.png" if identity_a else "B.png")
                self.segmenter.predict.return_value = source_result(a_path, self.a if identity_a else self.a1)
                result = TransferPipeline(self.calibration).run(a_path, b_path, self.segmenter)
                self.assertEqual(result.match_a.method, "identity" if identity_a else "features")
                self.assertEqual(result.match_b.method, "features" if identity_a else "identity")
                expected = self.hb @ self.hab if identity_a else self.hab @ np.linalg.inv(self.ha)
                probes = np.array([[[100., 100.]], [[250., 180.]], [[450., 300.]]])
                np.testing.assert_allclose(cv.perspectiveTransform(probes, result.matrix_a1_to_b1),
                                           cv.perspectiveTransform(probes, expected), atol=1.5)

    def test_real_sift_a_is_crop_of_b_with_different_image_sizes(self):
        # A 为 B 的真实子图；A1 和 B1 分别旋转/缩放/透视，尺寸彼此不同。
        b = textured_image(23)
        offset_x, offset_y = 100, 80
        a = b[offset_y:350, offset_x:500].copy()
        ha = np.array([[1.04, .04, 35], [-.03, 1.01, 45], [.00008, -.00004, 1]])
        hb = np.array([[.96, -.035, 65], [.04, 1.02, 40], [-.00006, .00008, 1]])
        a1 = cv.warpPerspective(a, ha, (510, 410))
        b1 = cv.warpPerspective(b, hb, (780, 590))
        for name, image in (("A.png", a), ("B.png", b), ("A1.png", a1), ("B1.png", b1)):
            write_image(self.inputs / name, image)
        pa = np.array([[25, 25], [370, 25], [370, 240], [25, 240], [150, 100], [270, 170]], dtype=float)
        pb = pa + [offset_x, offset_y]
        calibration = Calibration(self.inputs / "A.png", self.inputs / "B.png", tuple(map(tuple, pa)), tuple(map(tuple, pb)))
        template_mask = np.zeros(a.shape[:2], dtype=np.uint8)
        template_mask[60:210, 80:310] = 1
        template_mask[100:160, 140:220] = 0
        source_mask = cv.warpPerspective(template_mask, ha, (510, 410), flags=cv.INTER_NEAREST).astype(bool)
        self.segmenter.predict.return_value = SegmentationResult(self.inputs / "A1.png", a1,
            (Segment(0, "bond", .95, (0, 0, 510, 410), source_mask),), 0)
        result = TransferPipeline(calibration).run(self.inputs / "A1.png", self.inputs / "B1.png", self.segmenter)

        # 使用未参与标定的检查点，预期 B1 坐标不通过被测链式矩阵生成。
        probes_a = np.array([[[70., 70.]], [[185., 125.]], [[315., 205.]]])
        probes_a1 = cv.perspectiveTransform(probes_a, ha)
        expected_b1 = cv.perspectiveTransform(probes_a + [offset_x, offset_y], hb)
        actual_b1 = cv.perspectiveTransform(probes_a1, result.matrix_a1_to_b1)
        np.testing.assert_allclose(actual_b1, expected_b1, atol=1.5)
        b_mask = np.zeros(b.shape[:2], dtype=np.uint8)
        b_mask[offset_y:350, offset_x:500] = template_mask
        expected_mask = cv.warpPerspective(b_mask, hb, (780, 590), flags=cv.INTER_NEAREST).astype(bool)
        actual_mask = result.instances[0].mask
        iou = np.count_nonzero(actual_mask & expected_mask) / np.count_nonzero(actual_mask | expected_mask)
        self.assertGreater(iou, .97)
        self.assertEqual(actual_mask.shape, b1.shape[:2])
        self.segmenter.predict.assert_called_once()

    def test_matching_failure_stops_before_yolo(self):
        write_image(self.inputs / "B1.png", np.full_like(self.b1, 128))
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.segmenter.predict.assert_not_called()

    def test_model_image_mismatch_rejected(self):
        self.segmenter.predict.return_value = replace(self.source, image_path=self.inputs / "B1.png")
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.run_pipeline()

    def test_calibration_round_trip_and_relative_paths(self):
        path = self.root / "标定" / "pair.json"
        save_calibration(path, self.calibration)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(data["template_a"].startswith("../"))
        loaded = load_calibration(path)
        self.assertEqual(loaded, self.calibration)
        data["points_b"] = [[0, 0]]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_calibration(path)

    def test_circle_metadata_validation_and_legacy_load(self):
        path = self.root / "calibration.json"
        save_calibration(path, self.calibration)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("radii_a", document)
        self.assertEqual(load_calibration(path).radii_a, ())
        for radii in ([1], [None] * 7 + [-1], [None] * 7 + [float("nan")], "bad", None):
            with self.subTest(radii=radii):
                invalid = dict(document, radii_a=radii)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_calibration(path)
        calibration = replace(self.calibration, radii_a=(8.25,) + (None,) * 7,
                              radii_b=(12.75,) + (None,) * 7)
        save_calibration(path, calibration)
        self.assertEqual(load_calibration(path), calibration)


    def test_export_masks_metadata_and_no_overwrite(self):
        result = self.run_pipeline()
        output = export_result(self.root / "outputs" / "run1", result)
        document = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
        self.assertEqual(document["instances"][0]["source_id"], 1)
        self.assertEqual(document["instances"][0]["source_confidence"], .95)
        mask = read_image(output / document["instances"][0]["mask"])
        self.assertEqual(mask.shape[:2], self.b1.shape[:2])
        np.testing.assert_array_equal(mask[:, :, 0] > 0, result.instances[0].mask)
        np.testing.assert_allclose(document["h_a1_to_b1"], result.matrix_a1_to_b1)
        with self.assertRaisesRegex(ValueError, "已存在"):
            export_result(output, result)
        self.assertFalse(list(output.parent.glob(".transfer-*")))


if __name__ == "__main__":
    unittest.main()
