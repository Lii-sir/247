"""验证暗区轮廓/RANSAC 圆检测及 Hough 回退。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import circle_mask


class RobustCircleDetectionTests(unittest.TestCase):
    @staticmethod
    def params(**overrides) -> dict:
        values = {
            "roi": [0.30, 0.15, 0.60, 0.75],
            "detection_method": "hybrid",
            "group_target": "best_score",
            "circle_target": "outer",
            "min_radius_ratio": 0.10,
            "max_radius_ratio": 0.40,
            "min_contour_score": 0.35,
            "require_circle_inside_roi": True,
            "ransac_seed": 7,
        }
        values.update(overrides)
        return values

    def test_dark_circle_is_fitted_despite_noise_and_distracting_line(self) -> None:
        image = np.full((400, 520, 3), 185, dtype=np.uint8)
        cv2.circle(image, (315, 205), 62, (25, 25, 25), thickness=-1)
        cv2.circle(image, (315, 205), 30, (5, 5, 5), thickness=-1)
        cv2.line(image, (200, 100), (480, 100), (35, 35, 35), thickness=3)
        rng = np.random.default_rng(12)
        noise = rng.normal(0, 6, image.shape[:2]).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)

        result = circle_mask.detect_circle(image, self.params())
        selected = result["selected"]

        self.assertEqual(result["detection"]["detector_used"], "dark_contour_ransac")
        self.assertLessEqual(abs(selected["center_x"] - 315), 2)
        self.assertLessEqual(abs(selected["center_y"] - 205), 2)
        self.assertLessEqual(abs(selected["radius"] - 62), 3)
        self.assertGreater(selected["candidate_score"], 0.75)
        self.assertGreater(selected["angular_coverage"], 0.85)
        self.assertGreater(selected["ransac_inlier_ratio"], 0.85)
        json.dumps(result)  # 检测记录必须能够直接保存为 JSON。

    def test_hybrid_uses_hough_only_when_contour_has_no_candidate(self) -> None:
        image = np.full((200, 300, 3), 180, dtype=np.uint8)
        hough_candidate = {
            "center_x": 70,
            "center_y": 60,
            "radius": 30,
            "edge_score": 80.0,
            "candidate_score": 0.6,
            "detector": "hough",
        }
        with patch.object(circle_mask, "_dark_contour_candidates", return_value=([], {})), \
                patch.object(circle_mask, "_hough_candidates", return_value=([hough_candidate], 0)):
            result = circle_mask.detect_circle(image, self.params())

        self.assertTrue(result["detection"]["hough_fallback_used"])
        self.assertEqual(result["detection"]["detector_used"], "hough")
        # Hough mock 使用 ROI 局部坐标，结果必须转换回原图坐标。
        self.assertEqual(result["selected"]["center_x"], 160)
        self.assertEqual(result["selected"]["center_y"], 90)

    def test_dark_contour_mode_does_not_silently_fall_back(self) -> None:
        image = np.full((200, 300, 3), 180, dtype=np.uint8)
        with patch.object(circle_mask, "_dark_contour_candidates", return_value=([], {})), \
                patch.object(circle_mask, "_hough_candidates") as hough:
            with self.assertRaisesRegex(ValueError, "未检测到圆候选"):
                circle_mask.detect_circle(
                    image, self.params(detection_method="dark_contour")
                )
        hough.assert_not_called()

    def test_black_ring_mode_finds_a_strongly_offset_inner_circle(self) -> None:
        image = np.full((420, 540, 3), 190, dtype=np.uint8)
        outer_center = (310, 215)
        inner_center = (344, 190)  # 与外圆明显偏心，不能依靠同心分组。
        cv2.circle(image, outer_center, 108, (25, 25, 25), thickness=-1)
        cv2.circle(image, inner_center, 43, (165, 165, 165), thickness=-1)
        # 模拟亮色工件局部遮挡黑环。
        cv2.rectangle(image, (329, 125), (340, 255), (175, 175, 175), thickness=-1)
        rng = np.random.default_rng(123)
        noise = rng.normal(0, 4, image.shape[:2]).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)

        result = circle_mask.detect_circle(
            image,
            self.params(
                detection_method="outer_inner_ring",
                min_radius_ratio=0.10,
                max_radius_ratio=0.42,
                inner_radius_min_ratio=0.25,
                inner_radius_max_ratio=0.60,
                black_ring_width_ratio=0.06,
                min_black_ring_coverage=0.55,
                min_inner_angular_coverage=0.35,
            ),
        )
        selected = result["selected"]
        outer = result["detection"]["outer_circle"]

        self.assertEqual(result["detection"]["detector_used"], "outer_inner_black_ring")
        self.assertLessEqual(abs(outer["center_x"] - outer_center[0]), 3)
        self.assertLessEqual(abs(outer["center_y"] - outer_center[1]), 3)
        self.assertLessEqual(abs(outer["radius"] - 108), 4)
        self.assertLessEqual(abs(selected["center_x"] - inner_center[0]), 4)
        self.assertLessEqual(abs(selected["center_y"] - inner_center[1]), 4)
        self.assertLessEqual(abs(selected["radius"] - 43), 4)
        self.assertGreater(selected["black_ring_coverage"], 0.70)
        self.assertEqual(selected["circle_target"], "inner_black_ring")


if __name__ == "__main__":
    unittest.main()
