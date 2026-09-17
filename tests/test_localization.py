"""验证 Score 对应空间响应、mask、阈值和连通域框。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ccd_localization import (
    build_localization,
    localization_summary,
    normalize_localization_params,
)
from ccd_report import save_heatmap
from circle_mask import mask_aware_average_pool, pooled_topk_score


TEST_PARAMS = {
    "min_area_ratio": 0.0,
    "morph_kernel": 1,
    "open_iterations": 0,
    "close_iterations": 0,
    "merge_iou": 0.15,
    "merge_containment": 0.80,
    "merge_distance_ratio": 0.0,
    "padding_ratio": 0.0,
    "fallback_size_ratio": 0.1,
    "normalized_display_max": 3.0,
}


class LocalizationTests(unittest.TestCase):
    def test_top_uses_raw_map_threshold_and_strict_image_gate(self) -> None:
        anomaly = torch.zeros((1, 1, 8, 8))
        anomaly[0, 0, 2:4, 3:5] = 2.0
        mask = torch.zeros_like(anomaly, dtype=torch.bool)
        result = build_localization(
            anomaly, mask, score_mode="top", score_method={},
            score_values={"score": 2.0}, threshold=1.0, raw_display_max=2.0,
            params=TEST_PARAMS,
        )
        self.assertTrue(np.array_equal(result["binary_map"], result["raw_map"] > 1.0))
        self.assertEqual(len(result["boxes"]), 1)
        self.assertEqual(
            {key: result["boxes"][0][key] for key in ("x0", "y0", "x1", "y1")},
            {"x0": 3, "y0": 2, "x1": 5, "y1": 4},
        )

        normal = build_localization(
            anomaly, mask, score_mode="top", score_method={},
            score_values={"score": 1.0}, threshold=1.0, raw_display_max=2.0,
            params=TEST_PARAMS,
        )
        self.assertFalse(normal["binary_map"].any())
        self.assertEqual(normal["boxes"], [])

    def test_pool_localization_reuses_exact_mask_aware_pool(self) -> None:
        anomaly = torch.zeros((1, 1, 9, 9))
        anomaly[0, 0, 4, 4] = 90.0  # 被 mask 的极端值不能泄漏到池化边缘。
        anomaly[0, 0, 1:4, 1:4] = 9.0
        mask = torch.zeros_like(anomaly, dtype=torch.bool)
        mask[0, 0, 4, 4] = True
        expected_map, _ = mask_aware_average_pool(anomaly, mask, pool_kernel=3)
        score = float(pooled_topk_score(
            anomaly, mask, pool_kernel=3, topk_ratio=1 / 80
        )[0])
        original_anomaly = anomaly.clone()
        result = build_localization(
            anomaly, mask, score_mode="pool_topk",
            score_method={"pool_kernel": 3, "topk_ratio": 1 / 80},
            score_values={"score": score}, threshold=1.0, raw_display_max=10.0,
            params=TEST_PARAMS,
        )
        self.assertTrue(np.allclose(result["response_map"], expected_map.squeeze().numpy()))
        self.assertEqual(result["response_map"][4, 4], 0.0)
        self.assertLessEqual(result["response_map"].max(), 9.0)
        self.assertGreater(len(result["boxes"]), 0)
        self.assertTrue(torch.equal(anomaly, original_anomaly))

    def test_two_dimensional_mask_expands_without_leaking_masked_value(self) -> None:
        anomaly = torch.zeros((1, 1, 7, 7))
        anomaly[0, 0, 3, 3] = 100.0
        mask = torch.zeros((7, 7), dtype=torch.bool)
        mask[3, 3] = True
        pooled, valid = mask_aware_average_pool(anomaly, mask, pool_kernel=3)
        self.assertEqual(tuple(valid.shape), tuple(anomaly.shape))
        self.assertFalse(bool(valid[0, 0, 3, 3]))
        self.assertEqual(float(pooled.max()), 0.0)

    def test_negative_boundary_is_clamped_for_top_and_pool_localization(self) -> None:
        anomaly = torch.zeros((1, 1, 8, 8))
        anomaly[0, 0, 5, 5] = 0.5
        mask = torch.zeros_like(anomaly, dtype=torch.bool)
        for mode, method in (
            ("top", {}),
            ("pool_topk", {"pool_kernel": 3, "topk_ratio": 0.01}),
        ):
            with self.subTest(mode=mode):
                result = build_localization(
                    anomaly, mask, score_mode=mode, score_method=method,
                    score_values={"score": 0.0}, threshold=-5e-324,
                    raw_display_max=1.0, params=TEST_PARAMS,
                )
                self.assertEqual(result["response_threshold"], 0.0)
                self.assertTrue(result["threshold_adjusted_for_localization"])
                self.assertLess(int(result["binary_map"].sum()), anomaly.numel())

    def test_flat_fallback_uses_valid_region_center_instead_of_first_pixel(self) -> None:
        anomaly = torch.zeros((1, 1, 9, 9))
        mask = torch.zeros_like(anomaly, dtype=torch.bool)
        mask[:, :, :2, :] = True
        result = build_localization(
            anomaly, mask, score_mode="top", score_method={},
            score_values={"score": 0.0}, threshold=-5e-324,
            raw_display_max=1.0, params=TEST_PARAMS,
        )
        box = result["boxes"][0]
        self.assertEqual(box["fallback_position"], "valid_region_center")
        self.assertGreater(box["x0"], 0)
        self.assertGreater(box["y0"], 1)

    def test_multiscale_uses_only_active_scales_and_merges_boxes(self) -> None:
        anomaly = torch.zeros((1, 1, 12, 12))
        anomaly[0, 0, 4:8, 4:8] = 4.0
        mask = torch.zeros_like(anomaly, dtype=torch.bool)
        method = {
            "pool_kernels": [1, 3],
            "topk_ratio": 0.01,
            "normalization": {
                "1": {"median": 0.0, "q99": 1.0, "denominator": 1.0},
                "3": {"median": 0.0, "q99": 1.0, "denominator": 1.0},
            },
        }
        result = build_localization(
            anomaly, mask, score_mode="multiscale_pool", score_method=method,
            score_values={
                "score": 3.0,
                "score_kernel_1": 4.0, "score_normalized_kernel_1": 3.0,
                "score_kernel_3": 2.0, "score_normalized_kernel_3": 2.0,
            },
            threshold=1.0, raw_display_max=5.0, params=TEST_PARAMS,
        )
        self.assertEqual([scale["active"] for scale in result["scales"]], [True, True])
        self.assertTrue(np.allclose(
            result["response_map"],
            np.maximum(result["scales"][0]["normalized_map"], result["scales"][1]["normalized_map"]),
        ))
        self.assertEqual(len(result["boxes"]), 1)
        self.assertEqual(result["boxes"][0]["source_scales"], [1, 3])

        only_first = build_localization(
            anomaly, mask, score_mode="multiscale_pool", score_method=method,
            score_values={
                "score": 3.0,
                "score_kernel_1": 4.0, "score_normalized_kernel_1": 3.0,
                "score_kernel_3": 2.0, "score_normalized_kernel_3": 0.5,
            },
            threshold=1.0, raw_display_max=5.0, params=TEST_PARAMS,
        )
        self.assertEqual([scale["active"] for scale in only_first["scales"]], [True, False])
        self.assertTrue(np.allclose(
            only_first["response_map"], only_first["scales"][0]["normalized_map"]
        ))

    def test_nested_multiscale_boxes_merge_by_containment(self) -> None:
        anomaly = torch.zeros((1, 1, 31, 31))
        anomaly[0, 0, 14:17, 14:17] = 8.0
        method = {
            "pool_kernels": [1, 15], "topk_ratio": 0.001,
            "normalization": {
                "1": {"median": 0.0, "q99": 1.0, "denominator": 1.0},
                "15": {"median": 0.0, "q99": 0.01, "denominator": 0.01},
            },
        }
        result = build_localization(
            anomaly, torch.zeros_like(anomaly, dtype=torch.bool),
            score_mode="multiscale_pool", score_method=method,
            score_values={
                "score": 2.0,
                "score_kernel_1": 8.0, "score_normalized_kernel_1": 8.0,
                "score_kernel_15": 0.2, "score_normalized_kernel_15": 2.0,
            },
            threshold=1.0, raw_display_max=8.0, params=TEST_PARAMS,
        )
        self.assertEqual(len(result["scales"][0]["boxes"]), 1)
        self.assertEqual(len(result["scales"][1]["boxes"]), 1)
        self.assertEqual(len(result["boxes"]), 1)
        self.assertEqual(result["boxes"][0]["source_scales"], [1, 15])

    def test_multiscale_visualization_saves_main_and_scale_diagnostic(self) -> None:
        anomaly = torch.zeros((1, 1, 12, 12))
        anomaly[0, 0, 4:8, 4:8] = 4.0
        method = {
            "pool_kernels": [1, 3, 5], "topk_ratio": 0.01,
            "normalization": {
                str(kernel): {"median": 0.0, "q99": 1.0, "denominator": 1.0}
                for kernel in (1, 3, 5)
            },
        }
        scores = {"score": 4.0}
        for kernel in (1, 3, 5):
            scores[f"score_kernel_{kernel}"] = 4.0
            scores[f"score_normalized_kernel_{kernel}"] = 4.0
        localization = build_localization(
            anomaly, torch.zeros_like(anomaly, dtype=torch.bool),
            score_mode="multiscale_pool", score_method=method,
            score_values=scores, threshold=1.0, raw_display_max=5.0,
            params=TEST_PARAMS,
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image_path = folder / "image.png"
            Image.new("RGB", (120, 80), "white").save(image_path)
            main_path, detail_path = folder / "main.png", folder / "scales.png"
            save_heatmap(
                image_path, anomaly.squeeze().numpy(), main_path,
                display_max=5.0, score=4.0, threshold=1.0,
                localization=localization, multiscale_output_path=detail_path,
            )
            self.assertTrue(main_path.is_file())
            self.assertTrue(detail_path.is_file())
            with Image.open(main_path) as main:
                self.assertEqual(main.width, 1500)
            with Image.open(detail_path) as detail:
                self.assertEqual(detail.width, 1500)

    def test_zero_boundary_does_not_turn_entire_multiscale_map_positive(self) -> None:
        anomaly = torch.zeros((1, 1, 8, 8))
        anomaly[0, 0, 6, 6] = 0.5
        method = {
            "pool_kernels": [1], "topk_ratio": 0.01,
            "normalization": {
                "1": {"median": 1.0, "q99": 2.0, "denominator": 1.0},
            },
        }
        result = build_localization(
            anomaly, torch.zeros_like(anomaly, dtype=torch.bool),
            score_mode="multiscale_pool", score_method=method,
            score_values={
                "score": 0.0, "score_kernel_1": 0.0,
                "score_normalized_kernel_1": 0.0,
            },
            threshold=-5e-324, raw_display_max=0.1, params=TEST_PARAMS,
        )
        self.assertEqual(result["response_threshold"], 0.0)
        self.assertTrue(result["threshold_adjusted_for_localization"])
        self.assertFalse(result["binary_map"].any())
        self.assertEqual(len(result["boxes"]), 1)
        self.assertTrue(result["boxes"][0]["fallback"])
        self.assertEqual(result["boxes"][0]["fallback_source"], "raw_anomaly_map")
        self.assertGreater(result["boxes"][0]["x0"], 0)
        self.assertGreater(result["boxes"][0]["y0"], 0)
        # 定位摘要必须能直接进入 prediction.json。
        json.dumps(localization_summary(result))

    def test_localization_parameter_validation(self) -> None:
        for invalid in (
            {"morph_kernel": 2}, {"open_iterations": -1},
            {"min_area_ratio": -0.1}, {"merge_containment": 1.1},
            {"normalized_display_max": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_localization_params(invalid)


if __name__ == "__main__":
    unittest.main()
