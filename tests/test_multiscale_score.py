"""验证多尺度整图分数的校准、融合和分类型阈值。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import efficientad_ccd as cli
from circle_mask import multiscale_topk_scores


class MultiscaleScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # efficientad_ccd 延迟加载 NumPy；这些纯计算测试不需要加载 anomalib。
        cli.np = np

    def test_pooling_scales_respond_differently_to_small_peak(self) -> None:
        anomaly_map = torch.zeros((1, 1, 9, 9))
        anomaly_map[0, 0, 4, 4] = 9.0
        mask = torch.zeros_like(anomaly_map, dtype=torch.bool)
        scores = multiscale_topk_scores(
            anomaly_map, mask, pool_kernels=(1, 3), topk_ratio=1 / 81
        )
        self.assertAlmostEqual(float(scores["1"][0]), 9.0)
        self.assertAlmostEqual(float(scores["3"][0]), 1.0)

    def test_top_mode_uses_masked_single_pixel_maximum(self) -> None:
        anomaly_map = torch.zeros((1, 1, 5, 5))
        anomaly_map[0, 0, 2, 2] = 9.0
        anomaly_map[0, 0, 0, 0] = 20.0
        ignore_mask = torch.zeros((1, 5, 5), dtype=torch.bool)
        ignore_mask[0, 0, 0] = True
        prediction = SimpleNamespace(anomaly_map=anomaly_map)
        batch = SimpleNamespace(ignore_mask=ignore_mask)

        result = cli.prediction_scores(
            prediction, batch, {"device": "cpu"},
            {"score_method": {"name": cli.TOP_SCORE_METHOD}},
        )
        self.assertEqual(result["score"], 9.0)
        self.assertEqual(result["score_max"], 9.0)

    def test_legacy_checkpoint_without_score_method_uses_top_mode(self) -> None:
        anomaly_map = torch.zeros((1, 1, 5, 5))
        anomaly_map[0, 0, 2, 2] = 9.0
        prediction = SimpleNamespace(anomaly_map=anomaly_map)
        batch = SimpleNamespace(ignore_mask=torch.zeros((1, 5, 5), dtype=torch.bool))

        result = cli.prediction_scores(
            prediction, batch, {"device": "cpu", "score_pool_kernel": 3}, {}
        )
        self.assertEqual(result["score"], 9.0)

    def test_pool_topk_mode_uses_saved_kernel_and_ratio(self) -> None:
        anomaly_map = torch.zeros((1, 1, 5, 5))
        anomaly_map[0, 0, 2, 2] = 9.0
        prediction = SimpleNamespace(anomaly_map=anomaly_map)
        batch = SimpleNamespace(ignore_mask=torch.zeros((1, 5, 5), dtype=torch.bool))
        method = {
            "name": cli.SINGLE_SCALE_SCORE_METHOD,
            "pool_kernel": 3,
            "topk_ratio": 1 / 25,
        }

        result = cli.prediction_scores(
            prediction, batch,
            {"device": "cpu", "score_pool_kernel": 1, "score_topk_ratio": 1.0},
            {"score_method": method},
        )
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(result["score_max"], 9.0)

    def test_score_mode_aliases_are_parsed(self) -> None:
        parser = cli.build_parser()
        cases = {
            "top": cli.SCORE_MODE_TOP,
            "pool+top": cli.SCORE_MODE_POOL_TOPK,
            "pool_topk": cli.SCORE_MODE_POOL_TOPK,
            "multiscale_pool": cli.SCORE_MODE_MULTISCALE,
            "checkpoint": cli.SCORE_MODE_CHECKPOINT,
        }
        for argument, expected in cases.items():
            with self.subTest(argument=argument):
                args = parser.parse_args([
                    "evaluate", "--checkpoint", "model.pt", "--score-mode", argument,
                ])
                self.assertEqual(args.score_mode, expected)

    def test_normalization_and_max_fusion(self) -> None:
        normal_rows = [
            {"score_kernel_1": 1.0, "score_kernel_7": 0.10, "score_kernel_21": 0.01},
            {"score_kernel_1": 2.0, "score_kernel_7": 0.20, "score_kernel_21": 0.02},
            {"score_kernel_1": 3.0, "score_kernel_7": 0.30, "score_kernel_21": 0.03},
        ]
        config = {"score_pool_kernels": [1, 7, 21], "score_topk_ratio": 0.001}
        method = cli.build_score_method(normal_rows, config)
        result = cli.fuse_prediction_scores({
            "score_max": 9.0,
            "score_kernel_1": 2.0,
            "score_kernel_7": 0.20,
            "score_kernel_21": 0.05,
        }, method)
        self.assertEqual(result["score_normalized_kernel_1"], 0.0)
        self.assertEqual(result["score_normalized_kernel_7"], 0.0)
        self.assertGreater(result["score_normalized_kernel_21"], 1.0)
        self.assertEqual(result["score"], result["score_normalized_kernel_21"])

    def test_threshold_constrains_every_defect_type(self) -> None:
        rows = [
            {"label": 0, "score": 0.5, "defect_type": "good"},
            {"label": 0, "score": 1.0, "defect_type": "good"},
            {"label": 1, "score": 3.0, "defect_type": "small"},
            {"label": 1, "score": 2.0, "defect_type": "small"},
            {"label": 1, "score": 1.5, "defect_type": "large"},
            {"label": 1, "score": 1.2, "defect_type": "large"},
        ]
        threshold, stats = cli.threshold_for_target_recall(rows, 1.0)
        self.assertLess(threshold, 1.2)
        self.assertEqual(stats["per_defect"]["small"]["achieved_recall"], 1.0)
        self.assertEqual(stats["per_defect"]["large"]["achieved_recall"], 1.0)
        self.assertFalse(stats["degenerate_zero_boundary"])

    def test_zero_anomaly_boundary_is_marked_degenerate(self) -> None:
        rows = [
            {"label": 0, "score": 0.0, "defect_type": "good"},
            {"label": 1, "score": 0.0, "defect_type": "hard_defect"},
        ]
        threshold, stats = cli.threshold_for_target_recall(rows, 1.0)
        self.assertLess(threshold, 0.0)
        self.assertTrue(stats["degenerate_zero_boundary"])

    def test_inference_uses_calibration_kernels_not_runtime_config(self) -> None:
        method = {
            "name": cli.MULTISCALE_SCORE_METHOD,
            "pool_kernels": [3],
            "topk_ratio": 0.25,
            "normalization": {"3": {"median": 1.0, "q99": 2.0, "denominator": 1.0}},
        }
        raw = {"score_max": 4.0, "score_kernel_3": 3.0}
        with patch.object(cli, "raw_prediction_scores", return_value=raw) as compute:
            result = cli.prediction_scores(
                SimpleNamespace(), SimpleNamespace(),
                {"device": "cpu", "score_pool_kernels": [1, 7, 21]},
                {"score_method": method},
            )
        self.assertEqual(result["score"], 2.0)
        self.assertEqual(compute.call_args.kwargs["pool_kernels"], [3])
        self.assertEqual(compute.call_args.kwargs["topk_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()
