"""验证图像级指标、报告编码和跨图一致的热图显示范围。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ccd_report import compute_metrics, save_heatmap, write_report


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"path": "正常1.png", "label": 0, "score": 0.1, "defect_type": "normal"},
            {"path": "正常2.png", "label": 0, "score": 0.8, "defect_type": "normal"},
            {"path": "划伤.png", "label": 1, "score": 0.9, "defect_type": "划伤"},
            {"path": "污点.png", "label": 1, "score": 0.5, "defect_type": "污点", "pred_label": 1},
        ]

    def test_metrics_and_strict_threshold(self):
        # 分数等于阈值时应为正常；外部 pred_label 不影响统一判定。
        metrics = compute_metrics(self.rows, 0.5)
        self.assertEqual(metrics["counts"], {"total": 4, "normal": 2, "anomaly": 2})
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        for name in ["accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]:
            self.assertAlmostEqual(metrics[name], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertEqual(metrics["defect_type_recall"]["划伤"]["recall"], 1)
        self.assertEqual(metrics["defect_type_recall"]["污点"]["recall"], 0)
        self.assertNotIn("normal", metrics["defect_type_recall"])

    def test_normal_only_has_explicit_undefined_metrics(self):
        metrics = compute_metrics(self.rows[:1], 0.5)
        for name in ["roc_auc", "average_precision", "precision", "recall", "false_negative_rate", "f1"]:
            self.assertIsNone(metrics[name])
        self.assertEqual(metrics["false_positive_rate"], 0)
        self.assertTrue(any("一个类别" in note for note in metrics["notes"]))

    def test_anomaly_only_and_no_predicted_anomalies(self):
        metrics = compute_metrics(self.rows[2:], 1.0)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["false_positive_rate"])
        self.assertEqual(metrics["average_precision"], 1)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "为空"):
            compute_metrics([], 0.5)
        for score in [float("nan"), float("inf"), -float("inf"), "0.5"]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                compute_metrics([{**self.rows[0], "score": score}], 0.5)
        for label in [-1, 2, 0.5, "1", None, float("nan"), float("inf")]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                compute_metrics([{**self.rows[0], "label": label}], 0.5)
        with self.assertRaises(ValueError):
            compute_metrics(self.rows, float("nan"))

    def test_report_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            speed = {"image_count": len(self.rows), "model_and_score_average_ms": 2.5}
            metrics = write_report(
                folder,
                self.rows,
                0.5,
                {"source": "normal_validation", "quantile": np.float64(0.995)},
                speed=speed,
            )
            stored = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["confusion_matrix"], metrics["confusion_matrix"])
            self.assertEqual(stored["calibration"]["source"], "normal_validation")
            self.assertEqual(stored["inference_speed"], speed)
            self.assertTrue((folder / "predictions.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(predictions[0]["path"], "正常1.png")
            self.assertEqual(predictions[-1]["pred_label"], "0")
            with Image.open(folder / "score_distribution.png") as plot:
                self.assertGreater(plot.width, 500)


class HeatmapTests(unittest.TestCase):
    def test_fixed_scale_and_output_dimensions(self):
        import matplotlib.axes

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            original_path = folder / "原图.png"
            Image.new("RGB", (200, 100), "white").save(original_path)
            calls = []
            real_imshow = matplotlib.axes.Axes.imshow

            def capture(axis, array, *args, **kwargs):
                if "vmax" in kwargs:
                    calls.append((np.asarray(array).copy(), kwargs["vmin"], kwargs["vmax"]))
                return real_imshow(axis, array, *args, **kwargs)

            # 两张不同分数范围的图必须保持相同色阶，负分数只在显示时截断。
            with patch.object(matplotlib.axes.Axes, "imshow", new=capture):
                for index, values in enumerate([np.array([[-2.0, 0.2], [0.1, 9.0]]), np.full((2, 2), 0.5)]):
                    save_heatmap(original_path, values, folder / f"heatmap_{index}.png", display_max=2.0, score=0.4, threshold=0.5)
            # 每张 2×3 主图包含 4 次连续热图着色和 1 次二值图显示。
            self.assertEqual(len(calls), 10)
            color_calls = [call for call in calls if call[2] == 2.0]
            binary_calls = [call for call in calls if call[2] == 1]
            self.assertEqual(len(color_calls), 8)
            self.assertEqual(len(binary_calls), 2)
            for array, lower, upper in color_calls:
                self.assertEqual((lower, upper), (0, 2.0))
                self.assertGreaterEqual(array.min(), 0)
                self.assertLessEqual(array.max(), 2.0)
            self.assertTrue(any(np.allclose(array, 0.5) for array, _, _ in color_calls))
            with Image.open(folder / "heatmap_0.png") as output:
                self.assertEqual(output.width, 1500)

    def test_invalid_heatmaps(self):
        for values in [np.array([[np.nan]]), np.empty((0, 3)), np.zeros(3)]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                save_heatmap(Path("unused.png"), values, Path("unused_output.png"), display_max=1.0, score=0.4, threshold=0.5)
        with self.assertRaises(ValueError):
            save_heatmap(Path("unused.png"), np.zeros((2, 2)), Path("unused_output.png"), display_max=0, score=0.4, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
