"""CPU-only threshold tests; no checkpoint loading or real detector execution."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
chip = import_module("algorithm.2D.chip_anomaly")
import detect_and_view as client


def config(threshold=None):
    return {"device": "cpu", "chip_class": "chip", "score_mode": "pool+top",
            "score_pool_kernel": 7, "threshold": threshold,
            "aspect_ratio_models": [{"range": [0.9, 1.1], "model": "chip_short"},
                                    {"range": [1.3, 1.7], "model": "chip_long"}]}


def fake_predictor(name, threshold):
    predictor = chip._LoadedPredictor.__new__(chip._LoadedPredictor)
    predictor.name = name
    predictor.category = name
    predictor.image_size = 16
    predictor.device = torch.device("cpu")
    predictor._lock = threading.RLock()
    predictor.config = {"score_topk_ratio": 0.001, "localization": {}}
    predictor.calibration = {"threshold": threshold, "score_method": {
        "name": chip.MULTISCALE_SCORE_METHOD, "pool_kernels": [1, 7, 21],
        "normalization": {str(k): {"median": 0, "denominator": 1} for k in (1, 7, 21)}}}
    predictor.model = lambda batch: SimpleNamespace(anomaly_map=torch.full((1, 1, 16, 16), 0.5))
    return predictor


def instances():
    short, long = np.zeros((48, 64), np.uint8), np.zeros((48, 64), np.uint8)
    short[5:25, 5:25] = 255
    long[5:35, 32:52] = 255
    return [{"instance_id": 1, "class_name": "chip", "is_ok": True, "mask": short},
            {"instance_id": 2, "class_name": "chip", "is_ok": True, "mask": long}]


def inspector(threshold=None):
    value = chip.ChipAnomalyInspector(config(threshold), {})
    value.predictors = {"chip_short": fake_predictor("chip_short", 0.25),
                        "chip_long": fake_predictor("chip_long", 0.75)}
    return value


class ThresholdTests(unittest.TestCase):
    def test_independent_defaults(self):
        value = inspector()
        result = value.inspect(np.zeros((48, 64, 3), np.uint8), instances(), [0, 0, 64, 48])
        short, long = result["chips"]
        self.assertEqual(short["threshold"], 0.25)
        self.assertEqual(long["threshold"], 0.75)
        self.assertFalse(short["is_ok"])
        self.assertTrue(long["is_ok"])

    def test_model_specific_configuration(self):
        value = inspector({"chip_short": 0.6, "chip_long": 0.4})
        result = value.inspect(np.zeros((48, 64, 3), np.uint8), instances(), [0, 0, 64, 48])
        self.assertEqual([r["threshold"] for r in result["chips"]], [0.6, 0.4])
        self.assertEqual([r["is_ok"] for r in result["chips"]], [True, False])
        self.assertEqual([r["threshold_source"] for r in result["chips"]], ["runtime_threshold"] * 2)

    def test_partial_request_preserves_other_model(self):
        value = inspector({"chip_short": 0.25, "chip_long": 0.75})
        before = copy.deepcopy(value.config)
        result = value.inspect(np.zeros((48, 64, 3), np.uint8), instances(), [0, 0, 64, 48], threshold={"chip_short": 0.6})
        self.assertEqual([r["threshold"] for r in result["chips"]], [0.6, 0.75])
        self.assertEqual(value.config, before)

    def test_sparse_request_uses_checkpoint_for_unspecified_model(self):
        value = inspector()
        result = value.inspect(np.zeros((48, 64, 3), np.uint8), instances(), [0, 0, 64, 48], threshold={"chip_short": 0.6})
        self.assertEqual([r["threshold"] for r in result["chips"]], [0.6, 0.75])

    def test_threshold_controls_boxes_and_verdict_together(self):
        value = fake_predictor("chip_short", 0.25)
        low, low_artifacts = value.predict(np.zeros((20, 20, 3), np.uint8), score_mode="pool+top", score_pool_kernel=7, threshold=0.2)
        high, high_artifacts = value.predict(np.zeros((20, 20, 3), np.uint8), score_mode="pool+top", score_pool_kernel=7, threshold=0.8)
        self.assertEqual(low["score"], high["score"])
        self.assertFalse(low["is_ok"])
        self.assertTrue(high["is_ok"])
        self.assertTrue(low_artifacts["boxes"])
        self.assertEqual(high_artifacts["boxes"], [])

    def test_threshold_zero_is_not_treated_as_missing(self):
        result, _ = fake_predictor("chip_short", 0.25).predict(np.zeros((20, 20, 3), np.uint8), score_mode="pool+top", threshold=0)
        self.assertEqual(result["threshold"], 0)
        self.assertFalse(result["is_ok"])

    def test_score_equal_to_threshold_is_ok(self):
        result, artifacts = fake_predictor("chip_short", 0.25).predict(np.zeros((20, 20, 3), np.uint8), score_mode="pool+top", threshold=0.5)
        self.assertTrue(result["is_ok"])
        self.assertEqual(artifacts["boxes"], [])

    def test_invalid_threshold_values(self):
        for threshold in (-1, float("nan"), float("inf"), True, {"chip_short": None}):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                chip.validate_config(config(threshold))

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            chip.validate_config(config({"wrong_model": 0.5}))

    def test_validate_config_idempotent(self):
        first = chip.validate_config(config({"chip_short": 0.4, "chip_long": 2}))
        self.assertEqual(chip.validate_config(first), first)

    def test_cli_named_values(self):
        self.assertEqual(client.parse_threshold("chip_short=0.4,chip_long=2.0"), {"chip_short": 0.4, "chip_long": 2.0})
        self.assertEqual(client.parse_threshold("0"), 0)
        for text in ("nan", "inf", "-1", "chip_short=-1", "chip_short=0.4,chip_short=1"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                client.parse_threshold(text)

    def test_cli_parameters_keep_scoring_config_and_send_threshold_per_request(self):
        request = client.algorithm_request("pool+top", 7)
        parameters = json.loads(request["config_text"])["chip_anomaly_2d"]["parameters"]
        self.assertNotIn("threshold", parameters)
        self.assertEqual(parameters["score_mode"], "pool+top")
        threshold = {"chip_short": 0.4, "chip_long": 2.0}
        self.assertEqual(client.threshold_request(threshold), {"threshold": threshold})
        self.assertEqual(client.threshold_request(None), {})

    def test_stream_gpu_phase_receives_this_piece_threshold(self):
        module = import_module("algorithm.detect_board")
        detector = module.DetectBoard(lambda text: None)
        detector.model = object()
        detector._synchronize_model = Mock()
        detector._predict_batch = Mock(return_value=[object()])
        detector._extract_instances = Mock(return_value=[])
        detector.chip_anomaly_inspector = SimpleNamespace(inspect=Mock(return_value={"is_ok": True}))
        image = np.zeros((48, 64, 3), np.uint8)
        override = {"chip_short": 0.4, "chip_long": 2.0}
        prepared = {"crop": image, "image": image, "crop_box": (0, 0, 64, 48), "threshold": override}
        detector.infer_2d_piece(prepared)
        self.assertEqual(detector.chip_anomaly_inspector.inspect.call_args.kwargs["threshold"], override)

    def test_prepare_preserves_threshold_without_changing_detector_config(self):
        module = import_module("algorithm.detect_board")
        detector = module.DetectBoard(lambda text: None)
        image = np.zeros((48, 64, 3), np.uint8)
        detector._read_2d = Mock(return_value=image)
        detector._stream_crop = Mock(return_value=(image, (0, 0, 64, 48), (20, 20, 5), []))
        before = copy.deepcopy(detector.chip_anomaly_config)
        with patch.object(module, "platform_mask_2d", return_value=np.zeros((48, 64), np.uint8)):
            first = detector.prepare_2d_piece({"board_id": "A", "piece_id": 1, "threshold": {"chip_short": 0.4}})
            second = detector.prepare_2d_piece({"board_id": "B", "piece_id": 1, "threshold": {"chip_short": 0.8}})
        self.assertEqual(first["threshold"], {"chip_short": 0.4})
        self.assertEqual(second["threshold"], {"chip_short": 0.8})
        self.assertEqual(detector.chip_anomaly_config, before)


if __name__ == "__main__":
    unittest.main()
