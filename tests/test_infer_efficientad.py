"""独立推理入口的离线测试；随机权重仅验证链路，不代表实际检测效果。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ccd_efficientad import cli
from ccd_efficientad import inference


class InputTests(unittest.TestCase):
    def test_parser_accepts_single_image_and_rejects_conflicting_inputs(self):
        parser = inference.build_parser()
        args = parser.parse_args(["--checkpoint", "model.pt", "--image", "图.bmp"])
        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.mask)
        self.assertIsNone(args.threshold)
        self.assertEqual(args.score_mode, "checkpoint")
        self.assertIsNone(args.score_pool_kernel)
        self.assertIsNone(args.score_topk_ratio)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--checkpoint", "model.pt", "--image", "a.png", "--image-dir", "."])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--checkpoint", "model.pt", "--image", "a.png",
                               "--mask", "a.png", "--circle-config", "config.json"])

    def test_parser_accepts_finite_thresholds_and_rejects_invalid_values(self):
        parser = inference.build_parser()
        arguments = ["--checkpoint", "model.pt", "--image", "a.png"]
        for value in ("0", "0.5", "2.5", "-0.25", "1e3"):
            with self.subTest(value=value):
                args = parser.parse_args([*arguments, f"--threshold={value}"])
                self.assertEqual(args.threshold, float(value))
        for value in ("nan", "inf", "-inf", "1e309", "not-a-number"):
            with self.subTest(value=value), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parser.parse_args([*arguments, f"--threshold={value}"])
            self.assertEqual(error.exception.code, 2)

    def test_api_rejects_invalid_threshold_before_loading_runtime(self):
        with patch.object(cli, "load_runtime") as runtime:
            for value in (float("nan"), float("inf"), -float("inf"), "invalid"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "threshold"):
                    inference.EfficientAdPredictor("missing.pt", threshold=value)
            runtime.assert_not_called()

    def test_parser_accepts_score_modes_aliases_and_pool_parameters(self):
        parser = inference.build_parser()
        arguments = ["--checkpoint", "model.pt", "--image", "a.png"]
        for value, expected in (("checkpoint", "checkpoint"), ("saved", "checkpoint"),
                                ("top", "top"), ("max", "top"), ("pool+top", "pool_topk"),
                                ("pool_topk", "pool_topk"), ("multiscale_pool", "multiscale_pool"),
                                ("multi-pool", "multiscale_pool")):
            with self.subTest(value=value):
                args = parser.parse_args([*arguments, "--score-mode", value])
                self.assertEqual(args.score_mode, expected)
        args = parser.parse_args([*arguments, "--score-mode", "pool+top",
                                  "--score-pool-kernel", "7", "--score-topk-ratio", "0.01"])
        self.assertEqual(args.score_pool_kernel, 7)
        self.assertEqual(args.score_topk_ratio, 0.01)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args([*arguments, "--score-mode", "unknown"])
        with patch.object(cli, "load_runtime") as runtime, self.assertRaisesRegex(ValueError, "score-mode"):
            inference.EfficientAdPredictor("missing.pt", score_mode="unknown")
        runtime.assert_not_called()

    def test_collect_images_excludes_results_and_keeps_duplicate_stems(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "results").mkdir()
            for name in ("same.png", "same.BMP", "nested/same.png", "results/old.png"):
                Image.new("L", (8, 8)).save(root / name)
            (root / "notes.txt").write_text("not an image")
            images = inference.collect_images(root, root / "results")
            self.assertEqual({path.relative_to(root).as_posix() for path in images},
                             {"same.png", "same.BMP", "nested/same.png"})
            for output in (root, root.parent):
                with self.assertRaisesRegex(ValueError, "不能等于或包含"):
                    inference.collect_images(root, output)

    def test_empty_or_missing_image_directory_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "没有支持的图片"):
                inference.collect_images(root, root / "out")
            with self.assertRaisesRegex(ValueError, "不存在"):
                inference.collect_images(root / "missing", root / "out")

    def test_missing_checkpoint_fails_before_loading_runtime(self):
        with TemporaryDirectory() as directory, patch.object(cli, "load_runtime") as runtime:
            with self.assertRaises(FileNotFoundError):
                inference.EfficientAdPredictor(Path(directory) / "missing.pt")
            runtime.assert_not_called()

    def test_invalid_input_fails_before_creating_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                inference.main(["--checkpoint", "missing.pt", "--image", str(root / "missing.png"),
                                "--output-dir", str(root / "out")])
            self.assertFalse((root / "out").exists())


class OfflineInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cli.load_runtime()
        cls.old_thread_count = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "推理测试"
        cls.root.mkdir()
        cls.image = cls.root / "原图.png"
        Image.fromarray(np.random.default_rng(5).integers(0, 256, (45, 71, 3), dtype=np.uint8)).save(cls.image)
        cls.config = dict(imagenette_dir="missing-training-data", model_size="small",
                          lr=1e-4, weight_decay=1e-5, device="cpu", num_workers=0,
                          backbone="resnet18_layer2", image_size=256,
                          resnet_architecture_version=2, resnet_feature_mode="valid")
        cls.model = cli.new_model(cls.config).eval()
        cls.model.model.mean_std["std"].data.fill_(1)
        for key, value in {"qa_st": 0., "qb_st": 1., "qa_ae": 0., "qb_ae": 1.}.items():
            cls.model.model.quantiles[key].data.fill_(value)
        cls.calibration = dict(threshold=0.0, display_max=1.0,
                               score_method={"name": cli.TOP_SCORE_METHOD})
        cls.checkpoint = cls.root / "model.pt"
        cli.save_checkpoint(cls.checkpoint, cls.model, cls.config, {"category": "CCD1"},
                            step=1, calibration=cls.calibration)
        cls.predictor = inference.EfficientAdPredictor(cls.checkpoint, "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_thread_count)

    def save_checkpoint(self, name, *, config=None, calibration=None):
        path = self.root / name
        cli.save_checkpoint(path, self.model, config or self.config, {"category": "CCD1"},
                            step=1, calibration=calibration)
        return path

    def test_predict_matches_existing_pipeline_and_restores_geometry(self):
        result = self.predictor.predict(self.image)
        batch, _ = self.predictor._prepare_batch(self.image)
        with torch.inference_mode():
            prediction = self.predictor.model.model(batch.image)
        reference = cli.prediction_scores(prediction, batch, self.predictor.config, self.calibration)
        self.assertEqual(result["score"], reference["score"])
        self.assertEqual(result["prediction"], "NG")
        self.assertEqual(result["anomaly_map_shape"], [256, 256])
        self.assertEqual(result["image_size_original"], {"width": 71, "height": 45})
        self.assertFalse(self.predictor.model.training)
        self.assertEqual(self.predictor.config["resnet_feature_mode"], "valid")
        self.assertTrue(result["boxes_original"])
        for box in result["boxes_original"]:
            self.assertTrue(0 <= box["x0"] < box["x1"] <= 71)
            self.assertTrue(0 <= box["y0"] < box["y1"] <= 45)

    def test_score_equal_to_threshold_is_ok(self):
        score = self.predictor.predict(self.image)["score"]
        predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=score)
        result = predictor.predict(self.image)
        self.assertEqual(result["prediction"], "OK")
        self.assertEqual(result["threshold"], score)
        self.assertFalse(result["boxes_original"])

    def test_threshold_override_keeps_checkpoint_and_loaded_calibration_unchanged(self):
        original_bytes = self.checkpoint.read_bytes()
        original_calibration = dict(self.calibration)
        with patch.object(cli, "restore_for_inference", return_value=(
            self.model, self.config, {"calibration": self.calibration},
        )):
            predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=2.5)
        self.assertEqual(predictor.threshold, 2.5)
        self.assertEqual(predictor.calibration["threshold"], 2.5)
        self.assertEqual(self.calibration, original_calibration)
        self.assertEqual(self.checkpoint.read_bytes(), original_bytes)
        self.assertEqual(self.predictor.threshold, original_calibration["threshold"])

    def test_threshold_override_controls_classification_localization_and_heatmaps(self):
        score = self.predictor.predict(self.image)["score"]
        for threshold, expected in ((score + 1, "OK"), (0.0, "NG"), (-0.25, "NG")):
            with self.subTest(threshold=threshold):
                predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=threshold)
                output = self.root / f"override-{threshold}"
                with patch.object(cli, "prediction_localization", wraps=cli.prediction_localization) as localize, \
                        patch("ccd_efficientad.report.save_heatmap") as heatmap:
                    result = predictor.predict(self.image, output)
                self.assertEqual(result["score"], score)
                self.assertEqual(result["threshold"], threshold)
                self.assertEqual(result["prediction"], expected)
                self.assertEqual(localize.call_args.args[3]["threshold"], threshold)
                self.assertEqual(heatmap.call_args.kwargs["threshold"], threshold)
                saved = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["threshold"], threshold)
                with Image.open(output / "anomaly_mask.png") as mask:
                    if expected == "OK":
                        self.assertFalse(result["boxes_original"])
                        self.assertEqual(mask.getextrema(), (0, 0))
                    else:
                        self.assertTrue(result["boxes_original"])
                        self.assertEqual(mask.getextrema()[1], 255)

    def test_saved_single_scale_and_multiscale_scores_are_reproduced(self):
        methods = [
            {"name": cli.SINGLE_SCALE_SCORE_METHOD, "pool_kernel": 3, "topk_ratio": 0.01},
            {"name": cli.MULTISCALE_SCORE_METHOD, "pool_kernels": [1, 3], "topk_ratio": 0.01,
             "normalization": {"1": {"median": 0., "denominator": 0.1},
                               "3": {"median": 0., "denominator": 0.2}}},
        ]
        for index, method in enumerate(methods):
            with self.subTest(method=method["name"]):
                calibration = {**self.calibration, "score_method": method}
                checkpoint = self.save_checkpoint(f"score-{index}.pt", calibration=calibration)
                predictor = inference.EfficientAdPredictor(checkpoint, "cpu")
                result = predictor.predict(self.image)
                batch, _ = predictor._prepare_batch(self.image)
                with torch.inference_mode():
                    prediction = predictor.model.model(batch.image)
                expected = cli.prediction_scores(prediction, batch, predictor.config, calibration)
                self.assertEqual(result["score"], expected["score"])
                self.assertEqual(result["threshold"], calibration["threshold"])
                if index == 1:
                    self.assertIn("score_normalized_kernel_3", result)
                overridden = inference.EfficientAdPredictor(checkpoint, "cpu", threshold=result["score"] + 1)
                overridden_result = overridden.predict(self.image)
                self.assertEqual(overridden_result["score"], result["score"])
                self.assertEqual(overridden_result["prediction"], "OK")
                self.assertFalse(overridden_result["boxes_original"])

    def test_uncalibrated_checkpoint_is_rejected(self):
        checkpoint = self.save_checkpoint("last.pt", calibration=None)
        with self.assertRaisesRegex(ValueError, "尚未校准"):
            inference.EfficientAdPredictor(checkpoint, "cpu")

    def test_explicit_same_score_mode_preserves_saved_threshold_and_method(self):
        method = {"name": cli.SINGLE_SCALE_SCORE_METHOD, "pool_kernel": 3, "topk_ratio": 0.01}
        calibration = {**self.calibration, "threshold": 2.5, "score_method": method}
        checkpoint = self.save_checkpoint("same-mode.pt", calibration=calibration)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu", score_mode="pool+top")
        self.assertEqual(predictor.threshold, 2.5)
        self.assertEqual(predictor.calibration["score_method"]["pool_kernel"], 3)
        self.assertEqual(predictor.calibration["score_method"]["topk_ratio"], 0.01)
        self.assertNotIn("inference_score_override", predictor.calibration)
        same = inference.EfficientAdPredictor(checkpoint, "cpu", score_mode="pool_topk",
                                            score_pool_kernel=3, score_topk_ratio=0.01)
        self.assertEqual(same.threshold, 2.5)

    def test_switching_score_formula_requires_threshold_and_missing_norm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "必须同时指定 --threshold"):
            inference.EfficientAdPredictor(self.checkpoint, "cpu", score_mode="pool+top")
        with self.assertRaisesRegex(ValueError, "归一化参数"):
            inference.EfficientAdPredictor(self.checkpoint, "cpu", score_mode="multiscale_pool", threshold=0.5)
        checkpoint = self.save_checkpoint("pool-mode.pt", calibration={**self.calibration, "score_method": {
            "name": cli.SINGLE_SCALE_SCORE_METHOD, "pool_kernel": 3, "topk_ratio": 0.01,
        }})
        for kwargs in ({"score_mode": "top"}, {"score_mode": "pool+top", "score_pool_kernel": 7},
                       {"score_mode": "pool+top", "score_topk_ratio": 0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "必须同时指定 --threshold"):
                inference.EfficientAdPredictor(checkpoint, "cpu", **kwargs)

    def test_pool_switch_uses_config_defaults_and_explicit_parameters(self):
        predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", score_mode="pool+top", threshold=0)
        self.assertEqual(predictor.calibration["score_method"]["pool_kernel"], 21)
        self.assertEqual(predictor.calibration["score_method"]["topk_ratio"], 0.001)
        config = {**self.config, "score_pool_kernel": 7, "score_topk_ratio": 0.02}
        checkpoint = self.save_checkpoint("pool-config.pt", config=config, calibration=self.calibration)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu", score_mode="pool+top", threshold=0)
        self.assertEqual(predictor.calibration["score_method"]["pool_kernel"], 7)
        self.assertEqual(predictor.calibration["score_method"]["topk_ratio"], 0.02)
        result = predictor.predict(self.image)
        batch, _ = predictor._prepare_batch(self.image)
        with torch.inference_mode():
            prediction = predictor.model.model(batch.image)
        expected = cli.prediction_scores(prediction, batch, predictor.config, {"score_method": {
            "name": cli.SINGLE_SCALE_SCORE_METHOD, "pool_kernel": 7, "topk_ratio": 0.02,
        }})
        self.assertEqual(result["score"], expected["score"])

    def test_pool_overrides_validate_parameters_and_reject_conflicting_modes(self):
        invalid = [{"score_pool_kernel": value} for value in (0, -3, 2, 3.5, True)]
        invalid += [{"score_topk_ratio": value} for value in (0, -0.1, 1.1, float("nan"), float("inf"))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=0.5, score_mode="pool+top", **kwargs)
        for mode in ("checkpoint", "top", "multiscale_pool"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "仅可与"):
                inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=0.5,
                                              score_mode=mode, score_pool_kernel=3)

    def test_switch_from_multiscale_uses_requested_formula_for_scores_and_localization(self):
        method = {"name": cli.MULTISCALE_SCORE_METHOD, "pool_kernels": [1, 3], "topk_ratio": 0.01,
                  "normalization": {"1": {"median": 0., "denominator": 0.1},
                                    "3": {"median": 0., "denominator": 0.2}}}
        calibration = {**self.calibration, "score_method": method}
        checkpoint = self.save_checkpoint("switch-mode.pt", calibration=calibration)
        original_bytes = checkpoint.read_bytes()
        for mode, kwargs in (("top", {}), ("pool+top", {"score_pool_kernel": 3, "score_topk_ratio": 0.01}),
                             ("multiscale_pool", {})):
            with self.subTest(mode=mode):
                predictor = inference.EfficientAdPredictor(checkpoint, "cpu", score_mode=mode, threshold=0, **kwargs)
                batch, _ = predictor._prepare_batch(self.image)
                with torch.inference_mode():
                    prediction = predictor.model.model(batch.image)
                expected_method = (cli.build_score_method([], self.config, "top") if mode == "top" else
                                   {"name": cli.SINGLE_SCALE_SCORE_METHOD, "pool_kernel": 3, "topk_ratio": 0.01}
                                   if mode == "pool+top" else method)
                expected = cli.prediction_scores(prediction, batch, predictor.config, {"score_method": expected_method})
                with patch.object(cli, "prediction_localization", wraps=cli.prediction_localization) as localize:
                    result = predictor.predict(self.image)
                self.assertEqual(result["score"], expected["score"])
                self.assertEqual(result["score_mode"], cli.normalize_score_mode(mode))
                self.assertEqual(result["localization"]["score_mode"], result["score_mode"])
                self.assertEqual(localize.call_args.args[3]["score_method"]["name"], expected_method["name"])
                self.assertEqual(result["threshold"], 0)
                self.assertEqual(result["score_method"]["name"], expected_method["name"])
        self.assertEqual(checkpoint.read_bytes(), original_bytes)
        self.assertEqual(calibration["score_method"], method)

    def test_explicit_multiscale_rejects_invalid_saved_normalization(self):
        methods = [
            {"pool_kernels": [1], "topk_ratio": 0.01, "normalization": {}},
            {"pool_kernels": [], "topk_ratio": 0.01, "normalization": {}},
            {"pool_kernels": [1], "topk_ratio": float("nan"),
             "normalization": {"1": {"median": 0., "denominator": 1.}}},
        ]
        methods += [{"pool_kernels": [1], "topk_ratio": 0.01,
                     "normalization": {"1": {"median": 0., "denominator": value}}}
                    for value in (0, -1, float("inf"))]
        for index, method in enumerate(methods):
            checkpoint = self.save_checkpoint(f"invalid-norm-{index}.pt", calibration={
                **self.calibration, "score_method": {"name": cli.MULTISCALE_SCORE_METHOD, **method},
            })
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "归一化参数"):
                inference.EfficientAdPredictor(checkpoint, "cpu", score_mode="multiscale_pool")

    def test_score_override_does_not_modify_loaded_config_or_calibration(self):
        original_config = dict(self.config)
        original_calibration = dict(self.calibration)
        with patch.object(cli, "restore_for_inference", return_value=(
            self.model, self.config, {"calibration": self.calibration},
        )):
            predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", threshold=0.5,
                                                      score_mode="pool+top", score_pool_kernel=7, score_topk_ratio=0.1)
        self.assertEqual(predictor.config["score_mode"], "pool_topk")
        self.assertEqual(predictor.calibration["score_method"]["pool_kernel"], 7)
        self.assertEqual(self.config, original_config)
        self.assertEqual(self.calibration, original_calibration)
        self.assertFalse(predictor.calibration["inference_score_override"]["recalibrated"])

    def test_invalid_statistics_are_rejected(self):
        for key in ("mean_std.std", "quantiles.qb_st"):
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            payload["model_state"][key].zero_()
            checkpoint = self.root / "invalid-statistics.pt"
            torch.save(payload, checkpoint)
            with self.subTest(key=key), self.assertRaises(ValueError):
                inference.EfficientAdPredictor(checkpoint, "cpu")

    def test_nonfinite_calibration_is_rejected(self):
        for overrides in ({"threshold": float("nan")}, {"display_max": 0.}):
            checkpoint = self.save_checkpoint("invalid-calibration.pt",
                                              calibration={**self.calibration, **overrides})
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "threshold"):
                inference.EfficientAdPredictor(checkpoint, "cpu")

    def test_nonfinite_prediction_is_rejected(self):
        with patch.object(cli, "prediction_scores", return_value={"score": float("inf")}), \
                self.assertRaisesRegex(ValueError, "NaN/Inf"):
            self.predictor.predict(self.image)

    def test_black_mask_keeps_all_pixels_and_white_mask_is_rejected(self):
        black, white = self.root / "black.png", self.root / "white.png"
        Image.new("L", (738, 1144), 0).save(black)
        Image.new("L", (32, 32), 255).save(white)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            black_predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", mask=black)
            white_predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", mask=white)
        black_result = black_predictor.predict(self.image)
        self.assertEqual(black_result["score"], self.predictor.predict(self.image)["score"])
        with self.assertRaisesRegex(ValueError, "全部像素"):
            white_predictor.predict(self.image)

    def test_mask_does_not_modify_network_input_and_ignores_nonzero_pixels(self):
        mask = self.root / "partial-mask.png"
        data = np.zeros((45, 71), dtype=np.uint8)
        data[:, :35] = 1
        Image.fromarray(data).save(mask)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu", mask=mask)
        batch, _ = predictor._prepare_batch(self.image)
        reference, _ = self.predictor._prepare_batch(self.image)
        torch.testing.assert_close(batch.image, reference.image, rtol=0, atol=0)
        self.assertTrue(batch.ignore_mask[..., :120].all())
        self.assertFalse(batch.ignore_mask[..., 140:].any())

    def test_checkpoint_embedded_mask_config_works_without_training_json(self):
        black = self.root / "embedded-black.png"
        Image.new("L", (71, 45), 0).save(black)
        config = {**self.config, "circle_config": str(self.root / "missing-config.json"),
                  "circle_params": {"default_mask": str(black)}}
        checkpoint = self.save_checkpoint("embedded-mask.pt", config=config, calibration=self.calibration)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu")
        self.assertEqual(predictor.predict(self.image)["mask_source"], "checkpoint")

    def test_missing_training_mask_is_not_silently_disabled_and_can_be_relocated(self):
        config = {**self.config, "circle_config": str(self.root / "missing-config.json"),
                  "circle_params": {"default_mask": "missing-mask.png"}}
        checkpoint = self.save_checkpoint("missing-mask.pt", config=config, calibration=self.calibration)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu")
        with self.assertRaisesRegex(ValueError, "不会自动改为无 mask"):
            predictor.predict(self.image)
        black = self.root / "relocated-mask.png"
        Image.new("L", (10, 10), 0).save(black)
        config_path = self.root / "local-mask.json"
        config_path.write_text(json.dumps({"CCD1": {"default_mask": black.name}}), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            relocated = inference.EfficientAdPredictor(checkpoint, "cpu", circle_config=config_path)
        self.assertEqual(relocated.predict(self.image)["mask_source"], "explicit_config")

    def test_saved_arrays_and_masks_have_expected_shape_and_values(self):
        output = self.root / "saved-arrays"
        result = self.predictor.predict(self.image, output, save_heatmaps=False)
        anomaly_map = np.load(output / "anomaly_map.npy")
        self.assertEqual(anomaly_map.dtype, np.float32)
        self.assertEqual(anomaly_map.shape, (256, 256))
        self.assertTrue(np.isfinite(anomaly_map).all())
        with Image.open(output / "ignore_mask.png") as mask:
            self.assertEqual(mask.size, (256, 256))
            self.assertEqual(mask.getextrema(), (0, 0))
        with Image.open(output / "anomaly_mask.png") as mask:
            self.assertEqual(mask.size, (71, 45))
        saved = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["score"], result["score"])
        self.assertFalse((output / "prediction.png").exists())

    def test_single_image_cli_and_json_only_options(self):
        with patch("sys.stdout"):
            output = inference.main(["--checkpoint", str(self.checkpoint), "--image", str(self.image),
                                     "--device", "cpu", "--output-dir", str(self.root / "single-output"),
                                     "--no-heatmaps", "--no-maps"])
        destination = output / "images" / self.image.name
        self.assertEqual({p.name for p in destination.iterdir()}, {"prediction.json"})
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["ng_count"] + summary["ok_count"], 1)
        self.assertEqual(summary["threshold"], self.calibration["threshold"])

    def test_single_image_cli_score_and_threshold_overrides_are_saved_in_all_reports(self):
        threshold = self.predictor.predict(self.image)["score"] + 1
        with patch("sys.stdout"):
            output = inference.main(["--checkpoint", str(self.checkpoint), "--image", str(self.image),
                                     "--device", "cpu", "--output-dir", str(self.root / "override-output"),
                                     "--threshold", str(threshold), "--score-mode", "pool+top",
                                     "--score-pool-kernel", "3", "--score-topk-ratio", "0.01",
                                     "--no-heatmaps", "--no-maps"])
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        result = json.loads((output / "images" / self.image.name / "prediction.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["threshold"], threshold)
        self.assertEqual(summary["calibration"]["threshold"], threshold)
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["ng_count"], 0)
        self.assertEqual(result["threshold"], threshold)
        self.assertEqual(result["prediction"], "OK")
        self.assertEqual(summary["score_mode"], "pool_topk")
        self.assertEqual(summary["calibration"]["score_mode"], "pool_topk")
        self.assertEqual(summary["score_method"]["pool_kernel"], 3)
        self.assertEqual(summary["score_method"]["topk_ratio"], 0.01)
        self.assertEqual(result["score_mode"], "pool_topk")
        self.assertEqual(result["localization"]["score_mode"], "pool_topk")
        with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            row, = list(csv.DictReader(stream))
        self.assertEqual(float(row["threshold"]), threshold)
        self.assertEqual(row["prediction"], "OK")
        self.assertEqual(row["score_mode"], "pool_topk")

    def test_invalid_score_override_does_not_create_outputs(self):
        output = self.root / "invalid-score-output"
        with self.assertRaisesRegex(ValueError, "必须同时指定 --threshold"):
            inference.main(["--checkpoint", str(self.checkpoint), "--image", str(self.image),
                            "--device", "cpu", "--score-mode", "pool+top", "--output-dir", str(output)])
        self.assertFalse(output.exists())

    def test_delivery_batch_entry_supports_score_override_in_isolated_mode(self):
        inputs = self.root / "delivery-inputs"
        inputs.mkdir()
        for name in ("one.png", "two.bmp"):
            with Image.open(self.image) as image:
                image.save(inputs / name)
        output_root = self.root / "delivery-output"
        script = Path(__file__).resolve().parents[1] / "delivery" / "efficientad" / "run_inference.py"
        process = subprocess.run([
            sys.executable, "-I", str(script), "--checkpoint", str(self.checkpoint),
            "--image-dir", str(inputs), "--device", "cpu", "--output-dir", str(output_root),
            "--score-mode", "pool+top", "--threshold", "1000000",
            "--score-pool-kernel", "7", "--score-topk-ratio", "0.01", "--no-heatmaps", "--no-maps",
        ], cwd=self.root, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        run, = output_root.iterdir()
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["score_mode"], "pool_topk")
        self.assertEqual(summary["score_method"]["pool_kernel"], 7)
        self.assertEqual(summary["threshold"], 1000000)
        self.assertEqual(summary["ok_count"], 2)
        for name in ("one.png", "two.bmp"):
            result = json.loads((run / "images" / name / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(result["score_mode"], "pool_topk")
            self.assertEqual(result["score_method"]["pool_kernel"], 7)
            self.assertEqual(result["prediction"], "OK")

    def test_batch_cli_offline_keeps_paths_and_renders_multiscale_heatmaps(self):
        calibration = {**self.calibration, "score_method": {
            "name": cli.MULTISCALE_SCORE_METHOD, "pool_kernels": [1, 3], "topk_ratio": 0.01,
            "normalization": {"1": {"median": 0., "denominator": 0.1},
                              "3": {"median": 0., "denominator": 0.2}},
        }}
        checkpoint = self.save_checkpoint("batch-model.pt", calibration=calibration)
        inputs = self.root / "batch-inputs"
        (inputs / "nested").mkdir(parents=True)
        for name in ("same.png", "same.bmp", "nested/same.png"):
            with Image.open(self.image) as image:
                image.save(inputs / name)
        output_root = inputs / "results"
        output_root.mkdir()
        with Image.open(self.image) as image:
            image.save(output_root / "old-result.png")
        script = Path(__file__).resolve().parents[1] / "run.py"
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUTF8": "1"}
        process = subprocess.run([
            sys.executable, str(script), "infer", "--checkpoint", str(checkpoint), "--image-dir", str(inputs),
            "--device", "cpu", "--output-dir", str(output_root),
            "--threshold", "1000000",
            "--score-mode", "multiscale_pool",
        ], capture_output=True, text=True, encoding="utf-8", timeout=120, env=environment)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        run, = [path for path in output_root.iterdir() if path.is_dir()]
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["image_count"], 3)
        self.assertEqual(summary["threshold"], 1000000)
        self.assertEqual(summary["ok_count"], 3)
        self.assertEqual(summary["ng_count"], 0)
        with (run / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(float(row["threshold"]) == 1000000 and row["prediction"] == "OK" for row in rows))
        for name in ("same.png", "same.bmp", "nested/same.png"):
            output = run / "images" / name
            result = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(result["image"], str((inputs / name).resolve()))
            self.assertEqual(result["score_mode"], "multiscale_pool")
            self.assertEqual(result["threshold"], 1000000)
            self.assertEqual(result["prediction"], "OK")
            self.assertFalse(result["boxes_original"])
            for png in ("prediction.png", "prediction_scales.png"):
                with Image.open(output / png) as rendered:
                    self.assertGreater(rendered.width, 100)
                    rendered.verify()

    def test_pdn_legacy_checkpoint_defaults_are_preserved(self):
        config = {**self.config, "backbone": "pdn_small"}
        model = cli.new_model(config).eval()
        model.model.mean_std["std"].data.fill_(1)
        for key in ("qb_st", "qb_ae"):
            model.model.quantiles[key].data.fill_(1)
        checkpoint = self.root / "legacy-pdn.pt"
        cli.save_checkpoint(checkpoint, model, config, {"category": "CCD1"}, 0,
                            calibration={"threshold": 0., "display_max": 1.})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key in ("resnet_architecture_version", "resnet_feature_mode", "resnet_teacher_output_activation"):
            payload["config"].pop(key, None)
        torch.save(payload, checkpoint)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu")
        result = predictor.predict(self.image)
        self.assertEqual(result["score_mode"], "top")
        self.assertEqual(result["anomaly_map_shape"], [256, 256])
        self.assertEqual(predictor.config["resnet_architecture_version"], 1)
        self.assertEqual(predictor.config["resnet_feature_mode"], "native")
        self.assertEqual(predictor.config["resnet_teacher_output_activation"], "relu")

    def test_resnet50_layer1v2_activation_and_signed_maps_are_preserved(self):
        config = {**self.config, "backbone": "resnet50_layer1v2", "resnet_teacher_output_activation": "none"}
        model = cli.new_model(config).eval()
        model.model.mean_std["std"].data.fill_(1)
        for key, value in {"qa_st": 10000., "qb_st": 10001., "qa_ae": 10000., "qb_ae": 10001.}.items():
            model.model.quantiles[key].data.fill_(value)
        checkpoint = self.root / "layer1v2.pt"
        cli.save_checkpoint(checkpoint, model, config, {"category": "CCD1"}, 0,
                            calibration=self.calibration)
        predictor = inference.EfficientAdPredictor(checkpoint, "cpu")
        output = self.root / "signed-maps"
        predictor.predict(self.image, output, save_heatmaps=False)
        self.assertEqual(predictor.config["resnet_teacher_output_activation"], "none")
        self.assertTrue((np.load(output / "anomaly_map.npy") < 0).any())


if __name__ == "__main__":
    unittest.main()
