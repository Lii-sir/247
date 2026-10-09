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
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--checkpoint", "model.pt", "--image", "a.png", "--image-dir", "."])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(["--checkpoint", "model.pt", "--image", "a.png",
                               "--mask", "a.png", "--circle-config", "config.json"])

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
        predictor = inference.EfficientAdPredictor(self.checkpoint, "cpu")
        predictor.threshold = score
        predictor.calibration = {**predictor.calibration, "threshold": score}
        self.assertEqual(predictor.predict(self.image)["prediction"], "OK")

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

    def test_uncalibrated_checkpoint_is_rejected(self):
        checkpoint = self.save_checkpoint("last.pt", calibration=None)
        with self.assertRaisesRegex(ValueError, "尚未校准"):
            inference.EfficientAdPredictor(checkpoint, "cpu")

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
        ], capture_output=True, text=True, encoding="utf-8", timeout=120, env=environment)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        run, = [path for path in output_root.iterdir() if path.is_dir()]
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["image_count"], 3)
        with (run / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        for name in ("same.png", "same.bmp", "nested/same.png"):
            output = run / "images" / name
            result = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(result["image"], str((inputs / name).resolve()))
            self.assertEqual(result["score_mode"], "multiscale_pool")
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
